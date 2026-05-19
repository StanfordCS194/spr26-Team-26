"""FastAPI bridge between the Vite frontend and the Manager graph.

The server deliberately keeps the v1 contract small: start a run, poll its
state, and surface the final Manager result. Detailed live log streaming can be
added later once observability emits run-scoped events.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.data_sources import looks_like_local_data_path, normalize_hf_dataset_source
from src.manager.manager import invoke_manager_graph
from src.runtime_context import output_root
from src.server.schemas import (
    ArtifactView,
    IterationView,
    LogEntryView,
    MetricPoint,
    PipelineStage,
    RunArtifactsView,
    RunCreated,
    RunRequest,
    RunState,
)


STAGE_LABELS = [
    "Manager Init",
    "Data Discovery",
    "Model Selection",
    "Training",
    "AutoResearch",
    "Finalization",
]
RUN_OUTPUT_ROOT = Path(os.getenv("MANAGER_RUN_OUTPUT_ROOT", "outputs/api-runs"))
ARTIFACT_DEFINITIONS = {
    "manifest": ("Manifest", "manifest.json", "application/json"),
    "metrics": ("Metrics", "metrics.json", "application/json"),
    "metrics_log": ("Metrics Log", "metrics.jsonl", "application/x-ndjson"),
    "sample": ("Sample", "sample.json", "application/json"),
    "diary": ("Research Diary", None, "application/x-ndjson"),
}


app = FastAPI(title="Nemoral ML Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class _RunRecord:
    run_id: str
    request: RunRequest
    output_dir: Path
    status: str = "running"
    stage: int = 0
    cost_spent: float = 0.0
    metrics: list[MetricPoint] = field(default_factory=list)
    iterations: list[IterationView] = field(default_factory=list)
    logs: list[LogEntryView] = field(default_factory=list)
    result: dict[str, Any] | None = None
    artifacts: RunArtifactsView | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    artifact_content_types: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    stages: list[PipelineStage] = field(default_factory=list)


_RUNS: dict[str, _RunRecord] = {}
_LOCK = threading.RLock()


def _initial_stages() -> list[PipelineStage]:
    return [
        PipelineStage(id=index, label=label, status="pending")
        for index, label in enumerate(STAGE_LABELS)
    ]


def _time_label() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _append_log(record: _RunRecord, component: str, message: str, kind: str = "default") -> None:
    record.logs.insert(
        0,
        LogEntryView(time=_time_label(), component=component, message=message, type=kind),
    )
    record.logs = record.logs[:100]


def _copy_request_with_data_path(request: RunRequest, data_path: str | None) -> RunRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"data_path": data_path})
    return request.copy(update={"data_path": data_path})


def _normalize_data_path_for_run(data_path: str | None) -> str | None:
    if data_path is None:
        return None

    value = data_path.strip()
    if not value:
        return None

    local_path = Path(value).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    if looks_like_local_data_path(value):
        raise HTTPException(
            status_code=400,
            detail="data_path must be an existing local path or Hugging Face source",
        )

    hf_source = normalize_hf_dataset_source(value)
    if hf_source:
        return hf_source

    raise HTTPException(
        status_code=400,
        detail="data_path must be an existing local path or Hugging Face source",
    )


def _mark_stage(record: _RunRecord, stage: int, status: str = "in-progress") -> None:
    record.stage = stage
    updated = []
    for item in record.stages:
        if item.id < stage:
            updated.append(PipelineStage(id=item.id, label=item.label, status="complete"))
        elif item.id == stage:
            updated.append(PipelineStage(id=item.id, label=item.label, status=status))
        else:
            updated.append(PipelineStage(id=item.id, label=item.label, status="pending"))
    record.stages = updated


def _complete_all_stages(record: _RunRecord) -> None:
    record.stage = len(STAGE_LABELS) - 1
    record.stages = [
        PipelineStage(id=item.id, label=item.label, status="complete")
        for item in record.stages
    ]


def _fail_current_stage(record: _RunRecord, message: str) -> None:
    record.status = "failed"
    record.error = message
    _append_log(record, "Manager", message, "error")


def _metric_from_result(result: dict[str, Any]) -> MetricPoint | None:
    score = result.get("metrics") or {}
    metrics = score.get("metrics") or {}
    scalar = float(score.get("scalar") or metrics.get("primary_metric") or 0.0)

    loss = metrics.get("val_loss")
    if loss is None:
        loss = metrics.get("train_loss")
    if loss is None:
        loss = max(0.0, 1.0 - scalar)

    return MetricPoint(loss=float(loss), accuracy=scalar, iteration=1)


def _iterations_from_diary(path: str | None) -> list[IterationView]:
    if not path:
        return []

    diary_path = Path(path)
    if not diary_path.exists():
        return []

    iterations: list[IterationView] = []
    for line in diary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        metrics = row.get("metrics") or {}
        iteration = int(row.get("iteration") or len(iterations) + 1)
        decision = row.get("decision") or "PENDING"
        if decision not in {"KEPT", "REVERTED", "PENDING"}:
            decision = "PENDING"

        iterations.append(
            IterationView(
                id=f"iter-{iteration}",
                experiment=str(row.get("hypothesis") or "AutoResearch iteration"),
                diff=row.get("patch"),
                loss=float(metrics.get("val_loss") or metrics.get("train_loss") or 0.0),
                f1=float(metrics.get("primary_metric") or 0.0),
                status=decision,
            )
        )
    return list(reversed(iterations))


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _resolve_reported_path(record: _RunRecord, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None

    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        raw_path,
        record.output_dir / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


def _artifact_view(
    *,
    run_id: str,
    name: str,
    label: str,
    path: Path | None,
    content_type: str,
    downloadable: bool,
) -> ArtifactView:
    exists = bool(path and path.is_file())
    return ArtifactView(
        name=name,
        label=label,
        path=str(path) if path else None,
        exists=exists,
        sizeBytes=path.stat().st_size if exists and path is not None else None,
        contentType=content_type,
        downloadPath=f"/api/runs/{run_id}/artifacts/{name}" if downloadable else None,
    )


def _is_run_artifact_path(record: _RunRecord, path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        path.resolve().relative_to(record.output_dir.resolve())
    except ValueError:
        return False
    return True


def _artifacts_from_result(record: _RunRecord, result: dict[str, Any]) -> RunArtifactsView:
    model_path = _resolve_reported_path(record, result.get("weights_path"))
    diary_path = _resolve_reported_path(record, result.get("research_diary_path"))

    files: list[ArtifactView] = []
    artifact_paths: dict[str, Path] = {}
    artifact_content_types: dict[str, str] = {}

    for name, (label, filename, content_type) in ARTIFACT_DEFINITIONS.items():
        if name == "diary":
            path = diary_path
        else:
            path = model_path / filename if model_path and filename else None
        downloadable = _is_run_artifact_path(record, path)
        view = _artifact_view(
            run_id=record.run_id,
            name=name,
            label=label,
            path=path,
            content_type=content_type,
            downloadable=downloadable,
        )
        files.append(view)
        if downloadable and path is not None:
            artifact_paths[name] = path
            artifact_content_types[name] = content_type

    manifest = _read_json_if_exists(artifact_paths.get("manifest"))
    checkpoints = manifest.get("checkpoints", {}) if manifest else {}
    metrics = _read_json_if_exists(artifact_paths.get("metrics"))
    sample = _read_json_if_exists(artifact_paths.get("sample"))

    record.artifact_paths = artifact_paths
    record.artifact_content_types = artifact_content_types
    return RunArtifactsView(
        modelPath=str(model_path) if model_path else None,
        checkpoints=checkpoints if isinstance(checkpoints, dict) else {},
        metrics=metrics,
        sample=sample,
        files=files,
    )


def _store_manager_result(record: _RunRecord, result: dict[str, Any]) -> None:
    record.result = result
    record.status = "complete"
    record.cost_spent = float((result.get("cost") or {}).get("total_usd") or 0.0)
    metric = _metric_from_result(result)
    record.metrics = [metric] if metric else []
    record.iterations = _iterations_from_diary(result.get("research_diary_path"))
    record.artifacts = _artifacts_from_result(record, result)
    _complete_all_stages(record)
    _append_log(record, "Manager", "Training pipeline complete", "success")


def _run_manager(record: _RunRecord) -> None:
    with _LOCK:
        _mark_stage(record, 0)
        _append_log(record, "Manager", "Run accepted")
        if record.request.data_path:
            _append_log(record, "DataGen", f"Dataset source: {record.request.data_path}")
        _append_log(record, "Manager", "Manager graph is running")

    try:
        with output_root(record.output_dir):
            result = invoke_manager_graph(
                record.request.prompt,
                record.request.budget,
                record.request.data_path,
            )
    except Exception as exc:  # surfaced to the browser as a failed run state
        with _LOCK:
            _fail_current_stage(record, str(exc))
        return

    with _LOCK:
        _store_manager_result(record, dict(result))


def _to_state(record: _RunRecord) -> RunState:
    return RunState(
        run_id=record.run_id,
        status=record.status,
        stage=record.stage,
        prompt=record.request.prompt,
        budget=record.request.budget,
        taskType=record.request.task_type,
        dataPath=record.request.data_path,
        costSpent=record.cost_spent,
        metrics=record.metrics,
        iterations=record.iterations,
        logs=record.logs,
        stages=record.stages,
        artifacts=record.artifacts,
        result=record.result,
        error=record.error,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunCreated)
def create_run(request: RunRequest) -> RunCreated:
    normalized_data_path = _normalize_data_path_for_run(request.data_path)
    request = _copy_request_with_data_path(request, normalized_data_path)

    run_id = str(uuid.uuid4())
    output_dir = RUN_OUTPUT_ROOT / run_id
    record = _RunRecord(
        run_id=run_id,
        request=request,
        output_dir=output_dir,
        stages=_initial_stages(),
    )
    with _LOCK:
        _RUNS[run_id] = record

    thread = threading.Thread(target=_run_manager, args=(record,), daemon=True)
    thread.start()
    return RunCreated(run_id=run_id, status="running")


@app.get("/api/runs/{run_id}", response_model=RunState)
def get_run(run_id: str) -> RunState:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return _to_state(record)


@app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
def get_run_artifact(run_id: str, artifact_name: str) -> FileResponse:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        if artifact_name not in ARTIFACT_DEFINITIONS:
            raise HTTPException(status_code=404, detail="artifact not found")
        path = record.artifact_paths.get(artifact_name)
        content_type = record.artifact_content_types.get(artifact_name)

    if path is None or not path.is_file() or content_type is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=content_type, filename=path.name)


def _reset_runs_for_tests() -> None:
    with _LOCK:
        _RUNS.clear()
