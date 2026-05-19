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
from fastapi.middleware.cors import CORSMiddleware

from src.data_sources import looks_like_local_data_path, normalize_hf_dataset_source
from src.manager.manager import invoke_manager_graph
from src.runtime_context import output_root
from src.server.schemas import (
    IterationView,
    LogEntryView,
    MetricPoint,
    PipelineStage,
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


def _copy_request_for_run(
    request: RunRequest,
    *,
    data_path: str | None,
    data_request: dict[str, Any] | None,
) -> RunRequest:
    updates = {"data_path": data_path, "data_request": data_request}
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    return request.copy(update=updates)


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


def _normalize_data_request_for_run(data_request: dict[str, Any] | None) -> dict[str, Any] | None:
    if data_request is None:
        return None

    if not isinstance(data_request, dict):
        raise HTTPException(status_code=400, detail="data_request must be an object")

    sources = data_request.get("sources")
    if sources is None:
        return None
    if not isinstance(sources, list):
        raise HTTPException(status_code=400, detail="data_request.sources must be a list")
    if not sources:
        return None

    normalized_sources: list[Any] = []
    for source in sources:
        normalized_sources.append(_normalize_data_request_source(source))

    normalized = dict(data_request)
    normalized["sources"] = normalized_sources
    return normalized


def _normalize_data_request_source(source: Any) -> Any:
    if isinstance(source, str):
        normalized = normalize_hf_dataset_source(source)
        if normalized:
            return {"type": "hf_dataset", "id": normalized.removeprefix("hf://")}
        raise HTTPException(
            status_code=400,
            detail="data_request.sources string values must be Hugging Face datasets",
        )

    if not isinstance(source, dict):
        raise HTTPException(
            status_code=400,
            detail="data_request.sources entries must be objects or Hugging Face dataset strings",
        )

    source_type = str(source.get("type") or "").strip().lower()
    if source_type in {"hf_dataset", "huggingface_dataset", "huggingface"}:
        for key in ("id", "url", "dataset_id", "dataset_url", "value"):
            field = source.get(key)
            if not isinstance(field, str):
                continue
            normalized = normalize_hf_dataset_source(field)
            if normalized:
                updated = dict(source)
                updated["type"] = "hf_dataset"
                updated["id"] = normalized.removeprefix("hf://")
                return updated
        raise HTTPException(
            status_code=400,
            detail="hf_dataset sources require a valid Hugging Face dataset id or url",
        )

    return dict(source)


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


def _store_manager_result(record: _RunRecord, result: dict[str, Any]) -> None:
    record.result = result
    record.status = "complete"
    record.cost_spent = float((result.get("cost") or {}).get("total_usd") or 0.0)
    metric = _metric_from_result(result)
    record.metrics = [metric] if metric else []
    record.iterations = _iterations_from_diary(result.get("research_diary_path"))
    _complete_all_stages(record)
    _append_log(record, "Manager", "Training pipeline complete", "success")


def _run_manager(record: _RunRecord) -> None:
    with _LOCK:
        _mark_stage(record, 0)
        _append_log(record, "Manager", "Run accepted")
        if record.request.data_path:
            _append_log(record, "DataGen", f"Dataset source: {record.request.data_path}")
        if record.request.data_request:
            source_count = len(record.request.data_request.get("sources") or [])
            _append_log(record, "DataGen", f"Structured data sources: {source_count}")
        _append_log(record, "Manager", "Manager graph is running")

    try:
        with output_root(record.output_dir):
            result = invoke_manager_graph(
                record.request.prompt,
                record.request.budget,
                data_path=record.request.data_path,
                data_request=record.request.data_request,
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
        dataRequest=record.request.data_request,
        costSpent=record.cost_spent,
        metrics=record.metrics,
        iterations=record.iterations,
        logs=record.logs,
        stages=record.stages,
        result=record.result,
        error=record.error,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunCreated)
def create_run(request: RunRequest) -> RunCreated:
    normalized_data_path = _normalize_data_path_for_run(request.data_path)
    normalized_data_request = _normalize_data_request_for_run(request.data_request)
    request = _copy_request_for_run(
        request,
        data_path=normalized_data_path,
        data_request=normalized_data_request,
    )

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


def _reset_runs_for_tests() -> None:
    with _LOCK:
        _RUNS.clear()
