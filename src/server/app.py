"""FastAPI bridge between the Vite frontend and the Manager graph.

The server deliberately keeps the v1 contract small: start a run, poll its
state, and surface the final Manager result. Detailed live log streaming can be
added later once observability emits run-scoped events.
"""

from __future__ import annotations

import json
import math
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
from src.runtime_context import (
    RunCancelled,
    cancellation_context,
    output_root,
    raise_if_cancelled,
)
from src.server.schemas import (
    ArtifactView,
    IterationView,
    LogEntryView,
    MetricPoint,
    PipelineStage,
    RunArtifactsView,
    RunProvenanceView,
    RunCreated,
    RunRequest,
    RunState,
)
from src.tinker_api.tinker_api import cancel_job


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
DATA_GENERATOR_ARTIFACT_DEFINITIONS = {
    "data_manifest": (
        "Data Manifest",
        "manifest",
        "artifact_manifest.json",
        "application/json",
    ),
    "data_handoff": (
        "Data Handoff",
        "handoff_payload",
        "raw_handoff_data.json",
        "application/json",
    ),
    "data_curation_report": (
        "Data Curation Report",
        "curation_human_readable",
        "human_readable.md",
        "text/markdown",
    ),
    "data_debug_context": (
        "Data Debug Context",
        "debug_context",
        "debug_context.json",
        "application/json",
    ),
    "data_source_report": (
        "Data Source Report",
        "source_human_readable",
        "source_human_readable.md",
        "text/markdown",
    ),
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
    cancel_event: threading.Event = field(default_factory=threading.Event)
    active_tinker_jobs: set[str] = field(default_factory=set)
    worker_thread: threading.Thread | None = None


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


def _is_terminal(record: _RunRecord) -> bool:
    return record.status in {"cancelled", "complete", "failed"}


def _mark_cancelled(record: _RunRecord, message: str = "Run cancelled") -> None:
    record.status = "cancelled"
    record.error = None
    _append_log(record, "Manager", message, "warning")


def _request_cancel(record: _RunRecord) -> None:
    if _is_terminal(record):
        return
    record.cancel_event.set()
    for job_id in list(record.active_tinker_jobs):
        cancel_job(job_id)
    if record.status != "cancelling":
        record.status = "cancelling"
        _append_log(record, "Manager", "Cancellation requested", "warning")


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


def _primary_metric_value_and_label(
    metrics: dict[str, Any],
    *,
    scalar: Any = None,
) -> tuple[float, str]:
    for key, label in (
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("primary_metric", "Primary Score"),
    ):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score, label

    try:
        score = float(scalar)
    except (TypeError, ValueError):
        return 0.0, "Primary Score"
    return (score, "Primary Score") if math.isfinite(score) else (0.0, "Primary Score")


def _metric_from_result(result: dict[str, Any]) -> MetricPoint | None:
    score = result.get("metrics") or {}
    metrics = score.get("metrics") or {}
    primary_metric, primary_metric_label = _primary_metric_value_and_label(
        metrics,
        scalar=score.get("scalar"),
    )

    loss = metrics.get("val_loss")
    if loss is None:
        loss = metrics.get("train_loss")
    if loss is None:
        loss = max(0.0, 1.0 - primary_metric)

    return MetricPoint(
        loss=float(loss),
        accuracy=primary_metric,
        primaryMetric=primary_metric,
        primaryMetricLabel=primary_metric_label,
        iteration=1,
    )


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
        primary_metric, primary_metric_label = _primary_metric_value_and_label(metrics)

        iterations.append(
            IterationView(
                id=f"iter-{iteration}",
                experiment=str(row.get("hypothesis") or "AutoResearch iteration"),
                diff=row.get("patch"),
                loss=float(metrics.get("val_loss") or metrics.get("train_loss") or 0.0),
                f1=primary_metric,
                primaryMetric=primary_metric,
                primaryMetricLabel=primary_metric_label,
                status=decision,
            )
        )
    return list(reversed(iterations))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _log_time(timestamp: Any) -> str:
    if isinstance(timestamp, str) and len(timestamp) >= 19:
        return timestamp[11:19]
    return _time_label()


def _log_kind(level: Any) -> str:
    if level == "ERROR":
        return "error"
    if level == "WARN":
        return "warning"
    return "default"


def _logs_from_observability(record: _RunRecord) -> list[LogEntryView]:
    rows = _jsonl_rows(record.output_dir / "logs" / "run.jsonl")
    return [
        LogEntryView(
            time=_log_time(row.get("timestamp")),
            component=str(row.get("agent") or "System"),
            message=str(row.get("message") or ""),
            type=_log_kind(row.get("level")),
        )
        for row in reversed(rows[-100:])
        if row.get("message")
    ]


def _merge_logs(record: _RunRecord, file_logs: list[LogEntryView]) -> None:
    if not file_logs:
        return
    seen = {(item.time, item.component, item.message, item.type) for item in file_logs}
    merged = file_logs + [
        item
        for item in record.logs
        if (item.time, item.component, item.message, item.type) not in seen
    ]
    record.logs = merged[:100]


def _cost_from_diary(path: Path) -> float:
    return sum(float(row.get("cost_usd") or 0.0) for row in _jsonl_rows(path))


def _experiment_dirs(record: _RunRecord) -> list[Path]:
    root = record.output_dir / "experiments"
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )


def _latest_experiment_dir(record: _RunRecord) -> Path | None:
    dirs = _experiment_dirs(record)
    return dirs[-1] if dirs else None


def _has_complete_experiment_artifacts(run_dir: Path) -> bool:
    return all(
        (run_dir / filename).is_file()
        for name, (_, filename, _) in ARTIFACT_DEFINITIONS.items()
        if name != "diary" and filename is not None
    )


def _latest_artifact_experiment_dir(record: _RunRecord) -> Path | None:
    for run_dir in reversed(_experiment_dirs(record)):
        if _has_complete_experiment_artifacts(run_dir):
            return run_dir
    return None


def _metric_point(row: dict[str, Any], iteration: int) -> MetricPoint | None:
    loss = row.get("val_loss", row.get("train_loss"))
    primary_metric, primary_metric_label = _primary_metric_value_and_label(row)
    try:
        loss_value = float(loss)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(loss_value):
        return None
    return MetricPoint(
        loss=loss_value,
        accuracy=primary_metric,
        primaryMetric=primary_metric,
        primaryMetricLabel=primary_metric_label,
        iteration=iteration,
    )


def _metrics_from_experiments(record: _RunRecord) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for run_dir in _experiment_dirs(record):
        rows = _jsonl_rows(run_dir / "metrics.jsonl")
        if not rows:
            metrics = _read_json_if_exists(run_dir / "metrics.json")
            rows = [metrics] if metrics else []
        for row in rows:
            point = _metric_point(row, len(points) + 1)
            if point:
                points.append(point)
    return points[-100:]


def _refresh_stage_from_files(record: _RunRecord, latest_experiment: Path | None) -> None:
    if _is_terminal(record):
        return

    stage = record.stage
    if (record.output_dir / "datasets" / "train_data.jsonl").exists():
        stage = max(stage, 2)
    if (
        (record.output_dir / "scripts" / "train.py").exists()
        or (record.output_dir / "configs" / "current.json").exists()
    ):
        stage = max(stage, 3)
    if latest_experiment and (
        (latest_experiment / "metrics.jsonl").exists()
        or (latest_experiment / "metrics.json").exists()
    ):
        stage = max(stage, 4)

    if stage > record.stage:
        _mark_stage(record, stage)


def _refresh_record_from_files(record: _RunRecord) -> None:
    _merge_logs(record, _logs_from_observability(record))

    diary_path = record.output_dir / "logs" / "research_diary.jsonl"
    if diary_path.exists():
        record.iterations = _iterations_from_diary(str(diary_path))
        record.cost_spent = max(record.cost_spent, _cost_from_diary(diary_path))

    metrics = _metrics_from_experiments(record)
    if metrics:
        record.metrics = metrics

    latest_experiment = _latest_experiment_dir(record)
    artifact_experiment = _latest_artifact_experiment_dir(record)
    _refresh_stage_from_files(record, latest_experiment)
    if artifact_experiment and record.result is None:
        record.artifacts = _artifacts_from_result(
            record,
            {
                "weights_path": str(artifact_experiment),
                "research_diary_path": str(diary_path) if diary_path.exists() else None,
            },
        )


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
        downloadPath=f"/runs/{run_id}/artifacts/{name}" if downloadable else None,
    )


def _is_run_artifact_path(record: _RunRecord, path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        path.resolve().relative_to(record.output_dir.resolve())
    except ValueError:
        return False
    return True


def _is_run_artifact_dir(record: _RunRecord, path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    try:
        path.resolve().relative_to(record.output_dir.resolve())
    except ValueError:
        return False
    return True


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _latest_data_generator_manifest(record: _RunRecord) -> tuple[dict[str, Any] | None, Path | None]:
    latest = _read_json_if_exists(record.output_dir / "data_generator" / "latest_run.json")
    if not latest:
        return None, None

    manifest_path = _resolve_reported_path(record, latest.get("manifest_path"))
    if (
        not _is_run_artifact_path(record, manifest_path)
        or manifest_path is None
        or manifest_path.name != "artifact_manifest.json"
    ):
        return None, None
    return _read_json_if_exists(manifest_path), manifest_path


def _data_generator_artifact_dir(
    record: _RunRecord,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> Path | None:
    artifact_dir = _resolve_reported_path(record, manifest.get("artifact_dir"))
    if not _is_run_artifact_dir(record, artifact_dir):
        artifact_dir = manifest_path.parent
    return artifact_dir if _is_run_artifact_dir(record, artifact_dir) else None


def _data_generator_artifact_paths(record: _RunRecord) -> dict[str, Path]:
    manifest, manifest_path = _latest_data_generator_manifest(record)
    if not manifest or manifest_path is None:
        return {}

    artifact_dir = _data_generator_artifact_dir(record, manifest, manifest_path)
    if artifact_dir is None:
        return {}

    files = manifest.get("files")
    files = files if isinstance(files, dict) else {}
    paths: dict[str, Path] = {"data_manifest": manifest_path}

    for name, (_, manifest_key, filename, _) in DATA_GENERATOR_ARTIFACT_DEFINITIONS.items():
        if name == "data_manifest":
            continue
        reported = files.get(manifest_key)
        path = _resolve_reported_path(record, reported) if reported else artifact_dir / filename
        if (
            _is_run_artifact_path(record, path)
            and path is not None
            and path.name == filename
            and _path_is_within(path, artifact_dir)
        ):
            paths[name] = path

    return paths


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

    data_paths = _data_generator_artifact_paths(record)
    if data_paths:
        for name, (label, _, _, content_type) in DATA_GENERATOR_ARTIFACT_DEFINITIONS.items():
            path = data_paths.get(name)
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


def _latest_tinker_manifest(record: _RunRecord) -> tuple[dict[str, Any] | None, Path | None]:
    path = record.artifact_paths.get("manifest")
    if path is None:
        run_dir = _latest_artifact_experiment_dir(record) or _latest_experiment_dir(record)
        path = run_dir / "manifest.json" if run_dir else None
    return _read_json_if_exists(path), path


def _latest_data_generator_debug(record: _RunRecord) -> tuple[dict[str, Any] | None, Path | None]:
    debug_path = _data_generator_artifact_paths(record).get("data_debug_context")
    if debug_path is None:
        return None, None
    return _read_json_if_exists(debug_path), debug_path


def _training_backend_from_manifest(manifest: dict[str, Any] | None) -> str | None:
    if not manifest:
        return None
    backend = manifest.get("backend") or manifest.get("training_backend")
    if isinstance(backend, str) and backend.strip():
        return backend.strip()

    checkpoints = manifest.get("checkpoints")
    if isinstance(checkpoints, dict):
        values = [str(value) for value in checkpoints.values() if value]
        if any(value.startswith("dry-run://") for value in values):
            return "dry_run"
        if any(value.startswith("tinker://") for value in values):
            return "tinker"
    return None


def _data_format_meta(debug: dict[str, Any] | None) -> dict[str, Any]:
    if not debug:
        return {}
    raw_data = debug.get("raw_data")
    if not isinstance(raw_data, dict):
        return {}
    format_meta = raw_data.get("format_meta")
    return format_meta if isinstance(format_meta, dict) else {}


def _provenance_for_record(record: _RunRecord) -> RunProvenanceView:
    manifest, manifest_path = _latest_tinker_manifest(record)
    data_debug, data_debug_path = _latest_data_generator_debug(record)

    training_backend = _training_backend_from_manifest(manifest)
    data_mode = None
    mode_c_fallback = None
    if data_debug:
        data_mode = data_debug.get("mode_used")
        mode_c_fallback = data_debug.get("mode_c_fallback")
        source_metadata = data_debug.get("source_metadata")
        if mode_c_fallback is None and isinstance(source_metadata, dict):
            mode_c_fallback = source_metadata.get("mode_c_fallback")

    budget_preflight_skipped = bool(manifest and manifest.get("budget_preflight_skipped"))
    cost = record.result.get("cost") if isinstance(record.result, dict) else None
    manifest_cost = manifest.get("cost") if isinstance(manifest, dict) else None
    budget_skip_reason = None
    if isinstance(manifest_cost, dict):
        budget_skip_reason = manifest_cost.get("termination_reason")
    if not budget_skip_reason and isinstance(cost, dict):
        budget_skip_reason = cost.get("termination_reason")
    if budget_skip_reason != "budget_limit":
        budget_skip_reason = None

    no_spend = _env_flag_enabled("NO_SPEND")
    backend_key = (training_backend or "").replace("-", "_").lower()
    configured_backend = os.getenv("TINKER_BACKEND", "").strip().replace("-", "_").lower()

    live_services: list[str] = []
    if not no_spend and backend_key in {"tinker", "tinker_sft"}:
        live_services.append("Tinker")

    format_meta = _data_format_meta(data_debug)
    if not no_spend and format_meta.get("search_backend"):
        live_services.append(str(format_meta["search_backend"]))
    if not no_spend and format_meta.get("teacher_used"):
        live_services.append("Teacher LLM")

    evidence: list[str] = []
    if _env_flag_enabled("NO_SPEND"):
        evidence.append("env:NO_SPEND=1")
    if configured_backend:
        evidence.append(f"env:TINKER_BACKEND={configured_backend}")
    if manifest_path and manifest_path.is_file():
        evidence.append(str(manifest_path))
    if data_debug_path and data_debug_path.is_file():
        evidence.append(str(data_debug_path))

    if no_spend:
        spend_mode = "no_spend"
    elif budget_preflight_skipped:
        spend_mode = "budget_skipped"
    elif backend_key == "dry_run" or configured_backend == "dry_run":
        spend_mode = "dry_run"
    elif live_services:
        spend_mode = "live"
    else:
        spend_mode = "local"

    return RunProvenanceView(
        spendMode=spend_mode,
        trainingBackend=training_backend,
        dataMode=str(data_mode) if data_mode else None,
        modeCFallback=str(mode_c_fallback) if mode_c_fallback else None,
        budgetPreflightSkipped=budget_preflight_skipped,
        budgetSkipReason=budget_skip_reason,
        liveServices=list(dict.fromkeys(live_services)),
        evidence=evidence,
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
        with output_root(record.output_dir), cancellation_context(
            record.cancel_event,
            record.active_tinker_jobs,
        ):
            raise_if_cancelled()
            result = invoke_manager_graph(
                record.request.prompt,
                record.request.budget,
                record.request.data_path,
            )
            raise_if_cancelled()
    except RunCancelled:
        with _LOCK:
            _refresh_record_from_files(record)
            _mark_cancelled(record)
        return
    except Exception as exc:  # surfaced to the browser as a failed run state
        with _LOCK:
            _refresh_record_from_files(record)
            _fail_current_stage(record, str(exc))
        return

    with _LOCK:
        _refresh_record_from_files(record)
        if record.cancel_event.is_set():
            _mark_cancelled(record)
        else:
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
        provenance=_provenance_for_record(record),
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
    record.worker_thread = thread
    thread.start()
    return RunCreated(run_id=run_id, status="running")


@app.get("/api/runs/{run_id}", response_model=RunState)
def get_run(run_id: str) -> RunState:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        _refresh_record_from_files(record)
        return _to_state(record)


@app.post("/api/runs/{run_id}/cancel", response_model=RunState)
def cancel_run(run_id: str) -> RunState:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        _request_cancel(record)
        _refresh_record_from_files(record)
        return _to_state(record)


@app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
def get_run_artifact(run_id: str, artifact_name: str) -> FileResponse:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        if (
            artifact_name not in ARTIFACT_DEFINITIONS
            and artifact_name not in DATA_GENERATOR_ARTIFACT_DEFINITIONS
        ):
            raise HTTPException(status_code=404, detail="artifact not found")
        _refresh_record_from_files(record)
        path = record.artifact_paths.get(artifact_name)
        content_type = record.artifact_content_types.get(artifact_name)

    if path is None or not path.is_file() or content_type is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=content_type, filename=path.name)


def _reset_runs_for_tests() -> None:
    with _LOCK:
        _RUNS.clear()
