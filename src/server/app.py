"""FastAPI bridge between the Vite frontend and the Manager graph.

The server deliberately keeps the v1 contract small: start a run, poll its
state, and surface the final Manager result. Detailed live log streaming can be
added later once observability emits run-scoped events.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.manager.manager import invoke_manager_graph
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
        _append_log(record, "Manager", "Manager graph is running")

    try:
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
    if request.data_path and not Path(request.data_path).exists():
        raise HTTPException(status_code=400, detail="data_path does not exist")

    run_id = str(uuid.uuid4())
    record = _RunRecord(run_id=run_id, request=request, stages=_initial_stages())
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
