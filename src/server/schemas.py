"""Pydantic schemas for the browser-facing run API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=2000)
    budget: float = Field(gt=0, le=500)
    task_type: Literal["classification", "regression", "fine-tuning"] = "fine-tuning"
    data_path: str | None = Field(default=None, max_length=1000)


class RunCreated(BaseModel):
    run_id: str
    status: Literal["running"]


class PipelineStage(BaseModel):
    id: int
    label: str
    status: Literal["pending", "in-progress", "complete"]


class MetricPoint(BaseModel):
    loss: float
    accuracy: float
    iteration: int


class IterationView(BaseModel):
    id: str
    experiment: str
    diff: str | None = None
    loss: float
    f1: float
    status: Literal["KEPT", "REVERTED", "PENDING"]


class LogEntryView(BaseModel):
    time: str
    component: str
    message: str
    type: Literal["default", "success", "warning", "error"]


class RunState(BaseModel):
    run_id: str
    status: Literal["running", "complete", "failed"]
    stage: int
    prompt: str
    budget: float
    taskType: Literal["classification", "regression", "fine-tuning"]
    dataPath: str | None = None
    costSpent: float
    metrics: list[MetricPoint]
    iterations: list[IterationView]
    logs: list[LogEntryView]
    stages: list[PipelineStage]
    result: dict[str, Any] | None = None
    error: str | None = None
