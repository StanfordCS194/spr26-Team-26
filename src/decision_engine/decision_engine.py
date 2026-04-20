"""
Feature 2 — Decision Engine
Owner: Ron Polonsky, Angel Raychev

Plain Python — no LangGraph. Analyzes task + budget, selects base model,
estimates cost, and writes the training script passed to AutoResearch.
"""

from __future__ import annotations

from src.types import (
    CostEstimate,
    DatasetResult,
    LoRAConfig,
    OrchestrationConfig,
    TaskAnalysis,
    TrainingPlan,
)


def run_decision_engine(
    config: OrchestrationConfig,
    dataset: DatasetResult,
) -> TrainingPlan:
    """Top-level dispatcher. Runs task analysis → model selection → cost estimation → script generation."""
    raise NotImplementedError


def analyze_task(config: OrchestrationConfig) -> TaskAnalysis:
    """Classifies the task into a canonical type. Determines whether a pretrained base model likely exists."""
    raise NotImplementedError


def find_base_model(task: TaskAnalysis, budget: float) -> str | None:
    """Searches HuggingFace Hub for the best pretrained base model. Returns model ID or None."""
    raise NotImplementedError


def estimate_training_cost(
    model_id: str | None,
    dataset: DatasetResult,
    strategy: str,
) -> CostEstimate:
    """Estimates GPU-hours and USD cost for a training run based on model size, dataset size, strategy."""
    raise NotImplementedError


def configure_lora(base_model: str, task: TaskAnalysis) -> LoRAConfig:
    """Determines LoRA hyperparameters (rank, alpha, dropout, target_modules) for the model + task."""
    raise NotImplementedError


def write_finetune_script(
    base_model: str,
    dataset: DatasetResult,
    lora: LoRAConfig,
    config: OrchestrationConfig,
) -> str:
    """Generates a complete train.py fine-tuning script with LoRA. Returns path to written script."""
    raise NotImplementedError


def write_pretrain_script(
    task: TaskAnalysis,
    dataset: DatasetResult,
    config: OrchestrationConfig,
) -> str:
    """Generates model.py + train.py for from-scratch pre-training. Returns path to written script."""
    raise NotImplementedError
