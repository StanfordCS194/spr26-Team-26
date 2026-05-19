"""
Feature 2 — Decision Engine
Owner: Ron Polonsky, Angel Raychev

Plain Python — no LangGraph. Analyzes task + budget, selects base model,
estimates cost, and writes the training script passed to AutoResearch.
"""

from __future__ import annotations

import math
import os
from textwrap import dedent

from src.types import (
    CostEstimate,
    DatasetResult,
    LoRAConfig,
    OrchestrationConfig,
    TaskAnalysis,
    TrainingPlan,
)
from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL

# Cost per GPU-hour on Tinker (USD) — update when Tinker docs confirm pricing
_TINKER_GPU_COST_PER_HOUR = 2.50

# Rough model-size → GPU-hours-per-epoch lookup (for cost estimation)
_MODEL_COST_PROFILE = {
    DEFAULT_TINKER_MODEL:           {"gpu_hours_per_epoch": 0.15, "strategy": "fine-tune"},
    "bert-base-uncased":        {"gpu_hours_per_epoch": 0.1,  "strategy": "fine-tune"},
    "bert-large-uncased":       {"gpu_hours_per_epoch": 0.3,  "strategy": "fine-tune"},
    "distilbert-base-uncased":  {"gpu_hours_per_epoch": 0.05, "strategy": "fine-tune"},
    "roberta-base":             {"gpu_hours_per_epoch": 0.1,  "strategy": "fine-tune"},
    "t5-small":                 {"gpu_hours_per_epoch": 0.15, "strategy": "fine-tune"},
    "t5-base":                  {"gpu_hours_per_epoch": 0.4,  "strategy": "fine-tune"},
}

_TASK_TO_BASE_MODEL = {
    "text-classification":       DEFAULT_TINKER_MODEL,
    "token-classification":      DEFAULT_TINKER_MODEL,
    "seq2seq":                   DEFAULT_TINKER_MODEL,
    "question-answering":        DEFAULT_TINKER_MODEL,
    "summarization":             DEFAULT_TINKER_MODEL,
    "translation":               DEFAULT_TINKER_MODEL,
    "custom":                    DEFAULT_TINKER_MODEL,
}

_TASK_TO_METRIC = {
    "text-classification": "accuracy",
    "token-classification": "f1",
    "seq2seq": "rouge",
    "question-answering": "f1",
    "summarization": "rouge",
    "translation": "bleu",
}


def run_decision_engine(
    config: OrchestrationConfig,
    dataset: DatasetResult,
) -> TrainingPlan:
    """Top-level dispatcher. Runs task analysis → model selection → cost estimation → script generation."""
    task = analyze_task(config)
    budget = config["compute_budget"]
    model_id = find_base_model(task, budget)
    strategy = "fine-tune" if model_id else "pre-train"
    cost = estimate_training_cost(model_id, dataset, strategy)

    if strategy == "fine-tune" and model_id:
        lora = configure_lora(model_id, task)
        script_path = write_finetune_script(model_id, dataset, lora, config)
    else:
        lora = None
        script_path = write_pretrain_script(task, dataset, config)

    backend = "tinker_sft"
    return TrainingPlan(
        strategy=strategy,
        base_model=model_id,
        lora_config=lora,
        estimated_cost=cost["estimated_usd"],
        estimated_run_cost_usd=estimate_autoresearch_run_cost(cost, config, dataset),
        estimated_time_min=cost["estimated_time_min"],
        training_script_path=script_path,
        eval_metric="primary_metric" if backend == "tinker_sft" else task["eval_metric"],
        backend=backend,
        dataset_path=dataset["dataset"]["path"],
        dataset=dataset["dataset"],
    )


def estimate_autoresearch_run_cost(
    cost: CostEstimate,
    config: OrchestrationConfig,
    dataset: DatasetResult,
) -> float:
    """Estimate one bounded AutoResearch Tinker launch from the full plan cost."""
    total_cost = max(0.0, float(cost["estimated_usd"]))
    if total_cost == 0.0:
        return 0.0

    hp = config["training_procedure"].get("hyperparameters", {})
    max_steps = hp.get("max_steps")
    if max_steps is None:
        return round(total_cost, 4)

    train_size = _positive_int(dataset["dataset"].get("train_size"), default=1)
    batch_size = _positive_int(hp.get("batch_size"), default=16)
    epochs = _positive_int(hp.get("num_epochs", hp.get("epochs")), default=3)
    full_steps = max(1, math.ceil(train_size / batch_size) * epochs)
    bounded_steps = _positive_int(max_steps, default=full_steps)
    ratio = min(1.0, bounded_steps / full_steps)
    return round(min(total_cost, max(0.01, total_cost * ratio)), 4)


def _positive_int(value, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def analyze_task(config: OrchestrationConfig) -> TaskAnalysis:
    """Classifies the task into a canonical type. Determines whether a pretrained base model likely exists."""
    task_type = config["training_procedure"]["task_type"]
    has_base = task_type in _TASK_TO_BASE_MODEL
    return TaskAnalysis(
        task_type=task_type,
        modality="text",  # expand for image/tabular in future
        has_pretrained_base=has_base,
        eval_metric=_TASK_TO_METRIC.get(task_type, "accuracy"),
        complexity="medium",
    )


def find_base_model(task: TaskAnalysis, budget: float) -> str | None:
    """Returns a pretrained HuggingFace model ID if one fits the task and budget, else None."""
    if not task["has_pretrained_base"]:
        return None
    model_id = _TASK_TO_BASE_MODEL.get(task["task_type"])
    if model_id is None:
        return None
    # Quick budget check — reject if even 1 epoch would exceed budget
    profile = _MODEL_COST_PROFILE.get(model_id, {})
    min_cost = profile.get("gpu_hours_per_epoch", 0.5) * _TINKER_GPU_COST_PER_HOUR
    return model_id if min_cost <= budget else None


def estimate_training_cost(
    model_id: str | None,
    dataset: DatasetResult,
    strategy: str,
) -> CostEstimate:
    """Estimates GPU-hours and USD cost based on model size, dataset size, and strategy."""
    epochs = 3
    if model_id and model_id in _MODEL_COST_PROFILE:
        gpu_hours_per_epoch = _MODEL_COST_PROFILE[model_id]["gpu_hours_per_epoch"]
    else:
        # Pre-train from scratch: scale by dataset size
        n = dataset["dataset"].get("train_size", 10_000)
        gpu_hours_per_epoch = max(0.5, n / 50_000)

    total_gpu_hours = gpu_hours_per_epoch * epochs
    usd = round(total_gpu_hours * _TINKER_GPU_COST_PER_HOUR, 2)
    time_min = int(total_gpu_hours * 60)

    return CostEstimate(
        estimated_usd=usd,
        estimated_gpu_hours=round(total_gpu_hours, 2),
        estimated_time_min=time_min,
        confidence="medium",
    )


def configure_lora(base_model: str, task: TaskAnalysis) -> LoRAConfig:
    """Determines LoRA hyperparameters for the base model architecture and task."""
    # Larger rank for more complex tasks
    rank = 16 if task["complexity"] in ("medium", "high") else 8
    return LoRAConfig(
        rank=rank,
        alpha=rank * 2,
        dropout=0.05,
        target_modules=["query", "value"],
    )


def write_finetune_script(
    base_model: str,
    dataset: DatasetResult,
    lora: LoRAConfig,
    config: OrchestrationConfig,
) -> str:
    """Generates a complete LoRA fine-tuning train.py. Returns the path to the written script."""
    os.makedirs("outputs/scripts", exist_ok=True)
    path = "outputs/scripts/train.py"
    hp = config["training_procedure"]["hyperparameters"]

    script = dedent(f"""\
        # Auto-generated fine-tuning script — do not edit manually
        # Base model: {base_model}
        # Strategy: LoRA fine-tune
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model
        from datasets import load_from_disk

        # ── Config ──────────────────────────────────────────────────────────
        BASE_MODEL = "{base_model}"
        DATASET_PATH = "{dataset['dataset']['path']}"
        LEARNING_RATE = {hp.get('learning_rate', 2e-5)}
        BATCH_SIZE = {hp.get('batch_size', 16)}
        EPOCHS = {hp.get('epochs', 3)}
        MAX_SEQ_LEN = {hp.get('max_seq_len', 128)}
        LORA_RANK = {lora['rank']}
        LORA_ALPHA = {lora['alpha']}
        LORA_DROPOUT = {lora['dropout']}

        # ── Model ────────────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                                  lora_dropout=LORA_DROPOUT,
                                  target_modules={list(lora['target_modules'])})
        model = get_peft_model(model, lora_config)

        # ── Data ─────────────────────────────────────────────────────────────
        dataset = load_from_disk(DATASET_PATH)
        def tokenize(batch):
            return tokenizer(batch["input"], truncation=True, max_length=MAX_SEQ_LEN)
        dataset = dataset.map(tokenize, batched=True)

        # ── Training ─────────────────────────────────────────────────────────
        args = TrainingArguments(
            output_dir="outputs/model",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_steps=10,
        )
        trainer = Trainer(model=model, args=args,
                          train_dataset=dataset["train"],
                          eval_dataset=dataset["validation"])
        trainer.train()
        model.save_pretrained("outputs/model/final")
    """)
    with open(path, "w") as f:
        f.write(script)
    return os.path.abspath(path)


def write_pretrain_script(
    task: TaskAnalysis,
    dataset: DatasetResult,
    config: OrchestrationConfig,
) -> str:
    """Generates a from-scratch train.py for pre-training. Returns the path to the written script."""
    os.makedirs("outputs/scripts", exist_ok=True)
    path = "outputs/scripts/train.py"
    hp = config["training_procedure"]["hyperparameters"]

    script = dedent(f"""\
        # Auto-generated pre-training script — do not edit manually
        # Task type: {task['task_type']}
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from datasets import load_from_disk

        # ── Config ──────────────────────────────────────────────────────────
        DATASET_PATH = "{dataset['dataset']['path']}"
        LEARNING_RATE = {hp.get('learning_rate', 1e-4)}
        BATCH_SIZE = {hp.get('batch_size', 32)}
        EPOCHS = {hp.get('epochs', 10)}

        # ── Model (simple MLP baseline — replace with architecture for task) ─
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 2))
            def forward(self, x):
                return self.net(x)

        # ── Training ─────────────────────────────────────────────────────────
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        dataset = load_from_disk(DATASET_PATH)
        loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            for batch in loader:
                optimizer.zero_grad()
                loss = model(batch["input"]).mean()
                loss.backward()
                optimizer.step()
            print(f"Epoch {{epoch+1}} loss: {{loss.item():.4f}}")

        torch.save(model.state_dict(), "outputs/model/final.pt")
    """)
    with open(path, "w") as f:
        f.write(script)
    return os.path.abspath(path)
