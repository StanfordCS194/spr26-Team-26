"""Tests for Feature 2 — Decision Engine (owner: Ron Polonsky, Angel Raychev)"""

import os
import pytest
from src.decision_engine.decision_engine import (
    analyze_task,
    configure_lora,
    estimate_training_cost,
    find_base_model,
    run_decision_engine,
    write_finetune_script,
    write_pretrain_script,
)
from src.types import DatasetResult, OrchestrationConfig, StandardDataset, ValidationReport


def _make_config(task_type="text-classification", budget=50.0) -> OrchestrationConfig:
    return OrchestrationConfig(
        data=False,
        prompt="classify sentiment",
        compute_budget=budget,
        training_procedure={
            "task_type": task_type,
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {"learning_rate": 2e-5, "batch_size": 16, "epochs": 3, "max_seq_len": 128},
            "notes": "test",
        },
    )


def _make_dataset(train_size=1000) -> DatasetResult:
    return DatasetResult(
        dataset=StandardDataset(path="/tmp/data", format="jsonl",
                                train_size=train_size, val_size=100, test_size=100),
        mode_used="B",
        quality_notes="ok",
        validation_report=ValidationReport(passed=True, issues=[], sample_accuracy_estimate=0.95),
    )


def test_analyze_task_classifies_text_classification():
    config = _make_config("text-classification")
    task = analyze_task(config)
    assert task["task_type"] == "text-classification"
    assert task["has_pretrained_base"] is True
    assert task["eval_metric"] == "accuracy"


def test_find_base_model_returns_none_when_no_model_fits_budget():
    config = _make_config("text-classification", budget=50.0)
    task = analyze_task(config)
    # Budget is ample — should find a model
    model = find_base_model(task, 50.0)
    assert model is not None
    # Budget of $0 — should return None
    assert find_base_model(task, 0.0) is None


def test_estimate_training_cost_finetune_cheaper_than_pretrain():
    dataset = _make_dataset()
    fine_tune_cost = estimate_training_cost("distilbert-base-uncased", dataset, "fine-tune")
    pretrain_cost = estimate_training_cost(None, dataset, "pre-train")
    assert fine_tune_cost["estimated_usd"] < pretrain_cost["estimated_usd"]
    assert fine_tune_cost["confidence"] == "medium"


def test_configure_lora_returns_valid_config():
    config = _make_config("text-classification")
    task = analyze_task(config)
    lora = configure_lora("distilbert-base-uncased", task)
    assert lora["rank"] > 0
    assert lora["alpha"] == lora["rank"] * 2
    assert 0 < lora["dropout"] < 1
    assert len(lora["target_modules"]) > 0


def test_write_finetune_script_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config()
    task = analyze_task(config)
    lora = configure_lora("distilbert-base-uncased", task)
    path = write_finetune_script("distilbert-base-uncased", _make_dataset(), lora, config)
    assert os.path.exists(path)
    content = open(path).read()
    assert "distilbert-base-uncased" in content
    assert "LoraConfig" in content


def test_write_pretrain_script_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config("custom")
    task = analyze_task(config)
    path = write_pretrain_script(task, _make_dataset(), config)
    assert os.path.exists(path)
    content = open(path).read()
    assert "SimpleModel" in content


def test_run_decision_engine_returns_training_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config("text-classification", budget=50.0)
    plan = run_decision_engine(config, _make_dataset())
    assert plan["strategy"] in ("fine-tune", "pre-train")
    assert plan["eval_metric"] == "accuracy"
    assert os.path.exists(plan["training_script_path"])
    assert plan["estimated_cost"] > 0
