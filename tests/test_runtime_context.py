"""Tests for run-scoped mutable output routing."""

from __future__ import annotations

import json
from pathlib import Path

from src.autoresearch import autoresearch as ar
from src.data_generator.curation import curate_handoff_to_dataset_result
from src.decision_engine.decision_engine import write_finetune_script
from src.manager.manager import build_orchestration_config, log_decision
from src.runtime_context import output_root


def test_output_root_routes_manager_datagen_decision_and_autoresearch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    run_root = tmp_path / "runs" / "run-123"

    config = build_orchestration_config(
        {
            "task_type": "text-classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "suggested_base_model": None,
            "hyperparameters": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "max_seq_len": 128,
            },
            "notes": "test",
        },
        "classify sentiment",
        10.0,
        False,
    )

    with output_root(run_root):
        log_decision("build_config", "test rationale", config)
        dataset = curate_handoff_to_dataset_result(
            {
                "mode_used": "B",
                "raw_data": {
                    "records": [
                        {"input": "great", "output": "positive"},
                        {"input": "bad", "output": "negative"},
                        {"input": "ok", "output": "neutral"},
                    ]
                },
            }
        )
        script_path = write_finetune_script(
            "Qwen/Qwen3.5-9B",
            dataset,
            {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["query", "value"],
            },
            config,
        )
        ar.log_iteration(
            [],
            {
                "iteration": 1,
                "hypothesis": "test",
                "patch": "",
                "cost_usd": 0.0,
                "metrics": {
                    "train_loss": 1.0,
                    "val_loss": 1.0,
                    "test_loss": 1.0,
                    "primary_metric": 0.5,
                },
                "decision": "PENDING",
                "notes": "",
            },
        )

    assert (run_root / "decisions.jsonl").is_file()
    assert Path(dataset["dataset"]["path"]) == run_root / "datasets" / "train_data.jsonl"
    assert Path(script_path) == run_root / "scripts" / "train.py"
    assert (run_root / "logs" / "research_diary.jsonl").is_file()

    assert not Path("decisions.jsonl").exists()
    assert not Path("outputs/datasets/train_data.jsonl").exists()
    assert not Path("outputs/scripts/train.py").exists()

    diary_line = (run_root / "logs" / "research_diary.jsonl").read_text().strip()
    assert json.loads(diary_line)["decision"] == "PENDING"
