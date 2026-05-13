"""Tests for Feature 0 — Manager Agent (owner: Sid Potti)"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from src.manager.manager import (
    build_manager_graph,
    build_orchestration_config,
    log_decision,
    reason_about_task,
)
from src.types import OrchestrationConfig, TaskReasoning


MOCK_REASONING: TaskReasoning = {
    "task_type": "text-classification",
    "data_format": "jsonl with input/output fields",
    "training_type": "SFT",
    "suggested_base_model": "bert-base-uncased",
    "hyperparameters": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_seq_len": 128,
    },
    "notes": "Standard text classification — fine-tune BERT with SFT.",
}


def test_build_manager_graph_returns_compiled_graph():
    try:
        graph = build_manager_graph()
        assert graph is not None
    except ModuleNotFoundError:
        pytest.skip("langgraph not installed")


def test_build_orchestration_config_shape():
    config = build_orchestration_config(MOCK_REASONING, "classify sentiment", 50.0, False)
    assert config["prompt"] == "classify sentiment"
    assert config["compute_budget"] == 50.0
    assert config["data"] is False
    proc = config["training_procedure"]
    assert proc["task_type"] == "text-classification"
    assert proc["training_type"] == "SFT"
    assert proc["base_model"] == "bert-base-uncased"
    assert "learning_rate" in proc["hyperparameters"]


def test_log_decision_writes_to_disk(tmp_path):
    log_file = str(tmp_path / "decisions.jsonl")
    config = build_orchestration_config(MOCK_REASONING, "test task", 25.0, True)

    with patch("src.manager.manager.LOG_PATH", log_file):
        log_decision("test_step", "some rationale", config)

    lines = open(log_file).readlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["step"] == "test_step"
    assert entry["rationale"] == "some rationale"
    assert "timestamp" in entry
    assert entry["config_snapshot"]["prompt"] == "test task"


def test_reason_about_task_returns_task_reasoning():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(MOCK_REASONING))]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = reason_about_task("classify movie sentiment", 50.0, False)

    assert result["task_type"] == "text-classification"
    assert result["training_type"] == "SFT"
    assert "learning_rate" in result["hyperparameters"]


def test_invoke_manager_graph_returns_trained_model():
    pytest.skip("orchestrate_node not yet implemented — end-to-end test deferred")
