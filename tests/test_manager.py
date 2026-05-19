"""Tests for Feature 0 — Manager Agent (owner: Sid Potti)"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from src.manager.manager import (
    _handoff_to_dataset_result,
    build_manager_graph,
    build_orchestration_config,
    log_decision,
    orchestrate_node,
    query_data_node,
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


def test_query_data_node_uses_programmatic_data_path_without_input(tmp_path, monkeypatch):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"input":"x","output":"y"}\n', encoding="utf-8")

    def fail_input(_prompt):
        raise AssertionError("query_data_node should not prompt when data_path is set")

    monkeypatch.setattr("builtins.input", fail_input)

    out = query_data_node(
        {
            "prompt": "classify",
            "budget": 5.0,
            "data_path": str(data_path),
            "has_data": True,
            "task_reasoning": None,
            "config": None,
            "result": None,
        }
    )

    assert out == {"has_data": True, "data_path": str(data_path.resolve())}


def test_query_data_node_preserves_hf_dataset_source(monkeypatch):
    def fail_input(_prompt):
        raise AssertionError("query_data_node should not prompt when data_path is set")

    monkeypatch.setattr("builtins.input", fail_input)

    out = query_data_node(
        {
            "prompt": "classify",
            "budget": 5.0,
            "data_path": "hf://SetFit/sst2",
            "has_data": True,
            "task_reasoning": None,
            "config": None,
            "result": None,
        }
    )

    assert out == {"has_data": True, "data_path": "hf://SetFit/sst2"}


def test_query_data_node_does_not_treat_missing_relative_file_as_hf(monkeypatch):
    def fail_input(_prompt):
        raise AssertionError("query_data_node should not prompt when data_path is set")

    monkeypatch.setattr("builtins.input", fail_input)

    out = query_data_node(
        {
            "prompt": "classify",
            "budget": 5.0,
            "data_path": "data/train.jsonl",
            "has_data": True,
            "task_reasoning": None,
            "config": None,
            "result": None,
        }
    )

    assert out == {"has_data": False, "data_path": None}


def test_query_data_node_treats_eof_as_no_data(monkeypatch):
    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    out = query_data_node(
        {
            "prompt": "build an assistant",
            "budget": 5.0,
            "data_path": None,
            "has_data": False,
            "task_reasoning": None,
            "config": None,
            "result": None,
        }
    )

    assert out == {"has_data": False, "data_path": None}


def test_invoke_manager_graph_returns_trained_model():
    pytest.skip("end-to-end graph test requires LangGraph + live sub-agents")


# ── _handoff_to_dataset_result ────────────────────────────────────────────────

def test_handoff_to_dataset_result_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff = {
        "mode_used": "B",
        "raw_data": {"records": [{"input": "hello", "output": "1"}] * 10},
    }
    result = _handoff_to_dataset_result(handoff)
    assert result["mode_used"] == "B"
    assert result["dataset"]["train_size"] == 8
    assert result["dataset"]["format"] == "jsonl"
    assert os.path.exists(result["dataset"]["path"])
    assert result["validation_report"]["passed"] is True


def test_handoff_to_dataset_result_empty_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff = {"mode_used": "C", "raw_data": {"records": []}}
    result = _handoff_to_dataset_result(handoff)
    assert result["validation_report"]["passed"] is False
    assert result["validation_report"]["issues"] != []


def test_handoff_to_dataset_result_preserves_datagen_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff = {
        "mode_used": "C",
        "raw_data": {"records": [{"input": "hello", "output": "1"}] * 10},
        "validation_report": {
            "passed": True,
            "issues": ["synthetic data generated by fallback"],
            "sample_accuracy_estimate": 0.72,
        },
    }
    result = _handoff_to_dataset_result(handoff)

    assert result["validation_report"]["passed"] is True
    assert result["validation_report"]["issues"] == [
        "synthetic data generated by fallback"
    ]
    assert result["validation_report"]["sample_accuracy_estimate"] == 0.72


# ── orchestrate_node (mocked sub-agents) ─────────────────────────────────────

def _make_orchestrate_state():
    config = build_orchestration_config(MOCK_REASONING, "classify sentiment", 50.0, False)
    return {
        "prompt": "classify sentiment",
        "budget": 50.0,
        "data_path": None,
        "has_data": False,
        "task_reasoning": MOCK_REASONING,
        "config": config,
        "result": None,
    }


def test_orchestrate_node_returns_trained_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fake_handoff = {
        "mode_used": "B",
        "raw_data": {"records": [{"input": "x", "output": "1"}] * 20},
    }
    fake_result = {
        "weights_path": "outputs/model/final",
        "metrics": {"scalar": 0.85, "metrics": {}, "critique": "good"},
        "cost": {"data_gen_usd": 0.0, "training_usd": 1.0, "llm_calls_usd": 0.0,
                 "total_usd": 1.0, "termination_reason": "training_complete"},
        "n_iterations": 3,
        "research_diary_path": "outputs/logs/diary.jsonl",
    }

    with patch("src.data_generator.graph.invoke_data_generator_graph",
               return_value=fake_handoff), \
         patch("src.decision_engine.decision_engine.run_decision_engine") as mock_de, \
         patch("src.autoresearch.autoresearch.invoke_autoresearch_graph",
               return_value=fake_result), \
         patch("src.observability.observability.log_event"):

        from src.types import TrainingPlan
        mock_plan: TrainingPlan = {
            "strategy": "fine-tune",
            "base_model": "distilbert-base-uncased",
            "lora_config": None,
            "estimated_cost": 1.0,
            "estimated_time_min": 10,
            "training_script_path": str(tmp_path / "train.py"),
            "eval_metric": "accuracy",
        }
        # write a dummy script so autoresearch doesn't choke
        (tmp_path / "train.py").write_text("print('ok')")
        mock_de.return_value = mock_plan

        state = _make_orchestrate_state()
        out = orchestrate_node(state)

    assert "result" in out
    assert out["result"]["n_iterations"] == 3
    assert out["result"]["cost"]["total_usd"] == 1.0


def test_orchestrate_node_stops_before_training_on_untrainable_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_handoff = {
        "mode_used": "C",
        "curation_payload": {
            "records": [
                {
                    "content": "A raw web page without an assistant target.",
                    "source_kind": "web_page",
                    "source_locator": "https://example.com/raw",
                }
            ]
        },
        "validation_report": {
            "passed": False,
            "issues": ["raw web pages require structuring before training"],
            "sample_accuracy_estimate": 0.0,
        },
    }

    with patch("src.data_generator.graph.invoke_data_generator_graph",
               return_value=fake_handoff), \
         patch("src.decision_engine.decision_engine.run_decision_engine") as mock_de, \
         patch("src.autoresearch.autoresearch.invoke_autoresearch_graph") as mock_ar, \
         patch("src.observability.observability.log_event"):

        with pytest.raises(ValueError, match="DataGen did not produce a trainable dataset"):
            orchestrate_node(_make_orchestrate_state())

    mock_de.assert_not_called()
    mock_ar.assert_not_called()
