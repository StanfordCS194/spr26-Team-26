"""Tests for Feature 0 — Manager Agent (owner: Sid Potti)"""

import json
import os
import pytest
from unittest.mock import MagicMock, call, patch
from src.manager.manager import (
    _handoff_to_dataset_result,
    build_manager_graph,
    build_orchestration_config,
    log_decision,
    merge_dataset_results,
    orchestrate_node,
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
    "dataset_queries": ["movie review sentiment", "product review sentiment"],
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
    assert "dataset_queries" in result
    assert isinstance(result["dataset_queries"], list)


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


def test_handoff_to_dataset_result_index_produces_unique_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff = {"mode_used": "B", "raw_data": {"records": [{"input": "x"}] * 5}}
    r0 = _handoff_to_dataset_result(handoff, index=0)
    r1 = _handoff_to_dataset_result(handoff, index=1)
    assert r0["dataset"]["path"] != r1["dataset"]["path"]
    assert os.path.exists(r0["dataset"]["path"])
    assert os.path.exists(r1["dataset"]["path"])


def test_handoff_to_dataset_result_empty_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff = {"mode_used": "C", "raw_data": {"records": []}}
    result = _handoff_to_dataset_result(handoff)
    assert result["validation_report"]["passed"] is False
    assert result["validation_report"]["issues"] != []


# ── merge_dataset_results ─────────────────────────────────────────────────────

def test_merge_dataset_results_combines_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff_a = {"mode_used": "B", "raw_data": {"records": [{"input": "a", "output": "1"}] * 10}}
    handoff_b = {"mode_used": "C", "raw_data": {"records": [{"input": "b", "output": "0"}] * 10}}
    r_a = _handoff_to_dataset_result(handoff_a, index=0)
    r_b = _handoff_to_dataset_result(handoff_b, index=1)

    merged = merge_dataset_results([r_a, r_b])

    assert merged["dataset"]["train_size"] == 16  # 80% of 20
    assert merged["dataset"]["val_size"] == 2     # 10% of 20
    assert merged["validation_report"]["passed"] is True
    assert "B" in merged["mode_used"] and "C" in merged["mode_used"]
    assert "2 dataset(s)" in merged["quality_notes"]

    # verify the merged file actually has all 20 records
    with open(merged["dataset"]["path"]) as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 20


def test_merge_dataset_results_single_passthrough(tmp_path, monkeypatch):
    """Merging a single result should still work."""
    monkeypatch.chdir(tmp_path)
    handoff = {"mode_used": "A", "raw_data": {"records": [{"input": "x"}] * 5}}
    r = _handoff_to_dataset_result(handoff, index=0)
    merged = merge_dataset_results([r])
    assert merged["dataset"]["train_size"] == 4


def test_merge_dataset_results_propagates_issues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handoff_good = {"mode_used": "B", "raw_data": {"records": [{"input": "x"}] * 5}}
    handoff_bad  = {"mode_used": "C", "raw_data": {"records": []}}
    r_good = _handoff_to_dataset_result(handoff_good, index=0)
    r_bad  = _handoff_to_dataset_result(handoff_bad,  index=1)
    merged = merge_dataset_results([r_good, r_bad])
    assert merged["validation_report"]["issues"] != []


# ── orchestrate_node (mocked sub-agents) ─────────────────────────────────────

def _make_orchestrate_state(queries=None):
    reasoning = {**MOCK_REASONING, "dataset_queries": queries or ["movie review sentiment"]}
    config = build_orchestration_config(reasoning, "classify sentiment", 50.0, False)
    return {
        "prompt": "classify sentiment",
        "budget": 50.0,
        "data_path": None,
        "has_data": False,
        "task_reasoning": reasoning,
        "config": config,
        "result": None,
    }


def _fake_result():
    return {
        "weights_path": "outputs/model/final",
        "metrics": {"scalar": 0.85, "metrics": {}, "critique": "good"},
        "cost": {"data_gen_usd": 0.0, "training_usd": 1.0, "llm_calls_usd": 0.0,
                 "total_usd": 1.0, "termination_reason": "training_complete"},
        "n_iterations": 3,
        "research_diary_path": "outputs/logs/diary.jsonl",
    }


def _fake_plan(tmp_path):
    (tmp_path / "train.py").write_text("print('ok')")
    from src.types import TrainingPlan
    return TrainingPlan(
        strategy="fine-tune", base_model="distilbert-base-uncased",
        lora_config=None, estimated_cost=1.0, estimated_time_min=10,
        training_script_path=str(tmp_path / "train.py"), eval_metric="accuracy",
    )


def test_orchestrate_node_single_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_handoff = {"mode_used": "B", "raw_data": {"records": [{"input": "x", "output": "1"}] * 20}}

    with patch("src.data_generator.graph.invoke_data_generator_graph",
               return_value=fake_handoff) as mock_dg, \
         patch("src.decision_engine.decision_engine.run_decision_engine",
               return_value=_fake_plan(tmp_path)), \
         patch("src.autoresearch.autoresearch.invoke_autoresearch_graph",
               return_value=_fake_result()), \
         patch("src.observability.observability.log_event"):

        state = _make_orchestrate_state(queries=["movie review sentiment"])
        out = orchestrate_node(state)

    # DataGen should have been called exactly once
    assert mock_dg.call_count == 1
    assert out["result"]["n_iterations"] == 3


def test_orchestrate_node_multiple_queries_calls_datagen_per_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_handoff = {"mode_used": "B", "raw_data": {"records": [{"input": "x", "output": "1"}] * 10}}

    with patch("src.data_generator.graph.invoke_data_generator_graph",
               return_value=fake_handoff) as mock_dg, \
         patch("src.decision_engine.decision_engine.run_decision_engine",
               return_value=_fake_plan(tmp_path)), \
         patch("src.autoresearch.autoresearch.invoke_autoresearch_graph",
               return_value=_fake_result()), \
         patch("src.observability.observability.log_event"):

        state = _make_orchestrate_state(queries=["movie sentiment", "product sentiment", "tweet sentiment"])
        out = orchestrate_node(state)

    # DataGen called once per query
    assert mock_dg.call_count == 3
    # Each call used a different prompt
    prompts = [c.args[0]["prompt"] for c in mock_dg.call_args_list]
    assert prompts == ["movie sentiment", "product sentiment", "tweet sentiment"]
    assert out["result"]["n_iterations"] == 3


def test_orchestrate_node_merged_dataset_passed_to_decision_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_handoff = {"mode_used": "B", "raw_data": {"records": [{"input": "x", "output": "1"}] * 10}}

    captured = {}

    def capture_de(config, dataset):
        captured["dataset"] = dataset
        return _fake_plan(tmp_path)

    with patch("src.data_generator.graph.invoke_data_generator_graph",
               return_value=fake_handoff), \
         patch("src.decision_engine.decision_engine.run_decision_engine",
               side_effect=capture_de), \
         patch("src.autoresearch.autoresearch.invoke_autoresearch_graph",
               return_value=_fake_result()), \
         patch("src.observability.observability.log_event"):

        state = _make_orchestrate_state(queries=["q1", "q2"])
        orchestrate_node(state)

    # The dataset passed to DE should be the merged one (20 records from 2×10)
    assert captured["dataset"]["dataset"]["train_size"] == 16  # 80% of 20
    assert "2 dataset(s)" in captured["dataset"]["quality_notes"]
