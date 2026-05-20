"""Tests for Feature 3 — AutoResearch Loop (owner: Matthew Torre, Hayley Antczak)"""

import json
import math
import sys
from pathlib import Path

import pytest

from src.autoresearch.autoresearch import (
    apply_patch,
    check_early_stop,
    compare_scores,
    continue_edge,
    create_eval_suite,
    decide_keep_or_revert,
    flag_regression,
    log_iteration,
    revert_patch,
)
from src.autoresearch.config import TrainingConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return TrainingConfig(model_name="distilbert-base-uncased")


@pytest.fixture
def config_file(tmp_path, base_config):
    p = tmp_path / "current.json"
    base_config.save(p)
    return p


def _minimal_task():
    return {
        "task_type": "text-classification",
        "modality": "text",
        "has_pretrained_base": True,
        "eval_metric": "f1",
        "complexity": "medium",
    }


def _minimal_dataset(path="/tmp/data"):
    return {
        "dataset": {
            "path": path,
            "format": "jsonl",
            "train_size": 1000,
            "val_size": 100,
            "test_size": 100,
        },
        "mode_used": "A",
        "quality_notes": "",
        "validation_report": {"passed": True, "issues": [], "sample_accuracy_estimate": 0.9},
    }


def _fail_live_service(name: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be used in this test")

    return _fail


# ─── build_autoresearch_graph ─────────────────────────────────────────────────

def test_build_autoresearch_graph_returns_compiled_graph():
    # langgraph may not be installed in CI; skip gracefully
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from src.autoresearch.autoresearch import build_autoresearch_graph
    graph = build_autoresearch_graph()
    assert graph is not None


def test_autoresearch_graph_stops_on_budget_boundary_after_tinker_run(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("langgraph", reason="langgraph not installed")

    import threading

    import src.autoresearch.autoresearch as ar
    import src.cost_manager.cost_manager as cost_module
    from src.cost_manager.cost_manager import CostManager
    from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "claude")
    monkeypatch.setattr("anthropic.Anthropic", _fail_live_service("Claude"))

    def fake_cost_monitor(*_args, **_kwargs):
        thread = threading.Thread(target=lambda: None)
        thread.stop_event = threading.Event()  # type: ignore[attr-defined]
        thread.start()
        return thread

    monkeypatch.setattr(cost_module, "start_cost_monitor", fake_cost_monitor)

    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        json.dumps({"input": "urgent ticket", "output": "escalate"}) + "\n",
        encoding="utf-8",
    )

    runner_calls = []
    proposal_calls = []

    def fake_propose_hypothesis(current_config, diary, task, allowed_params=None):
        proposal_calls.append(
            {
                "current_config": dict(current_config),
                "diary_len": len(diary),
                "allowed_params": allowed_params,
            }
        )
        if len(proposal_calls) > 1:
            raise AssertionError("graph should stop at the budget boundary before another proposal")
        assert "learning_rate" in allowed_params
        return {
            "description": "Increase learning_rate for one bounded fake Tinker run.",
            "patch": json.dumps({"learning_rate": 2e-4}),
            "expected_effect": "Improve the fake validation metric.",
            "search_strategy": "test-double",
        }

    def fake_tinker_runner(config, dataset, *, run_id=None, max_steps=None, **_kwargs):
        assert dataset["dataset"]["path"] == str(dataset_path)
        assert config.model_name == DEFAULT_TINKER_MODEL
        assert max_steps == 2

        call_number = len(runner_calls) + 1
        runner_calls.append(
            {
                "run_id": run_id,
                "learning_rate": config.learning_rate,
            }
        )

        val_loss = 0.50 if call_number == 1 else 0.25
        metrics = {
            "train_loss": val_loss + 0.05,
            "val_loss": val_loss,
            "test_loss": val_loss + 0.02,
            "primary_metric": 1.0 / (1.0 + val_loss),
        }
        run_dir = Path("outputs/experiments") / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, **metrics}) + "\n",
            encoding="utf-8",
        )
        return {
            "job_id": str(run_id),
            "status": "COMPLETED",
            "metrics": metrics,
            "model_path": str(run_dir),
            "cost_usd": 0.01,
            "logs_path": str(run_dir / "metrics.jsonl"),
        }

    monkeypatch.setattr(ar, "propose_hypothesis", fake_propose_hypothesis)
    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fake_tinker_runner)

    plan = {
        "strategy": "fine-tune",
        "base_model": DEFAULT_TINKER_MODEL,
        "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
        "estimated_cost": 0.02,
        "estimated_time_min": 5,
        "training_script_path": str(tmp_path / "train.py"),
        "eval_metric": "primary_metric",
        "backend": "tinker_sft",
        "dataset_path": str(dataset_path),
    }
    config = {
        "data": False,
        "prompt": "classify support tickets",
        "compute_budget": 0.02,
        "training_procedure": {
            "task_type": "text-classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": DEFAULT_TINKER_MODEL,
            "hyperparameters": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "num_epochs": 1,
                "max_seq_length": 256,
                "max_steps": 2,
            },
            "notes": "offline graph budget-boundary test",
        },
    }

    result = ar.invoke_autoresearch_graph(plan, config, CostManager(0.02))

    assert result["n_iterations"] == 1
    assert result["cost"]["total_usd"] == pytest.approx(0.02)
    assert result["cost"]["termination_reason"] == "budget_limit"
    assert result["metrics"]["scalar"] == pytest.approx(0.8)
    assert len(proposal_calls) == 1
    assert len(runner_calls) == 2
    assert runner_calls[0]["run_id"].startswith("autoresearch-baseline-0-")
    assert runner_calls[1]["run_id"].startswith("autoresearch-iteration-0-")
    assert [call["learning_rate"] for call in runner_calls] == [1e-4, 2e-4]

    diary_records = [
        json.loads(line)
        for line in Path(result["research_diary_path"]).read_text().splitlines()
        if line.strip()
    ]
    assert len(diary_records) == 1
    assert diary_records[0]["decision"] == "KEPT"


# ─── apply_patch / revert_patch ───────────────────────────────────────────────

def test_apply_patch_and_revert_patch_are_inverses(config_file, base_config):
    original = apply_patch(str(config_file), json.dumps({"batch_size": 64}))
    assert TrainingConfig.load(config_file).batch_size == 64
    revert_patch(str(config_file), original)
    assert TrainingConfig.load(config_file).batch_size == base_config.batch_size


# ─── check_early_stop ─────────────────────────────────────────────────────────

def test_check_early_stop_true_on_nan_loss():
    metrics = {
        "train_loss": float("nan"),
        "val_loss": 0.3,
        "test_loss": 0.35,
        "primary_metric": 0.88,
    }
    assert check_early_stop(metrics) is True


def test_check_early_stop_true_on_inf_loss():
    metrics = {
        "train_loss": float("inf"),
        "val_loss": 0.3,
        "test_loss": 0.35,
        "primary_metric": 0.88,
    }
    assert check_early_stop(metrics) is True


def test_check_early_stop_false_on_high_finite_sft_loss():
    metrics = {
        "train_loss": 15.0,
        "val_loss": 14.0,
        "test_loss": 14.5,
        "primary_metric": 0.10,
    }
    assert check_early_stop(metrics) is False


def test_check_early_stop_true_on_accuracy_collapse():
    metrics = {
        "train_loss": 0.4,
        "val_loss": 0.45,
        "test_loss": 0.50,
        "primary_metric": 0.001,
    }
    assert check_early_stop(metrics) is True


def test_check_early_stop_true_on_nonfinite_primary_metric():
    metrics = {
        "train_loss": 0.4,
        "val_loss": 0.45,
        "test_loss": 0.50,
        "primary_metric": float("nan"),
    }
    assert check_early_stop(metrics) is True


def test_check_early_stop_false_on_normal_metrics():
    metrics = {
        "train_loss": 0.32,
        "val_loss": 0.38,
        "test_loss": 0.40,
        "primary_metric": 0.87,
    }
    assert check_early_stop(metrics) is False


# ─── compare_scores ───────────────────────────────────────────────────────────

def test_compare_scores_improved_flag():
    new_score = {"scalar": 0.90, "metrics": {}, "critique": ""}
    baseline  = {"scalar": 0.85, "metrics": {}, "critique": ""}
    delta = compare_scores(new_score, baseline)
    assert delta["improved"] is True
    assert delta["absolute"] == pytest.approx(0.05, abs=1e-9)
    assert delta["relative_pct"] == pytest.approx(5.88, abs=0.01)


def test_compare_scores_not_improved_when_lower():
    new_score = {"scalar": 0.80, "metrics": {}, "critique": ""}
    baseline  = {"scalar": 0.85, "metrics": {}, "critique": ""}
    delta = compare_scores(new_score, baseline)
    assert delta["improved"] is False


def test_compare_scores_tiny_positive_delta_is_not_improved():
    new_score = {"scalar": 0.057829344672345545, "metrics": {}, "critique": ""}
    baseline = {"scalar": 0.0577940194157755, "metrics": {}, "critique": ""}
    delta = compare_scores(new_score, baseline)
    assert delta["relative_pct"] == pytest.approx(0.0611, abs=0.001)
    assert delta["improved"] is False


def test_compare_scores_zero_baseline_allows_positive_improvement():
    new_score = {"scalar": 0.01, "metrics": {}, "critique": ""}
    baseline = {"scalar": 0.0, "metrics": {}, "critique": ""}
    delta = compare_scores(new_score, baseline)
    assert delta["improved"] is True


def test_compare_scores_tie_not_improved():
    score = {"scalar": 0.85, "metrics": {}, "critique": ""}
    delta = compare_scores(score, score)
    assert delta["improved"] is False
    assert delta["absolute"] == pytest.approx(0.0)


# ─── decide_keep_or_revert ────────────────────────────────────────────────────

def test_decide_keep_or_revert_tie_defaults_to_revert():
    delta = {"absolute": 0.0, "relative_pct": 0.0, "improved": False}
    assert decide_keep_or_revert(delta) == "REVERT"


def test_decide_keep_or_revert_positive_keeps():
    delta = {"absolute": 0.03, "relative_pct": 3.0, "improved": True}
    assert decide_keep_or_revert(delta) == "KEEP"


def test_decide_keep_or_revert_negative_reverts():
    delta = {"absolute": -0.02, "relative_pct": -2.0, "improved": False}
    assert decide_keep_or_revert(delta) == "REVERT"


# ─── flag_regression ──────────────────────────────────────────────────────────

def test_flag_regression_triggers_below_threshold():
    delta = {"absolute": -0.05, "relative_pct": -5.0, "improved": False}
    assert flag_regression(delta) is True


def test_flag_regression_does_not_trigger_above_threshold():
    delta = {"absolute": 0.02, "relative_pct": 2.0, "improved": True}
    assert flag_regression(delta) is False


def test_flag_regression_at_boundary_is_not_triggered():
    # absolute == threshold: strict < means no flag
    delta = {"absolute": -0.01, "relative_pct": -1.0, "improved": False}
    assert flag_regression(delta, threshold=-0.01) is False


def test_evaluate_node_compares_against_current_best(monkeypatch):
    import src.autoresearch.autoresearch as ar

    monkeypatch.setattr(
        ar,
        "run_evals",
        lambda _model_path, _suite: {"scalar": 0.70, "metrics": {}, "critique": ""},
    )
    state = _make_state(
        eval_suite={"primary_metric": "accuracy", "metrics": [], "test_split_path": "",
                    "use_llm_grading": False},
        baseline_score={"scalar": 0.50, "metrics": {}, "critique": ""},
        best_score={"scalar": 0.80, "metrics": {}, "critique": ""},
        last_result={
            "job_id": "candidate",
            "status": "COMPLETED",
            "metrics": {"train_loss": 0.3, "val_loss": 0.4,
                        "test_loss": 0.5, "primary_metric": 0.70},
            "model_path": "candidate-model",
            "cost_usd": 0.01,
            "logs_path": "candidate.jsonl",
        },
    )

    out = ar.evaluate_node(state)

    assert out["last_delta"]["improved"] is False
    assert out["last_delta"]["absolute"] == pytest.approx(-0.10)


# ─── log_iteration ────────────────────────────────────────────────────────────

def test_log_iteration_appends_to_diary(tmp_path, monkeypatch):
    # Redirect the diary path to a temp location
    import src.autoresearch.autoresearch as ar
    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"

    try:
        record = {
            "iteration": 1,
            "hypothesis": "Decrease lr to reduce loss spikes.",
            "patch": "- learning_rate: 0.0003\n+ learning_rate: 0.00015",
            "cost_usd": 0.0,
            "metrics": {"train_loss": 0.3, "val_loss": 0.35,
                        "test_loss": 0.38, "primary_metric": 0.88},
            "decision": "KEPT",
            "notes": "+5% on f1",
        }
        updated = log_iteration([], record)
        assert len(updated) == 1
        assert updated[0]["decision"] == "KEPT"

        lines = [l for l in ar._DIARY_PATH.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["iteration"] == 1
    finally:
        ar._DIARY_PATH = original_path


def test_log_iteration_returns_extended_diary(tmp_path, monkeypatch):
    import src.autoresearch.autoresearch as ar
    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"

    try:
        existing = [{"iteration": 1, "hypothesis": "x", "patch": "",
                     "cost_usd": 0.0, "metrics": {}, "decision": "KEPT", "notes": ""}]
        new_record = {**existing[0], "iteration": 2, "decision": "REVERTED"}
        result = log_iteration(existing, new_record)
        assert len(result) == 2
        assert result[-1]["iteration"] == 2
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_uses_pre_patch_config_for_kept_diff(tmp_path):
    import src.autoresearch.autoresearch as ar

    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    try:
        state = _make_state(
            eval_suite={"primary_metric": "accuracy", "metrics": [], "test_split_path": "",
                        "use_llm_grading": False},
            current_config={"learning_rate": 2e-4, "batch_size": 4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            original_content=json.dumps({"learning_rate": 1e-4, "batch_size": 4}),
            last_description="Increase learning rate.",
            last_delta={"absolute": 0.1, "relative_pct": 20.0, "improved": True},
            last_result={
                "job_id": "candidate",
                "status": "COMPLETED",
                "metrics": {"train_loss": 0.3, "val_loss": 0.4,
                            "test_loss": 0.5, "primary_metric": 0.70},
                "model_path": "candidate-model",
                "cost_usd": 0.01,
                "logs_path": "candidate.jsonl",
            },
        )

        out = ar.log_node(state)

        assert out["diary"][0]["decision"] == "KEPT"
        assert "- learning_rate: 0.0001" in out["diary"][0]["patch"]
        assert "+ learning_rate: 0.0002" in out["diary"][0]["patch"]
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_records_early_stop_once(tmp_path):
    import src.autoresearch.autoresearch as ar

    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    try:
        state = _make_state(
            eval_suite={"primary_metric": "primary_metric", "metrics": [], "test_split_path": "",
                        "use_llm_grading": False},
            current_config={"learning_rate": 1e-4, "batch_size": 4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            original_content=None,
            last_description="Increase learning rate.",
            last_delta=None,
            last_result={
                "job_id": "candidate",
                "status": "COMPLETED",
                "metrics": {"train_loss": float("inf"), "val_loss": 0.4,
                            "test_loss": 0.5, "primary_metric": 0.70},
                "model_path": "candidate-model",
                "cost_usd": 0.01,
                "logs_path": "candidate.jsonl",
            },
        )

        out = ar.log_node(state)

        assert out["iteration"] == 1
        assert len(out["diary"]) == 1
        assert out["diary"][0]["iteration"] == 1
        assert out["diary"][0]["decision"] == "REVERTED"
        assert "Early-stopped" in out["diary"][0]["notes"]
        lines = [l for l in ar._DIARY_PATH.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_labels_budget_preflight_skip(tmp_path):
    import src.autoresearch.autoresearch as ar

    run_dir = tmp_path / "budget-skip"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"budget_preflight_skipped": True})
    )
    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    try:
        state = _make_state(
            eval_suite={"primary_metric": "primary_metric", "metrics": [], "test_split_path": "",
                        "use_llm_grading": False},
            current_config={"learning_rate": 1e-4, "batch_size": 4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            last_description="Increase learning rate.",
            last_delta=None,
            last_result={
                "job_id": "candidate",
                "status": "CANCELLED",
                "metrics": {"train_loss": 1e9, "val_loss": 1e9,
                            "test_loss": 1e9, "primary_metric": 0.0},
                "model_path": str(run_dir),
                "cost_usd": 0.0,
                "logs_path": str(run_dir / "metrics.jsonl"),
            },
        )

        out = ar.log_node(state)

        assert out["diary"][0]["decision"] == "REVERTED"
        assert "budget preflight" in out["diary"][0]["notes"]
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_skips_eval_adaptation_without_anthropic_key(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar

    def fail_anthropic():
        raise AssertionError("Anthropic should not be constructed in auto mode without a key")

    events = []
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AUTORESEARCH_EVAL_ADAPTATION", raising=False)
    monkeypatch.setattr(ar.anthropic, "Anthropic", fail_anthropic)
    monkeypatch.setattr(
        ar,
        "log_event",
        lambda agent, level, message, metadata=None, **kwargs: events.append(
            {"level": level, "message": message, "metadata": metadata or {}}
        ),
    )

    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    suite = {"primary_metric": "accuracy", "metrics": ["accuracy"],
             "test_split_path": "", "use_llm_grading": False}
    try:
        state = _make_state(
            eval_suite=suite,
            diary=[
                {"iteration": i, "hypothesis": "", "patch": "", "cost_usd": 0.0,
                 "metrics": {}, "decision": "REVERTED", "notes": f"revert {i}"}
                for i in range(1, 10)
            ],
            iteration=9,
            current_config={"learning_rate": 1e-4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            last_description="Increase learning rate.",
            last_delta={"absolute": -0.01, "relative_pct": -2.0, "improved": False},
            last_result={
                "job_id": "candidate",
                "status": "COMPLETED",
                "metrics": {"train_loss": 0.4, "val_loss": 0.5,
                            "test_loss": 0.6, "primary_metric": 0.65},
                "model_path": "candidate-model",
                "cost_usd": 0.01,
                "logs_path": "candidate.jsonl",
            },
        )

        out = ar.log_node(state)

        assert out["iteration"] == 10
        assert out["eval_suite"] == suite
        assert any("skipped eval suite update" in event["message"] for event in events)
        assert events[-1]["metadata"]["reason"] == "ANTHROPIC_API_KEY is not configured"
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_no_spend_skips_eval_adaptation_even_when_required(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar

    def fail_adapt_eval_suite(*args, **kwargs):
        raise AssertionError("Eval adaptation should not run when NO_SPEND=1")

    events = []
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-present")
    monkeypatch.setenv("AUTORESEARCH_EVAL_ADAPTATION", "required")
    monkeypatch.setattr(ar, "adapt_eval_suite", fail_adapt_eval_suite)
    monkeypatch.setattr(
        ar,
        "log_event",
        lambda agent, level, message, metadata=None, **kwargs: events.append(
            {"level": level, "message": message, "metadata": metadata or {}}
        ),
    )

    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    suite = {"primary_metric": "accuracy", "metrics": ["accuracy"],
             "test_split_path": "", "use_llm_grading": False}
    try:
        state = _make_state(
            eval_suite=suite,
            diary=[
                {"iteration": i, "hypothesis": "", "patch": "", "cost_usd": 0.0,
                 "metrics": {}, "decision": "REVERTED", "notes": f"revert {i}"}
                for i in range(1, 10)
            ],
            iteration=9,
            current_config={"learning_rate": 1e-4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            last_description="Increase learning rate.",
            last_delta={"absolute": -0.01, "relative_pct": -2.0, "improved": False},
            last_result={
                "job_id": "candidate",
                "status": "COMPLETED",
                "metrics": {"train_loss": 0.4, "val_loss": 0.5,
                            "test_loss": 0.6, "primary_metric": 0.65},
                "model_path": "candidate-model",
                "cost_usd": 0.01,
                "logs_path": "candidate.jsonl",
            },
        )

        out = ar.log_node(state)

        assert out["iteration"] == 10
        assert out["eval_suite"] == suite
        assert any("skipped eval suite update" in event["message"] for event in events)
        assert events[-1]["metadata"]["reason"] == "NO_SPEND=1"
    finally:
        ar._DIARY_PATH = original_path


def test_log_node_required_eval_adaptation_calls_adaptor(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar

    calls = []
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUTORESEARCH_EVAL_ADAPTATION", "required")
    monkeypatch.setattr(ar, "log_event", lambda *args, **kwargs: None)

    def fake_adapt_eval_suite(suite, weaknesses):
        calls.append({"suite": suite, "weaknesses": weaknesses})
        return {**suite, "metrics": suite["metrics"] + ["hard_case"]}

    monkeypatch.setattr(ar, "adapt_eval_suite", fake_adapt_eval_suite)

    original_path = ar._DIARY_PATH
    ar._DIARY_PATH = tmp_path / "diary.jsonl"
    suite = {"primary_metric": "accuracy", "metrics": ["accuracy"],
             "test_split_path": "", "use_llm_grading": False}
    try:
        state = _make_state(
            eval_suite=suite,
            diary=[
                {"iteration": i, "hypothesis": "", "patch": "", "cost_usd": 0.0,
                 "metrics": {}, "decision": "REVERTED", "notes": f"revert {i}"}
                for i in range(1, 10)
            ],
            iteration=9,
            current_config={"learning_rate": 1e-4},
            current_patch=json.dumps({"learning_rate": 2e-4}),
            last_description="Increase learning rate.",
            last_delta={"absolute": -0.01, "relative_pct": -2.0, "improved": False},
            last_result={
                "job_id": "candidate",
                "status": "COMPLETED",
                "metrics": {"train_loss": 0.4, "val_loss": 0.5,
                            "test_loss": 0.6, "primary_metric": 0.65},
                "model_path": "candidate-model",
                "cost_usd": 0.01,
                "logs_path": "candidate.jsonl",
            },
        )

        out = ar.log_node(state)

        assert calls
        assert calls[0]["weaknesses"][-1] == "-2.0% \u2014 no improvement"
        assert out["eval_suite"]["metrics"] == ["accuracy", "hard_case"]
    finally:
        ar._DIARY_PATH = original_path


def test_completed_high_finite_sft_loss_flows_to_evaluate():
    import src.autoresearch.autoresearch as ar

    state = _make_state(
        last_result={
            "job_id": "candidate",
            "status": "COMPLETED",
            "metrics": {"train_loss": 24.8, "val_loss": 24.8,
                        "test_loss": 24.8, "primary_metric": 0.0387},
            "model_path": "candidate-model",
            "cost_usd": 0.01,
            "logs_path": "candidate.jsonl",
        },
    )

    assert ar.early_stop_edge(state) == "evaluate"


def test_run_node_preflight_skips_unaffordable_candidate(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar
    from src.cost_manager.cost_manager import CostManager

    def fail_runner(*args, **kwargs):
        raise AssertionError("Tinker runner should not start when preflight rejects")

    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fail_runner)
    monkeypatch.chdir(tmp_path)
    state = _make_state(
        plan={
            "strategy": "fine-tune",
            "base_model": "Qwen/Qwen3.5-9B",
            "lora_config": None,
            "estimated_cost": 2.0,
            "estimated_time_min": 5,
            "training_script_path": "outputs/scripts/train.py",
            "eval_metric": "primary_metric",
            "backend": "tinker_sft",
            "dataset_path": str(tmp_path / "train.jsonl"),
            "estimated_run_cost_usd": 2.0,
        },
        config={
            "compute_budget": 1.0,
            "prompt": "",
            "data": False,
            "training_procedure": {
                "task_type": "text-classification",
                "data_format": "jsonl",
                "training_type": "SFT",
                "base_model": "Qwen/Qwen3.5-9B",
                "hyperparameters": {"learning_rate": 1e-4, "batch_size": 1},
                "notes": "",
            },
        },
        current_config={"learning_rate": 1e-4, "batch_size": 1},
        current_patch=json.dumps({"learning_rate": 2e-4}),
        cost_manager=CostManager(1.0),
    )

    out = ar.run_node(state)
    run_dir = Path(out["last_result"]["model_path"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics_row = json.loads((run_dir / "metrics.jsonl").read_text())

    assert out["last_result"]["status"] == "CANCELLED"
    assert out["last_result"]["cost_usd"] == 0.0
    assert out["should_stop"] is True
    assert manifest["budget_preflight_skipped"] is True
    assert all(
        math.isfinite(out["last_result"]["metrics"][key])
        for key in ("train_loss", "val_loss", "test_loss")
    )
    assert all(math.isfinite(metrics[key]) for key in ("train_loss", "val_loss", "test_loss"))
    assert all(math.isfinite(metrics_row[key]) for key in ("train_loss", "val_loss", "test_loss"))
    json.dumps(metrics, allow_nan=False)
    json.dumps(metrics_row, allow_nan=False)


def test_invoke_autoresearch_graph_canonicalizes_tinker_initial_config(monkeypatch):
    import src.autoresearch.autoresearch as ar

    captured = {}

    class FakeGraph:
        def invoke(self, initial_state):
            captured.update(initial_state["current_config"])
            return {
                **initial_state,
                "baseline_result": {
                    "job_id": "baseline",
                    "status": "COMPLETED",
                    "metrics": {"train_loss": 0.5, "val_loss": 0.5,
                                "test_loss": 0.5, "primary_metric": 0.5},
                    "model_path": "baseline-model",
                    "cost_usd": 0.0,
                    "logs_path": "baseline.jsonl",
                },
                "baseline_score": {"scalar": 0.5, "metrics": {}, "critique": ""},
                "best_score": {"scalar": 0.5, "metrics": {}, "critique": ""},
                "best_script": "baseline-model",
            }

    monkeypatch.setattr(ar, "build_autoresearch_graph", lambda: FakeGraph())
    state = _make_state()
    plan = {
        **state["plan"],
        "base_model": "Qwen/Qwen3.5-9B",
        "backend": "tinker_sft",
        "eval_metric": "primary_metric",
    }
    config = {
        **state["config"],
        "training_procedure": {
            **state["config"]["training_procedure"],
            "base_model": "Qwen/Qwen3.5-9B",
            "hyperparameters": {
                "learning_rate": 1e-4,
                "epochs": 2,
                "max_seq_len": 256,
            },
        },
    }

    ar.invoke_autoresearch_graph(plan, config, None)

    assert captured["learning_rate"] == 1e-4
    assert captured["num_epochs"] == 2
    assert captured["max_seq_length"] == 256
    assert "epochs" not in captured
    assert "max_seq_len" not in captured


def test_propose_hypothesis_canonicalizes_tinker_alias_patch(monkeypatch):
    import src.autoresearch.autoresearch as ar

    captured = {}

    class FakeContent:
        text = json.dumps(
            {
                "description": "Increase epochs for a longer bounded run.",
                "patch": {"epochs": 5, "max_seq_len": 1024},
                "expected_effect": "Improve validation loss.",
                "search_strategy": "local",
            }
        )

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("FakeMessage", (), {"content": [FakeContent()]})()

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    monkeypatch.setattr(ar.anthropic, "Anthropic", FakeAnthropic)

    hypothesis = ar.propose_hypothesis(
        {"learning_rate": 1e-4, "epochs": 3, "max_seq_len": 512},
        [],
        _minimal_task(),
        allowed_params=["learning_rate", "max_seq_length", "num_epochs"],
    )

    assert json.loads(hypothesis["patch"]) == {
        "num_epochs": 5,
        "max_seq_length": 1024,
    }
    assert '"num_epochs": 3' in captured["messages"][0]["content"]
    assert '"max_seq_length": 512' in captured["messages"][0]["content"]
    assert '"max_seq_len":' not in captured["messages"][0]["content"]


def test_propose_hypothesis_no_spend_blocks_anthropic_client(monkeypatch):
    import src.autoresearch.autoresearch as ar

    def fail_client():
        raise AssertionError("Anthropic client should not be constructed")

    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setattr(ar.anthropic, "Anthropic", fail_client)

    with pytest.raises(RuntimeError, match="NO_SPEND=1"):
        ar.propose_hypothesis({"learning_rate": 1e-4}, [], _minimal_task())


def test_propose_hypothesis_still_rejects_unsupported_tinker_patch(monkeypatch):
    import src.autoresearch.autoresearch as ar

    class FakeContent:
        text = json.dumps(
            {
                "description": "Change dropout.",
                "patch": {"dropout": 0.2},
                "expected_effect": "Improve validation loss.",
                "search_strategy": "local",
            }
        )

    class FakeMessages:
        def create(self, **kwargs):
            return type("FakeMessage", (), {"content": [FakeContent()]})()

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    monkeypatch.setattr(ar.anthropic, "Anthropic", FakeAnthropic)

    with pytest.raises(ValueError, match="Unsupported Tinker SFT hyperparameter"):
        ar.propose_hypothesis(
            {"learning_rate": 1e-4},
            [],
            _minimal_task(),
            allowed_params=["learning_rate", "num_epochs"],
        )


def test_adapt_eval_suite_no_spend_blocks_anthropic_client(monkeypatch):
    import src.autoresearch.autoresearch as ar

    def fail_client():
        raise AssertionError("Anthropic client should not be constructed")

    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setattr(ar.anthropic, "Anthropic", fail_client)

    suite = {
        "primary_metric": "accuracy",
        "metrics": ["accuracy"],
        "test_split_path": "",
        "use_llm_grading": False,
    }
    with pytest.raises(RuntimeError, match="NO_SPEND=1"):
        ar.adapt_eval_suite(suite, ["weakness"])


def test_cli_no_spend_rejects_claude_strategy_before_api_key_check(monkeypatch):
    import src.autoresearch.__main__ as cli

    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-no-spend")
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m src.autoresearch", "--strategy", "claude", "--n-iters", "1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1


def test_keep_node_canonicalizes_tinker_alias_patch():
    import src.autoresearch.autoresearch as ar

    state = _make_state(
        plan={"training_script_path": "train.py", "base_model": "Qwen/Qwen3.5-9B",
              "lora_config": None, "estimated_cost": 0.0,
              "estimated_time_min": 5, "eval_metric": "primary_metric",
              "backend": "tinker_sft"},
        current_config={"learning_rate": 1e-4, "epochs": 2},
        current_patch=json.dumps({"epochs": 3}),
        last_score={"scalar": 0.6, "metrics": {}, "critique": ""},
        best_score={"scalar": 0.5, "metrics": {}, "critique": ""},
        last_result={
            "job_id": "candidate",
            "status": "COMPLETED",
            "metrics": {"train_loss": 0.4, "val_loss": 0.4,
                        "test_loss": 0.4, "primary_metric": 0.6},
            "model_path": "candidate-model",
            "cost_usd": 0.01,
            "logs_path": "candidate.jsonl",
        },
    )

    out = ar.keep_node(state)

    assert out["current_config"]["num_epochs"] == 3
    assert "epochs" not in out["current_config"]


# ─── continue_edge ────────────────────────────────────────────────────────────

def _make_state(**overrides):
    base = {
        "plan": {"training_script_path": "train.py", "base_model": None,
                 "lora_config": None, "estimated_cost": 0.0,
                 "estimated_time_min": 5, "eval_metric": "f1"},
        "config": {"compute_budget": 50.0, "prompt": "", "data": False,
                   "training_procedure": {"task_type": "text-classification",
                                          "data_format": "", "training_type": "SFT",
                                          "base_model": None, "hyperparameters": {},
                                          "notes": ""}},
        "eval_suite": None,
        "current_script": "train.py",
        "current_config": {},
        "current_patch": None,
        "last_description": None,
        "original_content": None,
        "diary": [],
        "baseline_score": None,
        "baseline_result": None,
        "best_score": None,
        "best_script": "train.py",
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
        "cost_manager": None,
    }
    base.update(overrides)
    return base


def _tinker_propose_state():
    return _make_state(
        plan={
            "strategy": "fine-tune",
            "base_model": "Qwen/Qwen3.5-9B",
            "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
            "estimated_cost": 1.0,
            "estimated_time_min": 5,
            "training_script_path": "outputs/scripts/train.py",
            "eval_metric": "primary_metric",
            "backend": "tinker_sft",
            "dataset_path": "/tmp/train.jsonl",
        },
        config={
            "data": False,
            "prompt": "test",
            "compute_budget": 10.0,
            "training_procedure": {
                "task_type": "text-classification",
                "data_format": "jsonl",
                "training_type": "SFT",
                "base_model": "Qwen/Qwen3.5-9B",
                "hyperparameters": {"learning_rate": 1e-4, "batch_size": 4},
                "notes": "",
            },
        },
        current_script="outputs/scripts/train.py",
        current_config={"learning_rate": 1e-4, "batch_size": 4},
    )


def test_propose_node_auto_without_anthropic_key_uses_local_proposer(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar
    from src.tinker_api.sft_runner import SUPPORTED_TINKER_TUNABLES

    config_path = tmp_path / "current.json"
    TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).save(config_path)
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AUTORESEARCH_PROPOSER", raising=False)
    monkeypatch.setattr(
        ar.anthropic,
        "Anthropic",
        lambda: pytest.fail("Anthropic should not be constructed in auto mode without a key"),
    )
    monkeypatch.setattr(
        ar,
        "propose_hypothesis",
        lambda *args, **kwargs: pytest.fail("Claude proposer should not run without a key"),
    )

    out = ar.propose_node(_tinker_propose_state())
    patch = json.loads(out["current_patch"])
    key, value = next(iter(patch.items()))

    assert len(patch) == 1
    assert key in SUPPORTED_TINKER_TUNABLES
    assert out["current_script"] == "outputs/scripts/train.py"
    assert out["last_description"]
    before_config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).to_dict()
    patched_config = TrainingConfig.load(config_path).to_dict()
    assert value != before_config[key]
    if isinstance(value, (int, float)):
        assert patched_config[key] == pytest.approx(value)
    else:
        assert patched_config[key] == value

def test_propose_node_no_spend_uses_local_proposer_even_when_claude_requested(
    monkeypatch, tmp_path
):
    import src.autoresearch.autoresearch as ar
    from src.tinker_api.sft_runner import SUPPORTED_TINKER_TUNABLES

    config_path = tmp_path / "current.json"
    TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).save(config_path)
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-present")
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "claude")
    monkeypatch.setattr(
        ar,
        "propose_hypothesis",
        lambda *args, **kwargs: pytest.fail("NO_SPEND=1 should block Claude proposer"),
    )

    out = ar.propose_node(_tinker_propose_state())
    patch = json.loads(out["current_patch"])
    key, value = next(iter(patch.items()))

    assert len(patch) == 1
    assert key in SUPPORTED_TINKER_TUNABLES
    assert out["last_description"]
    before_config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).to_dict()
    assert value != before_config[key]
    TrainingConfig.load(config_path).apply_patch(patch)


def test_propose_node_local_mode_produces_tinker_supported_patch(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar
    from src.tinker_api.sft_runner import SUPPORTED_TINKER_TUNABLES

    config_path = tmp_path / "current.json"
    TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).save(config_path)
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-present")
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "local")
    monkeypatch.setattr(
        ar,
        "propose_hypothesis",
        lambda *args, **kwargs: pytest.fail("Explicit local proposer should not call Claude"),
    )

    out = ar.propose_node(_tinker_propose_state())
    patch = json.loads(out["current_patch"])

    assert len(patch) == 1
    assert set(patch).issubset(SUPPORTED_TINKER_TUNABLES)
    key, value = next(iter(patch.items()))
    before_config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).to_dict()
    assert value != before_config[key]
    TrainingConfig.load(config_path).apply_patch(patch)


def test_propose_node_explicit_claude_uses_propose_hypothesis(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar

    config_path = tmp_path / "current.json"
    TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
        batch_size=4,
    ).save(config_path)
    calls = []
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "claude")

    def fake_propose_hypothesis(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "description": "Increase learning rate for a bounded candidate run.",
            "patch": json.dumps({"learning_rate": 2e-4}),
            "expected_effect": "Lower validation loss.",
            "search_strategy": "local",
        }

    monkeypatch.setattr(ar, "propose_hypothesis", fake_propose_hypothesis)

    out = ar.propose_node(_tinker_propose_state())

    assert len(calls) == 1
    assert json.loads(out["current_patch"]) == {"learning_rate": 2e-4}


def test_continue_edge_ends_when_budget_exhausted():
    diary = [{"cost_usd": 60.0, "iteration": 1, "hypothesis": "", "patch": "",
              "metrics": {}, "decision": "KEPT", "notes": ""}]
    state = _make_state(diary=diary, iteration=1)
    # budget is 50.0, spent 60.0
    assert continue_edge(state) == "__end__"


def test_continue_edge_loops_when_budget_remaining():
    state = _make_state(diary=[], iteration=1)
    assert continue_edge(state) == "propose"


def test_continue_edge_ends_on_should_stop():
    state = _make_state(should_stop=True, iteration=1)
    assert continue_edge(state) == "__end__"


def test_continue_edge_ends_on_no_improve_streak():
    from src.autoresearch.autoresearch import _MAX_NO_IMPROVE
    state = _make_state(no_improve_streak=_MAX_NO_IMPROVE, iteration=5)
    assert continue_edge(state) == "__end__"


def test_continue_edge_ends_at_max_iterations():
    from src.autoresearch.autoresearch import _MAX_ITERATIONS
    state = _make_state(iteration=_MAX_ITERATIONS)
    assert continue_edge(state) == "__end__"


def test_dataset_result_from_plan_preserves_split_counts(tmp_path):
    import src.autoresearch.autoresearch as ar

    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"input": "x", "output": "y"}\n')
    plan = {
        "strategy": "fine-tune",
        "base_model": "Qwen/Qwen3.5-9B",
        "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
        "estimated_cost": 1.0,
        "estimated_time_min": 5,
        "training_script_path": "outputs/scripts/train.py",
        "eval_metric": "primary_metric",
        "backend": "tinker_sft",
        "dataset_path": "ignored-when-dataset-present.jsonl",
        "dataset": {
            "path": str(dataset_path),
            "format": "jsonl",
            "train_size": 2,
            "val_size": 1,
            "test_size": 1,
        },
    }

    dataset_result = ar._dataset_result_from_plan(plan)

    assert dataset_result["dataset"] == plan["dataset"]


def test_autoresearch_graph_passes_plan_dataset_splits_to_tinker(monkeypatch, tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    import src.autoresearch.autoresearch as ar

    config_path = tmp_path / "configs" / "current.json"
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps({"input": f"row {i}", "output": "label"}) + "\n"
            for i in range(4)
        )
    )
    calls = []

    class FakeCostManager:
        def __init__(self, budget):
            self.budget = budget
            self.spent_usd = 0.0

        @property
        def status(self):
            return "EXCEEDED" if self.spent_usd >= self.budget else "OK"

        def start(self, job_id):
            self.job_id = job_id

        def stop(self):
            pass

        def record_spend(self, amount, category="training"):
            self.spent_usd += amount
            return self.status

        def cost_breakdown(self, termination_reason=None):
            return {
                "data_gen_usd": 0.0,
                "training_usd": self.spent_usd,
                "llm_calls_usd": 0.0,
                "total_usd": self.spent_usd,
                "termination_reason": termination_reason or "training_complete",
            }

    def fake_runner(config, dataset, *, run_id=None, max_steps=None, **kwargs):
        index = len(calls)
        calls.append(
            {
                "path": dataset["dataset"]["path"],
                "splits": {
                    "train": dataset["dataset"]["train_size"],
                    "val": dataset["dataset"]["val_size"],
                    "test": dataset["dataset"]["test_size"],
                },
            }
        )
        val_loss = 0.5 if index == 0 else 0.4
        run_dir = tmp_path / f"run-{index}"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(
            json.dumps({"train_loss": val_loss, "val_loss": val_loss, "test_loss": val_loss})
        )
        return {
            "job_id": run_id,
            "status": "COMPLETED",
            "metrics": {
                "train_loss": val_loss,
                "val_loss": val_loss,
                "test_loss": val_loss,
                "primary_metric": 1 / (1 + val_loss),
            },
            "model_path": str(run_dir),
            "cost_usd": 0.02,
            "logs_path": str(run_dir / "metrics.jsonl"),
        }

    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fake_runner)
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "claude")
    monkeypatch.setattr(
        ar,
        "propose_hypothesis",
        lambda *args, **kwargs: {
            "description": "Increase learning rate for a bounded candidate run.",
            "patch": json.dumps({"learning_rate": 2e-4}),
            "expected_effect": "Improve heldout loss.",
            "search_strategy": "playbook",
        },
    )

    plan = {
        "strategy": "fine-tune",
        "base_model": "Qwen/Qwen3.5-9B",
        "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
        "estimated_cost": 0.005,
        "estimated_time_min": 5,
        "training_script_path": "outputs/scripts/train.py",
        "eval_metric": "primary_metric",
        "backend": "tinker_sft",
        "dataset_path": str(dataset_path),
        "dataset": {
            "path": str(dataset_path),
            "format": "jsonl",
            "train_size": 2,
            "val_size": 1,
            "test_size": 1,
        },
    }
    config = {
        "data": False,
        "prompt": "test",
        "compute_budget": 0.03,
        "training_procedure": {
            "task_type": "text-classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": "Qwen/Qwen3.5-9B",
            "hyperparameters": {"learning_rate": 1e-4, "batch_size": 1},
            "notes": "",
        },
    }

    result = ar.invoke_autoresearch_graph(plan, config, FakeCostManager(0.03))

    assert [call["path"] for call in calls] == [str(dataset_path), str(dataset_path)]
    assert [call["splits"] for call in calls] == [
        {"train": 2, "val": 1, "test": 1},
        {"train": 2, "val": 1, "test": 1},
    ]
    assert result["n_iterations"] == 1
    assert result["cost"]["termination_reason"] == "budget_limit"


def test_invoke_autoresearch_graph_cost_breakdown_includes_baseline_cost(monkeypatch):
    import src.autoresearch.autoresearch as ar
    from src.cost_manager.cost_manager import CostManager

    class FakeGraph:
        def invoke(self, initial_state):
            cost_manager = initial_state["cost_manager"]
            cost_manager.record_spend(0.20, category="training")
            cost_manager.record_spend(0.30, category="training")
            baseline_result = {
                "job_id": "baseline",
                "status": "COMPLETED",
                "metrics": {"train_loss": 0.5, "val_loss": 0.5,
                            "test_loss": 0.5, "primary_metric": 0.5},
                "model_path": "baseline-model",
                "cost_usd": 0.20,
                "logs_path": "baseline.jsonl",
            }
            return {
                **initial_state,
                "baseline_result": baseline_result,
                "baseline_score": {"scalar": 0.5, "metrics": {}, "critique": ""},
                "best_score": {"scalar": 0.6, "metrics": {}, "critique": ""},
                "best_script": "iteration-model",
                "diary": [
                    {
                        "iteration": 1,
                        "hypothesis": "x",
                        "patch": "",
                        "cost_usd": 0.30,
                        "metrics": {},
                        "decision": "KEPT",
                        "notes": "",
                    }
                ],
                "iteration": 1,
            }

    monkeypatch.setattr(ar, "build_autoresearch_graph", lambda: FakeGraph())
    state = _make_state()
    plan = {
        **state["plan"],
        "strategy": "fine-tune",
        "training_script_path": "train.py",
    }
    cost_manager = CostManager(10.0)

    result = ar.invoke_autoresearch_graph(plan, state["config"], cost_manager)

    assert result["cost"]["training_usd"] == pytest.approx(0.50)
    assert result["cost"]["total_usd"] == pytest.approx(0.50)
    assert result["cost"]["termination_reason"] == "training_complete"
