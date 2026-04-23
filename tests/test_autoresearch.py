"""Tests for Feature 3 — AutoResearch Loop (owner: Matthew Torre, Hayley Antczak)"""

import json
import math
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


# ─── build_autoresearch_graph ─────────────────────────────────────────────────

def test_build_autoresearch_graph_returns_compiled_graph():
    # langgraph may not be installed in CI; skip gracefully
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from src.autoresearch.autoresearch import build_autoresearch_graph
    graph = build_autoresearch_graph()
    assert graph is not None


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


def test_check_early_stop_true_on_exploding_loss():
    metrics = {
        "train_loss": 15.0,
        "val_loss": 14.0,
        "test_loss": 14.5,
        "primary_metric": 0.10,
    }
    assert check_early_stop(metrics) is True


def test_check_early_stop_true_on_accuracy_collapse():
    metrics = {
        "train_loss": 0.4,
        "val_loss": 0.45,
        "test_loss": 0.50,
        "primary_metric": 0.001,
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
        "best_score": None,
        "best_script": "train.py",
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
    }
    base.update(overrides)
    return base


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
