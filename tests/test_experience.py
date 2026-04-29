"""Tests for experience-level → capability profile mapping."""

import pytest

from src.manager.experience import map_experience_to_capabilities


def _req(toggle: str) -> dict:
    return {
        "user_prompt": "classify sentiment in product reviews",
        "budget_usd": 50.0,
        "experience_toggle": toggle,
    }


# ─── Normalisation ────────────────────────────────────────────────────────────

def test_beginner_normalises():
    p = map_experience_to_capabilities(_req("Beginner"))
    assert p["comfort_level"] == "BEGINNER"


def test_intermediate_normalises():
    p = map_experience_to_capabilities(_req("Intermediate"))
    assert p["comfort_level"] == "INTERMEDIATE"


def test_advanced_normalises():
    p = map_experience_to_capabilities(_req("Advanced"))
    assert p["comfort_level"] == "ADVANCED"


def test_invalid_toggle_raises():
    with pytest.raises(ValueError, match="Unrecognised"):
        map_experience_to_capabilities(_req("Expert"))


# ─── BEGINNER profile ─────────────────────────────────────────────────────────

def test_beginner_observability():
    obs = map_experience_to_capabilities(_req("Beginner"))["observability"]
    assert obs["run_status"] == "basic"
    assert obs["metrics_visibility"] == "summary"
    assert obs["autoresearch_diary_access"] == "none"
    assert obs["cost_visibility"] == "summary"


def test_beginner_control():
    ctrl = map_experience_to_capabilities(_req("Beginner"))["control"]
    assert ctrl["can_edit_hyperparameters"] is False
    assert ctrl["hyperparameter_scope"] == "none"
    assert ctrl["can_edit_training_script"] is False
    assert ctrl["can_constrain_autoresearch_space"] is False
    assert ctrl["can_set_custom_stopping_criteria"] is False
    assert ctrl["strategy_hints_allowed"] is True


# ─── INTERMEDIATE profile ─────────────────────────────────────────────────────

def test_intermediate_observability():
    obs = map_experience_to_capabilities(_req("Intermediate"))["observability"]
    assert obs["run_status"] == "detailed"
    assert obs["metrics_visibility"] == "summary"
    assert obs["autoresearch_diary_access"] == "summary"
    assert obs["cost_visibility"] == "summary"


def test_intermediate_control():
    ctrl = map_experience_to_capabilities(_req("Intermediate"))["control"]
    assert ctrl["can_edit_hyperparameters"] is True
    assert ctrl["hyperparameter_scope"] == "high_level"
    assert ctrl["can_edit_training_script"] is False
    assert ctrl["can_constrain_autoresearch_space"] is True
    assert ctrl["can_set_custom_stopping_criteria"] is False
    assert ctrl["strategy_hints_allowed"] is True


# ─── ADVANCED profile ─────────────────────────────────────────────────────────

def test_advanced_observability():
    obs = map_experience_to_capabilities(_req("Advanced"))["observability"]
    assert obs["run_status"] == "detailed"
    assert obs["metrics_visibility"] == "full"
    assert obs["autoresearch_diary_access"] == "full"
    assert obs["cost_visibility"] == "detailed"


def test_advanced_control():
    ctrl = map_experience_to_capabilities(_req("Advanced"))["control"]
    assert ctrl["can_edit_hyperparameters"] is True
    assert ctrl["hyperparameter_scope"] == "full"
    assert ctrl["can_edit_training_script"] is True
    assert ctrl["can_constrain_autoresearch_space"] is True
    assert ctrl["can_set_custom_stopping_criteria"] is True
    assert ctrl["strategy_hints_allowed"] is True


# ─── user_prompt and budget are ignored ───────────────────────────────────────

def test_does_not_modify_prompt_or_budget():
    req = _req("Advanced")
    result = map_experience_to_capabilities(req)
    assert "user_prompt" not in result
    assert "budget_usd" not in result


# ─── profiles are independent (no shared-reference mutation) ──────────────────

def test_profiles_are_independent():
    p1 = map_experience_to_capabilities(_req("Beginner"))
    p2 = map_experience_to_capabilities(_req("Advanced"))
    assert p1["comfort_level"] != p2["comfort_level"]
    assert p1["observability"] != p2["observability"]
