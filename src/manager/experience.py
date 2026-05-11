"""Maps the user's experience-level toggle to an observability + control capability profile."""

from __future__ import annotations

from src.types import UserCapabilityProfile

_PROFILES: dict[str, UserCapabilityProfile] = {
    "BEGINNER": {
        "comfort_level": "BEGINNER",
        "observability": {
            "run_status": "basic",
            "metrics_visibility": "summary",
            "autoresearch_diary_access": "none",
            "cost_visibility": "summary",
        },
        "control": {
            "can_edit_hyperparameters": False,
            "hyperparameter_scope": "none",
            "can_edit_training_script": False,
            "can_constrain_autoresearch_space": False,
            "can_set_custom_stopping_criteria": False,
            "strategy_hints_allowed": True,
        },
    },
    "INTERMEDIATE": {
        "comfort_level": "INTERMEDIATE",
        "observability": {
            "run_status": "detailed",
            "metrics_visibility": "summary",
            "autoresearch_diary_access": "summary",
            "cost_visibility": "summary",
        },
        "control": {
            "can_edit_hyperparameters": True,
            "hyperparameter_scope": "high_level",
            "can_edit_training_script": False,
            "can_constrain_autoresearch_space": True,
            "can_set_custom_stopping_criteria": False,
            "strategy_hints_allowed": True,
        },
    },
    "ADVANCED": {
        "comfort_level": "ADVANCED",
        "observability": {
            "run_status": "detailed",
            "metrics_visibility": "full",
            "autoresearch_diary_access": "full",
            "cost_visibility": "detailed",
        },
        "control": {
            "can_edit_hyperparameters": True,
            "hyperparameter_scope": "full",
            "can_edit_training_script": True,
            "can_constrain_autoresearch_space": True,
            "can_set_custom_stopping_criteria": True,
            "strategy_hints_allowed": True,
        },
    },
}

_NORMALIZE: dict[str, str] = {
    "beginner": "BEGINNER",
    "intermediate": "INTERMEDIATE",
    "advanced": "ADVANCED",
}


def map_experience_to_capabilities(request: dict) -> UserCapabilityProfile:
    """
    Args:
        request: {
            "user_prompt": str,
            "budget_usd": float,
            "experience_toggle": "Beginner" | "Intermediate" | "Advanced"
        }

    Returns:
        UserCapabilityProfile with comfort_level, observability, and control fields.

    Raises:
        ValueError: if experience_toggle is not one of the three recognised values.
    """
    raw = request.get("experience_toggle", "")
    tier = _NORMALIZE.get(raw.lower())
    if tier is None:
        raise ValueError(
            f"Unrecognised experience_toggle {raw!r}. "
            "Expected 'Beginner', 'Intermediate', or 'Advanced'."
        )
    return _PROFILES[tier]
