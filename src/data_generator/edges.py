from __future__ import annotations

from typing import Literal

from src.types import DataGenState


def select_mode_edge(state: DataGenState) -> Literal["acquire_user_data", "acquire_hf_data", "acquire_web_data"]:
    """Route acquisition by mode."""
    mode = state.get("mode")
    if mode == "A":
        return "acquire_user_data"
    if mode == "B":
        return "acquire_hf_data"
    return "acquire_web_data"


def select_curation_edge(state: DataGenState) -> Literal["structure_data", "validate_hf_data"]:
    """Route curation sub-agent behavior after acquisition."""
    mode = state.get("mode")
    if mode in {"A", "C"}:
        return "structure_data"
    return "validate_hf_data"
