from __future__ import annotations

from typing import Literal

from src.types import DataGenState


def select_mode_edge(
    state: DataGenState,
) -> Literal["acquire_user_data", "acquire_hf_data", "plan_web_acquisition"]:
    mode = state.get("mode")
    if mode == "A":
        return "acquire_user_data"
    if mode == "B":
        return "acquire_hf_data"
    return "plan_web_acquisition"