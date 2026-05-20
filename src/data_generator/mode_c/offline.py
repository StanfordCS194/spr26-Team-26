"""Shared offline/no-spend policy for Mode C data generation."""

from __future__ import annotations

import os

_OFFLINE_FLAGS = (
    "NO_SPEND",
    "DATA_GENERATOR_OFFLINE",
    "DATA_GENERATOR_SYNTHETIC_OFFLINE",
)


def env_flag_enabled(name: str) -> bool:
    """Return True for conventional truthy environment flag values."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def mode_c_offline() -> bool:
    """Return True when Mode C must avoid live web or teacher calls."""
    return any(env_flag_enabled(name) for name in _OFFLINE_FLAGS)


def mode_c_offline_reason() -> str:
    """Describe the first configured flag forcing Mode C offline."""
    for name in _OFFLINE_FLAGS:
        if env_flag_enabled(name):
            return f"{name}=1"
    return "offline mode"
