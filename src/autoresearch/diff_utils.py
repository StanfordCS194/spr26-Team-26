"""
Diff utilities for the AutoResearch Loop.

Produces and parses git-style config diffs that go into the ResearchDiary.
These are config-level (JSON key/value), not source-level — extending to
unified diffs of train.py is a natural next step once script patching lands.
"""

from __future__ import annotations

from typing import Any

from src.autoresearch.config import TrainingConfig


def format_patch_as_diff(patch: dict[str, Any], old_config: TrainingConfig) -> str:
    """Returns a git-style diff string for each changed key, stored in the ResearchDiary."""
    lines: list[str] = []
    config_dict = old_config.to_dict()
    for key, new_val in patch.items():
        old_val = config_dict.get(key, "<unset>")
        lines.append(f"- {key}: {old_val}")
        lines.append(f"+ {key}: {new_val}")
    return "\n".join(lines)


def parse_diff_to_patch(diff: str) -> dict[str, Any]:
    """Inverse of format_patch_as_diff — reconstructs the patch dict from the + lines."""
    patch: dict[str, Any] = {}
    for line in diff.splitlines():
        if not line.startswith("+ "):
            continue
        key, sep, raw_val = line[2:].partition(": ")
        key = key.strip()
        if not key or not sep:
            continue
        raw_val = raw_val.strip()
        # int before float so "32" stays an int, not 32.0
        for coerce in (int, float):
            try:
                patch[key] = coerce(raw_val)
                break
            except ValueError:
                pass
        else:
            patch[key] = raw_val
    return patch
