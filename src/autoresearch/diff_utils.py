"""
Diff utilities for the AutoResearch Loop.

format_patch_as_diff produces a git-style one-parameter diff string that
goes into the ResearchDiary's `patch` field. Currently this is a config-level
diff (JSON key/value), not a textual diff of train.py — consistent with the
PROPOSE-only phase.

Design note: the diff string is intentionally human-readable and minimal.
When RUN is wired up and patches can touch train.py directly, extend
format_script_diff below to produce a real unified diff.
"""

from __future__ import annotations

from typing import Any

from src.autoresearch.config import TrainingConfig


def format_patch_as_diff(patch: dict[str, Any], old_config: TrainingConfig) -> str:
    """
    Returns a git-style config diff, e.g.

      - learning_rate: 0.0003
      + learning_rate: 0.00015

    Each changed key gets its own - / + line pair. The string is stored in
    ResearchDiary.patch and is sufficient to reproduce the experiment later.
    """
    lines: list[str] = []
    config_dict = old_config.to_dict()
    for key, new_val in patch.items():
        old_val = config_dict.get(key, "<unset>")
        lines.append(f"- {key}: {old_val}")
        lines.append(f"+ {key}: {new_val}")
    return "\n".join(lines)


def parse_diff_to_patch(diff: str) -> dict[str, Any]:
    """
    Inverse of format_patch_as_diff. Parses the + lines to reconstruct the
    patch dict. Used by the DECIDE phase to re-apply or replay changes.
    """
    patch: dict[str, Any] = {}
    for line in diff.splitlines():
        if not line.startswith("+ "):
            continue
        key, sep, raw_val = line[2:].partition(": ")
        key = key.strip()
        if not key or not sep:   # guard against malformed "+" lines with no ": "
            continue
        raw_val = raw_val.strip()
        # Best-effort type inference: try int, float, then leave as string.
        # int must be tried before float to preserve integer types.
        for coerce in (int, float):
            try:
                patch[key] = coerce(raw_val)
                break
            except ValueError:
                pass
        else:
            patch[key] = raw_val
    return patch
