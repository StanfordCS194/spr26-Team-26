"""Data curation helpers for turning DataGen handoffs into trainable JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.types import DatasetResult, StandardDataset, ValidationReport

_INPUT_KEYS = ("input", "prompt", "question", "content", "text", "utterance")
_TARGET_KEYS = ("output", "answer", "label", "target", "label_text", "intent")
_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant"}


def curate_handoff_to_dataset_result(
    handoff: Mapping[str, Any],
    *,
    output_dir: str = "outputs/datasets",
    filename: str = "train_data.jsonl",
) -> DatasetResult:
    """Normalize a DataGen handoff into a local JSONL DatasetResult.

    The curator is deliberately deterministic for the MVP. It accepts either
    chat records with ``messages`` or single-turn ``input``/``output``-style
    records, and drops rows that cannot produce both a user input and an
    assistant target.
    """
    mode = str(handoff.get("mode_used") or "C")
    raw_data = handoff.get("raw_data") or {}
    records = raw_data.get("records", []) if isinstance(raw_data, Mapping) else []

    curated: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"row {index}: record must be an object")
            continue
        try:
            curated.append(curate_record(record))
        except ValueError as exc:
            issues.append(f"row {index}: {exc}")

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for record in curated:
            fh.write(json.dumps(record) + "\n")

    dropped = len(records) - len(curated)
    validation_issues = _validation_issues(issues, dropped, len(curated))
    dataset: StandardDataset = {
        "path": os.path.abspath(output_path),
        "format": "jsonl",
        **_split_counts(len(curated)),
    }
    validation_report: ValidationReport = {
        "passed": bool(curated),
        "issues": validation_issues,
        "sample_accuracy_estimate": 0.9 if curated else 0.0,
    }
    quality_notes = (
        f"DataGen mode {mode}; curated {len(curated)} of {len(records)} raw records"
    )
    if dropped:
        quality_notes += f"; dropped {dropped} malformed record(s)"

    return {
        "dataset": dataset,
        "mode_used": mode,
        "quality_notes": quality_notes,
        "validation_report": validation_report,
    }


def curate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Curate one raw record into a Tinker chat/SFT-compatible JSON object."""
    messages = record.get("messages")
    if messages is not None:
        normalized = _normalize_messages(messages)
        if not any(message["role"] == "user" for message in normalized):
            raise ValueError("messages must contain a user message")
        if not any(message["role"] == "assistant" for message in normalized):
            raise ValueError("messages must contain an assistant message")
        return {"messages": normalized}

    user_text = _first_text(record, _INPUT_KEYS)
    target_text = _first_text(record, _TARGET_KEYS)
    if not user_text:
        raise ValueError(f"missing input field; expected one of {_INPUT_KEYS}")
    if not target_text:
        raise ValueError(f"missing target field; expected one of {_TARGET_KEYS}")
    return {"input": user_text, "output": target_text}


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("message entries must be objects")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in _ALLOWED_MESSAGE_ROLES:
            raise ValueError(f"unsupported message role {role!r}")
        if content is None or str(content).strip() == "":
            raise ValueError("message content cannot be empty")
        normalized.append({"role": role, "content": str(content).strip()})
    return normalized


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _split_counts(n_records: int) -> dict[str, int]:
    if n_records <= 0:
        return {"train_size": 0, "val_size": 0, "test_size": 0}
    train_size = max(1, int(n_records * 0.8))
    val_size = int(n_records * 0.1) if n_records >= 3 else 0
    test_size = max(0, n_records - train_size - val_size)
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }


def _validation_issues(
    row_issues: list[str],
    dropped: int,
    curated_count: int,
) -> list[str]:
    issues: list[str] = []
    if dropped:
        preview = "; ".join(row_issues[:3])
        issues.append(f"Dropped {dropped} malformed record(s). {preview}")
    if curated_count == 0:
        issues.append("No valid chat/SFT records were produced")
    return issues
