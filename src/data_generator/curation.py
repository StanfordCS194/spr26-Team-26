"""Data curation helpers for turning DataGen handoffs into trainable JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime_context import get_output_root
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
    config = handoff.get("config") if isinstance(handoff.get("config"), Mapping) else {}
    records, record_source = _records_from_handoff(handoff)

    curated: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"row {index}: record must be an object")
            continue
        try:
            curated.append(
                curate_record(
                    record,
                    mode=mode,
                    config=config,
                    record_source=record_source,
                )
            )
        except ValueError as exc:
            issues.append(f"row {index}: {exc}")

    output_path = _curated_output_path(output_dir, filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for record in curated:
            fh.write(json.dumps(record) + "\n")

    dropped = len(records) - len(curated)
    validation_issues = _validation_issues(
        issues,
        dropped,
        len(curated),
        handoff.get("validation_report"),
    )
    upstream_validation = handoff.get("validation_report")
    upstream_passed = (
        bool(upstream_validation.get("passed", True))
        if isinstance(upstream_validation, Mapping)
        else True
    )
    sample_accuracy = (
        float(upstream_validation.get("sample_accuracy_estimate", 0.9))
        if isinstance(upstream_validation, Mapping)
        else (0.9 if curated else 0.0)
    )
    dataset: StandardDataset = {
        "path": os.path.abspath(output_path),
        "format": "jsonl",
        **_split_counts(len(curated)),
    }
    validation_report: ValidationReport = {
        "passed": bool(curated) and upstream_passed,
        "issues": validation_issues,
        "sample_accuracy_estimate": sample_accuracy if curated else 0.0,
    }
    quality_notes = (
        f"DataGen mode {mode}; curated {len(curated)} of {len(records)} "
        f"{record_source} record(s)"
    )
    if dropped:
        quality_notes += f"; dropped {dropped} malformed record(s)"

    return {
        "dataset": dataset,
        "mode_used": mode,
        "quality_notes": quality_notes,
        "validation_report": validation_report,
    }


def _curated_output_path(output_dir: str, filename: str) -> Path:
    root = get_output_root()
    if root is not None and output_dir == "outputs/datasets":
        return root / "datasets" / filename
    return Path(output_dir) / filename


def curate_record(
    record: Mapping[str, Any],
    *,
    mode: str | None = None,
    config: Mapping[str, Any] | None = None,
    record_source: str = "raw_data",
) -> dict[str, Any]:
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
        if _is_mode_c_web_record(record, mode, record_source):
            raise ValueError(
                "Mode C web source has no assistant target; run synthetic "
                "structuring before training"
            )
        raise ValueError(f"missing target field; expected one of {_TARGET_KEYS}")
    return {"input": user_text, "output": target_text}


def _records_from_handoff(handoff: Mapping[str, Any]) -> tuple[list[Any], str]:
    curation_payload = handoff.get("curation_payload")
    if isinstance(curation_payload, Mapping):
        records = curation_payload.get("records")
        if isinstance(records, list):
            return records, "curation_payload"

    raw_data = handoff.get("raw_data") or {}
    if isinstance(raw_data, Mapping) and isinstance(raw_data.get("records"), list):
        return raw_data["records"], "raw_data"

    return [], "raw_data"


def _is_mode_c_web_record(
    record: Mapping[str, Any],
    mode: str | None,
    record_source: str,
) -> bool:
    if mode != "C":
        return False
    source_kind = str(record.get("source_kind") or record.get("source_type") or "")
    source = str(record.get("source") or "")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    if record_source == "curation_payload" and source_kind in {
        "web",
        "web_page",
        "web_asset",
        "html",
        "pdf",
        "csv",
        "json",
        "image",
    }:
        return True
    if source in {"web_page", "web_asset", "web_search", "mode_c_web_acquisition"}:
        return True
    if record.get("url") or record.get("source_locator"):
        return True
    return str(metadata.get("extraction_method") or "") != ""


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
    upstream_validation: Any = None,
) -> list[str]:
    issues: list[str] = []
    if isinstance(upstream_validation, Mapping):
        upstream_issues = upstream_validation.get("issues", [])
        if isinstance(upstream_issues, list):
            issues.extend(str(issue) for issue in upstream_issues if str(issue).strip())
    if dropped:
        preview = "; ".join(row_issues[:3])
        issues.append(f"Dropped {dropped} malformed record(s). {preview}")
    if curated_count == 0:
        issues.append("No valid chat/SFT records were produced")
    return issues
