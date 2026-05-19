from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_generator.mode_a import load_raw_data
from src.data_generator.mode_b import (
    build_explicit_hf_candidates,
    fetch_hf_datasets,
    parse_explicit_hf_dataset_ids,
)
from src.data_generator.mode_c import acquire_synthetic_dataset
from src.types import DataGenState


def route_node(state: DataGenState) -> dict:
    """
    First sub-agent router:
    - Mode A: local user data_path exists
    - Mode B: explicit HF dataset IDs/URLs exist
    - Mode C: no local data and no explicit HF source
    """
    path = state.get("data_path")
    if path and Path(path).exists():
        return {"mode": "A"}

    explicit_ids = parse_explicit_hf_dataset_ids(state["config"], path)
    if explicit_ids:
        return {"mode": "B"}

    return {"mode": "C"}


def acquire_user_data_node(state: DataGenState) -> dict:
    data_path = state.get("data_path")
    if not data_path:
        raise ValueError("acquire_user_data_node requires data_path.")

    raw = load_raw_data(data_path)
    return {"raw_data": raw}


def acquire_hf_data_node(state: DataGenState) -> dict:
    config = state["config"]
    task_type = config["training_procedure"].get("task_type", "text_classification")

    source_ids = parse_explicit_hf_dataset_ids(config, state.get("data_path"))
    if not source_ids:
        raise ValueError("Mode B selected but no explicit HF dataset IDs were provided.")

    candidates = build_explicit_hf_candidates(source_ids, task_type)
    raw = fetch_hf_datasets(candidates)

    return {
        "hf_candidates": candidates,
        "selected_candidate": candidates[0] if candidates else None,
        "raw_data": raw,
    }


def acquire_web_data_node(state: DataGenState) -> dict:
    """Synthetic Mode C acquisition node kept for non-web fallback/tests."""
    result = acquire_synthetic_dataset(state["config"])
    return {
        "schema": result.schema,
        "raw_data": result.raw_data,
        "validation_report": result.validation_report,
    }


def build_handoff_node(state: DataGenState) -> dict:
    """
    Final handoff from first sub-agent to second sub-agent.

    This packages acquired raw data, curation-facing normalized records, and
    provenance. The actual trainable JSONL is written by the curation gate.
    """
    raw_data = state.get("raw_data") or {}
    mode = state.get("mode")
    curation_payload = build_curation_payload(state, raw_data)
    curation_human_readable = build_curation_human_readable_report(curation_payload)
    preserved_human_readable = raw_data.get("human_readable") or state.get("human_readable")

    return {
        "handoff": {
            "target_subagent": "data_curation",
            "action": "structure_data",
            "verification_level": "strict",
            "mode_used": mode,
            "curation_payload": curation_payload,
            "curation_human_readable": curation_human_readable,
            "raw_data": raw_data,
            "human_readable": preserved_human_readable,
            "schema": state.get("schema"),
            "validation_report": state.get("validation_report"),
            "hf_candidates": state.get("hf_candidates", []),
            "selected_candidate": state.get("selected_candidate"),
            "web_plan": state.get("web_plan"),
            "web_search_results": state.get("web_search_results", []),
            "mode_c_fallback": state.get("mode_c_fallback"),
            "web_acquisition_error": state.get("web_acquisition_error"),
            "source_metadata": {
                "data_path": state.get("data_path"),
                "mode": mode,
                "mode_c_fallback": state.get("mode_c_fallback"),
                "web_acquisition_error": state.get("web_acquisition_error"),
                "raw_format": raw_data.get("format_meta"),
                "curation_contract_version": curation_payload.get("schema_version"),
                "curation_record_count": len(curation_payload.get("records", [])),
            },
            "config": state.get("config"),
        }
    }


def build_curation_payload(state: DataGenState, raw_data: dict[str, Any]) -> dict[str, Any]:
    mode = str(state.get("mode") or "unknown")
    records = normalize_records_for_curation(mode, raw_data.get("records", []))
    format_meta = raw_data.get("format_meta") or {}

    return {
        "schema_version": "data_curation_input.v1",
        "mode_hint": mode,
        "records": records,
        "record_count": len(records),
        "format_meta": format_meta,
        "modality": format_meta.get("modality"),
        "provenance_summary": {
            "data_path": state.get("data_path"),
            "hf_candidate_count": len(state.get("hf_candidates", [])),
            "web_result_count": len(state.get("web_search_results", [])),
        },
    }


def normalize_records_for_curation(mode: str, records: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    counter = 0

    for item in records:
        expanded_records = _expand_source_record(item)
        for expanded in expanded_records:
            counter += 1
            normalized.append(_normalize_single_record(mode, expanded, counter))

    return normalized


def _expand_source_record(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return [{"raw_value": item}]

    nested_records = item.get("records")
    if isinstance(nested_records, list):
        expanded: list[dict[str, Any]] = []
        for nested in nested_records:
            if isinstance(nested, dict):
                merged = dict(nested)
            else:
                merged = {"raw_value": nested}
            merged.setdefault("source_path", item.get("source_path"))
            merged.setdefault("container_file_type", item.get("file_type"))
            merged.setdefault("container_modality", item.get("modality"))
            expanded.append(merged)
        return expanded

    return [item]


def _normalize_single_record(mode: str, record: dict[str, Any], index: int) -> dict[str, Any]:
    input_value = _coerce_primary_input(record)
    output_value = _coerce_primary_output(record)
    source_locator = _coerce_source_locator(record)

    metadata = {
        "mode": mode,
        "source": record.get("source"),
        "source_type": record.get("source_type"),
        "source_path": record.get("source_path") or record.get("path"),
        "url": record.get("url"),
        "title": record.get("title"),
        "split": record.get("split"),
        "query": record.get("query"),
        "note": record.get("note"),
        "domain": record.get("domain"),
        "container_file_type": record.get("container_file_type"),
        "container_modality": record.get("container_modality"),
        "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    }

    return {
        "record_id": f"{mode.lower()}_{index:06d}",
        "input": input_value,
        "output": output_value,
        "content": str(record.get("content") or input_value),
        "source_locator": source_locator,
        "source_kind": _infer_source_kind(mode, record),
        "metadata": metadata,
    }


def _coerce_primary_input(record: dict[str, Any]) -> str:
    for key in (
        "input",
        "text",
        "sentence",
        "question",
        "prompt",
        "utterance",
        "content",
        "title",
        "path",
        "url",
    ):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    raw_value = record.get("raw_value")
    if raw_value is not None:
        return str(raw_value)

    return json.dumps(record, ensure_ascii=True, sort_keys=True)[:4000]


def _coerce_primary_output(record: dict[str, Any]) -> str:
    for key in ("output", "label", "label_text", "intent", "target", "answer"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _coerce_source_locator(record: dict[str, Any]) -> str:
    for key in ("url", "source_path", "path", "source", "filename", "title"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _infer_source_kind(mode: str, record: dict[str, Any]) -> str:
    if mode == "A":
        if record.get("path") or record.get("source_path"):
            return "local"
        return "local_structured"
    if mode == "B":
        return "hf_dataset"
    if record.get("source_type"):
        return str(record.get("source_type"))
    return str(record.get("source") or "web")


def build_curation_human_readable_report(curation_payload: dict[str, Any]) -> str:
    lines = [
        "Sub-Agent 2 Curation Input",
        f"Schema version: {curation_payload.get('schema_version', 'unknown')}",
        f"Mode hint: {curation_payload.get('mode_hint', 'unknown')}",
        f"Modality: {curation_payload.get('modality', 'unknown')}",
        f"Record count: {curation_payload.get('record_count', 0)}",
        "",
        "Sample normalized records:",
    ]

    sample_records = curation_payload.get("records", [])[:5]
    if not sample_records:
        lines.append("- (none)")
        return "\n".join(lines)

    for idx, rec in enumerate(sample_records, start=1):
        metadata = rec.get("metadata", {}) if isinstance(rec, dict) else {}
        lines.append(
            f"- {idx}. id={rec.get('record_id')} | source={rec.get('source_locator')} | "
            f"kind={rec.get('source_kind')} | output={str(rec.get('output', ''))[:80]}"
        )
        lines.append(
            f"  input={str(rec.get('input', ''))[:200].replace(chr(10), ' ')}"
        )
        if metadata.get("note"):
            lines.append(f"  note={metadata.get('note')}")

    return "\n".join(lines)
