from __future__ import annotations

from pathlib import Path

from src.data_generator.mode_a import load_raw_data
from src.data_generator.mode_b import build_explicit_hf_candidates, fetch_hf_datasets, parse_explicit_hf_dataset_ids
from src.data_generator.mode_c import acquire_web_data
from src.types import DataGenState


def route_node(state: DataGenState) -> dict:
    """
    First sub-agent router:
    - Mode A: user data_path exists
    - Mode B: explicit HF dataset IDs exist
    - Mode C: no user data and no explicit HF IDs
    """
    path = state.get("data_path")
    if path and Path(path).exists():
        return {"mode": "A"}

    explicit_ids = parse_explicit_hf_dataset_ids(state["config"], path)
    if explicit_ids:
        return {"mode": "B"}
    return {"mode": "C"}


def acquire_user_data_node(state: DataGenState) -> dict:
    """Acquisition for Mode A."""
    data_path = state.get("data_path")
    if not data_path:
        raise ValueError("acquire_user_data_node requires data_path.")
    raw = load_raw_data(data_path)
    return {"raw_data": raw}


def acquire_hf_data_node(state: DataGenState) -> dict:
    """Acquisition for Mode B from explicit HF dataset IDs."""
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
    """Acquisition backbone for Mode C (placeholder logic for now)."""
    query = str(state["config"].get("prompt", "")).strip() or "generic task"
    raw = acquire_web_data(query)
    return {"raw_data": raw}


def handoff_structure_data_node(state: DataGenState) -> dict:
    """Handoff payload for second sub-agent: structure data."""
    return {
        "handoff": {
            "target_subagent": "data_curation",
            "action": "structure_data",
            "verification_level": "strict",
            "mode_used": state.get("mode", "A"),
            "raw_data": state.get("raw_data"),
            "config": state.get("config"),
        }
    }


def handoff_validate_hf_node(state: DataGenState) -> dict:
    """Handoff payload for second sub-agent: validate HF dataset."""
    return {
        "handoff": {
            "target_subagent": "data_curation",
            "action": "validate_hf_dataset",
            "verification_level": "light",
            "mode_used": "B",
            "raw_data": state.get("raw_data"),
            "hf_candidates": state.get("hf_candidates", []),
            "config": state.get("config"),
        }
    }
