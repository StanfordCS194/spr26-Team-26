from __future__ import annotations

from pathlib import Path

from src.data_generator.mode_a import load_raw_data
from src.data_generator.mode_b import (
    build_explicit_hf_candidates,
    fetch_hf_datasets,
    parse_explicit_hf_dataset_ids,
)
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


def build_handoff_node(state: DataGenState) -> dict:
    """
    Final handoff from first sub-agent to second sub-agent.

    This does not structure/clean/split/validate the dataset.
    It only packages acquired raw data + provenance.
    """
    raw_data = state.get("raw_data") or {}

    return {
        "handoff": {
            "target_subagent": "data_curation",
            "action": "structure_data",
            "verification_level": "strict",
            "mode_used": state.get("mode"),
            "raw_data": raw_data,
            "human_readable": raw_data.get("human_readable") or state.get("human_readable"),
            "hf_candidates": state.get("hf_candidates", []),
            "selected_candidate": state.get("selected_candidate"),
            "web_plan": state.get("web_plan"),
            "web_search_results": state.get("web_search_results", []),
            "source_metadata": {
                "data_path": state.get("data_path"),
                "mode": state.get("mode"),
                "raw_format": raw_data.get("format_meta"),
            },
            "config": state.get("config"),
        }
    }