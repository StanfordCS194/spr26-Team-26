"""
Feature 1 — Data Generator
Owner: Ron Polonsky, Angel Raychev

First sub-agent façade: acquisition + handoff to second sub-agent.
"""

from __future__ import annotations

from src.data_generator.edges import select_curation_edge, select_mode_edge
from src.data_generator.graph import build_data_generator_graph, invoke_data_generator_graph
from src.data_generator.mode_a import detect_format, load_raw_data
from src.data_generator.mode_b import build_explicit_hf_candidates, fetch_hf_datasets, parse_explicit_hf_dataset_ids
from src.data_generator.mode_c import acquire_web_data
from src.data_generator.nodes import (
    acquire_hf_data_node,
    acquire_user_data_node,
    acquire_web_data_node,
    handoff_structure_data_node,
    handoff_validate_hf_node,
    route_node,
)

__all__ = [
    "build_data_generator_graph",
    "invoke_data_generator_graph",
    "route_node",
    "acquire_user_data_node",
    "acquire_hf_data_node",
    "acquire_web_data_node",
    "handoff_structure_data_node",
    "handoff_validate_hf_node",
    "select_mode_edge",
    "select_curation_edge",
    "load_raw_data",
    "detect_format",
    "parse_explicit_hf_dataset_ids",
    "build_explicit_hf_candidates",
    "fetch_hf_datasets",
    "acquire_web_data",
]
