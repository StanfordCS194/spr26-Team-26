"""
Feature 1 — Data Generator
Owner: Ron Polonsky, Angel Raychev

First sub-agent façade: acquisition + handoff to second sub-agent.
"""

from __future__ import annotations

from src.data_generator.curation import curate_handoff_to_dataset_result, curate_record
from src.data_generator.edges import select_mode_edge
from src.data_generator.graph import build_data_generator_graph, invoke_data_generator_graph
from src.data_generator.mode_a import detect_format, load_raw_data
from src.data_generator.mode_b import (
    build_explicit_hf_candidates,
    fetch_hf_datasets,
    parse_explicit_hf_dataset_ids,
)
from src.data_generator.mode_c import (
    acquire_synthetic_dataset,
    acquire_web_data,
    build_mode_c_dataset,
    determine_data_schema,
    generate_synthetic_data,
    infer_schema_without_teacher,
    plan_synthetic_generation,
    validate_synthetic_records,
)
from src.data_generator.nodes import (
    acquire_hf_data_node,
    acquire_user_data_node,
    acquire_web_data_node,
    build_handoff_node,
    route_node,
)

__all__ = [
    "build_data_generator_graph",
    "invoke_data_generator_graph",
    "route_node",
    "acquire_user_data_node",
    "acquire_hf_data_node",
    "acquire_web_data_node",
    "build_handoff_node",
    "select_mode_edge",
    "load_raw_data",
    "detect_format",
    "parse_explicit_hf_dataset_ids",
    "build_explicit_hf_candidates",
    "fetch_hf_datasets",
    "acquire_synthetic_dataset",
    "acquire_web_data",
    "build_mode_c_dataset",
    "determine_data_schema",
    "generate_synthetic_data",
    "infer_schema_without_teacher",
    "plan_synthetic_generation",
    "validate_synthetic_records",
    "curate_record",
    "curate_handoff_to_dataset_result",
]
