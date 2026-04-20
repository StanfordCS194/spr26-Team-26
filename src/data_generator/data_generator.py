"""
Feature 1 — Data Generator
Owner: Ron Polonsky, Angel Raychev

LangGraph StateGraph(DataGenState) with conditional routing:
  route_node → mode_a_node | mode_b_search_node → mode_b_download_node | mode_c_node → validate_node
"""

from __future__ import annotations

from typing import Literal

from src.types import (
    DataGenState,
    DataSchema,
    DatasetResult,
    HFCandidate,
    OrchestrationConfig,
    RawData,
    StandardDataset,
    ValidationReport,
)


def build_data_generator_graph():
    """Constructs and compiles the Data Generator LangGraph StateGraph. Called once at startup."""
    raise NotImplementedError


def invoke_data_generator_graph(
    config: OrchestrationConfig,
    data_path: str | None,
) -> DatasetResult:
    """Entry point called by the Manager's orchestrate_node. Returns DatasetResult."""
    raise NotImplementedError


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def route_node(state: DataGenState) -> dict:
    """LangGraph node. Determines mode (A/B/C) from state.data_path. Returns: { mode }."""
    raise NotImplementedError


def mode_a_node(state: DataGenState) -> dict:
    """LangGraph node (Mode A). Loads and cleans user-provided data. Returns: { raw_data, dataset }."""
    raise NotImplementedError


def mode_b_search_node(state: DataGenState) -> dict:
    """LangGraph node. Searches HuggingFace Hub. Returns: { hf_candidates, selected_candidate }."""
    raise NotImplementedError


def mode_b_download_node(state: DataGenState) -> dict:
    """LangGraph node. Downloads and normalizes the selected HF dataset. Returns: { raw_data, dataset }."""
    raise NotImplementedError


def mode_c_node(state: DataGenState) -> dict:
    """LangGraph node. Generates synthetic data via Claude API (or scrapes web). Returns: { schema, raw_data, dataset }."""
    raise NotImplementedError


def validate_node(state: DataGenState) -> dict:
    """LangGraph node. Runs validate_dataset(). Final node before END. Returns: { validation_report }."""
    raise NotImplementedError


# ─── CONDITIONAL EDGE FUNCTIONS ───────────────────────────────────────────────

def select_mode_edge(state: DataGenState) -> Literal["mode_a", "mode_b_search", "mode_c"]:
    """Conditional edge after route_node. Reads state.mode and returns the next node name."""
    raise NotImplementedError


def hf_found_edge(state: DataGenState) -> Literal["mode_b_download", "mode_c"]:
    """Conditional edge after mode_b_search_node. Returns 'mode_b_download' if candidate found, else 'mode_c'."""
    raise NotImplementedError


# ─── MODE A HELPERS ───────────────────────────────────────────────────────────

def load_raw_data(data_path: str) -> RawData:
    """Loads data from a file path, auto-detecting format (CSV, JSON, JSONL, Parquet, image dir, etc.)."""
    raise NotImplementedError


def detect_format(data_path: str):
    """Inspects file extension, magic bytes, and sample rows to determine format and modality."""
    raise NotImplementedError


def normalize_and_clean(raw: RawData, schema: DataSchema) -> StandardDataset:
    """Normalizes fields, removes nulls/dupes, and reindexes into standard {input, output, split} format."""
    raise NotImplementedError


def augment_with_synthetic(
    dataset: StandardDataset,
    n_extra: int,
    schema: DataSchema,
) -> StandardDataset:
    """Augments an existing dataset with LLM-generated synthetic examples when < 500 training examples."""
    raise NotImplementedError


# ─── MODE B HELPERS ───────────────────────────────────────────────────────────

def search_huggingface(task_description: str, task_type: str) -> list[HFCandidate]:
    """Queries HuggingFace Hub datasets API. Returns up to 10 ranked candidates."""
    raise NotImplementedError


def rank_hf_candidates(
    candidates: list[HFCandidate],
    config: OrchestrationConfig,
) -> HFCandidate | None:
    """Scores candidates by relevance, size, license, download size. Returns best or None."""
    raise NotImplementedError


def download_hf_dataset(candidate: HFCandidate) -> RawData:
    """Downloads the selected HuggingFace dataset to disk and loads it into a RawData object."""
    raise NotImplementedError


def validate_dataset(dataset: StandardDataset, schema: DataSchema) -> ValidationReport:
    """Checks label accuracy, distribution coverage, and completeness. Returns pass/fail report."""
    raise NotImplementedError


# ─── MODE C HELPERS ───────────────────────────────────────────────────────────

def determine_data_schema(config: OrchestrationConfig) -> DataSchema:
    """Uses Claude API to infer input/output schema from OrchestrationConfig training_procedure."""
    raise NotImplementedError


def generate_synthetic_data(
    schema: DataSchema,
    n_examples: int,
    teacher_model: str = "claude-haiku-4-5-20251001",
) -> RawData:
    """Generates n_examples synthetic (input, output) pairs using an LLM teacher."""
    raise NotImplementedError


def scrape_web(query: str, schema: DataSchema, max_examples: int = 500) -> RawData:
    """Fallback web scraping path. Returns raw scraped data ready for morph_to_standard."""
    raise NotImplementedError


def morph_to_standard(raw: RawData, schema: DataSchema) -> StandardDataset:
    """Transforms scraped/generated raw data into standard {input, output, split} format."""
    raise NotImplementedError
