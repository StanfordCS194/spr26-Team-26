from __future__ import annotations

import re
from typing import Any

from src.types import OrchestrationConfig


def build_base_plan(config: OrchestrationConfig) -> dict[str, Any]:
    prompt = str(config.get("prompt", "")).strip()
    tp = config.get("training_procedure", {})
    task_type = str(tp.get("task_type", "")).strip()
    data_format = str(tp.get("data_format", "")).strip()
    base = _clean_query(prompt or "machine learning training data")

    suffixes = [
        "dataset examples",
        "public dataset",
        "benchmark dataset",
        "labeled examples",
        "training data examples",
        "github dataset",
    ]

    queries = []
    for suffix in suffixes:
        q = _clean_query(f"{base} {suffix}")
        if task_type:
            q = _clean_query(f"{q} {task_type}")
        queries.append(q)

    return {
        "planner_backend": "mock_llm:base",
        "task_summary": prompt or "Acquire raw web data for the requested ML task.",
        "data_goal": "Find relevant public raw source material for downstream curation.",
        "task_type": task_type,
        "data_format": data_format,
        "search_queries": _dedupe(queries),
        "preferred_domains": [
            "huggingface.co",
            "github.com",
            "kaggle.com",
            "archive.ics.uci.edu",
            "paperswithcode.com",
            "tensorflow.org",
            "pytorch.org",
            "stanford.edu",
            "berkeley.edu",
        ],
        "avoid_domains": [],
        "quality_criteria": [
            "The page is relevant to the requested ML task.",
            "The page contains concrete data, examples, tables, benchmark descriptions, or dataset documentation.",
            "The page is publicly accessible.",
            "The extracted text is long enough for downstream structuring.",
            "The page has clear provenance via URL and title.",
        ],
        "max_search_results_per_query": 6,
        "max_pages": 12,
        "min_extracted_chars": 500,
    }


def _clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out