from __future__ import annotations

import re
from typing import Any

from src.types import OrchestrationConfig


def build_mixed_sources_plan(config: OrchestrationConfig) -> dict[str, Any]:
    prompt = str(config.get("prompt", "")).strip()
    tp = config.get("training_procedure", {})
    task_type = str(tp.get("task_type", "")).strip()
    data_format = str(tp.get("data_format", "")).strip()
    base = re.sub(r"\s+", " ", prompt or "machine learning mixed source data").strip()

    return {
        "planner_backend": "mock_llm:mixed_sources",
        "task_summary": prompt or "Acquire mixed web data sources for ML training.",
        "data_goal": "Find diverse raw source materials: HTML pages, datasets, GitHub repos, PDFs, CSVs, JSON, and image-heavy pages.",
        "task_type": task_type,
        "data_format": data_format,
        "search_queries": [
            f"{base} dataset csv",
            f"{base} filetype:csv dataset",
            f"{base} github dataset",
            f"{base} filetype:pdf dataset paper benchmark",
            f"{base} image dataset examples",
            f"{base} public benchmark dataset documentation",
        ],
        "preferred_domains": [
            "huggingface.co",
            "github.com",
            "kaggle.com",
            "archive.ics.uci.edu",
            "paperswithcode.com",
            "data.gov",
            "stanford.edu",
            "berkeley.edu",
        ],
        "avoid_domains": [],
        "quality_criteria": [
            "Relevant to the requested task.",
            "Contains raw data, links to raw data, benchmark docs, tables, PDFs, CSVs, JSON, or image datasets.",
            "Has clear URL provenance.",
            "Useful for downstream structuring by the data curation sub-agent.",
        ],
        "max_search_results_per_query": 5,
        "max_pages": 15,
        "min_extracted_chars": 200,
    }