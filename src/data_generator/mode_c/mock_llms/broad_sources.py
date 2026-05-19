from __future__ import annotations

import re
from typing import Any

from src.types import OrchestrationConfig


def build_broad_sources_plan(config: OrchestrationConfig) -> dict[str, Any]:
    prompt = str(config.get("prompt", "")).strip()
    tp = config.get("training_procedure", {})
    task_type = str(tp.get("task_type", "")).strip()
    data_format = str(tp.get("data_format", "")).strip()
    base = re.sub(r"\s+", " ", prompt or "public open data machine learning").strip()

    return {
        "planner_backend": "mock_llm:broad_sources",
        "task_summary": prompt or "Acquire broad public data sources.",
        "data_goal": (
            "Find diverse raw public data sources across government portals, academic PDFs, "
            "CSV/JSON files, image datasets, documentation pages, and open data catalogs. "
            "Do not structure examples yet."
        ),
        "task_type": task_type,
        "data_format": data_format,
        "search_queries": [
            f"{base} site:data.gov csv",
            f"{base} site:catalog.data.gov json dataset",
            f"{base} site:archive.ics.uci.edu dataset",
            f"{base} site:zenodo.org dataset",
            f"{base} site:figshare.com dataset",
            f"{base} filetype:pdf dataset benchmark",
            f"{base} filetype:csv raw data",
            f"{base} image dataset examples",
        ],
        "preferred_domains": [
            "data.gov",
            "catalog.data.gov",
            "archive.ics.uci.edu",
            "zenodo.org",
            "figshare.com",
            "openml.org",
            "paperswithcode.com",
            "stanford.edu",
        ],
        "avoid_domains": [],
        "quality_criteria": [
            "Publicly accessible.",
            "Contains raw downloadable data or concrete dataset documentation.",
            "Preserves provenance via URL/domain/title.",
            "Useful for downstream data curation.",
        ],
        "max_search_results_per_query": 4,
        "max_pages": 18,
        "min_extracted_chars": 100,
    }