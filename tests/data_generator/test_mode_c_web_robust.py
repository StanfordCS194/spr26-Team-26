from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.data_generator.graph import invoke_data_generator_graph


@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="Requires TAVILY_API_KEY for real Tavily search",
)
def test_mode_c_mixed_sources_real_search_crawl_and_artifact(monkeypatch):
    """
    Robust Mode C integration test.

    Mocked:
      - LLM planner output only

    Real:
      - Tavily search
      - HTTP fetching
      - HTML text extraction
      - asset URL preservation
      - handoff artifact generation
    """
    monkeypatch.setenv("DATA_GENERATOR_MOCK_SCENARIO", "mixed_sources")

    config = {
        "prompt": "Build a dataset for classifying food images and food review sentiment",
        "training_procedure": {
            "task_type": "multimodal_classification",
            "data_format": "raw_mixed_sources",
            "training_type": "SFT",
        },
    }

    handoff = invoke_data_generator_graph(config=config, data_path=None)

    assert handoff["mode_used"] == "C"
    assert handoff["target_subagent"] == "data_curation"
    assert handoff["action"] == "structure_data"

    raw_data = handoff["raw_data"]
    records = raw_data["records"]
    meta = raw_data["format_meta"]

    assert meta["file_type"] == "web_aggregated_sources"
    assert meta["planner_backend"] == "mock_llm"
    assert meta["search_backend"] == "tavily"
    assert meta["num_search_results"] > 0
    assert len(records) > 0

    # Basic record contract.
    for rec in records:
        assert rec.get("url")
        assert rec.get("title") is not None
        assert rec.get("query")
        assert rec.get("metadata")
        assert 200 <= rec["metadata"].get("http_status", 0) < 300
        assert rec["metadata"].get("extraction_method") in {
            "trafilatura",
            "pymupdf",
            "direct_text_asset",
            "image_metadata_only",
        }

    # We should preserve different kinds of sources when Tavily finds them.
    source_types = {rec.get("source_type", "html") for rec in records}

    # Do not make this too brittle: Tavily results change.
    # But require at least normal HTML plus one extra asset-ish or GitHub/dataset source.
    assert "html" in source_types or any(rec.get("source") == "web_page" for rec in records)

    domains = {rec.get("domain", "") for rec in records}
    joined_domains = " ".join(domains).lower()
    joined_urls = " ".join(rec.get("url", "").lower() for rec in records)

    assert any(
        token in joined_domains or token in joined_urls
        for token in [
            "github",
            "huggingface",
            "kaggle",
            "archive.ics.uci.edu",
            "paperswithcode",
            "data.gov",
            ".csv",
            ".pdf",
        ]
    )

    assert handoff["human_readable"]
    assert "Mode C Web Acquisition Report" in handoff["human_readable"]

    out_dir = Path("artifacts/data_generator")
    out_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = out_dir / "mode_c_mixed_sources_handoff.json"
    report_path = out_dir / "mode_c_mixed_sources_report.txt"

    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    report_path.write_text(handoff["human_readable"], encoding="utf-8")

    print(f"\nWrote robust Mode C handoff to: {handoff_path}")
    print(f"Wrote robust human-readable report to: {report_path}")
    print(f"Collected {len(records)} sources.")
    print(f"Source types: {sorted(source_types)}")
    print(f"Domains: {sorted(d for d in domains if d)[:20]}")

    print("\n=== SAVED ARTIFACTS ===")
    print(f"Handoff JSON: {handoff_path.resolve()}")
    print(f"Human report: {report_path.resolve()}")
    print(f"Collected {len(records)} sources.")
    print(f"Search results: {meta['num_search_results']}")
    print(f"Pages crawled: {meta['num_pages_crawled']}")
    print(f"Source types: {sorted(source_types)}")
    print(f"Domains: {sorted(d for d in domains if d)[:20]}")