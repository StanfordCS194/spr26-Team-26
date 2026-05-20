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
def test_mode_c_web_real_search_crawl_and_artifact():
    config = {
        "prompt": "Build a sentiment classifier for short customer reviews",
        "training_procedure": {
            "task_type": "text_classification",
            "data_format": "raw_text_sources",
            "training_type": "SFT",
        },
    }

    handoff = invoke_data_generator_graph(config=config, data_path=None)

    assert handoff["mode_used"] == "C"
    assert handoff["target_subagent"] == "data_curation"
    assert handoff["action"] == "structure_data"

    raw_data = handoff["raw_data"]
    records = raw_data["records"]

    assert raw_data["format_meta"]["file_type"] == "web_aggregated_sources"
    assert raw_data["format_meta"]["planner_backend"] == "mock_llm"
    assert raw_data["format_meta"]["search_backend"] == "tavily"
    assert len(records) > 0

    for rec in records:
    assert rec["source"] in {"web_page", "web_asset"}
    assert rec["url"]
    assert rec["title"] is not None
    assert rec["content"] is not None
    assert rec["metadata"]

    if rec["source"] == "web_page":
        assert len(rec["content"]) >= 300
        assert rec["metadata"]["extraction_method"] == "trafilatura"

    if rec["source"] == "web_asset":
        assert rec.get("source_type") in {"pdf", "csv", "json", "image"}
        assert rec["metadata"]["extraction_method"] in {
            "asset_metadata_only",
            "direct_text_asset",
        }

    assert handoff["human_readable"]
    assert "Mode C Web Acquisition Report" in handoff["human_readable"]

    out_dir = Path("artifacts/data_generator")
    out_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = out_dir / "mode_c_handoff.json"
    report_path = out_dir / "mode_c_human_report.txt"

    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    report_path.write_text(handoff["human_readable"], encoding="utf-8")

    print(f"\nWrote real Mode C handoff to: {handoff_path}")
    print(f"Wrote human-readable report to: {report_path}")
    print(f"Collected {len(records)} real crawled/extracted pages.")

    print("\n=== FIRST RECORD PREVIEW ===")
    first = records[0]
    print("Title:", first.get("title"))
    print("URL:", first.get("url"))
    print("Content preview:")
    print(first.get("content", "")[:800])