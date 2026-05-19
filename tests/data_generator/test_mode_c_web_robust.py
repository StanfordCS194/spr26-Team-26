from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.data_generator.artifacts import save_subagent2_artifacts
from src.data_generator.graph import invoke_data_generator_graph


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(
    _env_truthy("NO_SPEND") or not os.getenv("TAVILY_API_KEY"),
    reason="NO_SPEND=1 or TAVILY_API_KEY missing for real Tavily search",
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
    out_dir = Path("artifacts/data_generator/test_mode_c_web_robust")
    monkeypatch.setenv("DATA_GENERATOR_ARTIFACT_DIR", str(out_dir))

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

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = save_subagent2_artifacts(handoff)
    raw_handoff_path = Path(saved["handoff_payload"])
    debug_context_path = Path(saved["debug_context"])
    source_report_path = Path(saved["source_human_readable"])
    curation_path = out_dir / "curation_human_readable.md"
    curation_path.write_text(handoff.get("curation_human_readable", "") + "\n", encoding="utf-8")

    # Keep the rich source-facing report as the main human-readable artifact.
    human_path = out_dir / "human_readable.md"
    human_path.write_text(handoff["human_readable"], encoding="utf-8")

    manifest_path = Path(saved["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["human_readable"] = str(human_path)
    manifest["files"]["curation_human_readable"] = str(curation_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote robust Mode C raw handoff to: {raw_handoff_path}")
    print(f"Wrote robust Mode C debug context to: {debug_context_path}")
    print(f"Wrote robust human-readable report to: {human_path}")
    print(f"Collected {len(records)} sources.")
    print(f"Source types: {sorted(source_types)}")
    print(f"Domains: {sorted(d for d in domains if d)[:20]}")

    print("\n=== SAVED ARTIFACTS ===")
    print(f"Raw handoff JSON: {raw_handoff_path.resolve()}")
    print(f"Debug context: {debug_context_path.resolve()}")
    print(f"Human report: {human_path.resolve()}")
    print(f"Source report: {source_report_path.resolve()}")
    print(f"Curation summary: {curation_path.resolve()}")
    print(f"Collected {len(records)} sources.")
    print(f"Search results: {meta['num_search_results']}")
    print(f"Pages crawled: {meta['num_pages_crawled']}")
    print(f"Source types: {sorted(source_types)}")
    print(f"Domains: {sorted(d for d in domains if d)[:20]}")
