from __future__ import annotations

from src.data_generator.nodes import build_handoff_node


def _base_config() -> dict:
    return {
        "data": False,
        "prompt": "unit test prompt",
        "compute_budget": 10.0,
        "training_procedure": {
            "task_type": "text_classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {},
            "notes": "test",
        },
    }


def test_mode_a_handoff_preserves_raw_and_adds_standardized_curation_payload():
    state = {
        "config": _base_config(),
        "data_path": "/tmp/local.csv",
        "mode": "A",
        "raw_data": {
            "records": [
                {"input": "hello", "output": "world"},
                {
                    "source_path": "/tmp/dir/part.json",
                    "file_type": "json",
                    "modality": "tabular",
                    "records": [{"input": "nested", "output": "record"}],
                },
            ],
            "format_meta": {"modality": "tabular", "file_type": "csv", "encoding": "utf-8"},
        },
        "hf_candidates": [],
        "selected_candidate": None,
        "web_plan": None,
        "web_search_results": [],
        "human_readable": None,
    }

    handoff = build_handoff_node(state)["handoff"]

    assert handoff["raw_data"] == state["raw_data"]
    assert handoff["curation_payload"]["schema_version"] == "data_curation_input.v1"
    assert handoff["curation_payload"]["mode_hint"] == "A"
    assert handoff["curation_payload"]["record_count"] == 2
    assert handoff["curation_payload"]["records"][0]["record_id"].startswith("a_")
    assert handoff["curation_payload"]["records"][1]["source_locator"] == "/tmp/dir/part.json"
    assert "Sub-Agent 2 Curation Input" in handoff["curation_human_readable"]


def test_mode_b_handoff_keeps_hf_artifacts_and_standardizes_records():
    raw_data = {
        "records": [
            {"source": "SetFit/sst2", "split": "train", "input": "great", "label": "positive", "content": "great"},
            {"source": "SetFit/sst2", "note": "fetch_failed", "error": "boom"},
        ],
        "format_meta": {"modality": "text", "file_type": "hf_bundle", "encoding": "utf-8"},
    }
    state = {
        "config": _base_config(),
        "data_path": None,
        "mode": "B",
        "raw_data": raw_data,
        "hf_candidates": [{"id": "x", "name": "SetFit/sst2", "num_examples": 1, "license": "unknown", "task_categories": ["text_classification"], "download_size": 1}],
        "selected_candidate": {"id": "x", "name": "SetFit/sst2", "num_examples": 1, "license": "unknown", "task_categories": ["text_classification"], "download_size": 1},
        "web_plan": None,
        "web_search_results": [],
        "human_readable": None,
    }

    handoff = build_handoff_node(state)["handoff"]

    assert handoff["hf_candidates"] == state["hf_candidates"]
    assert handoff["selected_candidate"] == state["selected_candidate"]
    assert handoff["curation_payload"]["mode_hint"] == "B"
    assert {rec["source_kind"] for rec in handoff["curation_payload"]["records"]} == {"hf_dataset"}
    assert handoff["curation_payload"]["records"][0]["input"] == "great"
    assert handoff["curation_payload"]["records"][1]["input"]


def test_mode_c_handoff_preserves_existing_human_readable_and_standardizes_records():
    raw_report = "Mode C Web Acquisition Report\nCollected sources..."
    raw_data = {
        "records": [
            {
                "source": "web_page",
                "url": "https://example.com/a",
                "title": "Example A",
                "query": "test query",
                "content": "Page content",
                "metadata": {"http_status": 200, "extraction_method": "trafilatura"},
            }
        ],
        "human_readable": raw_report,
        "format_meta": {"modality": "text", "file_type": "web_aggregated_sources", "encoding": "utf-8"},
    }
    state = {
        "config": _base_config(),
        "data_path": None,
        "mode": "C",
        "raw_data": raw_data,
        "hf_candidates": [],
        "selected_candidate": None,
        "web_plan": {"planner_backend": "mock_llm"},
        "web_search_results": [{"url": "https://example.com/a"}],
        "human_readable": raw_report,
    }

    handoff = build_handoff_node(state)["handoff"]

    assert handoff["human_readable"] == raw_report
    assert handoff["curation_payload"]["mode_hint"] == "C"
    assert handoff["curation_payload"]["records"][0]["source_locator"] == "https://example.com/a"
    assert handoff["curation_payload"]["records"][0]["source_kind"] == "web_page"
    assert handoff["source_metadata"]["curation_contract_version"] == "data_curation_input.v1"
