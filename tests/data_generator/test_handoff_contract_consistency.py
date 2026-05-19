from __future__ import annotations

import json

from src.data_generator.curation import curate_handoff_to_dataset_result
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


def test_mode_a_handoff_preserves_chat_messages_through_curation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    messages = [
        {"role": "system", "content": "Answer with the escalation label."},
        {"role": "user", "content": "Payment service is down for all customers."},
        {"role": "assistant", "content": "urgent"},
    ]
    state = {
        "config": _base_config(),
        "data_path": "/tmp/chat.jsonl",
        "mode": "A",
        "raw_data": {
            "records": [
                {
                    "messages": messages,
                    "source_path": "/tmp/chat.jsonl",
                    "row_index": 1,
                }
            ],
            "format_meta": {"modality": "text", "file_type": "jsonl", "encoding": "utf-8"},
        },
        "hf_candidates": [],
        "selected_candidate": None,
        "web_plan": None,
        "web_search_results": [],
        "human_readable": None,
    }

    handoff = build_handoff_node(state)["handoff"]
    curation_record = handoff["curation_payload"]["records"][0]
    assert curation_record["messages"] == messages

    dataset = curate_handoff_to_dataset_result(handoff)
    rows = [
        json.loads(line)
        for line in open(dataset["dataset"]["path"], encoding="utf-8")
        if line.strip()
    ]
    assert rows == [{"messages": messages}]
    assert dataset["validation_report"]["passed"] is True


def test_mode_a_plain_text_is_source_only_and_fails_curation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "config": _base_config(),
        "data_path": "/tmp/notes.txt",
        "mode": "A",
        "raw_data": {
            "records": [
                {
                    "input": "unlabeled note",
                    "source_path": "/tmp/notes.txt",
                    "row_index": 1,
                }
            ],
            "format_meta": {"modality": "text", "file_type": "txt", "encoding": "utf-8"},
        },
        "hf_candidates": [],
        "selected_candidate": None,
        "web_plan": None,
        "web_search_results": [],
        "human_readable": None,
    }

    handoff = build_handoff_node(state)["handoff"]
    curation_record = handoff["curation_payload"]["records"][0]
    assert curation_record["input"] == "unlabeled note"
    assert curation_record["output"] == ""

    dataset = curate_handoff_to_dataset_result(handoff)

    assert dataset["dataset"]["train_size"] == 0
    assert dataset["validation_report"]["passed"] is False
    assert any("missing target field" in issue for issue in dataset["validation_report"]["issues"])
    assert any("No valid chat/SFT records" in issue for issue in dataset["validation_report"]["issues"])


def test_mode_a_chat_requires_assistant_after_user(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    messages = [
        {"role": "assistant", "content": "urgent"},
        {"role": "user", "content": "Payment service is down."},
    ]
    state = {
        "config": _base_config(),
        "data_path": "/tmp/chat.jsonl",
        "mode": "A",
        "raw_data": {
            "records": [
                {
                    "messages": messages,
                    "source_path": "/tmp/chat.jsonl",
                    "row_index": 1,
                }
            ],
            "format_meta": {"modality": "text", "file_type": "jsonl", "encoding": "utf-8"},
        },
        "hf_candidates": [],
        "selected_candidate": None,
        "web_plan": None,
        "web_search_results": [],
        "human_readable": None,
    }

    handoff = build_handoff_node(state)["handoff"]
    dataset = curate_handoff_to_dataset_result(handoff)

    assert dataset["validation_report"]["passed"] is False
    assert any(
        "assistant message after a user" in issue
        for issue in dataset["validation_report"]["issues"]
    )


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
