import json
import os

import pytest

from src.data_generator.curation import (
    curate_handoff_to_dataset_result,
    curate_record,
)


def test_curate_record_preserves_valid_messages():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "hi"},
    ]

    assert curate_record({"messages": messages}) == {"messages": messages}


def test_curate_record_converts_input_label_to_output():
    assert curate_record({"input": "Classify this", "label": 0}) == {
        "input": "Classify this",
        "output": "0",
    }


def test_curate_record_rejects_content_without_target():
    with pytest.raises(ValueError, match="missing target"):
        curate_record({"content": "Only source text"})


def test_curate_record_rejects_mode_c_web_source_without_target():
    with pytest.raises(ValueError, match="web source has no assistant target"):
        curate_record(
            {
                "source_kind": "web_page",
                "source_locator": "https://example.com/support",
                "input": "Urgent tickets include outages and security incidents.",
                "output": "",
            },
            mode="C",
            config={"prompt": "classify support tickets by urgency"},
            record_source="curation_payload",
        )


def test_curate_handoff_writes_only_valid_jsonl(tmp_path):
    handoff = {
        "mode_used": "B",
        "raw_data": {
            "records": [
                {"input": "A", "label": "alpha"},
                {"content": "missing target"},
                {
                    "messages": [
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "beta"},
                    ]
                },
            ]
        },
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["mode_used"] == "B"
    assert result["validation_report"]["passed"] is True
    assert result["validation_report"]["issues"]
    assert result["dataset"]["train_size"] == 1
    assert result["dataset"]["test_size"] == 1
    assert os.path.exists(result["dataset"]["path"])

    rows = [
        json.loads(line)
        for line in open(result["dataset"]["path"]).read().splitlines()
        if line.strip()
    ]
    assert rows == [
        {"input": "A", "output": "alpha"},
        {
            "messages": [
                {"role": "user", "content": "B"},
                {"role": "assistant", "content": "beta"},
            ]
        },
    ]


def test_curate_handoff_prefers_mode_c_curation_payload(tmp_path):
    handoff = {
        "mode_used": "C",
        "config": {"prompt": "build a classifier for support ticket urgency"},
        "raw_data": {
            "records": [
                {
                    "content": "raw page should not be used when curation payload exists",
                }
            ]
        },
        "curation_payload": {
            "records": [
                {
                    "record_id": "c_000001",
                    "source_kind": "web_page",
                    "source_locator": "https://example.com/urgency",
                    "input": "Outages and active security incidents are urgent.",
                    "output": "urgent",
                }
            ]
        },
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is True
    assert result["quality_notes"] == (
        "DataGen mode C; curated 1 of 1 curation_payload record(s)"
    )
    rows = [
        json.loads(line)
        for line in open(result["dataset"]["path"]).read().splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "input": "Outages and active security incidents are urgent.",
            "output": "urgent",
        }
    ]


def test_curate_handoff_keeps_validation_split_for_small_datasets(tmp_path):
    handoff = {
        "mode_used": "C",
        "raw_data": {
            "records": [
                {"input": f"ticket {idx}", "output": "normal"}
                for idx in range(6)
            ]
        },
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is True
    assert result["dataset"]["train_size"] == 4
    assert result["dataset"]["val_size"] == 1
    assert result["dataset"]["test_size"] == 1


def test_curate_handoff_preserves_source_split_order_and_counts(tmp_path):
    handoff = {
        "mode_used": "B",
        "raw_data": {
            "records": [
                {"input": "validation row", "output": "v", "split": "validation"},
                {"input": "test row", "output": "t", "metadata": {"split": "test"}},
                {"input": "train row", "output": "tr", "split": "train"},
                {"input": "dev row", "output": "d", "metadata": {"split": "dev"}},
                {"input": "unknown row", "output": "u"},
            ]
        },
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is True
    assert result["dataset"]["train_size"] == 2
    assert result["dataset"]["val_size"] == 2
    assert result["dataset"]["test_size"] == 1
    rows = [
        json.loads(line)
        for line in open(result["dataset"]["path"]).read().splitlines()
        if line.strip()
    ]
    assert rows == [
        {"input": "train row", "output": "tr"},
        {"input": "unknown row", "output": "u"},
        {"input": "validation row", "output": "v"},
        {"input": "dev row", "output": "d"},
        {"input": "test row", "output": "t"},
    ]


def test_curate_handoff_preserves_upstream_validation(tmp_path):
    handoff = {
        "mode_used": "C",
        "raw_data": {"records": [{"input": "A", "output": "alpha"}]},
        "validation_report": {
            "passed": False,
            "issues": ["synthetic quality warning"],
            "sample_accuracy_estimate": 0.42,
        },
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is False
    assert result["validation_report"]["issues"] == ["synthetic quality warning"]
    assert result["validation_report"]["sample_accuracy_estimate"] == 0.42


def test_curate_handoff_fails_validation_when_no_rows_survive(tmp_path):
    handoff = {
        "mode_used": "C",
        "raw_data": {"records": [{"content": "TODO with no target"}]},
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is False
    assert result["dataset"]["train_size"] == 0
    assert "No valid chat/SFT records" in result["validation_report"]["issues"][-1]
