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


def test_curate_handoff_fails_validation_when_no_rows_survive(tmp_path):
    handoff = {
        "mode_used": "C",
        "raw_data": {"records": [{"content": "TODO with no target"}]},
    }

    result = curate_handoff_to_dataset_result(handoff, output_dir=str(tmp_path))

    assert result["validation_report"]["passed"] is False
    assert result["dataset"]["train_size"] == 0
    assert "No valid chat/SFT records" in result["validation_report"]["issues"][-1]
