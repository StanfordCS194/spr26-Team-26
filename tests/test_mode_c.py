from src.data_generator.mode_c import acquire_web_data
from src.data_generator.nodes import acquire_web_data_node


def _config():
    return {
        "data": False,
        "prompt": "classify support tickets by urgency",
        "compute_budget": 25.0,
        "training_procedure": {
            "task_type": "text-classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {},
            "notes": "",
        },
    }


def test_mode_c_generates_chat_sft_records_without_placeholders():
    raw = acquire_web_data("classify support tickets by urgency", _config())

    assert raw["format_meta"]["file_type"] == "synthetic_chat_jsonl"
    assert len(raw["records"]) >= 8
    assert all(record["source"] == "mode_c_synthetic" for record in raw["records"])
    assert not any("TODO" in str(record) for record in raw["records"])
    for record in raw["records"]:
        messages = record["messages"]
        assert any(message["role"] == "user" for message in messages)
        assert any(message["role"] == "assistant" for message in messages)


def test_acquire_web_data_node_passes_task_config_to_mode_c():
    out = acquire_web_data_node(
        {
            "config": _config(),
            "data_path": None,
            "mode": "C",
            "raw_data": None,
            "hf_candidates": [],
            "selected_candidate": None,
            "schema": None,
            "dataset": None,
            "validation_report": None,
            "handoff": None,
        }
    )

    records = out["raw_data"]["records"]
    assert records
    assert {record["task_type"] for record in records} == {"text-classification"}
