import json
from types import SimpleNamespace

import pytest

from src.data_generator.mode_c.nodes import aggregate_web_sources_node
from src.data_generator.mode_c.structuring import (
    WebStructuringResult,
    structure_web_sources_for_sft,
)


def _config():
    return {
        "data": False,
        "prompt": "classify support tickets by urgency",
        "compute_budget": 10.0,
        "training_procedure": {
            "task_type": "text-classification",
            "data_format": "chat jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {},
            "notes": "",
        },
    }


def _pages():
    return [
        {
            "source": "web_page",
            "url": "https://example.com/urgency",
            "title": "Support urgency guide",
            "content": (
                "Urgent support tickets include production outages, active "
                "security incidents, and customer data loss."
            ),
            "metadata": {"extraction_method": "trafilatura"},
        }
    ]


class _FakeTeacher:
    def __init__(self):
        self.messages = _FakeMessages()


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "Infer a supervised fine-tuning data schema" in prompt:
            payload = {
                "input_format": "support ticket text",
                "output_format": "one of: urgent, normal",
                "input_description": "A support ticket.",
                "output_description": "The urgency label.",
                "example_pair": {
                    "input": "The checkout service is down for all users.",
                    "output": "urgent",
                },
            }
        else:
            payload = [
                {
                    "input": "The production checkout service is down for all users.",
                    "output": "urgent",
                    "messages": [
                        {
                            "role": "user",
                            "content": "The production checkout service is down for all users.",
                        },
                        {"role": "assistant", "content": "urgent"},
                    ],
                    "source_url": "https://example.com/urgency",
                },
                {
                    "input": "A user asks how to change their profile photo.",
                    "output": "normal",
                    "messages": [
                        {
                            "role": "user",
                            "content": "A user asks how to change their profile photo.",
                        },
                        {"role": "assistant", "content": "normal"},
                    ],
                    "source_url": "https://example.com/urgency",
                },
            ]
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class _RepairingTeacher:
    def __init__(self):
        self.messages = _RepairingMessages()


class _RepairingMessages(_FakeMessages):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "Infer a supervised fine-tuning data schema" in prompt:
            payload = {
                "input_format": "support ticket text",
                "output_format": "one of: urgent, normal",
                "input_description": "A support ticket.",
                "output_description": "The urgency label.",
                "example_pair": {
                    "input": "The checkout service is down for all users.",
                    "output": "urgent",
                },
            }
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])

        if "Repair this teacher response into valid JSON" in prompt:
            payload = [
                {
                    "input": "The checkout service is down for all users.",
                    "output": "urgent",
                    "messages": [
                        {
                            "role": "user",
                            "content": "The checkout service is down for all users.",
                        },
                        {"role": "assistant", "content": "urgent"},
                    ],
                    "source_url": "https://example.com/urgency",
                },
                {
                    "input": "A customer asks how to update a profile photo.",
                    "output": "normal",
                    "messages": [
                        {
                            "role": "user",
                            "content": "A customer asks how to update a profile photo.",
                        },
                        {"role": "assistant", "content": "normal"},
                    ],
                    "source_url": "https://example.com/urgency",
                }
            ]
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])

        bad_json = (
            "[{\"input\": \"The checkout service is down\", "
            "output: \"urgent\", "
            "\"messages\": [{\"role\": \"user\", \"content\": \"The checkout service is down\"}, "
            "{\"role\": \"assistant\", \"content\": \"urgent\"}]}]"
        )
        return SimpleNamespace(content=[SimpleNamespace(text=bad_json)])


def test_structure_web_sources_with_teacher_returns_trainable_records(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    teacher = _FakeTeacher()

    result = structure_web_sources_for_sft(
        _config(),
        _pages(),
        teacher_client=teacher,
        max_records=4,
    )

    assert result.validation_report["passed"] is True
    assert result.teacher_used is True
    assert result.raw_data["format_meta"]["file_type"] == "web_structured_chat_jsonl"
    assert result.raw_data["format_meta"]["teacher_used"] is True
    assert len(result.raw_data["records"]) == 2
    assert {record["output"] for record in result.raw_data["records"]} == {
        "urgent",
        "normal",
    }
    assert all(record["messages"] for record in result.raw_data["records"])
    assert result.raw_data["format_meta"]["requested_records"] == 4
    assert len(teacher.messages.calls) == 2


def test_structure_web_sources_repairs_malformed_teacher_json(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    teacher = _RepairingTeacher()

    result = structure_web_sources_for_sft(
        _config(),
        _pages(),
        teacher_client=teacher,
        max_records=4,
    )

    assert result.validation_report["passed"] is True
    assert result.raw_data["format_meta"]["teacher_repair_used"] is True
    assert len(result.raw_data["records"]) == 2
    assert result.raw_data["records"][0]["output"] == "urgent"
    assert len(teacher.messages.calls) == 3
    repair_call = teacher.messages.calls[-1]
    assert repair_call["temperature"] == 0
    assert "at most 4 objects" in repair_call["messages"][0]["content"]


def test_structure_web_sources_respects_synthetic_example_cap(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_EXAMPLES", "8")
    teacher = _FakeTeacher()

    result = structure_web_sources_for_sft(
        _config(),
        _pages(),
        teacher_client=teacher,
    )

    structuring_call = teacher.messages.calls[-1]
    assert result.raw_data["format_meta"]["requested_records"] == 8
    assert "at most 8 objects" in structuring_call["messages"][0]["content"]


def test_structure_web_sources_no_spend_does_not_use_teacher(monkeypatch):
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-should-not-be-used")
    teacher = _FakeTeacher()

    result = structure_web_sources_for_sft(
        _config(),
        _pages(),
        teacher_client=teacher,
        max_records=4,
    )

    assert teacher.messages.calls == []
    assert result.teacher_used is False
    assert result.validation_report["passed"] is False
    assert result.raw_data["format_meta"]["teacher_used"] is False
    assert any("NO_SPEND=1" in issue for issue in result.validation_report["issues"])


def test_structure_web_sources_prefers_web_structuring_cap(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_EXAMPLES", "12")
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING_MAX_RECORDS", "5")
    teacher = _FakeTeacher()

    result = structure_web_sources_for_sft(
        _config(),
        _pages(),
        teacher_client=teacher,
    )

    assert result.raw_data["format_meta"]["requested_records"] == 5


def test_structure_web_sources_uses_config_example_cap(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    config = _config()
    config["training_procedure"]["hyperparameters"]["synthetic_examples"] = 9
    teacher = _FakeTeacher()

    result = structure_web_sources_for_sft(
        config,
        _pages(),
        teacher_client=teacher,
    )

    assert result.raw_data["format_meta"]["requested_records"] == 9


def test_aggregate_web_sources_uses_structured_records_when_required(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "required")
    from src.data_generator.mode_c import nodes as mode_c_nodes

    schema = {
        "input_format": "support ticket text",
        "output_format": "one of: urgent, normal",
        "input_description": "A support ticket.",
        "output_description": "The urgency label.",
        "example_pair": {"input": "The site is down.", "output": "urgent"},
    }
    structured_raw = {
        "records": [
            {
                "input": "The site is down.",
                "output": "urgent",
                "messages": [
                    {"role": "user", "content": "The site is down."},
                    {"role": "assistant", "content": "urgent"},
                ],
            }
        ],
        "format_meta": {
            "modality": "text",
            "file_type": "web_structured_chat_jsonl",
            "encoding": "utf-8",
            "schema": schema,
            "teacher_used": True,
        },
    }

    monkeypatch.setattr(
        mode_c_nodes,
        "structure_web_sources_for_sft",
        lambda _config, _pages: WebStructuringResult(
            schema=schema,
            raw_data=structured_raw,
            validation_report={
                "passed": True,
                "issues": [],
                "sample_accuracy_estimate": 1.0,
            },
            teacher_used=True,
        ),
    )

    out = aggregate_web_sources_node(
        {
            "config": _config(),
            "web_plan": {"planner_backend": "test"},
            "web_search_results": [{"url": "https://example.com/urgency"}],
            "web_pages": _pages(),
            "mode_c_backend": "web",
        }
    )

    assert out["validation_report"]["passed"] is True
    assert out["schema"] == schema
    assert out["raw_data"]["format_meta"]["file_type"] == "web_structured_chat_jsonl"
    assert out["raw_data"]["records"][0]["output"] == "urgent"


def test_aggregate_web_sources_auto_without_teacher_falls_back_synthetic(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-not-used")

    out = aggregate_web_sources_node(
        {
            "config": _config(),
            "web_plan": {"planner_backend": "test"},
            "web_search_results": [{"url": "https://example.com/urgency"}],
            "web_pages": _pages(),
            "mode_c_backend": "web",
        }
    )

    assert out["validation_report"]["passed"] is True
    assert out["mode_c_fallback"] == "synthetic"
    assert out["raw_data"]["format_meta"]["file_type"] == "synthetic_chat_jsonl"
    assert out["raw_data"]["format_meta"]["mode_c_fallback"] == "synthetic"
    assert out["raw_data"]["format_meta"]["num_pages_crawled"] == 1
    assert out["raw_data"]["records"][0]["messages"][0]["role"] == "user"
    assert out["raw_data"]["records"][0]["messages"][-1]["role"] == "assistant"
    assert any(
        "web structuring requires a teacher" in issue
        for issue in out["raw_data"]["format_meta"]["web_structuring_issues"]
    )
    assert "Web acquisition report retained for provenance" in out["human_readable"]


def test_aggregate_web_sources_required_without_teacher_raises_structuring_failure(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "required")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="web structuring requires a teacher"):
        aggregate_web_sources_node(
            {
                "config": _config(),
                "web_plan": {"planner_backend": "test"},
                "web_search_results": [{"url": "https://example.com/urgency"}],
                "web_pages": _pages(),
                "mode_c_backend": "web",
            }
        )


def test_aggregate_web_sources_required_raises_invalid_teacher_output(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "required")
    from src.data_generator.mode_c import nodes as mode_c_nodes

    schema = {
        "input_format": "support ticket text",
        "output_format": "one of: urgent, normal",
        "input_description": "A support ticket.",
        "output_description": "The urgency label.",
        "example_pair": {"input": "The site is down.", "output": "urgent"},
    }
    structured_raw = {
        "records": [
            {
                "input": "The site is down.",
                "output": "urgent",
                "messages": [
                    {"role": "user", "content": "The site is down."},
                    {"role": "assistant", "content": "urgent"},
                ],
            }
        ],
        "format_meta": {
            "modality": "text",
            "file_type": "web_structured_chat_jsonl",
            "encoding": "utf-8",
            "schema": schema,
            "teacher_used": True,
        },
    }

    monkeypatch.setattr(
        mode_c_nodes,
        "structure_web_sources_for_sft",
        lambda _config, _pages: WebStructuringResult(
            schema=schema,
            raw_data=structured_raw,
            validation_report={
                "passed": False,
                "issues": ["Classification-style data should include at least two labels"],
                "sample_accuracy_estimate": 1.0,
            },
            teacher_used=True,
        ),
    )

    with pytest.raises(RuntimeError, match="at least two labels"):
        aggregate_web_sources_node(
            {
                "config": _config(),
                "web_plan": {"planner_backend": "test"},
                "web_search_results": [{"url": "https://example.com/urgency"}],
                "web_pages": _pages(),
                "mode_c_backend": "web",
            }
        )
