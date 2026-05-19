import json
import re
import sys
import types
from types import SimpleNamespace

import pytest

from src.data_generator.mode_c import (
    acquire_synthetic_dataset,
    acquire_web_data,
    determine_data_schema,
    generate_synthetic_data,
    infer_schema_without_teacher,
    plan_synthetic_generation,
    scrape_web,
    validate_synthetic_records,
)
from src.data_generator.graph import invoke_data_generator_graph
from src.data_generator.nodes import acquire_web_data_node


def _config(**hyperparams):
    return {
        "data": False,
        "prompt": "classify support tickets by urgency",
        "compute_budget": 25.0,
        "training_procedure": {
            "task_type": "text-classification",
            "data_format": "chat jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": hyperparams,
            "notes": "",
        },
    }


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
                "input_format": "support ticket chat text",
                "output_format": "one of: urgent, normal",
                "input_description": "A support ticket that may require urgent handling.",
                "output_description": "The urgency label for the ticket.",
                "example_pair": {
                    "input": "The production checkout flow is down.",
                    "output": "urgent",
                },
            }
        else:
            count_match = re.search(r"Generate (\d+) supervised", prompt)
            count = int(count_match.group(1)) if count_match else 1
            bucket_match = re.search(r'"name": "([^"]+)"', prompt)
            bucket = bucket_match.group(1) if bucket_match else "teacher_bucket"
            payload = [
                {
                    "input": f"{bucket} support ticket {index}",
                    "output": "urgent" if index % 2 == 0 else "normal",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{bucket} support ticket {index}",
                        },
                        {
                            "role": "assistant",
                            "content": "urgent" if index % 2 == 0 else "normal",
                        },
                    ],
                    "difficulty": "medium",
                    "tags": [bucket],
                }
                for index in range(count)
            ]
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(payload))]
        )


def test_infer_schema_without_teacher_builds_classification_contract(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")

    schema = determine_data_schema(_config())

    assert "support tickets" in schema["input_description"]
    assert schema["output_format"] == "one of: relevant, not_relevant"
    assert schema["example_pair"]["input"]
    assert schema["example_pair"]["output"] == "relevant"


def test_generation_plan_covers_requested_examples(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    schema = infer_schema_without_teacher(_config())

    plan = plan_synthetic_generation(_config(), schema, 23)

    assert sum(bucket["count"] for bucket in plan) == 23
    assert {bucket["name"] for bucket in plan} == {
        "core_positive",
        "core_negative",
        "boundary_cases",
        "format_variants",
        "hard_edge_cases",
    }
    assert {bucket["difficulty"] for bucket in plan} == {"easy", "medium", "hard"}


def test_generate_synthetic_data_offline_returns_valid_diverse_chat_records(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    config = _config(synthetic_examples=25)
    schema = infer_schema_without_teacher(config)

    raw = generate_synthetic_data(schema, 25, config=config)

    assert raw["format_meta"]["file_type"] == "synthetic_chat_jsonl"
    assert raw["format_meta"]["teacher_used"] is False
    assert len(raw["records"]) == 25
    assert raw["format_meta"]["quality_report"]["passed"] is True
    assert raw["format_meta"]["quality_report"]["label_counts"]["relevant"] > 0
    assert raw["format_meta"]["quality_report"]["label_counts"]["not_relevant"] > 0
    assert not any("TODO" in str(record) for record in raw["records"])
    for record in raw["records"]:
        messages = record["messages"]
        assert any(message["role"] == "user" for message in messages)
        assert any(message["role"] == "assistant" for message in messages)
        assert record["input"]
        assert record["output"]


def test_teacher_path_infers_schema_and_generates_batches(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    teacher = _FakeTeacher()

    result = acquire_synthetic_dataset(
        _config(synthetic_examples=12),
        teacher_client=teacher,
        n_examples=12,
    )

    assert result.schema["output_format"] == "one of: urgent, normal"
    assert result.validation_report["passed"] is True
    assert result.raw_data["format_meta"]["teacher_available"] is True
    assert result.raw_data["format_meta"]["teacher_used"] is True
    assert len(result.raw_data["records"]) == 12
    assert len(teacher.messages.calls) == 6
    assert teacher.messages.calls[0]["model"]
    assert all(call["model"] for call in teacher.messages.calls)


def test_no_spend_disables_synthetic_teacher_even_when_client_supplied(monkeypatch):
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-should-not-be-used")
    teacher = _FakeTeacher()

    result = acquire_synthetic_dataset(
        _config(synthetic_examples=12),
        teacher_client=teacher,
        n_examples=12,
    )

    assert teacher.messages.calls == []
    assert result.raw_data["format_meta"]["teacher_available"] is False
    assert result.raw_data["format_meta"]["teacher_used"] is False
    assert result.validation_report["passed"] is True


def test_validate_synthetic_records_reports_malformed_and_duplicate_rows(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    schema = infer_schema_without_teacher(_config())

    quality = validate_synthetic_records(
        [
            {"messages": [{"role": "user", "content": "missing target"}]},
            {"input": "same", "output": "relevant"},
            {"input": "same", "output": "relevant"},
            {"input": "other", "output": "not_relevant"},
        ],
        schema,
        min_examples=3,
    )

    assert quality["passed"] is False
    assert quality["valid_records"] == 2
    assert quality["duplicate_records"] == 1
    assert any("assistant" in issue for issue in quality["issues"])
    assert any("duplicate" in issue for issue in quality["issues"])


def test_validate_synthetic_records_enforces_schema_labels(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    schema = {
        **infer_schema_without_teacher(_config()),
        "output_format": "one of: urgent, normal",
    }

    quality = validate_synthetic_records(
        [{"input": "server is down", "output": "not_relevant"}],
        schema,
        min_examples=1,
    )

    assert quality["passed"] is False
    assert any("schema labels" in issue for issue in quality["issues"])


def test_acquire_web_data_compatibility_wrapper_returns_raw_data(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")

    raw = acquire_web_data(
        "classify support tickets by urgency",
        _config(synthetic_examples=11),
    )

    assert len(raw["records"]) == 11
    assert raw["format_meta"]["quality_report"]["passed"] is True
    assert raw["format_meta"]["schema"]["output_format"] == "one of: relevant, not_relevant"


def test_acquire_web_data_node_returns_schema_raw_data_and_validation(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")

    out = acquire_web_data_node(
        {
            "config": _config(synthetic_examples=14),
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

    assert out["schema"]["input_format"] == "chat jsonl"
    assert out["validation_report"]["passed"] is True
    assert len(out["raw_data"]["records"]) == 14
    assert {record["task_type"] for record in out["raw_data"]["records"]} == {
        "text-classification"
    }


def test_mode_c_graph_handoff_contains_schema_and_validation(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")

    handoff = invoke_data_generator_graph(_config(synthetic_examples=10), data_path=None)

    assert handoff["target_subagent"] == "data_curation"
    assert handoff["action"] == "structure_data"
    assert handoff["verification_level"] == "strict"
    assert handoff["mode_used"] == "C"
    assert handoff["schema"]["output_format"] == "one of: relevant, not_relevant"
    assert handoff["validation_report"]["passed"] is True
    assert len(handoff["raw_data"]["records"]) == 10


def test_mode_c_synthetic_backend_never_calls_web(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "synthetic")
    monkeypatch.setenv("TAVILY_API_KEY", "present-but-should-not-be-used")

    from src.data_generator.mode_c import nodes as mode_c_nodes

    def fail_search(_web_plan):
        raise AssertionError("synthetic backend should not call web search")

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fail_search)

    handoff = invoke_data_generator_graph(_config(synthetic_examples=8), data_path=None)

    assert handoff["mode_c_fallback"] == "synthetic"
    assert handoff["raw_data"]["format_meta"]["mode_c_backend"] == "synthetic"
    assert handoff["validation_report"]["passed"] is True


def test_mode_c_offline_env_overrides_web_backend(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")

    from src.data_generator.mode_c import nodes as mode_c_nodes

    def fail_search(_web_plan):
        raise AssertionError("offline synthetic mode should not call web search")

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fail_search)

    handoff = invoke_data_generator_graph(_config(synthetic_examples=8), data_path=None)

    assert handoff["mode_c_fallback"] == "synthetic"
    assert handoff["raw_data"]["format_meta"]["mode_c_backend"] == "synthetic"


@pytest.mark.parametrize("flag_name", ["DATA_GENERATOR_OFFLINE", "NO_SPEND"])
def test_mode_c_global_offline_flags_override_web_backend(monkeypatch, flag_name):
    monkeypatch.setenv(flag_name, "1")
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")
    monkeypatch.setenv("TAVILY_API_KEY", "present-but-should-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-should-not-be-used")

    from src.data_generator.mode_c import nodes as mode_c_nodes

    def fail_search(_web_plan):
        raise AssertionError(f"{flag_name}=1 should not call web search")

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fail_search)

    handoff = invoke_data_generator_graph(_config(synthetic_examples=8), data_path=None)

    assert handoff["mode_c_fallback"] == "synthetic"
    assert handoff["raw_data"]["format_meta"]["mode_c_backend"] == "synthetic"
    assert handoff["validation_report"]["passed"] is True


def test_mode_c_web_backend_fails_loudly_on_search_error(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")

    from src.data_generator.mode_c import nodes as mode_c_nodes

    def fail_search(_web_plan):
        raise RuntimeError("web unavailable")

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fail_search)

    with pytest.raises(RuntimeError, match="web unavailable"):
        invoke_data_generator_graph(_config(synthetic_examples=8), data_path=None)


def test_scrape_web_fallback_uses_same_standard_contract(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    schema = infer_schema_without_teacher(_config())

    raw = scrape_web("support-ticket urgency examples", schema, max_examples=9)

    assert len(raw["records"]) == 9
    assert raw["format_meta"]["quality_report"]["passed"] is True
    assert raw["format_meta"]["generation_plan"]


def test_scrape_web_global_offline_skips_mode_c_web_pipeline(monkeypatch):
    monkeypatch.setenv("DATA_GENERATOR_OFFLINE", "1")
    schema = infer_schema_without_teacher(_config())

    mock_llm = types.ModuleType("src.data_generator.mode_c.mock_llm")
    search = types.ModuleType("src.data_generator.mode_c.search")
    crawler = types.ModuleType("src.data_generator.mode_c.crawler")

    mock_llm.mock_plan_web_acquisition = lambda _config: pytest.fail(
        "offline scrape_web should not plan web acquisition"
    )
    search.search_web_sources = lambda _plan: pytest.fail(
        "offline scrape_web should not search"
    )
    crawler.crawl_and_extract_pages = lambda _results, _plan: pytest.fail(
        "offline scrape_web should not crawl"
    )
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.mock_llm", mock_llm)
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.search", search)
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.crawler", crawler)

    raw = scrape_web("support-ticket urgency examples", schema, max_examples=9)

    assert len(raw["records"]) == 9
    assert raw["format_meta"]["web_acquisition_used"] is False
    assert raw["format_meta"]["web_acquisition_fallback"] == "deterministic_synthetic"


@pytest.mark.parametrize("flag_name", ["DATA_GENERATOR_OFFLINE", "NO_SPEND"])
def test_direct_mode_c_web_helpers_honor_offline_flags(monkeypatch, flag_name):
    from src.data_generator.mode_c import crawler, search

    monkeypatch.setenv(flag_name, "1")
    monkeypatch.setenv("TAVILY_API_KEY", "present-but-not-used")

    def fail_get(*_args, **_kwargs):
        raise AssertionError("offline Mode C helpers should not call requests.get")

    monkeypatch.setattr(crawler.requests, "get", fail_get)

    result = {
        "url": "https://example.com/page",
        "domain": "example.com",
        "title": "Example",
        "query": "support ticket urgency",
        "snippet": "",
    }

    assert search.search_web_sources({"search_queries": ["support ticket urgency"]}) == []
    assert crawler.crawl_and_extract_pages([result], {"max_pages": 1}) == []

    fetched = crawler.fetch_and_extract_one(result)
    assert fetched["content"] == ""
    assert flag_name in fetched["error"]
    assert fetched["metadata"]["extraction_method"] == "offline_guard"


def test_scrape_web_uses_mode_c_web_pipeline_when_available(monkeypatch):
    monkeypatch.delenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", raising=False)
    schema = infer_schema_without_teacher(_config())

    mock_llm = types.ModuleType("src.data_generator.mode_c.mock_llm")
    search = types.ModuleType("src.data_generator.mode_c.search")
    crawler = types.ModuleType("src.data_generator.mode_c.crawler")

    mock_llm.mock_plan_web_acquisition = lambda _config: {
        "search_queries": ["support ticket urgency"],
        "max_pages": 2,
    }
    search.search_web_sources = lambda _plan: [
        {
            "url": "https://example.com/support",
            "title": "Support urgency guide",
            "snippet": "urgent outage escalation",
        }
    ]
    crawler.crawl_and_extract_pages = lambda _results, _plan: [
        {
            "url": "https://example.com/support",
            "title": "Support urgency guide",
            "content": "Urgent tickets include outages, data loss, and security incidents.",
            "metadata": {"extraction_method": "test"},
        }
    ]
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.mock_llm", mock_llm)
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.search", search)
    monkeypatch.setitem(sys.modules, "src.data_generator.mode_c.crawler", crawler)

    raw = scrape_web("classify support tickets by urgency", schema, max_examples=2)

    assert raw["format_meta"]["web_acquisition_used"] is True
    assert raw["format_meta"]["file_type"] == "web_acquired_chat_jsonl"
    assert len(raw["records"]) == 1
    assert raw["records"][0]["source"] == "mode_c_web_acquisition"
    assert "Urgent tickets" in raw["records"][0]["messages"][0]["content"]
