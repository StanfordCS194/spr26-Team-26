import json
from pathlib import Path

import pytest

from src.data_generator.mode_c.structuring import WebStructuringResult
from src.manager.manager import orchestrate_node
from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL


def test_manager_accepts_structured_mode_c_web_handoff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRICT", "1")
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "required")
    monkeypatch.setenv(
        "DATA_GENERATOR_ARTIFACT_DIR",
        str(tmp_path / "artifacts" / "data_generator"),
    )
    _disable_live_services(monkeypatch)

    from src.data_generator.mode_c import nodes as mode_c_nodes

    web_calls = {"search": 0, "crawl": 0, "structure": 0}

    def fake_search(web_plan):
        web_calls["search"] += 1
        assert web_plan["planner_backend"] == "mock_llm:base"
        return [
            {
                "url": "https://example.com/support-priority-guide",
                "title": "Support priority guide",
                "score": 0.95,
            }
        ]

    def fake_crawl(search_results, web_plan):
        web_calls["crawl"] += 1
        assert search_results[0]["url"] == "https://example.com/support-priority-guide"
        assert web_plan["task_type"] == "text-classification"
        return [
            {
                "source": "web_page",
                "url": "https://example.com/support-priority-guide",
                "title": "Support priority guide",
                "content": (
                    "Critical tickets include production outages, security "
                    "incidents, and confirmed data loss. Normal tickets include "
                    "how-to requests, cosmetic bugs, and account preference changes."
                ),
                "metadata": {"extraction_method": "test_fixture"},
            }
        ]

    def fake_structure(config, pages):
        web_calls["structure"] += 1
        assert config["prompt"] == "classify support tickets by urgency"
        assert pages[0]["url"] == "https://example.com/support-priority-guide"
        schema = {
            "input_format": "support ticket text",
            "output_format": "one of: urgent, normal",
            "input_description": "A short support ticket.",
            "output_description": "The escalation label.",
            "example_pair": {
                "input": "Checkout is down for every customer.",
                "output": "urgent",
            },
        }
        return WebStructuringResult(
            schema=schema,
            raw_data={
                "records": [
                    _chat_row("Checkout is down for every customer.", "urgent"),
                    _chat_row("How do I change my profile photo?", "normal"),
                    _chat_row("We see confirmed customer data loss.", "urgent"),
                ],
                "format_meta": {
                    "modality": "text",
                    "file_type": "web_structured_chat_jsonl",
                    "encoding": "utf-8",
                    "schema": schema,
                    "teacher_used": True,
                },
            },
            validation_report={
                "passed": True,
                "issues": [],
                "sample_accuracy_estimate": 0.92,
            },
            teacher_used=True,
        )

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fake_search)
    monkeypatch.setattr(mode_c_nodes, "crawl_and_extract_pages", fake_crawl)
    monkeypatch.setattr(mode_c_nodes, "structure_web_sources_for_sft", fake_structure)

    captured = {}

    def fake_decision_engine(config, dataset):
        captured["config"] = config
        captured["dataset"] = dataset
        rows = _read_jsonl(Path(dataset["dataset"]["path"]))
        assert len(rows) == 3
        assert all(_has_user_and_assistant(row) for row in rows)
        assert dataset["mode_used"] == "C"
        assert dataset["validation_report"]["passed"] is True
        assert dataset["dataset"]["train_size"] > 0
        return {
            "strategy": "fine-tune",
            "base_model": DEFAULT_TINKER_MODEL,
            "lora_config": None,
            "estimated_cost": 0.02,
            "estimated_time_min": 1,
            "training_script_path": str(tmp_path / "train.py"),
            "eval_metric": "primary_metric",
            "backend": "tinker_sft",
            "dataset_path": dataset["dataset"]["path"],
        }

    def fake_autoresearch(plan, config, cost_manager):
        captured["plan"] = plan
        captured["autoresearch_config"] = config
        captured["budget"] = cost_manager.budget
        assert plan["backend"] == "tinker_sft"
        assert Path(plan["dataset_path"]).is_file()
        return {
            "weights_path": "outputs/experiments/fake",
            "metrics": {
                "scalar": 0.71,
                "metrics": {"val_loss": 0.41},
                "critique": "structured Mode C web handoff reached training",
            },
            "cost": {
                "data_gen_usd": 0.0,
                "training_usd": 0.02,
                "llm_calls_usd": 0.0,
                "total_usd": 0.02,
                "termination_reason": "training_complete",
            },
            "n_iterations": 1,
            "research_diary_path": "outputs/logs/research_diary.jsonl",
        }

    monkeypatch.setattr(
        "src.decision_engine.decision_engine.run_decision_engine",
        fake_decision_engine,
    )
    monkeypatch.setattr(
        "src.autoresearch.autoresearch.invoke_autoresearch_graph",
        fake_autoresearch,
    )
    monkeypatch.setattr("src.observability.observability.log_event", lambda *_args, **_kwargs: None)

    out = orchestrate_node(_mode_c_state())

    assert out["result"]["metrics"]["scalar"] == 0.71
    assert captured["config"]["data"] is False
    assert captured["autoresearch_config"] == captured["config"]
    assert captured["budget"] == 0.25
    assert captured["dataset"]["quality_notes"].startswith("DataGen mode C")
    assert web_calls == {"search": 1, "crawl": 1, "structure": 1}
    assert (tmp_path / "artifacts" / "data_generator" / "raw_handoff_data.json").is_file()


def test_manager_rejects_unstructured_mode_c_web_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRICT", "1")
    monkeypatch.setenv("DATA_GENERATOR_WEB_STRUCTURING", "off")
    _disable_live_services(monkeypatch)

    from src.data_generator.mode_c import nodes as mode_c_nodes

    monkeypatch.setattr(
        mode_c_nodes,
        "search_web_sources",
        lambda _web_plan: [
            {
                "url": "https://example.com/raw-support-page",
                "title": "Raw support policy",
                "score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        mode_c_nodes,
        "crawl_and_extract_pages",
        lambda _results, _plan: [
            {
                "source": "web_page",
                "url": "https://example.com/raw-support-page",
                "title": "Raw support policy",
                "content": "Urgent tickets involve outages. This is source text only.",
                "metadata": {"extraction_method": "test_fixture"},
            }
        ],
    )

    def fail_downstream(*_args, **_kwargs):
        raise AssertionError("downstream training should not run for raw web records")

    monkeypatch.setattr(
        "src.decision_engine.decision_engine.run_decision_engine",
        fail_downstream,
    )
    monkeypatch.setattr(
        "src.autoresearch.autoresearch.invoke_autoresearch_graph",
        fail_downstream,
    )
    monkeypatch.setattr("src.observability.observability.log_event", lambda *_args, **_kwargs: None)

    with pytest.raises(
        ValueError,
        match="Mode C web source has no assistant target",
    ):
        orchestrate_node(_mode_c_state())

    dataset_path = tmp_path / "outputs" / "datasets" / "train_data.jsonl"
    assert dataset_path.is_file()
    assert dataset_path.read_text(encoding="utf-8") == ""


def _mode_c_state():
    return {
        "prompt": "classify support tickets by urgency",
        "budget": 0.25,
        "data_path": None,
        "has_data": False,
        "task_reasoning": None,
        "config": {
            "data": False,
            "prompt": "classify support tickets by urgency",
            "compute_budget": 0.25,
            "training_procedure": {
                "task_type": "text-classification",
                "data_format": "chat jsonl",
                "training_type": "SFT",
                "base_model": None,
                "hyperparameters": {
                    "learning_rate": 1e-4,
                    "batch_size": 4,
                    "epochs": 1,
                    "max_seq_len": 256,
                    "max_steps": 2,
                },
                "notes": "Manager boundary test config.",
            },
        },
        "result": None,
    }


def _disable_live_services(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    def fail_input(_prompt):
        raise AssertionError("test should not prompt for interactive input")

    monkeypatch.setattr("builtins.input", fail_input)


def _chat_row(user_text, assistant_text):
    return {
        "input": user_text,
        "output": assistant_text,
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "source_url": "https://example.com/support-priority-guide",
    }


def _read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _has_user_and_assistant(row):
    messages = row.get("messages")
    roles = {
        message.get("role")
        for message in messages or []
        if isinstance(message, dict)
    }
    return {"user", "assistant"}.issubset(roles)
