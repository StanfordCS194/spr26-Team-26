import json
import threading
from pathlib import Path
from typing import Any, Mapping

from src.manager.manager import orchestrate_node
from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL


def test_manager_to_real_autoresearch_graph_with_fake_tinker_offline(
    tmp_path,
    monkeypatch,
):
    fake_run_cost = 1.20
    fake_budget = fake_run_cost * 2
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "synthetic")
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_EXAMPLES", "8")
    monkeypatch.setenv(
        "DATA_GENERATOR_ARTIFACT_DIR",
        str(tmp_path / "artifacts" / "data_generator"),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "claude")

    monkeypatch.setattr("builtins.input", _fail_live_service("user input"))
    monkeypatch.setattr("anthropic.Anthropic", _fail_live_service("Claude"))

    from src.data_generator.mode_c import nodes as mode_c_nodes

    monkeypatch.setattr(
        mode_c_nodes,
        "search_web_sources",
        _fail_live_service("web search"),
    )
    monkeypatch.setattr(
        mode_c_nodes,
        "crawl_and_extract_pages",
        _fail_live_service("web crawl"),
    )

    import src.autoresearch.autoresearch as ar
    import src.cost_manager.cost_manager as cost_module

    monkeypatch.setattr(cost_module, "start_cost_monitor", _fake_cost_monitor)

    runner_calls: list[dict[str, Any]] = []

    def fake_propose_hypothesis(current_config, diary, task, allowed_params=None):
        assert "learning_rate" in allowed_params
        assert current_config["learning_rate"] == 1e-4
        assert diary == []
        return {
            "description": "Increase learning_rate to 2e-4 for a bounded integration smoke.",
            "patch": json.dumps({"learning_rate": 2e-4}),
            "expected_effect": "Lower validation loss in the fake Tinker runner.",
            "search_strategy": "playbook",
        }

    def fake_tinker_runner(config, dataset, *, run_id=None, max_steps=None, **_kwargs):
        dataset_path = _dataset_path(dataset)
        rows = _load_jsonl(dataset_path)
        assert rows
        assert all(_has_chat_or_io_contract(row) for row in rows)
        assert config.model_name == DEFAULT_TINKER_MODEL
        assert max_steps == 2

        call_number = len(runner_calls) + 1
        runner_calls.append(
            {
                "call_number": call_number,
                "run_id": run_id,
                "learning_rate": config.learning_rate,
                "dataset_path": str(dataset_path),
                "row_count": len(rows),
            }
        )

        val_loss = 0.50 if call_number == 1 else 0.25
        metrics = {
            "train_loss": val_loss + 0.05,
            "val_loss": val_loss,
            "test_loss": val_loss + 0.02,
            "primary_metric": 1.0 / (1.0 + val_loss),
        }
        run_dir = Path("outputs/experiments") / str(run_id or f"fake-{call_number}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, **metrics}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(
            json.dumps({"status": "COMPLETED"}),
            encoding="utf-8",
        )
        return {
            "job_id": str(run_id or f"fake-{call_number}"),
            "status": "COMPLETED",
            "metrics": metrics,
            "model_path": str(run_dir),
            "cost_usd": fake_run_cost,
            "logs_path": str(run_dir / "metrics.jsonl"),
        }

    monkeypatch.setattr(ar, "propose_hypothesis", fake_propose_hypothesis)
    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fake_tinker_runner)

    state = {
        "prompt": "classify short support tickets by whether they require escalation",
        "budget": fake_budget,
        "data_path": None,
        "has_data": False,
        "task_reasoning": None,
        "config": _offline_mode_c_config(compute_budget=fake_budget),
        "result": None,
    }

    out = orchestrate_node(state)

    result = out["result"]
    assert result["n_iterations"] == 1
    assert result["cost"]["total_usd"] == fake_budget
    assert result["cost"]["termination_reason"] == "budget_limit"
    assert result["metrics"]["scalar"] == 0.8
    assert len(runner_calls) == 2
    assert [call["learning_rate"] for call in runner_calls] == [1e-4, 2e-4]
    assert runner_calls[0]["row_count"] >= 8
    assert runner_calls[0]["dataset_path"] == runner_calls[1]["dataset_path"]

    diary_path = Path(result["research_diary_path"])
    assert diary_path.is_file()
    diary_records = [json.loads(line) for line in diary_path.read_text().splitlines()]
    assert diary_records[0]["decision"] == "KEPT"
    assert "learning_rate" in diary_records[0]["patch"]
    assert Path("configs/current.json").is_file()
    assert json.loads(Path("configs/current.json").read_text())["learning_rate"] == 2e-4


def _offline_mode_c_config(compute_budget: float = 1.5):
    return {
        "data": False,
        "prompt": "classify short support tickets by whether they require escalation",
        "compute_budget": compute_budget,
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
                "synthetic_examples": 8,
            },
            "notes": "Offline synthetic Mode C full-stack contract test config.",
        },
    }


def _fake_cost_monitor(*_args, **_kwargs):
    thread = threading.Thread(target=lambda: None)
    thread.stop_event = threading.Event()  # type: ignore[attr-defined]
    thread.start()
    return thread


def _fail_live_service(name: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be used in this test")

    return _fail


def _dataset_path(dataset: Mapping[str, Any]) -> Path:
    path = Path(str(dataset["dataset"]["path"]))
    assert path.is_file()
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _has_chat_or_io_contract(row: Mapping[str, Any]) -> bool:
    messages = row.get("messages")
    if isinstance(messages, list):
        roles = {
            message.get("role")
            for message in messages
            if isinstance(message, dict)
        }
        return {"user", "assistant"}.issubset(roles)
    return bool(row.get("input")) and bool(row.get("output"))
