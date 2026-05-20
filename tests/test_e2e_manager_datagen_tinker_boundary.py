import json
from pathlib import Path
from typing import Any, Mapping

from src.manager.manager import orchestrate_node
from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL


def test_manager_datagen_decision_autoresearch_tinker_boundary_offline(
    tmp_path,
    monkeypatch,
):
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

    runner_calls: list[dict[str, Any]] = []

    def fake_tinker_runner(config, dataset, *, max_steps=None, **_kwargs):
        dataset_path = _dataset_path(dataset)
        rows = _load_jsonl(dataset_path)
        assert rows
        assert all(_has_chat_or_io_contract(row) for row in rows)
        assert config.model_name == DEFAULT_TINKER_MODEL
        assert max_steps is not None
        assert 1 <= int(max_steps) <= 5

        runner_calls.append(
            {
                "model_name": config.model_name,
                "dataset_path": str(dataset_path),
                "row_count": len(rows),
                "max_steps": max_steps,
            }
        )
        run_dir = Path("outputs/experiments") / f"fake-tinker-{len(runner_calls)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "train_loss": 0.4,
            "val_loss": 0.5,
            "test_loss": 0.55,
            "primary_metric": 1.0 / 1.5,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, **metrics}) + "\n",
            encoding="utf-8",
        )
        return {
            "job_id": "fake-tinker-boundary",
            "status": "COMPLETED",
            "metrics": metrics,
            "model_path": str(run_dir),
            "cost_usd": 0.02,
            "logs_path": str(run_dir / "metrics.jsonl"),
        }

    def invoke_baseline_only_autoresearch(plan, config, cost_manager):
        assert cost_manager.budget == config["compute_budget"]
        assert plan["backend"] == "tinker_sft"
        assert plan["base_model"] == DEFAULT_TINKER_MODEL
        assert Path(plan["dataset_path"]).exists()

        state = {
            "plan": plan,
            "config": config,
            "eval_suite": None,
            "current_script": plan["training_script_path"],
            "current_config": config["training_procedure"]["hyperparameters"],
            "current_patch": None,
            "last_description": None,
            "original_content": None,
            "diary": [],
            "baseline_score": None,
            "best_score": None,
            "best_script": plan["training_script_path"],
            "last_result": None,
            "last_score": None,
            "last_delta": None,
            "iteration": 0,
            "no_improve_streak": 0,
            "should_stop": False,
        }
        state.update(ar.init_node(state))
        state.update(ar.baseline_node(state))

        return {
            "weights_path": state["best_script"],
            "metrics": state["best_score"],
            "cost": {
                "data_gen_usd": 0.0,
                "training_usd": state["last_result"]["cost_usd"],
                "llm_calls_usd": 0.0,
                "total_usd": state["last_result"]["cost_usd"],
                "termination_reason": "training_complete",
            },
            "n_iterations": state["iteration"],
            "research_diary_path": str(Path("outputs/logs/research_diary.jsonl")),
        }

    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fake_tinker_runner)
    monkeypatch.setattr(
        ar,
        "invoke_autoresearch_graph",
        invoke_baseline_only_autoresearch,
    )

    state = {
        "prompt": "classify short support tickets by whether they require escalation",
        "budget": 10.0,
        "data_path": None,
        "has_data": False,
        "task_reasoning": None,
        "config": _offline_mode_c_config(),
        "result": None,
    }

    out = orchestrate_node(state)

    result = out["result"]
    assert result["weights_path"]
    assert result["metrics"]["scalar"] > 0.0
    assert isinstance(result["metrics"]["metrics"]["train_loss"], float)
    assert result["cost"]["total_usd"] == 0.02
    assert result["n_iterations"] == 0
    assert len(runner_calls) == 1
    assert runner_calls[0]["model_name"] == DEFAULT_TINKER_MODEL
    assert runner_calls[0]["row_count"] >= 8
    assert runner_calls[0]["max_steps"] == 2

    dataset_path = Path(runner_calls[0]["dataset_path"])
    assert dataset_path.is_file()
    assert tmp_path in dataset_path.parents
    assert Path("decisions.jsonl").is_file()


def _offline_mode_c_config():
    return {
        "data": False,
        "prompt": "classify short support tickets by whether they require escalation",
        "compute_budget": 10.0,
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
            "notes": "Offline synthetic Mode C boundary test config.",
        },
    }


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
