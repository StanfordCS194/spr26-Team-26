"""Tests for the browser-facing FastAPI run bridge."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.runtime_context import get_output_root
from src.server import app as server_app


def _fake_model(tmp_path, *, total_cost=1.25):
    diary_path = tmp_path / "diary.jsonl"
    diary_path.write_text(
        json.dumps(
            {
                "iteration": 1,
                "hypothesis": "Decrease learning rate",
                "patch": "- learning_rate: 0.0002\n+ learning_rate: 0.0001",
                "metrics": {
                    "train_loss": 0.31,
                    "val_loss": 0.29,
                    "primary_metric": 0.775,
                },
                "decision": "KEPT",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "weights_path": "outputs/experiments/run/model",
        "metrics": {
            "scalar": 0.775,
            "metrics": {
                "train_loss": 0.31,
                "val_loss": 0.29,
                "primary_metric": 0.775,
            },
            "critique": "ok",
        },
        "cost": {
            "data_gen_usd": 0.0,
            "training_usd": total_cost,
            "llm_calls_usd": 0.0,
            "total_usd": total_cost,
            "termination_reason": "training_complete",
        },
        "n_iterations": 1,
        "research_diary_path": str(diary_path),
    }


def _wait_for_status(client: TestClient, run_id: str, status: str):
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] == status:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {status}; last state={last}")


def setup_function():
    server_app._reset_runs_for_tests()


def test_health_check():
    client = TestClient(server_app.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_run_completes_with_manager_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    roots = []

    def fake_invoke(prompt, budget, data_path=None):
        calls.append((prompt, budget, data_path))
        root = get_output_root()
        assert root is not None
        roots.append(root)

        from src.manager.manager import build_orchestration_config, log_decision

        log_decision(
            "server_run",
            "run-scoped output test",
            build_orchestration_config(
                {
                    "task_type": "text-classification",
                    "data_format": "jsonl",
                    "training_type": "SFT",
                    "suggested_base_model": None,
                    "hyperparameters": {},
                    "notes": "test",
                },
                prompt,
                budget,
                False,
            ),
        )
        return _fake_model(root)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a small chat assistant on support tickets",
            "budget": 25,
            "task_type": "fine-tuning",
            "data_path": None,
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")

    assert calls == [
        ("Fine tune a small chat assistant on support tickets", 25.0, None)
    ]
    assert len(roots) == 1
    assert roots[0] == Path("outputs/api-runs") / run_id
    assert (roots[0] / "decisions.jsonl").is_file()
    assert not Path("decisions.jsonl").exists()
    assert state["costSpent"] == 1.25
    assert state["metrics"][0]["loss"] == 0.29
    assert state["metrics"][0]["accuracy"] == 0.775
    assert state["iterations"][0]["status"] == "KEPT"
    assert state["result"]["weights_path"] == "outputs/experiments/run/model"


def test_create_run_surfaces_manager_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail_invoke(prompt, budget, data_path=None):
        assert get_output_root() is not None
        raise RuntimeError("DataGen did not produce a trainable dataset")

    monkeypatch.setattr(server_app, "invoke_manager_graph", fail_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Train a classifier from public web data",
            "budget": 20,
            "task_type": "classification",
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "failed")
    assert "DataGen did not produce a trainable dataset" in state["error"]
    assert state["logs"][0]["type"] == "error"


def test_missing_data_path_is_rejected_before_background_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail_invoke(prompt, budget, data_path=None):
        raise AssertionError("manager should not be called for missing data_path")

    monkeypatch.setattr(server_app, "invoke_manager_graph", fail_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune on my local data",
            "budget": 20,
            "task_type": "fine-tuning",
            "data_path": "/path/that/does/not/exist.jsonl",
        },
    )

    assert response.status_code == 400
    assert "data_path does not exist" in response.text
