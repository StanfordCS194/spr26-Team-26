"""Tests for the browser-facing FastAPI run bridge."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.runtime_context import get_output_root
from src.server import app as server_app


def _write_tinker_artifacts(
    run_dir: Path,
    *,
    run_id: str,
    state_path: str,
    sampler_path: str | None = None,
    train_loss: float = 0.31,
    val_loss: float = 0.29,
    test_loss: float = 0.33,
    primary_metric: float = 0.775,
    sample_text: str = "sample completion",
    metric_rows: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {"state_path": state_path}
    if sampler_path:
        checkpoints["sampler_path"] = sampler_path
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETED",
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "primary_metric": primary_metric,
            }
        ),
        encoding="utf-8",
    )
    rows = metric_rows or [
        {"step": 1, "val_loss": val_loss, "primary_metric": primary_metric}
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "sample.json").write_text(
        json.dumps({"text": sample_text}),
        encoding="utf-8",
    )


def _fake_model(tmp_path, *, total_cost=1.25, with_artifacts=False):
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
    weights_path = "outputs/experiments/run/model"
    if with_artifacts:
        run_dir = tmp_path / "experiments" / "tinker-run"
        weights_path = str(run_dir)
        _write_tinker_artifacts(
            run_dir,
            run_id="tinker-run",
            state_path="tinker://state/abc",
            sampler_path="tinker://sampler/abc",
        )
    return {
        "weights_path": weights_path,
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

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        calls.append((prompt, budget, data_path, task_type_hint))
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
        ("Fine tune a small chat assistant on support tickets", 25.0, None, "fine-tuning")
    ]
    assert len(roots) == 1
    assert roots[0] == Path("outputs/api-runs") / run_id
    assert (roots[0] / "decisions.jsonl").is_file()
    assert not Path("decisions.jsonl").exists()
    assert state["costSpent"] == 1.25
    assert state["dataPath"] is None
    assert state["metrics"][0]["loss"] == 0.29
    assert state["metrics"][0]["accuracy"] == 0.775
    assert state["iterations"][0]["status"] == "KEPT"
    assert state["result"]["weights_path"] == "outputs/experiments/run/model"


def test_create_run_surfaces_artifacts_and_allowlisted_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        return _fake_model(root, total_cost=0.5, with_artifacts=True)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a small chat assistant on support tickets",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")

    artifacts = state["artifacts"]
    assert artifacts["modelPath"].endswith(f"outputs/api-runs/{run_id}/experiments/tinker-run")
    assert artifacts["checkpoints"]["state_path"] == "tinker://state/abc"
    assert artifacts["metrics"]["val_loss"] == 0.29
    assert artifacts["sample"]["text"] == "sample completion"

    files = {item["name"]: item for item in artifacts["files"]}
    assert files["manifest"]["exists"] is True
    assert files["manifest"]["downloadPath"] == f"/api/runs/{run_id}/artifacts/manifest"
    assert files["metrics_log"]["contentType"] == "application/x-ndjson"

    manifest_response = client.get(files["manifest"]["downloadPath"])
    assert manifest_response.status_code == 200
    assert manifest_response.json()["checkpoints"]["sampler_path"] == "tinker://sampler/abc"

    metrics_log_response = client.get(f"/api/runs/{run_id}/artifacts/metrics_log")
    assert metrics_log_response.status_code == 200
    assert '"step": 1' in metrics_log_response.text

    missing_response = client.get(f"/api/runs/{run_id}/artifacts/not-real")
    assert missing_response.status_code == 404


def test_running_run_refreshes_logs_iterations_metrics_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ready = threading.Event()
    release = threading.Event()

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None

        from src.observability.observability import log_event
        from src.types import AgentName, LogLevel

        log_event(AgentName.DATA_GEN, LogLevel.INFO, "dataset found", {})
        diary_path = root / "logs" / "research_diary.jsonl"
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        diary_path.write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "hypothesis": "Increase learning rate",
                    "patch": "- learning_rate: 0.0001\n+ learning_rate: 0.0005",
                    "metrics": {"val_loss": 0.4, "primary_metric": 0.714},
                    "decision": "PENDING",
                    "cost_usd": 0.42,
                }
            )
            + "\nnot-json-yet",
            encoding="utf-8",
        )
        _write_tinker_artifacts(
            root / "experiments" / "live-progress",
            run_id="live-progress",
            state_path="tinker://state/progress",
            val_loss=0.4,
            primary_metric=0.714,
            sample_text="progress sample",
            metric_rows=[
                {"step": 1, "val_loss": 0.52, "primary_metric": 0.61},
                {"step": 2, "val_loss": 0.4, "primary_metric": 0.714},
            ],
        )
        ready.set()
        assert release.wait(timeout=5)
        return _fake_model(root, total_cost=0.5)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune while surfacing live progress",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert ready.wait(timeout=5)

    state = client.get(f"/api/runs/{run_id}").json()
    assert state["status"] == "running"
    assert state["stage"] >= 4
    assert state["costSpent"] == 0.42
    assert state["logs"][0]["component"] == "DataGen"
    assert state["iterations"][0]["status"] == "PENDING"
    assert [metric["loss"] for metric in state["metrics"]] == [0.52, 0.4]
    assert [metric["accuracy"] for metric in state["metrics"]] == [0.61, 0.714]
    assert state["artifacts"]["checkpoints"]["state_path"] == "tinker://state/progress"

    metrics_log = client.get(f"/api/runs/{run_id}/artifacts/metrics_log")
    assert metrics_log.status_code == 200
    assert '"step": 1' in metrics_log.text
    assert '"step": 2' in metrics_log.text

    release.set()
    state = _wait_for_status(client, run_id, "complete")
    assert state["status"] == "complete"
    assert [metric["loss"] for metric in state["metrics"]] == [0.52, 0.4]
    assert [metric["accuracy"] for metric in state["metrics"]] == [0.61, 0.714]


def test_running_run_keeps_previous_artifacts_when_newer_experiment_is_incomplete(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    first_ready = threading.Event()
    create_incomplete = threading.Event()
    incomplete_ready = threading.Event()
    release = threading.Event()

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        _write_tinker_artifacts(
            root / "experiments" / "complete-a",
            run_id="complete-a",
            state_path="tinker://state/a",
            val_loss=0.4,
            primary_metric=0.714,
        )
        first_ready.set()
        assert create_incomplete.wait(timeout=5)
        time.sleep(0.02)
        incomplete_dir = root / "experiments" / "in-progress-b"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        (incomplete_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, "val_loss": 0.8, "primary_metric": 0.556}) + "\n",
            encoding="utf-8",
        )
        incomplete_ready.set()
        assert release.wait(timeout=5)
        return _fake_model(root, total_cost=0.5, with_artifacts=True)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune while keeping stable artifact links",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert first_ready.wait(timeout=5)

    state = client.get(f"/api/runs/{run_id}").json()
    assert state["status"] == "running"
    assert state["artifacts"]["modelPath"].endswith("experiments/complete-a")
    assert state["artifacts"]["checkpoints"]["state_path"] == "tinker://state/a"
    assert client.get(f"/api/runs/{run_id}/artifacts/manifest").status_code == 200

    create_incomplete.set()
    assert incomplete_ready.wait(timeout=5)
    state = client.get(f"/api/runs/{run_id}").json()
    assert state["status"] == "running"
    assert state["metrics"][-1]["loss"] == 0.8
    assert state["artifacts"]["modelPath"].endswith("experiments/complete-a")
    assert state["artifacts"]["checkpoints"]["state_path"] == "tinker://state/a"
    manifest_response = client.get(f"/api/runs/{run_id}/artifacts/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.json()["checkpoints"]["state_path"] == "tinker://state/a"

    release.set()
    assert _wait_for_status(client, run_id, "complete")["status"] == "complete"


def test_running_run_switches_artifacts_after_newer_experiment_completes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    first_ready = threading.Event()
    create_second = threading.Event()
    second_ready = threading.Event()
    release = threading.Event()

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        _write_tinker_artifacts(
            root / "experiments" / "complete-a",
            run_id="complete-a",
            state_path="tinker://state/a",
        )
        first_ready.set()
        assert create_second.wait(timeout=5)
        time.sleep(0.02)
        _write_tinker_artifacts(
            root / "experiments" / "complete-b",
            run_id="complete-b",
            state_path="tinker://state/b",
            val_loss=0.2,
            primary_metric=0.833,
            sample_text="newer sample",
        )
        second_ready.set()
        assert release.wait(timeout=5)
        return _fake_model(root, total_cost=0.5, with_artifacts=True)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune while switching completed artifact links",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert first_ready.wait(timeout=5)

    state = client.get(f"/api/runs/{run_id}").json()
    assert state["artifacts"]["modelPath"].endswith("experiments/complete-a")
    assert state["artifacts"]["checkpoints"]["state_path"] == "tinker://state/a"

    create_second.set()
    assert second_ready.wait(timeout=5)
    state = client.get(f"/api/runs/{run_id}").json()
    assert state["status"] == "running"
    assert state["artifacts"]["modelPath"].endswith("experiments/complete-b")
    assert state["artifacts"]["checkpoints"]["state_path"] == "tinker://state/b"
    manifest_response = client.get(f"/api/runs/{run_id}/artifacts/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.json()["checkpoints"]["state_path"] == "tinker://state/b"

    release.set()
    assert _wait_for_status(client, run_id, "complete")["status"] == "complete"


def test_cancel_running_run_stops_at_safe_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entered = threading.Event()

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None

        from src.runtime_context import active_tinker_job, cancellation_requested

        with active_tinker_job("server-active-job"):
            entered.set()
            while not cancellation_requested():
                time.sleep(0.01)
        return _fake_model(root, total_cost=99.0, with_artifacts=True)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a cancellable support assistant",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert entered.wait(timeout=5)

    cancel_response = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelling"

    from src.tinker_api.tinker_api import is_cancelled

    assert is_cancelled("server-active-job")
    state = _wait_for_status(client, run_id, "cancelled")
    assert state["status"] == "cancelled"
    assert state["result"] is None
    assert any("cancel" in item["message"].lower() for item in state["logs"])


def test_cancel_missing_run_returns_404():
    client = TestClient(server_app.app)
    response = client.post("/api/runs/not-a-run/cancel")
    assert response.status_code == 404


def test_cancel_completed_run_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        return _fake_model(root, total_cost=0.5)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune and finish before cancel",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    run_id = response.json()["run_id"]
    _wait_for_status(client, run_id, "complete")
    cancel_response = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "complete"


def test_create_run_passes_existing_local_data_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"input":"great","output":"positive"}\n', encoding="utf-8")
    calls = []

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        calls.append((prompt, budget, data_path, task_type_hint))
        root = get_output_root()
        assert root is not None
        return _fake_model(root, total_cost=0.25)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune on this labeled local JSONL data",
            "budget": 20,
            "task_type": "fine-tuning",
            "data_path": str(data_path),
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "complete")
    assert calls[0][2] == str(data_path.resolve())
    assert calls[0][3] == "fine-tuning"
    assert state["dataPath"] == str(data_path.resolve())
    assert any("Dataset source:" in item["message"] for item in state["logs"])


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        ("hf://SetFit/sst2", "hf://SetFit/sst2"),
        ("https://huggingface.co/datasets/SetFit/sst2", "hf://SetFit/sst2"),
        ("SetFit/sst2", "hf://SetFit/sst2"),
    ],
)
def test_create_run_accepts_hugging_face_data_sources(tmp_path, monkeypatch, submitted, expected):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        calls.append((prompt, budget, data_path, task_type_hint))
        root = get_output_root()
        assert root is not None
        return _fake_model(root, total_cost=0.2)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune on a public Hugging Face sentiment dataset",
            "budget": 20,
            "task_type": "classification",
            "data_path": submitted,
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "complete")
    assert calls[0][2] == expected
    assert calls[0][3] == "classification"
    assert state["dataPath"] == expected


def test_create_run_surfaces_manager_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail_invoke(prompt, budget, data_path=None, task_type_hint=None):
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

    def fail_invoke(prompt, budget, data_path=None, task_type_hint=None):
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
    assert "existing local path or Hugging Face source" in response.text


def test_missing_relative_data_path_is_not_treated_as_hf_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail_invoke(prompt, budget, data_path=None, task_type_hint=None):
        raise AssertionError("manager should not be called for missing data_path")

    monkeypatch.setattr(server_app, "invoke_manager_graph", fail_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune on my relative local data path",
            "budget": 20,
            "task_type": "fine-tuning",
            "data_path": "data/train.jsonl",
        },
    )

    assert response.status_code == 400
    assert "existing local path or Hugging Face source" in response.text


def test_unsupported_remote_data_path_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail_invoke(prompt, budget, data_path=None, task_type_hint=None):
        raise AssertionError("manager should not be called for unsupported data_path")

    monkeypatch.setattr(server_app, "invoke_manager_graph", fail_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune on a remote source",
            "budget": 20,
            "task_type": "fine-tuning",
            "data_path": "https://example.com/train.jsonl",
        },
    )

    assert response.status_code == 400
    assert "existing local path or Hugging Face source" in response.text
