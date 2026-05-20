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
    backend: str | None = None,
    budget_preflight_skipped: bool = False,
    termination_reason: str | None = None,
    status: str = "COMPLETED",
    error: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {"state_path": state_path}
    if sampler_path:
        checkpoints["sampler_path"] = sampler_path
    manifest = {
        "run_id": run_id,
        "status": status,
        "checkpoints": checkpoints,
    }
    if backend:
        manifest["backend"] = backend
    if budget_preflight_skipped:
        manifest["budget_preflight_skipped"] = True
    if termination_reason:
        manifest["cost"] = {"termination_reason": termination_reason}
    if error:
        manifest["error"] = error
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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


def _write_budget_skip_artifacts(run_dir: Path, *, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sentinel_metrics = {
        "train_loss": 1_000_000_000.0,
        "val_loss": 1_000_000_000.0,
        "test_loss": 1_000_000_000.0,
        "primary_metric": 0.0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "CANCELLED",
                "budget_preflight_skipped": True,
                "budget_skip_reason": "estimated run cost exceeds remaining budget",
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(sentinel_metrics), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                **sentinel_metrics,
                "budget_preflight_skipped": True,
                "reason": "estimated run cost exceeds remaining budget",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "sample.json").write_text(json.dumps({"text": ""}), encoding="utf-8")


def _write_data_generator_debug_artifacts(
    root: Path,
    *,
    mode_used: str = "C",
    mode_c_fallback: str | None = "synthetic",
) -> None:
    artifact_dir = root / "data_generator" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    debug_path = artifact_dir / "debug_context.json"
    manifest_path = artifact_dir / "artifact_manifest.json"
    handoff_path = artifact_dir / "raw_handoff_data.json"
    report_path = artifact_dir / "human_readable.md"
    source_report_path = artifact_dir / "source_human_readable.md"
    handoff_path.write_text(
        json.dumps(
            {
                "target_subagent": "data_curation",
                "action": "curate_dataset",
                "mode_used": mode_used,
                "curation_payload": {"record_count": 1},
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text("# Curation report\n\n1 trainable row.\n", encoding="utf-8")
    source_report_path.write_text("# Source report\n\nSynthetic fallback.\n", encoding="utf-8")
    debug_path.write_text(
        json.dumps(
            {
                "mode_used": mode_used,
                "mode_c_fallback": mode_c_fallback,
                "source_metadata": {
                    "mode": mode_used,
                    "mode_c_fallback": mode_c_fallback,
                },
                "raw_data": {
                    "format_meta": {
                        "mode_c_backend": "synthetic",
                        "mode_c_fallback": mode_c_fallback,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "mode_used": mode_used,
                "artifact_dir": str(artifact_dir),
                "files": {
                    "handoff_payload": str(handoff_path),
                    "curation_human_readable": str(report_path),
                    "debug_context": str(debug_path),
                    "source_human_readable": str(source_report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "data_generator" / "latest_run.json").write_text(
        json.dumps({"mode_used": mode_used, "manifest_path": str(manifest_path)}),
        encoding="utf-8",
    )


def _fake_budget_skipped_model(tmp_path):
    run_dir = tmp_path / "experiments" / "budget-skipped-run"
    _write_budget_skip_artifacts(run_dir, run_id="budget-skipped-run")
    diary_path = tmp_path / "logs" / "research_diary.jsonl"
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    diary_path.write_text("", encoding="utf-8")
    return {
        "weights_path": str(run_dir),
        "metrics": {
            "scalar": 0.0,
            "metrics": {
                "train_loss": 1_000_000_000.0,
                "val_loss": 1_000_000_000.0,
                "primary_metric": 0.0,
            },
            "critique": "budget preflight skipped the Tinker launch",
        },
        "cost": {
            "data_gen_usd": 0.0,
            "training_usd": 0.0,
            "llm_calls_usd": 0.0,
            "total_usd": 0.0,
            "termination_reason": "budget_limit",
        },
        "n_iterations": 0,
        "research_diary_path": str(diary_path),
    }


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
            backend="tinker_sft",
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


def _fail_live_service(name: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be used in this test")

    return _fail


def _install_no_live_service_fences(monkeypatch):
    monkeypatch.setattr("builtins.input", _fail_live_service("user input"))
    monkeypatch.setattr("anthropic.Anthropic", _fail_live_service("Claude"))

    try:
        import requests

        monkeypatch.setattr(requests, "get", _fail_live_service("requests.get"))
    except Exception:
        pass

    from src.autoresearch import autoresearch as ar
    from src.data_generator import mode_b
    from src.data_generator.mode_c import nodes as mode_c_nodes
    from src.data_generator.mode_c import synthetic as mode_c_synthetic
    from src.tinker_api import sft_runner

    monkeypatch.setattr(ar, "_MAX_ITERATIONS", 1)
    monkeypatch.setattr(
        mode_b,
        "_fetch_with_hf_datasets",
        _fail_live_service("Hugging Face dataset fetch"),
    )
    monkeypatch.setattr(
        mode_c_nodes,
        "search_web_sources",
        _fail_live_service("Tavily web search"),
    )
    monkeypatch.setattr(
        mode_c_nodes,
        "crawl_and_extract_pages",
        _fail_live_service("web crawl"),
    )
    monkeypatch.setattr(
        mode_c_nodes,
        "structure_web_sources_for_sft",
        _fail_live_service("Mode C web teacher structuring"),
    )
    monkeypatch.setattr(
        mode_c_synthetic,
        "_anthropic_client",
        _fail_live_service("Mode C synthetic teacher"),
    )
    monkeypatch.setattr(
        sft_runner,
        "_load_tinker_deps",
        _fail_live_service("Tinker SDK"),
    )


def setup_function():
    server_app._reset_runs_for_tests()


def test_health_check():
    client = TestClient(server_app.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_alternate_local_dev_ports():
    client = TestClient(server_app.app)

    response = client.options(
        "/api/runs",
        headers={
            "Origin": "http://127.0.0.1:5177",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5177"


def test_cors_origins_can_be_extended_from_env(monkeypatch):
    monkeypatch.setenv(
        "MANAGER_API_CORS_ORIGINS",
        "https://preview.example.com, http://localhost:5173",
    )

    origins = server_app._cors_origins_from_env()

    assert origins.count("http://localhost:5173") == 1
    assert "https://preview.example.com" in origins


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
    assert state["metrics"][0]["primaryMetric"] == 0.775
    assert state["metrics"][0]["primaryMetricLabel"] == "Primary Score"
    assert state["iterations"][0]["status"] == "KEPT"
    assert state["iterations"][0]["primaryMetric"] == 0.775
    assert state["iterations"][0]["primaryMetricLabel"] == "Primary Score"
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
    assert state["provenance"]["trainingBackend"] == "tinker_sft"
    assert state["provenance"]["spendMode"] == "live"
    assert "Tinker" in state["provenance"]["liveServices"]

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


def test_run_state_surfaces_provenance_from_manifests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        run_dir = root / "experiments" / "dry-skip"
        _write_tinker_artifacts(
            run_dir,
            run_id="dry-skip",
            state_path="dry-run://state/abc",
            backend="dry_run",
            budget_preflight_skipped=True,
            termination_reason="budget_limit",
        )
        _write_data_generator_debug_artifacts(root, mode_used="C", mode_c_fallback="synthetic")
        result = _fake_model(root, total_cost=0.0)
        result["weights_path"] = str(run_dir)
        result["cost"]["termination_reason"] = "budget_limit"
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a dry-run support assistant",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "cancelled")

    provenance = state["provenance"]
    assert provenance["spendMode"] == "budget_skipped"
    assert provenance["trainingBackend"] == "dry_run"
    assert provenance["dataMode"] == "C"
    assert provenance["modeCFallback"] == "synthetic"
    assert provenance["budgetPreflightSkipped"] is True
    assert provenance["budgetSkipReason"] == "budget_limit"
    assert provenance["liveServices"] == []
    assert any(item.endswith("manifest.json") for item in provenance["evidence"])
    assert any(item.endswith("debug_context.json") for item in provenance["evidence"])

    files = {item["name"]: item for item in state["artifacts"]["files"]}
    assert files["data_manifest"]["exists"] is True
    assert files["data_handoff"]["exists"] is True
    assert files["data_curation_report"]["exists"] is True
    assert files["data_debug_context"]["downloadPath"] == (
        f"/api/runs/{response.json()['run_id']}/artifacts/data_debug_context"
    )
    debug_response = client.get(
        f"/api/runs/{response.json()['run_id']}/artifacts/data_debug_context"
    )
    assert debug_response.status_code == 200
    assert debug_response.json()["mode_used"] == "C"
    source_report_response = client.get(
        f"/api/runs/{response.json()['run_id']}/artifacts/data_source_report"
    )
    assert source_report_response.status_code == 200
    assert "Synthetic fallback" in source_report_response.text


def test_data_generator_artifact_downloads_are_allowlisted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        run_dir = root / "experiments" / "dry-run"
        _write_tinker_artifacts(
            run_dir,
            run_id="dry-run",
            state_path="dry-run://state/abc",
            backend="dry_run",
        )

        secret = tmp_path / "secret.json"
        secret.write_text('{"secret": true}\n', encoding="utf-8")
        artifact_dir = root / "data_generator" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "artifact_manifest.json"
        safe_debug_path = artifact_dir / "debug_context.json"
        safe_debug_path.write_text('{"mode_used": "C"}\n', encoding="utf-8")
        hidden_path = artifact_dir / "hidden.json"
        hidden_path.write_text('{"hidden": true}\n', encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "mode_used": "C",
                    "artifact_dir": str(artifact_dir),
                    "files": {
                        "debug_context": str(secret),
                        "handoff_payload": str(hidden_path),
                        "secret": str(hidden_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "data_generator" / "latest_run.json").write_text(
            json.dumps({"mode_used": "C", "manifest_path": str(manifest_path)}),
            encoding="utf-8",
        )

        result = _fake_model(root, total_cost=0.0)
        result["weights_path"] = str(run_dir)
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune with guarded DataGen artifact downloads",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")
    files = {item["name"]: item for item in state["artifacts"]["files"]}

    assert files["data_manifest"]["downloadPath"] == (
        f"/api/runs/{run_id}/artifacts/data_manifest"
    )
    assert files["data_debug_context"]["exists"] is False
    assert files["data_debug_context"]["downloadPath"] is None
    assert files["data_handoff"]["exists"] is False
    assert files["data_handoff"]["downloadPath"] is None
    assert "secret" not in files
    assert client.get(f"/api/runs/{run_id}/artifacts/data_manifest").status_code == 200
    assert client.get(f"/api/runs/{run_id}/artifacts/data_debug_context").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifacts/secret").status_code == 404


def test_data_generator_manifest_path_must_stay_inside_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        run_dir = root / "experiments" / "dry-run"
        _write_tinker_artifacts(
            run_dir,
            run_id="dry-run",
            state_path="dry-run://state/abc",
            backend="dry_run",
        )

        external_manifest = tmp_path / "external_manifest.json"
        external_manifest.write_text(
            json.dumps({"files": {"debug_context": str(tmp_path / "secret.json")}}),
            encoding="utf-8",
        )
        (root / "data_generator").mkdir(parents=True, exist_ok=True)
        (root / "data_generator" / "latest_run.json").write_text(
            json.dumps({"mode_used": "C", "manifest_path": str(external_manifest)}),
            encoding="utf-8",
        )

        result = _fake_model(root, total_cost=0.0)
        result["weights_path"] = str(run_dir)
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune with an escaped DataGen manifest",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")
    files = {item["name"]: item for item in state["artifacts"]["files"]}
    assert "data_manifest" not in files
    assert client.get(f"/api/runs/{run_id}/artifacts/data_manifest").status_code == 404


def test_api_run_routes_data_generator_artifacts_under_run_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    external_dir = tmp_path / "external-data-generator-artifacts"
    monkeypatch.setenv("DATA_GENERATOR_ARTIFACT_DIR", str(external_dir))

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        run_dir = root / "experiments" / "dry-run"
        _write_tinker_artifacts(
            run_dir,
            run_id="dry-run",
            state_path="dry-run://state/abc",
            backend="dry_run",
        )

        from src.data_generator.artifacts import save_subagent2_artifacts

        save_subagent2_artifacts(
            {
                "target_subagent": "data_curation",
                "action": "structure_data",
                "mode_used": "C",
                "curation_payload": {
                    "schema_version": "data_curation_input.v1",
                    "record_count": 1,
                    "records": [{"messages": [{"role": "user", "content": "Hi"}]}],
                },
                "curation_human_readable": "Sub-Agent 2 Curation Input\nRecord count: 1",
                "human_readable": "Mode C source report",
            }
        )

        result = _fake_model(root, total_cost=0.0)
        result["weights_path"] = str(run_dir)
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune while preserving DataGen evidence under run root",
            "budget": 25,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")
    files = {item["name"]: item for item in state["artifacts"]["files"]}

    assert not external_dir.exists()
    assert files["data_manifest"]["exists"] is True
    assert files["data_manifest"]["path"].endswith(
        f"outputs/api-runs/{run_id}/data_generator/artifacts/artifact_manifest.json"
    )
    assert files["data_debug_context"]["downloadPath"] == (
        f"/api/runs/{run_id}/artifacts/data_debug_context"
    )
    assert client.get(f"/api/runs/{run_id}/artifacts/data_manifest").status_code == 200
    assert client.get(f"/api/runs/{run_id}/artifacts/data_debug_context").status_code == 200


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


def test_running_run_hides_budget_skip_sentinel_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ready = threading.Event()
    release = threading.Event()

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        diary_path = root / "logs" / "research_diary.jsonl"
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        diary_path.write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "hypothesis": "Increase learning rate",
                    "patch": "- learning_rate: 0.0001\n+ learning_rate: 0.0002",
                    "metrics": {"val_loss": 0.42, "primary_metric": 0.704},
                    "decision": "REVERTED",
                    "cost_usd": 1.12,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "iteration": 2,
                    "hypothesis": "Increase batch size",
                    "patch": "- batch_size: 4\n+ batch_size: 5",
                    "metrics": {
                        "train_loss": 1_000_000_000.0,
                        "val_loss": 1_000_000_000.0,
                        "primary_metric": 0.0,
                    },
                    "decision": "REVERTED",
                    "cost_usd": 0.0,
                    "notes": "Skipped: budget preflight rejected the Tinker run",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_tinker_artifacts(
            root / "experiments" / "completed-run",
            run_id="completed-run",
            state_path="tinker://state/completed",
            val_loss=0.42,
            primary_metric=0.704,
            metric_rows=[
                {"step": 1, "val_loss": 0.6, "primary_metric": 0.625},
                {"step": 2, "val_loss": 0.42, "primary_metric": 0.704},
            ],
        )
        _write_budget_skip_artifacts(
            root / "experiments" / "budget-skipped-run",
            run_id="budget-skipped-run",
        )
        ready.set()
        assert release.wait(timeout=5)
        result = _fake_model(root, total_cost=1.12)
        result["research_diary_path"] = str(diary_path)
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune while hiding budget skip sentinels",
            "budget": 2,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert ready.wait(timeout=5)

    state = client.get(f"/api/runs/{run_id}").json()
    assert [metric["loss"] for metric in state["metrics"]] == [0.6, 0.42]
    assert [metric["accuracy"] for metric in state["metrics"]] == [0.625, 0.704]
    assert [item["experiment"] for item in state["iterations"]] == [
        "Increase learning rate"
    ]

    release.set()
    state = _wait_for_status(client, run_id, "complete")
    assert [metric["loss"] for metric in state["metrics"]] == [0.6, 0.42]
    assert [metric["accuracy"] for metric in state["metrics"]] == [0.625, 0.704]
    assert [item["experiment"] for item in state["iterations"]] == [
        "Increase learning rate"
    ]


def test_budget_preflight_skip_is_not_reported_as_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, **_kwargs):
        root = get_output_root()
        assert root is not None
        return _fake_budget_skipped_model(root)

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune with a budget too small to start Tinker",
            "budget": 1,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "cancelled")

    assert state["status"] == "cancelled"
    assert state["error"] == "estimated run cost exceeds remaining budget"
    assert state["costSpent"] == 0.0
    assert state["metrics"] == []
    assert state["iterations"] == []
    assert state["result"]["cost"]["termination_reason"] == "budget_limit"
    assert state["artifacts"]["modelPath"].endswith("experiments/budget-skipped-run")
    assert state["artifacts"]["checkpoints"] == {}
    assert any("budget guard" in item["message"].lower() for item in state["logs"])

    manifest_response = client.get(f"/api/runs/{run_id}/artifacts/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.json()["budget_preflight_skipped"] is True


@pytest.mark.parametrize(
    ("manifest_status", "api_status", "log_type"),
    [
        ("FAILED", "failed", "error"),
        ("CANCELLED", "cancelled", "warning"),
    ],
)
def test_selected_training_artifact_terminal_status_controls_api_status(
    tmp_path,
    monkeypatch,
    manifest_status,
    api_status,
    log_type,
):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(prompt, budget, data_path=None, **_kwargs):
        root = get_output_root()
        assert root is not None
        run_dir = root / "experiments" / "terminal-artifact"
        _write_tinker_artifacts(
            run_dir,
            run_id="terminal-artifact",
            state_path="tinker://state/terminal",
            status=manifest_status,
            error="runner reported terminal artifact status",
        )
        result = _fake_model(root, total_cost=1.25)
        result["weights_path"] = str(run_dir)
        return result

    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)
    client = TestClient(server_app.app)

    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune with a terminal artifact status",
            "budget": 10,
            "task_type": "fine-tuning",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, api_status)

    assert state["status"] == api_status
    assert state["error"] == "runner reported terminal artifact status"
    assert state["artifacts"]["modelPath"].endswith("experiments/terminal-artifact")
    assert any(
        item["type"] == log_type and "terminal artifact status" in item["message"]
        for item in state["logs"]
    )
    manifest_response = client.get(f"/api/runs/{run_id}/artifacts/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.json()["status"] == manifest_status


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
    assert state["stages"][state["stage"]]["status"] == "cancelled"
    assert state["result"] is None
    assert any("cancel" in item["message"].lower() for item in state["logs"])


def test_cancel_no_spend_dry_run_tinker_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    from src.autoresearch.config import TrainingConfig
    from src.tinker_api import sft_runner, tinker_api

    started = threading.Event()
    original_record_tokens = tinker_api.record_tokens

    def slow_first_step(job_id: str, n_tokens: int) -> None:
        if not started.is_set():
            started.set()
            time.sleep(0.2)
        original_record_tokens(job_id, n_tokens)

    def fake_invoke(prompt, budget, data_path=None, task_type_hint=None):
        root = get_output_root()
        assert root is not None
        return sft_runner.run_tinker_sft_experiment(
            TrainingConfig(model_name="Qwen/Qwen3.5-9B", batch_size=1),
            [{"input": "Question", "output": "Answer"}],
            run_id="server-dry-run",
            max_steps=250,
            output_dir=str(root / "experiments"),
        )

    monkeypatch.setattr(sft_runner, "record_tokens", slow_first_step)
    monkeypatch.setattr(server_app, "invoke_manager_graph", fake_invoke)

    client = TestClient(server_app.app)
    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a cancellable no-spend dry-run assistant",
            "budget": 3,
            "task_type": "fine-tuning",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert started.wait(timeout=5)

    cancel_response = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] in {"cancelling", "cancelled"}

    state = _wait_for_status(client, run_id, "cancelled")
    manifest_path = (
        tmp_path
        / "outputs"
        / "api-runs"
        / run_id
        / "experiments"
        / "server-dry-run"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert state["status"] == "cancelled"
    assert manifest["status"] == "CANCELLED"
    assert tinker_api.is_cancelled("server-dry-run")


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


def test_create_run_local_jsonl_reaches_tinker_dry_run_without_live_credentials(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"input": "I loved the movie.", "output": "positive"}),
                json.dumps({"input": "The wait was frustrating.", "output": "negative"}),
                json.dumps({"input": "Service was quick.", "output": "positive"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("MANAGER_REASONER", "local")
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "local")
    monkeypatch.setenv("AUTORESEARCH_EVAL_ADAPTATION", "off")
    monkeypatch.delenv("TINKER_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    _install_no_live_service_fences(monkeypatch)

    client = TestClient(server_app.app)
    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a sentiment classifier using this labeled JSONL file",
            "budget": 20,
            "task_type": "classification",
            "data_path": str(data_path),
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "complete")

    assert state["dataPath"] == str(data_path.resolve())
    assert state["result"]["n_iterations"] == 1
    assert any(
        f"Dataset source: {data_path.resolve()}" in item["message"]
        for item in state["logs"]
    )

    artifacts = state["artifacts"]
    assert artifacts is not None
    assert artifacts["sample"]["backend"] == "dry_run"
    assert artifacts["checkpoints"]["state_path"].startswith("dry-run://state/")

    manifest_file = next(
        file
        for file in artifacts["files"]
        if file["name"] == "manifest" and file["path"]
    )
    manifest = json.loads(Path(manifest_file["path"]).read_text(encoding="utf-8"))
    assert manifest["backend"] == "dry_run"
    assert manifest["training_examples"] > 0

    download = client.get(manifest_file["downloadPath"])
    assert download.status_code == 200
    assert download.json()["backend"] == "dry_run"


def test_create_run_hf_data_path_reaches_tinker_dry_run_without_live_credentials(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("MANAGER_REASONER", "local")
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "local")
    monkeypatch.setenv("AUTORESEARCH_EVAL_ADAPTATION", "off")
    monkeypatch.setenv("DATA_GENERATOR_OFFLINE", "1")
    monkeypatch.setenv("DATA_GENERATOR_MAX_ROWS_PER_DATASET", "6")
    monkeypatch.delenv("TINKER_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    _install_no_live_service_fences(monkeypatch)

    client = TestClient(server_app.app)
    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a classifier using this Hugging Face sentiment dataset",
            "budget": 20,
            "task_type": "classification",
            "data_path": "hf://SetFit/sst2",
        },
    )

    assert response.status_code == 200
    state = _wait_for_status(client, response.json()["run_id"], "complete")

    assert state["dataPath"] == "hf://SetFit/sst2"
    assert 0.0 <= state["costSpent"] <= 20.0
    assert state["result"]["n_iterations"] == 1
    assert any(
        "Dataset source: hf://SetFit/sst2" in item["message"]
        for item in state["logs"]
    )

    artifacts = state["artifacts"]
    assert artifacts is not None
    assert artifacts["metrics"]["train_loss"] >= 0.0
    assert artifacts["sample"]["backend"] == "dry_run"
    assert artifacts["checkpoints"]["state_path"].startswith("dry-run://state/")

    manifest_path = next(
        Path(file["path"])
        for file in artifacts["files"]
        if file["name"] == "manifest" and file["path"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "dry_run"
    assert manifest["dataset_path"].endswith("train_data.jsonl")
    assert manifest["training_examples"] > 0


def test_create_run_without_data_path_reaches_mode_c_tinker_dry_run_without_live_credentials(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_SPEND", "1")
    monkeypatch.setenv("MANAGER_REASONER", "local")
    monkeypatch.setenv("AUTORESEARCH_PROPOSER", "local")
    monkeypatch.setenv("AUTORESEARCH_EVAL_ADAPTATION", "off")
    monkeypatch.setenv("DATA_GENERATOR_SYNTHETIC_OFFLINE", "1")
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "synthetic")
    monkeypatch.setenv("DATA_GENERATOR_MAX_ROWS_PER_DATASET", "6")
    monkeypatch.delenv("TINKER_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    _install_no_live_service_fences(monkeypatch)

    client = TestClient(server_app.app)
    response = client.post(
        "/api/runs",
        json={
            "prompt": "Fine tune a classifier for urgent support ticket triage",
            "budget": 20,
            "task_type": "classification",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    state = _wait_for_status(client, run_id, "complete")

    assert state["dataPath"] is None
    assert 0.0 <= state["costSpent"] <= 20.0
    assert state["result"]["n_iterations"] == 1
    assert not any(
        "Dataset source:" in item["message"]
        for item in state["logs"]
    )

    artifacts = state["artifacts"]
    assert artifacts is not None
    assert f"outputs/api-runs/{run_id}/experiments/" in artifacts["modelPath"]
    assert artifacts["metrics"]["train_loss"] >= 0.0
    assert artifacts["sample"]["backend"] == "dry_run"
    assert artifacts["sample"]["text"]
    assert artifacts["checkpoints"]["state_path"].startswith("dry-run://state/")

    files = {item["name"]: item for item in artifacts["files"]}
    for name in ("manifest", "metrics", "metrics_log", "sample", "diary"):
        assert files[name]["exists"] is True
        assert files[name]["downloadPath"] == f"/api/runs/{run_id}/artifacts/{name}"

    manifest_response = client.get(files["manifest"]["downloadPath"])
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["backend"] == "dry_run"
    assert manifest["dataset_path"].endswith("train_data.jsonl")
    assert manifest["training_examples"] > 0
    assert manifest["split_examples"]["train"] > 0

    metrics_log_response = client.get(files["metrics_log"]["downloadPath"])
    assert metrics_log_response.status_code == 200
    assert '"step": 1' in metrics_log_response.text

    sample_response = client.get(files["sample"]["downloadPath"])
    assert sample_response.status_code == 200
    assert sample_response.json()["backend"] == "dry_run"


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
    assert state["stages"][state["stage"]]["status"] == "failed"
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
