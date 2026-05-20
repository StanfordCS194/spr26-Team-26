from __future__ import annotations

import json
from pathlib import Path

from src.data_generator.artifacts import save_subagent2_artifacts
from src.data_generator.graph import build_data_generator_graph, invoke_data_generator_graph


def _base_config() -> dict:
    return {
        "data": False,
        "prompt": "deployment-style test prompt",
        "compute_budget": 5.0,
        "training_procedure": {
            "task_type": "text_classification",
            "data_format": "jsonl",
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {},
            "notes": "artifact smoke test",
        },
    }


def _mode_a_orchestrator_config() -> dict:
    config = _base_config()
    config["prompt"] = "Train a classifier on a user-provided CSV of text pairs."
    config["data"] = True
    config["training_procedure"]["data_format"] = "csv"
    config["training_procedure"]["notes"] = "Mode A orchestrator handoff"
    return config


def _mode_b_orchestrator_config() -> dict:
    config = _base_config()
    config["prompt"] = "Train a sentiment classifier from an explicit Hugging Face dataset."
    config["data_request"] = {"sources": [{"type": "hf_dataset", "id": "SetFit/sst2"}]}
    config["training_procedure"]["notes"] = "Mode B orchestrator handoff"
    return config


def _mode_c_orchestrator_config() -> dict:
    config = _base_config()
    config["prompt"] = "Build a dataset for product review classification from public web sources."
    config["training_procedure"]["data_format"] = "raw_web_sources"
    config["training_procedure"]["notes"] = "Mode C orchestrator handoff"
    return config


def _latest_artifact_dir() -> Path:
    latest_pointer = Path("artifacts") / "data_generator" / "latest_run.json"
    assert latest_pointer.exists(), f"Missing latest pointer: {latest_pointer}"
    latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
    return Path(str(latest["artifact_dir"]))


def _test_artifact_dir(test_name: str) -> Path:
    return Path("artifacts") / "data_generator" / test_name


def _fake_user_data_dir(name: str) -> Path:
    return Path("artifacts") / "fake_user_data" / name


def _assert_saved_artifacts(out_dir: Path, expected_mode: str) -> dict:
    manifest_path = out_dir / "artifact_manifest.json"
    assert manifest_path.exists(), f"Missing manifest artifact: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_path = Path(str(manifest["files"]["handoff_payload"]))
    curation_path = Path(str(manifest["files"]["curation_human_readable"]))
    debug_path = Path(str(manifest["files"]["debug_context"]))
    assert exact_path.exists(), f"Missing exact payload artifact: {exact_path}"
    assert curation_path.exists(), f"Missing curation human-readable artifact: {curation_path}"
    assert debug_path.exists(), f"Missing debug context artifact: {debug_path}"

    handoff = json.loads(exact_path.read_text(encoding="utf-8"))
    debug_context = json.loads(debug_path.read_text(encoding="utf-8"))
    curation_human = curation_path.read_text(encoding="utf-8")

    assert handoff["mode_used"] == expected_mode
    assert handoff["curation_payload"]["schema_version"] == "data_curation_input.v1"
    assert manifest["mode_used"] == expected_mode
    assert "Sub-Agent 2 Curation Input" in curation_human
    assert debug_context["mode_used"] == expected_mode
    assert debug_context["curation_human_readable"].strip() == curation_human.strip()
    return handoff


def test_mode_a_fake_user_data(monkeypatch):
    artifact_dir = _test_artifact_dir("test_mode_a_fake_user_data")
    monkeypatch.setenv("DATA_GENERATOR_ARTIFACT_DIR", str(artifact_dir))

    data_dir = _fake_user_data_dir("mode_a_rich_input")
    assert data_dir.exists(), f"Missing fake user data directory: {data_dir}"

    graph = build_data_generator_graph()
    initial_state = {
        "config": _mode_a_orchestrator_config(),
        "data_path": str(data_dir),
        "mode": None,
        "raw_data": None,
        "hf_candidates": [],
        "selected_candidate": None,
        "schema": None,
        "dataset": None,
        "validation_report": None,
        "handoff": None,
        "web_plan": None,
        "web_search_results": [],
        "web_pages": [],
        "human_readable": None,
    }

    final_state = graph.invoke(initial_state)
    handoff = final_state["handoff"]
    save_subagent2_artifacts(handoff)

    assert handoff["mode_used"] == "A"
    assert handoff["source_metadata"]["data_path"] == str(data_dir)
    assert handoff["raw_data"]["format_meta"]["file_type"] == "directory"
    assert len(handoff["raw_data"]["records"]) == 7

    saved_handoff = _assert_saved_artifacts(artifact_dir, "A")
    curation_payload = saved_handoff["curation_payload"]
    assert curation_payload["record_count"] == 21
    assert curation_payload["format_meta"]["file_type"] == "directory"
    assert curation_payload["provenance_summary"]["data_path"] == str(data_dir)
    assert any(rec["metadata"].get("container_file_type") == "tsv" for rec in curation_payload["records"])
    assert any(rec["metadata"].get("container_file_type") == "json" for rec in curation_payload["records"])
    assert any(rec["source_locator"].endswith("reference.png") for rec in curation_payload["records"])
    assert any("ticket_id" in rec["input"] or rec["metadata"].get("source_path", "").endswith("support.jsonl") for rec in curation_payload["records"])

    source_report = artifact_dir / "source_human_readable.md"
    source_report_text = source_report.read_text(encoding="utf-8")
    assert "Detected file type: directory" in source_report_text
    assert "Records loaded: 7" in source_report_text
    assert "source file:" in source_report_text
    assert "reference.png" in source_report_text

    curation_report = artifact_dir / "human_readable.md"
    curation_report_text = curation_report.read_text(encoding="utf-8")
    assert "Record count: 21" in curation_report_text
    assert "Mode hint: A" in curation_report_text
    assert "kind=local" in curation_report_text

    latest_pointer = Path("artifacts") / "data_generator" / "latest_run.json"
    latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
    assert latest["artifact_dir"] == str(artifact_dir)


def test_mode_b_saves_deployment_style_handoff_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "DATA_GENERATOR_ARTIFACT_DIR",
        str(_test_artifact_dir("test_mode_b_saves_deployment_style_handoff_artifacts")),
    )
    monkeypatch.setenv("DATA_GENERATOR_OFFLINE", "1")

    handoff = invoke_data_generator_graph(config=_mode_b_orchestrator_config(), data_path=None)

    assert handoff["mode_used"] == "B"
    assert handoff["action"] == "structure_data"
    assert handoff["hf_candidates"]
    artifact_dir = _test_artifact_dir("test_mode_b_saves_deployment_style_handoff_artifacts")
    saved_handoff = _assert_saved_artifacts(artifact_dir, "B")
    assert saved_handoff["curation_payload"]["record_count"] >= 1


def test_mode_c_saves_deployment_style_handoff_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "DATA_GENERATOR_ARTIFACT_DIR",
        str(_test_artifact_dir("test_mode_c_saves_deployment_style_handoff_artifacts")),
    )
    monkeypatch.setenv("DATA_GENERATOR_MOCK_SCENARIO", "mixed_sources")

    from src.data_generator.mode_c import nodes as mode_c_nodes

    def fake_search(_web_plan):
        return [
            {
                "source": "web_search",
                "provider": "tavily",
                "query": "sample query",
                "url": "https://example.com/data",
                "domain": "example.com",
                "title": "Example dataset",
                "snippet": "example snippet",
                "provider_score": 0.9,
            }
        ]

    def fake_crawl(_search_results, _web_plan):
        return [
            {
                "source": "web_page",
                "url": "https://example.com/data",
                "domain": "example.com",
                "title": "Example dataset",
                "query": "sample query",
                "snippet": "example snippet",
                "content": "Example page content for mode C artifact test.",
                "metadata": {"http_status": 200, "extraction_method": "trafilatura"},
                "error": None,
            }
        ]

    monkeypatch.setattr(mode_c_nodes, "search_web_sources", fake_search)
    monkeypatch.setattr(mode_c_nodes, "crawl_and_extract_pages", fake_crawl)

    handoff = invoke_data_generator_graph(config=_mode_c_orchestrator_config(), data_path=None)

    assert handoff["mode_used"] == "C"
    assert "Mode C Web Acquisition Report" in handoff["human_readable"]
    artifact_dir = _test_artifact_dir("test_mode_c_saves_deployment_style_handoff_artifacts")
    saved_handoff = _assert_saved_artifacts(artifact_dir, "C")
    assert saved_handoff["curation_payload"]["records"][0]["source_locator"] == "https://example.com/data"

    source_report = artifact_dir / "source_human_readable.md"
    assert source_report.exists()
    assert "Mode C Web Acquisition Report" in source_report.read_text(encoding="utf-8")


def test_default_artifact_saving_uses_unique_run_directories(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DATA_GENERATOR_ARTIFACT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    handoff = {
        "target_subagent": "data_curation",
        "action": "structure_data",
        "mode_used": "A",
        "curation_payload": {"record_count": 1},
        "curation_human_readable": "Sub-Agent 2 Curation Input\nRecord count: 1",
    }

    saved_one = save_subagent2_artifacts(handoff)
    saved_two = save_subagent2_artifacts(handoff)

    first_dir = Path(saved_one["manifest"]).parent
    second_dir = Path(saved_two["manifest"]).parent

    assert first_dir != second_dir
    assert first_dir.exists()
    assert second_dir.exists()

    latest_pointer = tmp_path / "artifacts" / "data_generator" / "latest_run.json"
    assert latest_pointer.exists()
    latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
    assert latest["artifact_dir"] == str(second_dir)
