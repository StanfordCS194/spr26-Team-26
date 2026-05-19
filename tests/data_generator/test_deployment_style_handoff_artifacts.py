from __future__ import annotations

import json
from pathlib import Path

from src.data_generator.artifacts import save_subagent2_artifacts
from src.data_generator.graph import invoke_data_generator_graph


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


def test_mode_a_saves_deployment_style_handoff_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "DATA_GENERATOR_ARTIFACT_DIR",
        str(_test_artifact_dir("test_mode_a_saves_deployment_style_handoff_artifacts")),
    )

    data_path = tmp_path / "mode_a_input.csv"
    data_path.write_text("input,output\nhello,world\nfoo,bar\n", encoding="utf-8")

    handoff = invoke_data_generator_graph(config=_mode_a_orchestrator_config(), data_path=str(data_path))

    assert handoff["mode_used"] == "A"
    assert handoff["raw_data"]["records"]
    artifact_dir = _test_artifact_dir("test_mode_a_saves_deployment_style_handoff_artifacts")
    saved_handoff = _assert_saved_artifacts(artifact_dir, "A")
    assert saved_handoff["curation_payload"]["records"][0]["input"] == "hello"
    source_report = artifact_dir / "source_human_readable.md"
    assert source_report.exists()
    assert "Mode A Local Data Acquisition Report" in source_report.read_text(encoding="utf-8")


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
    monkeypatch.setenv("DATA_GENERATOR_MODE_C_BACKEND", "web")

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
    assert saved_handoff["curation_payload"]["format_meta"]["mode_c_fallback"] == "synthetic"
    assert saved_handoff["curation_payload"]["format_meta"]["web_source_urls"] == [
        "https://example.com/data"
    ]
    assert saved_handoff["curation_payload"]["records"][0]["messages"][-1]["role"] == "assistant"

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
