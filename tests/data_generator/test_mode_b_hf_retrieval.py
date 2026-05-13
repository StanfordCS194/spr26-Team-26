from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_generator.graph import invoke_data_generator_graph

EXAMPLE_REPORT_PATH = (
    REPO_ROOT
    / "tests"
    / "data_generator"
    / "fixtures"
    / "hf_robust_structuring_run_v1"
    / "example_orchestrator_mode_b.json"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "tests"
    / "data_generator"
    / "results"
    / "hf_robust_structuring_run_v1"
)
FIXTURE_RUN_DIR = (
    REPO_ROOT
    / "tests"
    / "data_generator"
    / "fixtures"
    / "hf_robust_structuring_run_v1"
)

TARGET_LIVE_DATASET_COUNT = 50
MIN_EXAMPLES_PER_RETRIEVED_DATASET = 3
MAX_EXAMPLES_PER_PREVIEW_DATASET = 6

# Seed list ensures we always request known valid public datasets first.
LIVE_HF_DATASET_SEEDS = [
    "OpenAssistant/oasst1",
    "Anthropic/hh-rlhf",
    "mteb/banking77",
    "rajpurkar/squad",
    "stanfordnlp/snli",
    "DeepPavlov/clinc150",
    "SetFit/sst2",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
]


def _extract_hf_dataset_ids_from_report(report: dict) -> list[str]:
    dataset_ids: list[str] = []
    seen: set[str] = set()

    report_config = report.get("config")
    if isinstance(report_config, dict):
        explicit_ids = report_config.get("hf_dataset_ids")
        if isinstance(explicit_ids, list):
            for item in explicit_ids:
                ds_id = str(item).strip()
                if "/" in ds_id and ds_id not in seen:
                    seen.add(ds_id)
                    dataset_ids.append(ds_id)
            if dataset_ids:
                return dataset_ids

    data_request = report.get("data_request")
    if isinstance(data_request, dict):
        sources = data_request.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue
                if source.get("type") != "hf_dataset":
                    continue
                ds_id = str(source.get("id", "")).strip()
                if "/" in ds_id and ds_id not in seen:
                    seen.add(ds_id)
                    dataset_ids.append(ds_id)
            if dataset_ids:
                return dataset_ids

    for candidate in report.get("candidates", []):
        if candidate.get("source_type") != "explicit":
            continue
        url = str(candidate.get("url", ""))
        marker = "/datasets/"
        if marker not in url:
            continue
        dataset_id = url.split(marker, 1)[1].strip("/")
        if dataset_id and dataset_id not in seen:
            seen.add(dataset_id)
            dataset_ids.append(dataset_id)

    return dataset_ids


def _build_config_from_report(report: dict, hf_dataset_ids: list[str]) -> dict:
    report_config = report.get("config")
    if isinstance(report_config, dict):
        report_config["hf_dataset_ids"] = hf_dataset_ids
        return report_config

    req = report.get("training_requirements", {})
    if not isinstance(req, dict):
        req = {}
    budget = report.get("budget", {})
    if not isinstance(budget, dict):
        budget = {}

    return {
        "data": False,
        "prompt": report.get("objective", report.get("task_name", "mode_b_hf_run")),
        "compute_budget": float(budget.get("total_usd", 20.0)),
        "training_procedure": {
            "task_type": req.get("task_type", "text-classification"),
            "data_format": req.get("data_format", "jsonl"),
            "training_type": req.get("training_type", "SFT"),
            "base_model": req.get("base_model", "bert-base-uncased"),
            "hyperparameters": req.get("hyperparameters", {"lr": 2e-5, "epochs": 2}),
            "notes": req.get("notes", "Derived from orchestrator report fixture"),
        },
        "hf_dataset_ids": hf_dataset_ids,
    }


def _request_live_hf_dataset_ids(target_count: int) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()

    for dataset_id in LIVE_HF_DATASET_SEEDS:
        if dataset_id not in seen:
            seen.add(dataset_id)
            requested.append(dataset_id)
        if len(requested) >= target_count:
            return requested

    # Expand to the requested count using live Hub listing from known orgs.
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        orgs = ["SetFit", "mteb", "cardiffnlp", "allenai", "google-research-datasets"]
        for org in orgs:
            for ds in api.list_datasets(author=org, limit=200):
                dataset_id = str(getattr(ds, "id", "")).strip()
                if "/" not in dataset_id:
                    continue
                if dataset_id in seen:
                    continue
                seen.add(dataset_id)
                requested.append(dataset_id)
                if len(requested) >= target_count:
                    return requested
    except Exception:
        # Keep seed-only fallback if listing fails.
        return requested

    return requested


def _collect_fixture_labeled_examples(
    max_examples_per_file: int = 6,
    max_total_examples: int = 240,
) -> dict[str, list[dict[str, Any]]]:
    parquet_dir = FIXTURE_RUN_DIR / "raw"
    if not parquet_dir.exists():
        return {}

    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}

    previews_by_file: dict[str, list[dict[str, Any]]] = {}
    total_examples = 0
    text_candidates = ["text", "input", "sentence", "question", "utterance", "content"]
    label_candidates = ["label_text", "label", "intent", "target", "output"]

    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        if total_examples >= max_total_examples:
            break
        try:
            table = pq.read_table(parquet_file)
        except Exception:
            continue

        columns = table.column_names
        text_col = next((c for c in text_candidates if c in columns), None)
        label_col = next((c for c in label_candidates if c in columns), None)
        if not text_col or not label_col:
            continue

        data = table.select([text_col, label_col]).to_pydict()
        texts = data.get(text_col, [])
        labels = data.get(label_col, [])
        file_examples: list[dict[str, Any]] = []

        for idx in range(min(len(texts), len(labels))):
            text_val = str(texts[idx]).strip()
            label_val = str(labels[idx]).strip()
            if not text_val:
                continue
            file_examples.append(
                {
                    "source_file": parquet_file.name,
                    "row_index": idx,
                    "input": text_val[:240],
                    "label": label_val,
                }
            )
            if len(file_examples) >= max_examples_per_file:
                break
            if total_examples + len(file_examples) >= max_total_examples:
                break

        if file_examples:
            previews_by_file[parquet_file.name] = file_examples
            total_examples += len(file_examples)

    return previews_by_file


def _collect_live_labeled_examples_by_source(
    records: list[Any], max_examples_per_source: int = MAX_EXAMPLES_PER_PREVIEW_DATASET
) -> dict[str, list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("note") == "fetch_failed":
            continue
        source = str(rec.get("source", "unknown")).strip() or "unknown"
        input_text = str(rec.get("input") or rec.get("content") or "").strip()
        label_text = str(rec.get("label") or rec.get("label_text") or "").strip()
        if not input_text:
            continue
        by_source.setdefault(source, [])
        if len(by_source[source]) >= max_examples_per_source:
            continue
        by_source[source].append(
            {
                "input": input_text[:240],
                "label": label_text,
            }
        )
    return by_source


def _collect_raw_source_previews(
    records: list[Any], max_examples_per_source: int = MAX_EXAMPLES_PER_PREVIEW_DATASET
) -> dict[str, list[dict[str, str]]]:
    previews_by_source: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        source = str(rec.get("source", "unknown")).strip() or "unknown"
        content = str(rec.get("content", "")).strip()
        item = {"content": content[:280]}
        previews_by_source.setdefault(source, [])
        if len(previews_by_source[source]) < max_examples_per_source:
            previews_by_source[source].append(item)
    return previews_by_source


def _write_subagent2_artifacts(handoff: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exact_path = OUTPUT_DIR / "subagent2_input_exact.json"
    exact_path.write_text(
        json.dumps(handoff, indent=2),
        encoding="utf-8",
    )

    raw_data = handoff.get("raw_data", {})
    records = raw_data.get("records", []) if isinstance(raw_data, dict) else []
    hf_candidates = handoff.get("hf_candidates", [])

    candidate_lines: list[str] = []
    for idx, cand in enumerate(hf_candidates, start=1):
        if isinstance(cand, dict):
            candidate_lines.append(f"{idx}. {cand.get('name', 'unknown')}")
        else:
            candidate_lines.append(f"{idx}. {str(cand)}")

    source_previews = _collect_raw_source_previews(
        records, max_examples_per_source=MAX_EXAMPLES_PER_PREVIEW_DATASET
    )

    source_lines: list[str] = []
    for source in sorted(source_previews):
        source_lines.append(f"### {source}")
        examples = source_previews[source]
        if not examples:
            source_lines.append("(no records found)")
            continue
        for idx, ex in enumerate(examples, start=1):
            compact = ex["content"].replace("\n", " ").strip()
            source_lines.append(f"{idx}. {compact}")

    live_examples_by_source = _collect_live_labeled_examples_by_source(
        records, max_examples_per_source=MAX_EXAMPLES_PER_PREVIEW_DATASET
    )
    used_live_examples = bool(live_examples_by_source)
    labeled_lines: list[str] = []
    labeled_examples_total = 0

    if used_live_examples:
        for source in sorted(live_examples_by_source):
            labeled_lines.append(f"### {source}")
            for idx, ex in enumerate(live_examples_by_source[source], start=1):
                labeled_examples_total += 1
                labeled_lines.append(
                    f"{idx}. label={ex['label']} | input={ex['input']}"
                )
    else:
        labeled_examples_by_file = _collect_fixture_labeled_examples(
            max_examples_per_file=6, max_total_examples=240
        )
        for source_file in sorted(labeled_examples_by_file):
            labeled_lines.append(f"### {source_file}")
            for idx, ex in enumerate(labeled_examples_by_file[source_file], start=1):
                labeled_examples_total += 1
                labeled_lines.append(
                    f"{idx}. label={ex['label']} | input={ex['input']}"
                )

    (OUTPUT_DIR / "subagent2_labeled_examples_preview.json").write_text(
        json.dumps(
            {
                "preview_mode": "live_records" if used_live_examples else "fixture_files",
                "total_examples": labeled_examples_total,
                "total_sources": (
                    len(live_examples_by_source)
                    if used_live_examples
                    else len(_collect_fixture_labeled_examples(max_examples_per_file=6))
                ),
                "examples_by_source": live_examples_by_source if used_live_examples else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "subagent2_raw_source_previews.json").write_text(
        json.dumps(
            {
                "total_sources": len(source_previews),
                "examples_per_source_limit": MAX_EXAMPLES_PER_PREVIEW_DATASET,
                "records_by_source": source_previews,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md = "\n".join(
        [
            "# Sub-Agent 2 Input (Human Readable)",
            "",
            "This is the same payload sent to the Data Curation sub-agent.",
            "",
            f"- target_subagent: `{handoff.get('target_subagent')}`",
            f"- action: `{handoff.get('action')}`",
            f"- verification_level: `{handoff.get('verification_level')}`",
            f"- mode_used: `{handoff.get('mode_used')}`",
            f"- raw_data_records: `{len(records)}`",
            f"- raw_data_sources: `{len(source_previews)}`",
            f"- hf_candidates: `{len(hf_candidates)}`",
            f"- labeled_examples_preview: `{labeled_examples_total}`",
            f"- labeled_preview_mode: `{'live_records' if used_live_examples else 'fixture_files'}`",
            "",
            "## Candidate Datasets (all)",
            *(candidate_lines if candidate_lines else ["(none)"]),
            "",
            "## Raw Data Preview By Retrieved Dataset/Source",
            *(
                source_lines
                if source_lines
                else ["(none found - raw_data.records was empty)"]
            ),
            "",
            "## Labeled Examples Preview (real fetched rows grouped by dataset/source)",
            *(
                labeled_lines
                if labeled_lines
                else ["(none found - check fixture/parquet availability)"]
            ),
            "",
            "## Exact JSON",
            "- `subagent2_input_exact.json`",
            "- `subagent2_labeled_examples_preview.json`",
            "- `subagent2_raw_source_previews.json`",
        ]
    )

    human_path = OUTPUT_DIR / "subagent2_input_human_readable.md"
    labeled_path = OUTPUT_DIR / "subagent2_labeled_examples_preview.json"
    raw_source_path = OUTPUT_DIR / "subagent2_raw_source_previews.json"
    human_path.write_text(md, encoding="utf-8")
    print("")
    print("[artifact] Saved sub-agent artifacts:")
    print(f"  - exact payload json: {exact_path}")
    print(f"  - human readable md:  {human_path}")
    print(f"  - labeled preview:    {labeled_path}")
    print(f"  - raw source preview: {raw_source_path}")
    print("")


def test_hf_retreival(monkeypatch, tmp_path: Path) -> None:
    assert EXAMPLE_REPORT_PATH.exists(), f"Missing example report: {EXAMPLE_REPORT_PATH}"

    report = json.loads(EXAMPLE_REPORT_PATH.read_text(encoding="utf-8"))
    hf_dataset_ids = _extract_hf_dataset_ids_from_report(report)
    assert hf_dataset_ids, "No explicit HF dataset IDs extracted from example report."

    # Force live behavior by removing offline override entirely.
    monkeypatch.delenv("DATA_GENERATOR_OFFLINE", raising=False)
    monkeypatch.setenv("DATA_GENERATOR_MAX_ROWS_PER_DATASET", "6")
    monkeypatch.setenv("DATA_GENERATOR_MAX_TOTAL_RECORDS", "600")
    print("")
    print("[run] Data Generator HF retrieval test")
    print(
        "[run] DATA_GENERATOR_MAX_ROWS_PER_DATASET="
        "6"
    )
    print(
        "[run] DATA_GENERATOR_MAX_TOTAL_RECORDS="
        "600"
    )
    # Force a fresh cache location each run so datasets are uncached and
    # HF progress bars are visible (download + preparation steps).
    hf_home = tmp_path / "hf_home"
    hf_hub_cache = tmp_path / "hf_hub_cache"
    hf_datasets_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_datasets_cache))
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("DATASETS_DISABLE_PROGRESS_BAR", raising=False)
    print(
        "[run] Live mode is ON. The graph will attempt real Hugging Face retrieval."
    )
    print(f"[run] HF_HOME={hf_home}")
    print(f"[run] HUGGINGFACE_HUB_CACHE={hf_hub_cache}")
    print(f"[run] HF_DATASETS_CACHE={hf_datasets_cache}")
    print("")

    requested_dataset_ids = _request_live_hf_dataset_ids(TARGET_LIVE_DATASET_COUNT)
    print(f"[run] Requested dataset count: {len(requested_dataset_ids)}")
    print(f"[run] Requested datasets: {requested_dataset_ids}")
    assert len(requested_dataset_ids) >= TARGET_LIVE_DATASET_COUNT, (
        f"Could only request {len(requested_dataset_ids)} datasets; expected "
        f"{TARGET_LIVE_DATASET_COUNT}. Check HF Hub listing/network."
    )
    config = _build_config_from_report(report, requested_dataset_ids)
    handoff = invoke_data_generator_graph(config=config, data_path=None)

    assert handoff["target_subagent"] == "data_curation"
    assert handoff["action"] == "validate_hf_dataset"
    assert handoff["verification_level"] == "light"
    assert handoff["mode_used"] == "B"
    assert handoff["hf_candidates"], "Expected non-empty HF candidates in handoff."
    records = handoff.get("raw_data", {}).get("records", [])
    successful_rows = [
        rec
        for rec in records
        if isinstance(rec, dict)
        and rec.get("note") != "fetch_failed"
        and str(rec.get("input", "")).strip()
    ]
    failed_rows = [
        rec for rec in records if isinstance(rec, dict) and rec.get("note") == "fetch_failed"
    ]
    print(f"[run] Retrieved records: total={len(records)}")
    print(
        "[run] Retrieval results: "
        f"success_rows={len(successful_rows)} failed_rows={len(failed_rows)}"
    )
    assert successful_rows, (
        "Live HF retrieval produced zero usable rows. "
        "Check network access, huggingface credentials/rate limits, "
        "or run a smaller dataset subset."
    )
    placeholder_hits = [
        rec
        for rec in records
        if isinstance(rec, dict)
        and "offline_placeholder" in str(rec.get("content", ""))
    ]
    assert not placeholder_hits, "Live mode unexpectedly returned offline placeholders."
    success_by_source: dict[str, int] = {}
    for rec in successful_rows:
        source = str(rec.get("source", "unknown"))
        success_by_source[source] = success_by_source.get(source, 0) + 1
    sources_with_multiple = [
        src
        for src, count in success_by_source.items()
        if count >= MIN_EXAMPLES_PER_RETRIEVED_DATASET
    ]
    print(
        f"[run] Sources with >={MIN_EXAMPLES_PER_RETRIEVED_DATASET} examples: "
        f"{len(sources_with_multiple)} / {len(success_by_source)}"
    )
    assert len(success_by_source) >= 25, (
        "Expected at least 25 datasets to return usable rows in 50-dataset run, "
        f"got {len(success_by_source)}."
    )
    assert len(sources_with_multiple) == len(success_by_source), (
        "Every retrieved dataset must have multiple examples for preview quality. "
        f"Expected all >= {MIN_EXAMPLES_PER_RETRIEVED_DATASET}, got "
        f"{len(sources_with_multiple)} / {len(success_by_source)}."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "source_report": str(EXAMPLE_REPORT_PATH.relative_to(REPO_ROOT)),
        "extracted_hf_dataset_ids": hf_dataset_ids,
        "handoff": handoff,
    }
    handoff_path = OUTPUT_DIR / "mode_b_handoff_from_report.json"
    handoff_path.write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print("[artifact] Saved mode-B handoff:")
    print(f"  - handoff json: {handoff_path}")
    _write_subagent2_artifacts(handoff)
