from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_generator.graph import invoke_data_generator_graph

EXAMPLE_REPORT_PATH = REPO_ROOT / "tests" / "data_generator" / "fixtures" / "hf_robust_structuring_run_v1" / "example_orchestrator_mode_b.json"
OUTPUT_DIR = REPO_ROOT / "tests" / "data_generator" / "results" / "hf_robust_structuring_run_v1"
FIXTURE_RUN_DIR = REPO_ROOT / "tests" / "data_generator" / "fixtures" / "hf_robust_structuring_run_v1"


def extract_hf_dataset_ids_from_report(report: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    report_config = report.get("config")
    if isinstance(report_config, dict):
        explicit_ids = report_config.get("hf_dataset_ids")
        if isinstance(explicit_ids, list):
            for item in explicit_ids:
                ds_id = str(item).strip()
                if "/" in ds_id and ds_id not in seen:
                    seen.add(ds_id)
                    ids.append(ds_id)
            if ids:
                return ids

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
                    ids.append(ds_id)
            if ids:
                return ids

    for candidate in report.get("candidates", []):
        if candidate.get("source_type") != "explicit":
            continue
        url = str(candidate.get("url", ""))
        marker = "/datasets/"
        if marker not in url:
            continue
        ds_id = url.split(marker, 1)[1].strip("/")
        if ds_id and ds_id not in seen:
            seen.add(ds_id)
            ids.append(ds_id)
    return ids


def build_config_from_report(report: dict, hf_dataset_ids: list[str]) -> dict:
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
            "notes": req.get("notes", "Mode B smoke run from orchestrator report fixture"),
        },
        "hf_dataset_ids": hf_dataset_ids,
    }


def collect_fixture_labeled_examples(max_examples: int = 12) -> list[dict[str, Any]]:
    parquet_dir = FIXTURE_RUN_DIR / "raw"
    if not parquet_dir.exists():
        return []

    try:
        import pyarrow.parquet as pq
    except Exception:
        return []

    previews: list[dict[str, Any]] = []
    text_candidates = ["text", "input", "sentence", "question", "utterance", "content"]
    label_candidates = ["label_text", "label", "intent", "target", "output"]

    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        if len(previews) >= max_examples:
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
        for idx in range(min(len(texts), len(labels))):
            text_val = str(texts[idx]).strip()
            label_val = str(labels[idx]).strip()
            if not text_val:
                continue
            previews.append(
                {
                    "source_file": parquet_file.name,
                    "input": text_val[:240],
                    "label": label_val,
                }
            )
            if len(previews) >= max_examples:
                break

    return previews


def write_subagent2_artifacts(handoff: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exact_path = OUTPUT_DIR / "subagent2_input_exact.json"
    exact_path.write_text(
        json.dumps(handoff, indent=2),
        encoding="utf-8",
    )

    raw_data = handoff.get("raw_data", {})
    records = raw_data.get("records", []) if isinstance(raw_data, dict) else []
    hf_candidates = handoff.get("hf_candidates", [])

    sample_lines: list[str] = []
    for idx, rec in enumerate(records[:5], start=1):
        if isinstance(rec, dict):
            source = str(rec.get("source", "unknown"))
            note = str(rec.get("note", "")).strip()
            if note:
                err = str(rec.get("error", ""))[:160].replace("\n", " ")
                sample_lines.append(f"{idx}. source={source} | note={note} | error={err}")
            else:
                input_text = str(rec.get("input", rec.get("content", "")))[:140].replace("\n", " ")
                label = str(rec.get("label", "")).strip()
                split = str(rec.get("split", "")).strip()
                sample_lines.append(f"{idx}. source={source} | split={split} | label={label} | input={input_text}")
        else:
            sample_lines.append(f"{idx}. {str(rec)[:140]}")

    candidate_lines: list[str] = []
    for idx, cand in enumerate(hf_candidates[:10], start=1):
        if isinstance(cand, dict):
            candidate_lines.append(f"{idx}. {cand.get('name', 'unknown')}")
        else:
            candidate_lines.append(f"{idx}. {str(cand)}")

    labeled_examples = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("note") == "fetch_failed":
            continue
        text_val = str(rec.get("input") or rec.get("content") or "").strip()
        label_val = str(rec.get("label") or rec.get("label_text") or "").strip()
        if not text_val:
            continue
        labeled_examples.append(
            {
                "source_file": str(rec.get("source", "raw_data")),
                "input": text_val[:240],
                "label": label_val,
            }
        )
        if len(labeled_examples) >= 12:
            break

    failed_records = [rec for rec in records if isinstance(rec, dict) and rec.get("note") == "fetch_failed"]
    labeled_lines = [
        f"{idx}. [{ex['source_file']}] label={ex['label']} | input={ex['input']}"
        for idx, ex in enumerate(labeled_examples[:8], start=1)
    ]

    (OUTPUT_DIR / "subagent2_labeled_examples_preview.json").write_text(
        json.dumps({"examples": labeled_examples}, indent=2),
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
            f"- hf_candidates: `{len(hf_candidates)}`",
            f"- labeled_examples_preview: `{len(labeled_examples)}`",
            f"- fetch_failed_records: `{len(failed_records)}`",
            "",
            "## Candidate Datasets (first 10)",
            *(candidate_lines if candidate_lines else ["(none)"]),
            "",
            "## Raw Data Sample (first 5 records)",
            *(sample_lines if sample_lines else ["(none)"]),
            "",
            "## Labeled Examples Preview (from fetched handoff data, first 8)",
            *(labeled_lines if labeled_lines else ["(none - dataset fetch likely failed)"]),
            "",
            "## Exact JSON",
            "- `subagent2_input_exact.json`",
            "- `subagent2_labeled_examples_preview.json`",
        ]
    )

    human_path = OUTPUT_DIR / "subagent2_input_human_readable.md"
    human_path.write_text(md, encoding="utf-8")
    print(f"[saved] sub-agent2 exact payload: {exact_path}")
    print(f"[saved] sub-agent2 human summary: {human_path}")
    print(f"[saved] labeled examples preview: {OUTPUT_DIR / 'subagent2_labeled_examples_preview.json'}")


def main() -> None:
    if not EXAMPLE_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing: {EXAMPLE_REPORT_PATH}")

    report = json.loads(EXAMPLE_REPORT_PATH.read_text(encoding="utf-8"))
    hf_dataset_ids = extract_hf_dataset_ids_from_report(report)
    if not hf_dataset_ids:
        raise RuntimeError("No explicit HF dataset IDs found in example report.")

    offline = "--offline" in set(sys.argv[1:])
    if offline:
        os.environ["DATA_GENERATOR_OFFLINE"] = "1"
    else:
        os.environ.setdefault("DATA_GENERATOR_OFFLINE", "0")
    os.environ.setdefault("DATA_GENERATOR_MAX_ROWS_PER_DATASET", "40")
    os.environ.setdefault("DATA_GENERATOR_MAX_TOTAL_RECORDS", "400")
    print(f"[mode] DATA_GENERATOR_OFFLINE={os.environ.get('DATA_GENERATOR_OFFLINE')}")
    print(f"[mode] DATA_GENERATOR_MAX_ROWS_PER_DATASET={os.environ.get('DATA_GENERATOR_MAX_ROWS_PER_DATASET')}")
    print(f"[mode] DATA_GENERATOR_MAX_TOTAL_RECORDS={os.environ.get('DATA_GENERATOR_MAX_TOTAL_RECORDS')}")

    config = build_config_from_report(report, hf_dataset_ids)

    handoff = invoke_data_generator_graph(config=config, data_path=None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mode_b_handoff_from_report_script.json"
    out_path.write_text(
        json.dumps(
            {
                "source_report": str(EXAMPLE_REPORT_PATH.relative_to(REPO_ROOT)),
                "extracted_hf_dataset_ids": hf_dataset_ids,
                "handoff": handoff,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_subagent2_artifacts(handoff)
    print(f"[saved] mode-b handoff artifact: {out_path}")


if __name__ == "__main__":
    main()
