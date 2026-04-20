# Retrieval Agent (Data Acquisition)

This module is the **external data acquisition planner** in the pipeline.

It answers:
- where relevant raw data may exist
- which sources to prioritize
- what to collect first, with safety constraints
- downloads raw artifacts and marks relevance against the acquisition spec

It does **not** do labeling, cleaning, structuring, or model training.

## Input contract

Manager should pass an acquisition spec shaped like:

```json
{
  "task_name": "support ticket urgency classification",
  "task_type": "text_classification",
  "target_schema": {
    "example_unit": "one ticket",
    "input_fields": ["ticket_text"],
    "output_fields": ["label"],
    "label_space": ["urgent", "normal"]
  },
  "data_requirements": {
    "preferred_sources": ["public web", "huggingface", "company docs"],
    "query_keywords": ["support ticket", "customer complaint", "incident", "priority"],
    "min_examples": 1000,
    "languages": ["en"]
  },
  "constraints": {
    "allow_scraping": true,
    "allow_api_sources": true,
    "allow_synthetic": false
  },
  "explicit_sources": []
}
```

## Current behavior

Implemented output:

`spec -> retrieval plan -> candidate sources -> ranked source list -> artifact download -> relevance check -> raw bundle report`

Every run now also writes a human-readable companion bundle:

- `SUMMARY.md` (plain-language summary of sources, artifacts, and outcomes)
- `artifact_previews/*.txt` (per-artifact text previews when available)
- `candidates.json` and `artifacts.json` for easy inspection

The JSON report also includes:

- `human_readable_dir`
- `human_readable_summary`

This version uses a LangGraph workflow:

`parse_spec -> decide_mode -> build_plan -> find_candidates -> rank -> collect_artifacts -> report`

The collector currently supports direct downloads and simple page link extraction for:

- `.zip`
- `.pdf`
- `.csv` / `.tsv`
- `.json` / `.jsonl`
- `.txt` / `.md`
- `.html` / `.htm`

Post-download sanity checks:

- ZIP integrity and presence of data-like files
- CSV/TSV minimum non-empty row checks
- JSON/JSONL parse and non-empty checks
- text/PDF minimum content-signal checks

Artifacts that are relevant but fail sanity checks are marked `filtered_out` with reasonableness notes.

Collection scope note:

- Search/discovery portal candidates are used for ranking and discovery.
- By default, artifact downloading is attempted from concrete sources (for example explicit source URLs), not generic search result pages.

## Run

```bash
python -m retrieval_agent.agent path/to/spec.json path/to/retrieval_report.json --raw-output-dir path/to/raw
```

If you use the project virtualenv:

```bash
.venv/bin/python -m retrieval_agent.agent path/to/spec.json path/to/retrieval_report.json --raw-output-dir path/to/raw
```

Optional human-readable output directory:

```bash
.venv/bin/python -m retrieval_agent.agent path/to/spec.json path/to/retrieval_report.json \
  --raw-output-dir path/to/raw \
  --human-output-dir path/to/human_readable
```

If omitted, human-readable output defaults to `human_readable/`.

Optional environment variables for LLM-backed planning (LangChain + Ollama):

- `RETRIEVAL_AGENT_MODEL` (default: `qwen2.5:7b-instruct`)
- `RETRIEVAL_AGENT_API_BASE` (default: `http://localhost:11434`)
- `RETRIEVAL_AGENT_ENABLE_LLM_RERANK` (`true`/`false`, default: `false`)
- `RETRIEVAL_AGENT_LLM_RERANK_TOP_K` (default: `12`)
- `RETRIEVAL_AGENT_LLM_RERANK_ALPHA` (default: `0.4`)

SSL/TLS troubleshooting env vars:

- `RETRIEVAL_AGENT_SSL_VERIFY` (`true`/`false`, default: `true`)
- `RETRIEVAL_AGENT_CA_BUNDLE` (path to custom CA bundle PEM)

CLI flags for hybrid rerank:

```bash
.venv/bin/python -m retrieval_agent.agent path/to/spec.json path/to/retrieval_report.json \
  --raw-output-dir path/to/raw \
  --enable-llm-rerank \
  --llm-rerank-top-k 12 \
  --llm-rerank-alpha 0.4
```

Ranking behavior with hybrid mode:

- deterministic pre-rank for all candidates
- single LLM rerank call for top-K candidates
- blended score: `final = (1 - alpha) * deterministic + alpha * llm_score`
- deterministic fallback if LLM rerank is unavailable/fails

If downloads fail with certificate errors (`CERTIFICATE_VERIFY_FAILED`):

1. Prefer setting a CA bundle:
   `export RETRIEVAL_AGENT_CA_BUNDLE=/path/to/cacert.pem`
2. Temporary debugging-only workaround:
   `export RETRIEVAL_AGENT_SSL_VERIFY=false`

To inspect exactly what URLs were attempted during collection:

```bash
python3 - <<'PY'
import json
r=json.load(open("test_report.json"))
for a in r.get("collected_artifacts", []):
    print(a["status"], a["artifact_url"])
PY
```

## Architecture

- `models.py`: acquisition contract + report schema
- `spec_parser.py`: spec validation, keyword extraction, retrieval mode decision
- `planner.py`: strategy and query generation (search-before-scrape)
- `source_finder.py`: candidate source generation
- `source_ranker.py`: weighted ranking (relevance/structure/ease/risk)
- `report.py`: final retrieval report

## Next milestone (not yet implemented)

- richer source-specific adapters (e.g., Hugging Face API client and robust dataset-page parsing)
- deeper PDF parsing and richer relevance checks
- retry/backoff/caching controls for larger crawls
