# Data Generator Test Guide

This README explains how to activate the Python environment, run the Data Generator tests, and find any saved artifact outputs.

## Activate The Environment

From the repository root:

```bash
source .venv/bin/activate
```

If your virtual environment uses a different folder name, replace `.venv` with that name.

## Why `PYTHONPATH=.` Is Needed

The tests import modules using the repo-root package path, for example:

```python
from src.data_generator.mode_a import load_raw_data
```

Running tests with `PYTHONPATH=.` tells Python to treat the repository root as an import root.

## Run The Main Mode A Local-Loading Test

From the repository root:

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q tests/data_generator/test_mode_a_local_data.py
```

This test checks local file and directory loading for Mode A. It verifies behavior for CSV, TSV, JSON, JSONL, text, image files, and generic directories.

## Run The Artifact-Writing Tests

To run the deployment-style artifact tests:

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q tests/data_generator/test_deployment_style_handoff_artifacts.py
```

To run only the richer persistent fake-user-data Mode A test:

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q tests/data_generator/test_deployment_style_handoff_artifacts.py -k test_mode_a_fake_user_data
```

## Where The Fake User Data Lives

The persistent deployment-style fake local input used by the richer Mode A artifact test is stored here:

- [`artifacts/fake_user_data/mode_a_rich_input`](</Users/ronpolonsky/Desktop/CS194W/spr26-Team-26/artifacts/fake_user_data/mode_a_rich_input>)

This directory contains mixed local inputs including:

- `reviews.csv`
- `support.jsonl`
- `eval.tsv`
- `examples.json`
- `notes.txt`
- `image_caption_seed.json`
- `reference.png`

## Where To Find Test Results

There are two kinds of results:

1. Pytest pass/fail output

This appears directly in the terminal after the test finishes.

Example:

```text
8 passed, 5 warnings in 2.22s
```

2. Persistent artifact outputs

Only some tests write saved artifacts to disk. Those outputs are written under:

- [`artifacts/data_generator`](</Users/ronpolonsky/Desktop/CS194W/spr26-Team-26/artifacts/data_generator>)

Useful output directories include:

- [`artifacts/data_generator/test_mode_a_fake_user_data`](</Users/ronpolonsky/Desktop/CS194W/spr26-Team-26/artifacts/data_generator/test_mode_a_fake_user_data>)
- [`artifacts/data_generator/test_mode_c_web_robust`](</Users/ronpolonsky/Desktop/CS194W/spr26-Team-26/artifacts/data_generator/test_mode_c_web_robust>)

## Which Files To Open Inside An Artifact Directory

The most useful files are:

- `artifact_manifest.json`: summary of what was saved
- `raw_handoff_data.json`: normalized handoff payload for sub-agent 2
- `debug_context.json`: richer debug context including the original handoff data
- `human_readable.md`: curation-facing human-readable summary
- `source_human_readable.md`: source acquisition report from the original mode

## Quick Interpretation Tip

One source file can produce many `record_id` values in the final handoff payload.

Examples:

- one CSV file -> one normalized record per row
- one JSONL file -> one normalized record per line/object
- one text file -> one normalized record per non-empty line
- one image file -> usually one normalized record
