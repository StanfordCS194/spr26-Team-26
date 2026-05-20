# Data Generator Tests

This is the dedicated test area for `src/data_generator`.

## Current Test Scope

- Mode B acquisition routing from explicit Hugging Face dataset IDs.
- Handoff contract to sub-agent 2 (`validate_hf_dataset`).
- Uses the previous real example report at:
- `tests/data_generator/fixtures/hf_robust_structuring_run_v1/example_orchestrator_mode_b.json`
- Live Hugging Face retrieval is opt-in:
  - `RUN_LIVE_HF_RETRIEVAL=1 python -m pytest tests/data_generator/test_mode_b_hf_retrieval.py -q -s`
  - `LIVE_HF_DATASET_COUNT` overrides the default 50-dataset run.
  - `LIVE_HF_SMALL_DATASET_COUNT` overrides the default 10-dataset run.

## Output Location

Generated test artifacts are written to:
- `artifacts/data_generator/`
