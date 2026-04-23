# Data Generator Tests

This is the dedicated test area for `src/data_generator`.

## Current Test Scope

- Mode B acquisition routing from explicit Hugging Face dataset IDs.
- Handoff contract to sub-agent 2 (`validate_hf_dataset`).
- Uses the previous real example report at:
  - `src/data_generator/tests/fixtures/hf_robust_structuring_run_v1/example_orchestrator_mode_b.json`

## Output Location

Generated test artifacts are written to:
- `src/data_generator/tests/results/hf_robust_structuring_run_v1/`
This keeps both fixtures and outputs colocated with the Data Generator module.
