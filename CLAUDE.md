# CLAUDE.md — Autonomous ML Training Agent

## Before writing any code, read the spec

All function signatures, inputs/outputs, shared types, and control flow are defined in:

```
spec-site/content/spec.ts
```

This is the single source of truth. Before implementing any function, verify:
1. The exact signature matches what's in `spec.ts`
2. Parameters use the correct types (TypedDicts for LangGraph state, named types like `OrchestrationConfig`, `DatasetResult`, etc.)
3. The return type matches the spec

## Architecture rules

**LangGraph** is used for three features. All others are plain Python.

| Feature | Pattern |
|---|---|
| Manager Agent | `StateGraph(ManagerState)` — built by `build_manager_graph()` |
| Data Generator | `StateGraph(DataGenState)` — built by `build_data_generator_graph()` |
| AutoResearch Loop | `StateGraph(AutoResearchState)` — built by `build_autoresearch_graph()`, cyclic |
| Decision Engine | Plain functions — no LangGraph |
| Cost Manager | Background thread — no LangGraph |
| Observability | Utility module — no LangGraph |
| Tinker API | Tinker SDK wrapper (`import tinker`) — no LangGraph |

**LangGraph node functions** must:
- Accept a single `state: XState` argument
- Return a `dict` with only the fields being updated (partial state update)
- Never mutate the state object directly

**LangGraph conditional edge functions** must:
- Accept `state: XState`
- Return a `Literal` string matching a node name or `"__end__"`
- Have no side effects

## Inter-agent data contracts

Every agent communicates through these typed objects — do not invent new top-level interfaces:

- `OrchestrationConfig` — emitted by Manager, read by everyone
- `DatasetResult` — emitted by Data Generator, read by Decision Engine
- `TrainingPlan` — emitted by Decision Engine, read by AutoResearch
- `TrainedModel` — emitted by AutoResearch, returned to user
- `CostBreakdown` — emitted by Cost Manager, included in TrainedModel

## Logging

No agent writes to stdout directly. Always call:
```python
log_event(AgentName.X, LogLevel.INFO, "message", metadata={})
```

## LLM calls

Use the Claude API (Anthropic SDK) for:
- `reason_node` in Manager
- `propose_node` in AutoResearch  
- `mode_c_node` in Data Generator (synthetic data generation)

Use `claude-haiku-4-5-20251001` for high-frequency calls (AutoResearch proposals, synthetic data batches). Use `claude-sonnet-4-6` for one-time reasoning calls (Manager task reasoning).

## Tinker SDK

The training backend is Thinking Machines Lab's **Tinker SDK** (`pip install tinker`), **not** a REST API.
Use the wrapper in `src/tinker_api/tinker_api.py`.

Key call sequence:
```python
svc = create_service_client()                              # reads TINKER_API_KEY
tc  = create_lora_training_client(svc, base_model, rank)
tok = get_tokenizer(tc)
dat = make_datum(input_tokens)                             # auto-shifts for causal LM
result = run_training_step(tc, batch, learning_rate, job_id)
sc  = save_checkpoint(tc, "checkpoint-name")               # returns SamplingClient
```

Cost tracking is **token-based at $0.40/million tokens** via `record_tokens` / `get_cumulative_spend`.
There is no REST billing endpoint.

## Testing rules

- Every source module must have a corresponding test in `tests/`.
- Tests must never make live network calls. Mock with `unittest.mock.patch`.
- To mock the Tinker SDK, inject a fake `tinker` module into `sys.modules` before import
  (see `tests/test_tinker_api.py` for the pattern).
- Run the full suite before any commit:
  ```
  python -m pytest tests/ --ignore=tests/test_mode_b_hf_retrieval.py -v
  ```

## Owner assignments

Each feature has a designated owner — coordinate before modifying another owner's feature:

| Feature | Owner |
|---|---|
| Manager + Cost Manager + Tinker API | Sid Potti |
| Data Generator + Decision Engine | Ron Polonsky, Angel Raychev |
| AutoResearch Loop + Evaluator | Matthew Torre, Hayley Antczak |
