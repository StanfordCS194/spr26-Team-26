# AutoResearch Loop — Punch List

*What is actually in `src/autoresearch/` today, what's broken, what's stubbed, what's missing. Prioritized by severity so you can hit blockers first.*

---

## Map of what exists

```
src/autoresearch/
├── autoresearch.py   graph + all nodes + edges + evaluator + Claude integration
├── loop.py           PROPOSE-only CLI scaffold (parallel to the graph, for testing)
├── proposer.py       algorithmic proposers (Random, LocalPerturbation) — no API
├── config.py         TrainingConfig dataclass + BOUNDS (safety bounds)
├── diff_utils.py     format_patch_as_diff / parse_diff_to_patch
└── __main__.py       CLI runner for loop.py
```

Two parallel paths live here:
- **The LangGraph graph** in `autoresearch.py` (the spec's Feature 3).
- **The CLI scaffold** in `loop.py` + `__main__.py` that only does PROPOSE. It's decoupled from the graph and uses algorithmic proposers (no API cost) for local testing.

Both use the same `TrainingConfig`, `BOUNDS`, and diff utilities. That's fine — the scaffold is a useful test harness while RUN/EVALUATE aren't real yet.

---

## Done (works as specified)

- `AutoResearchState` TypedDict (`src/types.py`). Matches spec, plus two spec-implied extras the implementation added: `current_patch`, `last_description`.
- All ancillary TypedDicts (`Hypothesis`, `EvalSuite`, `EvalScore`, `ScoreDelta`, `ExperimentResult`, `IterationRecord`, `TrainingMetrics`, `TrainingPlan`, etc.).
- Graph construction `build_autoresearch_graph()` — all 9 nodes, 3 conditional edges, compiles cleanly.
- Node skeletons: every node function exists and matches its spec signature.
- Conditional-edge skeletons: `early_stop_edge`, `decision_edge`, `continue_edge`.
- `propose_hypothesis()` — real Anthropic call, claude-haiku-4-5-20251001, JSON output, diary context, bounded search rules in the prompt.
- `check_early_stop()` — NaN/inf, exploding loss (>10), accuracy collapse (<0.01).
- `compare_scores()`, `decide_keep_or_revert()`, `flag_regression()` — correct behavior, full unit tests.
- `log_iteration()` — append-only JSONL to `outputs/logs/research_diary.jsonl`. Unit tested.
- `create_eval_suite()` — metric map for text-classification / seq2seq / custom, overrides for eval_metric, llm_grading flag on high-complexity.
- `adapt_eval_suite()` — Claude call, JSON array parsing, merge-without-duplicates.
- `apply_patch()` / `revert_patch()` — atomic JSON write via temp file. Unit-tested roundtrip.
- `continue_edge()` — checks `should_stop`, budget exhaustion, `_MAX_NO_IMPROVE`, `_MAX_ITERATIONS`. Fully unit-tested (5 scenarios).
- `TrainingConfig` dataclass with `BOUNDS`-validated `apply_patch`.
- Three `ProposalStrategy` classes — Random, LocalPerturbation, Claude wrapper.
- CLI `python -m src.autoresearch` with `--strategy`, `--n-iters`, `--show-diary`, etc.
- `diff_utils.format_patch_as_diff` / `parse_diff_to_patch` (invertible on numeric values).
- Test coverage on the algorithmic/scaffold side: `test_autoresearch.py`, `test_propose_infra.py`, `test_e2e_propose.py` — 20+ tests.

---

## Critical blockers — cannot run end-to-end

### B1. `run_evals` has an empty body
`src/autoresearch/autoresearch.py:721`

```python
def run_evals(model_path: str, eval_suite: EvalSuite) -> EvalScore:
    """Runs the evaluation suite against the model and returns a scalar score + per-metric breakdown."""
    # Wire-in point for the model inference layer (transformers / PEFT / Tinker artifacts).
    # ... long comment describing what to do ...
    # Blocked on: transformers model loading + Tinker artifact download (F4).
    # <— no return statement —>
```

Both `baseline_node` and `evaluate_node` call this. It currently returns `None` which will blow up on `last_score["scalar"]`. **Nothing can score. The graph cannot produce any EvalScore until this is written.**

To unblock: implement it for your primary task type first. For a pretraining run against the reference repo's `train.py`, this can be a 15-line function that reads `val_bpb` out of the Tinker job's `metrics.json` (or the run log) and wraps it as `EvalScore(scalar=-val_bpb, metrics={"val_bpb": val_bpb}, critique="")`.

### B2. No LangGraph checkpointer
`autoresearch.py:103` — `return graph.compile()` with no checkpointer.

Spec (line 747): "Pass a LangGraph checkpointer (e.g. SqliteSaver) to `graph.compile()` so the loop can be resumed after crashes." The whole reason Feature 3 is LangGraph and not a while-loop is resumability — without the checkpointer, you have the same crash behavior as a plain loop but with a LangGraph dependency on top.

Fix: `return graph.compile(checkpointer=SqliteSaver.from_conn_string("autoresearch_checkpoints.db"))` and thread a `thread_id` through `graph.invoke(..., config={"configurable": {"thread_id": ...}})`.

### B3. Tinker API is entirely NotImplementedError (Sid's feature)
`src/tinker_api/tinker_api.py` — every function raises `NotImplementedError`. `submit_experiment` and `wait_for_experiment` in `autoresearch.py` depend on `submit_job`, `get_job_status`, `get_cumulative_spend`.

Not your code to write, but `run_node` and `baseline_node` do nothing useful until Sid lands the Tinker wrapper. In the meantime you can stub these in tests.

### B4. Cost Manager is entirely NotImplementedError (Sid's feature)
`src/cost_manager/cost_manager.py` — every function raises `NotImplementedError`. `invoke_autoresearch_graph` takes a `cost_manager` parameter but doesn't use it.

Same situation as B3 — not yours, but you can't hit the "budget exhausted" stop condition for real yet.

---

## Real bugs — will silently produce wrong behavior

### G1. `evaluate_node` compares against `baseline_score`, not `best_score`
`autoresearch.py:341`

```python
last_delta = compare_scores(last_score, state["baseline_score"])
```

Should be `state["best_score"]`. As written, once the loop finds any improvement over baseline, **every subsequent run that beats baseline but is worse than the current best still gets KEEP**. `best_score` will walk downward over time.

### G2. `revert_and_continue_node` reverts the wrong file
`autoresearch.py:302`

```python
revert_patch(state["plan"]["training_script_path"], state["original_content"])
```

But `propose_node` (line 254) patched `_CONFIG_PATH = configs/current.json`, not the training script:

```python
original_content = apply_patch(str(_CONFIG_PATH), hypothesis["patch"])
```

So on catastrophic failure the revert writes the config JSON's contents to the **training script path**. That either corrupts `train.py` (if the path is real) or fails silently. Either way, the JSON config is never actually reverted.

Same bug could hit `revert_node` if the paths diverge — that one correctly reverts `_CONFIG_PATH` (line 393), but it relies on `original_content` being the JSON contents, which is correct. So `revert_node` is fine; `revert_and_continue_node` is broken.

### G3. `log_node`'s diff for KEPT iterations is empty/wrong
`autoresearch.py:432-433`

```python
patch_dict = json.loads(state.get("current_patch", "{}"))
diff = _patch_to_diff(patch_dict, state["current_config"])
```

`_patch_to_diff` reads `old_val = current_config.get(key)` and emits `- key: old_val / + key: new_val`.

But the graph edges are `keep → log`, and `keep_node` (line 380-388) has already updated `current_config` with the patch. By the time `log_node` runs for a KEPT iteration, `current_config[key] == new_val`, so the diff prints:

```
- learning_rate: 0.00015     <-- says "was 0.00015"
+ learning_rate: 0.00015     <-- now "is 0.00015"
```

All diary entries for KEPT iterations will show old == new. Fix by either (a) moving the config-update into `log_node` after the diff is computed, or (b) snapshotting `current_config` in `propose_node` as e.g. `pre_patch_config` and using that for the diff in `log_node`.

### G4. `invoke_autoresearch_graph` returns the script path as `weights_path`
`autoresearch.py:148`

```python
return {
    "weights_path": final_state["best_script"],  # <-- this is a .py path, not weights
    ...
}
```

`best_script` is the training script path (`plan.training_script_path`). `TrainedModel.weights_path` is supposed to point at the model checkpoint. Should come from `last_result.model_path` on the last KEPT iteration. Either store `best_model_path` alongside `best_script` in state, or pull it from the diary.

### G5. `cost_manager` parameter is ignored
`autoresearch.py:109-110, 139-145`

```python
def invoke_autoresearch_graph(plan, config, cost_manager) -> TrainedModel:
    ...
    cost_breakdown: CostBreakdown = {
        "data_gen_usd": 0.0,
        "training_usd": total_cost,        # summed from diary
        "llm_calls_usd": 0.0,
        ...
    }
```

The `cost_manager` argument is never touched. The real CostBreakdown should come from `cost_manager.generate_cost_report()`. Also, no node calls `start_cost_monitor(job_id)` when submitting to Tinker — the spec says every mini-run must be registered (Feature 4 architecture block, line 999).

### G6. `apply_patch` for `.py` files raises NotImplementedError
`autoresearch.py:611-620`

```python
elif path.suffix == ".py":
    raise NotImplementedError(
        "Script-level patching not yet implemented. "
        "Requires RUN phase and Tinker job submission."
    )
```

The spec and the reference repo both assume patches edit `train.py` via unified diff (see spec line 846-855 — patch is a unified-diff string). The implementation pivoted to a simpler JSON config approach. That's a reasonable V1 **but the design drift is not reflected in the spec**. You need to either:

(a) implement .py patching (subprocess to `patch`, or `unidiff.PatchSet`, or regex-replace one line) and change `propose_hypothesis`'s prompt to emit unified diffs, or

(b) formally switch the spec to "config-level patches only" and document that train.py is fixed per-iteration — which means Claude can only tune hyperparameters, not architecture.

Option (b) is a legitimate V1 scope cut. Just write it down.

---

## Design drift from spec

### D1. `init_node` fabricates a DatasetResult
`autoresearch.py:170-185` — builds a `DatasetResult` with `train_size=0`, `val_size=0`, `test_size=0` and a made-up path (`script_dir`). The real one should come from DataGen (Feature 2). Either:
- Add `dataset_result: DatasetResult` as an argument to `invoke_autoresearch_graph` and thread it through state, or
- Add it to `TrainingPlan` so it rides along with the plan.

`create_eval_suite` uses `dataset["dataset"]["path"]` to set `test_split_path`, so the current value is `<script_dir>/test` — likely nonexistent.

### D2. `current_config` is a plain dict in state, but `TrainingConfig` is a dataclass
`types.py:275` declares `current_config: dict`, and `autoresearch.py` uses `dict.update` patterns. But `config.py` has `TrainingConfig` with validated `BOUNDS`. The two never meet — the graph path bypasses `TrainingConfig` entirely.

Consequences: the graph can write any value into `current_config` without bounds validation. Claude can propose `learning_rate=100` and no one catches it. The algorithmic proposers in `proposer.py` DO validate via `TrainingConfig.apply_patch`, but the Claude path in `propose_hypothesis` doesn't.

Fix: in `propose_node`, after Claude returns the patch, validate it with `TrainingConfig.from_dict(current_config).apply_patch(patch)` — raise or retry if invalid.

### D3. `adapt_eval_suite` unconditionally calls Claude
`autoresearch.py:807` starts with `client = anthropic.Anthropic()`. If `ANTHROPIC_API_KEY` isn't set, this raises. `log_node` calls `adapt_eval_suite` every 10 iterations whenever there were any recent REVERTED entries — which is essentially guaranteed.

Fix: guard with an env-var check and fall back to the unchanged suite on missing key.

### D4. No idempotence guards
`run_node` re-submits a Tinker job every time it's entered. On checkpoint-resume, LangGraph will re-run the in-flight node. If the first job is still running, you now have **two jobs** billing concurrently.

Minimum fix: stash `pending_job_id` in state at submit time; on entry, if `pending_job_id` is set and still RUNNING, skip submit and go straight to `wait_for_experiment`. Clear after completion.

Same pattern applies to `baseline_node`.

### D5. `log_node` dedupe is fragile
`autoresearch.py:414-445` — checks `any(r["iteration"] == iteration_number for r in state["diary"])` to detect the "revert_and_continue already wrote it" case. Works, but it's cleaner to just skip `log_node` on the early-stop path (add an edge from `revert_and_continue` straight to `propose` OR to a special "post-log" marker).

Spec (line 676, 713) shows `revert_and_continue` looping *directly back to propose*, skipping `log_node` entirely. The implementation (line 96) adds `revert_and_continue → log`. Dedupe is the compensation. Either:
- Match the spec: remove line 96 and have `revert_and_continue` go straight to `propose` (but then `continue_edge` also needs to be reachable — move the stop-check into `revert_and_continue` itself or duplicate it).
- Keep the current shape and document it.

### D6. Hard-coded limits
`_MAX_NO_IMPROVE = 3` and `_MAX_ITERATIONS = 20` at the top of `autoresearch.py`. Should come from `OrchestrationConfig` or at least be module-level constants with a reason.

`_MAX_NO_IMPROVE = 3` is aggressive for hyperparameter search — you'll exit after 3 bad guesses in a row, which is normal noise. Consider 10-15.

---

## Test coverage gaps

What HAS tests (good):
- `apply_patch` / `revert_patch` roundtrip
- `check_early_stop` — 4 scenarios
- `compare_scores` — 3 scenarios
- `decide_keep_or_revert` — 3 scenarios
- `flag_regression` — 3 scenarios
- `log_iteration` — 2 scenarios
- `continue_edge` — 5 scenarios
- `build_autoresearch_graph` — smoke (compiles)
- `create_eval_suite` — 3 scenarios
- Algorithmic proposers — in-bounds, one-param-at-a-time, history-aware
- Claude `propose_hypothesis` — integration-marked (skipped without API key)

What has NO tests:
- `init_node`
- `baseline_node`
- `propose_node`
- `run_node`
- `revert_and_continue_node`
- `evaluate_node`
- `keep_node`
- `revert_node`
- `log_node`
- `early_stop_edge`
- `decision_edge`
- `adapt_eval_suite`
- Full-graph integration: invoke the compiled graph end-to-end with stubbed Tinker and stubbed Claude, assert the diary has the expected KEPT/REVERTED pattern.

The integration test is the big one. Once `run_evals` has a body and Tinker is stubbed, you can write:

```python
def test_full_loop_with_fakes(monkeypatch, tmp_path):
    # Stub Tinker: return canned ExperimentResults from a scripted sequence.
    # Stub Claude: return canned Hypotheses from a scripted sequence.
    # Run invoke_autoresearch_graph with _MAX_ITERATIONS=5.
    # Assert: diary length == 5, at least one KEPT, best_score improved, no crashes.
```

---

## Prioritized fix order

Do them top-to-bottom — each unblocks the next.

1. **G1** — one-line fix (`best_score` not `baseline_score`). High-severity correctness bug.
2. **G2** — one-line fix (revert the correct path in `revert_and_continue_node`).
3. **G3** — refactor where `current_config` is updated, or snapshot pre-patch in propose.
4. **D3** — guard Claude call in `adapt_eval_suite` (prevents crash every 10 iters).
5. **B1** — implement `run_evals` for your primary task type. This alone unblocks the full loop running against stubbed Tinker.
6. **B2** — add `SqliteSaver` to `graph.compile()`.
7. **D1** — plumb real `DatasetResult` through `invoke_autoresearch_graph`.
8. **G4** — track `best_model_path` in state and return it as `weights_path`.
9. **G5** — thread `cost_manager` into nodes; register jobs; pull CostBreakdown from it.
10. **D4** — idempotence guards for `run_node` / `baseline_node`.
11. **G6** — resolve the spec-vs-implementation design drift on `.py` patching. Document either way.
12. **D2** — validate Claude's patches through `TrainingConfig.apply_patch` before accepting.
13. **D5** — clean up `revert_and_continue` edge to match spec shape or document deviation.
14. **D6** — surface `_MAX_NO_IMPROVE` / `_MAX_ITERATIONS` into config.
15. **Tests** — full node unit tests + one integration test with stubs.

Items 1-4 are tiny. You can do them in an afternoon. 5-6 are the real work. 7-15 are the polish that makes this actually ship.

---

## Dependencies on other features

These are not your code but your loop blocks on them:

| Need | Status | Owner |
|---|---|---|
| `submit_job` / `get_job_status` / `get_cumulative_spend` | NotImplementedError | Sid |
| `start_cost_monitor` / `poll_spend` / `kill_job` | NotImplementedError | Sid |
| Real `DatasetResult` piped through | Unknown — check DataGen | Ron / Angel |
| Real `TrainingPlan.training_script_path` | Unknown — check Decision Engine | Ron / Angel |
| `log_event` / observability | Implemented (see `src/observability`) | — |

Until Tinker is real, write a `FakeTinker` module in `tests/fakes/` that:
- `submit_job` writes `{"metrics": {...}, "model_path": "..."}` to `outputs/experiments/<fake_id>/metrics.json` and returns a UUID.
- `get_job_status` returns `COMPLETED` after one call.
- `get_cumulative_spend` returns a small canned cost (e.g. `0.05`).

Monkeypatch `src.tinker_api.tinker_api` to the fake in your integration test. That lets you exercise the whole graph today.
