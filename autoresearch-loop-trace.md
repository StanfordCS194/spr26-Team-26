# AutoResearch Loop — Execution Trace

*Companion to `autoresearch-deep-dive.md`. This one is laser-focused on the LOOP: exact execution order, state transitions, all four branches traced with concrete example values, and pseudocode for every node.*

---

## 0. Mental model — the loop in one paragraph

The AutoResearch loop takes a starting `train.py` (from the Decision Engine) and, for some number of iterations bounded by budget, repeatedly: (1) asks Claude to propose a single code diff that might improve the training, (2) applies the diff, (3) submits a short training run to Tinker, (4) evaluates the resulting model, (5) compares it to the current best, (6) keeps the diff if it helped or reverts the file if it didn't, (7) writes what happened to a JSONL diary, and (8) decides whether to loop again. The whole thing is a LangGraph StateGraph so that if the Python process crashes mid-iteration, it resumes from the last completed node via the checkpointer. When the loop stops, `invoke_autoresearch_graph` returns a `TrainedModel` pointing at the best script + its weights + the full diary.

---

## 1. The call site — how the loop gets kicked off

The Manager's `orchestrate_node` runs this (pseudocode):

```python
# inside Manager's orchestrate_node
dataset_result = invoke_data_generator_graph(config)
plan = run_decision_engine(config, dataset_result)
trained_model = invoke_autoresearch_graph(plan, config, cost_manager)
return { "result": trained_model }
```

`invoke_autoresearch_graph` is the ONE function the rest of the system calls:

```python
def invoke_autoresearch_graph(
    plan: TrainingPlan,
    config: OrchestrationConfig,
    cost_manager: CostManager,
) -> TrainedModel:
    graph = build_autoresearch_graph()          # compiled once, cached module-global

    initial_state: AutoResearchState = {
        "plan": plan,
        "config": config,
        # everything else is filled by init_node / baseline_node
        "eval_suite": None,
        "current_script": None,
        "current_config": None,
        "original_content": None,
        "diary": [],
        "baseline_score": None,
        "best_score": None,
        "best_script": None,
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
    }

    # The cost_manager is NOT stored in state (it's a live threading.Thread object
    # that the checkpointer can't serialize). Access it via a module-level registry
    # or closure.
    _register_cost_manager(cost_manager)

    final_state = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": f"autoresearch-{plan.training_script_path}"}},
    )

    return TrainedModel(
        weights_path=_get_weights_path(final_state["best_script"]),
        metrics=final_state["best_score"],
        cost=cost_manager.generate_cost_report(),
        n_iterations=final_state["iteration"],
        research_diary_path=_diary_path(final_state),
    )
```

The `thread_id` matters — that's what the checkpointer keys on for resumption.

---

## 2. Graph construction

`build_autoresearch_graph` is called once at module import:

```python
def build_autoresearch_graph() -> CompiledStateGraph[AutoResearchState]:
    g = StateGraph(AutoResearchState)

    # Nodes
    g.add_node("init",                  init_node)
    g.add_node("baseline",              baseline_node)
    g.add_node("propose",               propose_node)
    g.add_node("run",                   run_node)
    g.add_node("revert_and_continue",   revert_and_continue_node)
    g.add_node("evaluate",              evaluate_node)
    g.add_node("keep",                  keep_node)
    g.add_node("revert",                revert_node)
    g.add_node("log",                   log_node)

    # Linear edges
    g.set_entry_point("init")
    g.add_edge("init",                "baseline")
    g.add_edge("baseline",            "propose")
    g.add_edge("propose",             "run")
    g.add_edge("revert_and_continue", "propose")   # fast-path loop back
    g.add_edge("keep",                "log")
    g.add_edge("revert",              "log")

    # Conditional edges
    g.add_conditional_edges("run", early_stop_edge, {
        "evaluate":             "evaluate",
        "revert_and_continue":  "revert_and_continue",
    })
    g.add_conditional_edges("evaluate", decision_edge, {
        "keep":   "keep",
        "revert": "revert",
    })
    g.add_conditional_edges("log", continue_edge, {
        "propose":  "propose",
        "__end__":  END,
    })

    checkpointer = SqliteSaver.from_conn_string("autoresearch_checkpoints.db")
    return g.compile(checkpointer=checkpointer)
```

That's the whole control flow. Three `add_conditional_edges` calls is why this is LangGraph and not a while-loop.

---

## 3. State at START

Let's trace a concrete run. Pretend the Decision Engine just handed us:

```python
plan = TrainingPlan(
    strategy="pre-train",
    base_model=None,
    lora_config=None,
    estimated_cost=4.50,
    estimated_time_min=120,
    training_script_path="/workdir/run_abc/train.py",
    eval_metric="val_bpb",
)

config = OrchestrationConfig(
    data=False,
    prompt="Pretrain a small LM on general web text",
    compute_budget=50.00,
    training_procedure=TrainingProcedure(
        task_type="language-modeling",
        data_format="parquet",
        training_type="pre-train",
        base_model=None,
        hyperparameters={"depth": 8, "total_batch_size": 2**19, ...},
        notes="...",
    ),
)
```

`/workdir/run_abc/train.py` is a literal copy of the reference repo's `train.py` at this point. That's important — the loop works by editing this file.

Initial state entering `init_node`:

```
plan = <the TrainingPlan above>
config = <the OrchestrationConfig above>
eval_suite = None
current_script = None
current_config = None
original_content = None
diary = []
baseline_score = None
best_score = None
best_script = None
last_result = None
last_score = None
last_delta = None
iteration = 0
no_improve_streak = 0
should_stop = False
```

---

## 4. `init_node` — first node after START

```python
def init_node(state: AutoResearchState) -> dict:
    log_event(AgentName.AUTORESEARCH, LogLevel.INFO, "init", metadata={
        "script": state["plan"].training_script_path,
    })
    # Task and dataset come from the pipeline; pull them from plan/config as needed.
    task = _task_analysis_from(state["config"])
    dataset = _dataset_from(state["plan"])
    eval_suite = create_eval_suite(task, dataset)

    return {
        "eval_suite":      eval_suite,
        "current_script":  state["plan"].training_script_path,
        "current_config":  dict(state["config"].training_procedure.hyperparameters),
        "iteration":       0,
    }
```

**State after `init_node`:**
```
eval_suite = EvalSuite(
    primary_metric="val_bpb",
    metrics=["val_bpb"],
    test_split_path="/workdir/run_abc/val_shard.parquet",
    use_llm_grading=False,
)
current_script = "/workdir/run_abc/train.py"
current_config = {"depth": 8, "total_batch_size": 524288, ...}
iteration = 0
# other fields unchanged
```

Note: returning a partial dict is a LangGraph convention. **Never** mutate `state` directly. LangGraph merges the returned dict into the state for you.

---

## 5. `baseline_node` — run the unmodified script once

```python
def baseline_node(state: AutoResearchState) -> dict:
    log_event(AgentName.AUTORESEARCH, LogLevel.INFO, "baseline: starting", metadata={})

    job_id = submit_experiment(state["current_script"], state["plan"], timeout_min=5)
    cost_manager = _get_cost_manager()
    cost_manager.start_cost_monitor(job_id, budget=_mini_run_budget(state["config"]))

    result = wait_for_experiment(job_id, timeout_min=5)   # ExperimentResult
    score = run_evals(result.model_path, state["eval_suite"])  # EvalScore

    log_event(AgentName.AUTORESEARCH, LogLevel.INFO, "baseline: done", metadata={
        "scalar": score.scalar, "cost_usd": result.cost_usd,
    })

    return {
        "baseline_score": score,
        "best_score":     score,
        "best_script":    state["current_script"],
    }
```

**State after `baseline_node`** (imagine the reference repo's baseline scored 0.9979 BPB — remember *lower* BPB is better, but `EvalScore.scalar` is defined as "higher = better" in the spec, so we flip signs):

```
baseline_score = EvalScore(
    scalar=-0.9979,                    # -val_bpb so higher is better
    metrics={"val_bpb": 0.9979},
    critique="Baseline. No changes yet.",
)
best_score = <same>
best_script = "/workdir/run_abc/train.py"
iteration = 0
```

This is the equivalent of "the first run should always be to establish the baseline" from `program.md` in the reference repo.

---

## 6. Iteration 1 — happy path (a KEPT improvement)

### 6.1 `propose_node`

```python
def propose_node(state: AutoResearchState) -> dict:
    task = _task_analysis_from(state["config"])

    hypothesis: Hypothesis = propose_hypothesis(
        current_config=state["current_config"],
        diary=state["diary"],
        task=task,
    )
    # hypothesis = Hypothesis(
    #   description="Increase matrix_lr from 0.04 to 0.05",
    #   patch="--- a/train.py\n+++ b/train.py\n@@\n-MATRIX_LR = 0.04\n+MATRIX_LR = 0.05\n",
    #   expected_effect="Small BPB improvement; may slightly increase instability.",
    #   search_strategy="local",
    # )

    original_content = apply_patch(state["current_script"], hypothesis.patch)
    # apply_patch reads the file, saves its content, then writes the patched version.

    log_event(AgentName.AUTORESEARCH, LogLevel.INFO, "proposed", metadata={
        "iteration": state["iteration"] + 1,
        "description": hypothesis.description,
        "strategy": hypothesis.search_strategy,
    })

    return {
        "original_content": original_content,
        "last_hypothesis":  hypothesis,  # convenience; used by log_node and propose_hypothesis next round
        # current_script path unchanged; file on disk is now patched
    }
```

Claude call. Uses `claude-haiku-4-5-20251001` per CLAUDE.md. The prompt should include the full diary (past hypotheses + outcomes) so Claude doesn't re-propose things that didn't work.

**State after `propose_node`** (file on disk is now patched):
```
original_content = "<entire pre-patch contents of train.py>"
# current_script path unchanged
# last_hypothesis populated
```

### 6.2 `run_node`

```python
def run_node(state: AutoResearchState) -> dict:
    job_id = submit_experiment(state["current_script"], state["plan"], timeout_min=5)
    cost_manager = _get_cost_manager()
    cost_manager.start_cost_monitor(job_id, budget=_mini_run_budget(state["config"]))

    result = wait_for_experiment(job_id, timeout_min=5)
    return { "last_result": result }
```

Returns an `ExperimentResult`:

```python
last_result = ExperimentResult(
    job_id="tinker-xyz-42",
    status="completed",
    metrics=TrainingMetrics(
        train_loss=2.51, val_loss=2.49,
        test_loss=None,
        primary_metric=0.9961,   # val_bpb
        # ...
    ),
    model_path="/tinker/jobs/tinker-xyz-42/final.pt",
    cost_usd=0.11,
    logs_path="/tinker/jobs/tinker-xyz-42/run.log",
)
```

### 6.3 `early_stop_edge` — catastrophic check

```python
def early_stop_edge(state: AutoResearchState) -> Literal["evaluate", "revert_and_continue"]:
    return "revert_and_continue" if check_early_stop(state["last_result"].metrics) else "evaluate"

def check_early_stop(metrics: TrainingMetrics) -> bool:
    if math.isnan(metrics.train_loss) or math.isnan(metrics.val_loss):
        return True
    if metrics.train_loss > 10 * _baseline_train_loss():
        return True
    if metrics.primary_metric < _chance_level():     # for classification; for BPB, use loss bound
        return True
    return False
```

In iteration 1 the metrics look fine (`val_bpb=0.9961`), so this returns `"evaluate"`.

### 6.4 `evaluate_node`

```python
def evaluate_node(state: AutoResearchState) -> dict:
    score: EvalScore = run_evals(state["last_result"].model_path, state["eval_suite"])
    # score.scalar = -0.9961 (higher is better — lower BPB)

    delta: ScoreDelta = compare_scores(score, state["best_score"])
    # delta.absolute = -0.9961 - (-0.9979) = 0.0018
    # delta.relative_pct = 0.0018 / 0.9979 ≈ 0.18%
    # delta.improved = True

    # flag_regression is separate — it catches big drops even when decide_keep_or_revert says keep
    if flag_regression(delta, threshold=-0.01):
        log_event(AgentName.AUTORESEARCH, LogLevel.WARN, "regression flagged", metadata={...})

    return { "last_score": score, "last_delta": delta }
```

**State after `evaluate_node`:**
```
last_score = EvalScore(scalar=-0.9961, metrics={"val_bpb": 0.9961}, critique="...")
last_delta = ScoreDelta(absolute=0.0018, relative_pct=0.1803, improved=True)
```

### 6.5 `decision_edge`

```python
def decision_edge(state: AutoResearchState) -> Literal["keep", "revert"]:
    return "keep" if decide_keep_or_revert(state["last_delta"]) == "KEEP" else "revert"

def decide_keep_or_revert(delta: ScoreDelta) -> Literal["KEEP", "REVERT"]:
    # Ties go to REVERT (simplicity criterion from reference program.md)
    if delta.improved and delta.relative_pct > 0:
        return "KEEP"
    return "REVERT"
```

Here `delta.improved=True` → `"keep"`.

### 6.6 `keep_node`

```python
def keep_node(state: AutoResearchState) -> dict:
    # The file on disk already contains the good patch. We just update best pointers.
    return {
        "best_score":        state["last_score"],
        "best_script":       state["current_script"],
        "no_improve_streak": 0,
    }
```

**State after `keep_node`:**
```
best_score = EvalScore(scalar=-0.9961, ...)
best_script = "/workdir/run_abc/train.py"     # same path, but the file has the good patch applied
no_improve_streak = 0
```

### 6.7 `log_node`

```python
def log_node(state: AutoResearchState) -> dict:
    new_iter = state["iteration"] + 1

    record = IterationRecord(
        iteration=new_iter,
        hypothesis=state["last_hypothesis"].description,
        patch=state["last_hypothesis"].patch,
        cost_usd=state["last_result"].cost_usd,
        metrics=state["last_result"].metrics,
        decision="KEPT",                       # or "REVERTED" — determined by no_improve_streak reset pattern
        notes=f"delta={state['last_delta'].relative_pct:.2%}",
    )
    new_diary = log_iteration(state["diary"], record)  # appends, writes JSONL

    eval_suite = state["eval_suite"]
    if new_iter % 10 == 0:
        weaknesses = _extract_weaknesses(state["diary"])
        eval_suite = adapt_eval_suite(eval_suite, weaknesses)

    return {
        "diary":      new_diary,
        "eval_suite": eval_suite,
        "iteration":  new_iter,
    }
```

**State after `log_node`:**
```
diary = [IterationRecord(iteration=1, decision="KEPT", ...)]
iteration = 1
# eval_suite unchanged (not a multiple of 10)
```

The diary is an append-only JSONL file on disk. Path convention: `/workdir/run_abc/research_diary.jsonl`.

### 6.8 `continue_edge`

```python
def continue_edge(state: AutoResearchState) -> Literal["propose", "__end__"]:
    cost_manager = _get_cost_manager()
    budget_exhausted = cost_manager.is_budget_exhausted()
    plateaued = state["no_improve_streak"] >= _NO_IMPROVE_LIMIT   # e.g. 15
    target_hit  = _target_metric_reached(state["best_score"], state["config"])

    if budget_exhausted or plateaued or target_hit:
        return "__end__"
    return "propose"
```

Returns `"propose"` → loops back to iteration 2.

---

## 7. Iteration 2 — regression (a REVERTED hypothesis)

Now imagine Claude proposes: `"Switch MLP from ReLU² to GELU"`.

- `propose_node` applies the patch. `original_content` snapshots the current (good!) file.
- `run_node` runs Tinker. `val_bpb=1.0050` (worse than our best 0.9961).
- `early_stop_edge` sees normal metrics → `"evaluate"`.
- `evaluate_node`: `delta.absolute = -1.0050 - (-0.9961) = -0.0089`, `delta.improved = False`.
- `decision_edge` → `"revert"`.
- `revert_node`:

```python
def revert_node(state: AutoResearchState) -> dict:
    revert_patch(state["current_script"], state["original_content"])
    return { "no_improve_streak": state["no_improve_streak"] + 1 }
```

The file on disk is now restored to the state it had at the START of this iteration (i.e. the good iteration-1 version).

- `log_node` writes `IterationRecord(iteration=2, decision="REVERTED", ...)`.
- `continue_edge` → `"propose"`.

**Invariant:** after `revert_node`, `current_script` points at a file whose contents equal `best_script`'s contents. This is critical: every iteration starts from best.

---

## 8. Iteration 3 — catastrophic failure (REVERT-AND-CONTINUE)

Claude proposes `"Halve weight decay to 0.0"`.

- `propose_node` applies. `original_content` snapshotted.
- `run_node` runs. Something goes wrong — loss explodes. Either (a) the job finishes with NaN metrics, or (b) the job returns early because `train.py` itself hit the `if math.isnan(loss) or loss > 100: exit(1)` line.
- `early_stop_edge` sees `metrics.train_loss > 10 * baseline` → returns `"revert_and_continue"`.
- `revert_and_continue_node`:

```python
def revert_and_continue_node(state: AutoResearchState) -> dict:
    revert_patch(state["current_script"], state["original_content"])

    record = IterationRecord(
        iteration=state["iteration"] + 1,
        hypothesis=state["last_hypothesis"].description,
        patch=state["last_hypothesis"].patch,
        cost_usd=state["last_result"].cost_usd,
        metrics=state["last_result"].metrics,
        decision="REVERTED",
        notes="catastrophic failure — early stop",
    )
    new_diary = log_iteration(state["diary"], record)

    return {
        "diary":             new_diary,
        "iteration":         state["iteration"] + 1,
        "no_improve_streak": state["no_improve_streak"] + 1,
    }
```

- This node loops DIRECTLY back to `propose_node`. It does NOT pass through `evaluate → keep/revert → log`. The whole point of this fast path is to skip the evaluator when the run is garbage — evaluating a NaN model is pointless.

Note the duplication: this node has to replicate the logging and iter-bump that `log_node` would've done. Keep that in mind when implementing — the easiest bug to write is logging twice or not incrementing the counter.

---

## 9. Termination — how the loop ends and what gets returned

`continue_edge` returns `"__end__"` when any of:

1. `cost_manager.is_budget_exhausted()` → True (Cost Manager decides this by polling Tinker billing).
2. `state["no_improve_streak"] >= N` → conventional stop, say N=15 (spec leaves this unpinned — pick something and justify it).
3. Target metric reached (rarely used for BPB; more relevant for classification accuracy targets).

When the graph reaches `END`, `graph.invoke()` returns the final state. `invoke_autoresearch_graph` then packs it into a `TrainedModel`:

```python
return TrainedModel(
    weights_path=_weights_for(final_state["best_script"]),
    metrics=final_state["best_score"],
    cost=cost_manager.generate_cost_report(),
    n_iterations=final_state["iteration"],
    research_diary_path="/workdir/run_abc/research_diary.jsonl",
)
```

That's what the Manager ultimately hands back to the user.

---

## 10. Checkpointing behavior — what happens if the process dies

LangGraph checkpoints after every node completes. State is serialized to SQLite via `SqliteSaver`. If the Python process is killed mid-iteration:

- **During `propose_node`:** the file might be patched but state not yet committed. On resume, LangGraph re-runs `propose_node` from the last checkpoint (which was `log_node` of the previous iter). Claude generates a fresh hypothesis. No real harm. **Caveat:** you must make `propose_node` idempotent — calling it twice shouldn't leave the file double-patched. The simplest way: always read `best_script` from disk at the start of `propose_node`, never trust that the on-disk file matches the in-state pointer.
- **During `run_node`:** the Tinker job keeps running server-side. On resume, `run_node` re-submits a new job, wasting the first one's cost. This is the main source of cost leakage from crashes. Mitigation: store `job_id` in state at submit time and, on resume, call `wait_for_experiment` on the existing job instead of submitting a new one.
- **During `evaluate_node` / `keep_node` / `revert_node`:** model_path is already on disk; just re-run.
- **During `log_node`:** diary file already has the record (if it was written before the crash). Make `log_iteration` idempotent by checking if a record for this iteration already exists before appending.

Rule of thumb: **every node must be safe to run twice.** Treat each node like an HTTP handler. Pure functions of state + deterministic side effects are easy. Non-idempotent side effects (appending to files, submitting jobs) need a guard.

---

## 11. A full annotated transition table

Here's every possible path through one iteration, shown as state diffs. "same" means the field is unchanged from before the node.

| Node | eval_suite | current_script (file) | original_content | last_result | last_score | last_delta | best_score | best_script | iteration | no_improve_streak | diary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **enter propose** | set | best contents | None | stale | stale | stale | set | set | N-1 | k | N-1 entries |
| **after propose** | same | **patched** | **snapshot** | same | same | same | same | same | same | same | same |
| **after run** | same | same | same | **new** | same | same | same | same | same | same | same |
| *early_stop = revert_and_continue* | | | | | | | | | | | |
| **after revert_and_continue** | same | **reverted** | same | same | stale | stale | same | same | **N** | **k+1** | **N entries** |
| *early_stop = evaluate* | | | | | | | | | | | |
| **after evaluate** | same | same | same | same | **new** | **new** | same | same | same | same | same |
| *decision = keep* | | | | | | | | | | | |
| **after keep** | same | same (good) | same | same | same | same | **new** | **new** | same | **0** | same |
| *decision = revert* | | | | | | | | | | | |
| **after revert** | same | **reverted** | same | same | same | same | same | same | same | **k+1** | same |
| **after log** | maybe adapted | same | same | same | same | same | same | same | **N** | same | **N entries** |

Two invariants that fall out of this table and that you should assert at node boundaries:

- After any node, the file at `current_script` EITHER equals `best_script`'s contents OR equals `best_script`'s contents + one active patch.
- After any node except `propose`, `original_content` is not needed until the next `propose_node`.

---

## 12. Where the Cost Manager plugs in

The Cost Manager is a background threading.Thread — it is *not* stored in `AutoResearchState` because checkpointers can't pickle live threads. Keep a module-level registry:

```python
# autoresearch/runtime.py (or similar)
_cost_manager: CostManager | None = None

def _register_cost_manager(cm: CostManager) -> None:
    global _cost_manager
    _cost_manager = cm

def _get_cost_manager() -> CostManager:
    assert _cost_manager is not None, "Cost manager not registered"
    return _cost_manager
```

Inside the loop:

- `baseline_node` and `run_node` both register the Tinker job they submit: `cost_manager.start_cost_monitor(job_id, budget=mini_run_budget)`. "Mini-run budget" is typically a small fraction of the total (e.g. 10% per run as a soft cap).
- `continue_edge` reads `cost_manager.is_budget_exhausted()` (or equivalent) to decide whether to stop.
- After END, `invoke_autoresearch_graph` calls `cost_manager.generate_cost_report()` to get the `CostBreakdown` for the final `TrainedModel`.

The Cost Manager also kills running Tinker instances at 100% budget on its own — your loop doesn't need to poll or react, it just eventually sees `is_budget_exhausted()` go True and terminates cleanly at `continue_edge`.

---

## 13. Where Tinker plugs in

`submit_experiment` and `wait_for_experiment` are the only two Tinker touchpoints in the loop. Both live in Sid's Tinker API wrapper, not in your module:

```python
# tinker_api.py (Sid's module, called from run_node + baseline_node)
def submit_experiment(script_path: str, plan: TrainingPlan, timeout_min: int = 5) -> str: ...
def wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult: ...
```

Your job is to treat them as black boxes. For local testing, stub them with `subprocess`:

```python
# tests/fakes/tinker.py
def submit_experiment(script_path, plan, timeout_min=5) -> str:
    job_id = uuid4().hex
    # spawn a subprocess; write logs to /tmp/<job_id>/run.log
    return job_id

def wait_for_experiment(job_id, timeout_min) -> ExperimentResult:
    # poll /tmp/<job_id> for completion; parse val_bpb from the summary block
    ...
```

The reference repo's `train.py` works perfectly as the "training script" for the stub — it prints the exact summary block your fake can parse.

---

## 14. Where Claude plugs in

Exactly ONE place: `propose_hypothesis`. Sketch:

```python
def propose_hypothesis(current_config: dict, diary: ResearchDiary, task: TaskAnalysis) -> Hypothesis:
    client = anthropic.Anthropic()
    system = _PROPOSE_SYSTEM_PROMPT
    user = _build_propose_user_prompt(current_config, diary, task)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        system=system,
        max_tokens=4096,
        messages=[{"role": "user", "content": user}],
    )
    return _parse_hypothesis_from(resp.content[0].text)
```

The system prompt should:
1. Describe the task and the goal metric.
2. Explain the three search strategies (`random`, `local`, `playbook`).
3. Require the output as a strict JSON object with keys `{description, patch, expected_effect, search_strategy}`.
4. Embed the diary so Claude can see what's been tried.
5. Embed the current `train.py` file contents so Claude knows what it's patching.
6. Include an anti-bloat rule from the simplicity criterion.

Use tool-use / structured output if you can, to avoid parsing free-form JSON.

---

## 15. The Evaluator sub-feature, in loop context

You own this in addition to the loop. Where each evaluator function runs:

| Function | Called from | How often |
|---|---|---|
| `create_eval_suite(task, dataset)` | `init_node` | Once per run |
| `run_evals(model_path, eval_suite)` | `baseline_node` and `evaluate_node` | Every iteration + baseline |
| `compare_scores(new, baseline)` | `evaluate_node` | Every iteration (except early-stopped) |
| `flag_regression(delta, threshold)` | `evaluate_node` | Every iteration (except early-stopped) |
| `adapt_eval_suite(suite, weaknesses)` | `log_node` | Every 10 iterations |

For the pre-train task using the reference repo's `train.py`, `run_evals` is nearly a one-liner: read `val_bpb` out of the run log (or from the model checkpoint), wrap it as `EvalScore(scalar=-val_bpb, metrics={"val_bpb": val_bpb}, critique="")`. For classification tasks you'd compute accuracy/F1/etc. on `eval_suite.test_split_path`.

`adapt_eval_suite` is the fancy part — it's supposed to look at systematic failure modes in the diary and add targeted test cases. For v1, you can have it be a no-op and come back to it. Flag it with a `# TODO` and move on.

---

## 16. Suggested unit-test sketch

Each of these should be its own test file.

```python
# test_patch.py
def test_apply_patch_and_revert_roundtrip(tmp_path): ...
def test_apply_patch_saves_original(tmp_path): ...

# test_compare.py
def test_compare_scores_improvement(): ...
def test_compare_scores_tie_reverts(): ...
def test_flag_regression(): ...

# test_early_stop.py
def test_check_early_stop_nan(): ...
def test_check_early_stop_loss_explosion(): ...
def test_check_early_stop_normal(): ...

# test_nodes.py  (state-in, state-out unit tests on each node, Tinker stubbed)
def test_init_node_sets_eval_suite_and_zero_iteration(): ...
def test_propose_node_snapshots_original_content(): ...
def test_keep_node_resets_no_improve_streak(): ...
def test_revert_node_restores_file(): ...
def test_revert_and_continue_node_increments_iteration_once(): ...

# test_edges.py
def test_early_stop_edge_routes_on_nan(): ...
def test_decision_edge_routes_on_delta(): ...
def test_continue_edge_ends_on_budget_exhausted(): ...

# test_graph_integration.py  (full loop with Tinker stub + fake Claude)
def test_full_loop_improves_monotonically_with_always_good_hypothesis(): ...
def test_full_loop_reverts_when_hypothesis_regresses(): ...
def test_full_loop_recovers_from_catastrophic_failure(): ...
```

The integration test is the important one. Stub `propose_hypothesis` to return a sequence of scripted hypotheses (some good, some bad, some catastrophic), stub `submit_experiment`/`wait_for_experiment` to run `autoresearch/train.py` locally (or return canned metrics), and assert the diary ends up with the expected `KEPT`/`REVERTED` pattern.

---

## 17. TL;DR for your whiteboard

```
TRANSIENT:        last_result, last_score, last_delta, original_content, last_hypothesis
BEST SO FAR:      best_score, best_script
COUNTERS:         iteration, no_improve_streak
PERSISTED:        diary, eval_suite

EVERY ITERATION:
    1. propose:   ask Claude, apply patch, snapshot original
    2. run:       tinker submit, tinker wait
    3. early?     if metrics garbage → revert file, log, loop
                  else → continue
    4. evaluate:  score, delta = compare(score, best)
    5. decide:    improved → keep; else → revert file
    6. log:       append iteration record; every 10, adapt eval suite
    7. continue?  budget / plateau / target → END
                  else → back to propose

INVARIANTS:
    - file on disk always equals best OR best+one active patch
    - every node is idempotent
    - diary is append-only
    - iteration increments exactly once per trip past log OR revert_and_continue
```

That's the whole thing.
