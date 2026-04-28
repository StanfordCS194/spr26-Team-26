# AutoResearch — Deep Dive

*Reference doc for Hayley. Two things in this repo share the name "autoresearch". This doc untangles them and walks every component, in execution order, with connections to the rest of the project.*

---

## 0. The two "autoresearch"s

| | What it is | Where it lives | State |
|---|---|---|---|
| **Reference repo** | A working Karpathy-style research loop. A human-driven coding agent (Claude/Codex) edits `train.py`, runs 5-min GPT training, logs results, iterates. | `autoresearch/` (train.py, prepare.py, program.md) | Works today |
| **AutoResearch Loop feature** | A LangGraph StateGraph that **automates** that same loop as part of your team's AutoTrain Agent. You and Matthew own it. | `spec-site/content/spec.ts` (Feature 3) | Spec only, no Python yet |

The feature is "the reference repo, turned into a graph that runs inside the larger pipeline." Understand the reference repo first, then the feature becomes much clearer.

---

## 1. The reference repo (`autoresearch/`)

### 1.1 File map

```
autoresearch/
├── prepare.py       # read-only. data download, tokenizer, dataloader, eval
├── train.py         # the ONLY file the agent edits. model + optimizer + loop
├── program.md       # the agent's instructions (the "skill")
├── pyproject.toml   # deps — cannot be changed
├── README.md        # repo context
├── analysis.ipynb   # post-hoc analysis notebook
└── progress.png     # teaser image
```

### 1.2 `prepare.py` — fixed infrastructure (do not modify)

Everything here is locked in. It defines the contract train.py runs against.

**Top-level constants (the three that matter most):**
- `MAX_SEQ_LEN = 2048` — context length used for both training and eval.
- `TIME_BUDGET = 300` — 5 minutes of wall-clock training time. `train.py` must respect this.
- `EVAL_TOKENS = 40 * 524288` — ~21M tokens used for validation.

**What it does, in order, when run as `uv run prepare.py`:**
1. `download_data(num_shards)` — pulls parquet shards from HuggingFace (`karpathy/climbmix-400b-shuffle`) into `~/.cache/autoresearch/data/`. Shard 6542 is pinned as val. Uses a `multiprocessing.Pool` of 8 workers, retries 5x with backoff.
2. `train_tokenizer()` — trains an 8192-vocab BPE using `rustbpe` on the downloaded text, wraps it as a `tiktoken.Encoding`, pickles to `tokenizer.pkl`. Also builds `token_bytes.pt`, a tensor where `token_bytes[i] = len(decode([i]).encode('utf-8'))` — this is what makes BPB evaluation vocab-size-independent.

**Runtime utilities imported by `train.py`:**
- `Tokenizer.from_directory()` — loads the pickled encoder.
- `make_dataloader(tokenizer, B, T, split)` — infinite generator that yields `(inputs, targets, epoch)` on CUDA. Packs documents into rows using **best-fit**: every row starts with BOS, and when no doc fits the remaining space, the shortest doc is cropped to fill exactly. 100% token utilization, no padding. Triple-buffers CPU→GPU with `pin_memory`.
- `evaluate_bpb(model, tokenizer, batch_size)` — the **fixed** metric. For ~21M val tokens it sums per-token cross-entropy in nats, sums target byte lengths (excluding special tokens), and returns `nats / (ln(2) * bytes)` = bits per byte. Vocab-size-independent, so architectural changes compete fairly.

### 1.3 `train.py` — the agent's playground

This is the single file the agent edits. Everything below is fair game; only the imports from `prepare.py` and the call shape to `evaluate_bpb` must stay intact.

#### 1.3.1 Execution order, top to bottom

The file runs as a script — no `main()`. Order matters.

1. **Env + imports (lines 1-26).** Enables `expandable_segments` CUDA allocator, detects GPU capability to pick between `varunneal/flash-attention-3` (Hopper H100) and `kernels-community/flash-attn3` (non-Hopper), imports `fa3.flash_attn_interface` for the attention kernel.
2. **Model definitions (lines 32-291).** `GPTConfig`, `norm`, `has_ve`, `apply_rotary_emb`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`. Classes only — no instantiation yet.
3. **Optimizer definitions (lines 297-426).** `polar_express_coeffs`, two `@torch.compile`'d fused step functions (`adamw_step_fused`, `muon_step_fused`), `MuonAdamW` class.
4. **Hyperparameter constants (lines 433-451).** Module-level knobs. **These are the first place the agent changes things.**
5. **Setup block (lines 457-514).** Seeds RNGs, loads tokenizer, builds config from `DEPTH`, constructs model on the `meta` device then `to_empty(device="cuda")` + `init_weights()` (saves VRAM during construction), computes param counts and FLOPs/token, builds optimizer, `torch.compile`s the model, constructs the dataloader, prefetches the first batch.
6. **Schedule functions (lines 518-532).** `get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay` — all computed from `progress = total_training_time / TIME_BUDGET`.
7. **Training loop (lines 538-604).** See below.
8. **Final eval + summary print (lines 610-630).** Runs `evaluate_bpb`, then prints the exact summary block the agent greps for.

#### 1.3.2 Model architecture at a glance

Not vanilla GPT. Several nanochat tricks baked in:

- **Config computed from DEPTH.** `model_dim = depth * ASPECT_RATIO` rounded up to a multiple of `HEAD_DIM`. With defaults (depth=8, aspect=64, head_dim=128) you get `n_embd=512, n_head=4`. Changing `DEPTH` proportionally changes width.
- **RMSNorm** via `F.rms_norm` (no learnable scale — `norm(x)` wraps it).
- **Rotary embeddings** precomputed once as buffers `cos` / `sin`. `rotary_seq_len = sequence_len * 10` so the buffer is reusable if the loop later changes T.
- **Grouped-Query Attention** (`n_kv_head` can be < `n_head`, defaults equal). Q and K are RMSNormed *after* rotary (QK-norm).
- **Value Embeddings (ResFormer).** Every other layer (alternating, last always included — see `has_ve`) has a per-layer `nn.Embedding(vocab_size, kv_dim)` that feeds into V through an input-dependent sigmoid gate (`ve_gate` reads the first 32 channels of x). These are called "value embeds" throughout.
- **Sliding window attention via FA3.** `WINDOW_PATTERN = "SSSL"` means layers alternate short-window (half context) / long-window (full context). The last layer is always full. Computed once in `_compute_window_sizes`.
- **Residual mixing.** Each block input is `resid_lambdas[i] * x + x0_lambdas[i] * x0`, where `x0` is the initial embedded input. Lambdas are learnable scalars (init 1.0 and 0.1).
- **MLP is ReLU², not GELU.** `F.relu(x).square()`. Cheaper and works well.
- **Zero-init "projection-out" trick.** `c_proj` in attention and MLP init to zero so early in training each block is near-identity.
- **Soft-capped logits.** `logits = 15 * tanh(logits / 15)` caps the output and stabilizes early steps.
- **BF16 embeddings.** `wte` and value_embeds cast to bf16 after init (see end of `init_weights`).

#### 1.3.3 The MuonAdamW optimizer

`setup_optimizer` builds **one** optimizer with a mix of AdamW and Muon param groups:

- **AdamW groups** (5 total):
  - `lm_head` (unembedding) — lr = 0.004 × scale
  - `wte` (embedding) — lr = 0.6 × scale (yes, much higher than matrix LR — classic trick)
  - `value_embeds` — same LR as wte
  - `resid_lambdas` — lr = 0.5 × 0.01 (tiny, they're scalars)
  - `x0_lambdas` — lr = 0.5
- **Muon groups** — one group per unique matrix `shape` in the transformer blocks. Grouping by shape lets `muon_step_fused` stack grads into a single `(num_params, *shape)` tensor and do the Newton-Schulz iteration batched.
- **LR scale.** `dmodel_lr_scale = (model_dim / 768) ** -0.5`. AdamW LRs scale by this; Muon LRs scale by `max(1, rows/cols)**0.5` (different scaling rule).

**The Muon step** (inside `muon_step_fused`):
1. Nesterov momentum update.
2. **Polar-express orthogonalization** — 5-step matrix Newton-Schulz iteration using `polar_express_coeffs` that produces a matrix close to `U @ V.T` (the SVD without singular values). This is what "Muon" is.
3. **NorMuon variance reduction** — a second-moment buffer tracks row-wise (or column-wise, depending on shape orientation) variance and renormalizes.
4. **Cautious weight decay + update** — only apply WD where sign(g) == sign(p), standard "cautious" trick.

Both fused step functions are `@torch.compile(fullgraph=True)`. To avoid recompilation when LRs change across steps, the optimizer keeps 0-D CPU tensors and `.fill_()`s them each step instead of passing Python floats.

#### 1.3.4 The training loop

```
while True:
    for micro_step in range(grad_accum_steps):       # gradient accumulation
        loss = model(x, y)                            # forward
        (loss / grad_accum_steps).backward()          # backward
        x, y, epoch = next(train_loader)              # prefetch next batch

    progress = total_training_time / TIME_BUDGET
    # update LR (warmup + stable + linear cooldown to FINAL_LR_FRAC)
    # update Muon momentum (0.85 → 0.95 over first 300 steps)
    # update Muon weight_decay (linearly decays to 0 as progress → 1)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    if nan(loss) or loss > 100: exit(1)               # fast fail

    if step > 10: total_training_time += dt           # skip compilation warmup
    if step == 0: gc.collect(); gc.freeze(); gc.disable()
    elif (step+1) % 5000 == 0: gc.collect()           # manual GC to avoid 500ms stalls
    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break
```

Key subtleties:
- `TOTAL_BATCH_SIZE = 2**19` (~524K tokens/step), `DEVICE_BATCH_SIZE * MAX_SEQ_LEN = 128*2048 = 262K`, so `grad_accum_steps = 2`.
- The first 10 steps don't count against the time budget — they absorb `torch.compile` warmup.
- GC is frozen after step 0 because Python GC pauses otherwise cost ~500ms at random.
- The schedules are **time-based**, not step-based, so faster hardware does more steps automatically.

#### 1.3.5 The summary print (the contract with `program.md`)

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

The agent greps this with `grep "^val_bpb:\|^peak_vram_mb:" run.log`. If grep comes back empty the run crashed.

### 1.4 `program.md` — the agent loop

The human-driven research loop is specified here. In execution order:

**Setup phase (once per run):**
1. Pick a run tag (e.g. `mar5`), create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py`, `train.py` into context.
3. Verify `~/.cache/autoresearch/` has data + tokenizer. Otherwise tell human to run `uv run prepare.py`.
4. Create `results.tsv` with just the header row: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`.
5. Confirm and go.

**Experiment loop (forever):**
1. Note current branch/commit.
2. Edit `train.py` with one experimental idea.
3. `git commit`.
4. `uv run train.py > run.log 2>&1` (redirect — don't flood context with training output).
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`. Empty → crashed, `tail -n 50 run.log` to see trace.
6. Append row to `results.tsv` (kept *untracked*).
7. If val_bpb improved → keep the commit ("advance" the branch). If equal/worse → `git reset` back.
8. 10-minute hard timeout per run; kill and treat as failure.
9. **NEVER STOP.** The loop is meant to run overnight.

The "simplicity criterion" is important: a tiny BPB gain that adds 20 lines of hacky code is not worth keeping; a zero-delta change that *removes* code is always worth keeping.

### 1.5 End-to-end flow of a single experiment (reference repo)

```
agent edits train.py      →  git commit
                          ↓
uv run train.py > run.log  →  prepare.py utilities
                          ↓       ↓
                          ↓   download_data / load tokenizer (cached, no-op after first)
                          ↓       ↓
setup model + optimizer   ←   make_dataloader
                          ↓
training loop (≈5 min)     ↓
                          ↓
evaluate_bpb(model, ...)  ←   make_dataloader(split="val") + token_bytes
                          ↓
print summary block        →  run.log
                          ↓
agent greps val_bpb        →  keep or revert commit
                          ↓
append row to results.tsv
```

---

## 2. The AutoResearch Loop feature (spec.ts, Feature 3)

This is what your team is building. It's the reference repo's agent loop, but:
- written in Python as a **LangGraph StateGraph**,
- resumable (checkpointed) so crashes don't restart from scratch,
- slotted into the larger 5-component pipeline,
- invoked by the Manager after the Decision Engine produces a `TrainingPlan`.

### 2.1 How it fits into the overall pipeline

```
User prompt + budget
        │
        ▼
Manager Agent        (LangGraph, linear, Sid)
  └─ orchestrate_node:
       │
       ├──► Data Generator   (LangGraph, Ron/Angel) ──► DatasetResult
       │
       ├──► Decision Engine  (plain Python, Ron/Angel) ──► TrainingPlan (path to train.py)
       │
       ├──► AutoResearch Loop  ◄── YOU ARE HERE ──► TrainedModel
       │
       └─ Cost Manager (background thread, Sid) runs the whole time
```

The AutoResearch entry point is `invoke_autoresearch_graph(plan, config, cost_manager) -> TrainedModel`. That signature is the contract with the Manager.

### 2.2 `AutoResearchState` (the TypedDict)

This is the single object that flows through every node. Every field, grouped by who writes it:

**Set once at the entry point** (by `invoke_autoresearch_graph`):
- `plan: TrainingPlan` — what Decision Engine produced (includes `training_script_path`, `eval_metric`, etc.).
- `config: OrchestrationConfig` — the global config dict (budget, task type, prompt).

**Set by `init_node`:**
- `eval_suite: EvalSuite` — from `create_eval_suite(task, dataset)`. Reused every iteration.
- `current_script: str` — starts as `plan.training_script_path`.
- `current_config: dict` — the live hyperparameter dict.
- `iteration: int = 0`.

**Set by `baseline_node`:**
- `baseline_score: EvalScore` — unmodified baseline.
- `best_score: EvalScore` — starts equal to baseline.
- `best_script: str` — starts equal to `current_script`.

**Set by `propose_node` every iteration:**
- `original_content: str` — file contents BEFORE the patch is applied. Needed for revert.
- `current_script` — same path, but the file on disk has now been patched.

**Set by `run_node`:**
- `last_result: ExperimentResult` — `{ job_id, status, metrics, model_path, cost_usd, logs_path }`.

**Set by `evaluate_node`:**
- `last_score: EvalScore`
- `last_delta: ScoreDelta` — `{ absolute, relative_pct, improved }`.

**Updated by `keep_node` / `revert_node` / `log_node`:**
- `best_score`, `best_script` (keep_node only)
- `no_improve_streak: int` (reset to 0 on keep, incremented on revert)
- `diary: ResearchDiary` — append-only list of `IterationRecord`s.
- `iteration` — incremented at log_node.
- `should_stop: bool` — set when continue_edge decides to END.

### 2.3 The graph (execution order, every node)

```
START
  │
  ▼
init_node ─────────► baseline_node ─────────► propose_node  ◄──────────┐
                                                   │                    │
                                                   ▼                    │
                                              run_node                  │
                                                   │                    │
                                 ┌─early_stop_edge─┴─────────────────┐  │
                                 │                                   │  │
                                 ▼  (catastrophic failure)           ▼  │
                   revert_and_continue_node                   evaluate_node
                                 │                                   │
                                 └───────────────► log_node ◄────────┤
                                                      ▲              │
                         ┌────────────────────────────┤              │
                         │                            │              │
                     keep_node                    revert_node        │
                         ▲                            ▲              │
                         │                            │              │
                         └────────decision_edge───────┘              │
                                      ▲                              │
                                      │                              │
                                      └──────────────────────────────┘
                                                      │
                                                      ▼
                                               continue_edge
                                              ┌──────┴──────┐
                                              ▼             ▼
                                          propose_node      END
                                          (loop back)    (returns best TrainedModel)
```

Three conditional edges drive all the branching:

| Edge | Reads from state | Returns |
|---|---|---|
| `early_stop_edge` | `state.last_result.metrics` | `'evaluate'` or `'revert_and_continue'` |
| `decision_edge` | `state.last_delta` | `'keep'` or `'revert'` |
| `continue_edge` | budget, `no_improve_streak`, target metric | `'propose'` or `'__end__'` |

### 2.4 Node-by-node walkthrough

**`init_node(state) -> dict`**
- Calls `create_eval_suite(task, dataset)`.
- Returns `{ eval_suite, current_script: plan.training_script_path, current_config: plan.hyperparameters_or_similar, iteration: 0 }`.

**`baseline_node(state) -> dict`**
- Submits the unmodified script to Tinker via `submit_experiment(script_path, plan, timeout_min=5)`.
- Waits with `wait_for_experiment(job_id, timeout_min)`.
- Scores with `run_evals(model_path, eval_suite)`.
- Returns `{ baseline_score, best_score: baseline_score, best_script: current_script }`.
- This corresponds to "the first run is the baseline" rule from the reference repo's `program.md`.

**`propose_node(state) -> dict`** — the Claude call
- Calls `propose_hypothesis(current_config, diary, task) -> Hypothesis`. Uses `claude-haiku-4-5-20251001` (CLAUDE.md — high-frequency call).
- A `Hypothesis` is `{ description, patch, expected_effect, search_strategy }` where `search_strategy ∈ {'random', 'local', 'playbook'}`.
- Calls `apply_patch(script_path, patch)` which:
  1. Reads the current file → saves as `original_content`.
  2. Writes the patched version to disk.
  3. Returns the original content.
- Returns `{ current_script (patched on disk), original_content, last_hypothesis }`.

**`run_node(state) -> dict`**
- `submit_experiment(state.current_script, state.plan, timeout_min=5)` → job_id.
- `wait_for_experiment(job_id, timeout_min=5)` → `ExperimentResult`.
- Returns `{ last_result }`.
- **Cost Manager is registered for this job** — each mini-run is budgeted (see `start_cost_monitor` in Cost Manager spec).

**`early_stop_edge(state) -> Literal['evaluate', 'revert_and_continue']`**
- Calls `check_early_stop(metrics)` — True on: exploding loss (>10× baseline), NaN, or accuracy below chance.
- True → `revert_and_continue`; False → `evaluate`.
- This is the equivalent of `train.py`'s `if nan or loss > 100: exit(1)` line.

**`revert_and_continue_node(state) -> dict`**
- `revert_patch(script_path, original_content)` restores the file.
- Logs a REVERTED iteration record to the diary.
- Increments `iteration`.
- Loops back to `propose_node` (NOT through log_node — this is the fast catastrophic-failure path).

**`evaluate_node(state) -> dict`**
- `run_evals(model_path, eval_suite) -> EvalScore`.
- `compare_scores(new_score, best_score) -> ScoreDelta`.
- `flag_regression(delta, threshold=-0.01)` — separate from the keep/revert decision, can trigger automatic revert on a hard regression.
- Returns `{ last_score, last_delta }`.

**`decision_edge(state) -> Literal['keep', 'revert']`**
- `decide_keep_or_revert(last_delta)`.
- `delta.improved` → `'keep'`; else → `'revert'`. Ties revert (mirrors the simplicity criterion).

**`keep_node(state) -> dict`**
- `{ best_score: last_score, best_script: current_script, no_improve_streak: 0 }`.

**`revert_node(state) -> dict`**
- `revert_patch(script_path, original_content)`.
- `{ no_improve_streak: state.no_improve_streak + 1 }`.

**`log_node(state) -> dict`**
- `log_iteration(diary, IterationRecord) -> ResearchDiary` — appends to diary and writes JSONL to disk.
- Every 10 iterations: `adapt_eval_suite(eval_suite, weaknesses)` to add harder test cases.
- Increments `iteration`.
- Returns `{ diary, eval_suite, iteration }`.

**`continue_edge(state) -> Literal['propose', '__end__']`**
- Ends if ANY of:
  - budget exhausted (CostManager says so),
  - `no_improve_streak >= N` (some threshold — spec doesn't pin it),
  - target metric reached.
- Otherwise loops to `propose_node`.

### 2.5 The "working" functions (called by the nodes)

These are the pieces that are plain functions — easier to unit-test than nodes.

**Propose phase:**
- `propose_hypothesis(current_config, diary, task) -> Hypothesis` — the Claude-backed generator. Uses three search strategies: random (over bounded ranges), local (perturbations around current best), playbook (heuristics from prior knowledge).
- `apply_patch(script_path, patch) -> str` — unified-diff application, returns original content.
- `revert_patch(script_path, original_content) -> None` — rewrites the file.

**Run phase:**
- `submit_experiment(script_path, plan, timeout_min=5) -> str` — Tinker job ID.
- `wait_for_experiment(job_id, timeout_min) -> ExperimentResult`.
- `check_early_stop(metrics) -> bool`.

**Evaluate phase:**
- `run_evals(model_path, eval_suite) -> EvalScore`.
- `compare_scores(new_score, baseline_score) -> ScoreDelta`.

**Decide phase:**
- `decide_keep_or_revert(delta) -> Literal['KEEP', 'REVERT']`.
- `log_iteration(diary, record) -> ResearchDiary`.

**Evaluator sub-feature (your other responsibility):**
- `create_eval_suite(task, dataset) -> EvalSuite` — called once in `init_node`.
- `adapt_eval_suite(suite, weaknesses) -> EvalSuite` — called every 10 iters in `log_node`.
- `flag_regression(delta, threshold=-0.01) -> bool` — called in `evaluate_node`.

### 2.6 Data contracts (the typed objects flowing in and out)

**In:**
- `TrainingPlan` (from Decision Engine): has `training_script_path`, `eval_metric`, `strategy`, `base_model`, `lora_config`, `estimated_cost`, `estimated_time_min`.
- `OrchestrationConfig` (from Manager): has `prompt`, `compute_budget`, `training_procedure` (nested).
- `CostManager` instance (already running): you register each mini-run with it.

**Out:**
- `TrainedModel`: `{ weights_path, metrics: EvalScore, cost: CostBreakdown, n_iterations, research_diary_path }`.

**Internal objects:**
- `EvalSuite { primary_metric, metrics, test_split_path, use_llm_grading }`.
- `EvalScore { scalar, metrics: dict[str,float], critique }`.
- `ScoreDelta { absolute, relative_pct, improved }`.
- `ExperimentResult { job_id, status, metrics: TrainingMetrics, model_path, cost_usd, logs_path }`.
- `Hypothesis { description, patch, expected_effect, search_strategy }`.
- `IterationRecord { iteration, hypothesis, patch, cost_usd, metrics, decision, notes }`.
- `ResearchDiary` — list of `IterationRecord`, persisted as JSONL.

### 2.7 Logging / LLM / checkpointing rules (from CLAUDE.md)

- **No stdout.** Every node logs with `log_event(AgentName.AUTORESEARCH, LogLevel.INFO, "...", metadata={...})`.
- **LLM model selection.** `propose_node` uses `claude-haiku-4-5-20251001` (high-frequency). `Manager.reason_node` uses `claude-sonnet-4-6` (one-time). Synthetic data generation uses haiku.
- **Checkpointing.** `build_autoresearch_graph` must pass a LangGraph checkpointer (e.g. `SqliteSaver`) to `graph.compile()`. This is the whole reason this feature is LangGraph and not a while-loop: a Tinker crash mid-run resumes from the last completed node.
- **No state mutation.** Nodes accept `state` and **return a dict** of only the fields to update. Never mutate `state` directly.
- **No hidden globals.** Everything lives in `AutoResearchState`.

---

## 3. How the reference repo maps to the feature

Useful because "what does `propose_node` actually DO" is most intuitive if you picture the reference repo's human loop.

| Reference repo (human + Claude + git) | Feature (LangGraph) |
|---|---|
| Human creates `autoresearch/<tag>` branch and `results.tsv` header | `init_node` — builds `EvalSuite`, zeros `iteration`, seeds state |
| Agent runs baseline `uv run train.py` first, writes first row of results.tsv | `baseline_node` — runs unmodified script, sets `baseline_score` + `best_score` |
| Agent edits `train.py` with an idea, `git commit` | `propose_node` — Claude generates `Hypothesis`, `apply_patch` writes to disk, saves `original_content` |
| `uv run train.py > run.log` | `run_node` — `submit_experiment` + `wait_for_experiment` on Tinker |
| Crash detection: `tail -n 50 run.log`, "if idea is fundamentally broken, skip" | `early_stop_edge` + `revert_and_continue_node` |
| Agent greps `val_bpb`, compares to previous best | `evaluate_node` — `run_evals` + `compare_scores` |
| If improved, keep the commit ("advance" the branch) | `keep_node` — updates `best_score`, `best_script` |
| If equal/worse, `git reset` back to previous commit | `revert_node` — `revert_patch` restores pre-patch file |
| Append row to `results.tsv` | `log_node` — `log_iteration` writes JSONL diary |
| "NEVER STOP" until manually interrupted | `continue_edge` — stops only on budget exhausted or N-no-improve streak |
| Human's "simplicity criterion" for ties | `decide_keep_or_revert` — tie defaults to REVERT |

The reference repo's tiny `train.py` is literally *an instance* of what the Decision Engine will hand to the feature as `plan.training_script_path` for the pre-train task type. Which means when you're debugging the feature end-to-end, you can (and probably will) use the reference `train.py` as your test input.

---

## 4. Suggested build order (for you + Matthew)

Build bottom-up. The graph nodes are thin wrappers — the "working" functions are where the real code lives and they're testable in isolation.

1. **Types first.** Write `AutoResearchState`, `Hypothesis`, `EvalSuite`, `EvalScore`, `ScoreDelta`, `ExperimentResult`, `IterationRecord` as TypedDicts / dataclasses. Cross-reference `spec.ts` line-by-line. This prevents the "return dict" type drift that will bite you once the graph is running.
2. **Patch machinery.** `apply_patch` / `revert_patch` on a dummy file. Easiest to test. Use `unidiff` or `patch -p0` via subprocess.
3. **Evaluator functions.** `create_eval_suite`, `run_evals`, `compare_scores`, `flag_regression`, `adapt_eval_suite`. Stub `run_evals` with a fake model_path that just returns a random score — you can wire real evals later.
4. **Propose.** `propose_hypothesis` — this is the Claude call. Start with a mock that returns a hand-written patch, then wire up Anthropic SDK with haiku.
5. **Submit/wait stubs.** `submit_experiment`/`wait_for_experiment` — Sid owns the Tinker wrapper. You can stub these returning a fake `ExperimentResult` that runs `train.py` locally and reads the `val_bpb` out of stdout.
6. **Graph nodes.** Each is 5–15 lines that reads from state, calls the function, returns a partial state dict.
7. **Conditional edges.** Tiny (3-line) pure functions.
8. **`build_autoresearch_graph`.** Wire `add_node` / `add_edge` / `add_conditional_edges`. Pass a `SqliteSaver` checkpointer.
9. **`invoke_autoresearch_graph`.** The Manager-facing entry point. Assemble initial state, `graph.invoke(state)`, pull `best_script`/`best_score`/`diary` out of final state and wrap in `TrainedModel`.

**First end-to-end test** you can run without touching Tinker: stub `submit_experiment` to run `autoresearch/train.py` locally with `subprocess.run`, parse the summary block, and route the val_bpb into `EvalScore.scalar`. That's a full closed loop on your laptop using the reference repo as the training script.

---

## 5. Glossary of the acronyms and tricks

- **BPB** — bits per byte. `nats / (ln(2) * bytes)`. Primary metric in reference repo. Vocab-independent.
- **BPE** — Byte-Pair Encoding. Tokenizer algorithm. rustbpe is a fast Rust implementation.
- **FA3** — Flash Attention 3. The fused attention kernel used in `CausalSelfAttention.forward`.
- **GQA** — Grouped-Query Attention. `n_kv_head < n_head`; each KV head is shared across `n_head/n_kv_head` Q heads.
- **MFU** — Model FLOPs Utilization. `flops_per_token * tokens_per_sec / peak_flops`. H100 BF16 peak is `989.5e12`.
- **Muon** — matrix optimizer that orthogonalizes gradients via Newton-Schulz iteration before updating.
- **NorMuon** — Muon variant with second-moment row/column normalization.
- **Polar express** — the specific coefficient set for the 5-step Newton-Schulz iteration used in `muon_step_fused`.
- **ResFormer / Value Embeddings** — per-token, per-layer additive signal into V, gated by a learned function of x.
- **VRAM** — GPU memory. Soft budget in reference repo.
- **LangGraph checkpointer** — mechanism (e.g. `SqliteSaver`) that persists `AutoResearchState` after each node so the graph can resume after a crash.
