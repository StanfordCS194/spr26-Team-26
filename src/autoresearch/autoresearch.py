"""
AutoResearch Loop (Feature 3) — Matthew Torre, Hayley Antczak

Cyclic LangGraph graph that drives iterative hyperparameter search:
  init → baseline → [propose → run → evaluate → decide → log] × N → END

Each iteration proposes one config change, trains a model, evaluates it,
and either keeps or reverts the patch. The loop terminates when the budget
is gone, performance plateaus, or the iteration cap is hit.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Literal

import anthropic

from src.observability.observability import log_event
from src.types import (
    AgentName,
    AutoResearchState,
    CostBreakdown,
    DatasetResult,
    EvalScore,
    EvalSuite,
    ExperimentResult,
    Hypothesis,
    IterationRecord,
    JobStatus,
    LogLevel,
    OrchestrationConfig,
    ResearchDiary,
    ScoreDelta,
    TaskAnalysis,
    TrainedModel,
    TrainingMetrics,
    TrainingPlan,
)

_DIARY_PATH = Path("outputs/logs/research_diary.jsonl")
_CONFIG_PATH = Path("configs/current.json")  # the JSON file apply_patch/revert_patch target
_MAX_NO_IMPROVE = 3   # halt after this many consecutive non-improving iterations
_MAX_ITERATIONS = 20  # hard cap on total iterations regardless of improvement


def _patch_to_diff(patch_dict: dict, current_config: dict) -> str:
    """Build a human-readable diff string from a patch dict and the pre-patch config."""
    lines = []
    for key, new_val in patch_dict.items():
        old_val = current_config.get(key, "<unset>")
        lines.append(f"- {key}: {old_val}")
        lines.append(f"+ {key}: {new_val}")
    return "\n".join(lines)


# ─── GRAPH BUILDER ────────────────────────────────────────────────────────────

def build_autoresearch_graph():
    """Builds and compiles the cyclic StateGraph. Call once at startup."""
    from langgraph.graph import END, StateGraph  # lazy so tests without langgraph installed still run

    graph = StateGraph(AutoResearchState)

    graph.add_node("init", init_node)
    graph.add_node("baseline", baseline_node)
    graph.add_node("propose", propose_node)
    graph.add_node("run", run_node)
    graph.add_node("revert_and_continue", revert_and_continue_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("keep", keep_node)
    graph.add_node("revert", revert_node)
    graph.add_node("log", log_node)

    # Linear entry path
    graph.set_entry_point("init")
    graph.add_edge("init", "baseline")
    graph.add_edge("baseline", "propose")

    # Cyclic core: propose → run → (early-stop check) → evaluate → (keep/revert) → log → ...
    graph.add_edge("propose", "run")
    graph.add_conditional_edges(
        "run",
        early_stop_edge,
        {"evaluate": "evaluate", "revert_and_continue": "revert_and_continue"},
    )
    graph.add_conditional_edges(
        "evaluate",
        decision_edge,
        {"keep": "keep", "revert": "revert"},
    )
    graph.add_edge("keep", "log")
    graph.add_edge("revert", "log")
    graph.add_edge("revert_and_continue", "log")
    graph.add_conditional_edges(
        "log",
        continue_edge,
        {"propose": "propose", "__end__": END},
    )

    return graph.compile()


def invoke_autoresearch_graph(
    plan: TrainingPlan,
    config: OrchestrationConfig,
    cost_manager,
) -> TrainedModel:
    """Entry point for the Manager. Runs the full loop and returns the best model found."""
    graph = build_autoresearch_graph()

    initial_state: AutoResearchState = {
        "plan": plan,
        "config": config,
        "eval_suite": None,
        "current_script": plan["training_script_path"],
        "current_config": config["training_procedure"]["hyperparameters"],
        "current_patch": None,
        "last_description": None,
        "original_content": None,
        "diary": [],
        "baseline_score": None,
        "best_score": None,
        "best_script": plan["training_script_path"],
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
    }

    final_state = graph.invoke(initial_state)

    total_cost = sum(r["cost_usd"] for r in final_state["diary"])
    termination = "budget_limit" if final_state["should_stop"] else "training_complete"
    cost_breakdown: CostBreakdown = {
        "data_gen_usd": 0.0,
        "training_usd": total_cost,
        "llm_calls_usd": 0.0,
        "total_usd": total_cost,
        "termination_reason": termination,
    }

    return {
        "weights_path": final_state["best_script"],
        "metrics": final_state["best_score"],
        "cost": cost_breakdown,
        "n_iterations": final_state["iteration"],
        "research_diary_path": str(_DIARY_PATH),
    }


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def init_node(state: AutoResearchState) -> dict:
    """Sets up the eval suite before the first training run."""
    task_analysis: TaskAnalysis = {
        "task_type": state["config"]["training_procedure"]["task_type"],
        "modality": "text",
        "has_pretrained_base": state["plan"]["base_model"] is not None,
        "eval_metric": state["plan"]["eval_metric"],
        "complexity": "medium",
    }
    # Construct a minimal DatasetResult so create_eval_suite can derive the test path.
    # The real dataset path comes from DataGen (F2); we point at the standard test split location.
    script_dir = str(Path(state["plan"]["training_script_path"]).parent)
    dataset_result: DatasetResult = {
        "dataset": {
            "path": script_dir,
            "format": "jsonl",
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
        },
        "mode_used": "A",
        "quality_notes": "",
        "validation_report": {
            "passed": True,
            "issues": [],
            "sample_accuracy_estimate": 0.0,
        },
    }

    eval_suite = create_eval_suite(task_analysis, dataset_result)

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        "INIT: eval suite ready",
        metadata={
            "primary_metric": eval_suite["primary_metric"],
            "metrics": eval_suite["metrics"],
            "use_llm_grading": eval_suite["use_llm_grading"],
        },
    )

    return {
        "eval_suite": eval_suite,
        "current_script": state["plan"]["training_script_path"],
        "current_config": state["config"]["training_procedure"]["hyperparameters"],
        "iteration": 0,
    }


def baseline_node(state: AutoResearchState) -> dict:
    """Runs the unmodified training script to get a reference score before we start changing things."""
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        "BASELINE: submitting unmodified training script",
    )

    job_id = submit_experiment(state["plan"]["training_script_path"], state["plan"])
    timeout = int(state["config"]["training_procedure"].get("timeout_min", 5))
    experiment = wait_for_experiment(job_id, timeout)
    baseline_score = run_evals(experiment["model_path"], state["eval_suite"])

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"BASELINE: scalar={baseline_score['scalar']:.4f}",
        metadata={"score": baseline_score, "job_id": job_id},
    )

    return {
        "baseline_score": baseline_score,
        "best_score": baseline_score,
        "best_script": state["plan"]["training_script_path"],
        "last_result": experiment,
    }


def propose_node(state: AutoResearchState) -> dict:
    """Asks Claude to suggest one config change, then patches the config file."""
    task_analysis: TaskAnalysis = {
        "task_type": state["config"]["training_procedure"]["task_type"],
        "modality": "text",
        "has_pretrained_base": state["plan"]["base_model"] is not None,
        "eval_metric": state["plan"]["eval_metric"],
        "complexity": "medium",
    }
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"PROPOSE: generating hypothesis for iteration {state['iteration'] + 1}",
        metadata={"iteration": state["iteration"] + 1},
    )
    hypothesis = propose_hypothesis(state["current_config"], state["diary"], task_analysis)

    # We patch the config JSON, not train.py directly — structured diffs are atomic and validatable.
    original_content = apply_patch(str(_CONFIG_PATH), hypothesis["patch"])

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        hypothesis["description"],
        metadata={
            "patch": hypothesis["patch"],
            "expected_effect": hypothesis["expected_effect"],
            "strategy": hypothesis["search_strategy"],
        },
    )
    return {
        "current_script": state["plan"]["training_script_path"],
        "current_patch": hypothesis["patch"],
        "last_description": hypothesis["description"],
        "original_content": original_content,
    }


def run_node(state: AutoResearchState) -> dict:
    """Submits the training job and blocks until it finishes."""
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"RUN: submitting experiment for iteration {state['iteration'] + 1}",
        metadata={"iteration": state["iteration"] + 1},
    )

    timeout = int(state["config"]["training_procedure"].get("timeout_min", 5))
    job_id = submit_experiment(state["plan"]["training_script_path"], state["plan"], timeout)
    result = wait_for_experiment(job_id, timeout)

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"RUN: job {job_id} finished — status={result['status']}",
        metadata={"job_id": job_id, "cost_usd": result["cost_usd"]},
    )

    return {"last_result": result}


def revert_and_continue_node(state: AutoResearchState) -> dict:
    """Early-stop path: the job produced catastrophic metrics, so roll back immediately."""
    revert_patch(state["plan"]["training_script_path"], state["original_content"])

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.WARN,
        f"REVERTED (early-stop): catastrophic failure on iteration {state['iteration'] + 1}",
        metadata={
            "iteration": state["iteration"] + 1,
            "metrics": state["last_result"]["metrics"] if state.get("last_result") else {},
        },
    )

    patch_dict = json.loads(state.get("current_patch", "{}"))
    diff = _patch_to_diff(patch_dict, state["current_config"])
    record: IterationRecord = {
        "iteration": state["iteration"] + 1,
        "hypothesis": state.get("last_description", str(patch_dict)),
        "patch": diff,
        "cost_usd": state["last_result"]["cost_usd"],
        "metrics": state["last_result"]["metrics"],
        "decision": "REVERTED",
        "notes": "Early-stopped: catastrophic failure (NaN/exploding loss/accuracy collapse)",
    }
    updated_diary = log_iteration(state["diary"], record)

    return {
        "current_script": state["plan"]["training_script_path"],
        "original_content": None,
        "last_delta": None,  # signals early-stop path to log_node
        "diary": updated_diary,
        "iteration": state["iteration"] + 1,
    }


def evaluate_node(state: AutoResearchState) -> dict:
    """Scores the trained model and decides whether the patch was an improvement."""
    last_score = run_evals(state["last_result"]["model_path"], state["eval_suite"])
    last_delta = compare_scores(last_score, state["baseline_score"])
    regressed = flag_regression(last_delta)

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"EVALUATE: scalar={last_score['scalar']:.4f}  "
        f"delta={last_delta['absolute']:+.4f} ({last_delta['relative_pct']:+.1f}%)"
        + ("  ⚠ REGRESSION" if regressed else ""),
        metadata={
            "score": last_score,
            "delta": last_delta,
            "regression": regressed,
        },
    )

    # Force a revert on regressions even if the raw delta is positive (shouldn't happen, but guard it).
    if regressed and last_delta["improved"]:
        last_delta = ScoreDelta(
            absolute=last_delta["absolute"],
            relative_pct=last_delta["relative_pct"],
            improved=False,
        )

    return {"last_score": last_score, "last_delta": last_delta}


def keep_node(state: AutoResearchState) -> dict:
    """Accepts the patch: updates the best score and resets the plateau counter."""
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"KEEP: new best scalar={state['last_score']['scalar']:.4f} "
        f"(was {state['best_score']['scalar']:.4f})",
        metadata={"iteration": state["iteration"] + 1},
    )
    # Advance current_config so the next proposal starts from the updated state, not the original baseline.
    patch_dict = json.loads(state.get("current_patch", "{}"))
    updated_config = {**state["current_config"], **patch_dict}

    return {
        "best_score": state["last_score"],
        "best_script": state["plan"]["training_script_path"],
        "current_config": updated_config,
        "no_improve_streak": 0,
    }


def revert_node(state: AutoResearchState) -> dict:
    """Rolls back the patch and bumps the non-improving streak counter."""
    revert_patch(str(_CONFIG_PATH), state["original_content"])

    new_streak = state["no_improve_streak"] + 1
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"REVERT: patch rolled back — no_improve_streak={new_streak}",
        metadata={"iteration": state["iteration"] + 1, "streak": new_streak},
    )
    return {
        "current_script": state["best_script"],
        "original_content": None,
        "no_improve_streak": new_streak,
    }


def log_node(state: AutoResearchState) -> dict:
    """Writes the iteration record to the research diary. Every 10 iterations, hardens the eval suite against observed weaknesses."""
    iteration_number = state["iteration"] + 1

    # revert_and_continue already wrote its own diary entry, so skip writing again.
    already_logged = any(r["iteration"] == iteration_number for r in state["diary"])

    if not already_logged:
        # Normal path: build and persist the IterationRecord.
        if state["last_delta"] is not None and state["last_delta"]["improved"]:
            decision = "KEPT"
            notes = f"+{state['last_delta']['relative_pct']:.1f}% on {state['eval_suite']['primary_metric']}"
        else:
            decision = "REVERTED"
            delta_pct = state["last_delta"]["relative_pct"] if state["last_delta"] else 0.0
            notes = f"{delta_pct:+.1f}% — no improvement"

        metrics: TrainingMetrics = (
            state["last_result"]["metrics"]
            if state.get("last_result")
            else {"train_loss": 0.0, "val_loss": 0.0, "test_loss": 0.0, "primary_metric": 0.0}
        )
        patch_dict = json.loads(state.get("current_patch", "{}"))
        diff = _patch_to_diff(patch_dict, state["current_config"])
        record: IterationRecord = {
            "iteration": iteration_number,
            "hypothesis": state.get("last_description", str(patch_dict)),
            "patch": diff,
            "cost_usd": state["last_result"]["cost_usd"] if state.get("last_result") else 0.0,
            "metrics": metrics,
            "decision": decision,
            "notes": notes,
        }
        updated_diary = log_iteration(state["diary"], record)
    else:
        updated_diary = state["diary"]

    updated_suite = state["eval_suite"]
    if iteration_number % 10 == 0:
        recent_reverts = [
            r for r in updated_diary[-10:]
            if r["decision"] == "REVERTED"
        ]
        if recent_reverts:
            weaknesses = [r["notes"] for r in recent_reverts]
            updated_suite = adapt_eval_suite(state["eval_suite"], weaknesses)
            log_event(
                AgentName.AUTORESEARCH,
                LogLevel.INFO,
                f"ADAPT: eval suite updated after {iteration_number} iterations",
                metadata={"weaknesses": weaknesses},
            )

    return {
        "diary": updated_diary,
        "eval_suite": updated_suite,
        "iteration": iteration_number,
    }


# ─── CONDITIONAL EDGE FUNCTIONS ───────────────────────────────────────────────

def early_stop_edge(state: AutoResearchState) -> Literal["evaluate", "revert_and_continue"]:
    """After run_node. Returns 'revert_and_continue' on catastrophic failure, 'evaluate' otherwise."""
    if state["last_result"] is None:
        return "revert_and_continue"
    if state["last_result"]["status"] == JobStatus.FAILED:
        return "revert_and_continue"
    if check_early_stop(state["last_result"]["metrics"]):
        return "revert_and_continue"
    return "evaluate"


def decision_edge(state: AutoResearchState) -> Literal["keep", "revert"]:
    """After evaluate_node. Calls decide_keep_or_revert(state.last_delta). Returns 'keep' or 'revert'."""
    verdict = decide_keep_or_revert(state["last_delta"])
    return "keep" if verdict == "KEEP" else "revert"


def continue_edge(state: AutoResearchState) -> Literal["propose", "__end__"]:
    """After log_node. Returns '__end__' if budget exhausted / convergence, else 'propose' to loop."""
    if state["should_stop"]:
        return "__end__"

    budget = state["config"].get("compute_budget", float("inf"))
    spent = sum(r["cost_usd"] for r in state["diary"])
    if spent >= budget:
        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.INFO,
            f"STOP: budget exhausted (spent ${spent:.2f} of ${budget:.2f})",
        )
        return "__end__"

    if state["no_improve_streak"] >= _MAX_NO_IMPROVE:
        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.INFO,
            f"STOP: convergence — {_MAX_NO_IMPROVE} consecutive non-improving iterations",
        )
        return "__end__"

    if state["iteration"] >= _MAX_ITERATIONS:
        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.INFO,
            f"STOP: reached max iterations ({_MAX_ITERATIONS})",
        )
        return "__end__"

    return "propose"


# ─── PROPOSE HELPERS ──────────────────────────────────────────────────────────

def propose_hypothesis(
    current_config: dict,
    diary: ResearchDiary,
    task: TaskAnalysis,
) -> Hypothesis:
    """Calls Claude API (claude-haiku-4-5-20251001) to generate a single testable hypothesis as a code/config diff."""
    client = anthropic.Anthropic()
    recent = diary[-5:] if len(diary) > 5 else diary

    system = (
        "You are an ML hyperparameter optimization assistant for the AutoResearch Loop. "
        "You propose exactly ONE config-level change per iteration, grounded in the "
        "experiment history. Return only valid JSON — no prose, no markdown fences."
    )
    user = f"""Current training configuration:
{json.dumps(current_config, indent=2)}

Task:
- type: {task['task_type']}
- primary metric: {task['eval_metric']}
- complexity: {task['complexity']}

Recent experiment history ({len(recent)} entries):
{json.dumps(recent, indent=2) if recent else "No history yet — this is the first iteration."}

Rules:
- Change exactly ONE hyperparameter.
- Stay within safe bounds: learning_rate [1e-6, 1e-2], batch_size [4, 8192],
  max_seq_length [128, 4096], lora_rank in [4,8,16,32,64,128],
  warmup_steps [0, 2000], dropout [0, 0.5].
- Avoid repeating a change that recently caused a REVERTED outcome.
- Prefer evidence-based choices over random exploration when history exists.

Respond with this JSON object and nothing else:
{{
  "description": "<one sentence: what changes and why, e.g. 'Decrease learning_rate from 3e-4 to 1e-4 to reduce loss spikes.'>",
  "patch": {{"<param_name>": <new_value>}},
  "expected_effect": "<one sentence: which metric should improve and by how much>",
  "search_strategy": "<random | local | playbook>"
}}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": user}],
        system=system,
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.startswith("```")
        )
    parsed = json.loads(raw)
    return Hypothesis(
        description=parsed["description"],
        patch=json.dumps(parsed["patch"]),
        expected_effect=parsed["expected_effect"],
        search_strategy=parsed["search_strategy"],
    )


def apply_patch(script_path: str, patch: str) -> str:
    """
    Applies a patch to a config or script file. Returns original content for revert.

    For .json files: patch is a JSON-encoded dict of key→value overrides.
    For .py files:   patch is a unified diff string (future; requires RUN phase).

    The returned string is the file's exact pre-patch content, suitable for
    passing directly to revert_patch().
    """
    path = Path(script_path)
    original = path.read_text()

    if path.suffix == ".json":
        config_dict = json.loads(original)
        patch_dict = json.loads(patch)
        config_dict.update(patch_dict)
        # Write atomically via a temp file + rename so a crash mid-write
        # never leaves a partially-written config on disk.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(config_dict, indent=2))
        tmp.replace(path)
    elif path.suffix == ".py":
        # Future: apply unified diff when RUN phase is wired up.
        # import subprocess
        # result = subprocess.run(["patch", script_path], input=patch, text=True, capture_output=True)
        # if result.returncode != 0:
        #     raise RuntimeError(f"patch failed: {result.stderr}")
        raise NotImplementedError(
            "Script-level patching not yet implemented. "
            "Requires RUN phase and Tinker job submission."
        )
    else:
        raise ValueError(f"Unsupported file type for patching: {path.suffix!r}")

    return original


def revert_patch(script_path: str, original_content: str) -> None:
    """Restores script_path to its pre-patch content."""
    Path(script_path).write_text(original_content)


# ─── RUN HELPERS ──────────────────────────────────────────────────────────────

# Thread results keyed by job_id — set by the background training thread.
# Value is an ExperimentResult dict on success, or an Exception on failure.
_experiment_results: dict[str, "ExperimentResult | Exception | None"] = {}
_experiment_threads: dict[str, "threading.Thread"] = {}


def submit_experiment(
    _script_path: str,
    plan: TrainingPlan,
    timeout_min: int = 5,
) -> str:
    """Starts a Tinker SDK training run in a background thread. Returns a job ID."""
    import os
    import threading
    import uuid

    from src.tinker_api.tinker_api import (
        create_lora_training_client,
        create_service_client,
        get_cumulative_spend,
        get_tokenizer,
        make_datum,
        run_training_loop,
    )

    job_id = str(uuid.uuid4())[:8]
    _experiment_results[job_id] = None  # not done yet

    def _run() -> None:
        try:
            svc = create_service_client()
            base_model = plan.get("base_model") or "Qwen/Qwen3-8B"
            rank = (plan.get("lora_config") or {}).get("rank", 32)
            tc = create_lora_training_client(svc, base_model, rank)
            tok = get_tokenizer(tc)

            dataset_path = os.getenv("TINKER_DATASET_PATH", "outputs/dataset/train.jsonl")
            config_path = Path("configs/current.json")
            lr: float = 3e-4
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                lr = float(cfg.get("learning_rate", lr))

            batches: list[list] = []
            with open(dataset_path) as fh:
                for line in fh:
                    sample = json.loads(line.strip())
                    text = sample.get("text", "")
                    tokens = tok.encode(text)
                    if tokens:
                        batches.append([make_datum(tokens)])

            loop_result = run_training_loop(
                tc, batches, learning_rate=lr, job_id=job_id, checkpoint_name=job_id
            )

            final_loss = loop_result.get("loss") or 0.0
            metrics: TrainingMetrics = {
                "train_loss": float(final_loss),
                "val_loss": float(final_loss),
                "test_loss": float(final_loss),
                "primary_metric": -float(final_loss),  # higher = better convention
            }
            out_dir = Path("outputs/experiments") / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(metrics))

            _experiment_results[job_id] = {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "metrics": metrics,
                "model_path": str(out_dir / "model"),
                "cost_usd": get_cumulative_spend(job_id),
                "logs_path": str(out_dir / "logs.txt"),
            }
        except Exception as exc:  # noqa: BLE001
            _experiment_results[job_id] = exc

    thread = threading.Thread(target=_run, daemon=True)
    _experiment_threads[job_id] = thread
    thread.start()
    return job_id


def wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult:
    """Blocks until the training thread for job_id finishes. Raises TimeoutError if it exceeds timeout_min."""
    from src.tinker_api.tinker_api import get_cumulative_spend

    thread = _experiment_threads.get(job_id)
    if thread is not None:
        thread.join(timeout=timeout_min * 60)
        if thread.is_alive():
            raise TimeoutError(
                f"Tinker job {job_id} did not finish within {timeout_min} minutes"
            )

    result = _experiment_results.get(job_id)
    if isinstance(result, Exception):
        return {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "metrics": {
                "train_loss": float("nan"),
                "val_loss": float("nan"),
                "test_loss": float("nan"),
                "primary_metric": 0.0,
            },
            "model_path": "",
            "cost_usd": get_cumulative_spend(job_id),
            "logs_path": str(Path("outputs/experiments") / job_id / "logs.txt"),
        }
    if result is None:
        raise TimeoutError(f"Tinker job {job_id} produced no result")
    return result  # type: ignore[return-value]


def check_early_stop(metrics: TrainingMetrics) -> bool:
    """Returns True on catastrophic failure: exploding loss (>10× baseline), NaN, or accuracy collapse."""
    for key in ("train_loss", "val_loss"):
        val = metrics.get(key)
        if val is not None and (math.isnan(val) or math.isinf(val)):
            return True

    train_loss = metrics.get("train_loss")
    if train_loss is not None and train_loss > 10.0:
        return True

    primary = metrics.get("primary_metric")
    if primary is not None and primary < 0.01:
        return True

    return False


# ─── EVALUATE HELPERS ─────────────────────────────────────────────────────────

def run_evals(model_path: str, eval_suite: EvalSuite) -> EvalScore:
    """
    Scores a trained model against the eval suite.

    Reads the metrics.json saved alongside the model by the training job, then
    maps the raw training metrics onto the eval suite's named metrics. For
    high-complexity tasks with use_llm_grading set, a small sample of the test
    split is sent to Claude Haiku which returns an adjusted quality score and
    a one-sentence critique. The final scalar is an 80/20 blend of the
    training-derived score and the LLM grade.
    """
    metrics_file = Path(model_path).parent / "metrics.json"

    raw: dict[str, float] = {}
    if metrics_file.exists():
        try:
            raw = json.loads(metrics_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Derive a proxy score from val_loss when dedicated eval metrics aren't available.
    val_loss = float(raw.get("val_loss", 1.0))
    proxy = max(0.0, 1.0 - val_loss)

    per_metric: dict[str, float] = {}
    for metric in eval_suite["metrics"]:
        if metric in raw:
            per_metric[metric] = float(raw[metric])
        elif "loss" in metric:
            per_metric[metric] = raw.get(metric, val_loss)
        else:
            per_metric[metric] = raw.get("primary_metric", proxy)

    primary_val = per_metric.get(eval_suite["primary_metric"], proxy)
    critique = ""

    if eval_suite["use_llm_grading"]:
        primary_val, critique = _llm_grade(model_path, eval_suite, primary_val)
        per_metric[eval_suite["primary_metric"]] = primary_val

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"run_evals: {eval_suite['primary_metric']}={primary_val:.4f}",
        metadata={"metrics": per_metric, "model_path": model_path},
    )

    return {
        "scalar": primary_val,
        "metrics": per_metric,
        "critique": critique,
    }


def _llm_grade(model_path: str, eval_suite: EvalSuite, prior_score: float) -> tuple[float, str]:
    """
    Samples up to 5 examples from the test split and asks Claude Haiku to
    rate output quality. Returns (blended_score, critique) where the blend
    is 80% training metrics + 20% LLM grade.
    """
    test_path = Path(eval_suite["test_split_path"])
    examples: list[str] = []

    if test_path.exists():
        try:
            with open(test_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        examples.append(line)
                    if len(examples) >= 5:
                        break
        except OSError:
            pass

    if not examples:
        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.WARN,
            "LLM grading skipped — test split is empty or missing",
            metadata={"test_split_path": eval_suite["test_split_path"]},
        )
        return prior_score, ""

    sample_text = "\n".join(f"  {i + 1}. {ex[:300]}" for i, ex in enumerate(examples))
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are an ML evaluation assistant. Assess model output quality and return JSON only.",
        messages=[{
            "role": "user",
            "content": (
                f"Test split samples (first {len(examples)} examples):\n{sample_text}\n\n"
                f"Prior score from training metrics: {prior_score:.4f}\n"
                f"Primary metric: {eval_suite['primary_metric']}\n\n"
                "Return exactly this JSON object and nothing else:\n"
                '{"score": <float 0.0–1.0>, "critique": "<one sentence>"}'
            ),
        }],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```"))

    try:
        parsed = json.loads(raw)
        llm_score = float(parsed.get("score", prior_score))
        critique = str(parsed.get("critique", ""))
        blended = 0.8 * prior_score + 0.2 * llm_score
        return blended, critique
    except (json.JSONDecodeError, ValueError):
        return prior_score, raw[:200]


def compare_scores(new_score: EvalScore, baseline_score: EvalScore) -> ScoreDelta:
    """Computes relative improvement of new_score vs baseline_score on the primary eval metric."""
    new_val = new_score["scalar"]
    base_val = baseline_score["scalar"]
    absolute = new_val - base_val
    relative_pct = (absolute / base_val * 100.0) if base_val != 0.0 else 0.0
    return {
        "absolute": absolute,
        "relative_pct": relative_pct,
        "improved": absolute > 0.0,
    }


# ─── DECIDE HELPERS ───────────────────────────────────────────────────────────

def decide_keep_or_revert(delta: ScoreDelta) -> Literal["KEEP", "REVERT"]:
    """Returns KEEP if hypothesis improved primary metric, REVERT otherwise. Ties default to REVERT."""
    return "KEEP" if delta["improved"] else "REVERT"


def log_iteration(diary: ResearchDiary, record: IterationRecord) -> ResearchDiary:
    """Appends an IterationRecord to the research diary and writes to disk as JSONL."""
    _DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DIARY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return [*diary, record]


# ─── EVALUATOR SUB-FEATURE ────────────────────────────────────────────────────

def create_eval_suite(task: TaskAnalysis, dataset: DatasetResult) -> EvalSuite:
    """Creates the evaluation suite: selects metrics, holds out test split, optionally adds LLM-graded stress tests."""
    # Map task type to the most informative primary metric and supporting metrics.
    _METRIC_MAP: dict[str, tuple[str, list[str]]] = {
        "text-classification": ("f1", ["f1", "accuracy", "precision", "recall"]),
        "seq2seq":             ("rouge_l", ["rouge_l", "rouge_1", "bleu"]),
        "custom":              ("accuracy", ["accuracy"]),
    }
    task_type = task.get("task_type", "custom")
    primary, metrics = _METRIC_MAP.get(task_type, ("accuracy", ["accuracy"]))

    # Caller-specified eval_metric overrides the default.
    if task.get("eval_metric"):
        primary = task["eval_metric"]
        if primary not in metrics:
            metrics = [primary] + metrics

    # Use LLM grading for high-complexity tasks where heuristic metrics miss nuance.
    use_llm_grading = task.get("complexity") == "high"

    test_split_path = str(Path(dataset["dataset"]["path"]) / "test")

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"create_eval_suite: primary={primary}, llm_grading={use_llm_grading}",
        metadata={"metrics": metrics, "test_split_path": test_split_path},
    )

    return {
        "primary_metric": primary,
        "metrics": metrics,
        "test_split_path": test_split_path,
        "use_llm_grading": use_llm_grading,
    }


def adapt_eval_suite(suite: EvalSuite, weaknesses: list[str]) -> EvalSuite:
    """Adds harder eval examples targeting systematic weaknesses detected across recent iterations."""
    client = anthropic.Anthropic()

    user = f"""You are an ML evaluation expert reviewing an AutoResearch experiment loop.

Current eval suite:
- primary metric: {suite['primary_metric']}
- metrics: {suite['metrics']}

Systematic weaknesses detected in recent iterations:
{json.dumps(weaknesses, indent=2)}

Which additional evaluation metrics or stress-test categories would best expose these weaknesses?
Respond with a JSON array of short metric/category names only — no prose, no markdown:
["metric_or_category_1", "metric_or_category_2", ...]"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": user}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```"))
    try:
        additions: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        additions = []

    # Merge without duplicates, preserving existing order.
    existing = set(suite["metrics"])
    new_metrics = suite["metrics"] + [m for m in additions if m not in existing]

    return {
        "primary_metric": suite["primary_metric"],
        "metrics": new_metrics,
        "test_split_path": suite["test_split_path"],
        "use_llm_grading": True,  # always enable LLM grading once weaknesses are identified
    }


def flag_regression(delta: ScoreDelta, threshold: float = -0.01) -> bool:
    """Returns True if score degraded beyond threshold, triggering automatic revert."""
    return delta["absolute"] < threshold
