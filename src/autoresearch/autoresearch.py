"""
Feature 3 — AutoResearch Loop
Owner: Matthew Torre, Hayley Antczak

Cyclic LangGraph StateGraph(AutoResearchState):
  init → baseline → [propose → run → evaluate → decide → log] × N → END

Includes the Evaluator sub-feature (create_eval_suite, run_evals, adapt_eval_suite).
"""

from __future__ import annotations

import json
import math
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
    JobConfig,
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
    """Constructs and compiles the AutoResearch cyclic LangGraph StateGraph with checkpointer. Called once at startup."""
    from langgraph.graph import END, StateGraph  # lazy import — not available in test environments without the dep

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
    """Entry point called by the Manager's orchestrate_node. Returns the best TrainedModel."""
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
    """LangGraph node. Calls create_eval_suite(). Returns: { eval_suite, current_script, current_config, iteration: 0 }."""
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
    """LangGraph node. Submits and evaluates the unmodified baseline script. Returns: { baseline_score, best_score }."""
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
    """LangGraph node. Calls propose_hypothesis() and apply_patch(). Returns: { current_script, current_patch, last_description, original_content }."""
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

    # Bug fix: patch the config JSON, not the training script (.py files are not patchable).
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
        # current_script keeps the training script path — it is never changed by PROPOSE.
        "current_script": state["plan"]["training_script_path"],
        # current_patch carries the JSON patch string so log_node can build a diary entry.
        "current_patch": hypothesis["patch"],
        # last_description carries Claude's human-readable explanation for the diary.
        "last_description": hypothesis["description"],
        "original_content": original_content,
    }


def run_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls submit_experiment() and wait_for_experiment(). Returns: { last_result }."""
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
    """LangGraph node (early stop path). Calls revert_patch(), logs REVERTED, increments iteration. Returns: { current_script, diary, iteration }."""
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

    # Build and persist the diary entry for this early-stopped iteration.
    # last_result always exists here because run_node already wrote it.
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
    """LangGraph node. Calls run_evals(), compare_scores(), flag_regression(). Returns: { last_score, last_delta }."""
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

    # A flagged regression forces a REVERT without waiting for decide_keep_or_revert.
    # We express this by overwriting improved=False so decision_edge returns "revert".
    if regressed and last_delta["improved"]:
        last_delta = ScoreDelta(
            absolute=last_delta["absolute"],
            relative_pct=last_delta["relative_pct"],
            improved=False,
        )

    return {"last_score": last_score, "last_delta": last_delta}


def keep_node(state: AutoResearchState) -> dict:
    """LangGraph node. Updates best_score and best_script. Resets no_improve_streak. Returns: { best_score, best_script, no_improve_streak: 0 }."""
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"KEEP: new best scalar={state['last_score']['scalar']:.4f} "
        f"(was {state['best_score']['scalar']:.4f})",
        metadata={"iteration": state["iteration"] + 1},
    )
    # Bug fix: advance current_config to reflect the accepted patch so the next
    # call to propose_hypothesis() sees the real current state, not the original baseline.
    patch_dict = json.loads(state.get("current_patch", "{}"))
    updated_config = {**state["current_config"], **patch_dict}

    return {
        "best_score": state["last_score"],
        "best_script": state["plan"]["training_script_path"],
        "current_config": updated_config,
        "no_improve_streak": 0,
    }


def revert_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls revert_patch(). Increments no_improve_streak. Returns: { current_script, no_improve_streak }."""
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
    """LangGraph node. Calls log_iteration(). Calls adapt_eval_suite() every 10 iters. Returns: { diary, eval_suite, iteration }."""
    iteration_number = state["iteration"] + 1

    # revert_and_continue_node already logged its entry and updated state["diary"].
    # Detect that path by checking if iteration_number is already recorded.
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

    # Every 10 iterations identify systematic weaknesses and harden the eval suite.
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

def submit_experiment(
    script_path: str,
    plan: TrainingPlan,
    timeout_min: int = 5,
) -> str:
    """Submits a constrained training run to Tinker. Returns the Tinker job ID."""
    # Translate our TrainingPlan into a Tinker JobConfig.
    # GPU allocation is kept minimal: single A100 with the plan's time budget.
    from src.tinker_api.tinker_api import submit_job

    job_config: JobConfig = {
        "gpu_type": "A100",
        "num_gpus": 1,
        "timeout_min": timeout_min,
        "env_vars": {},
        "output_dir": "outputs/experiments",
    }
    return submit_job(script_path, job_config)


def wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult:
    """Polls Tinker for job completion. Raises TimeoutError if timeout_min exceeded."""
    import time
    from src.tinker_api.tinker_api import get_job_status, get_cumulative_spend

    deadline = time.time() + timeout_min * 60
    poll_interval = 15  # seconds between status checks

    while time.time() < deadline:
        status = get_job_status(job_id)
        if status == JobStatus.COMPLETED:
            cost = get_cumulative_spend(job_id)
            # Tinker writes metrics to outputs/experiments/<job_id>/metrics.json.
            metrics_path = Path("outputs/experiments") / job_id / "metrics.json"
            metrics: TrainingMetrics = json.loads(metrics_path.read_text())
            model_path = str(Path("outputs/experiments") / job_id / "model")
            return {
                "job_id": job_id,
                "status": status,
                "metrics": metrics,
                "model_path": model_path,
                "cost_usd": cost,
                "logs_path": str(Path("outputs/experiments") / job_id / "logs.txt"),
            }
        if status == JobStatus.FAILED:
            return {
                "job_id": job_id,
                "status": status,
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
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Tinker job {job_id} did not finish within {timeout_min} minutes"
    )


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
    """Runs the evaluation suite against the model and returns a scalar score + per-metric breakdown."""
    # Wire-in point for the model inference layer (transformers / PEFT / Tinker artifacts).
    #
    # Expected implementation:
    #   1. Load the model from model_path (AutoModelForSequenceClassification or equivalent).
    #   2. Load the test split from eval_suite["test_split_path"].
    #   3. Run inference and compute each metric in eval_suite["metrics"].
    #   4. If eval_suite["use_llm_grading"], call Claude to score free-form outputs.
    #   5. Return EvalScore with scalar = primary metric value, metrics = per-metric dict.
    #
    # Blocked on: transformers model loading + Tinker artifact download (F4).
    raise NotImplementedError(
        "run_evals not yet implemented. "
        "Requires model loading from Tinker artifacts and inference infrastructure."
    )


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
