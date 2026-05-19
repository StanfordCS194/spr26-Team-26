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
from typing import Any, Literal, Mapping
from uuid import uuid4

import anthropic

from src.autoresearch.config import TrainingConfig
from src.observability.observability import log_event
from src.runtime_context import raise_if_cancelled, resolve_output_path
from src.tinker_api.sft_runner import (
    DEFAULT_LIVE_SMOKE_STEPS,
    DEFAULT_TINKER_MODEL,
    SUPPORTED_TINKER_TUNABLES,
    run_tinker_sft_experiment,
)
from src.types import (
    AgentName,
    AutoResearchState,
    BudgetStatus,
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
_MIN_RELATIVE_IMPROVEMENT_PCT = 1.0
_MIN_ABSOLUTE_IMPROVEMENT = 1e-9
_EXPERIMENT_CACHE: dict[str, ExperimentResult] = {}
_TINKER_HYPERPARAMETER_ALIASES = {
    "epochs": "num_epochs",
    "max_seq_len": "max_seq_length",
}


def _diary_path() -> Path:
    return resolve_output_path(_DIARY_PATH, "logs", "research_diary.jsonl")


def _config_path() -> Path:
    return resolve_output_path(_CONFIG_PATH, "configs", "current.json")


def _experiments_output_dir() -> Path:
    return resolve_output_path(Path("outputs/experiments"), "experiments")


def _patch_to_diff(patch_dict: dict, current_config: dict) -> str:
    """Build a human-readable diff string from a patch dict and the pre-patch config."""
    lines = []
    for key, new_val in patch_dict.items():
        old_val = current_config.get(key, "<unset>")
        lines.append(f"- {key}: {old_val}")
        lines.append(f"+ {key}: {new_val}")
    return "\n".join(lines)


def _canonicalize_tinker_hyperparameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Tinker V1 hyperparameters with legacy Manager aliases removed."""
    canonical = dict(config)
    for legacy_key, canonical_key in _TINKER_HYPERPARAMETER_ALIASES.items():
        if legacy_key in canonical and canonical_key not in canonical:
            canonical[canonical_key] = canonical[legacy_key]
        canonical.pop(legacy_key, None)
    return canonical


def _current_config_for_plan(
    plan: TrainingPlan,
    config: OrchestrationConfig,
) -> dict[str, Any]:
    current_config = dict(config["training_procedure"]["hyperparameters"])
    if plan.get("backend") == "tinker_sft":
        return _canonicalize_tinker_hyperparameters(current_config)
    return current_config


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
    graph.add_conditional_edges(
        "baseline",
        continue_edge,
        {"propose": "propose", "__end__": END},
    )

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
        "cost_manager": cost_manager,
        "eval_suite": None,
        "current_script": plan["training_script_path"],
        "current_config": _current_config_for_plan(plan, config),
        "current_patch": None,
        "last_description": None,
        "original_content": None,
        "diary": [],
        "baseline_score": None,
        "baseline_result": None,
        "best_score": None,
        "best_script": plan["training_script_path"],
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
    }

    raise_if_cancelled()
    final_state = graph.invoke(initial_state)
    raise_if_cancelled()

    termination = "budget_limit" if _budget_exhausted(final_state) else "training_complete"
    cost_breakdown = _cost_breakdown_from_state(final_state, termination)

    return {
        "weights_path": final_state["best_script"],
        "metrics": final_state["best_score"],
        "cost": cost_breakdown,
        "n_iterations": final_state["iteration"],
        "research_diary_path": str(_diary_path()),
    }


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def init_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls create_eval_suite(). Returns: { eval_suite, current_script, current_config, iteration: 0 }."""
    raise_if_cancelled()
    _write_current_config(_training_config_from_state(state))
    task_analysis: TaskAnalysis = {
        "task_type": state["config"]["training_procedure"]["task_type"],
        "modality": "text",
        "has_pretrained_base": state["plan"]["base_model"] is not None,
        "eval_metric": state["plan"]["eval_metric"],
        "complexity": "medium",
    }
    dataset_result = _dataset_result_from_plan(state["plan"])

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
        "current_config": _current_config_for_plan(state["plan"], state["config"]),
        "iteration": 0,
    }


def baseline_node(state: AutoResearchState) -> dict:
    """LangGraph node. Submits and evaluates the unmodified baseline script. Returns: { baseline_score, best_score }."""
    raise_if_cancelled()
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        "BASELINE: submitting unmodified training script",
    )

    experiment = _run_tinker_experiment_for_state(state, phase="baseline")
    budget_status = _record_experiment_cost(state, experiment)
    raise_if_cancelled()
    baseline_score = run_evals(experiment["model_path"], state["eval_suite"])

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"BASELINE: scalar={baseline_score['scalar']:.4f}",
        metadata={"score": baseline_score, "job_id": experiment["job_id"]},
    )

    return {
        "baseline_score": baseline_score,
        "baseline_result": experiment,
        "best_score": baseline_score,
        "best_script": experiment["model_path"],
        "last_result": experiment,
        "should_stop": (
            budget_status == BudgetStatus.EXCEEDED
            or experiment["status"] == JobStatus.CANCELLED
        ),
    }


def propose_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls propose_hypothesis() and apply_patch(). Returns: { current_script, current_patch, last_description, original_content }."""
    raise_if_cancelled()
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
    allowed_params = (
        sorted(SUPPORTED_TINKER_TUNABLES)
        if state["plan"].get("backend") == "tinker_sft"
        else None
    )
    hypothesis = propose_hypothesis(
        state["current_config"],
        state["diary"],
        task_analysis,
        allowed_params=allowed_params,
    )
    raise_if_cancelled()

    # Bug fix: patch the config JSON, not the training script (.py files are not patchable).
    original_content = apply_patch(str(_config_path()), hypothesis["patch"])

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
    """LangGraph node. Runs a bounded SDK-native Tinker experiment. Returns: { last_result }."""
    raise_if_cancelled()
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"RUN: submitting experiment for iteration {state['iteration'] + 1}",
        metadata={"iteration": state["iteration"] + 1},
    )

    result = _run_tinker_experiment_for_state(state, phase="iteration")
    budget_status = _record_experiment_cost(state, result)

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"RUN: job {result['job_id']} finished — status={result['status']}",
        metadata={"job_id": result["job_id"], "cost_usd": result["cost_usd"]},
    )
    raise_if_cancelled()

    return {
        "last_result": result,
        "should_stop": (
            budget_status == BudgetStatus.EXCEEDED
            or result["status"] == JobStatus.CANCELLED
        ),
    }


def revert_and_continue_node(state: AutoResearchState) -> dict:
    """LangGraph node (early stop path). Reverts the patch before normal logging."""
    raise_if_cancelled()
    revert_patch(str(_config_path()), state["original_content"])

    budget_skipped = _last_result_was_budget_skipped(state)
    reason = (
        "budget preflight skipped iteration"
        if budget_skipped
        else "catastrophic failure"
    )
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.WARN,
        f"REVERTED (early-stop): {reason} on iteration {state['iteration'] + 1}",
        metadata={
            "iteration": state["iteration"] + 1,
            "metrics": state["last_result"]["metrics"] if state.get("last_result") else {},
        },
    )

    return {
        "current_script": state["plan"]["training_script_path"],
        "original_content": None,
        "last_delta": None,  # signals early-stop path to log_node.
    }


def evaluate_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls run_evals(), compare_scores(), flag_regression(). Returns: { last_score, last_delta }."""
    raise_if_cancelled()
    last_score = run_evals(state["last_result"]["model_path"], state["eval_suite"])
    reference_score = state.get("best_score") or state["baseline_score"]
    last_delta = compare_scores(last_score, reference_score)
    regressed = flag_regression(last_delta)

    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"EVALUATE: scalar={last_score['scalar']:.4f}  "
        f"delta={last_delta['absolute']:+.4f} ({last_delta['relative_pct']:+.1f}%)"
        + ("  ⚠ REGRESSION" if regressed else ""),
        metadata={
            "score": last_score,
            "reference_score": reference_score,
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
    raise_if_cancelled()
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
    if state["plan"].get("backend") == "tinker_sft":
        updated_config = _canonicalize_tinker_hyperparameters(updated_config)

    return {
        "best_score": state["last_score"],
        "best_script": state["last_result"]["model_path"],
        "current_config": updated_config,
        "no_improve_streak": 0,
    }


def revert_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls revert_patch(). Increments no_improve_streak. Returns: { current_script, no_improve_streak }."""
    raise_if_cancelled()
    revert_patch(str(_config_path()), state["original_content"])

    new_streak = state["no_improve_streak"] + 1
    log_event(
        AgentName.AUTORESEARCH,
        LogLevel.INFO,
        f"REVERT: patch rolled back — no_improve_streak={new_streak}",
        metadata={"iteration": state["iteration"] + 1, "streak": new_streak},
    )
    return {
        "current_script": state["plan"]["training_script_path"],
        "original_content": None,
        "no_improve_streak": new_streak,
    }


def log_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls log_iteration(). Calls adapt_eval_suite() every 10 iters. Returns: { diary, eval_suite, iteration }."""
    raise_if_cancelled()
    iteration_number = state["iteration"] + 1

    # revert_and_continue_node already logged its entry and updated state["diary"].
    # Detect that path by checking if iteration_number is already recorded.
    already_logged = any(r["iteration"] == iteration_number for r in state["diary"])

    if not already_logged:
        # Normal path: build and persist the IterationRecord.
        if state["last_delta"] is not None and state["last_delta"]["improved"]:
            decision = "KEPT"
            notes = f"+{state['last_delta']['relative_pct']:.1f}% on {state['eval_suite']['primary_metric']}"
        elif _last_result_was_early_stopped(state):
            decision = "REVERTED"
            notes = (
                "Skipped: budget preflight rejected the Tinker run"
                if _last_result_was_budget_skipped(state)
                else "Early-stopped: catastrophic failure (NaN/Inf loss or primary metric collapse)"
            )
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
        diff = _patch_to_diff(patch_dict, _pre_patch_config_from_state(state))
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
    if state["last_result"]["status"] in {JobStatus.FAILED, JobStatus.CANCELLED}:
        return "revert_and_continue"
    if check_early_stop(state["last_result"]["metrics"]):
        return "revert_and_continue"
    return "evaluate"


def _last_result_was_early_stopped(state: AutoResearchState) -> bool:
    if state.get("last_delta") is not None:
        return False
    last_result = state.get("last_result")
    if last_result is None:
        return True
    if last_result["status"] in {JobStatus.FAILED, JobStatus.CANCELLED}:
        return True
    return check_early_stop(last_result["metrics"])


def _last_result_was_budget_skipped(state: AutoResearchState) -> bool:
    last_result = state.get("last_result")
    if not last_result:
        return False
    try:
        manifest_path = Path(last_result["model_path"]) / "manifest.json"
    except (KeyError, TypeError):
        return False
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(manifest.get("budget_preflight_skipped"))


def decision_edge(state: AutoResearchState) -> Literal["keep", "revert"]:
    """After evaluate_node. Calls decide_keep_or_revert(state.last_delta). Returns 'keep' or 'revert'."""
    verdict = decide_keep_or_revert(state["last_delta"])
    return "keep" if verdict == "KEEP" else "revert"


def continue_edge(state: AutoResearchState) -> Literal["propose", "__end__"]:
    """After log_node. Returns '__end__' if budget exhausted / convergence, else 'propose' to loop."""
    raise_if_cancelled()
    if state["should_stop"]:
        return "__end__"

    budget = state["config"].get("compute_budget", float("inf"))
    spent = _spent_usd_from_state(state)
    if _budget_exhausted(state):
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


def _pre_patch_config_from_state(state: AutoResearchState) -> dict[str, Any]:
    original_content = state.get("original_content")
    if original_content:
        try:
            original_config = json.loads(original_content)
        except json.JSONDecodeError:
            original_config = None
        if isinstance(original_config, dict):
            return original_config
    return state["current_config"]


def _spent_usd_from_state(state: AutoResearchState) -> float:
    cost_manager = state.get("cost_manager")
    if cost_manager is not None and hasattr(cost_manager, "spent_usd"):
        return float(cost_manager.spent_usd)
    baseline_cost = (
        state.get("baseline_result", {}).get("cost_usd", 0.0)
        if state.get("baseline_result")
        else 0.0
    )
    return baseline_cost + sum(r["cost_usd"] for r in state["diary"])


def _budget_exhausted(state: AutoResearchState) -> bool:
    if state.get("should_stop"):
        return True
    cost_manager = state.get("cost_manager")
    if cost_manager is not None and getattr(cost_manager, "status", None) == BudgetStatus.EXCEEDED:
        return True
    budget = state["config"].get("compute_budget", float("inf"))
    return _spent_usd_from_state(state) >= budget


def _cost_breakdown_from_state(
    state: AutoResearchState,
    termination_reason: str,
) -> CostBreakdown:
    cost_manager = state.get("cost_manager")
    if cost_manager is not None and hasattr(cost_manager, "cost_breakdown"):
        return cost_manager.cost_breakdown(termination_reason=termination_reason)

    total_cost = _spent_usd_from_state(state)
    return {
        "data_gen_usd": 0.0,
        "training_usd": total_cost,
        "llm_calls_usd": 0.0,
        "total_usd": total_cost,
        "termination_reason": termination_reason,
    }


# ─── PROPOSE HELPERS ──────────────────────────────────────────────────────────

def propose_hypothesis(
    current_config: dict,
    diary: ResearchDiary,
    task: TaskAnalysis,
    allowed_params: list[str] | None = None,
) -> Hypothesis:
    """Calls Claude API (claude-haiku-4-5-20251001) to generate a single testable hypothesis as a code/config diff."""
    client = anthropic.Anthropic()
    recent = diary[-5:] if len(diary) > 5 else diary
    prompt_config = (
        _canonicalize_tinker_hyperparameters(current_config)
        if allowed_params
        else current_config
    )

    system = (
        "You are an ML hyperparameter optimization assistant for the AutoResearch Loop. "
        "You propose exactly ONE config-level change per iteration, grounded in the "
        "experiment history. Return only valid JSON — no prose, no markdown fences."
    )
    change_rule = (
        f"Change exactly ONE of these supported hyperparameters: {', '.join(allowed_params)}."
        if allowed_params
        else "Change exactly ONE hyperparameter."
    )
    if allowed_params:
        bounds_rule = (
            "Stay within safe bounds: learning_rate [1e-6, 1e-2], batch_size [4, 8192], "
            "max_seq_length [128, 4096], lora_rank in [4,8,16,32,64,128], "
            "num_epochs [1, 100]."
        )
    else:
        bounds_rule = (
            "Stay within safe bounds: learning_rate [1e-6, 1e-2], batch_size [4, 8192], "
            "max_seq_length [128, 4096], lora_rank in [4,8,16,32,64,128], "
            "warmup_steps [0, 2000], dropout [0, 0.5]."
        )
    user = f"""Current training configuration:
{json.dumps(prompt_config, indent=2)}

Task:
- type: {task['task_type']}
- primary metric: {task['eval_metric']}
- complexity: {task['complexity']}

Recent experiment history ({len(recent)} entries):
{json.dumps(recent, indent=2) if recent else "No history yet — this is the first iteration."}

Rules:
- {change_rule}
- {bounds_rule}
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
    patch = parsed["patch"]
    if allowed_params:
        patch = _canonicalize_tinker_hyperparameters(patch)
        unsupported = [key for key in patch if key not in set(allowed_params)]
        if unsupported:
            raise ValueError(
                f"Unsupported Tinker SFT hyperparameter(s): {unsupported}. "
                f"Allowed: {allowed_params}"
            )
    return Hypothesis(
        description=parsed["description"],
        patch=json.dumps(patch),
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
        config_dict = TrainingConfig.from_dict(config_dict).apply_patch(patch_dict).to_dict()
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
            "The Tinker SFT v1 path patches JSON config only."
        )
    else:
        raise ValueError(f"Unsupported file type for patching: {path.suffix!r}")

    return original


def revert_patch(script_path: str, original_content: str) -> None:
    """Restores script_path to its pre-patch content."""
    Path(script_path).write_text(original_content)


# ─── RUN HELPERS ──────────────────────────────────────────────────────────────

def _dataset_result_from_plan(plan: TrainingPlan) -> DatasetResult:
    dataset_meta = plan.get("dataset")
    if isinstance(dataset_meta, Mapping):
        dataset_path = str(
            dataset_meta.get("path")
            or plan.get("dataset_path")
            or Path(plan["training_script_path"]).parent
        )
        dataset_format = str(dataset_meta.get("format") or "jsonl")
        train_size = max(0, int(dataset_meta.get("train_size") or 0))
        val_size = max(0, int(dataset_meta.get("val_size") or 0))
        test_size = max(0, int(dataset_meta.get("test_size") or 0))
    else:
        dataset_path = plan.get("dataset_path") or str(
            Path(plan["training_script_path"]).parent
        )
        dataset_format = "jsonl"
        train_size = 0
        val_size = 0
        test_size = 0

    return {
        "dataset": {
            "path": dataset_path,
            "format": dataset_format,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        },
        "mode_used": "A",
        "quality_notes": "AutoResearch Tinker SFT dataset",
        "validation_report": {
            "passed": True,
            "issues": [],
            "sample_accuracy_estimate": 0.0,
        },
    }


def _training_config_from_state(
    state: AutoResearchState,
    *,
    include_pending_patch: bool = False,
) -> TrainingConfig:
    data: dict[str, Any] = (
        _canonicalize_tinker_hyperparameters(state["current_config"])
        if state["plan"].get("backend") == "tinker_sft"
        else dict(state["current_config"])
    )
    data.setdefault("model_name", state["plan"].get("base_model") or DEFAULT_TINKER_MODEL)
    lora = state["plan"].get("lora_config")
    if lora and "lora_rank" not in data:
        data["lora_rank"] = lora["rank"]
    if lora and "lora_alpha" not in data:
        data["lora_alpha"] = lora["alpha"]
    config = TrainingConfig.from_dict(data)
    if include_pending_patch and state.get("current_patch"):
        config = config.apply_patch(json.loads(state["current_patch"]))
    return config


def _write_current_config(config: TrainingConfig) -> None:
    config.save(_config_path())


def _training_config_from_plan(plan: TrainingPlan) -> TrainingConfig:
    config_path = _config_path()
    if config_path.exists():
        data = TrainingConfig.load(config_path).to_dict()
    else:
        data = {"model_name": plan.get("base_model") or DEFAULT_TINKER_MODEL}
    data["model_name"] = plan.get("base_model") or data.get("model_name") or DEFAULT_TINKER_MODEL
    lora = plan.get("lora_config")
    if lora:
        data.setdefault("lora_rank", lora["rank"])
        data.setdefault("lora_alpha", lora["alpha"])
    return TrainingConfig.from_dict(data)


def _max_steps_from_state(state: AutoResearchState) -> int:
    procedure = state["config"].get("training_procedure", {})
    hyperparameters = procedure.get("hyperparameters", {})
    for source in (state["current_config"], hyperparameters, procedure):
        value = source.get("max_steps") if isinstance(source, dict) else None
        if value is not None:
            return max(1, int(value))
    return DEFAULT_LIVE_SMOKE_STEPS


def _estimated_run_cost_from_state(state: AutoResearchState) -> float:
    """Returns the best available USD estimate for the next Tinker launch."""
    plan = state.get("plan", {})
    procedure = state.get("config", {}).get("training_procedure", {})
    hyperparameters = (
        procedure.get("hyperparameters", {})
        if isinstance(procedure, Mapping)
        else {}
    )
    candidates = (
        plan.get("estimated_run_cost_usd"),
        hyperparameters.get("estimated_run_cost_usd")
        if isinstance(hyperparameters, Mapping)
        else None,
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _remaining_budget_from_state(state: AutoResearchState) -> float:
    cost_manager = state.get("cost_manager")
    if cost_manager is not None and hasattr(cost_manager, "remaining_budget"):
        try:
            return max(0.0, float(cost_manager.remaining_budget))
        except (TypeError, ValueError):
            pass

    budget = state.get("config", {}).get("compute_budget", float("inf"))
    try:
        return max(0.0, float(budget) - _spent_usd_from_state(state))
    except (TypeError, ValueError):
        return float("inf")


def _budget_preflight_skip_reason(
    state: AutoResearchState,
    estimated_cost: float,
) -> str | None:
    cost_manager = state.get("cost_manager")
    remaining = _remaining_budget_from_state(state)

    if cost_manager is not None and hasattr(cost_manager, "can_start_run"):
        can_start = cost_manager.can_start_run(estimated_cost)
    else:
        can_start = remaining > 0 and estimated_cost <= remaining

    if can_start:
        return None
    if remaining <= 0:
        return "remaining budget is exhausted before launch"
    if estimated_cost > remaining:
        return (
            f"estimated run cost ${estimated_cost:.2f} exceeds remaining "
            f"budget ${remaining:.2f}"
        )
    return "budget preflight rejected launch"


def _budget_limited_experiment_result(
    run_id: str,
    *,
    reason: str,
    estimated_cost: float,
    remaining_budget: float,
    output_dir: str = "outputs/experiments",
) -> ExperimentResult:
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    loss = float("inf")
    metrics: TrainingMetrics = {
        "train_loss": loss,
        "val_loss": loss,
        "test_loss": loss,
        "primary_metric": 0.0,
    }
    metrics_row = {
        "step": 0,
        **metrics,
        "budget_preflight_skipped": True,
        "reason": reason,
    }
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics_row) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": JobStatus.CANCELLED,
                "backend": "tinker_sft",
                "budget_preflight_skipped": True,
                "budget_skip_reason": reason,
                "estimated_cost_usd": round(float(estimated_cost), 6),
                "remaining_budget_usd": round(float(remaining_budget), 6),
                "completed_steps": 0,
                "checkpoints": {},
            },
            indent=2,
        )
    )
    (run_dir / "sample.json").write_text(
        json.dumps({"prompt": [], "text": "", "tokens": [], "error": reason}, indent=2)
    )
    return {
        "job_id": run_id,
        "status": JobStatus.CANCELLED,
        "metrics": metrics,
        "model_path": str(run_dir),
        "cost_usd": 0.0,
        "logs_path": str(metrics_path),
    }


def _run_tinker_experiment_for_state(
    state: AutoResearchState,
    *,
    phase: str,
) -> ExperimentResult:
    run_id = f"autoresearch-{phase}-{state['iteration']}-{uuid4().hex[:8]}"
    estimated_cost = _estimated_run_cost_from_state(state)
    remaining_budget = _remaining_budget_from_state(state)
    skip_reason = _budget_preflight_skip_reason(state, estimated_cost)
    if skip_reason:
        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.WARN,
            "BUDGET: skipped Tinker launch before spend",
            metadata={
                "run_id": run_id,
                "phase": phase,
                "estimated_cost_usd": estimated_cost,
                "remaining_budget_usd": remaining_budget,
                "reason": skip_reason,
            },
        )
        return _budget_limited_experiment_result(
            run_id,
            reason=skip_reason,
            estimated_cost=estimated_cost,
            remaining_budget=remaining_budget,
            output_dir=str(_experiments_output_dir()),
        )

    cost_manager = state.get("cost_manager")
    if cost_manager is not None and hasattr(cost_manager, "start"):
        cost_manager.start(run_id)
    try:
        return run_tinker_sft_experiment(
            _training_config_from_state(
                state,
                include_pending_patch=phase == "iteration",
            ),
            _dataset_result_from_plan(state["plan"]),
            run_id=run_id,
            max_steps=_max_steps_from_state(state),
            output_dir=str(_experiments_output_dir()),
        )
    finally:
        if cost_manager is not None and hasattr(cost_manager, "stop"):
            cost_manager.stop()


def _record_experiment_cost(
    state: AutoResearchState,
    result: ExperimentResult,
    category: str = "training",
) -> str | None:
    cost_manager = state.get("cost_manager")
    if cost_manager is None or not hasattr(cost_manager, "record_spend"):
        return None
    accounted_cost = _accounted_experiment_cost(state, result)
    result["cost_usd"] = accounted_cost
    return cost_manager.record_spend(accounted_cost, category=category)


def _accounted_experiment_cost(
    state: AutoResearchState,
    result: ExperimentResult,
) -> float:
    actual_cost = max(0.0, float(result.get("cost_usd", 0.0)))
    if _experiment_result_was_budget_skipped(result):
        return actual_cost
    return max(actual_cost, _estimated_run_cost_from_state(state))


def _experiment_result_was_budget_skipped(result: ExperimentResult) -> bool:
    try:
        manifest_path = Path(result["model_path"]) / "manifest.json"
    except (KeyError, TypeError):
        return False
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(manifest.get("budget_preflight_skipped"))


def submit_experiment(
    script_path: str,
    plan: TrainingPlan,
    timeout_min: int = 5,
) -> str:
    """Run a constrained SDK-native Tinker experiment and return its run ID.

    Kept as a compatibility wrapper for older call sites; it no longer submits
    a REST job.
    """
    _ = script_path, timeout_min
    result = run_tinker_sft_experiment(
        _training_config_from_plan(plan),
        _dataset_result_from_plan(plan),
        max_steps=DEFAULT_LIVE_SMOKE_STEPS,
        output_dir=str(_experiments_output_dir()),
    )
    _EXPERIMENT_CACHE[result["job_id"]] = result
    return result["job_id"]


def wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult:
    """Return a completed SDK-native Tinker experiment result."""
    _ = timeout_min
    if job_id in _EXPERIMENT_CACHE:
        return _EXPERIMENT_CACHE[job_id]

    run_dir = _experiments_output_dir() / job_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise TimeoutError(f"Tinker run {job_id} has no metrics artifact")
    metrics: TrainingMetrics = json.loads(metrics_path.read_text())
    manifest_path = run_dir / "manifest.json"
    status = JobStatus.COMPLETED
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        status = manifest.get("status", status)
    return {
        "job_id": job_id,
        "status": status,
        "metrics": metrics,
        "model_path": str(run_dir),
        "cost_usd": 0.0,
        "logs_path": str(run_dir / "metrics.jsonl"),
    }


def check_early_stop(metrics: TrainingMetrics) -> bool:
    """Returns True on catastrophic failure: non-finite loss or primary metric collapse."""
    for key in ("train_loss", "val_loss"):
        val = metrics.get(key)
        if val is not None and (math.isnan(val) or math.isinf(val)):
            return True

    primary = metrics.get("primary_metric")
    if primary is not None and (
        math.isnan(primary) or math.isinf(primary) or primary < 0.01
    ):
        return True

    return False


# ─── EVALUATE HELPERS ─────────────────────────────────────────────────────────

def run_evals(model_path: str, eval_suite: EvalSuite) -> EvalScore:
    """Score Tinker SFT artifacts using val loss as the v1 proxy metric."""
    _ = eval_suite
    metrics_path = Path(model_path) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Tinker metrics artifact: {metrics_path}")
    raw_metrics = json.loads(metrics_path.read_text())
    val_loss = float(raw_metrics.get("val_loss", raw_metrics.get("train_loss", float("nan"))))
    scalar = 0.0 if not math.isfinite(val_loss) or val_loss < 0 else 1.0 / (1.0 + val_loss)
    metrics = {
        "train_loss": float(raw_metrics.get("train_loss", val_loss)),
        "val_loss": val_loss,
        "test_loss": float(raw_metrics.get("test_loss", val_loss)),
        "primary_metric": scalar,
    }
    return {
        "scalar": scalar,
        "metrics": metrics,
        "critique": "Tinker SFT v1 score uses primary_metric = 1 / (1 + val_loss).",
    }


def compare_scores(new_score: EvalScore, baseline_score: EvalScore) -> ScoreDelta:
    """Computes relative improvement of new_score vs baseline_score on the primary eval metric."""
    new_val = new_score["scalar"]
    base_val = baseline_score["scalar"]
    absolute = new_val - base_val
    relative_pct = (absolute / base_val * 100.0) if base_val != 0.0 else 0.0
    if base_val == 0.0:
        improved = absolute > _MIN_ABSOLUTE_IMPROVEMENT
    else:
        improved = (
            absolute > _MIN_ABSOLUTE_IMPROVEMENT
            and relative_pct >= _MIN_RELATIVE_IMPROVEMENT_PCT
        )
    return {
        "absolute": absolute,
        "relative_pct": relative_pct,
        "improved": improved,
    }


# ─── DECIDE HELPERS ───────────────────────────────────────────────────────────

def decide_keep_or_revert(delta: ScoreDelta) -> Literal["KEEP", "REVERT"]:
    """Returns KEEP if hypothesis improved primary metric, REVERT otherwise. Ties default to REVERT."""
    return "KEEP" if delta["improved"] else "REVERT"


def log_iteration(diary: ResearchDiary, record: IterationRecord) -> ResearchDiary:
    """Appends an IterationRecord to the research diary and writes to disk as JSONL."""
    diary_path = _diary_path()
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diary_path, "a") as f:
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
