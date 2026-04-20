"""
Feature 3 — AutoResearch Loop
Owner: Matthew Torre, Hayley Antczak

Cyclic LangGraph StateGraph(AutoResearchState):
  init → baseline → [propose → run → evaluate → decide → log] × N → END

Includes the Evaluator sub-feature (create_eval_suite, run_evals, adapt_eval_suite).
"""

from __future__ import annotations

from typing import Literal

from src.types import (
    AutoResearchState,
    DatasetResult,
    EvalScore,
    EvalSuite,
    ExperimentResult,
    Hypothesis,
    IterationRecord,
    OrchestrationConfig,
    ResearchDiary,
    ScoreDelta,
    TaskAnalysis,
    TrainedModel,
    TrainingMetrics,
    TrainingPlan,
)


def build_autoresearch_graph():
    """Constructs and compiles the AutoResearch cyclic LangGraph StateGraph with checkpointer. Called once at startup."""
    raise NotImplementedError


def invoke_autoresearch_graph(
    plan: TrainingPlan,
    config: OrchestrationConfig,
    cost_manager,
) -> TrainedModel:
    """Entry point called by the Manager's orchestrate_node. Returns the best TrainedModel."""
    raise NotImplementedError


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def init_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls create_eval_suite(). Returns: { eval_suite, current_script, current_config, iteration: 0 }."""
    raise NotImplementedError


def baseline_node(state: AutoResearchState) -> dict:
    """LangGraph node. Submits and evaluates the unmodified baseline script. Returns: { baseline_score, best_score }."""
    raise NotImplementedError


def propose_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls propose_hypothesis() and apply_patch(). Returns: { current_script, original_content, last_hypothesis }."""
    raise NotImplementedError


def run_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls submit_experiment() and wait_for_experiment(). Returns: { last_result }."""
    raise NotImplementedError


def revert_and_continue_node(state: AutoResearchState) -> dict:
    """LangGraph node (early stop path). Calls revert_patch(), logs REVERTED, increments iteration. Returns: { current_script, diary, iteration }."""
    raise NotImplementedError


def evaluate_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls run_evals(), compare_scores(), flag_regression(). Returns: { last_score, last_delta }."""
    raise NotImplementedError


def keep_node(state: AutoResearchState) -> dict:
    """LangGraph node. Updates best_score and best_script. Resets no_improve_streak. Returns: { best_score, best_script, no_improve_streak: 0 }."""
    raise NotImplementedError


def revert_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls revert_patch(). Increments no_improve_streak. Returns: { current_script, no_improve_streak }."""
    raise NotImplementedError


def log_node(state: AutoResearchState) -> dict:
    """LangGraph node. Calls log_iteration(). Calls adapt_eval_suite() every 10 iters. Returns: { diary, eval_suite, iteration }."""
    raise NotImplementedError


# ─── CONDITIONAL EDGE FUNCTIONS ───────────────────────────────────────────────

def early_stop_edge(state: AutoResearchState) -> Literal["evaluate", "revert_and_continue"]:
    """After run_node. Returns 'revert_and_continue' on catastrophic failure, 'evaluate' otherwise."""
    raise NotImplementedError


def decision_edge(state: AutoResearchState) -> Literal["keep", "revert"]:
    """After evaluate_node. Calls decide_keep_or_revert(state.last_delta). Returns 'keep' or 'revert'."""
    raise NotImplementedError


def continue_edge(state: AutoResearchState) -> Literal["propose", "__end__"]:
    """After log_node. Returns '__end__' if budget exhausted / convergence, else 'propose' to loop."""
    raise NotImplementedError


# ─── PROPOSE HELPERS ──────────────────────────────────────────────────────────

def propose_hypothesis(
    current_config: dict,
    diary: ResearchDiary,
    task: TaskAnalysis,
) -> Hypothesis:
    """Calls Claude API (claude-haiku-4-5-20251001) to generate a single testable hypothesis as a code/config diff."""
    raise NotImplementedError


def apply_patch(script_path: str, patch: str) -> str:
    """Applies a unified diff patch to train.py. Returns original content for revert."""
    raise NotImplementedError


def revert_patch(script_path: str, original_content: str) -> None:
    """Restores train.py to its pre-patch content."""
    raise NotImplementedError


# ─── RUN HELPERS ──────────────────────────────────────────────────────────────

def submit_experiment(
    script_path: str,
    plan: TrainingPlan,
    timeout_min: int = 5,
) -> str:
    """Submits a constrained training run to Tinker. Returns the Tinker job ID."""
    raise NotImplementedError


def wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult:
    """Polls Tinker for job completion. Raises TimeoutError if timeout_min exceeded."""
    raise NotImplementedError


def check_early_stop(metrics: TrainingMetrics) -> bool:
    """Returns True on catastrophic failure: exploding loss (>10× baseline), NaN, or accuracy collapse."""
    raise NotImplementedError


# ─── EVALUATE HELPERS ─────────────────────────────────────────────────────────

def run_evals(model_path: str, eval_suite: EvalSuite) -> EvalScore:
    """Runs the evaluation suite against the model and returns a scalar score + per-metric breakdown."""
    raise NotImplementedError


def compare_scores(new_score: EvalScore, baseline_score: EvalScore) -> ScoreDelta:
    """Computes relative improvement of new_score vs baseline_score on the primary eval metric."""
    raise NotImplementedError


# ─── DECIDE HELPERS ───────────────────────────────────────────────────────────

def decide_keep_or_revert(delta: ScoreDelta) -> Literal["KEEP", "REVERT"]:
    """Returns KEEP if hypothesis improved primary metric, REVERT otherwise. Ties default to REVERT."""
    raise NotImplementedError


def log_iteration(diary: ResearchDiary, record: IterationRecord) -> ResearchDiary:
    """Appends an IterationRecord to the research diary and writes to disk as JSONL."""
    raise NotImplementedError


# ─── EVALUATOR SUB-FEATURE ────────────────────────────────────────────────────

def create_eval_suite(task: TaskAnalysis, dataset: DatasetResult) -> EvalSuite:
    """Creates the evaluation suite: selects metrics, holds out test split, optionally adds LLM-graded stress tests."""
    raise NotImplementedError


def adapt_eval_suite(suite: EvalSuite, weaknesses: list[str]) -> EvalSuite:
    """Adds harder eval examples targeting systematic weaknesses detected across recent iterations."""
    raise NotImplementedError


def flag_regression(delta: ScoreDelta, threshold: float = -0.01) -> bool:
    """Returns True if score degraded beyond threshold, triggering automatic revert."""
    raise NotImplementedError
