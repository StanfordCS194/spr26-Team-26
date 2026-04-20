"""
Feature 0 — Manager Agent
Owner: Sid Potti

LangGraph StateGraph(ManagerState) with 4 linear nodes:
  query_data_node → reason_node → build_config_node → orchestrate_node
"""

from __future__ import annotations

from src.types import (
    ManagerState,
    OrchestrationConfig,
    TaskReasoning,
    TrainedModel,
)


def build_manager_graph():
    """Constructs and compiles the Manager LangGraph StateGraph. Called once at startup."""
    raise NotImplementedError


def invoke_manager_graph(
    prompt: str,
    budget: float,
    data_path: str | None = None,
) -> TrainedModel:
    """Main entry point for the entire system. Invokes the compiled Manager graph."""
    raise NotImplementedError


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def query_data_node(state: ManagerState) -> dict:
    """LangGraph node. Calls query_user_for_data(). Returns: { has_data, data_path }."""
    raise NotImplementedError


def reason_node(state: ManagerState) -> dict:
    """LangGraph node. Calls reason_about_task() via Claude API. Returns: { task_reasoning }."""
    raise NotImplementedError


def build_config_node(state: ManagerState) -> dict:
    """LangGraph node. Calls build_orchestration_config() and log_decision(). Returns: { config }."""
    raise NotImplementedError


def orchestrate_node(state: ManagerState) -> dict:
    """LangGraph node. Sequences DataGen → DecisionEngine → AutoResearch. Returns: { result }."""
    raise NotImplementedError


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def query_user_for_data() -> str | None:
    """Interactively asks the user whether they have existing training data."""
    raise NotImplementedError


def reason_about_task(prompt: str, budget: float, has_data: bool) -> TaskReasoning:
    """Calls Claude API (claude-sonnet-4-6) to infer task type, training type, base model, hyperparams."""
    raise NotImplementedError


def build_orchestration_config(
    reasoning: TaskReasoning,
    prompt: str,
    budget: float,
    has_data: bool,
) -> OrchestrationConfig:
    """Assembles the OrchestrationConfig dict passed to all downstream agents."""
    raise NotImplementedError


def log_decision(step: str, rationale: str, config: OrchestrationConfig) -> None:
    """Appends a timestamped entry to the audit trail log (decisions.jsonl)."""
    raise NotImplementedError
