"""
Feature 0 — Manager Agent
Owner: Sid Potti

LangGraph StateGraph(ManagerState) with 4 linear nodes:
  query_data_node → reason_node → build_config_node → orchestrate_node
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.types import (
    ManagerState,
    OrchestrationConfig,
    TaskReasoning,
    TrainedModel,
)

LOG_PATH = os.environ.get("DECISIONS_LOG", "decisions.jsonl")


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
    path = query_user_for_data()
    return {"has_data": path is not None, "data_path": path}


def reason_node(state: ManagerState) -> dict:
    """LangGraph node. Calls reason_about_task() via Claude API. Returns: { task_reasoning }."""
    reasoning = reason_about_task(state["prompt"], state["budget"], state["has_data"])
    return {"task_reasoning": reasoning}


def build_config_node(state: ManagerState) -> dict:
    """LangGraph node. Builds OrchestrationConfig and logs the decision. Returns: { config }."""
    config = build_orchestration_config(
        state["task_reasoning"],
        state["prompt"],
        state["budget"],
        state["has_data"],
    )
    log_decision(
        step="build_config",
        rationale=state["task_reasoning"]["notes"],
        config=config,
    )
    return {"config": config}


def orchestrate_node(state: ManagerState) -> dict:
    """LangGraph node. Sequences DataGen → DecisionEngine → AutoResearch. Returns: { result }."""
    raise NotImplementedError


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def query_user_for_data() -> str | None:
    """Interactively asks the user whether they have existing training data.
    Returns the file path if provided, else None.
    """
    answer = input("Do you have existing training data? (y/n): ").strip().lower()
    if answer != "y":
        return None
    path = input("Enter the path to your data file or directory: ").strip()
    if not path or not os.path.exists(path):
        print(f"[Manager] Path '{path}' not found — proceeding without user data.")
        return None
    return os.path.abspath(path)


def log_decision(step: str, rationale: str, config: OrchestrationConfig) -> None:
    """Appends a timestamped entry to the audit trail log (decisions.jsonl)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "rationale": rationale,
        "config_snapshot": dict(config),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def reason_about_task(prompt: str, budget: float, has_data: bool) -> TaskReasoning:
    """Calls Claude API to infer task type, training type, base model, and starting hyperparams."""
    import anthropic

    client = anthropic.Anthropic()
    system = (
        "You are an ML infrastructure planner. Given a task description and budget, "
        "return a JSON object with these exact keys:\n"
        "  task_type: str (e.g. 'text-classification', 'seq2seq', 'token-classification', 'custom')\n"
        "  data_format: str (e.g. 'jsonl with input/output fields', 'csv', 'image directory')\n"
        "  training_type: str ('SFT', 'RL', or 'pre-train')\n"
        "  suggested_base_model: str or null (HuggingFace model ID, e.g. 'bert-base-uncased')\n"
        "  hyperparameters: dict with keys learning_rate, batch_size, epochs, max_seq_len\n"
        "  notes: str (one sentence reasoning summary)\n"
        "Respond with raw JSON only. No markdown, no explanation."
    )
    user_msg = (
        f"Task: {prompt}\n"
        f"Budget: ${budget:.2f}\n"
        f"User has existing data: {has_data}"
    )
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return json.loads(message.content[0].text)


def build_orchestration_config(
    reasoning: TaskReasoning,
    prompt: str,
    budget: float,
    has_data: bool,
) -> OrchestrationConfig:
    """Assembles the OrchestrationConfig dict passed to all downstream agents."""
    return OrchestrationConfig(
        data=has_data,
        prompt=prompt,
        compute_budget=budget,
        training_procedure={
            "task_type": reasoning["task_type"],
            "data_format": reasoning["data_format"],
            "training_type": reasoning["training_type"],
            "base_model": reasoning["suggested_base_model"],
            "hyperparameters": reasoning["hyperparameters"],
            "notes": reasoning["notes"],
        },
    )


def log_decision(step: str, rationale: str, config: OrchestrationConfig) -> None:
    """Appends a timestamped entry to the audit trail log (decisions.jsonl)."""
    raise NotImplementedError
