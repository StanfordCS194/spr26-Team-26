"""
Feature 0 — Manager Agent
Owner: Sid Potti

LangGraph StateGraph(ManagerState) with 4 linear nodes:
  query_data_node → reason_node → build_config_node → orchestrate_node
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from src.runtime_context import get_output_root
from src.types import (
    DatasetResult,
    ManagerState,
    OrchestrationConfig,
    TaskReasoning,
    TrainedModel,
)

LOG_PATH = os.environ.get("DECISIONS_LOG", "decisions.jsonl")


def build_manager_graph():
    """Constructs and compiles the Manager LangGraph StateGraph. Called once at startup.

    Graph: query_data → reason → build_config → orchestrate (linear, no branching)
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(ManagerState)

    graph.add_node("query_data", query_data_node)
    graph.add_node("reason", reason_node)
    graph.add_node("build_config", build_config_node)
    graph.add_node("orchestrate", orchestrate_node)

    graph.set_entry_point("query_data")
    graph.add_edge("query_data", "reason")
    graph.add_edge("reason", "build_config")
    graph.add_edge("build_config", "orchestrate")
    graph.add_edge("orchestrate", END)

    return graph.compile()


def invoke_manager_graph(
    prompt: str,
    budget: float,
    data_path: str | None = None,
) -> TrainedModel:
    """Main entry point for the entire system. Builds and invokes the Manager graph."""
    graph = build_manager_graph()
    initial_state: ManagerState = {
        "prompt": prompt,
        "budget": budget,
        "data_path": data_path,
        "has_data": data_path is not None,
        "task_reasoning": None,
        "config": None,
        "result": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state["result"]


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def query_data_node(state: ManagerState) -> dict:
    """LangGraph node. Calls query_user_for_data(). Returns: { has_data, data_path }."""
    existing_path = state.get("data_path")
    if existing_path:
        if os.path.exists(existing_path):
            resolved_path = os.path.abspath(existing_path)
            return {"has_data": True, "data_path": resolved_path}
        return {"has_data": False, "data_path": None}

    try:
        path = query_user_for_data()
    except EOFError:
        path = None
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
    from src.data_generator.graph import invoke_data_generator_graph
    from src.decision_engine.decision_engine import run_decision_engine
    from src.autoresearch.autoresearch import invoke_autoresearch_graph
    from src.cost_manager.cost_manager import CostManager
    from src.observability.observability import log_event
    from src.types import AgentName, LogLevel

    config = state["config"]
    budget = config["compute_budget"]

    # ── 1. Data acquisition ──────────────────────────────────────────────────
    log_event(AgentName.MANAGER, LogLevel.INFO, "Starting DataGen sub-agent", {})
    handoff = invoke_data_generator_graph(config, state.get("data_path"))
    dataset = _handoff_to_dataset_result(handoff)
    _ensure_dataset_is_trainable(dataset)

    # ── 2. Decision Engine — pick model, estimate cost, write training script
    log_event(AgentName.MANAGER, LogLevel.INFO, "Running Decision Engine", {})
    plan = run_decision_engine(config, dataset)
    log_decision(
        step="orchestrate",
        rationale=f"strategy={plan['strategy']} model={plan['base_model']} "
                  f"estimated_cost=${plan['estimated_cost']:.2f}",
        config=config,
    )

    # ── 3. Cost monitor (background thread) ─────────────────────────────────
    cost_manager = CostManager(budget)

    # ── 4. AutoResearch loop ─────────────────────────────────────────────────
    log_event(AgentName.MANAGER, LogLevel.INFO, "Launching AutoResearch loop", {})
    result = invoke_autoresearch_graph(plan, config, cost_manager)

    log_event(
        AgentName.MANAGER,
        LogLevel.INFO,
        f"Training complete — {result['n_iterations']} iterations, "
        f"cost=${result['cost']['total_usd']:.2f}",
        {},
    )
    return {"result": result}


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
    log_path = _decision_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _decision_log_path() -> Path:
    root = get_output_root()
    if root is not None:
        return root / "decisions.jsonl"
    return Path(os.environ.get("DECISIONS_LOG", LOG_PATH))


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
    raw_text = message.content[0].text
    return _parse_task_reasoning_response(raw_text)


def _parse_task_reasoning_response(raw_text: str) -> TaskReasoning:
    """Parse and validate the Manager LLM's task-reasoning JSON."""
    try:
        parsed = json.loads(_extract_json_payload(raw_text))
    except json.JSONDecodeError as exc:
        preview = raw_text.strip().replace("\n", " ")[:200]
        raise ValueError(
            f"Manager reasoning response was not valid JSON: {preview!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Manager reasoning response must be a JSON object.")

    required = {
        "task_type",
        "data_format",
        "training_type",
        "suggested_base_model",
        "hyperparameters",
        "notes",
    }
    missing = sorted(required - parsed.keys())
    if missing:
        raise ValueError(f"Manager reasoning response missing keys: {missing}")
    if not isinstance(parsed["hyperparameters"], dict):
        raise ValueError("Manager reasoning hyperparameters must be a JSON object.")

    return TaskReasoning(
        task_type=str(parsed["task_type"]),
        data_format=str(parsed["data_format"]),
        training_type=str(parsed["training_type"]),
        suggested_base_model=(
            None
            if parsed["suggested_base_model"] is None
            else str(parsed["suggested_base_model"])
        ),
        hyperparameters=dict(parsed["hyperparameters"]),
        notes=str(parsed["notes"]),
    )


def _extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _handoff_to_dataset_result(handoff: dict) -> DatasetResult:
    """Converts a DataGen handoff payload into a DatasetResult for the Decision Engine.

    Persists records to outputs/datasets/train_data.jsonl and computes an 80/10/10 split.
    """
    from src.data_generator.curation import curate_handoff_to_dataset_result

    return curate_handoff_to_dataset_result(handoff)


def _ensure_dataset_is_trainable(dataset: DatasetResult) -> None:
    validation = dataset["validation_report"]
    split = dataset["dataset"]
    if validation["passed"] and split["train_size"] > 0:
        return

    issues = "; ".join(validation.get("issues") or [])
    if not issues:
        issues = "no trainable examples were produced"
    raise ValueError(f"DataGen did not produce a trainable dataset: {issues}")


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
