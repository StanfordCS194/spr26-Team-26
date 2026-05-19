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
from pathlib import Path

from src.data_sources import looks_like_local_data_path, normalize_hf_dataset_source
from src.runtime_context import get_output_root, raise_if_cancelled
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
    *,
    task_type_hint: str | None = None,
) -> TrainedModel:
    """Main entry point for the entire system. Builds and invokes the Manager graph."""
    graph = build_manager_graph()
    initial_state: ManagerState = {
        "prompt": prompt,
        "budget": budget,
        "data_path": data_path,
        "has_data": data_path is not None,
        "task_type_hint": task_type_hint,
        "task_reasoning": None,
        "config": None,
        "result": None,
    }
    raise_if_cancelled()
    final_state = graph.invoke(initial_state)
    raise_if_cancelled()
    return final_state["result"]


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def query_data_node(state: ManagerState) -> dict:
    """LangGraph node. Calls query_user_for_data(). Returns: { has_data, data_path }."""
    raise_if_cancelled()
    existing_path = state.get("data_path")
    if existing_path:
        expanded_path = Path(existing_path).expanduser()
        if expanded_path.exists():
            resolved_path = str(expanded_path.resolve())
            return {"has_data": True, "data_path": resolved_path}
        if looks_like_local_data_path(existing_path):
            return {"has_data": False, "data_path": None}
        hf_source = normalize_hf_dataset_source(existing_path)
        if hf_source:
            return {"has_data": True, "data_path": hf_source}
        return {"has_data": False, "data_path": None}

    try:
        path = query_user_for_data()
    except EOFError:
        path = None
    return {"has_data": path is not None, "data_path": path}


def reason_node(state: ManagerState) -> dict:
    """LangGraph node. Calls reason_about_task() via Claude API. Returns: { task_reasoning }."""
    raise_if_cancelled()
    reasoning = reason_about_task(
        state["prompt"],
        state["budget"],
        state["has_data"],
        task_type_hint=state.get("task_type_hint"),
    )
    raise_if_cancelled()
    return {"task_reasoning": reasoning}


def build_config_node(state: ManagerState) -> dict:
    """LangGraph node. Builds OrchestrationConfig and logs the decision. Returns: { config }."""
    raise_if_cancelled()
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
    raise_if_cancelled()
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
    raise_if_cancelled()
    log_event(AgentName.MANAGER, LogLevel.INFO, "Starting DataGen sub-agent", {})
    handoff = invoke_data_generator_graph(config, state.get("data_path"))
    raise_if_cancelled()
    dataset = _handoff_to_dataset_result(handoff)
    _ensure_dataset_is_trainable(dataset)

    # ── 2. Decision Engine — pick model, estimate cost, write training script
    raise_if_cancelled()
    log_event(AgentName.MANAGER, LogLevel.INFO, "Running Decision Engine", {})
    plan = run_decision_engine(config, dataset)
    raise_if_cancelled()
    log_decision(
        step="orchestrate",
        rationale=f"strategy={plan['strategy']} model={plan['base_model']} "
                  f"estimated_cost=${plan['estimated_cost']:.2f}",
        config=config,
    )

    # ── 3. Cost monitor (background thread) ─────────────────────────────────
    cost_manager = CostManager(budget)

    # ── 4. AutoResearch loop ─────────────────────────────────────────────────
    raise_if_cancelled()
    log_event(AgentName.MANAGER, LogLevel.INFO, "Launching AutoResearch loop", {})
    result = invoke_autoresearch_graph(plan, config, cost_manager)
    raise_if_cancelled()

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


def reason_about_task(
    prompt: str,
    budget: float,
    has_data: bool,
    *,
    task_type_hint: str | None = None,
) -> TaskReasoning:
    """Calls Claude API to infer task type, training type, base model, and starting hyperparams."""
    import anthropic

    hint = _normalize_task_type_hint(task_type_hint)
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
        "If a requested UI task type is provided, treat it as an operator hint: "
        "classification usually maps to text-classification, regression may be "
        "a custom supervised task, and fine-tuning means infer the concrete task "
        "from the objective and data.\n"
        "Respond with raw JSON only. No markdown, no explanation."
    )
    user_msg = (
        f"Task: {prompt}\n"
        f"Budget: ${budget:.2f}\n"
        f"User has existing data: {has_data}"
    )
    if hint:
        user_msg += f"\nRequested UI task type: {hint}"
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return json.loads(message.content[0].text)


def _normalize_task_type_hint(task_type_hint: str | None) -> str | None:
    if task_type_hint is None:
        return None
    value = str(task_type_hint).strip().lower()
    return value if value in {"classification", "regression", "fine-tuning"} else None


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
