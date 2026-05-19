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

from src.types import (
    DatasetResult,
    ManagerState,
    OrchestrationConfig,
    TaskReasoning,
    TrainedModel,
)

LOG_PATH = os.environ.get("DECISIONS_LOG", "decisions.jsonl")
_MANAGER_REASONER_ENV = "MANAGER_REASONER"


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
    interactive_data_prompt: bool = False,
) -> TrainedModel:
    """Main entry point for the system.

    Programmatic callers default to no-data Mode C when ``data_path`` is not
    supplied. CLI-style callers can opt into the legacy stdin prompt with
    ``interactive_data_prompt=True``.
    """
    graph = build_manager_graph()
    initial_state: ManagerState = {
        "prompt": prompt,
        "budget": budget,
        "data_path": data_path,
        "has_data": data_path is not None,
        "interactive_data_prompt": interactive_data_prompt,
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

    if not state.get("interactive_data_prompt", True):
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
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def reason_about_task(
    prompt: str,
    budget: float,
    has_data: bool,
    *,
    task_type_hint: str | None = None,
) -> TaskReasoning:
    """Infer task type, training type, base model, and starting hyperparams."""
    hint = _normalize_task_type_hint(task_type_hint)
    if _use_local_reasoner():
        return _local_task_reasoning(prompt, budget, has_data, task_type_hint=hint)

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
    raw_text = message.content[0].text
    return _parse_task_reasoning_response(raw_text)


def _manager_reasoner_mode() -> str:
    raw = os.getenv(_MANAGER_REASONER_ENV, "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"local", "offline", "heuristic"}:
        return "local"
    if raw in {"claude", "anthropic", "live", "required"}:
        return "claude"
    raise ValueError(f"{_MANAGER_REASONER_ENV} must be one of: auto, local, claude")


def _use_local_reasoner() -> bool:
    mode = _manager_reasoner_mode()
    if _env_flag_enabled("NO_SPEND"):
        return True
    if mode in {"auto", "local"}:
        return True
    if mode == "claude":
        return False
    return True


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_task_type_hint(task_type_hint: str | None) -> str | None:
    if task_type_hint is None:
        return None
    value = str(task_type_hint).strip().lower()
    return value if value in {"classification", "regression", "fine-tuning"} else None


def _local_task_reasoning(
    prompt: str,
    budget: float,
    has_data: bool,
    *,
    task_type_hint: str | None = None,
) -> TaskReasoning:
    """Deterministic fallback planner for no-live Manager runs."""
    hint = _normalize_task_type_hint(task_type_hint)
    task_type = _infer_local_task_type(prompt, task_type_hint=hint)
    data_format = (
        "jsonl with messages or input/output fields"
        if has_data
        else "generated jsonl with messages or input/output fields"
    )
    hyperparameters = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 1,
        "num_epochs": 1,
        "max_seq_len": 512,
        "max_seq_length": 512,
        "lora_rank": 8,
        "max_steps": 5,
    }
    return TaskReasoning(
        task_type=task_type,
        data_format=data_format,
        training_type="SFT",
        suggested_base_model=None,
        hyperparameters=hyperparameters,
        notes=(
            "Local deterministic planner used because live Manager reasoning "
            "was not enabled."
            + (f" Operator task hint: {hint}." if hint else "")
        ),
    )


def _infer_local_task_type(prompt: str, *, task_type_hint: str | None = None) -> str:
    hint = _normalize_task_type_hint(task_type_hint)
    if hint == "classification":
        return "text-classification"
    if hint == "regression":
        return "custom"

    text = prompt.lower()
    if any(term in text for term in ("translate", "translation")):
        return "translation"
    if any(term in text for term in ("summarize", "summarise", "summary")):
        return "summarization"
    if any(term in text for term in ("question answering", "question-answering", "qa")):
        return "question-answering"
    if any(
        term in text
        for term in (
            "classify",
            "classification",
            "sentiment",
            "label",
            "category",
            "categorize",
            "categorise",
            "intent",
        )
    ):
        return "text-classification"
    return "custom"


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
