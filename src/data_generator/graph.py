from __future__ import annotations

from typing import Any

from src.data_generator.edges import select_curation_edge, select_mode_edge
from src.data_generator.nodes import (
    acquire_hf_data_node,
    acquire_user_data_node,
    acquire_web_data_node,
    handoff_structure_data_node,
    handoff_validate_hf_node,
    route_node,
)
from src.types import DataGenState, OrchestrationConfig

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = START = StateGraph = None


def build_data_generator_graph():
    """First sub-agent graph: acquire data and handoff to second sub-agent."""
    if StateGraph is None:
        return _FallbackDataGeneratorGraph()

    graph = StateGraph(DataGenState)
    graph.add_node("route_node", route_node)
    graph.add_node("acquire_user_data", acquire_user_data_node)
    graph.add_node("acquire_hf_data", acquire_hf_data_node)
    graph.add_node("acquire_web_data", acquire_web_data_node)
    graph.add_node("structure_data", handoff_structure_data_node)
    graph.add_node("validate_hf_data", handoff_validate_hf_node)

    graph.add_edge(START, "route_node")
    graph.add_conditional_edges(
        "route_node",
        select_mode_edge,
        {
            "acquire_user_data": "acquire_user_data",
            "acquire_hf_data": "acquire_hf_data",
            "acquire_web_data": "acquire_web_data",
        },
    )
    graph.add_conditional_edges(
        "acquire_user_data",
        select_curation_edge,
        {"structure_data": "structure_data", "validate_hf_data": "validate_hf_data"},
    )
    graph.add_conditional_edges(
        "acquire_hf_data",
        select_curation_edge,
        {"structure_data": "structure_data", "validate_hf_data": "validate_hf_data"},
    )
    graph.add_conditional_edges(
        "acquire_web_data",
        select_curation_edge,
        {"structure_data": "structure_data", "validate_hf_data": "validate_hf_data"},
    )
    graph.add_edge("structure_data", END)
    graph.add_edge("validate_hf_data", END)
    return graph.compile()


def invoke_data_generator_graph(config: OrchestrationConfig, data_path: str | None) -> dict[str, Any]:
    """
    Entry point for the first sub-agent only.
    Returns handoff payload for second sub-agent.
    """
    graph = build_data_generator_graph()
    initial_state: DataGenState = {
        "config": config,
        "data_path": data_path,
        "mode": None,
        "raw_data": None,
        "hf_candidates": [],
        "selected_candidate": None,
        "schema": None,
        "dataset": None,
        "validation_report": None,
        "handoff": None,
    }
    final_state = graph.invoke(initial_state)
    handoff = final_state.get("handoff")
    if not handoff:
        raise RuntimeError("Data generator did not produce a handoff payload.")
    return handoff


class _FallbackDataGeneratorGraph:
    """Sequential fallback when LangGraph is unavailable."""

    def invoke(self, state: DataGenState) -> DataGenState:
        mutable_state: DataGenState = dict(state)
        mutable_state.update(route_node(mutable_state))
        if select_mode_edge(mutable_state) == "acquire_user_data":
            mutable_state.update(acquire_user_data_node(mutable_state))
        elif select_mode_edge(mutable_state) == "acquire_hf_data":
            mutable_state.update(acquire_hf_data_node(mutable_state))
        else:
            mutable_state.update(acquire_web_data_node(mutable_state))

        if select_curation_edge(mutable_state) == "structure_data":
            mutable_state.update(handoff_structure_data_node(mutable_state))
        else:
            mutable_state.update(handoff_validate_hf_node(mutable_state))
        return mutable_state
