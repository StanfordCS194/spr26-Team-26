from __future__ import annotations

from typing import Any

from src.data_generator.artifacts import save_subagent2_artifacts
from src.data_generator.edges import select_mode_edge
from src.data_generator.mode_c.nodes import (
    aggregate_web_sources_node,
    crawl_web_pages_node,
    plan_web_acquisition_node,
    search_web_sources_node,
)
from src.data_generator.nodes import (
    acquire_hf_data_node,
    acquire_user_data_node,
    build_handoff_node,
    route_node,
)
from src.types import DataGenState, OrchestrationConfig

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = START = StateGraph = None


def build_data_generator_graph():
    """
    First sub-agent graph.

    Mode A: acquire local user data.
    Mode B: acquire explicit HF dataset.
    Mode C: mock LLM planner + real web search/crawl/extraction.
    """
    if StateGraph is None:
        return _FallbackDataGeneratorGraph()

    graph = StateGraph(DataGenState)

    graph.add_node("route_node", route_node)
    graph.add_node("acquire_user_data", acquire_user_data_node)
    graph.add_node("acquire_hf_data", acquire_hf_data_node)

    graph.add_node("plan_web_acquisition", plan_web_acquisition_node)
    graph.add_node("search_web_sources", search_web_sources_node)
    graph.add_node("crawl_web_pages", crawl_web_pages_node)
    graph.add_node("aggregate_web_sources", aggregate_web_sources_node)

    graph.add_node("build_handoff", build_handoff_node)

    graph.add_edge(START, "route_node")

    graph.add_conditional_edges(
        "route_node",
        select_mode_edge,
        {
            "acquire_user_data": "acquire_user_data",
            "acquire_hf_data": "acquire_hf_data",
            "plan_web_acquisition": "plan_web_acquisition",
        },
    )

    graph.add_edge("acquire_user_data", "build_handoff")
    graph.add_edge("acquire_hf_data", "build_handoff")

    graph.add_edge("plan_web_acquisition", "search_web_sources")
    graph.add_edge("search_web_sources", "crawl_web_pages")
    graph.add_edge("crawl_web_pages", "aggregate_web_sources")
    graph.add_edge("aggregate_web_sources", "build_handoff")

    graph.add_edge("build_handoff", END)

    return graph.compile()


def invoke_data_generator_graph(config: OrchestrationConfig, data_path: str | None) -> dict[str, Any]:
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

        # Mode C fields
        "web_plan": None,
        "web_search_results": [],
        "web_pages": [],
        "human_readable": None,
        "mode_c_fallback": None,
        "mode_c_backend": None,
        "web_acquisition_error": None,
    }

    final_state = graph.invoke(initial_state)
    handoff = final_state.get("handoff")
    if not handoff:
        raise RuntimeError("Data generator did not produce a handoff payload.")
    try:
        save_subagent2_artifacts(handoff)
    except Exception:
        # Artifact persistence is for observability only; do not fail the main flow.
        pass
    return handoff


class _FallbackDataGeneratorGraph:
    def invoke(self, state: DataGenState) -> DataGenState:
        mutable_state: DataGenState = dict(state)

        mutable_state.update(route_node(mutable_state))
        next_node = select_mode_edge(mutable_state)

        if next_node == "acquire_user_data":
            mutable_state.update(acquire_user_data_node(mutable_state))
        elif next_node == "acquire_hf_data":
            mutable_state.update(acquire_hf_data_node(mutable_state))
        else:
            mutable_state.update(plan_web_acquisition_node(mutable_state))
            mutable_state.update(search_web_sources_node(mutable_state))
            mutable_state.update(crawl_web_pages_node(mutable_state))
            mutable_state.update(aggregate_web_sources_node(mutable_state))

        mutable_state.update(build_handoff_node(mutable_state))
        return mutable_state
