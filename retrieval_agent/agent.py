from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypedDict

from .collectors import CollectorConfig, collect_raw_artifacts
from .human_readable import generate_human_readable_bundle
from .models import (
    AcquisitionSpec,
    CollectedArtifact,
    CollectionSummary,
    RetrievalModeDecision,
    RetrievalPlan,
    RetrievalReport,
    SourceCandidate,
)
from .planner import build_retrieval_plan
from .report import build_report
from .reranker import LLMRerankResult, apply_llm_hybrid_rerank
from .source_finder import find_source_candidates
from .source_ranker import rank_sources
from .spec_parser import build_keyword_bank, decide_retrieval_mode, parse_acquisition_spec

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = START = StateGraph = None


class RetrievalAgentState(TypedDict, total=False):
    spec_payload: dict[str, Any]
    raw_output_dir: str
    spec: dict[str, Any]
    decision: dict[str, Any]
    keyword_bank: list[str]
    plan: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    collected_artifacts: list[dict[str, Any]]
    collection_summary: dict[str, Any]
    report: dict[str, Any]
    planner_backend: str
    rerank_backend: str
    warnings: list[str]


class RetrievalAgent:
    """LangGraph-based retrieval agent with optional LangChain LLM planning."""

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        collector_config: CollectorConfig | None = None,
        enable_llm_rerank: bool | None = None,
        llm_rerank_top_k: int | None = None,
        llm_rerank_alpha: float | None = None,
    ) -> None:
        self.model = model or os.environ.get("RETRIEVAL_AGENT_MODEL", "qwen2.5:7b-instruct")
        self.api_base = api_base or os.environ.get("RETRIEVAL_AGENT_API_BASE", "http://localhost:11434")
        self.temperature = temperature
        self.collector_config = collector_config or CollectorConfig()
        self.enable_llm_rerank = (
            enable_llm_rerank
            if enable_llm_rerank is not None
            else os.environ.get("RETRIEVAL_AGENT_ENABLE_LLM_RERANK", "false").lower() == "true"
        )
        self.llm_rerank_top_k = llm_rerank_top_k or int(os.environ.get("RETRIEVAL_AGENT_LLM_RERANK_TOP_K", "12"))
        self.llm_rerank_alpha = llm_rerank_alpha if llm_rerank_alpha is not None else float(
            os.environ.get("RETRIEVAL_AGENT_LLM_RERANK_ALPHA", "0.4")
        )
        self.graph = self._build_graph() if StateGraph is not None else None

    def run(
        self,
        spec_payload: dict[str, Any],
        raw_output_dir: str | Path = "raw",
        human_output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.graph is None:
            report_payload = self._run_fallback(spec_payload, raw_output_dir=raw_output_dir)
            return self._attach_human_readable_output(report_payload, human_output_dir)
        state = self.graph.invoke(
            {
                "spec_payload": spec_payload,
                "raw_output_dir": str(raw_output_dir),
                "warnings": [],
                "planner_backend": "heuristic",
                "rerank_backend": "disabled",
            }
        )
        return self._attach_human_readable_output(state["report"], human_output_dir)

    def _run_fallback(self, spec_payload: dict[str, Any], raw_output_dir: str | Path) -> dict[str, Any]:
        spec = parse_acquisition_spec(spec_payload)
        decision = decide_retrieval_mode(spec)
        keyword_bank = build_keyword_bank(spec)
        plan = build_retrieval_plan(spec, decision, keyword_bank)
        candidates = find_source_candidates(spec, plan)
        ranked = rank_sources(candidates)
        rerank_backend = "disabled"
        if self.enable_llm_rerank:
            llm = self._get_reranker_llm()
            if llm is not None:
                try:
                    ranked = apply_llm_hybrid_rerank(
                        spec=spec,
                        ranked_candidates=ranked,
                        llm=llm,
                        top_k=min(self.llm_rerank_top_k, len(ranked)),
                        blend_alpha=self.llm_rerank_alpha,
                    )
                    rerank_backend = "langchain+ollama_hybrid"
                except Exception:
                    rerank_backend = "heuristic"
            else:
                rerank_backend = "heuristic"
        artifacts, collection_summary = collect_raw_artifacts(
            spec=spec,
            ranked_candidates=ranked,
            keyword_bank=keyword_bank,
            output_dir=Path(raw_output_dir),
            config=self.collector_config,
        )
        report = build_report(
            spec.task_name,
            decision.mode,
            plan,
            ranked,
            collected_artifacts=artifacts,
            collection_summary=collection_summary,
        )
        report.concerns.append("LangGraph not installed in current interpreter; used sequential fallback.")
        if self.enable_llm_rerank and rerank_backend != "langchain+ollama_hybrid":
            report.concerns.append("LLM rerank fallback: deterministic ranking used.")
        return report.model_dump()

    def _attach_human_readable_output(
        self,
        report_payload: dict[str, Any],
        human_output_dir: str | Path | None,
    ) -> dict[str, Any]:
        report = RetrievalReport.model_validate(report_payload)
        target_dir = Path(human_output_dir) if human_output_dir is not None else Path("human_readable")
        bundle_dir, summary_path = generate_human_readable_bundle(report, target_dir)
        report.human_readable_dir = str(bundle_dir)
        report.human_readable_summary = str(summary_path)
        return report.model_dump()

    def _build_graph(self):
        graph = StateGraph(RetrievalAgentState)
        graph.add_node("parse_spec", self._parse_spec_node)
        graph.add_node("decide_mode", self._decide_mode_node)
        graph.add_node("build_plan", self._build_plan_node)
        graph.add_node("find_candidates", self._find_candidates_node)
        graph.add_node("rank_candidates", self._rank_candidates_node)
        graph.add_node("collect_artifacts", self._collect_artifacts_node)
        graph.add_node("build_report", self._build_report_node)

        graph.add_edge(START, "parse_spec")
        graph.add_edge("parse_spec", "decide_mode")
        graph.add_edge("decide_mode", "build_plan")
        graph.add_edge("build_plan", "find_candidates")
        graph.add_edge("find_candidates", "rank_candidates")
        graph.add_edge("rank_candidates", "collect_artifacts")
        graph.add_edge("collect_artifacts", "build_report")
        graph.add_edge("build_report", END)
        return graph.compile()

    def _parse_spec_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = parse_acquisition_spec(state["spec_payload"])
        keyword_bank = build_keyword_bank(spec)
        return {
            "spec": spec.model_dump(),
            "keyword_bank": keyword_bank,
        }

    def _decide_mode_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        decision = decide_retrieval_mode(spec)
        return {"decision": decision.model_dump()}

    def _build_plan_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        decision = RetrievalModeDecision.model_validate(state["decision"])
        keyword_bank = state.get("keyword_bank", [])

        heuristic_plan = build_retrieval_plan(spec, decision, keyword_bank)
        llm = self._get_planner_llm()
        if llm is None:
            return {"plan": heuristic_plan.model_dump(), "planner_backend": "heuristic"}

        prompt = (
            "You are a data acquisition planner.\n"
            "Return a RetrievalPlan for external raw-data discovery.\n"
            "Prioritize: existing_datasets, public_apis, structured_repositories, web_pages, targeted_scraping.\n"
            "Use scraping as a last resort.\n"
            "Do not include labeling/cleaning/structuring steps.\n\n"
            f"AcquisitionSpec:\n{json.dumps(spec.model_dump(), indent=2)}\n\n"
            f"Retrieval decision:\n{json.dumps(decision.model_dump(), indent=2)}\n\n"
            f"Keyword bank:\n{json.dumps(keyword_bank, indent=2)}"
        )

        try:
            result = llm.invoke(prompt)
            llm_plan = result if isinstance(result, RetrievalPlan) else RetrievalPlan.model_validate(result)
            constrained_plan = self._apply_constraints_to_plan(spec, llm_plan)
            return {"plan": constrained_plan.model_dump(), "planner_backend": "langchain+ollama"}
        except Exception:
            warnings = state.get("warnings", []) + ["LLM planner failed; used heuristic planner."]
            return {
                "plan": heuristic_plan.model_dump(),
                "planner_backend": "heuristic",
                "warnings": warnings,
            }

    def _find_candidates_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        plan = RetrievalPlan.model_validate(state["plan"])
        candidates = find_source_candidates(spec, plan)
        return {"candidates": [candidate.model_dump() for candidate in candidates]}

    def _rank_candidates_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        candidates = [SourceCandidate.model_validate(item) for item in state.get("candidates", [])]
        ranked = rank_sources(candidates)
        rerank_backend = "disabled"

        if self.enable_llm_rerank and ranked:
            llm = self._get_reranker_llm()
            if llm is None:
                rerank_backend = "heuristic"
            else:
                try:
                    ranked = apply_llm_hybrid_rerank(
                        spec=spec,
                        ranked_candidates=ranked,
                        llm=llm,
                        top_k=min(self.llm_rerank_top_k, len(ranked)),
                        blend_alpha=self.llm_rerank_alpha,
                    )
                    rerank_backend = "langchain+ollama_hybrid"
                except Exception:
                    warnings = state.get("warnings", []) + ["LLM rerank failed; deterministic ranking retained."]
                    return {
                        "ranked_candidates": [candidate.model_dump() for candidate in ranked],
                        "rerank_backend": "heuristic",
                        "warnings": warnings,
                    }

        return {
            "ranked_candidates": [candidate.model_dump() for candidate in ranked],
            "rerank_backend": rerank_backend,
        }

    def _collect_artifacts_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        ranked = [SourceCandidate.model_validate(item) for item in state.get("ranked_candidates", [])]
        keyword_bank = state.get("keyword_bank", [])
        raw_output_dir = Path(state.get("raw_output_dir", "raw"))

        artifacts, summary = collect_raw_artifacts(
            spec=spec,
            ranked_candidates=ranked,
            keyword_bank=keyword_bank,
            output_dir=raw_output_dir,
            config=self.collector_config,
        )
        return {
            "collected_artifacts": [artifact.model_dump() for artifact in artifacts],
            "collection_summary": summary.model_dump(),
        }

    def _build_report_node(self, state: RetrievalAgentState) -> RetrievalAgentState:
        spec = AcquisitionSpec.model_validate(state["spec"])
        decision = RetrievalModeDecision.model_validate(state["decision"])
        plan = RetrievalPlan.model_validate(state["plan"])
        ranked = [SourceCandidate.model_validate(item) for item in state.get("ranked_candidates", [])]
        artifacts = [CollectedArtifact.model_validate(item) for item in state.get("collected_artifacts", [])]
        summary = (
            CollectionSummary.model_validate(state["collection_summary"])
            if state.get("collection_summary")
            else None
        )

        report = build_report(
            spec.task_name,
            decision.mode,
            plan,
            ranked,
            collected_artifacts=artifacts,
            collection_summary=summary,
        )
        if summary and summary.relevant_artifacts == 0:
            report.concerns.append("No relevant artifacts were identified from downloaded content.")
        if summary and summary.reasonable_artifacts == 0:
            report.concerns.append("No artifacts passed basic reasonableness checks.")
        if state.get("planner_backend") == "heuristic":
            report.concerns.append("Planner backend: heuristic fallback.")
        if self.enable_llm_rerank and state.get("rerank_backend") != "langchain+ollama_hybrid":
            report.concerns.append("LLM rerank fallback: deterministic ranking used.")
        if state.get("warnings"):
            report.concerns.extend(state["warnings"])
        return {"report": report.model_dump()}

    def _get_planner_llm(self):
        if ChatOllama is None:
            return None
        try:
            llm = ChatOllama(
                model=self.model,
                base_url=self.api_base,
                temperature=self.temperature,
            )
            return llm.with_structured_output(RetrievalPlan)
        except Exception:
            return None

    def _get_reranker_llm(self):
        if ChatOllama is None:
            return None
        try:
            llm = ChatOllama(
                model=self.model,
                base_url=self.api_base,
                temperature=self.temperature,
            )
            return llm.with_structured_output(LLMRerankResult)
        except Exception:
            return None

    def _apply_constraints_to_plan(self, spec: AcquisitionSpec, plan: RetrievalPlan) -> RetrievalPlan:
        priority_order = list(plan.priority_order)
        safety_checks = list(plan.safety_checks)

        if not spec.constraints.allow_api_sources:
            priority_order = [item for item in priority_order if item != "public_apis"]
            if "API sources disabled by spec constraints." not in safety_checks:
                safety_checks.append("API sources disabled by spec constraints.")

        if spec.constraints.allow_scraping and "targeted_scraping" not in priority_order:
            priority_order.append("targeted_scraping")
        if not spec.constraints.allow_scraping:
            priority_order = [item for item in priority_order if item != "targeted_scraping"]
            if "Scraping disabled by spec constraints." not in safety_checks:
                safety_checks.append("Scraping disabled by spec constraints.")

        return RetrievalPlan(
            strategy_summary=plan.strategy_summary,
            priority_order=priority_order,
            search_queries=plan.search_queries[:12],
            safety_checks=safety_checks,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangGraph retrieval-agent from an acquisition spec JSON file.")
    parser.add_argument("spec_json", type=Path, help="Path to input acquisition spec JSON.")
    parser.add_argument("output_json", type=Path, help="Path to output retrieval report JSON.")
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=Path("raw"),
        help="Directory where downloaded raw artifacts are saved.",
    )
    parser.add_argument(
        "--human-output-dir",
        type=Path,
        default=None,
        help="Directory where human-readable summaries and previews are saved.",
    )
    parser.add_argument(
        "--enable-llm-rerank",
        action="store_true",
        help="Enable hybrid LLM reranking on top-K deterministically-ranked candidates.",
    )
    parser.add_argument(
        "--llm-rerank-top-k",
        type=int,
        default=12,
        help="Number of top candidates to rerank with a single LLM call.",
    )
    parser.add_argument(
        "--llm-rerank-alpha",
        type=float,
        default=0.4,
        help="Blend weight for LLM score in final score. final=(1-alpha)*det + alpha*llm",
    )
    args = parser.parse_args()

    spec_payload = json.loads(args.spec_json.read_text(encoding="utf-8"))
    result = RetrievalAgent(
        enable_llm_rerank=args.enable_llm_rerank,
        llm_rerank_top_k=max(args.llm_rerank_top_k, 0),
        llm_rerank_alpha=min(max(args.llm_rerank_alpha, 0.0), 1.0),
    ).run(
        spec_payload,
        raw_output_dir=args.raw_output_dir,
        human_output_dir=args.human_output_dir,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
