from __future__ import annotations

import os

from src.data_generator.mode_c.synthetic import build_mode_c_dataset
from src.types import DataGenState

_MODE_C_BACKENDS = {"auto", "synthetic", "web"}


def plan_web_acquisition_node(state: DataGenState) -> dict:
    """
    Mock LLM planner.
    Later this function becomes the real LLM call.
    """
    backend = _mode_c_backend(state)
    if backend == "synthetic":
        return {
            "mode_c_backend": backend,
            "mode_c_fallback": "synthetic",
            "web_acquisition_error": "Mode C synthetic backend selected",
        }

    web_plan = mock_plan_web_acquisition(state["config"])
    return {"web_plan": web_plan, "mode_c_backend": backend}


def search_web_sources_node(state: DataGenState) -> dict:
    """
    Real web search.
    """
    backend = _mode_c_backend(state)
    if backend == "synthetic" or state.get("mode_c_fallback") == "synthetic":
        return {"web_search_results": []}

    web_plan = state.get("web_plan")
    if not web_plan:
        raise RuntimeError("search_web_sources_node requires web_plan.")

    try:
        results = search_web_sources(web_plan)
    except Exception as exc:
        return _mode_c_synthetic_fallback_state("web_search", exc, backend)

    if not results:
        return _mode_c_synthetic_fallback_state(
            "web_search",
            RuntimeError("Mode C search returned no results."),
            backend,
        )

    return {"web_search_results": results}


def crawl_web_pages_node(state: DataGenState) -> dict:
    """
    Real crawl + real text extraction.
    """
    backend = _mode_c_backend(state)
    web_plan = state.get("web_plan")
    search_results = state.get("web_search_results", [])

    if backend == "synthetic" or state.get("mode_c_fallback") == "synthetic":
        return {"web_pages": []}

    if not web_plan:
        raise RuntimeError("crawl_web_pages_node requires web_plan.")
    if not search_results:
        return _mode_c_synthetic_fallback_state(
            "web_crawl",
            RuntimeError("crawl_web_pages_node requires web_search_results."),
            backend,
        )

    try:
        pages = crawl_and_extract_pages(search_results, web_plan)
    except Exception as exc:
        return _mode_c_synthetic_fallback_state("web_crawl", exc, backend)

    if not pages:
        return _mode_c_synthetic_fallback_state(
            "web_crawl",
            RuntimeError("Mode C crawling produced no usable extracted pages."),
            backend,
        )

    return {"web_pages": pages}


def aggregate_web_sources_node(state: DataGenState) -> dict:
    """
    Package real extracted web pages as RawData.
    No structuring into training examples happens here.
    """
    web_plan = state.get("web_plan")
    search_results = state.get("web_search_results", [])
    pages = state.get("web_pages", [])
    backend = _mode_c_backend(state)

    if backend == "synthetic" or state.get("mode_c_fallback") == "synthetic":
        result = build_mode_c_dataset(state["config"])
        raw_data = result.raw_data
        raw_data["format_meta"]["mode_c_fallback"] = "synthetic"
        raw_data["format_meta"]["mode_c_backend"] = backend
        raw_data["format_meta"]["web_acquisition_error"] = state.get(
            "web_acquisition_error"
        )
        report = (
            "Mode C Synthetic Fallback Report\n"
            f"Reason: {state.get('web_acquisition_error', 'web acquisition unavailable')}\n"
            f"Records generated: {len(raw_data.get('records', []))}"
        )
        return {
            "schema": result.schema,
            "raw_data": raw_data,
            "validation_report": result.validation_report,
            "human_readable": report,
        }

    report = build_web_human_readable_report(
        web_plan=web_plan,
        search_results=search_results,
        pages=pages,
    )

    raw_data = {
        "records": pages,
        "human_readable": report,
        "format_meta": {
            "modality": "text",
            "file_type": "web_aggregated_sources",
            "encoding": "utf-8",
            "mode_c_backend": backend,
            "planner_backend": "mock_llm",
            "search_backend": "tavily",
            "web_plan": web_plan,
            "num_search_results": len(search_results),
            "num_pages_crawled": len(pages),
            "num_records": len(pages),
        },
    }

    return {
        "raw_data": raw_data,
        "human_readable": report,
        "validation_report": {
            "passed": False,
            "issues": [
                (
                    "Mode C web acquisition produced raw source records, not "
                    "trainable chat/SFT targets; run synthetic structuring before training."
                )
            ],
            "sample_accuracy_estimate": 0.0,
        },
    }


def _mode_c_synthetic_fallback_state(
    stage: str,
    exc: Exception,
    backend: str | None = None,
) -> dict:
    if (
        (backend or _mode_c_backend({})) == "web"
        or os.getenv("DATA_GENERATOR_WEB_STRICT") == "1"
    ):
        raise exc
    return {
        "mode_c_fallback": "synthetic",
        "web_acquisition_error": f"{stage}: {exc}",
        "web_search_results": [],
    }


def _mode_c_backend(state: DataGenState | dict) -> str:
    if os.getenv("DATA_GENERATOR_SYNTHETIC_OFFLINE") == "1":
        return "synthetic"

    selected = str(state.get("mode_c_backend") or "").strip().lower()
    if not selected:
        selected = os.getenv("DATA_GENERATOR_MODE_C_BACKEND", "").strip().lower()
    if not selected:
        selected = "web" if os.getenv("DATA_GENERATOR_WEB_STRICT") == "1" else "auto"
    if selected not in _MODE_C_BACKENDS:
        raise ValueError(
            "DATA_GENERATOR_MODE_C_BACKEND must be one of: auto, synthetic, web"
        )
    return selected


def mock_plan_web_acquisition(config):
    from src.data_generator.mode_c.mock_llm import mock_plan_web_acquisition as _impl

    return _impl(config)


def search_web_sources(web_plan):
    from src.data_generator.mode_c.search import search_web_sources as _impl

    return _impl(web_plan)


def crawl_and_extract_pages(search_results, web_plan):
    from src.data_generator.mode_c.crawler import crawl_and_extract_pages as _impl

    return _impl(search_results, web_plan)


def build_web_human_readable_report(*, web_plan, search_results, pages):
    from src.data_generator.mode_c.report import build_web_human_readable_report as _impl

    return _impl(web_plan=web_plan, search_results=search_results, pages=pages)
