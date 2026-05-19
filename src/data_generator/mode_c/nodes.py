from __future__ import annotations

import os

from src.data_generator.mode_c.offline import mode_c_offline
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
        return _mode_c_synthetic_fallback_output(
            state["config"],
            backend=backend,
            reason=state.get(
                "web_acquisition_error",
                "web acquisition unavailable",
            ),
        )

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

    unstructured = {
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

    structuring_mode = _web_structuring_mode()
    if structuring_mode == "off":
        return unstructured

    structured = structure_web_sources_for_sft(state["config"], pages)
    if structured.validation_report["passed"] or structuring_mode == "required":
        structured.raw_data["human_readable"] = report
        structured.raw_data["format_meta"]["mode_c_backend"] = backend
        structured.raw_data["format_meta"]["web_plan"] = web_plan
        structured.raw_data["format_meta"]["num_search_results"] = len(search_results)
        structured.raw_data["format_meta"]["num_pages_crawled"] = len(pages)
        return {
            "schema": structured.schema,
            "raw_data": structured.raw_data,
            "validation_report": structured.validation_report,
            "human_readable": report,
        }

    return _mode_c_synthetic_fallback_output(
        state["config"],
        backend=backend,
        reason="Mode C web structuring produced no trainable records.",
        web_report=report,
        web_plan=web_plan,
        search_results=search_results,
        pages=pages,
        structuring_issues=structured.validation_report.get("issues", []),
    )


def _mode_c_synthetic_fallback_output(
    config,
    *,
    backend: str,
    reason: str | None,
    web_report: str | None = None,
    web_plan=None,
    search_results=None,
    pages=None,
    structuring_issues=None,
) -> dict:
    result = build_mode_c_dataset(config)
    raw_data = result.raw_data
    format_meta = raw_data.setdefault("format_meta", {})
    format_meta["mode_c_fallback"] = "synthetic"
    format_meta["mode_c_backend"] = backend
    format_meta["web_acquisition_error"] = reason
    if web_plan is not None:
        format_meta["web_plan"] = web_plan
    if search_results is not None:
        format_meta["num_search_results"] = len(search_results)
        search_urls = [item.get("url") for item in search_results if item.get("url")]
        if search_urls:
            format_meta["web_search_urls"] = search_urls
    if pages is not None:
        format_meta["num_pages_crawled"] = len(pages)
        source_urls = [page.get("url") for page in pages if page.get("url")]
        if source_urls:
            format_meta["web_source_urls"] = source_urls
    if structuring_issues:
        format_meta["web_structuring_issues"] = list(structuring_issues)

    report = (
        "Mode C Synthetic Fallback Report\n"
        f"Reason: {reason or 'web acquisition unavailable'}\n"
        f"Records generated: {len(raw_data.get('records', []))}"
    )
    if web_report:
        report = (
            f"{report}\n\n"
            "Web acquisition report retained for provenance:\n"
            f"{web_report}"
        )
    raw_data["human_readable"] = report
    return {
        "schema": result.schema,
        "raw_data": raw_data,
        "validation_report": result.validation_report,
        "human_readable": report,
        "mode_c_fallback": "synthetic",
        "web_acquisition_error": reason,
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
    if mode_c_offline():
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


def _web_structuring_mode() -> str:
    selected = os.getenv("DATA_GENERATOR_WEB_STRUCTURING", "auto").strip().lower()
    if selected in {"1", "true", "yes", "on"}:
        return "required"
    if selected in {"0", "false", "no", "off"}:
        return "off"
    if selected not in {"auto", "required"}:
        raise ValueError(
            "DATA_GENERATOR_WEB_STRUCTURING must be one of: auto, required, off"
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


def structure_web_sources_for_sft(config, pages):
    from src.data_generator.mode_c.structuring import structure_web_sources_for_sft as _impl

    return _impl(config, pages)
