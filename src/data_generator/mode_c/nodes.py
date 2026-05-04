from __future__ import annotations

from src.data_generator.mode_c.crawler import crawl_and_extract_pages
from src.data_generator.mode_c.mock_llm import mock_plan_web_acquisition
from src.data_generator.mode_c.report import build_web_human_readable_report
from src.data_generator.mode_c.search import search_web_sources
from src.types import DataGenState


def plan_web_acquisition_node(state: DataGenState) -> dict:
    """
    Mock LLM planner.
    Later this function becomes the real LLM call.
    """
    web_plan = mock_plan_web_acquisition(state["config"])
    return {"web_plan": web_plan}


def search_web_sources_node(state: DataGenState) -> dict:
    """
    Real web search.
    """
    web_plan = state.get("web_plan")
    if not web_plan:
        raise RuntimeError("search_web_sources_node requires web_plan.")

    results = search_web_sources(web_plan)
    if not results:
        raise RuntimeError("Mode C search returned no results.")

    return {"web_search_results": results}


def crawl_web_pages_node(state: DataGenState) -> dict:
    """
    Real crawl + real text extraction.
    """
    web_plan = state.get("web_plan")
    search_results = state.get("web_search_results", [])

    if not web_plan:
        raise RuntimeError("crawl_web_pages_node requires web_plan.")
    if not search_results:
        raise RuntimeError("crawl_web_pages_node requires web_search_results.")

    pages = crawl_and_extract_pages(search_results, web_plan)
    if not pages:
        raise RuntimeError("Mode C crawling produced no usable extracted pages.")

    return {"web_pages": pages}


def aggregate_web_sources_node(state: DataGenState) -> dict:
    """
    Package real extracted web pages as RawData.
    No structuring into training examples happens here.
    """
    web_plan = state.get("web_plan")
    search_results = state.get("web_search_results", [])
    pages = state.get("web_pages", [])

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
    }