from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


def search_web_sources(web_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Real web search using Tavily.
    LLM is mocked, but search is real.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Mode C real web search requires TAVILY_API_KEY. "
            "Set it with: export TAVILY_API_KEY='...'"
        )

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    queries = web_plan.get("search_queries", [])
    max_results = int(web_plan.get("max_search_results_per_query", 6))
    avoid_domains = web_plan.get("avoid_domains", [])

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for query in queries:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        for item in response.get("results", []):
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            if _blocked(url, avoid_domains):
                continue

            seen_urls.add(url)
            results.append(
                {
                    "source": "web_search",
                    "provider": "tavily",
                    "query": query,
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "title": str(item.get("title", "")).strip(),
                    "snippet": str(item.get("content", "")).strip(),
                    "provider_score": item.get("score"),
                }
            )

    return results


def _blocked(url: str, avoid_domains: list[str]) -> bool:
    domain = urlparse(url).netloc.lower()
    for blocked in avoid_domains:
        blocked = str(blocked).lower().strip()
        if blocked and blocked in domain:
            return True
    return False