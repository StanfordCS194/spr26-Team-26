from __future__ import annotations

from typing import Any


def build_web_human_readable_report(
    web_plan: dict[str, Any],
    search_results: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    max_preview_chars: int = 900,
) -> str:
    lines: list[str] = []

    lines.append("Mode C Web Acquisition Report")
    lines.append("=" * 40)
    lines.append(f"Planner backend: {web_plan.get('planner_backend', 'unknown')}")
    lines.append(f"Task summary: {web_plan.get('task_summary', '')}")
    lines.append(f"Data goal: {web_plan.get('data_goal', '')}")
    lines.append("")

    lines.append("Search queries:")
    for q in web_plan.get("search_queries", []):
        lines.append(f"- {q}")
    lines.append("")

    lines.append(f"Search results found: {len(search_results)}")
    lines.append(f"Pages successfully crawled/extracted: {len(pages)}")
    lines.append("")

    for idx, page in enumerate(pages, start=1):
        content = str(page.get("content", "")).strip()
        preview = content[:max_preview_chars]
        if len(content) > max_preview_chars:
            preview += "..."

        lines.append(f"[{idx}] {page.get('title') or 'Untitled'}")
        lines.append(f"URL: {page.get('url')}")
        lines.append(f"Domain: {page.get('domain')}")
        lines.append(f"Query: {page.get('query')}")
        lines.append(f"Extracted chars: {page.get('metadata', {}).get('content_chars')}")
        lines.append("Preview:")
        lines.append(preview)
        lines.append("-" * 40)

    return "\n".join(lines)