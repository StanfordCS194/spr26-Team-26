from __future__ import annotations

from .models import AcquisitionSpec, RetrievalModeDecision, RetrievalPlan


def build_retrieval_plan(
    spec: AcquisitionSpec,
    decision: RetrievalModeDecision,
    keyword_bank: list[str],
) -> RetrievalPlan:
    priority_order: list[str] = []

    # Search-before-scrape policy.
    priority_order.extend(["existing_datasets", "public_apis", "structured_repositories", "web_pages"])
    if spec.constraints.allow_scraping:
        priority_order.append("targeted_scraping")

    search_queries = _build_search_queries(spec, keyword_bank)

    strategy_summary = (
        "Use structured datasets/APIs first, then repository archives, then public web pages; "
        "reserve scraping for gaps only."
    )

    safety_checks = [
        "Respect source license and terms before downloading artifacts.",
        "Check robots.txt and site policies before scraping.",
        "Use rate limits and retries with backoff for automated requests.",
        "Store raw artifacts with provenance metadata (source, time, method).",
        "Avoid login-protected or anti-bot bypass collection in MVP.",
    ]

    if not spec.constraints.allow_scraping:
        safety_checks.append("Scraping disabled by spec constraints.")

    if not spec.constraints.allow_api_sources:
        priority_order = [item for item in priority_order if item != "public_apis"]
        safety_checks.append("API sources disabled by spec constraints.")

    if decision.mode == "pointed_source":
        strategy_summary = "Start from spec-pointed sources first, then expand only if coverage is insufficient."
    if decision.mode == "hybrid":
        strategy_summary = "Use pointed sources as anchor data, then add complementary external sources to hit volume/coverage targets."

    return RetrievalPlan(
        strategy_summary=strategy_summary,
        priority_order=priority_order,
        search_queries=search_queries,
        safety_checks=safety_checks,
    )


def _build_search_queries(spec: AcquisitionSpec, keyword_bank: list[str]) -> list[str]:
    joined = " ".join(keyword_bank[:6]).strip()
    queries: list[str] = []

    if joined:
        queries.append(f"{joined} dataset")
        queries.append(f"{joined} corpus")
        queries.append(f"{joined} labeled data")

    for language in spec.data_requirements.languages:
        language = language.strip().lower()
        if language:
            queries.append(f"{joined} dataset language {language}")

    # Keep deterministic and compact.
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped[:12]
