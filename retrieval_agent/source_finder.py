from __future__ import annotations

import hashlib
from urllib.parse import quote_plus

from .models import AcquisitionSpec, RetrievalPlan, SourceCandidate


def find_source_candidates(spec: AcquisitionSpec, plan: RetrievalPlan) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    preferred = {item.strip().lower() for item in spec.data_requirements.preferred_sources if item.strip()}
    use_all = not preferred

    for explicit in spec.explicit_sources:
        explicit = explicit.strip()
        if not explicit:
            continue
        candidates.append(
            SourceCandidate(
                source_id=_source_id("explicit", explicit),
                title=f"Explicit source: {explicit}",
                url=explicit,
                source_type="explicit",
                collection_method="download" if explicit.startswith("http") else "search_portal",
                expected_artifact_type="unknown",
                relevance_score=0.95,
                structure_score=0.6,
                collection_ease_score=0.7,
                risk_score=0.5,
                total_score=0.0,
                notes=["Directly provided by upstream spec."],
            )
        )

    for query in plan.search_queries:
        q = quote_plus(query)
        if use_all or "huggingface" in preferred:
            candidates.append(
                _candidate(
                    "hf",
                    "Hugging Face Datasets Search",
                    f"https://huggingface.co/datasets?search={q}",
                    "huggingface",
                    "search_portal",
                    "dataset_listing",
                    0.9,
                    0.9,
                    0.9,
                    0.35,
                )
            )
        if use_all or "kaggle" in preferred:
            candidates.append(
                _candidate(
                    "kaggle",
                    "Kaggle Datasets Search",
                    f"https://www.kaggle.com/datasets?search={q}",
                    "kaggle",
                    "search_portal",
                    "dataset_listing",
                    0.8,
                    0.8,
                    0.8,
                    0.45,
                )
            )
        if use_all or "github" in preferred or "public repositories" in preferred:
            candidates.append(
                _candidate(
                    "github",
                    "GitHub Dataset Search",
                    f"https://github.com/search?q={q}+dataset&type=repositories",
                    "github",
                    "search_portal",
                    "repository_listing",
                    0.75,
                    0.6,
                    0.85,
                    0.55,
                )
            )
        if use_all or "public web" in preferred or "web" in preferred:
            candidates.append(
                _candidate(
                    "web",
                    "General Web Discovery",
                    f"https://duckduckgo.com/?q={q}",
                    "web",
                    "search_portal",
                    "mixed",
                    0.6,
                    0.45,
                    0.6,
                    0.65,
                )
            )

    # Constraint-aware filtering.
    if not spec.constraints.allow_scraping:
        for c in candidates:
            if c.source_type == "web":
                c.notes.append("Scraping disabled; use for discovery only.")

    return _dedupe_candidates(candidates)


def _candidate(
    source_prefix: str,
    title: str,
    url: str,
    source_type: str,
    collection_method: str,
    artifact: str,
    relevance: float,
    structure: float,
    ease: float,
    risk: float,
) -> SourceCandidate:
    return SourceCandidate(
        source_id=_source_id(source_prefix, url),
        title=title,
        url=url,
        source_type=source_type,
        collection_method=collection_method,
        expected_artifact_type=artifact,
        relevance_score=relevance,
        structure_score=structure,
        collection_ease_score=ease,
        risk_score=risk,
        total_score=0.0,
        notes=[],
    )


def _source_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _dedupe_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    seen_urls: set[str] = set()
    deduped: list[SourceCandidate] = []
    for candidate in candidates:
        if candidate.url in seen_urls:
            continue
        seen_urls.add(candidate.url)
        deduped.append(candidate)
    return deduped
