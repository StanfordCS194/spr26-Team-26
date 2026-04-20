from __future__ import annotations

from .models import CollectedArtifact, CollectionSummary, RetrievalPlan, RetrievalReport, SourceCandidate


def build_report(
    task_name: str,
    retrieval_mode: str,
    plan: RetrievalPlan,
    ranked_candidates: list[SourceCandidate],
    collected_artifacts: list[CollectedArtifact] | None = None,
    collection_summary: CollectionSummary | None = None,
) -> RetrievalReport:
    concerns: list[str] = []

    if not ranked_candidates:
        concerns.append("No source candidates discovered from current spec.")

    if any(c.risk_score >= 0.7 for c in ranked_candidates[:5]):
        concerns.append("Top candidates include elevated legal/collection risk; verify licensing before collection.")

    artifacts = collected_artifacts or []
    summary = collection_summary
    confidence = "high" if len(ranked_candidates) >= 5 else "medium"
    if not ranked_candidates:
        confidence = "low"
    if summary is not None and summary.relevant_artifacts == 0:
        confidence = "low"
    if summary is not None and summary.reasonable_artifacts == 0:
        confidence = "low"

    return RetrievalReport(
        task_name=task_name,
        retrieval_mode=retrieval_mode,
        plan=plan,
        candidates=ranked_candidates,
        ranked_source_ids=[candidate.source_id for candidate in ranked_candidates],
        collected_artifacts=artifacts,
        collection_summary=summary,
        concerns=concerns,
        confidence=confidence,
    )
