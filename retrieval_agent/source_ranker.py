from __future__ import annotations

from .models import SourceCandidate


def rank_sources(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    ranked: list[SourceCandidate] = []
    for candidate in candidates:
        # Higher is better; risk penalizes score.
        total = (
            0.45 * candidate.relevance_score
            + 0.25 * candidate.structure_score
            + 0.20 * candidate.collection_ease_score
            - 0.15 * candidate.risk_score
        )
        candidate.total_score = round(max(total, 0.0), 4)
        ranked.append(candidate)

    ranked.sort(key=lambda c: c.total_score, reverse=True)
    return ranked
