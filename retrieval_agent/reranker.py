from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .models import AcquisitionSpec, SourceCandidate


class LLMRerankItem(BaseModel):
    source_id: str
    llm_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(default="")


class LLMRerankResult(BaseModel):
    items: list[LLMRerankItem] = Field(default_factory=list)


def apply_llm_hybrid_rerank(
    spec: AcquisitionSpec,
    ranked_candidates: list[SourceCandidate],
    llm: Any,
    top_k: int,
    blend_alpha: float,
) -> list[SourceCandidate]:
    if not ranked_candidates or top_k <= 0 or llm is None:
        return ranked_candidates

    head = ranked_candidates[:top_k]
    tail = ranked_candidates[top_k:]

    payload = [
        {
            "source_id": c.source_id,
            "title": c.title,
            "url": c.url,
            "source_type": c.source_type,
            "collection_method": c.collection_method,
            "expected_artifact_type": c.expected_artifact_type,
            "deterministic_score": c.total_score,
            "notes": c.notes[:3],
        }
        for c in head
    ]

    prompt = (
        "You are reranking candidate raw-data sources for an ML data acquisition pipeline.\n"
        "Score each candidate from 0.0 to 1.0 for relevance and likely usefulness to this spec.\n"
        "Higher score means better acquisition priority.\n"
        "Return one score per source_id provided.\n"
        "Do not invent new source_ids.\n\n"
        f"AcquisitionSpec:\n{json.dumps(spec.model_dump(), indent=2)}\n\n"
        f"Candidates:\n{json.dumps(payload, indent=2)}"
    )

    result = llm.invoke(prompt)
    parsed = result if isinstance(result, LLMRerankResult) else LLMRerankResult.model_validate(result)

    llm_scores = {item.source_id: item.llm_score for item in parsed.items}
    rationales = {item.source_id: item.rationale for item in parsed.items}

    blended: list[SourceCandidate] = []
    for candidate in head:
        llm_score = llm_scores.get(candidate.source_id)
        if llm_score is None:
            blended.append(candidate)
            continue

        old_score = candidate.total_score
        new_score = (1.0 - blend_alpha) * old_score + blend_alpha * llm_score
        candidate.total_score = round(max(min(new_score, 1.0), 0.0), 4)

        rationale = rationales.get(candidate.source_id, "")
        candidate.notes.append(f"llm_rerank_score={llm_score:.4f}")
        if rationale:
            candidate.notes.append(f"llm_rationale={rationale[:220]}")
        blended.append(candidate)

    blended.sort(key=lambda c: c.total_score, reverse=True)
    return blended + tail
