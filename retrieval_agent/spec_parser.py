from __future__ import annotations

import re
from typing import Any

from .models import AcquisitionSpec, RetrievalModeDecision


def parse_acquisition_spec(payload: dict[str, Any]) -> AcquisitionSpec:
    """Validate and normalize acquisition-spec payload from the manager agent."""
    return AcquisitionSpec.model_validate(payload)


def decide_retrieval_mode(spec: AcquisitionSpec) -> RetrievalModeDecision:
    explicit = [s for s in spec.explicit_sources if s.strip()]
    preferred = [s.lower().strip() for s in spec.data_requirements.preferred_sources]

    if explicit:
        mode = "pointed_source"
        reasoning = ["Spec contains explicit source hints."]
    elif preferred:
        mode = "pointed_source"
        reasoning = ["Spec lists preferred source families to follow."]
    else:
        mode = "no_data"
        reasoning = ["No explicit sources provided; discovery-first retrieval is required."]

    if explicit and spec.data_requirements.min_examples > 0:
        reasoning.append("Will augment pointed sources if they cannot satisfy minimum example target.")
        mode = "hybrid"

    return RetrievalModeDecision(mode=mode, reasoning=reasoning)


def build_keyword_bank(spec: AcquisitionSpec) -> list[str]:
    base = [spec.task_name, spec.task_type, spec.target_schema.example_unit]
    base.extend(spec.target_schema.label_space)
    base.extend(spec.data_requirements.query_keywords)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in base:
        for token in re.split(r"[,/|]", item):
            cleaned = " ".join(token.split()).strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
    return normalized
