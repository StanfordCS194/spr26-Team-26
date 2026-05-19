"""Mode C acquisition: no user data, so synthesize chat/SFT data.

The public ``acquire_web_data`` name is kept for compatibility with the
original graph/spec wording. The implementation follows the newer Mode C
shape: infer a data schema, generate synthetic examples, standardize them, and
return validation signals alongside the raw records.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.data_generator.synthetic import (
    DEFAULT_SYNTHETIC_EXAMPLES,
    DEFAULT_TEACHER_MODEL,
    SyntheticGenerationResult,
    build_mode_c_dataset,
    determine_data_schema,
    generate_synthetic_data,
    infer_schema_without_teacher,
    morph_to_standard,
    plan_synthetic_generation,
    scrape_web,
    validate_synthetic_records,
)
from src.types import OrchestrationConfig, RawData


def acquire_synthetic_dataset(
    config: OrchestrationConfig | Mapping[str, Any],
    *,
    teacher_client: Any = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    n_examples: int | None = None,
) -> SyntheticGenerationResult:
    """Build the full Mode C synthetic dataset bundle."""
    return build_mode_c_dataset(
        config,
        teacher_client=teacher_client,
        teacher_model=teacher_model,
        n_examples=n_examples,
    )


def acquire_web_data(
    query: str,
    config: Mapping[str, Any] | None = None,
    *,
    teacher_client: Any = None,
    n_examples: int | None = None,
) -> RawData:
    """Compatibility wrapper for the Mode C no-data acquisition path."""
    mode_config = _config_with_prompt(query, config)
    result = acquire_synthetic_dataset(
        mode_config,
        teacher_client=teacher_client,
        n_examples=n_examples,
    )
    return result.raw_data


def _config_with_prompt(
    query: str,
    config: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    clean_query = " ".join(str(query or "generic ML task").split()) or "generic ML task"
    if config is None:
        return {
            "data": False,
            "prompt": clean_query,
            "compute_budget": 0.0,
            "training_procedure": {
                "task_type": "custom",
                "data_format": "chat JSONL",
                "training_type": "SFT",
                "base_model": None,
                "hyperparameters": {
                    "synthetic_examples": DEFAULT_SYNTHETIC_EXAMPLES,
                },
                "notes": "Mode C synthetic fallback",
            },
        }

    merged = dict(config)
    merged["prompt"] = clean_query or str(config.get("prompt") or "generic ML task")
    return merged


__all__ = [
    "DEFAULT_SYNTHETIC_EXAMPLES",
    "DEFAULT_TEACHER_MODEL",
    "SyntheticGenerationResult",
    "acquire_synthetic_dataset",
    "acquire_web_data",
    "build_mode_c_dataset",
    "determine_data_schema",
    "generate_synthetic_data",
    "infer_schema_without_teacher",
    "morph_to_standard",
    "plan_synthetic_generation",
    "scrape_web",
    "validate_synthetic_records",
]
