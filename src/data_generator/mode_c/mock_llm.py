from __future__ import annotations

import os
from typing import Any

from src.data_generator.mode_c.mock_llms.base import build_base_plan
from src.data_generator.mode_c.mock_llms.broad_sources import build_broad_sources_plan
from src.data_generator.mode_c.mock_llms.mixed_sources import build_mixed_sources_plan
from src.types import OrchestrationConfig


def mock_plan_web_acquisition(config: OrchestrationConfig) -> dict[str, Any]:
    scenario = os.getenv("DATA_GENERATOR_MOCK_SCENARIO", "").strip().lower()

    if scenario == "mixed_sources":
        return build_mixed_sources_plan(config)

    if scenario == "broad_sources":
        return build_broad_sources_plan(config)

    return build_base_plan(config)