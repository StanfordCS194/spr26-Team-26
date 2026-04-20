from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "text_classification",
    "text_to_text",
    "token_classification",
    "ranking",
    "regression",
    "image_classification",
    "multimodal",
    "other",
]


class TargetSchema(BaseModel):
    example_unit: str = Field(..., min_length=1)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    label_space: list[str] = Field(default_factory=list)


class DataRequirements(BaseModel):
    preferred_sources: list[str] = Field(default_factory=list)
    query_keywords: list[str] = Field(default_factory=list)
    min_examples: int = Field(default=0, ge=0)
    languages: list[str] = Field(default_factory=list)


class Constraints(BaseModel):
    allow_scraping: bool = True
    allow_api_sources: bool = True
    allow_synthetic: bool = False


class AcquisitionSpec(BaseModel):
    task_name: str = Field(..., min_length=1)
    task_type: TaskType = "other"
    target_schema: TargetSchema
    data_requirements: DataRequirements = Field(default_factory=DataRequirements)
    constraints: Constraints = Field(default_factory=Constraints)
    explicit_sources: list[str] = Field(default_factory=list, description="Direct source hints (URLs, domains, dataset ids).")


class RetrievalModeDecision(BaseModel):
    mode: Literal["no_data", "pointed_source", "hybrid"]
    reasoning: list[str] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    strategy_summary: str
    priority_order: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    safety_checks: list[str] = Field(default_factory=list)


class SourceCandidate(BaseModel):
    source_id: str
    title: str
    url: str
    source_type: str
    collection_method: Literal["api", "download", "html_fetch", "search_portal"]
    expected_artifact_type: str
    relevance_score: float
    structure_score: float
    collection_ease_score: float
    risk_score: float
    total_score: float
    notes: list[str] = Field(default_factory=list)


class CollectedArtifact(BaseModel):
    artifact_id: str
    source_id: str
    source_url: str
    artifact_url: str
    local_path: str
    content_type: str
    size_bytes: int
    retrieved_at: str
    relevance_score: float
    matched_terms: list[str] = Field(default_factory=list)
    is_relevant: bool = False
    is_reasonable: bool = False
    reasonableness_reason: str = "unknown"
    reasonableness_details: list[str] = Field(default_factory=list)
    status: Literal["downloaded", "filtered_out", "failed"] = "downloaded"
    notes: list[str] = Field(default_factory=list)


class CollectionSummary(BaseModel):
    attempted_sources: int = 0
    attempted_artifacts: int = 0
    downloaded_artifacts: int = 0
    relevant_artifacts: int = 0
    reasonable_artifacts: int = 0
    failed_artifacts: int = 0
    output_dir: str


class RetrievalReport(BaseModel):
    task_name: str
    retrieval_mode: str
    plan: RetrievalPlan
    candidates: list[SourceCandidate]
    ranked_source_ids: list[str]
    collected_artifacts: list[CollectedArtifact] = Field(default_factory=list)
    collection_summary: CollectionSummary | None = None
    human_readable_dir: str | None = None
    human_readable_summary: str | None = None
    concerns: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
