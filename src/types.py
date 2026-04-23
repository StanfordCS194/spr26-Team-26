"""
Shared TypedDicts and named types used across all agents.
Single source of truth for inter-agent data contracts.
See spec-site/content/spec.ts for the canonical definitions.
"""

from __future__ import annotations

from typing import TypedDict


# ─── ENUMS ───────────────────────────────────────────────────────────────────

class AgentName:
    MANAGER = "Manager"
    DATA_GEN = "DataGen"
    DECISION_ENGINE = "DecisionEngine"
    AUTORESEARCH = "AutoResearch"
    COST_MANAGER = "CostManager"
    TINKER_API = "TinkerAPI"


class LogLevel:
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class BudgetStatus:
    OK = "OK"
    WARNING = "WARNING"
    EXCEEDED = "EXCEEDED"


class JobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ─── NESTED TYPES ─────────────────────────────────────────────────────────────

class TrainingProcedure(TypedDict):
    task_type: str        # e.g. 'text-classification', 'seq2seq', 'custom'
    data_format: str      # expected format for training data
    training_type: str    # 'SFT', 'RL', or 'pre-train'
    base_model: str | None
    hyperparameters: dict
    notes: str


class OrchestrationConfig(TypedDict):
    data: bool
    prompt: str
    compute_budget: float
    training_procedure: TrainingProcedure


class TaskReasoning(TypedDict):
    task_type: str
    data_format: str
    training_type: str
    suggested_base_model: str | None
    hyperparameters: dict
    notes: str


class TaskAnalysis(TypedDict):
    task_type: str
    modality: str
    has_pretrained_base: bool
    eval_metric: str
    complexity: str  # 'low' | 'medium' | 'high'


class DataFormat(TypedDict):
    modality: str   # 'text' | 'image' | 'tabular'
    file_type: str
    encoding: str


class DataSchema(TypedDict):
    input_format: str
    output_format: str
    input_description: str
    output_description: str
    example_pair: dict


class RawData(TypedDict):
    records: list
    format_meta: DataFormat


class StandardDataset(TypedDict):
    path: str
    format: str   # 'jsonl' | 'csv' | 'parquet'
    train_size: int
    val_size: int
    test_size: int


class ValidationReport(TypedDict):
    passed: bool
    issues: list[str]
    sample_accuracy_estimate: float


class DatasetResult(TypedDict):
    dataset: StandardDataset
    mode_used: str   # 'A' | 'B' | 'C'
    quality_notes: str
    validation_report: ValidationReport


class HFCandidate(TypedDict):
    id: str
    name: str
    num_examples: int
    license: str
    task_categories: list[str]
    download_size: int


class LoRAConfig(TypedDict):
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]


class CostEstimate(TypedDict):
    estimated_usd: float
    estimated_gpu_hours: float
    estimated_time_min: int
    confidence: str  # 'low' | 'medium' | 'high'


class TrainingPlan(TypedDict):
    strategy: str           # 'fine-tune' | 'pre-train'
    base_model: str | None
    lora_config: LoRAConfig | None
    estimated_cost: float
    estimated_time_min: int
    training_script_path: str
    eval_metric: str


class EvalScore(TypedDict):
    scalar: float
    metrics: dict[str, float]
    critique: str


class ScoreDelta(TypedDict):
    absolute: float
    relative_pct: float
    improved: bool


class EvalSuite(TypedDict):
    primary_metric: str
    metrics: list[str]
    test_split_path: str
    use_llm_grading: bool


class TrainingMetrics(TypedDict):
    train_loss: float
    val_loss: float
    test_loss: float
    primary_metric: float


class ExperimentResult(TypedDict):
    job_id: str
    status: str
    metrics: TrainingMetrics
    model_path: str
    cost_usd: float
    logs_path: str


class Hypothesis(TypedDict):
    description: str
    patch: str
    expected_effect: str
    search_strategy: str  # 'random' | 'local' | 'playbook'


class IterationRecord(TypedDict):
    iteration: int
    hypothesis: str
    patch: str
    cost_usd: float
    metrics: TrainingMetrics
    decision: str   # 'KEPT' | 'REVERTED'
    notes: str


ResearchDiary = list[IterationRecord]


class CostBreakdown(TypedDict):
    data_gen_usd: float
    training_usd: float
    llm_calls_usd: float
    total_usd: float
    termination_reason: str  # 'budget_limit' | 'training_complete' | 'error'


class TrainedModel(TypedDict):
    weights_path: str
    metrics: EvalScore
    cost: CostBreakdown
    n_iterations: int
    research_diary_path: str


class LogEntry(TypedDict):
    agent: str
    level: str
    message: str
    metadata: dict
    timestamp: str


class JobConfig(TypedDict):
    gpu_type: str
    num_gpus: int
    timeout_min: int
    env_vars: dict
    output_dir: str


class JobSummary(TypedDict):
    job_id: str
    status: str
    submitted_at: str
    cost_usd: float
    script_name: str


# ─── LANGGRAPH STATES ─────────────────────────────────────────────────────────

class ManagerState(TypedDict):
    prompt: str
    budget: float
    data_path: str | None
    has_data: bool
    task_reasoning: TaskReasoning | None
    config: OrchestrationConfig | None
    result: TrainedModel | None


class DataGenState(TypedDict):
    config: OrchestrationConfig
    data_path: str | None
    mode: str | None           # 'A' | 'B' | 'C'
    raw_data: RawData | None
    hf_candidates: list[HFCandidate]
    selected_candidate: HFCandidate | None
    schema: DataSchema | None
    dataset: StandardDataset | None
    validation_report: ValidationReport | None


class AutoResearchState(TypedDict):
    plan: TrainingPlan
    config: OrchestrationConfig
    eval_suite: EvalSuite | None
    current_script: str        # path to the training script (never mutated by PROPOSE)
    current_config: dict       # live hyperparams; updated by keep_node after each KEPT patch
    current_patch: str | None  # JSON-encoded patch from the last propose_node call
    last_description: str | None  # Claude's human-readable hypothesis for the diary
    original_content: str | None
    diary: ResearchDiary
    baseline_score: EvalScore | None
    best_score: EvalScore | None
    best_script: str
    last_result: ExperimentResult | None
    last_score: EvalScore | None
    last_delta: ScoreDelta | None
    iteration: int
    no_improve_streak: int
    should_stop: bool
