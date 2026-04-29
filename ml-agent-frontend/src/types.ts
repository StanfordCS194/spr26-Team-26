export type StageStatus = 'pending' | 'in-progress' | 'complete';
export type TaskType = 'classification' | 'regression' | 'fine-tuning';
export type LogType = 'default' | 'success' | 'warning';
export type IterationStatus = 'KEPT' | 'REVERTED' | 'PENDING';
export type ExperienceLevel = 'Beginner' | 'Intermediate' | 'Advanced';

export interface CapabilityProfile {
  comfort_level: 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';
  observability: {
    run_status: 'basic' | 'detailed';
    metrics_visibility: 'none' | 'summary' | 'full';
    autoresearch_diary_access: 'none' | 'summary' | 'full';
    cost_visibility: 'summary' | 'detailed';
  };
  control: {
    can_edit_hyperparameters: boolean;
    hyperparameter_scope: 'none' | 'high_level' | 'full';
    can_edit_training_script: boolean;
    can_constrain_autoresearch_space: boolean;
    can_set_custom_stopping_criteria: boolean;
    strategy_hints_allowed: boolean;
  };
}

export interface PipelineStage {
  id: number;
  label: string;
  status: StageStatus;
}

export interface MetricPoint {
  loss: number;
  accuracy: number;
  iteration: number;
}

export interface IterationReasoning {
  rationale: string;
  expected_effect: string;
  search_strategy: 'random' | 'local' | 'claude';
  outcome_vs_expected: string;
}

export interface Iteration {
  id: string;
  experiment: string;
  diff?: string;
  reasoning?: IterationReasoning;
  loss: number;
  f1: number;
  status: IterationStatus;
}

export interface LogEntry {
  time: string;
  component: string;
  message: string;
  type: LogType;
}

export interface TrainingState {
  status: 'idle' | 'running' | 'complete';
  stage: number;
  prompt: string;
  budget: number;
  taskType: TaskType;
  experience: ExperienceLevel;
  capabilities: CapabilityProfile;
  costSpent: number;
  metrics: MetricPoint[];
  iterations: Iteration[];
  logs: LogEntry[];
  stages: PipelineStage[];
}
