export type StageStatus = 'pending' | 'in-progress' | 'complete';
export type TaskType = 'classification' | 'regression' | 'fine-tuning';
export type LogType = 'default' | 'success' | 'warning';
export type IterationStatus = 'KEPT' | 'REVERTED' | 'PENDING';

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

export interface Iteration {
  id: string;
  experiment: string;
  diff?: string;
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
  costSpent: number;
  metrics: MetricPoint[];
  iterations: Iteration[];
  logs: LogEntry[];
  stages: PipelineStage[];
}
