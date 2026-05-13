export type StageStatus = 'pending' | 'in-progress' | 'complete';
export type TaskType = 'classification' | 'regression' | 'fine-tuning';
export type LogType = 'default' | 'success' | 'warning';
export type IterationStatus = 'KEPT' | 'REVERTED' | 'PENDING';
export type ExpertLevel = 'guided' | 'standard' | 'autonomous';

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

export interface DataSample {
  [field: string]: string | number;
}

export interface TrainingState {
  status: 'idle' | 'running' | 'complete';
  stage: number;
  prompt: string;
  budget: number;
  taskType: TaskType;
  expertLevel: ExpertLevel;
  costSpent: number;
  metrics: MetricPoint[];
  iterations: Iteration[];
  logs: LogEntry[];
  stages: PipelineStage[];
  dataSamples: DataSample[];
  datasetName: string;
}
