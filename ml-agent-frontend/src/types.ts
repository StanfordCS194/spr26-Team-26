export type StageStatus = 'pending' | 'in-progress' | 'complete';
export type TaskType = 'classification' | 'regression' | 'fine-tuning';
export type LogType = 'default' | 'success' | 'warning' | 'error';
export type IterationStatus = 'KEPT' | 'REVERTED' | 'PENDING';

export type StartTraining = (
  prompt: string,
  budget: number,
  taskType: TaskType,
  dataPath?: string | null
) => void | Promise<void>;

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

export interface ArtifactFile {
  name: string;
  label: string;
  path?: string | null;
  exists: boolean;
  sizeBytes?: number | null;
  contentType: string;
  downloadPath?: string | null;
}

export interface RunArtifacts {
  modelPath?: string | null;
  checkpoints: Record<string, unknown>;
  metrics?: Record<string, unknown> | null;
  sample?: Record<string, unknown> | null;
  files: ArtifactFile[];
}

export interface TrainingState {
  status: 'idle' | 'running' | 'complete' | 'failed';
  stage: number;
  prompt: string;
  budget: number;
  taskType: TaskType;
  dataPath?: string | null;
  costSpent: number;
  metrics: MetricPoint[];
  iterations: Iteration[];
  logs: LogEntry[];
  stages: PipelineStage[];
  artifacts?: RunArtifacts | null;
  result?: Record<string, unknown> | null;
  error?: string | null;
  runId?: string;
}
