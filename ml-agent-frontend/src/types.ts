export type StageStatus = 'pending' | 'in-progress' | 'complete';
export type TaskType = 'classification' | 'regression' | 'fine-tuning';
export type LogType = 'default' | 'success' | 'warning';
export type IterationStatus = 'KEPT' | 'REVERTED' | 'PENDING';
export type SkillLevel = 'beginner' | 'intermediate' | 'expert';
export type DataSource = 'huggingface' | 'scraped' | 'synthetic';
export type ApprovalStage = 'dataset' | 'model' | null;
export type ProposalStrategy = 'random_search' | 'local_perturbation' | 'claude_proposal';

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
  hypothesis: string;
  strategy: ProposalStrategy;
  expectedImpact: string;
  rationale?: string;
  lossBeforeAfter?: [number, number];
  f1BeforeAfter?: [number, number];
}

export interface Iteration {
  id: string;
  experiment: string;
  diff?: string;
  loss: number;
  f1: number;
  status: IterationStatus;
  reasoning?: IterationReasoning;
  // Search-space coordinates for visualization
  searchCoord?: { lr: number; loraRank: number };
}

export interface LogEntry {
  time: string;
  component: string;
  message: string;
  type: LogType;
  // Minimum skill level needed to see this log.
  // 'beginner' logs are shown to everyone; 'expert' logs only to Expert.
  minLevel: SkillLevel;
}

export interface DataSampleRow {
  [columnName: string]: string | number;
}

export interface SyntheticDetails {
  generationPrompt: string;
  diversityScore: number;
  validationsPassed: number;
  validationsTotal: number;
}

export interface DataPreview {
  source: DataSource;
  datasetName: string;
  totalSamples: number;
  splits: { train: number; val: number; test: number };
  columns: string[];
  samples: DataSampleRow[];
  classDistribution?: Record<string, number>;
  reasoning: string[];
  syntheticDetails?: SyntheticDetails;
}

export interface ModelPlan {
  baseModel: string;
  approach: 'fine-tune-lora' | 'pretrain' | 'full-finetune';
  reasoning: string[];
  hyperparams: Record<string, string | number>;
  estimatedCost: number;
}

export interface TrainingState {
  status: 'idle' | 'running' | 'awaiting-approval' | 'complete';
  stage: number;
  prompt: string;
  budget: number;
  taskType: TaskType;
  skillLevel: SkillLevel;
  pendingApproval: ApprovalStage;
  dataPreview?: DataPreview;
  modelPlan?: ModelPlan;
  costSpent: number;
  metrics: MetricPoint[];
  iterations: Iteration[];
  logs: LogEntry[];
  stages: PipelineStage[];
}
