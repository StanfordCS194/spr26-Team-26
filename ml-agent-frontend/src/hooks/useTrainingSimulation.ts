import { useState, useRef, useCallback } from 'react';
import type { TrainingState, PipelineStage, Iteration, LogEntry } from '../types';

const STAGE_LABELS = [
  'Manager Init',
  'Data Discovery',
  'Model Selection',
  'Training',
  'AutoResearch',
  'Finalization',
];

const STAGE_DURATIONS = [1200, 1800, 1500, 3000, 3000, 1200]; // ms

function makeStages(): PipelineStage[] {
  return STAGE_LABELS.map((label, i) => ({ id: i, label, status: 'pending' as const }));
}

function nowTime(): string {
  return new Date().toTimeString().slice(0, 8);
}

const STAGE_LOGS: Array<Array<{ component: string; message: string; type: LogEntry['type'] }>> = [
  [
    { component: 'Manager', message: 'Initializing orchestration pipeline', type: 'default' },
    { component: 'Manager', message: 'Task parsed: analyzing prompt and constraints', type: 'default' },
    { component: 'CostManager', message: 'Budget guardrail active — monitoring spend', type: 'success' },
  ],
  [
    { component: 'DataGen', message: 'Querying HuggingFace Hub for matching datasets', type: 'default' },
    { component: 'DataGen', message: 'Found 3 candidate datasets, scoring relevance', type: 'default' },
    { component: 'DataGen', message: 'Dataset selected: 42,000 samples loaded', type: 'success' },
  ],
  [
    { component: 'Decision', message: 'Running fine-tune vs pre-train analysis', type: 'default' },
    { component: 'Decision', message: 'Dataset size favors fine-tuning strategy', type: 'default' },
    { component: 'Decision', message: 'Base model selected: distilbert-base-uncased', type: 'success' },
  ],
  [
    { component: 'Tinker', message: 'Submitting training job to Tinker API', type: 'default' },
    { component: 'Tinker', message: 'Job accepted — GPU allocation confirmed', type: 'success' },
    { component: 'CostManager', message: 'Estimated job cost: $14.20', type: 'warning' },
    { component: 'Tinker', message: 'Training epoch 1/3 complete', type: 'default' },
  ],
  [
    { component: 'AutoResearch', message: 'Starting hyperparameter search loop', type: 'default' },
    { component: 'AutoResearch', message: 'Proposing experiment: LoRA rank sweep', type: 'default' },
    { component: 'CostManager', message: 'Cumulative spend approaching 60% of budget', type: 'warning' },
    { component: 'AutoResearch', message: 'Best config identified — committing weights', type: 'success' },
  ],
  [
    { component: 'Manager', message: 'Collecting final metrics and artifacts', type: 'default' },
    { component: 'Observability', message: 'Research diary serialized to JSON', type: 'success' },
    { component: 'Manager', message: 'Training pipeline complete', type: 'success' },
  ],
];

const ITERATIONS: Array<Omit<Iteration, 'id'>> = [
  { experiment: 'Baseline: lr=2e-4, batch=16', loss: 0.312, f1: 0.871, status: 'KEPT' },
  { experiment: 'LoRA rank=16, alpha=32', loss: 0.298, f1: 0.883, status: 'KEPT' },
  { experiment: 'LoRA rank=32, dropout=0.1', loss: 0.271, f1: 0.901, status: 'KEPT' },
  { experiment: 'lr=5e-4 (too aggressive)', loss: 0.334, f1: 0.862, status: 'REVERTED' },
  { experiment: 'LoRA rank=32, warmup=500', loss: 0.248, f1: 0.914, status: 'KEPT' },
  { experiment: 'Epoch +1 overfitting check', loss: 0.261, f1: 0.908, status: 'REVERTED' },
  { experiment: 'Final: rank=32, lr=2e-4, warmup=500', loss: 0.231, f1: 0.923, status: 'KEPT' },
];

export function useTrainingSimulation() {
  const [state, setState] = useState<TrainingState>({
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: makeStages(),
  });

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearTimers = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  };

  const schedule = useCallback((fn: () => void, delay: number) => {
    timersRef.current.push(setTimeout(fn, delay));
  }, []);

  const appendLog = useCallback((entry: Omit<LogEntry, 'time'>) => {
    setState(prev => {
      const newLog: LogEntry = { ...entry, time: nowTime() };
      const logs = [newLog, ...prev.logs].slice(0, 50);
      return { ...prev, logs };
    });
  }, []);

  const start = useCallback((prompt: string, budget: number, taskType: TrainingState['taskType']) => {
    clearTimers();
    setState({
      status: 'running',
      stage: 0,
      prompt,
      budget,
      taskType,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
    });

    let cursor = 0; // ms elapsed

    STAGE_DURATIONS.forEach((dur, stageIdx) => {
      const stageStart = cursor;

      // Mark stage in-progress
      schedule(() => {
        setState(prev => {
          const stages = prev.stages.map((s, i) =>
            i === stageIdx ? { ...s, status: 'in-progress' as const } : s
          );
          return { ...prev, stages, stage: stageIdx };
        });
      }, stageStart);

      // Emit logs staggered through the stage
      STAGE_LOGS[stageIdx].forEach((log, li) => {
        schedule(() => appendLog(log), stageStart + (li * dur) / (STAGE_LOGS[stageIdx].length + 1));
      });

      // Metric updates during Training (stage 3) and AutoResearch (stage 4)
      if (stageIdx === 3 || stageIdx === 4) {
        const tickCount = Math.floor(dur / 500);
        for (let t = 0; t < tickCount; t++) {
          const globalIter = stageIdx === 3 ? t : tickCount + t;
          const progress = globalIter / (tickCount * 2 - 1);
          const loss = +(0.42 - progress * 0.30 + (Math.random() - 0.5) * 0.02).toFixed(4);
          const accuracy = +(0.72 + progress * 0.22 + (Math.random() - 0.5) * 0.015).toFixed(4);
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, { loss, accuracy, iteration: prev.metrics.length + 1 }],
            }));
          }, stageStart + t * 500 + 200);
        }
      }

      // Cost ticks every ~600ms after stage 1
      if (stageIdx >= 1) {
        const totalBudgetUsed = budget * 0.76;
        const costPerStage = totalBudgetUsed / 5;
        const ticks = Math.floor(dur / 600);
        for (let t = 0; t < ticks; t++) {
          const amount = costPerStage / ticks;
          schedule(() => {
            setState(prev => ({
              ...prev,
              costSpent: Math.min(+(prev.costSpent + amount).toFixed(2), budget),
            }));
          }, stageStart + t * 600 + 300);
        }
      }

      // AutoResearch iterations during stage 4
      if (stageIdx === 4) {
        const iterDelay = dur / (ITERATIONS.length + 1);
        ITERATIONS.forEach((iter, i) => {
          schedule(() => {
            const iteration: Iteration = { ...iter, id: `iter-${i}` };
            setState(prev => ({
              ...prev,
              iterations: [iteration, ...prev.iterations],
            }));
          }, stageStart + iterDelay * (i + 1));
        });
      }

      // Mark stage complete
      schedule(() => {
        setState(prev => {
          const stages = prev.stages.map((s, i) =>
            i === stageIdx ? { ...s, status: 'complete' as const } : s
          );
          return { ...prev, stages };
        });
      }, stageStart + dur - 50);

      cursor += dur;
    });

    // Mark complete
    schedule(() => {
      setState(prev => ({ ...prev, status: 'complete', stage: 5 }));
    }, cursor + 100);
  }, [schedule, appendLog]);

  const reset = useCallback(() => {
    clearTimers();
    setState({
      status: 'idle',
      stage: -1,
      prompt: '',
      budget: 50,
      taskType: 'classification',
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
    });
  }, []);

  return { state, start, reset };
}
