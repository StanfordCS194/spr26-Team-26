import { useState, useRef, useCallback } from 'react';
import type { StartTraining, TrainingState, PipelineStage, Iteration, LogEntry } from '../types';

const STAGE_LABELS = [
  'Manager Init',
  'Data Discovery',
  'Model Selection',
  'Training',
  'AutoResearch',
  'Finalization',
];

const STAGE_DURATIONS = [1200, 1800, 1500, 3000, 3000, 1200]; // ms
const SIMULATION_PROVENANCE = {
  spendMode: 'simulation',
  trainingBackend: 'simulation',
  dataMode: null,
  modeCFallback: null,
  budgetPreflightSkipped: false,
  budgetSkipReason: null,
  liveServices: [],
  evidence: [],
};

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
    { component: 'DataGen', message: 'Simulating dataset discovery for matching sources', type: 'default' },
    { component: 'DataGen', message: 'Simulated 3 candidate datasets and scored relevance', type: 'default' },
    { component: 'DataGen', message: 'Simulation selected a 42,000-sample dataset profile', type: 'success' },
  ],
  [
    { component: 'Decision', message: 'Simulating fine-tune vs pre-train analysis', type: 'default' },
    { component: 'Decision', message: 'Simulated dataset size favors fine-tuning strategy', type: 'default' },
    { component: 'Decision', message: 'Simulation selected base model distilbert-base-uncased', type: 'success' },
  ],
  [
    { component: 'Tinker', message: 'Simulating Tinker job submission', type: 'default' },
    { component: 'Tinker', message: 'Simulation accepted job and reserved a GPU slot', type: 'success' },
    { component: 'CostManager', message: 'Simulated budget reserve: $14.20', type: 'warning' },
    { component: 'Tinker', message: 'Simulated training epoch 1/3 complete', type: 'default' },
  ],
  [
    { component: 'AutoResearch', message: 'Simulating eval-suite creation and baseline recording', type: 'default' },
    { component: 'AutoResearch', message: 'Simulated hypothesis: decrease learning_rate 3e-4→1.5e-4', type: 'default' },
    { component: 'AutoResearch', message: 'Simulated KEPT decision: val_loss improved 0.312→0.289', type: 'success' },
    { component: 'AutoResearch', message: 'Simulated hypothesis: increase lora_rank 16→32', type: 'default' },
    { component: 'AutoResearch', message: 'Simulated KEPT decision: F1 improved 0.871→0.901', type: 'success' },
    { component: 'AutoResearch', message: 'Simulated hypothesis: increase learning_rate 1.5e-4→6e-4', type: 'default' },
    { component: 'AutoResearch', message: 'Simulated REVERTED decision: loss spiked 0.289→0.334', type: 'warning' },
    { component: 'CostManager', message: 'Simulated budget reserve approaching 60% of cap', type: 'warning' },
    { component: 'AutoResearch', message: 'Simulated best config: lora_rank=32, lr=1.5e-4, warmup=500', type: 'success' },
  ],
  [
    { component: 'Manager', message: 'Collecting simulated final metrics and artifacts', type: 'default' },
    { component: 'Observability', message: 'Simulated research diary serialized to JSON', type: 'success' },
    { component: 'Manager', message: 'Simulation pipeline complete', type: 'success' },
  ],
];

const ITERATIONS: Array<Omit<Iteration, 'id'>> = [
  {
    experiment: 'Decrease learning_rate 3e-4→1.5e-4 to reduce loss spikes.',
    diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015',
    loss: 0.289, f1: 0.871, primaryMetric: 0.871, primaryMetricLabel: 'F1', status: 'KEPT',
  },
  {
    experiment: 'Increase lora_rank 16→32 to expand model capacity for task.',
    diff: '- lora_rank: 16\n+ lora_rank: 32',
    loss: 0.271, f1: 0.901, primaryMetric: 0.901, primaryMetricLabel: 'F1', status: 'KEPT',
  },
  {
    experiment: 'Increase learning_rate 1.5e-4→6.1e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00061',
    loss: 0.334, f1: 0.862, primaryMetric: 0.862, primaryMetricLabel: 'F1', status: 'REVERTED',
  },
  {
    experiment: 'Increase warmup_steps 100→500 to stabilize early training.',
    diff: '- warmup_steps: 100\n+ warmup_steps: 500',
    loss: 0.248, f1: 0.914, primaryMetric: 0.914, primaryMetricLabel: 'F1', status: 'KEPT',
  },
  {
    experiment: 'Decrease dropout 0.1→0.06 (local ±20% perturbation).',
    diff: '- dropout: 0.1\n+ dropout: 0.06',
    loss: 0.243, f1: 0.917, primaryMetric: 0.917, primaryMetricLabel: 'F1', status: 'KEPT',
  },
  {
    experiment: 'Increase num_epochs 3→4 to allow longer convergence.',
    diff: '- num_epochs: 3\n+ num_epochs: 4',
    loss: 0.261, f1: 0.908, primaryMetric: 0.908, primaryMetricLabel: 'F1', status: 'REVERTED',
  },
  {
    experiment: 'Decrease learning_rate 1.5e-4→1.2e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00012',
    loss: 0.231, f1: 0.923, primaryMetric: 0.923, primaryMetricLabel: 'F1', status: 'KEPT',
  },
];

export function useTrainingSimulation() {
  const [state, setState] = useState<TrainingState>({
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    dataPath: null,
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: makeStages(),
    artifacts: null,
    result: null,
    provenance: null,
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

  const start = useCallback<StartTraining>((prompt, budget, taskType, dataPath = null) => {
    const normalizedDataPath = dataPath?.trim() || null;
    clearTimers();
    setState({
      status: 'running',
      stage: 0,
      prompt,
      budget,
      taskType,
      dataPath: normalizedDataPath,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
      artifacts: null,
      result: null,
      provenance: SIMULATION_PROVENANCE,
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
              metrics: [
                ...prev.metrics,
                {
                  loss,
                  accuracy,
                  primaryMetric: accuracy,
                  primaryMetricLabel: 'Accuracy',
                  iteration: prev.metrics.length + 1,
                },
              ],
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

      // AutoResearch iterations during stage 4: appear as PENDING, then resolve
      if (stageIdx === 4) {
        const iterDelay = dur / (ITERATIONS.length + 1);
        const resolveDelay = Math.min(iterDelay * 0.7, 800); // resolve before next appears
        ITERATIONS.forEach((iter, i) => {
          const appearAt = stageStart + iterDelay * (i + 1);
          // Step 1: add as PENDING
          schedule(() => {
            const pending: Iteration = { ...iter, id: `iter-${i}`, status: 'PENDING' };
            setState(prev => ({ ...prev, iterations: [pending, ...prev.iterations] }));
          }, appearAt);
          // Step 2: resolve to final status
          schedule(() => {
            setState(prev => ({
              ...prev,
              iterations: prev.iterations.map(it =>
                it.id === `iter-${i}` ? { ...it, status: iter.status } : it
              ),
            }));
          }, appearAt + resolveDelay);
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
      setState(prev => ({ ...prev, status: 'complete', stage: 5, provenance: SIMULATION_PROVENANCE }));
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
      dataPath: null,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
      artifacts: null,
      result: null,
      provenance: null,
    });
  }, []);

  const cancel = useCallback(() => {
    clearTimers();
    setState(prev => ({
      ...prev,
      status: prev.status === 'running' ? 'cancelled' : prev.status,
      logs: [
        {
          time: new Date().toLocaleTimeString('en-US', { hour12: false }),
          component: 'Manager',
          message: 'Run cancelled',
          type: 'warning',
        },
        ...prev.logs,
      ],
    }));
  }, []);

  return { state, start, reset, cancel };
}
