import { useState, useRef, useCallback } from 'react';
import type { TrainingState, PipelineStage, Iteration, LogEntry, ExperienceLevel } from '../types';
import { getCapabilities } from '../utils/experience';

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
    { component: 'AutoResearch', message: 'PROPOSE: creating eval suite and recording baseline', type: 'default' },
    { component: 'AutoResearch', message: 'Hypothesis: decrease learning_rate 3e-4→1.5e-4 (local ±20%)', type: 'default' },
    { component: 'AutoResearch', message: 'KEPT — val_loss improved 0.312→0.289 (+7.4%)', type: 'success' },
    { component: 'AutoResearch', message: 'Hypothesis: increase lora_rank 16→32 (random search)', type: 'default' },
    { component: 'AutoResearch', message: 'KEPT — F1 improved 0.871→0.901 (+3.4%)', type: 'success' },
    { component: 'AutoResearch', message: 'Hypothesis: increase learning_rate 1.5e-4→6e-4 (local ±20%)', type: 'default' },
    { component: 'AutoResearch', message: 'REVERTED — loss spiked 0.289→0.334, patch rolled back', type: 'warning' },
    { component: 'CostManager', message: 'Cumulative spend approaching 60% of budget', type: 'warning' },
    { component: 'AutoResearch', message: 'Best config committed: lora_rank=32, lr=1.5e-4, warmup=500', type: 'success' },
  ],
  [
    { component: 'Manager', message: 'Collecting final metrics and artifacts', type: 'default' },
    { component: 'Observability', message: 'Research diary serialized to JSON', type: 'success' },
    { component: 'Manager', message: 'Training pipeline complete', type: 'success' },
  ],
];

const ITERATIONS: Array<Omit<Iteration, 'id'>> = [
  {
    experiment: 'Decrease learning_rate 3e-4→1.5e-4 to reduce loss spikes.',
    diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015',
    loss: 0.289, f1: 0.871, status: 'KEPT',
    reasoning: {
      rationale: 'Training loss showed high variance across batches, suggesting the learning rate is too large and causing the optimizer to overshoot minima. Halving it should smooth convergence.',
      expected_effect: 'val_loss should decrease by ~5–10% and loss variance should drop within 1 epoch.',
      search_strategy: 'local',
      outcome_vs_expected: 'Matched — val_loss dropped 7.4% and variance stabilised. Patch kept.',
    },
  },
  {
    experiment: 'Increase lora_rank 16→32 to expand model capacity for task.',
    diff: '- lora_rank: 16\n+ lora_rank: 32',
    loss: 0.271, f1: 0.901, status: 'KEPT',
    reasoning: {
      rationale: 'F1 plateau at 0.871 suggests the LoRA adapter lacks representational capacity for the task. Doubling rank adds more trainable parameters without touching the frozen base weights.',
      expected_effect: 'F1 should improve by 2–4% at the cost of ~15% more compute per step.',
      search_strategy: 'random',
      outcome_vs_expected: 'Matched — F1 improved 3.4%. Compute overhead acceptable within budget.',
    },
  },
  {
    experiment: 'Increase learning_rate 1.5e-4→6.1e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00061',
    loss: 0.334, f1: 0.862, status: 'REVERTED',
    reasoning: {
      rationale: 'After stabilising at a lower LR, probing upward tests whether the current minima is a saddle point that a larger step could escape. Local perturbation of ±20% selected as a conservative range.',
      expected_effect: 'Possible 1–2% F1 gain if a better basin exists nearby; risk of loss spike if not.',
      search_strategy: 'local',
      outcome_vs_expected: 'Did not match — loss spiked from 0.289 to 0.334. Patch reverted, previous config restored.',
    },
  },
  {
    experiment: 'Increase warmup_steps 100→500 to stabilize early training.',
    diff: '- warmup_steps: 100\n+ warmup_steps: 500',
    loss: 0.248, f1: 0.914, status: 'KEPT',
    reasoning: {
      rationale: 'Early training logs show loss spikes in the first 80 steps consistent with a learning rate that ramps too quickly. Extending warmup gives the optimizer more time to find a stable trajectory before reaching peak LR.',
      expected_effect: 'Smoother early loss curve and ~1–3% improvement in final val_loss.',
      search_strategy: 'claude',
      outcome_vs_expected: 'Matched — loss decreased a further 6.4% and early spike eliminated.',
    },
  },
  {
    experiment: 'Decrease dropout 0.1→0.06 (local ±20% perturbation).',
    diff: '- dropout: 0.1\n+ dropout: 0.06',
    loss: 0.243, f1: 0.917, status: 'KEPT',
    reasoning: {
      rationale: 'Train/val loss gap is small (~0.005), indicating the model is not overfitting. Current dropout may be over-regularising. Reducing it slightly should allow the model to use more capacity without increasing overfitting risk.',
      expected_effect: 'Minor F1 improvement of ~0.5–1% with negligible change in generalisation gap.',
      search_strategy: 'local',
      outcome_vs_expected: 'Matched — F1 improved 0.3% and train/val gap unchanged.',
    },
  },
  {
    experiment: 'Increase num_epochs 3→4 to allow longer convergence.',
    diff: '- num_epochs: 3\n+ num_epochs: 4',
    loss: 0.261, f1: 0.908, status: 'REVERTED',
    reasoning: {
      rationale: 'Val loss curve was still gently declining at epoch 3, suggesting the model has not fully converged. Adding an epoch might extract further signal from the data.',
      expected_effect: 'F1 improvement of 0.5–1.5% if convergence is genuinely incomplete.',
      search_strategy: 'claude',
      outcome_vs_expected: 'Did not match — val_loss increased on epoch 4, indicating overfitting onset. Patch reverted.',
    },
  },
  {
    experiment: 'Decrease learning_rate 1.5e-4→1.2e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00012',
    loss: 0.231, f1: 0.923, status: 'KEPT',
    reasoning: {
      rationale: 'With warmup now extended, the effective peak LR may still be slightly high for fine-tuning on this dataset size. A conservative downward nudge probes whether slower late-stage updates improve final convergence.',
      expected_effect: 'val_loss decrease of 2–5% with no regression in F1.',
      search_strategy: 'local',
      outcome_vs_expected: 'Matched — best result so far. val_loss 0.231, F1 0.923. Patch kept.',
    },
  },
];

export function useTrainingSimulation() {
  const [state, setState] = useState<TrainingState>({
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    experience: 'Intermediate',
    capabilities: getCapabilities('Intermediate'),
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

  const start = useCallback((prompt: string, budget: number, taskType: TrainingState['taskType'], experience: ExperienceLevel = 'Intermediate') => {
    clearTimers();
    setState({
      status: 'running',
      stage: 0,
      prompt,
      budget,
      taskType,
      experience,
      capabilities: getCapabilities(experience),
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
      experience: 'Intermediate',
      capabilities: getCapabilities('Intermediate'),
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
    });
  }, []);

  return { state, start, reset };
}
