import { useState, useRef, useCallback } from 'react';
import type { TrainingState, PipelineStage, Iteration, LogEntry } from '../types';

// ─── Stage definitions ────────────────────────────────────────────────────────
const STAGE_LABELS = [
  'Manager Init',       // 0
  'Data Discovery',     // 1
  'Model Selection',    // 2
  'Baseline Training',  // 3
  'AutoResearch Setup', // 4
  'Experiment Loop',    // 5
  'Final Run',          // 6
  'Finalization',       // 7
];

const STAGE_DURATIONS = [
  1500,   // 0  Manager Init
  2800,   // 1  Data Discovery
  2200,   // 2  Model Selection
  4000,   // 3  Baseline Training
  3000,   // 4  AutoResearch Setup
  11000,  // 5  Experiment Loop  ← main event
  4500,   // 6  Final Run
  1800,   // 7  Finalization
]; // total ≈ 30 s

// ─── Per-stage log scripts ────────────────────────────────────────────────────
const STAGE_LOGS: Array<Array<{ component: string; message: string; type: LogEntry['type'] }>> = [
  // 0 Manager Init
  [
    { component: 'Manager',      message: 'Initializing orchestration pipeline',            type: 'default' },
    { component: 'Manager',      message: 'Prompt parsed — task type: text-classification', type: 'default' },
    { component: 'Manager',      message: 'Claude reasoning complete — strategy: SFT fine-tune', type: 'success' },
    { component: 'CostManager',  message: 'Budget guardrail active — monitoring spend',     type: 'success' },
  ],
  // 1 Data Discovery
  [
    { component: 'DataGen',      message: 'Querying HuggingFace Hub — 2 dataset queries',  type: 'default' },
    { component: 'DataGen',      message: 'Query 1: "movie review sentiment" — 3 candidates found', type: 'default' },
    { component: 'DataGen',      message: 'Query 2: "product review sentiment" — 2 candidates found', type: 'default' },
    { component: 'DataGen',      message: 'Scoring relevance — SetFit/imdb selected (42k samples)', type: 'success' },
    { component: 'DataGen',      message: 'Scoring relevance — SetFit/amazon_polarity selected (36k samples)', type: 'success' },
    { component: 'DataGen',      message: 'Merging datasets — 78,000 total samples, 80/10/10 split', type: 'success' },
  ],
  // 2 Model Selection
  [
    { component: 'Decision',     message: 'Analyzing task type: text-classification',       type: 'default' },
    { component: 'Decision',     message: 'Dataset size favors fine-tuning over pre-train', type: 'default' },
    { component: 'Decision',     message: 'Candidate: distilbert-base-uncased (est. $0.38/epoch)', type: 'default' },
    { component: 'Decision',     message: 'Budget check passed — 3 epochs within $50 budget', type: 'success' },
    { component: 'Decision',     message: 'LoRA config: rank=16, alpha=32, dropout=0.05, modules=[query,value]', type: 'success' },
    { component: 'Decision',     message: 'Training script written → outputs/scripts/train.py', type: 'success' },
  ],
  // 3 Baseline Training
  [
    { component: 'Tinker',       message: 'Creating LoRA training client — distilbert-base-uncased', type: 'default' },
    { component: 'Tinker',       message: 'Tokenizing 62,400 train samples',                type: 'default' },
    { component: 'Tinker',       message: 'Epoch 1/3 — loss: 0.421, acc: 0.741',            type: 'default' },
    { component: 'Tinker',       message: 'Epoch 2/3 — loss: 0.367, acc: 0.801',            type: 'default' },
    { component: 'Tinker',       message: 'Epoch 3/3 — loss: 0.312, acc: 0.861',            type: 'default' },
    { component: 'CostManager',  message: 'Baseline training cost: $1.14',                  type: 'warning' },
    { component: 'Tinker',       message: 'Baseline weights saved → outputs/model/baseline', type: 'success' },
  ],
  // 4 AutoResearch Setup
  [
    { component: 'AutoResearch', message: 'Initializing AutoResearch loop',                 type: 'default' },
    { component: 'AutoResearch', message: 'Creating eval suite — primary metric: accuracy', type: 'default' },
    { component: 'AutoResearch', message: 'Loading test split — 7,800 samples',             type: 'default' },
    { component: 'AutoResearch', message: 'Running baseline evaluation on test set…',       type: 'default' },
    { component: 'AutoResearch', message: 'Baseline score — loss: 0.312, F1: 0.871, acc: 0.861', type: 'success' },
    { component: 'AutoResearch', message: 'Research diary initialized — max 20 iterations, early-stop at 3 no-improve', type: 'success' },
    { component: 'AutoResearch', message: 'Entering experiment loop…',                      type: 'success' },
  ],
  // 5 Experiment Loop — logs staggered around each iteration (see special handling below)
  [
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #1',    type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #2',    type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #3',    type: 'default' },
    { component: 'CostManager',  message: 'Spend at 28% of budget — status: OK',           type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #4',    type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #5',    type: 'default' },
    { component: 'CostManager',  message: 'Spend at 47% of budget — status: OK',           type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #6',    type: 'default' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #7',    type: 'default' },
    { component: 'CostManager',  message: 'Spend at 63% of budget — status: WARNING',      type: 'warning' },
    { component: 'AutoResearch', message: 'No-improve streak: 1 — continuing search',       type: 'warning' },
    { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #8',    type: 'default' },
    { component: 'CostManager',  message: 'Checkpoint saved at 70% budget threshold',       type: 'warning' },
    { component: 'AutoResearch', message: 'Best config locked — acc: 0.923, F1: 0.919',    type: 'success' },
  ],
  // 6 Final Run
  [
    { component: 'AutoResearch', message: 'Experiment loop complete — 8 iterations, best: iter #7', type: 'success' },
    { component: 'Manager',      message: 'Launching final training run with best config',  type: 'default' },
    { component: 'Tinker',       message: 'Final config: lr=1.2e-4, lora_rank=32, warmup=500, dropout=0.06', type: 'default' },
    { component: 'Tinker',       message: 'Final run epoch 1/3 — loss: 0.271, acc: 0.893', type: 'default' },
    { component: 'Tinker',       message: 'Final run epoch 2/3 — loss: 0.238, acc: 0.911', type: 'default' },
    { component: 'Tinker',       message: 'Final run epoch 3/3 — loss: 0.214, acc: 0.931', type: 'default' },
    { component: 'Tinker',       message: 'Final weights saved → outputs/model/final',      type: 'success' },
    { component: 'CostManager',  message: 'Final run cost: $1.14 — total: $14.80 (29.6%)', type: 'success' },
  ],
  // 7 Finalization
  [
    { component: 'Manager',      message: 'Collecting final metrics and artifacts',         type: 'default' },
    { component: 'Observability', message: 'Research diary serialized — 8 iterations logged', type: 'success' },
    { component: 'Observability', message: 'Agent log written → outputs/logs/agent_log.jsonl', type: 'success' },
    { component: 'Manager',      message: 'Training pipeline complete — final acc: 0.931', type: 'success' },
  ],
];

// ─── AutoResearch iterations ──────────────────────────────────────────────────
const ITERATIONS: Array<Omit<Iteration, 'id'>> = [
  {
    experiment: 'Decrease learning_rate 3e-4→1.5e-4 to reduce loss spikes.',
    diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015',
    loss: 0.289, f1: 0.882, status: 'KEPT',
  },
  {
    experiment: 'Increase lora_rank 16→32 to expand model capacity for task.',
    diff: '- lora_rank: 16\n+ lora_rank: 32',
    loss: 0.271, f1: 0.901, status: 'KEPT',
  },
  {
    experiment: 'Increase learning_rate 1.5e-4→6.1e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00061',
    loss: 0.334, f1: 0.862, status: 'REVERTED',
  },
  {
    experiment: 'Increase warmup_steps 100→500 to stabilize early training.',
    diff: '- warmup_steps: 100\n+ warmup_steps: 500',
    loss: 0.248, f1: 0.910, status: 'KEPT',
  },
  {
    experiment: 'Add weight_decay 0→0.01 to reduce overfitting.',
    diff: '- weight_decay: 0.0\n+ weight_decay: 0.01',
    loss: 0.253, f1: 0.906, status: 'REVERTED',
  },
  {
    experiment: 'Decrease dropout 0.1→0.06 (local ±20% perturbation).',
    diff: '- dropout: 0.1\n+ dropout: 0.06',
    loss: 0.243, f1: 0.914, status: 'KEPT',
  },
  {
    experiment: 'Decrease learning_rate 1.5e-4→1.2e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00012',
    loss: 0.231, f1: 0.923, status: 'KEPT',
  },
  {
    experiment: 'Increase num_epochs 3→4 to allow longer convergence.',
    diff: '- num_epochs: 3\n+ num_epochs: 4',
    loss: 0.228, f1: 0.919, status: 'REVERTED',
  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function makeStages(): PipelineStage[] {
  return STAGE_LABELS.map((label, i) => ({ id: i, label, status: 'pending' as const }));
}

function nowTime(): string {
  return new Date().toTimeString().slice(0, 8);
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
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
      return { ...prev, logs: [newLog, ...prev.logs].slice(0, 80) };
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

    let cursor = 0;

    STAGE_DURATIONS.forEach((dur, stageIdx) => {
      const stageStart = cursor;

      // Mark stage in-progress
      schedule(() => {
        setState(prev => ({
          ...prev,
          stage: stageIdx,
          stages: prev.stages.map((s, i) =>
            i === stageIdx ? { ...s, status: 'in-progress' as const } : s
          ),
        }));
      }, stageStart);

      // Stagger logs evenly through the stage
      const logs = STAGE_LOGS[stageIdx];
      logs.forEach((log, li) => {
        schedule(() => appendLog(log), stageStart + ((li + 1) * dur) / (logs.length + 1));
      });

      // ── Baseline Training (3): loss curve starting from high loss ──────────
      if (stageIdx === 3) {
        const tickCount = Math.floor(dur / 400);
        for (let t = 0; t < tickCount; t++) {
          const progress = t / tickCount;
          const loss     = +(0.44 - progress * 0.13 + (Math.random() - 0.5) * 0.015).toFixed(4);
          const accuracy = +(0.73 + progress * 0.13 + (Math.random() - 0.5) * 0.01).toFixed(4);
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, { loss, accuracy, iteration: prev.metrics.length + 1 }],
            }));
          }, stageStart + t * 400 + 150);
        }
      }

      // ── Experiment Loop (5): iterations appear one-by-one ─────────────────
      if (stageIdx === 5) {
        const iterSpacing = dur / (ITERATIONS.length + 1); // ~1.2s between iterations
        const resolveDelay = Math.min(iterSpacing * 0.65, 900);

        ITERATIONS.forEach((iter, i) => {
          const appearAt = stageStart + iterSpacing * (i + 1);

          // Per-experiment log entries right before and after each iteration
          schedule(() => appendLog({
            component: 'Tinker',
            message: `Running experiment ${i + 1}/${ITERATIONS.length}…`,
            type: 'default',
          }), appearAt - 200);

          // Appear as PENDING
          schedule(() => {
            setState(prev => ({
              ...prev,
              iterations: [{ ...iter, id: `iter-${i}`, status: 'PENDING' }, ...prev.iterations],
            }));
          }, appearAt);

          // Resolve to final status
          schedule(() => {
            setState(prev => ({
              ...prev,
              iterations: prev.iterations.map(it =>
                it.id === `iter-${i}` ? { ...it, status: iter.status } : it
              ),
            }));
            appendLog({
              component: 'AutoResearch',
              message: iter.status === 'KEPT'
                ? `KEPT — loss ${iter.loss}, F1 ${iter.f1} (iter ${i + 1})`
                : `REVERTED — regression detected (iter ${i + 1})`,
              type: iter.status === 'KEPT' ? 'success' : 'warning',
            });
          }, appearAt + resolveDelay);

          // Metric tick for this experiment
          const expProgress = (i + 1) / ITERATIONS.length;
          const loss     = +(0.312 - expProgress * 0.08 + (Math.random() - 0.5) * 0.012).toFixed(4);
          const accuracy = +(0.861 + expProgress * 0.065 + (Math.random() - 0.5) * 0.008).toFixed(4);
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, { loss, accuracy, iteration: prev.metrics.length + 1 }],
            }));
          }, appearAt + resolveDelay + 100);
        });
      }

      // ── Final Run (6): cleaner, faster-dropping loss curve ─────────────────
      if (stageIdx === 6) {
        const tickCount = Math.floor(dur / 350);
        for (let t = 0; t < tickCount; t++) {
          const progress = t / tickCount;
          const loss     = +(0.271 - progress * 0.06 + (Math.random() - 0.5) * 0.008).toFixed(4);
          const accuracy = +(0.893 + progress * 0.04 + (Math.random() - 0.5) * 0.006).toFixed(4);
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, { loss, accuracy, iteration: prev.metrics.length + 1 }],
            }));
          }, stageStart + t * 350 + 150);
        }
      }

      // ── Cost ticks (all stages after Manager Init) ─────────────────────────
      if (stageIdx >= 1) {
        const totalBudgetUsed = budget * 0.76;
        // Weight cost heavily toward training stages
        const stageWeights = [0, 0.04, 0.02, 0.14, 0.05, 0.52, 0.20, 0.03];
        const costThisStage = totalBudgetUsed * (stageWeights[stageIdx] ?? 0.05);
        const ticks = Math.max(1, Math.floor(dur / 500));
        for (let t = 0; t < ticks; t++) {
          const amount = costThisStage / ticks;
          schedule(() => {
            setState(prev => ({
              ...prev,
              costSpent: Math.min(+(prev.costSpent + amount).toFixed(2), budget),
            }));
          }, stageStart + t * 500 + 300);
        }
      }

      // Mark stage complete
      schedule(() => {
        setState(prev => ({
          ...prev,
          stages: prev.stages.map((s, i) =>
            i === stageIdx ? { ...s, status: 'complete' as const } : s
          ),
        }));
      }, stageStart + dur - 50);

      cursor += dur;
    });

    // Mark pipeline complete
    schedule(() => {
      setState(prev => ({ ...prev, status: 'complete', stage: STAGE_LABELS.length - 1 }));
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
