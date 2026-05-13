import { useState, useRef, useCallback } from 'react';
import type { TrainingState, PipelineStage, LogEntry } from '../types';
import { matchTask } from '../utils/matchTask';
import type { TaskConfig } from '../data/taskConfigs';

// ─── Stage definitions ────────────────────────────────────────────────────────
const STAGE_LABELS = [
  'Manager Init',
  'Data Discovery',
  'Model Selection',
  'Baseline Training',
  'AutoResearch Setup',
  'Experiment Loop',
  'Final Run',
  'Finalization',
];

const STAGE_DURATIONS = [
  1500,   // 0  Manager Init
  2800,   // 1  Data Discovery
  2200,   // 2  Model Selection
  4000,   // 3  Baseline Training
  3000,   // 4  AutoResearch Setup
  11000,  // 5  Experiment Loop
  4500,   // 6  Final Run
  1800,   // 7  Finalization
];

// ─── Dynamic log builders (driven by TaskConfig) ──────────────────────────────
function buildStageLogs(cfg: TaskConfig): Array<Array<{ component: string; message: string; type: LogEntry['type'] }>> {
  const ds0 = cfg.datasets[0];
  const ds1 = cfg.datasets[1];
  const dsLine = ds1
    ? `${ds0.name} (${ds0.size}) + ${ds1.name} (${ds1.size})`
    : `${ds0.name} (${ds0.size})`;
  const totalSamples = cfg.datasets.reduce((acc, d) => {
    const n = parseInt(d.size.replace(/[^0-9]/g, ''), 10);
    const mult = d.size.includes('M') ? 1_000_000 : d.size.includes('k') ? 1_000 : 1;
    return acc + n * mult;
  }, 0);
  const sampleLabel = totalSamples >= 1_000_000
    ? `${(totalSamples / 1_000_000).toFixed(1)}M`
    : `${Math.round(totalSamples / 1000)}k`;

  const bl = cfg.baseline;
  const fn = cfg.final;
  const metricFmt = (v: number) =>
    cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L'
      ? v.toFixed(3)
      : `${(v * 100).toFixed(1)}%`;

  return [
    // 0 Manager Init
    [
      { component: 'Manager',      message: 'Initializing orchestration pipeline',                      type: 'default' },
      { component: 'Manager',      message: `Prompt parsed — task: ${cfg.taskLabel}`,                   type: 'default' },
      { component: 'Manager',      message: `Claude reasoning: strategy=${cfg.strategy}, type=${cfg.trainingType}`, type: 'success' },
      { component: 'CostManager',  message: 'Budget guardrail active — monitoring spend',               type: 'success' },
    ],
    // 1 Data Discovery
    [
      { component: 'DataGen',      message: `Querying HuggingFace Hub — ${cfg.datasets.length} dataset quer${cfg.datasets.length > 1 ? 'ies' : 'y'}`, type: 'default' },
      ...cfg.datasets.map((d, i) => ({
        component: 'DataGen',
        message: `Query ${i + 1}: "${d.name}" — ${d.size} samples found`,
        type: 'default' as LogEntry['type'],
      })),
      { component: 'DataGen',      message: `Scoring relevance — selected: ${dsLine}`,                  type: 'success' },
      { component: 'DataGen',      message: `Merging datasets — ${sampleLabel} total, 80/10/10 split`,  type: 'success' },
    ],
    // 2 Model Selection
    [
      { component: 'Decision',     message: `Analyzing task: ${cfg.taskLabel}`,                         type: 'default' },
      { component: 'Decision',     message: `Dataset size favors ${cfg.strategy}`,                      type: 'default' },
      { component: 'Decision',     message: `Selected model: ${cfg.model}`,                             type: 'default' },
      { component: 'Decision',     message: `LoRA config: rank=${cfg.loraRank}, alpha=${cfg.loraRank * 2}, dropout=0.05, modules=${cfg.targetModules}`, type: 'success' },
      { component: 'Decision',     message: 'Training script written → outputs/scripts/train.py',       type: 'success' },
    ],
    // 3 Baseline Training
    [
      { component: 'Tinker',       message: `Creating LoRA training client — ${cfg.model}`,             type: 'default' },
      { component: 'Tinker',       message: `Tokenizing ${Math.round(totalSamples * 0.8 / 1000)}k train samples`, type: 'default' },
      { component: 'Tinker',       message: `Epoch 1/3 — loss: ${(bl.loss * 1.12).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(bl.metric * 0.91)}`, type: 'default' },
      { component: 'Tinker',       message: `Epoch 2/3 — loss: ${(bl.loss * 1.06).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(bl.metric * 0.96)}`, type: 'default' },
      { component: 'Tinker',       message: `Epoch 3/3 — loss: ${bl.loss.toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(bl.metric)}`,                  type: 'default' },
      { component: 'CostManager',  message: 'Baseline training cost: $1.14',                            type: 'warning' },
      { component: 'Tinker',       message: 'Baseline weights saved → outputs/model/baseline',          type: 'success' },
    ],
    // 4 AutoResearch Setup
    [
      { component: 'AutoResearch', message: 'Initializing AutoResearch loop',                           type: 'default' },
      { component: 'AutoResearch', message: `Creating eval suite — primary metric: ${cfg.evalMetric}`,  type: 'default' },
      { component: 'AutoResearch', message: `Loading test split — ${Math.round(totalSamples * 0.1 / 1000)}k samples`, type: 'default' },
      { component: 'AutoResearch', message: 'Running baseline evaluation on test set…',                 type: 'default' },
      { component: 'AutoResearch', message: `Baseline score — loss: ${bl.loss}, ${cfg.evalMetric}: ${metricFmt(bl.metric)}`, type: 'success' },
      { component: 'AutoResearch', message: 'Research diary initialized — max 20 iterations, early-stop at 3 no-improve', type: 'success' },
      { component: 'AutoResearch', message: 'Entering experiment loop…',                                type: 'success' },
    ],
    // 5 Experiment Loop (per-iteration logs injected separately in scheduleExperimentLoop)
    [
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #1',              type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #2',              type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #3',              type: 'default' },
      { component: 'CostManager',  message: 'Spend at 28% of budget — status: OK',                     type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #4',              type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #5',              type: 'default' },
      { component: 'CostManager',  message: 'Spend at 47% of budget — status: OK',                     type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #6',              type: 'default' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #7',              type: 'default' },
      { component: 'CostManager',  message: 'Spend at 63% of budget — status: WARNING',                type: 'warning' },
      { component: 'AutoResearch', message: 'No-improve streak: 1 — continuing search',                 type: 'warning' },
      { component: 'AutoResearch', message: 'PROPOSE — querying Claude for hypothesis #8',              type: 'default' },
      { component: 'CostManager',  message: 'Checkpoint saved at 70% budget threshold',                 type: 'warning' },
      { component: 'AutoResearch', message: `Best config locked — ${cfg.evalMetric}: ${metricFmt(cfg.iterations.filter(i => i.status === 'KEPT').slice(-1)[0]?.metricAfter ?? fn.metric)}`, type: 'success' },
    ],
    // 6 Final Run
    [
      { component: 'AutoResearch', message: `Experiment loop complete — ${cfg.iterations.length} iterations`, type: 'success' },
      { component: 'Manager',      message: 'Launching final training run with best config',             type: 'default' },
      { component: 'Tinker',       message: `Final config: ${cfg.model}, rank=${cfg.loraRank}, modules=${cfg.targetModules}`, type: 'default' },
      { component: 'Tinker',       message: `Final epoch 1/3 — loss: ${(fn.loss * 1.14).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric * 0.93)}`, type: 'default' },
      { component: 'Tinker',       message: `Final epoch 2/3 — loss: ${(fn.loss * 1.06).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric * 0.97)}`, type: 'default' },
      { component: 'Tinker',       message: `Final epoch 3/3 — loss: ${fn.loss.toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric)}`,                  type: 'default' },
      { component: 'Tinker',       message: 'Final weights saved → outputs/model/final',                type: 'success' },
      { component: 'CostManager',  message: 'Final run cost: $1.14 — pipeline total: $14.80 (29.6%)',   type: 'success' },
    ],
    // 7 Finalization
    [
      { component: 'Manager',      message: 'Collecting final metrics and artifacts',                   type: 'default' },
      { component: 'Observability', message: `Research diary serialized — ${cfg.iterations.length} iterations logged`, type: 'success' },
      { component: 'Observability', message: 'Agent log written → outputs/logs/agent_log.jsonl',        type: 'success' },
      { component: 'Manager',      message: `Training complete — ${cfg.evalMetric}: ${metricFmt(fn.metric)}`, type: 'success' },
    ],
  ];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function makeStages(): PipelineStage[] {
  return STAGE_LABELS.map((label, i) => ({ id: i, label, status: 'pending' as const }));
}

function nowTime(): string {
  return new Date().toTimeString().slice(0, 8);
}

function jitter(base: number, mag = 0.012): number {
  return +(base + (Math.random() - 0.5) * mag).toFixed(4);
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
    setState(prev => ({
      ...prev,
      logs: [{ ...entry, time: nowTime() }, ...prev.logs].slice(0, 80),
    }));
  }, []);

  const start = useCallback((prompt: string, budget: number, taskType: TrainingState['taskType']) => {
    clearTimers();

    // ── Match prompt to task config ─────────────────────────────────────────
    const cfg: TaskConfig = matchTask(prompt);
    const stageLogs = buildStageLogs(cfg);

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

      // Stagger logs evenly
      const logs = stageLogs[stageIdx];
      logs.forEach((log, li) => {
        schedule(() => appendLog(log), stageStart + ((li + 1) * dur) / (logs.length + 1));
      });

      // ── Baseline Training (3): loss curve from high → baseline ─────────────
      if (stageIdx === 3) {
        const { baseline: bl } = cfg;
        const tickCount = Math.floor(dur / 400);
        for (let t = 0; t < tickCount; t++) {
          const p = t / tickCount;
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, {
                loss:     jitter(bl.loss * (1.22 - p * 0.22), 0.012),
                accuracy: jitter(bl.metric * (0.87 + p * 0.13), 0.008),
                iteration: prev.metrics.length + 1,
              }],
            }));
          }, stageStart + t * 400 + 150);
        }
      }

      // ── Experiment Loop (5): iterations appear one-by-one ──────────────────
      if (stageIdx === 5) {
        const iters = cfg.iterations;
        const iterSpacing = dur / (iters.length + 1);
        const resolveDelay = Math.min(iterSpacing * 0.65, 900);

        iters.forEach((iter, i) => {
          const appearAt = stageStart + iterSpacing * (i + 1);

          schedule(() => appendLog({
            component: 'Tinker',
            message: `Running experiment ${i + 1}/${iters.length}…`,
            type: 'default',
          }), appearAt - 200);

          // Appear as PENDING (map IterationTemplate → Iteration shape)
          schedule(() => {
            setState(prev => ({
              ...prev,
              iterations: [{
                id: `iter-${i}`,
                experiment: iter.experiment,
                diff: iter.diff,
                loss: iter.lossAfter,
                f1: iter.metricAfter,
                status: 'PENDING' as const,
              }, ...prev.iterations],
            }));
          }, appearAt);

          // Resolve
          schedule(() => {
            setState(prev => ({
              ...prev,
              iterations: prev.iterations.map(it =>
                it.id === `iter-${i}` ? { ...it, status: iter.status } : it
              ),
            }));
            const metricFmt = (v: number) =>
              cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L'
                ? v.toFixed(3)
                : `${(v * 100).toFixed(1)}%`;
            appendLog({
              component: 'AutoResearch',
              message: iter.status === 'KEPT'
                ? `KEPT — loss ${iter.lossAfter.toFixed(3)}, ${cfg.evalMetric} ${metricFmt(iter.metricAfter)} (iter ${i + 1})`
                : `REVERTED — regression detected (iter ${i + 1})`,
              type: iter.status === 'KEPT' ? 'success' : 'warning',
            });
          }, appearAt + resolveDelay);

          // Metric tick
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, {
                loss:     jitter(iter.lossAfter, 0.01),
                accuracy: jitter(iter.metricAfter, 0.007),
                iteration: prev.metrics.length + 1,
              }],
            }));
          }, appearAt + resolveDelay + 120);
        });
      }

      // ── Final Run (6): cleaner, faster-converging loss curve ───────────────
      if (stageIdx === 6) {
        const { baseline: bl, final: fn } = cfg;
        const tickCount = Math.floor(dur / 350);
        for (let t = 0; t < tickCount; t++) {
          const p = t / tickCount;
          const startLoss   = bl.loss * 0.88;
          const startMetric = bl.metric * 1.03;
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, {
                loss:     jitter(startLoss     - p * (startLoss - fn.loss),     0.007),
                accuracy: jitter(startMetric   + p * (fn.metric - startMetric), 0.005),
                iteration: prev.metrics.length + 1,
              }],
            }));
          }, stageStart + t * 350 + 150);
        }
      }

      // ── Cost ticks ─────────────────────────────────────────────────────────
      if (stageIdx >= 1) {
        const stageWeights = [0, 0.04, 0.02, 0.14, 0.05, 0.52, 0.20, 0.03];
        const costThisStage = budget * 0.76 * (stageWeights[stageIdx] ?? 0.05);
        const ticks = Math.max(1, Math.floor(dur / 500));
        for (let t = 0; t < ticks; t++) {
          schedule(() => {
            setState(prev => ({
              ...prev,
              costSpent: Math.min(+(prev.costSpent + costThisStage / ticks).toFixed(2), budget),
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

    // Pipeline complete
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
