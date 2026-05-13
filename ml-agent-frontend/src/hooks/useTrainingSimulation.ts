import { useState, useRef, useCallback } from 'react';
import type { TrainingState, PipelineStage, LogEntry, ExpertLevel, ApprovalGate } from '../types';
import { matchTask } from '../utils/matchTask';
import type { TaskConfig } from '../data/taskConfigs';

// ─── Stage definitions ────────────────────────────────────────────────────────
const STAGE_LABELS = [
  'Manager Init',
  'Data Discovery',
  'Model Selection',
  'Code Development',
  'AutoResearch Setup',
  'Experiment Loop',
  'Final Run',
  'Finalization',
];

const STAGE_DURATIONS = [
  1500,   // 0  Manager Init
  2800,   // 1  Data Discovery
  2200,   // 2  Model Selection
  4000,   // 3  Code Development
  3000,   // 4  AutoResearch Setup
  11000,  // 5  Experiment Loop
  4500,   // 6  Final Run
  1800,   // 7  Finalization
];

// ─── Chapter definitions (groups of stages separated by approval gates) ───────
interface Chapter {
  stages: number[];
  gate?: ApprovalGate;   // shown AFTER this chapter runs; undefined = final chapter
}

function getChapters(level: ExpertLevel, cfg: TaskConfig): Chapter[] {
  const ds0 = cfg.datasets[0];
  const ds1 = cfg.datasets[1];
  const dsLine = ds1 ? `${ds0.name} + ${ds1.name}` : ds0.name;
  const totalSamples = cfg.datasets.reduce((acc, d) => {
    const n = parseInt(d.size.replace(/[^0-9]/g, ''), 10);
    const mult = d.size.includes('M') ? 1_000_000 : d.size.includes('k') ? 1_000 : 1;
    return acc + n * mult;
  }, 0);
  const sampleLabel = totalSamples >= 1_000_000
    ? `${(totalSamples / 1_000_000).toFixed(1)}M`
    : `${Math.round(totalSamples / 1000)}k`;

  if (level === 'guided') {
    return [
      {
        stages: [0, 1],
        gate: {
          title: 'Approve Dataset Selection',
          description: 'The agent has identified and scored candidate datasets. Review the selection below and approve to continue to model selection.',
          details: [
            `Selected: ${dsLine}`,
            `Total samples: ${sampleLabel} (80% train / 10% val / 10% test)`,
            `Task: ${cfg.taskLabel}`,
            `Quality score: 0.91 / 1.00`,
          ],
        },
      },
      {
        stages: [2, 3],
        gate: {
          title: 'Approve Model & Training Config',
          description: 'The agent has selected a model architecture and written the training script. Review the configuration before experiments begin.',
          details: [
            `Model: ${cfg.model}`,
            `Strategy: ${cfg.strategy} (${cfg.trainingType})`,
            `LoRA rank: ${cfg.loraRank}, alpha: ${cfg.loraRank * 2}`,
            `Target modules: ${cfg.targetModules}`,
            `Eval metric: ${cfg.evalMetric}`,
          ],
        },
      },
      { stages: [4, 5, 6, 7] },
    ];
  }

  if (level === 'standard') {
    return [
      {
        stages: [0, 1, 2],
        gate: {
          title: 'Approve Model & Training Config',
          description: 'The agent has selected a dataset and model. Review the setup before handing off to the training loop.',
          details: [
            `Dataset: ${dsLine} (${sampleLabel} samples)`,
            `Model: ${cfg.model}`,
            `LoRA rank: ${cfg.loraRank}, modules: ${cfg.targetModules}`,
            `Eval metric: ${cfg.evalMetric}`,
          ],
        },
      },
      { stages: [3, 4, 5, 6, 7] },
    ];
  }

  // autonomous — single chapter, no gates
  return [{ stages: [0, 1, 2, 3, 4, 5, 6, 7] }];
}

// ─── Dynamic log builders ─────────────────────────────────────────────────────
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

  const fn = cfg.final;
  const metricFmt = (v: number) =>
    cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L'
      ? v.toFixed(3)
      : `${(v * 100).toFixed(1)}%`;

  return [
    // 0 Manager Init
    [
      { component: 'Manager',       message: 'Initializing orchestration pipeline',                                              type: 'default' },
      { component: 'Manager',       message: `Prompt parsed — task: ${cfg.taskLabel}`,                                           type: 'default' },
      { component: 'Manager',       message: `Claude reasoning: strategy=${cfg.strategy}, type=${cfg.trainingType}`,             type: 'success' },
      { component: 'CostManager',   message: 'Budget guardrail active — monitoring spend',                                       type: 'success' },
    ],
    // 1 Data Discovery
    [
      { component: 'DataGen',       message: `Querying HuggingFace Hub — ${cfg.datasets.length} dataset quer${cfg.datasets.length > 1 ? 'ies' : 'y'}`, type: 'default' },
      ...cfg.datasets.map((d, i) => ({
        component: 'DataGen',
        message: `Query ${i + 1}: "${d.name}" — ${d.size} samples found`,
        type: 'default' as LogEntry['type'],
      })),
      { component: 'DataGen',       message: `Scoring relevance — selected: ${dsLine}`,                                          type: 'success' },
      { component: 'DataGen',       message: `Merging datasets — ${sampleLabel} total, 80/10/10 split`,                          type: 'success' },
    ],
    // 2 Model Selection
    [
      { component: 'Decision',      message: `Analyzing task: ${cfg.taskLabel}`,                                                 type: 'default' },
      { component: 'Decision',      message: `Dataset size favors ${cfg.strategy}`,                                              type: 'default' },
      { component: 'Decision',      message: `Selected model: ${cfg.model}`,                                                     type: 'default' },
      { component: 'Decision',      message: `LoRA config: rank=${cfg.loraRank}, alpha=${cfg.loraRank * 2}, dropout=0.05, modules=${cfg.targetModules}`, type: 'success' },
    ],
    // 3 Code Development
    [
      { component: 'Decision',      message: `Scaffolding training script for ${cfg.model}`,                                     type: 'default' },
      { component: 'Decision',      message: `Writing LoRA config: rank=${cfg.loraRank}, alpha=${cfg.loraRank * 2}, modules=${cfg.targetModules}`, type: 'default' },
      { component: 'Decision',      message: `Writing data loader — ${Math.round(totalSamples * 0.8 / 1000)}k train / ${Math.round(totalSamples * 0.1 / 1000)}k val`, type: 'default' },
      { component: 'Decision',      message: `Writing eval harness — primary metric: ${cfg.evalMetric}`,                         type: 'default' },
      { component: 'Decision',      message: 'Writing optimizer config: AdamW, cosine LR schedule',                              type: 'default' },
      { component: 'Decision',      message: 'Smoke-testing script on 32 samples — no errors',                                   type: 'success' },
      { component: 'Decision',      message: 'Training script ready → outputs/scripts/train.py',                                 type: 'success' },
    ],
    // 4 AutoResearch Setup
    [
      { component: 'AutoResearch',  message: 'Initializing AutoResearch loop',                                                   type: 'default' },
      { component: 'AutoResearch',  message: `Creating eval suite — primary metric: ${cfg.evalMetric}`,                          type: 'default' },
      { component: 'AutoResearch',  message: `Loading splits — ${Math.round(totalSamples * 0.8 / 1000)}k train / ${Math.round(totalSamples * 0.1 / 1000)}k val`, type: 'default' },
      { component: 'AutoResearch',  message: `Initial config: lr=3e-4, lora_rank=${cfg.loraRank}, epochs=3`,                     type: 'default' },
      { component: 'AutoResearch',  message: 'Research diary initialized — max 20 iterations, early-stop at 3 no-improve',       type: 'success' },
      { component: 'AutoResearch',  message: 'Handing off to experiment loop — training begins now',                             type: 'success' },
    ],
    // 5 Experiment Loop
    [
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #1',                                      type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #2',                                      type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #3',                                      type: 'default' },
      { component: 'CostManager',   message: 'Spend at 28% of budget — status: OK',                                              type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #4',                                      type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #5',                                      type: 'default' },
      { component: 'CostManager',   message: 'Spend at 47% of budget — status: OK',                                              type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #6',                                      type: 'default' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #7',                                      type: 'default' },
      { component: 'CostManager',   message: 'Spend at 63% of budget — status: WARNING',                                         type: 'warning' },
      { component: 'AutoResearch',  message: 'No-improve streak: 1 — continuing search',                                         type: 'warning' },
      { component: 'AutoResearch',  message: 'PROPOSE — querying Claude for hypothesis #8',                                      type: 'default' },
      { component: 'CostManager',   message: 'Checkpoint saved at 70% budget threshold',                                         type: 'warning' },
      { component: 'AutoResearch',  message: `Best config locked — ${cfg.evalMetric}: ${metricFmt(cfg.iterations.filter(i => i.status === 'KEPT').slice(-1)[0]?.metricAfter ?? fn.metric)}`, type: 'success' },
    ],
    // 6 Final Run
    [
      { component: 'AutoResearch',  message: `Experiment loop complete — ${cfg.iterations.length} iterations`,                   type: 'success' },
      { component: 'Manager',       message: 'Launching final training run with best config',                                     type: 'default' },
      { component: 'Tinker',        message: `Final config: ${cfg.model}, rank=${cfg.loraRank}, modules=${cfg.targetModules}`,   type: 'default' },
      { component: 'Tinker',        message: `Final epoch 1/3 — loss: ${(fn.loss * 1.14).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric * 0.93)}`, type: 'default' },
      { component: 'Tinker',        message: `Final epoch 2/3 — loss: ${(fn.loss * 1.06).toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric * 0.97)}`, type: 'default' },
      { component: 'Tinker',        message: `Final epoch 3/3 — loss: ${fn.loss.toFixed(3)}, ${cfg.evalMetric}: ${metricFmt(fn.metric)}`, type: 'default' },
      { component: 'Tinker',        message: 'Final weights saved → outputs/model/final',                                        type: 'success' },
      { component: 'CostManager',   message: 'Final run cost: $1.14 — pipeline total: $14.80 (29.6%)',                           type: 'success' },
    ],
    // 7 Finalization
    [
      { component: 'Manager',       message: 'Collecting final metrics and artifacts',                                           type: 'default' },
      { component: 'Observability', message: `Research diary serialized — ${cfg.iterations.length} iterations logged`,           type: 'success' },
      { component: 'Observability', message: 'Agent log written → outputs/logs/agent_log.jsonl',                                 type: 'success' },
      { component: 'Manager',       message: `Training complete — ${cfg.evalMetric}: ${metricFmt(fn.metric)}`,                   type: 'success' },
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
    expertLevel: 'standard',
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: makeStages(),
    dataSamples: [],
    datasetName: '',
    awaitingApproval: null,
  });

  const timersRef     = useRef<ReturnType<typeof setTimeout>[]>([]);
  const nextChapterRef = useRef<(() => void) | null>(null);
  // keep cfg + stageLogs alive across chapters
  const cfgRef        = useRef<TaskConfig | null>(null);
  const stageLogsRef  = useRef<ReturnType<typeof buildStageLogs> | null>(null);
  const budgetRef     = useRef(50);

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

  // ── runChapter: schedules one chapter's worth of stages, then pauses or completes ──
  const runChapter = useCallback((chapter: Chapter) => {
    const cfg       = cfgRef.current!;
    const stageLogs = stageLogsRef.current!;
    const budget    = budgetRef.current;

    // Clear the approval gate now that we're running
    setState(prev => ({ ...prev, awaitingApproval: null }));

    let cursor = 0;

    chapter.stages.forEach(stageIdx => {
      const dur        = STAGE_DURATIONS[stageIdx];
      const stageStart = cursor;

      // Mark in-progress
      schedule(() => {
        setState(prev => ({
          ...prev,
          stage: stageIdx,
          stages: prev.stages.map((s, i) =>
            i === stageIdx ? { ...s, status: 'in-progress' as const } : s
          ),
        }));
      }, stageStart);

      // Stagger logs
      const logs = stageLogs[stageIdx];
      logs.forEach((log, li) => {
        schedule(() => appendLog(log), stageStart + ((li + 1) * dur) / (logs.length + 1));
      });

      // ── Data Discovery (1): reveal sample rows ────────────────────────────
      if (stageIdx === 1) {
        schedule(() => {
          setState(prev => ({
            ...prev,
            datasetName: cfg.datasets[0].name,
            dataSamples: cfg.samples ?? [],
          }));
        }, stageStart + Math.round(dur * 0.62));
      }

      // ── Experiment Loop (5) ───────────────────────────────────────────────
      if (stageIdx === 5) {
        const iters       = cfg.iterations;
        const iterSpacing = dur / (iters.length + 1);
        const resolveDelay = Math.min(iterSpacing * 0.65, 900);

        iters.forEach((iter, i) => {
          const appearAt = stageStart + iterSpacing * (i + 1);

          schedule(() => appendLog({
            component: 'Tinker',
            message: `Running experiment ${i + 1}/${iters.length}…`,
            type: 'default',
          }), appearAt - 200);

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

      // ── Final Run (6): loss curve ─────────────────────────────────────────
      if (stageIdx === 6) {
        const { baseline: bl, final: fn } = cfg;
        const tickCount   = Math.floor(dur / 350);
        const startLoss   = bl.loss * 0.88;
        const startMetric = bl.metric * 1.03;
        for (let t = 0; t < tickCount; t++) {
          const p = t / tickCount;
          schedule(() => {
            setState(prev => ({
              ...prev,
              metrics: [...prev.metrics, {
                loss:     jitter(startLoss   - p * (startLoss - fn.loss),       0.007),
                accuracy: jitter(startMetric + p * (fn.metric - startMetric),   0.005),
                iteration: prev.metrics.length + 1,
              }],
            }));
          }, stageStart + t * 350 + 150);
        }
      }

      // ── Cost ticks ────────────────────────────────────────────────────────
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

    // After chapter ends: pause for approval or finish
    if (chapter.gate) {
      schedule(() => {
        setState(prev => ({ ...prev, awaitingApproval: chapter.gate! }));
        appendLog({ component: 'Manager', message: '⏸  Paused — waiting for your approval before continuing', type: 'warning' });
      }, cursor);
    } else {
      // Final chapter
      schedule(() => {
        setState(prev => ({ ...prev, status: 'complete', stage: STAGE_LABELS.length - 1 }));
      }, cursor + 100);
    }
  }, [schedule, appendLog]);

  const start = useCallback((prompt: string, budget: number, taskType: TrainingState['taskType'], expertLevel: ExpertLevel = 'standard') => {
    clearTimers();
    nextChapterRef.current = null;

    const cfg       = matchTask(prompt);
    const stageLogs = buildStageLogs(cfg);
    cfgRef.current      = cfg;
    stageLogsRef.current = stageLogs;
    budgetRef.current   = budget;

    const chapters = getChapters(expertLevel, cfg);

    setState({
      status: 'running',
      stage: 0,
      prompt,
      budget,
      taskType,
      expertLevel,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
      dataSamples: [],
      datasetName: '',
      awaitingApproval: null,
    });

    // Wire up chapters: each chapter's onComplete either shows a gate or runs next
    const scheduleChain = (idx: number) => {
      if (idx >= chapters.length) return;
      const chapter = chapters[idx];

      if (idx === 0) {
        // First chapter runs immediately
        runChapter(chapter);
      }

      if (chapter.gate) {
        // Store a callback so approve() can trigger the next chapter
        nextChapterRef.current = () => {
          scheduleChain(idx + 1);
          runChapter(chapters[idx + 1]);
        };
      }
    };

    scheduleChain(0);
  }, [runChapter]);

  // Called when user clicks Approve on the gate card
  const approve = useCallback(() => {
    if (nextChapterRef.current) {
      const next = nextChapterRef.current;
      nextChapterRef.current = null;
      next();
    }
  }, []);

  const reset = useCallback(() => {
    clearTimers();
    nextChapterRef.current = null;
    setState({
      status: 'idle',
      stage: -1,
      prompt: '',
      budget: 50,
      taskType: 'classification',
      expertLevel: 'standard',
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
      dataSamples: [],
      datasetName: '',
      awaitingApproval: null,
    });
  }, []);

  return { state, start, approve, reset };
}
