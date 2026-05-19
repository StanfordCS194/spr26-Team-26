import { useState, useRef, useCallback } from 'react';
import type {
  TrainingState,
  PipelineStage,
  Iteration,
  LogEntry,
  SkillLevel,
  DataPreview,
  ModelPlan,
  TaskType,
} from '../types';

const STAGE_LABELS = [
  'Manager Init',
  'Data Discovery',
  'Model Selection',
  'Training',
  'AutoResearch',
  'Finalization',
];

// Durations in ms per stage.
const STAGE_DURATIONS = [1200, 1800, 1500, 3000, 3000, 1200];

function makeStages(): PipelineStage[] {
  return STAGE_LABELS.map((label, i) => ({ id: i, label, status: 'pending' as const }));
}

function nowTime(): string {
  return new Date().toTimeString().slice(0, 8);
}

// Log scripts per stage. `minLevel` tags who can see each log.
// beginner = everyone, intermediate = Intermediate + Expert, expert = Expert only.
type StageLog = { component: string; message: string; type: LogEntry['type']; minLevel: SkillLevel };

const STAGE_LOGS: StageLog[][] = [
  [
    { component: 'Manager', message: 'Starting your training run', type: 'default', minLevel: 'beginner' },
    { component: 'Manager', message: 'Initializing orchestration pipeline', type: 'default', minLevel: 'intermediate' },
    { component: 'Manager', message: 'StateGraph(ManagerState) compiled; checkpoints bound to /tmp/run.db', type: 'default', minLevel: 'expert' },
    { component: 'CostManager', message: 'Budget guardrail active — watching spend', type: 'success', minLevel: 'beginner' },
    { component: 'CostManager', message: 'Polling Tinker billing every 15s', type: 'default', minLevel: 'expert' },
  ],
  [
    { component: 'DataGen', message: 'Looking for data that matches your task', type: 'default', minLevel: 'beginner' },
    { component: 'DataGen', message: 'Querying HuggingFace Hub with inferred keywords', type: 'default', minLevel: 'intermediate' },
    { component: 'DataGen', message: 'HF API: GET /api/datasets?search=sentiment+imdb (200 OK, 47 hits)', type: 'default', minLevel: 'expert' },
    { component: 'DataGen', message: 'Scoring 47 candidates by size × license × coverage', type: 'default', minLevel: 'intermediate' },
    { component: 'DataGen', message: 'Top 3: imdb (97K rows), rotten_tomatoes (11K), sst2 (70K)', type: 'default', minLevel: 'expert' },
    { component: 'DataGen', message: 'Dataset picked: 42,000 samples loaded', type: 'success', minLevel: 'beginner' },
    { component: 'DataGen', message: 'Class balance verified: 51.3% pos / 48.7% neg', type: 'default', minLevel: 'intermediate' },
    { component: 'DataGen', message: 'Ready for your review before training', type: 'default', minLevel: 'intermediate' },
  ],
  [
    { component: 'Decision', message: 'Picking the best model for your task', type: 'default', minLevel: 'beginner' },
    { component: 'Decision', message: 'Running fine-tune vs pre-train analysis', type: 'default', minLevel: 'intermediate' },
    { component: 'Decision', message: 'Dataset size favors fine-tuning strategy', type: 'default', minLevel: 'intermediate' },
    { component: 'Decision', message: 'Budget $50 < pretrain threshold $400 → forcing fine-tune', type: 'default', minLevel: 'expert' },
    { component: 'Decision', message: 'Base model selected: distilbert-base-uncased', type: 'success', minLevel: 'beginner' },
    { component: 'Decision', message: 'LoRA config: r=16, alpha=32, dropout=0.1, targets=[q_proj, v_proj]', type: 'default', minLevel: 'expert' },
  ],
  [
    { component: 'Tinker', message: 'Spinning up a GPU for training', type: 'default', minLevel: 'beginner' },
    { component: 'Tinker', message: 'Submitting training job to Tinker API', type: 'default', minLevel: 'intermediate' },
    { component: 'Tinker', message: 'POST /v1/jobs → job_id=tnk_7f3a2b (accepted, 1x A100-40GB)', type: 'default', minLevel: 'expert' },
    { component: 'CostManager', message: 'Estimated run: $14.20', type: 'warning', minLevel: 'beginner' },
    { component: 'Tinker', message: 'Training progress: 1/3 epochs', type: 'default', minLevel: 'beginner' },
    { component: 'Tinker', message: 'step=312, loss=0.3121, grad_norm=1.47, lr=3.0e-4', type: 'default', minLevel: 'expert' },
  ],
  [
    { component: 'AutoResearch', message: 'Trying small tweaks to improve the model', type: 'default', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'PROPOSE: creating eval suite and recording baseline', type: 'default', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'baseline_bpb=0.312, baseline_f1=0.871, time_budget=300s', type: 'default', minLevel: 'expert' },
    { component: 'AutoResearch', message: 'Hypothesis: lower learning rate (local ±20%)', type: 'default', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'Improvement found — keeping it', type: 'success', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'KEPT — val_loss improved 0.312→0.289 (+7.4%)', type: 'success', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'decision_edge → keep_node; commit sha=ab12cd', type: 'default', minLevel: 'expert' },
    { component: 'AutoResearch', message: 'Hypothesis: increase LoRA rank (random search)', type: 'default', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'Another improvement — keeping it', type: 'success', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'KEPT — F1 improved 0.871→0.901 (+3.4%)', type: 'success', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'Tried a change that hurt — rolled it back', type: 'warning', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'REVERTED — loss spiked 0.289→0.334, patch rolled back', type: 'warning', minLevel: 'intermediate' },
    { component: 'AutoResearch', message: 'decision_edge → revert_node; restore from iter_2.pt', type: 'default', minLevel: 'expert' },
    { component: 'CostManager', message: 'Spend approaching 60% of budget', type: 'warning', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'Best config locked in', type: 'success', minLevel: 'beginner' },
    { component: 'AutoResearch', message: 'Best config committed: lora_rank=32, lr=1.5e-4, warmup=500', type: 'success', minLevel: 'intermediate' },
  ],
  [
    { component: 'Manager', message: 'Wrapping up and saving your model', type: 'default', minLevel: 'beginner' },
    { component: 'Manager', message: 'Collecting final metrics and artifacts', type: 'default', minLevel: 'intermediate' },
    { component: 'Observability', message: 'Research diary serialized to research_diary.json', type: 'success', minLevel: 'intermediate' },
    { component: 'Manager', message: 'Writing TrainedModel{weights_path, metrics, cost_breakdown}', type: 'default', minLevel: 'expert' },
    { component: 'Manager', message: 'Training pipeline complete', type: 'success', minLevel: 'beginner' },
  ],
];

// Seven iterations with rich reasoning metadata + search-space coordinates.
const ITERATIONS: Array<Omit<Iteration, 'id'>> = [
  {
    experiment: 'Decrease learning_rate 3e-4→1.5e-4 to reduce loss spikes.',
    diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015',
    loss: 0.289, f1: 0.871, status: 'KEPT',
    searchCoord: { lr: 1.5e-4, loraRank: 16 },
    reasoning: {
      hypothesis: 'Training loss is oscillating. Halving the learning rate should smooth out updates without stalling convergence.',
      strategy: 'local_perturbation',
      expectedImpact: 'val_loss should drop by 3–8%; no risk of catastrophic divergence.',
      rationale: 'val_loss dropped 0.312 → 0.289 (−7.4%), exceeding the +2% keep threshold. F1 held flat at 0.871.',
      lossBeforeAfter: [0.312, 0.289],
      f1BeforeAfter: [0.871, 0.871],
    },
  },
  {
    experiment: 'Increase lora_rank 16→32 to expand model capacity for task.',
    diff: '- lora_rank: 16\n+ lora_rank: 32',
    loss: 0.271, f1: 0.901, status: 'KEPT',
    searchCoord: { lr: 1.5e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'The model may be underfitting. Doubling LoRA rank gives it more capacity for the 42k-sample dataset.',
      strategy: 'random_search',
      expectedImpact: 'F1 should climb 2–5%; memory and cost increase ~15%.',
      rationale: 'F1 climbed 0.871 → 0.901 (+3.4%). Memory footprint grew 18% but stayed within budget.',
      lossBeforeAfter: [0.289, 0.271],
      f1BeforeAfter: [0.871, 0.901],
    },
  },
  {
    experiment: 'Increase learning_rate 1.5e-4→6.1e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00061',
    loss: 0.334, f1: 0.862, status: 'REVERTED',
    searchCoord: { lr: 6.1e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'Agent proposed a 4x learning-rate bump to escape a local minimum (Claude-driven).',
      strategy: 'claude_proposal',
      expectedImpact: 'Mixed — could either unlock fresh gradient directions or cause divergence.',
      rationale: 'loss spiked 0.271 → 0.334 (+23%), F1 dropped 0.901 → 0.862. Rolled back to iter_2 checkpoint.',
      lossBeforeAfter: [0.271, 0.334],
      f1BeforeAfter: [0.901, 0.862],
    },
  },
  {
    experiment: 'Increase warmup_steps 100→500 to stabilize early training.',
    diff: '- warmup_steps: 100\n+ warmup_steps: 500',
    loss: 0.248, f1: 0.914, status: 'KEPT',
    searchCoord: { lr: 1.5e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'Longer warmup should let the LR schedule find a more stable plateau before full-speed training.',
      strategy: 'local_perturbation',
      expectedImpact: 'Small but reliable loss drop; F1 improves slightly.',
      rationale: 'val_loss dropped 0.271 → 0.248 (−8.5%), F1 improved 0.901 → 0.914.',
      lossBeforeAfter: [0.271, 0.248],
      f1BeforeAfter: [0.901, 0.914],
    },
  },
  {
    experiment: 'Decrease dropout 0.1→0.06 (local ±20% perturbation).',
    diff: '- dropout: 0.1\n+ dropout: 0.06',
    loss: 0.243, f1: 0.917, status: 'KEPT',
    searchCoord: { lr: 1.5e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'With the higher LoRA rank the model has enough capacity; less dropout should help signal flow.',
      strategy: 'local_perturbation',
      expectedImpact: 'Marginal loss improvement, no overfitting signal yet.',
      rationale: 'val_loss dropped 0.248 → 0.243 (−2.0%). F1 inched up 0.914 → 0.917. No overfitting detected.',
      lossBeforeAfter: [0.248, 0.243],
      f1BeforeAfter: [0.914, 0.917],
    },
  },
  {
    experiment: 'Increase num_epochs 3→4 to allow longer convergence.',
    diff: '- num_epochs: 3\n+ num_epochs: 4',
    loss: 0.261, f1: 0.908, status: 'REVERTED',
    searchCoord: { lr: 1.5e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'Extra epoch might help the model converge further.',
      strategy: 'random_search',
      expectedImpact: 'Loss drops or stays flat; risk of overfitting.',
      rationale: 'val_loss rose 0.243 → 0.261 (+7.4%) and F1 dropped 0.917 → 0.908 — classic overfitting. Reverted.',
      lossBeforeAfter: [0.243, 0.261],
      f1BeforeAfter: [0.917, 0.908],
    },
  },
  {
    experiment: 'Decrease learning_rate 1.5e-4→1.2e-4 (local ±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00012',
    loss: 0.231, f1: 0.923, status: 'KEPT',
    searchCoord: { lr: 1.2e-4, loraRank: 32 },
    reasoning: {
      hypothesis: 'Fine-grained LR drop to squeeze the last bit of convergence out of the current config.',
      strategy: 'local_perturbation',
      expectedImpact: 'Tiny but positive movement; safe change.',
      rationale: 'val_loss dropped 0.243 → 0.231 (−4.9%), F1 climbed 0.917 → 0.923. Final best config.',
      lossBeforeAfter: [0.243, 0.231],
      f1BeforeAfter: [0.917, 0.923],
    },
  },
];

// ── Data preview factory ──────────────────────────────────────────────────
function buildDataPreview(prompt: string, taskType: TaskType): DataPreview {
  const lowered = prompt.toLowerCase();

  // Pick a plausible source+dataset based on prompt heuristics.
  let source: DataPreview['source'] = 'huggingface';
  let datasetName = 'imdb';
  let columns = ['text', 'label'];
  let samples: DataPreview['samples'] = [
    { text: 'A masterpiece — beautifully shot and emotionally honest.', label: 'positive' },
    { text: 'Plodding, derivative, and twenty minutes too long.', label: 'negative' },
    { text: 'The lead performance alone is worth the price of admission.', label: 'positive' },
    { text: 'I have never checked my phone more during a film.', label: 'negative' },
    { text: 'Smart, compact, and genuinely moving.', label: 'positive' },
    { text: 'The plot collapses under its own weight by act three.', label: 'negative' },
    { text: 'Not perfect, but refreshingly original.', label: 'positive' },
  ];
  let classDistribution: Record<string, number> = { positive: 21420, negative: 20580 };

  if (lowered.includes('spam') || lowered.includes('email')) {
    datasetName = 'enron_spam_filtered';
    samples = [
      { text: 'Your Amazon order has shipped. Tracking #: 1Z999AA10123456784', label: 'ham' },
      { text: 'URGENT: Claim your $5,000 prize now — click here!!!', label: 'spam' },
      { text: 'Hey, lunch tomorrow at noon?', label: 'ham' },
      { text: 'Viagra 90% off today only, no prescription needed', label: 'spam' },
      { text: 'Quarterly review meeting moved to Thursday 3pm.', label: 'ham' },
      { text: 'Congratulations you have been selected for our reward program', label: 'spam' },
      { text: 'Project update attached. Let me know if you have questions.', label: 'ham' },
    ];
    classDistribution = { ham: 32840, spam: 9160 };
  } else if (lowered.includes('news') || lowered.includes('topic')) {
    datasetName = 'ag_news';
    samples = [
      { text: 'Federal Reserve signals potential rate cut this quarter', label: 'Business' },
      { text: 'Lakers edge Celtics in triple overtime thriller', label: 'Sports' },
      { text: 'New quantum chip achieves 1000-qubit milestone', label: 'Sci/Tech' },
      { text: 'Diplomatic summit concludes with joint climate statement', label: 'World' },
      { text: 'Tech giant announces layoffs across engineering divisions', label: 'Business' },
      { text: 'Underdog team advances to World Cup semifinals', label: 'Sports' },
      { text: 'Breakthrough in room-temperature superconductor research', label: 'Sci/Tech' },
    ];
    classDistribution = { World: 10500, Sports: 10500, Business: 10500, 'Sci/Tech': 10500 };
  } else if (taskType === 'regression') {
    datasetName = 'california_housing';
    columns = ['median_income', 'house_age', 'avg_rooms', 'population', 'target_value'];
    samples = [
      { median_income: 8.32, house_age: 41, avg_rooms: 6.98, population: 322, target_value: 452600 },
      { median_income: 8.30, house_age: 21, avg_rooms: 6.24, population: 2401, target_value: 358500 },
      { median_income: 7.26, house_age: 52, avg_rooms: 8.29, population: 496, target_value: 352100 },
      { median_income: 5.64, house_age: 52, avg_rooms: 5.82, population: 558, target_value: 341300 },
      { median_income: 3.85, house_age: 52, avg_rooms: 6.28, population: 565, target_value: 342200 },
      { median_income: 4.03, house_age: 52, avg_rooms: 4.76, population: 413, target_value: 269700 },
      { median_income: 3.66, house_age: 52, avg_rooms: 4.93, population: 1094, target_value: 299200 },
    ];
    classDistribution = {};
  } else if (lowered.includes('medical') || lowered.includes('rare') || lowered.includes('legal')) {
    // Heuristic: narrow domain → fall back to synthetic.
    source = 'synthetic';
    datasetName = 'synth-domain-2026-04-22';
    samples = [
      { text: 'Patient presents with persistent dry cough and mild fever for five days.', label: 'viral_infection' },
      { text: 'Chest X-ray reveals bilateral infiltrates; oxygen saturation 92%.', label: 'pneumonia' },
      { text: 'Sudden onset severe headache with photophobia and neck stiffness.', label: 'meningitis' },
      { text: 'Intermittent abdominal pain localized to right lower quadrant.', label: 'appendicitis' },
      { text: 'Polyuria and polydipsia with unexplained weight loss over two weeks.', label: 'diabetes' },
      { text: 'Shortness of breath on exertion, ankle swelling, fatigue.', label: 'heart_failure' },
      { text: 'Recurrent joint pain in small joints of hands, worse in morning.', label: 'rheumatoid_arthritis' },
    ];
    classDistribution = { viral_infection: 500, pneumonia: 500, meningitis: 500, appendicitis: 500, diabetes: 500, heart_failure: 500, rheumatoid_arthritis: 500 };
  }

  const total = Object.values(classDistribution).reduce((a, b) => a + b, 0) || samples.length * 6000;
  const reasoning: string[] = [];

  if (source === 'huggingface') {
    reasoning.push(`Parsed prompt → task=\"${taskType}\", domain keywords extracted.`);
    reasoning.push(`Queried HuggingFace Hub with inferred search terms; scored 47 candidates.`);
    reasoning.push(`Ranked by (dataset_size × license_permissive × topic_coverage).`);
    reasoning.push(`Selected \"${datasetName}\" — top score 0.91; permissive license; well-balanced classes.`);
    reasoning.push(`Stratified 80/10/10 split; verified class distribution is within 5% of uniform.`);
  } else if (source === 'synthetic') {
    reasoning.push(`Parsed prompt → task=\"${taskType}\", domain is too narrow for reliable HF match.`);
    reasoning.push(`Fallback to synthetic generation using claude-haiku-4-5.`);
    reasoning.push(`Generated 3,500 candidate rows across 7 target classes.`);
    reasoning.push(`Ran dedup → deduped 127 near-duplicates.`);
    reasoning.push(`Validated with schema checks + LLM judge; discarded 238 rows that failed validation.`);
    reasoning.push(`Final dataset: 3,135 rows, diversity score 0.78.`);
  }

  const syntheticDetails = source === 'synthetic' ? {
    generationPrompt: `Generate realistic ${taskType} examples for the domain inferred from: \"${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}\". Cover the full label space. Vary length, tone, and surface form.`,
    diversityScore: 0.78,
    validationsPassed: 3135,
    validationsTotal: 3373,
  } : undefined;

  const trainSize = Math.round(total * 0.8);
  const valSize = Math.round(total * 0.1);
  const testSize = total - trainSize - valSize;

  return {
    source,
    datasetName,
    totalSamples: total,
    splits: { train: trainSize, val: valSize, test: testSize },
    columns,
    samples,
    classDistribution: Object.keys(classDistribution).length ? classDistribution : undefined,
    reasoning,
    syntheticDetails,
  };
}

// ── Model plan factory ─────────────────────────────────────────────────────
function buildModelPlan(dataPreview: DataPreview, budget: number): ModelPlan {
  const smallDataset = dataPreview.totalSamples < 5000;
  const approach: ModelPlan['approach'] = smallDataset ? 'full-finetune' : 'fine-tune-lora';
  const baseModel = dataPreview.source === 'synthetic' ? 'distilbert-base-uncased' : 'distilbert-base-uncased';

  return {
    baseModel,
    approach,
    reasoning: [
      `Dataset has ${dataPreview.totalSamples.toLocaleString()} samples → fine-tuning preferred over pre-training.`,
      `Budget $${budget} is below pre-training threshold ($400); forcing fine-tune path.`,
      approach === 'fine-tune-lora'
        ? `Using LoRA (r=16, alpha=32) — trains ~0.5% of params, keeps cost & VRAM low.`
        : `Dataset is small enough for full fine-tune; no LoRA overhead needed.`,
      `Base ${baseModel}: strong starting point for classification-style tasks.`,
    ],
    hyperparams: {
      base_model: baseModel,
      approach,
      learning_rate: '3e-4',
      batch_size: 32,
      num_epochs: 3,
      warmup_steps: 100,
      ...(approach === 'fine-tune-lora' ? { lora_rank: 16, lora_alpha: 32, lora_dropout: 0.1 } : {}),
    },
    estimatedCost: +(budget * 0.28).toFixed(2),
  };
}

// ──────────────────────────────────────────────────────────────────────────

export function useTrainingSimulation() {
  const [state, setState] = useState<TrainingState>({
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    skillLevel: 'intermediate',
    pendingApproval: null,
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: makeStages(),
  });

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const resumeRef = useRef<(() => void) | null>(null);
  // Snapshots of budget + skillLevel for timers to close over.
  const runCtxRef = useRef<{ budget: number; skillLevel: SkillLevel; prompt: string; taskType: TaskType }>({
    budget: 50, skillLevel: 'intermediate', prompt: '', taskType: 'classification',
  });

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
      const logs = [newLog, ...prev.logs].slice(0, 80);
      return { ...prev, logs };
    });
  }, []);

  // Emit logs for a stage, staggered across its duration.
  const emitStageLogs = useCallback((stageIdx: number, stageStart: number, dur: number) => {
    const logs = STAGE_LOGS[stageIdx];
    logs.forEach((log, li) => {
      schedule(
        () => appendLog({ component: log.component, message: log.message, type: log.type, minLevel: log.minLevel }),
        stageStart + (li * dur) / (logs.length + 1)
      );
    });
  }, [schedule, appendLog]);

  // Set a stage to in-progress, then complete.
  const runStage = useCallback((stageIdx: number, stageStart: number, dur: number) => {
    schedule(() => {
      setState(prev => ({
        ...prev,
        stages: prev.stages.map((s, i) => i === stageIdx ? { ...s, status: 'in-progress' } : s),
        stage: stageIdx,
      }));
    }, stageStart);
    emitStageLogs(stageIdx, stageStart, dur);
    schedule(() => {
      setState(prev => ({
        ...prev,
        stages: prev.stages.map((s, i) => i === stageIdx ? { ...s, status: 'complete' } : s),
      }));
    }, stageStart + dur - 50);
  }, [schedule, emitStageLogs]);

  // ── Phase 1: Manager Init + Data Discovery ──────────────────────────────
  const startPhase1 = useCallback(() => {
    const ctx = runCtxRef.current;
    let cursor = 0;

    runStage(0, cursor, STAGE_DURATIONS[0]);
    cursor += STAGE_DURATIONS[0];

    runStage(1, cursor, STAGE_DURATIONS[1]);
    cursor += STAGE_DURATIONS[1];

    // Cost ticks during stage 1 (small).
    const stage1Cost = ctx.budget * 0.04;
    const stage1Ticks = 3;
    for (let t = 0; t < stage1Ticks; t++) {
      const amount = stage1Cost / stage1Ticks;
      schedule(() => {
        setState(prev => ({ ...prev, costSpent: Math.min(+(prev.costSpent + amount).toFixed(2), prev.budget) }));
      }, STAGE_DURATIONS[0] + t * 500 + 300);
    }

    // After phase 1 finishes → emit data preview and either pause (Int/Exp) or auto-continue (Beginner).
    schedule(() => {
      const dataPreview = buildDataPreview(ctx.prompt, ctx.taskType);
      setState(prev => {
        const isBeginner = ctx.skillLevel === 'beginner';
        return {
          ...prev,
          dataPreview,
          status: isBeginner ? 'running' : 'awaiting-approval',
          pendingApproval: isBeginner ? null : 'dataset',
        };
      });
      if (ctx.skillLevel === 'beginner') {
        appendLog({ component: 'DataGen', message: 'Dataset auto-accepted (beginner mode)', type: 'success', minLevel: 'beginner' });
        startPhase2();
      } else {
        appendLog({ component: 'DataGen', message: 'Waiting for your approval of the dataset', type: 'default', minLevel: 'beginner' });
        resumeRef.current = startPhase2;
      }
    }, cursor);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runStage, schedule, appendLog]);

  // ── Phase 2: Model Selection ────────────────────────────────────────────
  const startPhase2 = useCallback(() => {
    const ctx = runCtxRef.current;
    let cursor = 0;
    runStage(2, cursor, STAGE_DURATIONS[2]);
    cursor += STAGE_DURATIONS[2];

    // Model cost ticks.
    const stage2Cost = ctx.budget * 0.06;
    const ticks = 3;
    for (let t = 0; t < ticks; t++) {
      const amount = stage2Cost / ticks;
      schedule(() => {
        setState(prev => ({ ...prev, costSpent: Math.min(+(prev.costSpent + amount).toFixed(2), prev.budget) }));
      }, t * 500 + 300);
    }

    // At end of phase 2: build + stash model plan; pause for Int/Exp approval.
    schedule(() => {
      setState(prev => {
        const dp = prev.dataPreview!;
        const modelPlan = buildModelPlan(dp, ctx.budget);
        const isBeginner = ctx.skillLevel === 'beginner';
        return {
          ...prev,
          modelPlan,
          status: isBeginner ? 'running' : 'awaiting-approval',
          pendingApproval: isBeginner ? null : 'model',
        };
      });
      if (ctx.skillLevel === 'beginner') {
        appendLog({ component: 'Decision', message: 'Model plan auto-accepted (beginner mode)', type: 'success', minLevel: 'beginner' });
        startPhase3();
      } else {
        appendLog({ component: 'Decision', message: 'Waiting for your approval of the model plan', type: 'default', minLevel: 'beginner' });
        resumeRef.current = startPhase3;
      }
    }, cursor);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runStage, schedule, appendLog]);

  // ── Phase 3: Training + AutoResearch + Finalization ─────────────────────
  const startPhase3 = useCallback(() => {
    const ctx = runCtxRef.current;
    let cursor = 0;

    // Training + AutoResearch + Finalization stages.
    [3, 4, 5].forEach((stageIdx, localIdx) => {
      const stageStart = cursor;
      const dur = STAGE_DURATIONS[stageIdx];
      runStage(stageIdx, stageStart, dur);

      // Metric updates during Training (3) + AutoResearch (4).
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

      // Cost during phase 3 stages.
      const totalBudgetUsed = ctx.budget * 0.66; // leaves ~10% already from phases 1+2
      const costPerStage = totalBudgetUsed / 3;
      const ticks = Math.floor(dur / 600);
      for (let t = 0; t < ticks; t++) {
        const amount = costPerStage / ticks;
        schedule(() => {
          setState(prev => ({ ...prev, costSpent: Math.min(+(prev.costSpent + amount).toFixed(2), prev.budget) }));
        }, stageStart + t * 600 + 300);
      }

      // AutoResearch iterations (stage 4).
      if (stageIdx === 4) {
        const iterDelay = dur / (ITERATIONS.length + 1);
        const resolveDelay = Math.min(iterDelay * 0.7, 800);
        ITERATIONS.forEach((iter, i) => {
          const appearAt = stageStart + iterDelay * (i + 1);
          // Step 1: add as PENDING.
          schedule(() => {
            const pending: Iteration = { ...iter, id: `iter-${i}`, status: 'PENDING' };
            setState(prev => ({ ...prev, iterations: [pending, ...prev.iterations] }));
          }, appearAt);
          // Step 2: resolve.
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

      cursor += dur;
      // silence unused warning in certain TS configs
      void localIdx;
    });

    // Mark complete.
    schedule(() => {
      setState(prev => ({ ...prev, status: 'complete', stage: 5 }));
    }, cursor + 100);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runStage, schedule]);

  // ── Public API ──────────────────────────────────────────────────────────
  const start = useCallback((prompt: string, budget: number, taskType: TaskType, skillLevel: SkillLevel) => {
    clearTimers();
    resumeRef.current = null;
    runCtxRef.current = { prompt, budget, taskType, skillLevel };
    setState({
      status: 'running',
      stage: 0,
      prompt,
      budget,
      taskType,
      skillLevel,
      pendingApproval: null,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
    });
    startPhase1();
  }, [startPhase1]);

  const approve = useCallback(() => {
    const next = resumeRef.current;
    resumeRef.current = null;
    setState(prev => ({ ...prev, status: 'running', pendingApproval: null }));
    if (next) next();
  }, []);

  const reject = useCallback(() => {
    clearTimers();
    resumeRef.current = null;
    setState(prev => ({
      ...prev,
      status: 'idle',
      stage: -1,
      pendingApproval: null,
    }));
  }, []);

  const reset = useCallback(() => {
    clearTimers();
    resumeRef.current = null;
    setState({
      status: 'idle',
      stage: -1,
      prompt: '',
      budget: 50,
      taskType: 'classification',
      skillLevel: 'intermediate',
      pendingApproval: null,
      costSpent: 0,
      metrics: [],
      iterations: [],
      logs: [],
      stages: makeStages(),
    });
  }, []);

  return { state, start, approve, reject, reset };
}
