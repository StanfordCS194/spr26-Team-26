import type { TrainingState } from '../types';
import { matchTask } from '../utils/matchTask';
import type { TaskConfig } from '../data/taskConfigs';

interface Props {
  state: TrainingState;
  onReset: () => void;
}

// ─── Plain-language summary generator ────────────────────────────────────────

function qualityLabel(metric: number, evalMetric: string): { grade: string; color: string; meaning: string } {
  if (evalMetric === 'BLEU') {
    if (metric >= 0.32) return { grade: 'Excellent',      color: '#22c55e', meaning: 'near human-level translation quality' };
    if (metric >= 0.25) return { grade: 'Good',           color: '#84cc16', meaning: 'fluent translations with occasional errors' };
    if (metric >= 0.18) return { grade: 'Solid',          color: '#eab308', meaning: 'mostly intelligible, some awkward phrasing' };
    return               { grade: 'Developing',           color: '#f97316', meaning: 'understandable but needs refinement' };
  }
  if (evalMetric === 'ROUGE-L') {
    if (metric >= 0.42) return { grade: 'Excellent',      color: '#22c55e', meaning: 'summaries capture key facts with high fidelity' };
    if (metric >= 0.35) return { grade: 'Good',           color: '#84cc16', meaning: 'summaries are coherent and mostly accurate' };
    if (metric >= 0.28) return { grade: 'Solid',          color: '#eab308', meaning: 'summaries convey main points, some detail lost' };
    return               { grade: 'Developing',           color: '#f97316', meaning: 'summaries capture the gist but miss detail' };
  }
  // accuracy / F1 / reward_score
  if (metric >= 0.92) return { grade: 'Excellent',        color: '#22c55e', meaning: 'production-ready for most applications' };
  if (metric >= 0.85) return { grade: 'Strong',           color: '#84cc16', meaning: 'solid performance; ready for real-world use' };
  if (metric >= 0.75) return { grade: 'Good',             color: '#eab308', meaning: 'good baseline; more data could push it further' };
  return               { grade: 'Developing',             color: '#f97316', meaning: 'promising start; needs more training or data' };
}

function practicalLine(cfg: TaskConfig, finalMetric: number): string {
  const pct = (finalMetric * 100).toFixed(1);
  switch (cfg.evalMetric) {
    case 'accuracy':
      return `The model correctly ${cfg.taskLabel.toLowerCase().includes('classif') ? 'classifies' : 'predicts'} about ${pct} out of every 100 examples — that's the real-world accuracy on data it has never seen.`;
    case 'F1':
      return `An F1 of ${pct}% means the model balances precision and recall well — it catches most true cases while keeping false positives low.`;
    case 'ROUGE-L':
      return `A ROUGE-L of ${(finalMetric).toFixed(3)} means the model's summaries share substantial word-sequence overlap with human-written references — a strong proxy for readability and factual coverage.`;
    case 'BLEU':
      return `A BLEU score of ${(finalMetric).toFixed(3)} puts this model in the range of fluent machine translation — sentences are grammatically correct and semantically accurate the majority of the time.`;
    case 'reward_score':
      return `A reward score of ${pct}% means human raters — or a reward model trained on human preferences — prefer this model's responses ${pct}% of the time.`;
    case 'perplexity':
      return `A lower perplexity score (${pct}) indicates the model assigns high probability to held-out conversation data — it has learned natural dialogue patterns well.`;
    default:
      return `The model scored ${pct}% on the validation set — a strong indicator of real-world performance on this task.`;
  }
}

function improvementLine(cfg: TaskConfig, keptCount: number, totalCount: number): string {
  const bl = (cfg.baseline.metric * 100).toFixed(1);
  const fn = (cfg.final.metric * 100).toFixed(1);
  const isRaw = cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L';
  const blRaw = cfg.baseline.metric.toFixed(3);
  const fnRaw = cfg.final.metric.toFixed(3);
  const score = isRaw ? `${blRaw} → ${fnRaw}` : `${bl}% → ${fn}%`;
  return `AutoResearch ran ${totalCount} experiments, kept ${keptCount} improvements, and pushed ${cfg.evalMetric} from ${score} — a ${isRaw
    ? ((cfg.final.metric - cfg.baseline.metric) / cfg.baseline.metric * 100).toFixed(0)
    : ((cfg.final.metric - cfg.baseline.metric) / cfg.baseline.metric * 100).toFixed(0)}% relative gain over the starting config.`;
}

function deployLine(cfg: TaskConfig): string {
  if (cfg.strategy === 'fine-tune' && cfg.trainingType === 'SFT') {
    return `The fine-tuned LoRA adapter (rank ${cfg.loraRank}) is lightweight — it can be merged into ${cfg.model} and served at near-baseline inference cost.`;
  }
  return `The trained model is ready to serve predictions via a standard inference API.`;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function FinalResults({ state, onReset }: Props) {
  const cfg: TaskConfig = matchTask(state.prompt);
  const lastMetric = state.metrics[state.metrics.length - 1];
  const finalMetric = lastMetric?.accuracy ?? cfg.final.metric;
  const finalLoss   = lastMetric?.loss     ?? cfg.final.loss;

  const keptIters     = state.iterations.filter(i => i.status === 'KEPT');
  const revertedIters = state.iterations.filter(i => i.status === 'REVERTED');
  const topKept       = [...keptIters].slice(0, 3);

  const quality = qualityLabel(finalMetric, cfg.evalMetric);

  const handleExportDiary = () => {
    const diary = {
      prompt: state.prompt,
      budget: state.budget,
      taskType: state.taskType,
      costSpent: state.costSpent,
      iterations: state.iterations,
      metrics: state.metrics,
      logs: state.logs,
    };
    const blob = new Blob([JSON.stringify(diary, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research-diary.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section style={{
      marginTop: '1.5rem',
      marginBottom: '1.5rem',
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>

      {/* ── Header badge ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 16px',
        background: '#0d1f14',
        border: '0.5px solid var(--success)',
        borderRadius: 'var(--radius)',
      }}>
        <span style={{
          fontSize: 11, fontWeight: 700, letterSpacing: '0.1em',
          color: 'var(--success)', background: 'var(--success-dim)',
          border: '0.5px solid var(--success)', borderRadius: 4,
          padding: '2px 8px',
        }}>✓ TRAINING COMPLETE</span>
        <span style={{ fontSize: 13, color: 'var(--text-secondary)', flex: 1 }}>{state.prompt}</span>
        <span style={{
          fontSize: 12, fontWeight: 600,
          color: quality.color,
          background: quality.color + '18',
          border: `0.5px solid ${quality.color}`,
          borderRadius: 4,
          padding: '2px 10px',
          whiteSpace: 'nowrap',
        }}>{quality.grade}</span>
      </div>

      {/* ── Metric summary cards ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 10,
      }}>
        {[
          { label: cfg.metricLabel,   value: cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L' ? finalMetric.toFixed(3) : `${(finalMetric * 100).toFixed(1)}%` },
          { label: 'Final Loss',      value: finalLoss.toFixed(3) },
          { label: 'Experiments Run', value: String(state.iterations.length) },
          { label: 'Total Cost',      value: `$${state.costSpent.toFixed(2)}` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-surface)',
            border: '0.5px solid var(--border)',
            borderRadius: 8,
            padding: '12px 14px',
            textAlign: 'center',
          }}>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</p>
            <p style={{ fontSize: 22, fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</p>
          </div>
        ))}
      </div>

      {/* ── Plain-language summary ── */}
      <div style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1.25rem 1.5rem',
      }}>
        <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12 }}>
          Model Summary
        </p>

        {/* Quality headline */}
        <p style={{ fontSize: 15, fontWeight: 500, color: quality.color, marginBottom: 10 }}>
          {quality.grade} — {quality.meaning}
        </p>

        {/* Prose paragraphs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {practicalLine(cfg, finalMetric)}
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {improvementLine(cfg, keptIters.length, state.iterations.length)}
            {revertedIters.length > 0 && ` ${revertedIters.length} experiment${revertedIters.length > 1 ? 's' : ''} were reverted after causing regressions, keeping the model on the optimal path.`}
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {deployLine(cfg)}
          </p>
        </div>
      </div>

      {/* ── Top improvements ── */}
      {topKept.length > 0 && (
        <div style={{
          background: 'var(--bg-surface)',
          border: '0.5px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '1.25rem 1.5rem',
        }}>
          <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12 }}>
            Key Improvements Found
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {topKept.map((iter, i) => (
              <div key={iter.id} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <span style={{
                  flexShrink: 0,
                  fontSize: 10, fontWeight: 700,
                  color: 'var(--success)',
                  background: 'var(--success-dim)',
                  border: '0.5px solid var(--success)',
                  borderRadius: 4,
                  padding: '1px 6px',
                  marginTop: 2,
                }}>#{i + 1}</span>
                <div>
                  <p style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 2 }}>{iter.experiment}</p>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                    {cfg.evalMetric}: {cfg.evalMetric === 'BLEU' || cfg.evalMetric === 'ROUGE-L'
                      ? iter.f1.toFixed(3)
                      : `${(iter.f1 * 100).toFixed(1)}%`}
                    {' · '}loss: {iter.loss.toFixed(3)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Actions ── */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button onClick={handleExportDiary} style={{
          padding: '0.5rem 1.1rem',
          background: 'var(--accent-dim)',
          border: '0.5px solid var(--accent)',
          borderRadius: 6,
          color: 'var(--accent)',
          fontSize: 13,
          cursor: 'pointer',
          fontFamily: 'inherit',
        }}>
          Export Research Diary
        </button>
        <button onClick={() => alert('Deploy endpoint: configure in production')} style={{
          padding: '0.5rem 1.1rem',
          background: 'var(--success-dim)',
          border: '0.5px solid var(--success)',
          borderRadius: 6,
          color: 'var(--success)',
          fontSize: 13,
          cursor: 'pointer',
          fontFamily: 'inherit',
        }}>
          Deploy Model
        </button>
        <button onClick={onReset} style={{
          padding: '0.5rem 1.1rem',
          background: 'transparent',
          border: '0.5px solid var(--border)',
          borderRadius: 6,
          color: 'var(--text-secondary)',
          fontSize: 13,
          cursor: 'pointer',
          fontFamily: 'inherit',
        }}>
          Try Another
        </button>
      </div>
    </section>
  );
}
