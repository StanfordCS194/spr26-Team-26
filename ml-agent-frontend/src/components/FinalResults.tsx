import type { TrainingState } from '../types';
import Tooltip from './Tooltip';

interface Props {
  state: TrainingState;
  onReset: () => void;
}

const RESULT_TOOLTIPS: Record<string, { label: string; body: string }> = {
  'Final F1': {
    label: 'Final F1 Score',
    body: 'Balances precision and recall into one number. Ranges from 0 to 1 — higher is better, especially useful when classes are imbalanced.',
  },
  'Val Accuracy': {
    label: 'Validation Accuracy',
    body: 'The model\'s correct-prediction rate on held-out data it has never seen. This is the more honest measure of real-world performance.',
  },
  'Total Cost': {
    label: 'Total Compute Cost',
    body: 'The full cost of this training run across all pipeline stages, billed against your original budget cap.',
  },
};

export default function FinalResults({ state, onReset }: Props) {
  const lastMetric = state.metrics[state.metrics.length - 1];
  const bestIter = state.iterations.find(i => i.status === 'KEPT') ?? state.iterations[0];

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

  const results = [
    { label: 'Final F1', value: bestIter ? bestIter.f1.toFixed(3) : '—' },
    { label: 'Val Accuracy', value: lastMetric ? `${(lastMetric.accuracy * 100).toFixed(1)}%` : '—' },
    { label: 'Total Cost', value: `$${state.costSpent.toFixed(2)}` },
  ];

  return (
    <section
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--success)',
        borderRadius: 'var(--radius)',
        padding: '1.5rem',
        marginBottom: '1.5rem',
        textAlign: 'center',
      }}
      aria-label="Training complete"
    >
      <div style={{ marginBottom: '1.25rem' }}>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.4rem',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '0.1em',
          color: 'var(--success)',
          background: 'var(--success-dim)',
          border: '0.5px solid var(--success)',
          borderRadius: '4px',
          padding: '3px 10px',
          marginBottom: '0.75rem',
        }}>
          ✓ TRAINING COMPLETE
        </span>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
          {state.prompt}
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        marginBottom: '1.5rem',
      }}>
        {results.map(({ label, value }) => {
          const tip = RESULT_TOOLTIPS[label];
          return (
            <div key={label} style={{
              background: 'var(--bg-elevated)',
              border: '0.5px solid var(--border)',
              borderRadius: '6px',
              padding: '0.75rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.25rem' }}>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{label}</p>
                {tip && <Tooltip label={tip.label} body={tip.body} />}
              </div>
              <p style={{ fontSize: '22px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</p>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={handleExportDiary}
            style={{
              padding: '0.5rem 1rem',
              background: 'var(--accent-dim)',
              border: '0.5px solid var(--accent)',
              borderRadius: '6px',
              color: 'var(--accent)',
              fontSize: '13px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
            aria-label="Export research diary as JSON"
          >
            Export Research Diary
          </button>
          <Tooltip
            label="Export Research Diary"
            body="Downloads a JSON file with every experiment, metric, and log event from this run. Useful for reproducibility and offline analysis."
            placement="top"
          />
        </span>
        <button
          onClick={() => alert('Deploy endpoint: configure in production')}
          style={{
            padding: '0.5rem 1rem',
            background: 'var(--success-dim)',
            border: '0.5px solid var(--success)',
            borderRadius: '6px',
            color: 'var(--success)',
            fontSize: '13px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
          aria-label="Deploy model"
        >
          Deploy Model
        </button>
        <button
          onClick={onReset}
          style={{
            padding: '0.5rem 1rem',
            background: 'transparent',
            border: '0.5px solid var(--border)',
            borderRadius: '6px',
            color: 'var(--text-secondary)',
            fontSize: '13px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
          aria-label="Try another training run"
        >
          Try Another
        </button>
      </div>
    </section>
  );
}
