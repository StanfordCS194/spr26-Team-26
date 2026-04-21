import type { TrainingState } from '../types';

interface Props {
  state: TrainingState;
  onReset: () => void;
}

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
        {[
          { label: 'Final F1', value: bestIter ? bestIter.f1.toFixed(3) : '—' },
          { label: 'Val Accuracy', value: lastMetric ? `${(lastMetric.accuracy * 100).toFixed(1)}%` : '—' },
          { label: 'Total Cost', value: `$${state.costSpent.toFixed(2)}` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--border)',
            borderRadius: '6px',
            padding: '0.75rem',
          }}>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</p>
            <p style={{ fontSize: '22px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</p>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
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
