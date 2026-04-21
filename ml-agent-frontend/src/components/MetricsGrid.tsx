import type { MetricPoint } from '../types';

interface Props {
  metrics: MetricPoint[];
  costSpent: number;
  budget?: number;
}

interface CardProps {
  label: string;
  value: string;
}

function MetricCard({ label, value }: CardProps) {
  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1rem',
      }}
      role="status"
      aria-label={`${label}: ${value}`}
    >
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '0.375rem' }}>{label}</p>
      <p style={{ fontSize: '24px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.5px' }}>
        {value}
      </p>
    </div>
  );
}

export default function MetricsGrid({ metrics, costSpent }: Props) {
  const last = metrics[metrics.length - 1];
  const loss = last ? last.loss.toFixed(4) : '—';
  const accuracy = last ? `${(last.accuracy * 100).toFixed(1)}%` : '—';
  const cost = `$${costSpent.toFixed(2)}`;
  const iter = `${metrics.length}/${Math.max(metrics.length, 12)}`;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '12px',
        marginBottom: '1.5rem',
      }}
      aria-label="Training metrics"
    >
      <MetricCard label="Training Loss" value={loss} />
      <MetricCard label="Val Accuracy" value={accuracy} />
      <MetricCard label="Cost Spent" value={cost} />
      <MetricCard label="Iterations" value={iter} />
    </div>
  );
}
