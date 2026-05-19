import type { MetricPoint } from '../types';
import { formatPrimaryMetric, metricPointValue, primaryMetricLabel } from '../utils/metricDisplay';
import Tooltip from './Tooltip';

interface Props {
  metrics: MetricPoint[];
  costSpent: number;
  budget?: number;
}

const TOOLTIPS: Record<string, { label: string; body: string }> = {
  loss: {
    label: 'Training Loss',
    body: 'Measures how wrong the model\'s predictions are on training data. Lower is better — a steadily decreasing value means the model is learning.',
  },
  primary: {
    label: 'Primary Metric',
    body: 'The higher-is-better score returned by the evaluation step. Tinker SFT runs report the normalized primary score derived from validation loss.',
  },
  cost: {
    label: 'Budget Accounted',
    body: 'Amount counted against the run budget so far. Dry-run and no-spend runs use reserved or estimated budget, not provider-billed spend.',
  },
  iter: {
    label: 'Metric Points',
    body: 'How many metric points the backend has reported for this run. The API does not expose a planned total for every backend.',
  },
};

interface CardProps {
  label: string;
  value: string;
  tooltipKey: keyof typeof TOOLTIPS;
}

function MetricCard({ label, value, tooltipKey }: CardProps) {
  const tip = TOOLTIPS[tooltipKey];
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
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.375rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{label}</p>
        <Tooltip label={tip.label} body={tip.body} />
      </div>
      <p style={{ fontSize: '24px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.5px' }}>
        {value}
      </p>
    </div>
  );
}

export default function MetricsGrid({ metrics, costSpent }: Props) {
  const last = metrics[metrics.length - 1];
  const loss = last ? last.loss.toFixed(4) : '—';
  const scoreLabel = primaryMetricLabel(last?.primaryMetricLabel);
  const score = formatPrimaryMetric(metricPointValue(last), scoreLabel);
  const cost = `$${costSpent.toFixed(2)}`;
  const metricCount = metrics.length ? String(metrics.length) : '—';

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
      <MetricCard label="Training Loss" value={loss} tooltipKey="loss" />
      <MetricCard label={scoreLabel} value={score} tooltipKey="primary" />
      <MetricCard label="Budget Used" value={cost} tooltipKey="cost" />
      <MetricCard label="Metric Points" value={metricCount} tooltipKey="iter" />
    </div>
  );
}
