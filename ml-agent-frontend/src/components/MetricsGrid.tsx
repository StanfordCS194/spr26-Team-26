import type { MetricPoint } from '../types';
import Tooltip from './Tooltip';

interface Props {
  metrics: MetricPoint[];
  costSpent: number;
  budget?: number;
  showCostDetail?: boolean;
}

const TOOLTIPS: Record<string, { label: string; body: string }> = {
  loss: {
    label: 'Training Loss',
    body: 'Measures how wrong the model\'s predictions are on training data. Lower is better — a steadily decreasing value means the model is learning.',
  },
  accuracy: {
    label: 'Validation Accuracy',
    body: 'The model\'s correct-prediction rate on held-out data it has never seen. This is the more honest measure of real-world performance.',
  },
  cost: {
    label: 'Accumulated Cost',
    body: 'Total compute spend so far across all pipeline stages. Updates in real time so you can monitor burn against your budget.',
  },
  iter: {
    label: 'Experiment Iterations',
    body: 'How many hyperparameter configurations have been tried out of the planned total. More iterations generally improve the final model but cost more.',
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

export default function MetricsGrid({ metrics, costSpent, budget, showCostDetail = false }: Props) {
  const last = metrics[metrics.length - 1];
  const loss = last ? last.loss.toFixed(4) : '—';
  const accuracy = last ? `${(last.accuracy * 100).toFixed(1)}%` : '—';
  const cost = showCostDetail && budget
    ? `$${costSpent.toFixed(2)} / $${budget}`
    : `$${costSpent.toFixed(2)}`;
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
      <MetricCard label="Training Loss" value={loss} tooltipKey="loss" />
      <MetricCard label="Val Accuracy" value={accuracy} tooltipKey="accuracy" />
      <MetricCard label="Cost Spent" value={cost} tooltipKey="cost" />
      <MetricCard label="Iterations" value={iter} tooltipKey="iter" />
    </div>
  );
}
