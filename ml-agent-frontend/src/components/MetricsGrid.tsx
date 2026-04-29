import type { MetricPoint, SkillLevel } from '../types';
import Tooltip from './Tooltip';

interface Props {
  metrics: MetricPoint[];
  costSpent: number;
  budget?: number;
  skillLevel: SkillLevel;
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
  progress: {
    label: 'Training Progress',
    body: 'How far the training run has gotten — a friendly rollup that combines loss improvement and iteration count.',
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

export default function MetricsGrid({ metrics, costSpent, budget, skillLevel }: Props) {
  const last = metrics[metrics.length - 1];
  const loss = last ? last.loss.toFixed(4) : '—';
  const accuracy = last ? `${(last.accuracy * 100).toFixed(1)}%` : '—';
  const cost = `$${costSpent.toFixed(2)}`;
  const iter = `${metrics.length}/${Math.max(metrics.length, 12)}`;

  // Beginner-friendly progress: map loss down toward a 0–100 score.
  // Interpolates 0.42 (start) → 0% to 0.12 (great) → 100%, clamped.
  const progress = last
    ? Math.max(0, Math.min(100, Math.round(((0.42 - last.loss) / (0.42 - 0.12)) * 100)))
    : 0;
  const progressPct = last ? `${progress}%` : '—';
  const budgetPct = budget ? `${Math.round((costSpent / budget) * 100)}% of budget` : cost;
  void budgetPct;

  if (skillLevel === 'beginner') {
    // 3 friendly cards — no raw loss.
    return (
      <div
        className="metrics-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px',
          marginBottom: '1.5rem',
        }}
        aria-label="Training metrics"
      >
        <MetricCard label="Progress" value={progressPct} tooltipKey="progress" />
        <MetricCard label="Accuracy" value={accuracy} tooltipKey="accuracy" />
        <MetricCard label="Spent" value={cost} tooltipKey="cost" />
      </div>
    );
  }

  // Intermediate + Expert: full 4-card grid.
  return (
    <div
      className="metrics-grid"
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
