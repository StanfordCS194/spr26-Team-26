import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import type { MetricPoint } from '../types';
import { formatPrimaryMetric, metricPointValue, primaryMetricLabel } from '../utils/metricDisplay';
import TooltipInfo from './Tooltip';

interface Props {
  metrics: MetricPoint[];
}

const tooltipStyle: React.CSSProperties = {
  background: 'var(--bg-elevated)',
  border: '0.5px solid var(--border)',
  borderRadius: '6px',
  fontSize: '12px',
  color: 'var(--text-primary)',
};

export default function MetricsChart({ metrics }: Props) {
  const latestLabel = primaryMetricLabel(metrics[metrics.length - 1]?.primaryMetricLabel);
  const data = metrics.map(m => ({
    iter: m.iteration,
    loss: m.loss,
    primaryScore: metricPointValue(m) ?? 0,
  }));

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1rem',
        height: '260px',
      }}
      aria-label="Training metrics chart"
    >
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.75rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Training Curves</p>
        <TooltipInfo
          label="Training Curves"
          body="Plots loss and the primary evaluation score together over time. Tinker SFT score is normalized from validation loss, so higher is better."
        />
      </div>
      {data.length === 0 ? (
        <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Awaiting training data…</span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 4, right: 16, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="iter"
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ value: 'Iteration', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 10 }}
            />
            <YAxis
              yAxisId="loss"
              domain={[0, 0.5]}
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
            />
            <YAxis
              yAxisId="acc"
              orientation="right"
              domain={[0, 1]}
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              tickFormatter={v => formatPrimaryMetric(Number(v), latestLabel)}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={v => `Iteration ${v}`}
              formatter={(value, name) => {
                const v = Number(value);
                return name === 'loss'
                  ? [v.toFixed(4), 'Loss']
                  : [formatPrimaryMetric(v, latestLabel), latestLabel];
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', color: 'var(--text-muted)' }}
              formatter={(value) => value === 'loss' ? 'Training Loss' : latestLabel}
            />
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="loss"
              stroke="var(--accent)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="acc"
              type="monotone"
              dataKey="primaryScore"
              stroke="var(--success)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
