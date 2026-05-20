import type { Iteration, MetricPoint } from '../types';

const DEFAULT_PRIMARY_LABEL = 'Primary Score';

export function primaryMetricLabel(label?: string | null): string {
  return label?.trim() || DEFAULT_PRIMARY_LABEL;
}

export function metricPointValue(metric?: MetricPoint | null): number | null {
  if (!metric) return null;
  return metric.primaryMetric ?? metric.accuracy ?? null;
}

export function iterationMetricValue(iteration?: Iteration | null): number | null {
  if (!iteration) return null;
  return iteration.primaryMetric ?? iteration.f1 ?? null;
}

export function shortMetricLabel(label?: string | null): string {
  const normalized = primaryMetricLabel(label);
  if (normalized.toLowerCase().includes('primary')) return 'Score';
  if (normalized.toLowerCase().includes('accuracy')) return 'Accuracy';
  return normalized;
}

export function formatPrimaryMetric(value: number | null | undefined, label?: string | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const normalized = primaryMetricLabel(label).toLowerCase();
  if (normalized.includes('accuracy')) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(3);
}

