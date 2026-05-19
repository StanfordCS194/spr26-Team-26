import { resolveApiHref } from '../api/runs';
import type { TrainingState } from '../types';
import {
  formatPrimaryMetric,
  iterationMetricValue,
  metricPointValue,
  primaryMetricLabel,
  shortMetricLabel,
} from '../utils/metricDisplay';
import Tooltip from './Tooltip';

interface Props {
  state: TrainingState;
  onReset: () => void;
}

const RESULT_TOOLTIPS: Record<string, { label: string; body: string }> = {
  score: {
    label: 'Primary Metric',
    body: 'The higher-is-better score returned by the evaluation step. Tinker SFT runs report the normalized primary score derived from validation loss.',
  },
  'Budget Used': {
    label: 'Budget Accounted',
    body: 'The amount counted against this run\'s budget cap. Dry-run and no-spend runs use reserved or estimated budget, not provider-billed spend.',
  },
};
type ResultTooltipKey = keyof typeof RESULT_TOOLTIPS;

export default function FinalResults({ state, onReset }: Props) {
  const lastMetric = state.metrics[state.metrics.length - 1];
  const bestIter = state.iterations.find(i => i.status === 'KEPT') ?? state.iterations[0];
  const artifactFiles = state.artifacts?.files.filter(file => file.exists) ?? [];
  const checkpointEntries = Object.entries(state.artifacts?.checkpoints ?? {}).filter(([, value]) => value);
  const bestLabel = primaryMetricLabel(bestIter?.primaryMetricLabel ?? lastMetric?.primaryMetricLabel);
  const finalScoreLabel = `Final ${shortMetricLabel(bestLabel)}`;
  const latestLabel = primaryMetricLabel(lastMetric?.primaryMetricLabel ?? bestLabel);
  const finalScoreValue = bestIter ? iterationMetricValue(bestIter) : metricPointValue(lastMetric);
  const diaryArtifact = artifactFiles.find(file => file.name === 'diary' && file.downloadPath);
  const diaryHref = resolveApiHref(diaryArtifact?.downloadPath);

  const handleExportSnapshot = () => {
    const snapshot = {
      prompt: state.prompt,
      budget: state.budget,
      taskType: state.taskType,
      dataPath: state.dataPath ?? null,
      costSpent: state.costSpent,
      iterations: state.iterations,
      metrics: state.metrics,
      logs: state.logs,
      artifacts: state.artifacts ?? null,
      result: state.result ?? null,
    };
    if (state.provenance) {
      Object.assign(snapshot, { provenance: state.provenance });
    }
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'run-snapshot.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const results: Array<{ key: string; label: string; value: string; tooltipKey: ResultTooltipKey }> = [
    {
      key: 'final-score',
      label: finalScoreLabel,
      value: formatPrimaryMetric(finalScoreValue, bestLabel),
      tooltipKey: 'score',
    },
    {
      key: 'latest-primary-score',
      label: latestLabel,
      value: formatPrimaryMetric(metricPointValue(lastMetric), latestLabel),
      tooltipKey: 'score',
    },
    { key: 'budget-used', label: 'Budget Used', value: `$${state.costSpent.toFixed(2)}`, tooltipKey: 'Budget Used' },
  ];
  const compactPath = (value?: string | null) => {
    if (!value) return '—';
    if (value.length <= 72) return value;
    return `…${value.slice(-69)}`;
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
        {results.map(({ key, label, value, tooltipKey }) => {
          const tip = RESULT_TOOLTIPS[tooltipKey];
          return (
            <div key={key} style={{
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

      {state.artifacts && (
        <div style={{
          borderTop: '0.5px solid var(--border)',
          paddingTop: '1rem',
          marginBottom: '1.5rem',
          textAlign: 'left',
        }}>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Artifacts
          </p>
          <div style={{ display: 'grid', gap: '0.4rem' }}>
            {state.artifacts.modelPath && (
              <div style={{ display: 'grid', gridTemplateColumns: '96px 1fr', gap: '0.75rem', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Model</span>
                <code style={{ fontSize: '11px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {compactPath(state.artifacts.modelPath)}
                </code>
              </div>
            )}
            {artifactFiles.map(file => {
              const href = resolveApiHref(file.downloadPath);
              return (
                <div
                  key={file.downloadPath ?? file.path ?? file.name}
                  style={{ display: 'grid', gridTemplateColumns: '96px 1fr auto', gap: '0.75rem', alignItems: 'center' }}
                >
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{file.label}</span>
                  <code style={{ fontSize: '11px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {compactPath(file.path)}
                  </code>
                  {href && (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      style={{ fontSize: '11px', color: 'var(--accent)', textDecoration: 'none' }}
                    >
                      Open
                    </a>
                  )}
                </div>
              );
            })}
            {checkpointEntries.map(([key, value]) => (
              <div key={key} style={{ display: 'grid', gridTemplateColumns: '96px 1fr', gap: '0.75rem', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{key}</span>
                <code style={{ fontSize: '11px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {String(value)}
                </code>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
          {diaryHref ? (
            <a
              href={diaryHref}
              target="_blank"
              rel="noreferrer"
              style={{
                padding: '0.5rem 1rem',
                background: 'var(--accent-dim)',
                border: '0.5px solid var(--accent)',
                borderRadius: '6px',
                color: 'var(--accent)',
                fontSize: '13px',
                cursor: 'pointer',
                fontFamily: 'inherit',
                textDecoration: 'none',
              }}
              aria-label="Open backend research diary artifact"
            >
              Open Research Diary
            </a>
          ) : (
            <button
              onClick={handleExportSnapshot}
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
              aria-label="Export run snapshot as JSON"
            >
              Export Run Snapshot
            </button>
          )}
          <Tooltip
            label={diaryHref ? 'Research Diary' : 'Run Snapshot'}
            body={
              diaryHref
                ? 'Opens the backend research diary artifact produced by AutoResearch for this run.'
                : 'Exports the visible run state as JSON. This is a UI snapshot, not a backend research diary artifact.'
            }
            placement="top"
          />
        </span>
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
