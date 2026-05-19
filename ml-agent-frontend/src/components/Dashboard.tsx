import { resolveApiHref } from '../api/runs';
import type { TrainingState } from '../types';
import PipelineProgress from './PipelineProgress';
import MetricsGrid from './MetricsGrid';
import MetricsChart from './MetricsChart';
import IterationsList from './IterationsList';
import ActivityLog from './ActivityLog';
import FinalResults from './FinalResults';
import {
  formatPrimaryMetric,
  iterationMetricValue,
  metricPointValue,
  primaryMetricLabel,
  shortMetricLabel,
} from '../utils/metricDisplay';

interface Props {
  state: TrainingState;
  onReset: () => void;
  onCancel: () => void;
}

const headerStyle: React.CSSProperties = {
  position: 'sticky',
  top: 0,
  zIndex: 10,
  background: 'var(--bg-base)',
  borderBottom: '0.5px solid var(--border)',
  padding: '0.75rem 1.5rem',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
};

function StatusDot({ status }: { status: TrainingState['status'] }) {
  const color =
    status === 'complete'
      ? 'var(--success)'
      : status === 'failed'
        ? 'var(--danger)'
        : status === 'running' || status === 'cancelling'
          ? 'var(--accent)'
          : 'var(--text-muted)';
  const label =
    status === 'complete'
      ? 'COMPLETE'
      : status === 'failed'
        ? 'FAILED'
        : status === 'cancelled'
          ? 'CANCELLED'
          : status === 'cancelling'
            ? 'CANCELLING'
            : status === 'running'
          ? 'RUNNING'
          : 'IDLE';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          display: 'inline-block',
          animation: status === 'running' || status === 'cancelling' ? 'pulse 1.5s ease-in-out infinite' : 'none',
        }}
      />
      <span style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.1em', color }}>
        {label}
      </span>
    </div>
  );
}

function titleCaseToken(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
}

function spendModeLabel(value?: string | null) {
  switch (value) {
    case 'no_spend':
      return 'NO_SPEND';
    case 'dry_run':
      return 'Dry-run';
    case 'budget_skipped':
      return 'Budget skipped';
    case 'live':
      return 'Live services';
    case 'local':
      return 'Local';
    default:
      return value ? titleCaseToken(value) : null;
  }
}

function backendLabel(value?: string | null) {
  const normalized = value?.replace('-', '_').toLowerCase();
  if (normalized === 'dry_run') return 'Tinker dry-run';
  if (normalized === 'tinker' || normalized === 'tinker_sft') return 'Live Tinker';
  return value ? titleCaseToken(value) : null;
}

function ProvenanceBadges({ state }: { state: TrainingState }) {
  const provenance = state.provenance;
  if (!provenance) return null;

  const badges = Array.from(new Set([
    spendModeLabel(provenance.spendMode),
    backendLabel(provenance.trainingBackend),
    provenance.dataMode ? `Mode ${provenance.dataMode}` : null,
    provenance.modeCFallback ? `${titleCaseToken(provenance.modeCFallback)} fallback` : null,
    provenance.budgetPreflightSkipped ? 'Budget skipped' : null,
    provenance.liveServices.length > 0 ? `Live: ${provenance.liveServices.join(', ')}` : null,
  ].filter((label): label is string => Boolean(label))));

  if (badges.length === 0) return null;

  return (
    <div
      aria-label="Run provenance"
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.4rem',
        marginTop: '-0.75rem',
        marginBottom: '1.25rem',
      }}
    >
      {badges.map(label => (
        <span
          key={label}
          style={{
            fontSize: '11px',
            color: 'var(--text-secondary)',
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--border)',
            borderRadius: '4px',
            padding: '2px 8px',
            lineHeight: 1.6,
          }}
        >
          {label}
        </span>
      ))}
    </div>
  );
}

function TerminalArtifacts({ state, onReset }: Props) {
  const isFailed = state.status === 'failed';
  const tone = isFailed ? 'var(--danger)' : 'var(--warning)';
  const lastMetric = state.metrics[state.metrics.length - 1];
  const bestIter = state.iterations.find(i => i.status === 'KEPT') ?? state.iterations[0];
  const artifactFiles = state.artifacts?.files.filter(file => file.exists) ?? [];
  const checkpointEntries = Object.entries(state.artifacts?.checkpoints ?? {}).filter(([, value]) => value);
  const bestLabel = primaryMetricLabel(bestIter?.primaryMetricLabel ?? lastMetric?.primaryMetricLabel);
  const latestLabel = primaryMetricLabel(lastMetric?.primaryMetricLabel ?? bestLabel);
  const compactPath = (value?: string | null) => {
    if (!value) return '—';
    if (value.length <= 72) return value;
    return `…${value.slice(-69)}`;
  };
  const results = [
    {
      label: `Checkpoint ${shortMetricLabel(bestLabel)}`,
      value: formatPrimaryMetric(iterationMetricValue(bestIter), bestLabel),
    },
    {
      label: latestLabel,
      value: formatPrimaryMetric(metricPointValue(lastMetric), latestLabel),
    },
    { label: 'Budget Used', value: `$${state.costSpent.toFixed(2)}` },
  ];

  return (
    <section
      style={{
        background: 'var(--bg-surface)',
        border: `0.5px solid ${tone}`,
        borderRadius: 'var(--radius)',
        padding: '1.5rem',
        marginBottom: '1.5rem',
      }}
      aria-label={isFailed ? 'Failed run artifacts' : 'Cancelled run artifacts'}
    >
      <div style={{ marginBottom: '1.25rem', textAlign: 'center' }}>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.4rem',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '0.1em',
          color: tone,
          background: 'var(--bg-elevated)',
          border: `0.5px solid ${tone}`,
          borderRadius: '4px',
          padding: '3px 10px',
          marginBottom: '0.75rem',
        }}>
          {isFailed ? 'FAILED RUN ARTIFACTS' : 'CANCELLED RUN ARTIFACTS'}
        </span>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
          {isFailed
            ? 'This run failed before completion. Any checkpoint metrics and downloadable files produced before failure are still available.'
            : 'This run was cancelled before completion. The latest checkpoint metrics and downloadable files are still available.'}
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        marginBottom: '1.5rem',
      }}>
        {results.map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--border)',
            borderRadius: '6px',
            padding: '0.75rem',
            textAlign: 'center',
          }}>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</p>
            <p style={{ fontSize: '22px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</p>
          </div>
        ))}
      </div>

      <div style={{
        borderTop: '0.5px solid var(--border)',
        paddingTop: '1rem',
        marginBottom: '1.5rem',
      }}>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
          Available Files
        </p>
        <div style={{ display: 'grid', gap: '0.4rem' }}>
          {state.artifacts?.modelPath && (
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
                key={file.name}
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

      <div style={{ display: 'flex', justifyContent: 'center' }}>
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

export default function Dashboard({ state, onReset, onCancel }: Props) {
  const canCancel = state.status === 'running' || state.status === 'cancelling';
  const hasCancelledArtifacts = state.status === 'cancelled' && Boolean(state.artifacts);
  const hasTerminalArtifacts = (state.status === 'cancelled' || state.status === 'failed') && Boolean(state.artifacts);

  return (
    <div style={{ minHeight: '100vh' }}>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

      {/* Header */}
      <header style={headerStyle}>
        <div>
          <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
            ML Training Agent
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '0.75rem' }}>
            CS194 · Team 26
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <StatusDot status={state.status} />
          {canCancel ? (
            <button
              onClick={onCancel}
              disabled={state.status === 'cancelling'}
              style={{
                padding: '4px 10px',
                background: 'transparent',
                border: '0.5px solid var(--warning)',
                borderRadius: '6px',
                color: 'var(--warning)',
                fontSize: '12px',
                cursor: state.status === 'cancelling' ? 'not-allowed' : 'pointer',
                opacity: state.status === 'cancelling' ? 0.7 : 1,
                fontFamily: 'inherit',
              }}
              aria-label="Cancel training"
            >
              {state.status === 'cancelling' ? 'Cancelling...' : 'Cancel'}
            </button>
          ) : (
            <button
              onClick={onReset}
              style={{
                padding: '4px 10px',
                background: 'transparent',
                border: '0.5px solid var(--border)',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                fontSize: '12px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
              aria-label="Reset training"
            >
              Reset
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '1.5rem' }}>
        {/* Prompt summary */}
        <div style={{
          background: 'var(--bg-surface)',
          border: '0.5px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '0.75rem 1rem',
          marginBottom: '1.5rem',
          display: 'flex',
          gap: '1rem',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)', flexShrink: 0 }}>Objective</span>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', flex: 1 }}>{state.prompt}</span>
          {state.dataPath && (
            <span style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              border: '0.5px solid var(--border)',
              borderRadius: '4px',
              padding: '2px 8px',
              maxWidth: '320px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flexShrink: 1,
            }}>
              {state.dataPath}
            </span>
          )}
          <span style={{
            fontSize: '11px',
            color: 'var(--accent)',
            background: 'var(--accent-dim)',
            border: '0.5px solid var(--accent)',
            borderRadius: '4px',
            padding: '2px 8px',
            flexShrink: 0,
          }}>
            {state.taskType}
          </span>
        </div>

        <ProvenanceBadges state={state} />

        {/* Pipeline + budget */}
        <PipelineProgress stages={state.stages} costSpent={state.costSpent} budget={state.budget} />

        {state.status === 'failed' && (
          <section
            style={{
              background: 'var(--danger-dim)',
              border: '0.5px solid var(--danger)',
              borderRadius: 'var(--radius)',
              padding: '0.875rem 1rem',
              marginBottom: '1.5rem',
            }}
            role="alert"
          >
            <p style={{ fontSize: '12px', fontWeight: 600, color: 'var(--danger)', marginBottom: '0.25rem' }}>
              Run failed
            </p>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {state.error ?? 'The backend run ended before producing a model.'}
            </p>
          </section>
        )}

        {state.status === 'cancelled' && (
          <section
            style={{
              background: 'var(--bg-surface)',
              border: '0.5px solid var(--warning)',
              borderRadius: 'var(--radius)',
              padding: '0.875rem 1rem',
              marginBottom: '1.5rem',
            }}
            role="status"
          >
            <p style={{ fontSize: '12px', fontWeight: 600, color: 'var(--warning)', marginBottom: '0.25rem' }}>
              Run cancelled
            </p>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {hasCancelledArtifacts
                ? 'The backend stopped the training pipeline, but checkpoint and artifact files are available below.'
                : 'The backend stopped the training pipeline at the next safe checkpoint.'}
            </p>
          </section>
        )}

        {/* 4 metric cards */}
        <MetricsGrid metrics={state.metrics} costSpent={state.costSpent} budget={state.budget} />

        {/* 2-col: chart + iterations */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
          marginBottom: '1.5rem',
        }}>
          <MetricsChart metrics={state.metrics} />
          <IterationsList iterations={state.iterations} />
        </div>

        {/* Activity log */}
        <ActivityLog logs={state.logs} />

        {/* Final results */}
        {state.status === 'complete' && (
          <FinalResults state={state} onReset={onReset} />
        )}
        {hasTerminalArtifacts && <TerminalArtifacts state={state} onReset={onReset} onCancel={onCancel} />}
      </main>
    </div>
  );
}
