import type { TrainingState } from '../types';
import PipelineProgress from './PipelineProgress';
import MetricsGrid from './MetricsGrid';
import MetricsChart from './MetricsChart';
import IterationsList from './IterationsList';
import ActivityLog from './ActivityLog';
import FinalResults from './FinalResults';

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

export default function Dashboard({ state, onReset, onCancel }: Props) {
  const canCancel = state.status === 'running' || state.status === 'cancelling';

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
              The backend stopped the training pipeline at the next safe checkpoint.
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
      </main>
    </div>
  );
}
