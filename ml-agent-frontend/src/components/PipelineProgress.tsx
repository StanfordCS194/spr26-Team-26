import type { PipelineStage } from '../types';
import Tooltip from './Tooltip';

interface Props {
  stages: PipelineStage[];
  costSpent: number;
  budget: number;
}

function StageIcon({ status }: { status: PipelineStage['status'] }) {
  const base: React.CSSProperties = {
    width: 24,
    height: 24,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '11px',
    fontWeight: 600,
    flexShrink: 0,
    transition: 'all 0.3s',
  };
  if (status === 'complete') {
    return (
      <span style={{ ...base, background: 'var(--success-dim)', color: 'var(--success)', border: '0.5px solid var(--success)' }}>
        ✓
      </span>
    );
  }
  if (status === 'in-progress') {
    return (
      <span style={{ ...base, background: 'var(--accent-dim)', color: 'var(--accent)', border: '0.5px solid var(--accent)' }}>
        ●
      </span>
    );
  }
  return (
    <span style={{ ...base, background: 'transparent', color: 'var(--text-muted)', border: '0.5px solid var(--border)' }}>
      ○
    </span>
  );
}

export default function PipelineProgress({ stages, costSpent, budget }: Props) {
  const pct = budget > 0 ? Math.min((costSpent / budget) * 100, 100) : 0;
  const barColor = pct >= 70 ? 'var(--danger)' : pct >= 50 ? 'var(--warning)' : 'var(--accent)';

  return (
    <section style={{ marginBottom: '1.5rem' }}>
      {/* Section label */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.625rem' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Training Pipeline</span>
        <Tooltip
          label="Training Pipeline Stages"
          body="Shows which step of the workflow is currently running. Stages run in order — each one must finish before the next begins."
        />
      </div>

      {/* Stage timeline */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0',
          overflowX: 'auto',
          paddingBottom: '0.5rem',
        }}
        role="list"
        aria-label="Pipeline stages"
      >
        {stages.map((stage, i) => (
          <div
            key={stage.id}
            role="listitem"
            style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.375rem', padding: '0 0.25rem' }}>
              <StageIcon status={stage.status} />
              <span
                style={{
                  fontSize: '11px',
                  color: stage.status === 'in-progress' ? 'var(--accent)' : stage.status === 'complete' ? 'var(--text-secondary)' : 'var(--text-muted)',
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  transition: 'color 0.3s',
                }}
              >
                {stage.label}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div
                style={{
                  flex: 1,
                  height: '0.5px',
                  background: stage.status === 'complete' ? 'var(--success)' : 'var(--border)',
                  marginBottom: '16px',
                  transition: 'background 0.3s',
                }}
              />
            )}
          </div>
        ))}
      </div>

      {/* Budget bar */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Budget</span>
            <Tooltip
              label="Live Budget Usage"
              body="Tracks how much of your cost cap has been spent so far. Turns amber at 50% and red at 70% as a warning before the limit is hit."
              placement="bottom"
            />
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
            ${costSpent.toFixed(2)} / ${budget.toFixed(2)}
            <span style={{ marginLeft: '0.5rem', color: pct >= 70 ? 'var(--danger)' : pct >= 50 ? 'var(--warning)' : 'var(--text-muted)' }}>
              ({pct.toFixed(0)}%)
            </span>
          </span>
        </div>
        <div
          style={{
            height: 6,
            background: 'var(--bg-elevated)',
            borderRadius: 3,
            overflow: 'hidden',
          }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Budget usage"
        >
          <div
            style={{
              height: '100%',
              width: `${pct}%`,
              background: barColor,
              borderRadius: 3,
              transition: 'width 0.5s ease, background 0.3s',
            }}
          />
        </div>
      </div>
    </section>
  );
}
