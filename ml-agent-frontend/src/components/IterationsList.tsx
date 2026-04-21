import type { Iteration } from '../types';

interface Props {
  iterations: Iteration[];
}

function StatusBadge({ status }: { status: Iteration['status'] }) {
  const kept = status === 'KEPT';
  return (
    <span
      style={{
        fontSize: '10px',
        fontWeight: 600,
        letterSpacing: '0.06em',
        padding: '2px 7px',
        borderRadius: '4px',
        background: kept ? 'var(--success-dim)' : 'var(--warning-dim)',
        color: kept ? 'var(--success)' : 'var(--warning)',
        border: `0.5px solid ${kept ? 'var(--success)' : 'var(--warning)'}`,
        flexShrink: 0,
      }}
      aria-label={`Status: ${status}`}
    >
      {status}
    </span>
  );
}

export default function IterationsList({ iterations }: Props) {
  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1rem',
        height: '260px',
        display: 'flex',
        flexDirection: 'column',
      }}
      aria-label="AutoResearch iterations"
    >
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '0.75rem', flexShrink: 0 }}>
        AutoResearch Iterations
      </p>

      {iterations.length === 0 ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Awaiting AutoResearch…</span>
        </div>
      ) : (
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
          role="list"
        >
          {iterations.map((iter, idx) => (
            <div
              key={iter.id}
              role="listitem"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.625rem',
                padding: '0.5rem 0.625rem',
                background: 'var(--bg-elevated)',
                borderRadius: '6px',
                border: '0.5px solid var(--border-subtle)',
              }}
            >
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', flexShrink: 0, minWidth: '1.5rem' }}>
                #{iterations.length - idx}
              </span>
              <span style={{ flex: 1, fontSize: '12px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {iter.experiment}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                L {iter.loss.toFixed(3)}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                F1 {iter.f1.toFixed(3)}
              </span>
              <StatusBadge status={iter.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
