import type { Iteration } from '../types';
import Tooltip from './Tooltip';

interface Props {
  iterations: Iteration[];
}

function StatusBadge({ status }: { status: Iteration['status'] }) {
  const isPending = status === 'PENDING';
  const kept = status === 'KEPT';

  const color = kept ? 'var(--success)' : isPending ? 'var(--accent)' : 'var(--warning)';
  const bg    = kept ? 'var(--success-dim)' : isPending ? 'var(--accent-dim)' : 'var(--warning-dim)';
  const tip = isPending
    ? { label: 'Evaluating', body: 'This experiment is currently running. The agent is waiting for the training job to finish before deciding to keep or revert.' }
    : kept
    ? { label: 'Configuration Kept', body: 'This experiment produced a measurable improvement and was permanently applied to the model.' }
    : { label: 'Configuration Reverted', body: 'This experiment did not improve the model and was rolled back. The previous best configuration is still active.' };

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '2px', flexShrink: 0 }}>
      <span
        style={{
          fontSize: '10px',
          fontWeight: 600,
          letterSpacing: '0.06em',
          padding: '2px 7px',
          borderRadius: '4px',
          background: bg,
          color,
          border: `0.5px solid ${color}`,
          opacity: isPending ? 0.85 : 1,
        }}
        aria-label={`Status: ${status}`}
      >
        {isPending ? '⏳ RUNNING' : status}
      </span>
      <Tooltip label={tip.label} body={tip.body} placement="top" />
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
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.75rem', flexShrink: 0 }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>AutoResearch Iterations</p>
        <Tooltip
          label="AutoResearch Experiments"
          body="Each row is one hyperparameter configuration the agent tested automatically. KEPT means it improved the model; REVERTED means it was discarded."
        />
      </div>

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
                flexDirection: 'column',
                gap: '4px',
                padding: '0.5rem 0.625rem',
                background: 'var(--bg-elevated)',
                borderRadius: '6px',
                border: '0.5px solid var(--border-subtle)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
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
              {iter.diff && (
                <pre
                  aria-label="Config diff"
                  style={{
                    margin: 0,
                    marginLeft: '1.75rem',
                    fontSize: '10px',
                    lineHeight: 1.5,
                    fontFamily: 'monospace',
                    color: 'var(--text-muted)',
                    whiteSpace: 'pre',
                    overflow: 'hidden',
                  }}
                >
                  {iter.diff.split('\n').map((line, li) => (
                    <span
                      key={li}
                      style={{
                        display: 'block',
                        color: line.startsWith('+') ? 'var(--success)' : line.startsWith('-') ? 'var(--danger)' : 'var(--text-muted)',
                      }}
                    >
                      {line}
                    </span>
                  ))}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
