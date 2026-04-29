import type { Iteration, TrainingState } from '../types';

interface Props {
  iterations: Iteration[];
  status: TrainingState['status'];
  onBack: () => void;
}

function DiffLine({ line }: { line: string }) {
  const isAdd    = line.startsWith('+');
  const isRemove = line.startsWith('-');
  return (
    <div style={{
      fontFamily: 'monospace',
      fontSize: '12px',
      lineHeight: '1.6',
      padding: '0 0.75rem',
      background: isAdd ? 'rgba(34,197,94,0.08)' : isRemove ? 'rgba(239,68,68,0.08)' : 'transparent',
      color: isAdd ? 'var(--success)' : isRemove ? 'var(--danger)' : 'var(--text-muted)',
      whiteSpace: 'pre',
      borderLeft: isAdd
        ? '2px solid var(--success)'
        : isRemove
        ? '2px solid var(--danger)'
        : '2px solid transparent',
    }}>
      {line || ' '}
    </div>
  );
}

function DecisionBadge({ status }: { status: Iteration['status'] }) {
  const cfg = status === 'KEPT'
    ? { color: 'var(--success)', bg: 'rgba(34,197,94,0.1)', label: '✓ KEPT' }
    : status === 'PENDING'
    ? { color: 'var(--accent)',  bg: 'var(--accent-dim)',   label: '⏳ RUNNING' }
    : { color: 'var(--warning)', bg: 'rgba(234,179,8,0.1)', label: '✗ REVERTED' };

  return (
    <span style={{
      fontSize: '11px',
      fontWeight: 600,
      letterSpacing: '0.06em',
      padding: '3px 8px',
      borderRadius: '4px',
      background: cfg.bg,
      color: cfg.color,
      border: `0.5px solid ${cfg.color}`,
      flexShrink: 0,
    }}>
      {cfg.label}
    </span>
  );
}

export default function DiffViewer({ iterations, status, onBack }: Props) {
  const isEmpty = iterations.length === 0;

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>
      {/* Header */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 10,
        background: 'var(--bg-base)',
        borderBottom: '0.5px solid var(--border)',
        padding: '0.75rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
      }}>
        <button
          onClick={onBack}
          style={{
            padding: '4px 10px',
            background: 'transparent',
            border: '0.5px solid var(--border)',
            borderRadius: '6px',
            color: 'var(--text-secondary)',
            fontSize: '12px',
            cursor: 'pointer',
            fontFamily: 'inherit',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          ← Back
        </button>
        <div>
          <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
            Research Diary
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '0.75rem' }}>
            {iterations.length} experiment{iterations.length !== 1 ? 's' : ''}
            {status === 'running' ? ' · live' : ''}
          </span>
        </div>
        {status === 'running' && (
          <span style={{
            marginLeft: 'auto',
            fontSize: '11px',
            color: 'var(--accent)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <span style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: 'var(--accent)',
              display: 'inline-block',
              animation: 'pulse 1.5s ease-in-out infinite',
            }} />
            LIVE
          </span>
        )}
      </header>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>

      <main style={{ maxWidth: '860px', margin: '0 auto', padding: '1.5rem' }}>
        {isEmpty ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '40vh',
            color: 'var(--text-muted)',
            fontSize: '13px',
          }}>
            {status === 'running' ? 'Waiting for AutoResearch to start…' : 'No experiments recorded.'}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {iterations.map((iter, idx) => (
              <div
                key={iter.id}
                style={{
                  background: 'var(--bg-surface)',
                  border: '0.5px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  overflow: 'hidden',
                }}
              >
                {/* Row header */}
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  borderBottom: iter.diff ? '0.5px solid var(--border-subtle, var(--border))' : 'none',
                }}>
                  <span style={{
                    fontSize: '12px',
                    fontWeight: 600,
                    color: 'var(--accent)',
                    flexShrink: 0,
                    minWidth: '1.75rem',
                    paddingTop: '1px',
                  }}>
                    #{iterations.length - idx}
                  </span>
                  <span style={{
                    flex: 1,
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                  }}>
                    {iter.experiment}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                      loss {iter.loss.toFixed(3)}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                      F1 {iter.f1.toFixed(3)}
                    </span>
                    <DecisionBadge status={iter.status} />
                  </div>
                </div>

                {/* Diff block */}
                {iter.diff && (
                  <div style={{
                    background: 'var(--bg-elevated)',
                    padding: '0.5rem 0',
                  }}>
                    <div style={{
                      fontSize: '10px',
                      fontWeight: 500,
                      letterSpacing: '0.08em',
                      color: 'var(--text-muted)',
                      padding: '0 0.75rem',
                      marginBottom: '4px',
                    }}>
                      configs/current.json
                    </div>
                    {iter.diff.split('\n').map((line, li) => (
                      <DiffLine key={li} line={line} />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
