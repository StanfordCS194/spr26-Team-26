import type { LogEntry } from '../types';

interface Props {
  logs: LogEntry[];
}

const typeColor: Record<LogEntry['type'], string> = {
  success: 'var(--success)',
  warning: 'var(--warning)',
  default: 'var(--text-muted)',
};

export default function ActivityLog({ logs }: Props) {
  return (
    <section style={{ marginBottom: '1.5rem' }}>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Activity Log</p>
      <div
        style={{
          background: 'var(--bg-surface)',
          border: '0.5px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '0.625rem 0.75rem',
          maxHeight: '300px',
          overflowY: 'auto',
          fontFamily: 'ui-monospace, Consolas, monospace',
          fontSize: '11px',
          lineHeight: 1.6,
          display: 'flex',
          flexDirection: 'column',
          gap: '1px',
        }}
        role="log"
        aria-label="Activity log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {logs.length === 0 && (
          <span style={{ color: 'var(--text-muted)' }}>Waiting for pipeline to start…</span>
        )}
        {logs.map((entry, i) => (
          <div key={i} style={{ display: 'flex', gap: '0.625rem' }}>
            <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>[{entry.time}]</span>
            <span style={{ color: 'var(--accent)', flexShrink: 0 }}>{entry.component}:</span>
            <span style={{ color: typeColor[entry.type] }}>{entry.message}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
