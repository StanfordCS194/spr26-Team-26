import type { LogEntry, SkillLevel } from '../types';
import Tooltip from './Tooltip';

interface Props {
  logs: LogEntry[];
  skillLevel: SkillLevel;
}

const typeColor: Record<LogEntry['type'], string> = {
  success: 'var(--success)',
  warning: 'var(--warning)',
  default: 'var(--text-muted)',
};

// Each skill level sees this level and anything below it.
// beginner → shows beginner only (clean, high-level)
// intermediate → shows beginner + intermediate (component-tagged, moderate detail)
// expert → shows everything (raw HTTP calls, stepwise gradients, state graph edges)
function canSee(entry: LogEntry, level: SkillLevel): boolean {
  if (level === 'expert') return true;
  if (level === 'intermediate') return entry.minLevel !== 'expert';
  return entry.minLevel === 'beginner';
}

const verbosityHints: Record<SkillLevel, string> = {
  beginner: 'friendly milestones only',
  intermediate: 'component-tagged logs',
  expert: 'all technical detail',
};

export default function ActivityLog({ logs, skillLevel }: Props) {
  const visible = logs.filter(l => canSee(l, skillLevel));
  const hiddenCount = logs.length - visible.length;

  return (
    <section style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem', gap: '0.5rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Activity Log</p>
        <Tooltip
          label="System Activity Log"
          body="A real-time feed of events from each pipeline component, with timestamps. Verbosity adapts to your ML experience level."
          placement="bottom"
        />
        <span style={{
          fontSize: '10px',
          color: 'var(--text-muted)',
          marginLeft: 'auto',
          fontStyle: 'italic',
        }}>
          {verbosityHints[skillLevel]}
          {hiddenCount > 0 && ` · ${hiddenCount} technical log${hiddenCount === 1 ? '' : 's'} hidden`}
        </span>
      </div>
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
        {visible.length === 0 && (
          <span style={{ color: 'var(--text-muted)' }}>Waiting for pipeline to start…</span>
        )}
        {visible.map((entry, i) => (
          <div key={i} style={{ display: 'flex', gap: '0.625rem' }}>
            <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>[{entry.time}]</span>
            {skillLevel !== 'beginner' && (
              <span style={{ color: 'var(--accent)', flexShrink: 0 }}>{entry.component}:</span>
            )}
            <span style={{ color: typeColor[entry.type] }}>{entry.message}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
