import { useState, useRef } from 'react';

interface Props {
  label: string;
  body: string;
  /** Which side the bubble opens toward. Defaults to 'top'. */
  placement?: 'top' | 'bottom';
}

export default function Tooltip({ label, body, placement = 'top' }: Props) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(true);
  };
  const hide = () => {
    timerRef.current = setTimeout(() => setVisible(false), 80);
  };

  const bubbleBase: React.CSSProperties = {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 200,
    width: 220,
    background: 'var(--bg-elevated)',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    padding: '0.625rem 0.75rem',
    pointerEvents: 'none',
    textAlign: 'left',
  };

  const bubble: React.CSSProperties =
    placement === 'top'
      ? { ...bubbleBase, bottom: 'calc(100% + 8px)' }
      : { ...bubbleBase, top: 'calc(100% + 8px)' };

  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        type="button"
        onFocus={show}
        onBlur={hide}
        aria-label={`More info: ${label}`}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 14,
          height: 14,
          borderRadius: '50%',
          border: '0.5px solid var(--text-muted)',
          background: 'transparent',
          color: 'var(--text-muted)',
          fontSize: '9px',
          fontWeight: 700,
          cursor: 'default',
          padding: 0,
          lineHeight: 1,
          flexShrink: 0,
          transition: 'border-color 0.15s, color 0.15s',
          outline: 'none',
          fontFamily: 'inherit',
          marginLeft: '5px',
        }}
        onMouseEnter={(e) => {
          show();
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--accent)';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--accent)';
        }}
        onMouseLeave={(e) => {
          hide();
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--text-muted)';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)';
        }}
      >
        i
      </button>

      {visible && (
        <span style={bubble} role="tooltip">
          <span style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {label}
          </span>
          <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
            {body}
          </span>
        </span>
      )}
    </span>
  );
}
