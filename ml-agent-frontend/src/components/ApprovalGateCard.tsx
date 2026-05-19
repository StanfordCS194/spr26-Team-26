import type { ApprovalGate } from '../types';

interface Props {
  gate: ApprovalGate;
  onApprove: () => void;
}

export default function ApprovalGateCard({ gate, onApprove }: Props) {
  return (
    <div style={{
      background: '#0d1a0f',
      border: '1px solid #16a34a',
      borderRadius: 10,
      padding: '1.25rem 1.5rem',
      marginBottom: 12,
      boxShadow: '0 0 24px #16a34a22',
      animation: 'gateAppear 0.25s ease-out',
    }}>
      <style>{`
        @keyframes gateAppear {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{
          fontSize: 11, fontWeight: 700, letterSpacing: '0.1em',
          color: '#fbbf24', background: '#1c1500',
          border: '0.5px solid #d97706',
          borderRadius: 4, padding: '2px 8px',
        }}>⏸ AWAITING APPROVAL</span>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#f0fdf4' }}>{gate.title}</span>
      </div>

      {/* Description */}
      <p style={{ fontSize: 13, color: '#86efac', lineHeight: 1.6, marginBottom: 12 }}>
        {gate.description}
      </p>

      {/* Detail rows */}
      <div style={{
        background: '#0a150c',
        border: '0.5px solid #166534',
        borderRadius: 6,
        padding: '10px 14px',
        marginBottom: 14,
        display: 'flex',
        flexDirection: 'column',
        gap: 5,
      }}>
        {gate.details.map((detail, i) => {
          const [key, ...rest] = detail.split(':');
          const val = rest.join(':').trim();
          return (
            <div key={i} style={{ display: 'flex', gap: 8, fontSize: 12, fontFamily: 'monospace' }}>
              {val ? (
                <>
                  <span style={{ color: '#6b7280', flexShrink: 0 }}>{key}:</span>
                  <span style={{ color: '#d1fae5' }}>{val}</span>
                </>
              ) : (
                <span style={{ color: '#d1fae5' }}>{detail}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Approve button */}
      <button
        onClick={onApprove}
        style={{
          padding: '8px 24px',
          background: '#15803d',
          border: '0.5px solid #22c55e',
          borderRadius: 6,
          color: '#f0fdf4',
          fontSize: 13,
          fontWeight: 600,
          cursor: 'pointer',
          fontFamily: 'inherit',
          transition: 'background 0.15s',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = '#166534')}
        onMouseLeave={e => (e.currentTarget.style.background = '#15803d')}
      >
        ✓ Approve &amp; Continue
      </button>
    </div>
  );
}
