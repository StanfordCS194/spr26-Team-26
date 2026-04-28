import type { ModelPlan, SkillLevel } from '../types';

interface Props {
  plan: ModelPlan;
  skillLevel: SkillLevel;
  onApprove: () => void;
  onReject: () => void;
}

const approachLabel: Record<ModelPlan['approach'], string> = {
  'fine-tune-lora': 'Fine-tune with LoRA',
  'pretrain': 'Pre-train from scratch',
  'full-finetune': 'Full fine-tune',
};

export default function ApprovalCheckpoint({ plan, skillLevel, onApprove, onReject }: Props) {
  const isExpert = skillLevel === 'expert';

  return (
    <section
      aria-label="Model plan approval"
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--accent)',
        borderRadius: 'var(--radius)',
        padding: '1rem 1.25rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '4px' }}>
            <h2 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Model Plan
            </h2>
            <span style={{
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.08em',
              padding: '2px 7px',
              borderRadius: '4px',
              background: 'var(--accent-dim)',
              color: 'var(--accent)',
              border: '0.5px solid var(--accent)',
            }}>
              AWAITING APPROVAL
            </span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Confirm the model and strategy before the agent spends your budget.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Estimated cost</span>
          <span style={{ fontSize: '18px', color: 'var(--warning)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
            ${plan.estimatedCost.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Top summary */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '8px',
        marginBottom: '1rem',
      }}>
        <PlanStat label="Base model" value={plan.baseModel} />
        <PlanStat label="Approach" value={approachLabel[plan.approach]} />
      </div>

      {/* Reasoning */}
      <div style={{ marginBottom: '1rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
          Why this plan
        </p>
        <ol style={{
          background: 'var(--bg-elevated)',
          border: '0.5px solid var(--border-subtle)',
          borderRadius: '6px',
          padding: '0.5rem 1.25rem',
          margin: 0,
          fontSize: '12px',
          color: 'var(--text-secondary)',
          lineHeight: 1.65,
        }}>
          {plan.reasoning.map((step, i) => (
            <li key={i} style={{ marginBottom: i === plan.reasoning.length - 1 ? 0 : '2px' }}>
              {step}
            </li>
          ))}
        </ol>
      </div>

      {/* Hyperparameters (Expert sees full raw; Intermediate sees summary) */}
      {isExpert ? (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
            Starting hyperparameters
          </p>
          <div style={{
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '0.5rem 0.75rem',
            fontFamily: 'ui-monospace, Consolas, monospace',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '4px 1rem',
          }}>
            {Object.entries(plan.hyperparams).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}:</span>
                <span style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
            Starting configuration
          </p>
          <div style={{
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '0.625rem 0.875rem',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.75rem 1.25rem',
          }}>
            {Object.entries(plan.hyperparams).slice(0, 5).map(([k, v]) => (
              <span key={k}>
                <span style={{ color: 'var(--text-muted)' }}>{k.replace(/_/g, ' ')}: </span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{String(v)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Approval actions */}
      <div style={{
        display: 'flex',
        gap: '0.5rem',
        paddingTop: '0.875rem',
        borderTop: '0.5px solid var(--border)',
      }}>
        <button
          onClick={onApprove}
          aria-label="Approve model plan"
          style={{
            padding: '0.5rem 1.25rem',
            background: 'var(--accent-dim)',
            border: '0.5px solid var(--accent)',
            borderRadius: '6px',
            color: 'var(--accent)',
            fontSize: '13px',
            fontWeight: 500,
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          Approve &amp; start training
        </button>
        <button
          onClick={onReject}
          aria-label="Reject plan"
          style={{
            padding: '0.5rem 1rem',
            background: 'transparent',
            border: '0.5px solid var(--danger)',
            borderRadius: '6px',
            color: 'var(--danger)',
            fontSize: '13px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          Cancel run
        </button>
      </div>
    </section>
  );
}

function PlanStat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '0.5px solid var(--border-subtle)',
      borderRadius: '6px',
      padding: '0.5rem 0.75rem',
    }}>
      <p style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        {label}
      </p>
      <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)', fontFamily: 'ui-monospace, Consolas, monospace' }}>
        {value}
      </p>
    </div>
  );
}
