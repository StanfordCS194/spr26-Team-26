import type { TrainingState, SkillLevel } from '../types';
import PipelineProgress from './PipelineProgress';
import MetricsGrid from './MetricsGrid';
import MetricsChart from './MetricsChart';
import IterationsList from './IterationsList';
import ActivityLog from './ActivityLog';
import FinalResults from './FinalResults';
import DataPreview from './DataPreview';
import ApprovalCheckpoint from './ApprovalCheckpoint';
import SearchSpaceViz from './SearchSpaceViz';

interface Props {
  state: TrainingState;
  onReset: () => void;
  onApprove: () => void;
  onReject: () => void;
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
    status === 'complete' ? 'var(--success)' :
    status === 'running' ? 'var(--accent)' :
    status === 'awaiting-approval' ? 'var(--warning)' :
    'var(--text-muted)';
  const label =
    status === 'complete' ? 'COMPLETE' :
    status === 'running' ? 'RUNNING' :
    status === 'awaiting-approval' ? 'WAITING FOR YOU' :
    'IDLE';
  const pulsing = status === 'running' || status === 'awaiting-approval';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          display: 'inline-block',
          animation: pulsing ? 'pulse 1.5s ease-in-out infinite' : 'none',
        }}
      />
      <span style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.1em', color }}>
        {label}
      </span>
    </div>
  );
}

const levelColor: Record<SkillLevel, string> = {
  beginner: 'var(--success)',
  intermediate: 'var(--accent)',
  expert: 'var(--warning)',
};

function SkillLevelBadge({ level }: { level: SkillLevel }) {
  const color = levelColor[level];
  return (
    <span
      style={{
        fontSize: '10px',
        fontWeight: 600,
        letterSpacing: '0.08em',
        padding: '2px 7px',
        borderRadius: '4px',
        background: 'var(--bg-surface)',
        color,
        border: `0.5px solid ${color}`,
        textTransform: 'uppercase',
      }}
      aria-label={`Skill level: ${level}`}
    >
      {level}
    </span>
  );
}

export default function Dashboard({ state, onReset, onApprove, onReject }: Props) {
  const { skillLevel, pendingApproval, dataPreview, modelPlan } = state;
  const isExpert = skillLevel === 'expert';
  const isBeginner = skillLevel === 'beginner';

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
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <SkillLevelBadge level={skillLevel} />
          <StatusDot status={state.status} />
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

        {/* Data preview — shown once DataGen has finished. Highlights while awaiting approval. */}
        {dataPreview && (
          <DataPreview
            preview={dataPreview}
            skillLevel={skillLevel}
            awaitingApproval={pendingApproval === 'dataset'}
            onApprove={onApprove}
            onReject={onReject}
          />
        )}

        {/* Model plan checkpoint — only shown while awaiting model approval (Int/Exp). */}
        {modelPlan && pendingApproval === 'model' && (
          <ApprovalCheckpoint
            plan={modelPlan}
            skillLevel={skillLevel}
            onApprove={onApprove}
            onReject={onReject}
          />
        )}

        {/* Metric cards — Beginner gets 3 cards (loss hidden), everyone else gets 4. */}
        <MetricsGrid
          metrics={state.metrics}
          costSpent={state.costSpent}
          budget={state.budget}
          skillLevel={skillLevel}
        />

        {/* 2-col: chart + iterations */}
        <div
          className="two-col-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '12px',
            marginBottom: '1.5rem',
          }}
        >
          <MetricsChart metrics={state.metrics} />
          <IterationsList iterations={state.iterations} skillLevel={skillLevel} />
        </div>

        {/* Search-space viz — shown for Expert once any AutoResearch iteration has landed.
            Also shown for Intermediate since it's the most intuitive observability panel. */}
        {!isBeginner && state.iterations.length > 0 && (
          <SearchSpaceViz iterations={state.iterations} />
        )}

        {/* Activity log — filters by skill level internally. */}
        <ActivityLog logs={state.logs} skillLevel={skillLevel} />

        {/* Expert-only: raw state snapshot for debugging during demo */}
        {isExpert && state.status !== 'complete' && (
          <details style={{
            background: 'var(--bg-surface)',
            border: '0.5px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '0.5rem 0.875rem',
            marginBottom: '1.5rem',
            fontSize: '11px',
            color: 'var(--text-muted)',
          }}>
            <summary style={{ cursor: 'pointer', userSelect: 'none' }}>
              Raw state snapshot (Expert)
            </summary>
            <pre style={{
              fontFamily: 'ui-monospace, Consolas, monospace',
              fontSize: '11px',
              color: 'var(--text-secondary)',
              overflowX: 'auto',
              marginTop: '8px',
              background: 'var(--bg-base)',
              padding: '8px',
              borderRadius: '4px',
            }}>
{JSON.stringify({
  status: state.status,
  stage: state.stage,
  pendingApproval,
  costSpent: state.costSpent,
  iterations: state.iterations.length,
  metrics: state.metrics.length,
}, null, 2)}
            </pre>
          </details>
        )}

        {/* Final results */}
        {state.status === 'complete' && (
          <FinalResults state={state} onReset={onReset} />
        )}
      </main>
    </div>
  );
}
