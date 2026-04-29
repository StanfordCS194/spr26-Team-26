import { useState } from 'react';
import type { TrainingState } from '../types';
import PipelineProgress from './PipelineProgress';
import MetricsGrid from './MetricsGrid';
import MetricsChart from './MetricsChart';
import IterationsList from './IterationsList';
import ActivityLog from './ActivityLog';
import FinalResults from './FinalResults';

interface Props {
  state: TrainingState;
  onReset: () => void;
  onOpenDiffs?: () => void;
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

const TIER_COLORS: Record<string, string> = {
  BEGINNER:     'var(--text-muted)',
  INTERMEDIATE: 'var(--accent)',
  ADVANCED:     'var(--success)',
};

function StatusDot({ status }: { status: TrainingState['status'] }) {
  const color = status === 'complete' ? 'var(--success)' : status === 'running' ? 'var(--accent)' : 'var(--text-muted)';
  const label = status === 'complete' ? 'COMPLETE' : status === 'running' ? 'RUNNING' : 'IDLE';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          display: 'inline-block',
          animation: status === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
        }}
      />
      <span style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.1em', color }}>
        {label}
      </span>
    </div>
  );
}

export default function Dashboard({ state, onReset, onOpenDiffs }: Props) {
  const { capabilities } = state;
  const obs = capabilities.observability;

  const showMetricsChart   = obs.metrics_visibility !== 'none';
  const showIterations     = obs.autoresearch_diary_access !== 'none';
  const showActivityLog    = obs.run_status === 'detailed';
  const showCostDetail     = obs.cost_visibility === 'detailed';
  const showDiaryGateway   = capabilities.comfort_level === 'ADVANCED';

  const [gatewayHovered, setGatewayHovered] = useState(false);
  const latestDiff = state.iterations.find(it => it.diff)?.diff ?? null;

  const tierColor = TIER_COLORS[capabilities.comfort_level] ?? 'var(--text-muted)';

  return (
    <div style={{ minHeight: '100vh' }}>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
            ML Training Agent
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>CS194 · Team 26</span>
          {/* Experience tier badge */}
          <span style={{
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.08em',
            padding: '2px 7px',
            borderRadius: '4px',
            background: 'var(--bg-elevated)',
            border: `0.5px solid ${tierColor}`,
            color: tierColor,
          }}>
            {capabilities.comfort_level}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
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

        {/* Pipeline + budget — always visible */}
        <PipelineProgress stages={state.stages} costSpent={state.costSpent} budget={state.budget} />

        {/* Metric cards — always visible (detail controlled by cost_visibility) */}
        <MetricsGrid
          metrics={state.metrics}
          costSpent={state.costSpent}
          budget={state.budget}
          showCostDetail={showCostDetail}
        />

        {/* Chart + iterations — gated */}
        {(showMetricsChart || showIterations) && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: showMetricsChart && showIterations ? '1fr 1fr' : '1fr',
            gap: '12px',
            marginBottom: '1.5rem',
          }}>
            {showMetricsChart && <MetricsChart metrics={state.metrics} />}
            {showIterations && <IterationsList iterations={state.iterations} />}
          </div>
        )}

        {/* Activity log — only for detailed run_status */}
        {showActivityLog && <ActivityLog logs={state.logs} />}

        {/* Beginner: simple status message while running */}
        {!showActivityLog && state.status === 'running' && (
          <div style={{
            background: 'var(--bg-surface)',
            border: '0.5px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '1rem',
            marginBottom: '1.5rem',
            textAlign: 'center',
          }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              Training is running — results will appear when complete.
            </span>
          </div>
        )}

        {state.status === 'complete' && (
          <FinalResults state={state} onReset={onReset} />
        )}
      </main>

      {/* Advanced-only: floating Research Diary gateway */}
      {showDiaryGateway && onOpenDiffs && (
        <div style={{ position: 'fixed', bottom: '1.5rem', right: '1.5rem', zIndex: 50 }}>
          {/* Hover preview card */}
          {gatewayHovered && latestDiff && (
            <div style={{
              position: 'absolute',
              bottom: 'calc(100% + 10px)',
              right: 0,
              width: '280px',
              background: 'var(--bg-surface)',
              border: '0.5px solid var(--border)',
              borderRadius: 'var(--radius)',
              padding: '0.75rem',
              boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              pointerEvents: 'none',
            }}>
              <p style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '6px' }}>
                LATEST DIFF — configs/current.json
              </p>
              <div style={{ background: 'var(--bg-elevated)', borderRadius: '4px', padding: '4px 0', overflow: 'hidden' }}>
                {latestDiff.split('\n').map((line, i) => {
                  const isAdd = line.startsWith('+');
                  const isRem = line.startsWith('-');
                  return (
                    <div key={i} style={{
                      fontFamily: 'monospace',
                      fontSize: '11px',
                      lineHeight: '1.6',
                      padding: '0 0.5rem',
                      color: isAdd ? 'var(--success)' : isRem ? 'var(--danger)' : 'var(--text-muted)',
                      background: isAdd ? 'rgba(34,197,94,0.08)' : isRem ? 'rgba(239,68,68,0.08)' : 'transparent',
                      borderLeft: isAdd ? '2px solid var(--success)' : isRem ? '2px solid var(--danger)' : '2px solid transparent',
                      whiteSpace: 'pre',
                    }}>
                      {line}
                    </div>
                  );
                })}
              </div>
              <p style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '6px' }}>
                {state.iterations.length} experiment{state.iterations.length !== 1 ? 's' : ''} · click to view all
              </p>
            </div>
          )}

          {/* The button */}
          <button
            onClick={onOpenDiffs}
            onMouseEnter={() => setGatewayHovered(true)}
            onMouseLeave={() => setGatewayHovered(false)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0.5rem 0.875rem',
              background: gatewayHovered ? 'var(--bg-surface)' : 'var(--bg-elevated)',
              border: `0.5px solid ${gatewayHovered ? 'var(--success)' : 'var(--border)'}`,
              borderRadius: '8px',
              color: gatewayHovered ? 'var(--success)' : 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 0.15s',
              boxShadow: gatewayHovered ? '0 4px 16px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.2)',
            }}
          >
            <span style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: state.status === 'running' ? 'var(--success)' : 'var(--text-muted)',
              display: 'inline-block',
              animation: state.status === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
              flexShrink: 0,
            }} />
            Research Diary
            {state.iterations.length > 0 && (
              <span style={{
                fontSize: '10px',
                fontWeight: 600,
                padding: '1px 5px',
                borderRadius: '10px',
                background: 'var(--accent-dim)',
                color: 'var(--accent)',
                border: '0.5px solid var(--accent)',
              }}>
                {state.iterations.length}
              </span>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
