import { useState } from 'react';
import type { Iteration, SkillLevel, ProposalStrategy } from '../types';
import Tooltip from './Tooltip';

interface Props {
  iterations: Iteration[];
  skillLevel: SkillLevel;
}

const strategyLabel: Record<ProposalStrategy, string> = {
  random_search: 'Random Search',
  local_perturbation: 'Local Perturbation',
  claude_proposal: 'Claude Proposal',
};

const strategyColor: Record<ProposalStrategy, string> = {
  random_search: 'var(--text-muted)',
  local_perturbation: 'var(--accent)',
  claude_proposal: 'var(--success)',
};

function StatusBadge({ status }: { status: Iteration['status'] }) {
  const isPending = status === 'PENDING';
  const kept = status === 'KEPT';

  const color = kept ? 'var(--success)' : isPending ? 'var(--accent)' : 'var(--warning)';
  const bg = kept ? 'var(--success-dim)' : isPending ? 'var(--accent-dim)' : 'var(--warning-dim)';
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

function DeltaBlock({ label, before, after, lowerIsBetter }: { label: string; before: number; after: number; lowerIsBetter: boolean }) {
  const delta = after - before;
  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  const pct = before === 0 ? 0 : (delta / before) * 100;
  const color = Math.abs(delta) < 1e-9 ? 'var(--text-muted)' : improved ? 'var(--success)' : 'var(--warning)';
  const arrow = improved ? '↓' : '↑';
  return (
    <span style={{ fontSize: '11px', fontFamily: 'ui-monospace, Consolas, monospace' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label} </span>
      <span style={{ color: 'var(--text-secondary)' }}>{before.toFixed(3)}</span>
      <span style={{ color: 'var(--text-muted)' }}> → </span>
      <span style={{ color: color, fontWeight: 500 }}>{after.toFixed(3)}</span>
      <span style={{ color, marginLeft: '4px' }}>
        {arrow} {Math.abs(pct).toFixed(1)}%
      </span>
    </span>
  );
}

export default function IterationsList({ iterations, skillLevel }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const isExpert = skillLevel === 'expert';
  // Auto-expand everything at beginner — ironically, beginner gets the
  // clearest reasoning surfaced since they need the "why" most;
  // Intermediate and Expert default collapsed so the page feels dense-but-tidy.
  const defaultExpanded = skillLevel === 'beginner';

  const toggle = (id: string) => setExpanded(prev => ({ ...prev, [id]: !(prev[id] ?? defaultExpanded) }));

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
          body="Each row is one hyperparameter configuration the agent tested automatically. Click a row to see the hypothesis, strategy, and post-hoc rationale. KEPT means it improved the model; REVERTED means it was discarded."
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
          {iterations.map((iter, idx) => {
            const isOpen = expanded[iter.id] ?? defaultExpanded;
            const hasReasoning = !!iter.reasoning;
            return (
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
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', cursor: hasReasoning ? 'pointer' : 'default' }}
                  onClick={hasReasoning ? () => toggle(iter.id) : undefined}
                  role={hasReasoning ? 'button' : undefined}
                  aria-expanded={hasReasoning ? isOpen : undefined}
                  aria-label={hasReasoning ? `Toggle reasoning for iteration ${iterations.length - idx}` : undefined}
                  tabIndex={hasReasoning ? 0 : undefined}
                  onKeyDown={hasReasoning ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(iter.id); } } : undefined}
                >
                  {hasReasoning && (
                    <span style={{
                      fontSize: '9px',
                      color: 'var(--text-muted)',
                      flexShrink: 0,
                      width: '10px',
                      display: 'inline-block',
                      transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                      transition: 'transform 0.12s',
                    }}>
                      ▶
                    </span>
                  )}
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

                {/* Expanded reasoning panel */}
                {isOpen && iter.reasoning && (
                  <div
                    style={{
                      marginLeft: '1.75rem',
                      marginTop: '4px',
                      padding: '0.5rem 0.75rem',
                      background: 'var(--bg-base)',
                      border: '0.5px solid var(--border-subtle)',
                      borderRadius: '5px',
                      fontSize: '11.5px',
                      color: 'var(--text-secondary)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                      lineHeight: 1.55,
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Hypothesis
                        </span>
                        <span style={{
                          fontSize: '9px',
                          color: strategyColor[iter.reasoning.strategy],
                          background: 'var(--bg-elevated)',
                          border: `0.5px solid ${strategyColor[iter.reasoning.strategy]}`,
                          borderRadius: '3px',
                          padding: '1px 5px',
                          letterSpacing: '0.04em',
                          textTransform: 'uppercase',
                          fontWeight: 600,
                        }}>
                          {strategyLabel[iter.reasoning.strategy]}
                        </span>
                      </div>
                      <p>{iter.reasoning.hypothesis}</p>
                    </div>

                    <div>
                      <p style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '2px' }}>
                        Expected impact
                      </p>
                      <p>{iter.reasoning.expectedImpact}</p>
                    </div>

                    {iter.status !== 'PENDING' && iter.reasoning.rationale && (
                      <div>
                        <p style={{
                          fontSize: '10px',
                          color: iter.status === 'KEPT' ? 'var(--success)' : 'var(--warning)',
                          textTransform: 'uppercase',
                          letterSpacing: '0.06em',
                          marginBottom: '2px',
                        }}>
                          Decision rationale — {iter.status}
                        </p>
                        <p>{iter.reasoning.rationale}</p>
                        {iter.reasoning.lossBeforeAfter && iter.reasoning.f1BeforeAfter && (
                          <div style={{ display: 'flex', gap: '1rem', marginTop: '4px', flexWrap: 'wrap' }}>
                            <DeltaBlock
                              label="loss"
                              before={iter.reasoning.lossBeforeAfter[0]}
                              after={iter.reasoning.lossBeforeAfter[1]}
                              lowerIsBetter={true}
                            />
                            <DeltaBlock
                              label="f1  "
                              before={iter.reasoning.f1BeforeAfter[0]}
                              after={iter.reasoning.f1BeforeAfter[1]}
                              lowerIsBetter={false}
                            />
                          </div>
                        )}
                      </div>
                    )}

                    {/* Expert-only: full raw diff */}
                    {isExpert && iter.diff && (
                      <div>
                        <p style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '2px' }}>
                          Raw config diff
                        </p>
                        <pre
                          aria-label="Config diff"
                          style={{
                            margin: 0,
                            fontSize: '11px',
                            lineHeight: 1.5,
                            fontFamily: 'monospace',
                            color: 'var(--text-muted)',
                            whiteSpace: 'pre',
                            overflow: 'auto',
                            background: 'var(--bg-elevated)',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            border: '0.5px solid var(--border-subtle)',
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
                      </div>
                    )}
                  </div>
                )}

                {/* When collapsed, keep the small diff line visible (except for Beginner, which hides it entirely). */}
                {!isOpen && iter.diff && skillLevel !== 'beginner' && (
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
            );
          })}
        </div>
      )}
    </div>
  );
}
