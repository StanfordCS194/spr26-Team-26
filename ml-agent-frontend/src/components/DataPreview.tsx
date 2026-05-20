import type { DataPreview, SkillLevel } from '../types';

interface Props {
  preview: DataPreview;
  skillLevel: SkillLevel;
  awaitingApproval: boolean;
  onApprove: () => void;
  onReject: () => void;
}

const sourceLabel: Record<DataPreview['source'], string> = {
  huggingface: 'HuggingFace Hub',
  scraped: 'Web Scrape',
  synthetic: 'Synthetic (Claude)',
};

const sourceColor: Record<DataPreview['source'], string> = {
  huggingface: 'var(--accent)',
  scraped: 'var(--warning)',
  synthetic: 'var(--success)',
};

export default function DataPreview({ preview, skillLevel, awaitingApproval, onApprove, onReject }: Props) {
  const isExpert = skillLevel === 'expert';

  return (
    <section
      aria-label="Data generator preview"
      style={{
        background: 'var(--bg-surface)',
        border: awaitingApproval ? '0.5px solid var(--accent)' : '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1rem 1.25rem',
        marginBottom: '1.5rem',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem', gap: '1rem', flexWrap: 'wrap' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '4px' }}>
            <h2 style={{ fontSize: '25px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Dataset Preview
            </h2>
            {awaitingApproval && (
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
            )}
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Review the data the agent picked before training begins.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{
            fontSize: '11px',
            color: sourceColor[preview.source],
            background: 'var(--bg-elevated)',
            border: `0.5px solid ${sourceColor[preview.source]}`,
            borderRadius: '4px',
            padding: '3px 8px',
          }}>
            {sourceLabel[preview.source]}
          </span>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>
            {preview.datasetName}
          </span>
        </div>
      </div>

      {/* Stats row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '8px',
        marginBottom: '1rem',
      }}>
        <StatBlock label="Total rows" value={preview.totalSamples.toLocaleString()} />
        <StatBlock label="Train" value={preview.splits.train.toLocaleString()} />
        <StatBlock label="Val" value={preview.splits.val.toLocaleString()} />
        <StatBlock label="Test" value={preview.splits.test.toLocaleString()} />
        <StatBlock label="Columns" value={String(preview.columns.length)} />
      </div>

      {/* Sample rows */}
      <div style={{ marginBottom: '1rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
          Sample rows ({preview.samples.length})
        </p>
        <div style={{
          background: 'var(--bg-elevated)',
          border: '0.5px solid var(--border-subtle)',
          borderRadius: '6px',
          overflow: 'auto',
          maxHeight: '220px',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
            <thead>
              <tr>
                {preview.columns.map(col => (
                  <th
                    key={col}
                    style={{
                      textAlign: 'left',
                      padding: '6px 10px',
                      color: 'var(--text-muted)',
                      fontWeight: 500,
                      borderBottom: '0.5px solid var(--border)',
                      fontSize: '11px',
                      letterSpacing: '0.04em',
                      textTransform: 'uppercase',
                      position: 'sticky',
                      top: 0,
                      background: 'var(--bg-elevated)',
                    }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.samples.map((row, i) => (
                <tr key={i} style={{ borderBottom: '0.5px solid var(--border-subtle)' }}>
                  {preview.columns.map(col => {
                    const val = row[col];
                    const str = typeof val === 'number' ? val.toLocaleString() : String(val ?? '');
                    return (
                      <td
                        key={col}
                        style={{
                          padding: '6px 10px',
                          color: 'var(--text-secondary)',
                          maxWidth: '460px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontVariantNumeric: typeof val === 'number' ? 'tabular-nums' : 'normal',
                        }}
                        title={str}
                      >
                        {str}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Class distribution (if present) */}
      {preview.classDistribution && (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
            Class distribution
          </p>
          <ClassBars distribution={preview.classDistribution} />
        </div>
      )}

      {/* Reasoning trace */}
      <div style={{ marginBottom: '1rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
          How the agent picked this data
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
          {preview.reasoning.map((step, i) => (
            <li key={i} style={{ marginBottom: i === preview.reasoning.length - 1 ? 0 : '2px' }}>
              {step}
            </li>
          ))}
        </ol>
      </div>

      {/* Synthetic details */}
      {preview.syntheticDetails && (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
            Synthetic generation details
          </p>
          <div style={{
            background: 'var(--bg-elevated)',
            border: '0.5px solid var(--success)',
            borderRadius: '6px',
            padding: '0.625rem 0.875rem',
            fontSize: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <StatChip label="Diversity" value={preview.syntheticDetails.diversityScore.toFixed(2)} />
              <StatChip
                label="Validated"
                value={`${preview.syntheticDetails.validationsPassed.toLocaleString()} / ${preview.syntheticDetails.validationsTotal.toLocaleString()}`}
              />
              <StatChip label="Model" value="claude-haiku-4-5" />
            </div>
            {isExpert && (
              <div>
                <p style={{ color: 'var(--text-muted)', fontSize: '11px', marginBottom: '2px' }}>Generation prompt:</p>
                <pre style={{
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  fontFamily: 'ui-monospace, Consolas, monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  background: 'var(--bg-base)',
                  padding: '6px 8px',
                  borderRadius: '4px',
                  border: '0.5px solid var(--border-subtle)',
                  margin: 0,
                }}>
                  {preview.syntheticDetails.generationPrompt}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Approval actions */}
      {awaitingApproval && (
        <div style={{
          display: 'flex',
          gap: '0.5rem',
          marginTop: '1rem',
          paddingTop: '0.875rem',
          borderTop: '0.5px solid var(--border)',
        }}>
          <button
            onClick={onApprove}
            aria-label="Approve dataset and continue"
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
            Looks good — start training
          </button>
          <button
            onClick={onReject}
            aria-label="Reject dataset and abort run"
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
      )}
    </section>
  );
}

function StatBlock({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '0.5px solid var(--border-subtle)',
      borderRadius: '6px',
      padding: '0.5rem 0.625rem',
    }}>
      <p style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        {label}
      </p>
      <p style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </p>
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}: </span>
      <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{value}</span>
    </span>
  );
}

function ClassBars({ distribution }: { distribution: Record<string, number> }) {
  const entries = Object.entries(distribution);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  const max = Math.max(...entries.map(([, c]) => c));

  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '0.5px solid var(--border-subtle)',
      borderRadius: '6px',
      padding: '0.5rem 0.75rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
    }}>
      {entries.map(([label, count]) => {
        const pct = (count / total) * 100;
        const widthPct = (count / max) * 100;
        return (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '11px' }}>
            <span style={{ width: '120px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {label}
            </span>
            <div style={{ flex: 1, height: '8px', background: 'var(--bg-base)', borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ width: `${widthPct}%`, height: '100%', background: 'var(--accent)', opacity: 0.7 }} />
            </div>
            <span style={{ width: '70px', textAlign: 'right', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
              {count.toLocaleString()} ({pct.toFixed(1)}%)
            </span>
          </div>
        );
      })}
    </div>
  );
}
