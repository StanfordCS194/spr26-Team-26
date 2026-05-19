import type { DataSample } from '../types';

interface Props {
  datasetName: string;
  samples: DataSample[];
}

/**
 * Shows a table preview of the matched dataset during the Data Discovery stage.
 * Columns are inferred dynamically from the first sample row.
 */
export default function DataSamplePanel({ datasetName, samples }: Props) {
  if (!samples.length) return null;

  const columns = Object.keys(samples[0]);

  const truncate = (val: string | number, max = 90): string => {
    const s = String(val);
    return s.length > max ? s.slice(0, max) + '…' : s;
  };

  return (
    <div style={{
      background: '#0f1117',
      border: '1px solid #2a2d3a',
      borderRadius: 8,
      overflow: 'hidden',
      marginTop: 12,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 14px',
        borderBottom: '1px solid #2a2d3a',
        background: '#161820',
      }}>
        <span style={{ fontSize: 12, color: '#6b7280' }}>dataset preview</span>
        <span style={{
          fontSize: 11,
          color: '#a78bfa',
          background: '#1e1730',
          border: '1px solid #3b2f6a',
          borderRadius: 4,
          padding: '1px 7px',
          fontFamily: 'monospace',
        }}>
          {datasetName}
        </span>
        <span style={{ fontSize: 11, color: '#4b5563', marginLeft: 'auto' }}>
          {samples.length} sample rows
        </span>
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: 11,
          fontFamily: 'monospace',
        }}>
          <thead>
            <tr style={{ background: '#131520' }}>
              {columns.map(col => (
                <th key={col} style={{
                  padding: '5px 12px',
                  textAlign: 'left',
                  color: '#6b7280',
                  fontWeight: 600,
                  borderBottom: '1px solid #1e2130',
                  whiteSpace: 'nowrap',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {samples.map((row, i) => (
              <tr key={i} style={{
                borderBottom: i < samples.length - 1 ? '1px solid #1a1d2a' : 'none',
                background: i % 2 === 0 ? 'transparent' : '#0d0f18',
              }}>
                {columns.map(col => {
                  const val = row[col];
                  // Highlight label / target / entities columns
                  const isLabel = ['label', 'intent', 'answer', 'target', 'entities', 'summary'].includes(col);
                  return (
                    <td key={col} style={{
                      padding: '5px 12px',
                      color: isLabel ? '#a78bfa' : '#c9d1d9',
                      maxWidth: isLabel ? 160 : 320,
                      verticalAlign: 'top',
                      lineHeight: 1.5,
                    }}>
                      {truncate(val, isLabel ? 120 : 90)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
