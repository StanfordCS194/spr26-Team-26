import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ZAxis,
  Legend,
} from 'recharts';
import type { Iteration } from '../types';
import TooltipInfo from './Tooltip';

interface Props {
  iterations: Iteration[];
}

interface Point {
  lr: number;
  loraRank: number;
  label: string;
  idx: number;
  status: Iteration['status'];
}

const tooltipStyle: React.CSSProperties = {
  background: 'var(--bg-elevated)',
  border: '0.5px solid var(--border)',
  borderRadius: '6px',
  fontSize: '12px',
  color: 'var(--text-primary)',
  padding: '6px 8px',
};

function buildData(iterations: Iteration[]): Record<Iteration['status'], Point[]> {
  const out: Record<Iteration['status'], Point[]> = { KEPT: [], REVERTED: [], PENDING: [] };
  const withCoords = iterations.filter(i => i.searchCoord);
  const count = withCoords.length;

  // Oldest iteration at the bottom of the array since we push newest to front.
  withCoords.forEach((iter, i) => {
    const displayIdx = count - i; // iteration number from 1..N
    out[iter.status].push({
      lr: iter.searchCoord!.lr,
      loraRank: iter.searchCoord!.loraRank,
      label: iter.experiment,
      idx: displayIdx,
      status: iter.status,
    });
  });
  return out;
}

export default function SearchSpaceViz({ iterations }: Props) {
  const data = buildData(iterations);
  const empty = iterations.filter(i => i.searchCoord).length === 0;

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '0.5px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1rem',
        marginBottom: '1.5rem',
      }}
      aria-label="Hyperparameter search space"
    >
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.75rem' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Hyperparameter Search Space
        </p>
        <TooltipInfo
          label="Search Space"
          body="Each dot is one AutoResearch iteration plotted by learning rate (x) and LoRA rank (y). Green = KEPT, orange = REVERTED, blue = still running. Clusters show where the agent is focusing its search."
        />
      </div>

      {empty ? (
        <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Waiting for first iteration…
          </span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ top: 10, right: 20, left: -16, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              type="number"
              dataKey="lr"
              name="learning rate"
              scale="log"
              domain={['auto', 'auto']}
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              tickFormatter={v => {
                const exp = Math.log10(v);
                return `${v.toExponential(1).replace('e', 'e')}`;
                void exp;
              }}
              label={{ value: 'learning rate (log)', position: 'insideBottom', offset: -4, fill: 'var(--text-muted)', fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="loraRank"
              name="LoRA rank"
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ value: 'LoRA rank', angle: -90, position: 'insideLeft', offset: 20, fill: 'var(--text-muted)', fontSize: 10 }}
            />
            <ZAxis type="number" range={[60, 60]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3', stroke: 'var(--border)' }}
              contentStyle={tooltipStyle}
              content={({ active, payload }) => {
                if (!active || !payload || payload.length === 0) return null;
                const p = payload[0].payload as Point;
                const statusColor =
                  p.status === 'KEPT' ? 'var(--success)' :
                  p.status === 'REVERTED' ? 'var(--warning)' :
                  'var(--accent)';
                return (
                  <div style={tooltipStyle}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Iteration #{p.idx}</div>
                    <div style={{ fontSize: '11px', color: statusColor, fontWeight: 600, marginBottom: '4px' }}>{p.status}</div>
                    <div style={{ fontSize: '11px' }}>
                      lr = <span style={{ fontFamily: 'monospace' }}>{p.lr.toExponential(2)}</span>
                    </div>
                    <div style={{ fontSize: '11px' }}>
                      lora_rank = <span style={{ fontFamily: 'monospace' }}>{p.loraRank}</span>
                    </div>
                  </div>
                );
              }}
            />
            <Legend
              verticalAlign="top"
              align="right"
              iconType="circle"
              wrapperStyle={{ fontSize: '11px', color: 'var(--text-muted)', paddingBottom: '4px' }}
            />
            <Scatter name="Kept" data={data.KEPT} fill="var(--success)" fillOpacity={0.85} />
            <Scatter name="Reverted" data={data.REVERTED} fill="var(--warning)" fillOpacity={0.85} />
            <Scatter name="Running" data={data.PENDING} fill="var(--accent)" fillOpacity={0.85} />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
