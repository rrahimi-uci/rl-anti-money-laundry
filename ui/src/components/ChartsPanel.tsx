import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { EvalResult, TrainStatus } from '../types';

interface Props {
  trainStatus: TrainStatus | null;
  evalResult: EvalResult | null;
}

const VIOLET = '#8b5cf6';
const INDIGO = '#6366f1';
const GREEN = '#22c55e';
const RED = '#ef4444';
const AMBER = '#f59e0b';

const TICK_STYLE = { fontSize: 11, fill: '#9ca3af' };
const TOOLTIP_STYLE = {
  background: '#1f2937',
  border: '1px solid #374151',
  borderRadius: '10px',
  fontSize: 13,
  padding: '8px 12px',
};

export default function ChartsPanel({ trainStatus, evalResult }: Props) {
  const history = trainStatus?.metrics?.accuracy_history ?? [];
  const actions = evalResult?.action_distribution ?? [];
  const rewardValues = evalResult?.reward_distribution?.values ?? [];

  const hasAccuracyData = history.length > 0;
  const hasEvalData = actions.length > 0;

  if (!hasAccuracyData && !hasEvalData) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-10 shadow-sm border border-gray-200 dark:border-gray-800 text-center">
        <div className="text-5xl mb-4">📈</div>
        <p className="text-sm font-medium text-gray-400 dark:text-gray-500">Start training or run evaluation to see charts</p>
      </div>
    );
  }

  // Histogram bins for reward distribution
  const rewardBins = (() => {
    if (rewardValues.length === 0) return [];
    const min = Math.min(...rewardValues);
    const max = Math.max(...rewardValues);
    const nBins = 20;
    const step = (max - min) / nBins || 1;
    const bins: { range: string; count: number; from: number }[] = [];
    for (let i = 0; i < nBins; i++) {
      const from = min + i * step;
      const to = from + step;
      bins.push({
        range: `${from.toFixed(2)}`,
        count: rewardValues.filter(v => v >= from && (i === nBins - 1 ? v <= to : v < to)).length,
        from,
      });
    }
    return bins;
  })();

  // Color coding for action bars
  const actionColors = actions.map(a => {
    const name = a.action.toLowerCase();
    if (name.includes('lower') || name.includes('reduce')) return GREEN;
    if (name.includes('raise') || name.includes('boost')) return RED;
    if (name.includes('no_change')) return AMBER;
    return VIOLET;
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Accuracy curve */}
      {hasAccuracyData && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Rolling Accuracy</h3>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={VIOLET} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={VIOLET} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
              <XAxis
                dataKey="step"
                tick={TICK_STYLE}
                tickFormatter={v => `${(v / 1000).toFixed(0)}k`}
              />
              <YAxis
                domain={[0, 1]}
                tick={TICK_STYLE}
                tickFormatter={v => `${(Number(v) * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelFormatter={v => `Step ${Number(v).toLocaleString()}`}
                formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Accuracy']}
              />
              <Area type="monotone" dataKey="accuracy" stroke={VIOLET} fill="url(#accGrad)" strokeWidth={2.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Reward curve */}
      {hasAccuracyData && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Average Reward</h3>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="rwdGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={INDIGO} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={INDIGO} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
              <XAxis
                dataKey="step"
                tick={TICK_STYLE}
                tickFormatter={v => `${(v / 1000).toFixed(0)}k`}
              />
              <YAxis tick={TICK_STYLE} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelFormatter={v => `Step ${Number(v).toLocaleString()}`}
                formatter={(v: number) => [v.toFixed(3), 'Reward']}
              />
              <Area type="monotone" dataKey="avg_reward" stroke={INDIGO} fill="url(#rwdGrad)" strokeWidth={2.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Action distribution */}
      {hasEvalData && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Action Distribution</h3>
          <ResponsiveContainer width="100%" height={Math.max(240, actions.length * 24)}>
            <BarChart data={actions} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
              <XAxis type="number" tick={TICK_STYLE} />
              <YAxis
                dataKey="action"
                type="category"
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                width={95}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v: number, _: string, item: any) => [`${v} (${item.payload?.pct ?? 0}%)`, 'Count']}
              />
              <Bar dataKey="count" radius={[0, 5, 5, 0]}>
                {actions.map((_, i) => (
                  <Cell key={i} fill={actionColors[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Reward distribution histogram */}
      {rewardBins.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Reward Distribution</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={rewardBins}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
              <XAxis dataKey="range" tick={{ fontSize: 11, fill: '#9ca3af' }} interval="preserveStartEnd" />
              <YAxis tick={TICK_STYLE} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="count" fill={INDIGO} radius={[5, 5, 0, 0]}>
                {rewardBins.map((bin, i) => (
                  <Cell key={i} fill={bin.from >= 0 ? GREEN : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
