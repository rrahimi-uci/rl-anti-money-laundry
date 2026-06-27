import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { EvalResult } from '../types';

interface Props {
  evalResult: EvalResult | null;
  algorithm: string;
  onEvaluate: (n: number, algo?: string) => void;
  evaluating: boolean;
  modelExists: boolean;
}

const TOOLTIP_STYLE = {
  background: '#1f2937',
  border: '1px solid #374151',
  borderRadius: '10px',
  fontSize: 13,
};

export default function EvaluationPanel({ evalResult, algorithm, onEvaluate, evaluating, modelExists }: Props) {
  if (!evalResult) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-12 text-center">
        <div className="text-6xl mb-5">📊</div>
        <h3 className="text-lg font-bold mb-2 text-gray-800 dark:text-gray-200">No Evaluation Results</h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-7">Run evaluation to see model performance metrics</p>
        <button
          onClick={() => onEvaluate(200, algorithm)}
          disabled={evaluating || !modelExists}
          className="px-7 py-2.5 rounded-xl bg-gradient-to-r from-violet-500 to-indigo-500 text-white text-sm font-semibold shadow-md hover:shadow-lg disabled:opacity-50 transition-all"
        >
          {evaluating ? '⏳ Evaluating…' : !modelExists ? 'No Model Found' : '🚀 Run Evaluation (200 episodes)'}
        </button>
      </div>
    );
  }

  const { accuracy, avg_reward, baseline_accuracy, improvement, total_episodes, correct, action_distribution, per_episode } = evalResult;

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Accuracy" value={`${(accuracy * 100).toFixed(1)}%`} color="text-emerald-500" bgAccent="bg-emerald-400" />
        <MetricCard label="Baseline" value={`${(baseline_accuracy * 100).toFixed(1)}%`} color="text-gray-400" bgAccent="bg-gray-300 dark:bg-gray-600" />
        <MetricCard label="Improvement" value={`+${(improvement * 100).toFixed(1)}%`} color="text-violet-500" bgAccent="bg-gradient-to-r from-violet-500 to-indigo-500" />
        <MetricCard label="Avg Reward" value={avg_reward.toFixed(3)} color="text-indigo-500" bgAccent="bg-gradient-to-r from-indigo-500 to-blue-500" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Action distribution */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Action Distribution <span className="text-sm font-normal text-gray-400">({total_episodes} episodes)</span></h3>
          <ResponsiveContainer width="100%" height={Math.max(260, action_distribution.length * 24)}>
            <BarChart data={action_distribution} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis dataKey="action" type="category" tick={{ fontSize: 11, fill: '#9ca3af' }} width={95} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v: number, _: string, item: any) => [`${v} (${item.payload?.pct ?? 0}%)`, 'Count']}
              />
              <Bar dataKey="count" radius={[0, 5, 5, 0]}>
                {action_distribution.map((a, i) => {
                  const name = a.action.toLowerCase();
                  const fill = name.includes('lower') || name.includes('reduce')
                    ? '#22c55e'
                    : name.includes('raise') || name.includes('boost')
                    ? '#ef4444'
                    : name.includes('no_change')
                    ? '#f59e0b'
                    : '#8b5cf6';
                  return <Cell key={i} fill={fill} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Episode details table */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Episode Details <span className="text-sm font-normal text-gray-400">(first {per_episode.length})</span></h3>
          <div className="overflow-auto max-h-[400px] rounded-xl border border-gray-200 dark:border-gray-700">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
                <tr>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">#</th>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Action</th>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Predicted</th>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Truth</th>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Reward</th>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {per_episode.map(ep => (
                  <tr key={ep.episode} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                    <td className="px-3 py-2 font-mono text-xs text-gray-500">{ep.episode}</td>
                    <td className="px-3 py-2 font-mono text-xs">{ep.action}</td>
                    <td className="px-3 py-2 text-xs">{ep.predicted}</td>
                    <td className="px-3 py-2 text-xs">{ep.ground_truth}</td>
                    <td className={`px-3 py-2 font-mono text-xs font-semibold ${ep.reward >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      {ep.reward > 0 ? '+' : ''}{ep.reward.toFixed(2)}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-white text-xs font-bold ${ep.correct ? 'bg-emerald-500' : 'bg-red-500'}`}>
                        {ep.correct ? '✓' : '✗'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-xs font-medium text-gray-400 text-right">
            {correct} / {total_episodes} correct ({(accuracy * 100).toFixed(1)}%)
          </div>
        </div>
      </div>

      {/* Re-evaluate button */}
      <div className="text-center">
        <button
          onClick={() => onEvaluate(500, algorithm)}
          disabled={evaluating}
          className="px-7 py-2.5 rounded-xl border border-violet-300 dark:border-violet-700 text-violet-600 dark:text-violet-400 text-sm font-semibold hover:bg-violet-50 dark:hover:bg-violet-900/30 disabled:opacity-50 transition-colors"
        >
          {evaluating ? '⏳ Evaluating…' : '🔄 Re-evaluate (500 episodes)'}
        </button>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color, bgAccent }: { label: string; value: string; color: string; bgAccent: string }) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800 text-center relative overflow-hidden">
      <div className={`absolute top-0 left-0 right-0 h-1 ${bgAccent} rounded-t-2xl`} />
      <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest mt-1">{label}</div>
      <div className={`text-3xl font-bold mt-2 ${color}`}>{value}</div>
    </div>
  );
}
