import type { DataStats, EvalResult, ModelStatus, TrainStatus } from '../types';

interface Props {
  trainStatus: TrainStatus | null;
  modelStatus: ModelStatus | null;
  dataStats: DataStats | null;
  evalResult: EvalResult | null;
}

interface CardProps {
  label: string;
  value: string;
  sub?: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: string;
}

function Card({ label, value, sub, color, bgColor, borderColor, icon }: CardProps) {
  return (
    <div className={`bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border ${borderColor} flex flex-col gap-1 relative overflow-hidden`}>
      {/* Subtle top accent strip */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${bgColor} rounded-t-2xl`} />
      <div className="flex items-start justify-between mt-1">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">{label}</span>
        <span className={`text-xl ${color} opacity-80`}>{icon}</span>
      </div>
      <span className={`text-3xl font-bold tracking-tight ${color}`}>{value}</span>
      {sub && <span className="text-xs font-medium text-gray-400 dark:text-gray-500">{sub}</span>}
    </div>
  );
}

export default function KPICards({ trainStatus, modelStatus, dataStats, evalResult }: Props) {
  const statusMap: Record<string, { text: string; color: string; bgColor: string; borderColor: string; icon: string }> = {
    idle:      { text: 'Idle',       color: 'text-gray-400',                  bgColor: 'bg-gray-200 dark:bg-gray-700',      borderColor: 'border-gray-200 dark:border-gray-800', icon: '⏸' },
    training:  { text: 'Training…',  color: 'text-amber-500',                 bgColor: 'bg-amber-400',                      borderColor: 'border-amber-200 dark:border-amber-900', icon: '⚙️' },
    completed: { text: 'Completed',  color: 'text-emerald-500',               bgColor: 'bg-emerald-400',                    borderColor: 'border-emerald-200 dark:border-emerald-900', icon: '✅' },
    error:     { text: 'Error',      color: 'text-red-500',                   bgColor: 'bg-red-400',                        borderColor: 'border-red-200 dark:border-red-900', icon: '⚠️' },
  };

  const s = statusMap[trainStatus?.status ?? 'idle'] ?? statusMap.idle;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card
        label="Status"
        value={s.text}
        sub={trainStatus?.status === 'training' ? `${trainStatus.progress}% complete` : undefined}
        color={s.color}
        bgColor={s.bgColor}
        borderColor={s.borderColor}
        icon={s.icon}
      />
      <Card
        label="Accuracy"
        value={
          evalResult
            ? `${(evalResult.accuracy * 100).toFixed(1)}%`
            : trainStatus?.metrics?.final_accuracy != null
            ? `${(trainStatus.metrics.final_accuracy * 100).toFixed(1)}%`
            : '—'
        }
        sub={evalResult ? `+${(evalResult.improvement * 100).toFixed(1)}% vs baseline` : undefined}
        color="text-violet-600 dark:text-violet-400"
        bgColor="bg-gradient-to-r from-violet-500 to-indigo-500"
        borderColor="border-violet-200 dark:border-violet-900"
        icon="🎯"
      />
      <Card
        label="Model"
        value={modelStatus?.exists ? `${modelStatus.size_kb} KB` : 'None'}
        sub={modelStatus?.modified ? `Updated ${new Date(modelStatus.modified).toLocaleDateString()}` : undefined}
        color="text-indigo-600 dark:text-indigo-400"
        bgColor="bg-gradient-to-r from-indigo-500 to-blue-500"
        borderColor="border-indigo-200 dark:border-indigo-900"
        icon="💾"
      />
      <Card
        label="Episodes"
        value={dataStats ? dataStats.total_episodes.toLocaleString() : '—'}
        sub={dataStats ? `${Object.keys(dataStats.scenarios).length} scenarios` : undefined}
        color="text-purple-600 dark:text-purple-400"
        bgColor="bg-gradient-to-r from-purple-500 to-pink-500"
        borderColor="border-purple-200 dark:border-purple-900"
        icon="📂"
      />
    </div>
  );
}
