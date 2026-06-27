import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { RLConfig, TuningJob, TuningParamDef, TuningStatus, TuningTrialResult } from '../types';

/* ── Preset parameter spaces ─────────────────────────────────────────────── */

const PARAM_PRESETS: Record<string, TuningParamDef[]> = {
  core: [
    { name: 'learning_rate', low: 1e-5, high: 1e-2, log_scale: true, dtype: 'float' },
    { name: 'weight_step', low: 1.0, high: 5.0, log_scale: false, dtype: 'float' },
    { name: 'ent_coef', low: 0.001, high: 0.1, log_scale: true, dtype: 'float' },
  ],
  full: [
    { name: 'learning_rate', low: 1e-5, high: 1e-2, log_scale: true, dtype: 'float' },
    { name: 'weight_step', low: 1.0, high: 5.0, log_scale: false, dtype: 'float' },
    { name: 'ent_coef', low: 0.001, high: 0.1, log_scale: true, dtype: 'float' },
    { name: 'n_steps', low: 32, high: 512, log_scale: false, dtype: 'int' },
    { name: 'batch_size', low: 16, high: 256, log_scale: false, dtype: 'int' },
    { name: 'n_epochs', low: 3, high: 20, log_scale: false, dtype: 'int' },
    { name: 'gamma', low: 0.9, high: 0.999, log_scale: false, dtype: 'float' },
  ],
};

const STRATEGY_INFO: Record<string, { label: string; icon: string; desc: string; color: string }> = {
  grid: {
    label: 'Grid Search',
    icon: '⊞',
    desc: 'Exhaustive search over all parameter combinations. Best for small spaces.',
    color: 'from-blue-500 to-cyan-500',
  },
  random: {
    label: 'Random Search',
    icon: '🎲',
    desc: 'Randomly samples configurations. Efficient for medium-sized spaces.',
    color: 'from-amber-500 to-orange-500',
  },
  bayesian: {
    label: 'Bayesian (TPE)',
    icon: '🧠',
    desc: 'Uses past results to intelligently pick the next trial. Best for expensive searches.',
    color: 'from-violet-500 to-purple-500',
  },
};

interface Props {
  config: RLConfig | null;
  onTrainWithBest: (timesteps: number) => void;
}

export default function TuningPanel({ config, onTrainWithBest }: Props) {
  const [strategy, setStrategy] = useState<'grid' | 'random' | 'bayesian'>('random');
  const [algorithm, setAlgorithm] = useState('PPO');
  const [nTrials, setNTrials] = useState(10);
  const [preset, setPreset] = useState<'core' | 'full'>('core');
  const [paramSpace, setParamSpace] = useState<TuningParamDef[]>(PARAM_PRESETS.core);
  const [maxParallel, setMaxParallel] = useState(0); // 0 = auto
  const [tuningStatus, setTuningStatus] = useState<TuningStatus | null>(null);
  const [showParamEditor, setShowParamEditor] = useState(false);
  const [selectedTrial, setSelectedTrial] = useState<TuningTrialResult | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const algorithms = config?.algorithms ?? ['PPO', 'A2C', 'DQN'];

  const job = tuningStatus?.job ?? null;
  const isRunning = job?.status === 'running';
  const isCompleted = job?.status === 'completed';

  // Initial fetch
  useEffect(() => {
    api.tuningStatus().then(setTuningStatus).catch(() => {});
  }, []);

  // Poll when running
  useEffect(() => {
    if (isRunning) {
      pollRef.current = setInterval(async () => {
        try {
          const s = await api.tuningStatus();
          setTuningStatus(s);
          if (s.job?.status !== 'running') {
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch { /* ignore */ }
      }, 2000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [isRunning]);

  const handlePresetChange = useCallback((p: 'core' | 'full') => {
    setPreset(p);
    setParamSpace([...PARAM_PRESETS[p]]);
  }, []);

  const handleStart = useCallback(async () => {
    try {
      await api.tuningStart({ strategy, algorithm, n_trials: nTrials, param_space: paramSpace, max_parallel: maxParallel });
      const s = await api.tuningStatus();
      setTuningStatus(s);
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to start tuning');
    }
  }, [strategy, algorithm, nTrials, paramSpace, maxParallel]);

  const handleCancel = useCallback(async () => {
    try {
      await api.tuningCancel();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to cancel');
    }
  }, []);

  const updateParam = useCallback((idx: number, field: keyof TuningParamDef, value: any) => {
    setParamSpace(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: value };
      return next;
    });
  }, []);

  const removeParam = useCallback((idx: number) => {
    setParamSpace(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const addParam = useCallback(() => {
    setParamSpace(prev => [...prev, { name: '', low: 0, high: 1, log_scale: false, dtype: 'float' }]);
  }, []);

  // Find the best accuracy among all results for color scaling
  const bestAcc = job?.results?.length ? Math.max(...job.results.map(r => r.accuracy)) : 0;
  const worstAcc = job?.results?.length ? Math.min(...job.results.map(r => r.accuracy)) : 0;

  return (
    <div className="space-y-6">
      {/* ── Strategy Selection ─────────────────────────────────────────── */}
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
        <div className="p-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-lg font-semibold">Hyperparameter Tuning</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Search for optimal hyperparameters using Grid, Random, or Bayesian optimization
              </p>
            </div>
            {isRunning && (
              <button
                onClick={handleCancel}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-red-300 dark:border-red-700 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
              >
                ✕ Cancel
              </button>
            )}
          </div>

          {/* Strategy cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
            {Object.entries(STRATEGY_INFO).map(([key, info]) => (
              <button
                key={key}
                onClick={() => setStrategy(key as any)}
                disabled={isRunning}
                className={`relative p-4 rounded-xl border-2 text-left transition-all disabled:opacity-50 ${
                  strategy === key
                    ? 'border-violet-500 dark:border-violet-400 bg-violet-50 dark:bg-violet-900/20 shadow-md'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                {strategy === key && (
                  <div className="absolute top-2 right-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-gradient-to-r from-violet-500 to-indigo-500 text-white text-xs">✓</span>
                  </div>
                )}
                <div className="text-2xl mb-2">{info.icon}</div>
                <div className="font-semibold text-sm mb-1">{info.label}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{info.desc}</div>
              </button>
            ))}
          </div>

          {/* Config row */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Algorithm</label>
              <div className="flex gap-1 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-1">
                {algorithms.map(a => (
                  <button
                    key={a}
                    onClick={() => setAlgorithm(a)}
                    disabled={isRunning}
                    className={`flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
                      algorithm === a
                        ? 'bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-sm'
                        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    } disabled:opacity-50`}
                  >
                    {a}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                {strategy === 'grid' ? 'Points per Param' : 'Number of Trials'}
              </label>
              <input
                type="number"
                value={nTrials}
                onChange={e => setNTrials(Math.max(1, Number(e.target.value)))}
                min={1}
                max={200}
                disabled={isRunning}
                title="Number of trials"
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Parallel Workers
              </label>
              <input
                type="number"
                value={maxParallel}
                onChange={e => setMaxParallel(Math.max(0, Number(e.target.value)))}
                min={0}
                max={16}
                disabled={isRunning}
                title="Max parallel workers (0 = auto)"
                placeholder="0 = auto"
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Parameter Preset</label>
              <div className="flex gap-1 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-1">
                {(['core', 'full'] as const).map(p => (
                  <button
                    key={p}
                    onClick={() => handlePresetChange(p)}
                    disabled={isRunning}
                    className={`flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all capitalize ${
                      preset === p
                        ? 'bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-sm'
                        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    } disabled:opacity-50`}
                  >
                    {p} ({PARAM_PRESETS[p].length})
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-end">
              <button
                onClick={handleStart}
                disabled={isRunning || paramSpace.length === 0}
                className="w-full px-5 py-2 text-sm font-medium rounded-lg bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-md hover:shadow-lg disabled:opacity-50 transition-all"
              >
                {isRunning ? '⏳ Tuning…' : '🔍 Start Tuning'}
              </button>
            </div>
          </div>

          {/* Parameter space editor */}
          <button
            onClick={() => setShowParamEditor(!showParamEditor)}
            className="text-xs text-violet-500 hover:text-violet-600 font-medium"
          >
            {showParamEditor ? '▾ Hide Parameter Space' : '▸ Edit Parameter Space'}
          </button>

          {showParamEditor && (
            <div className="mt-3 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 dark:bg-gray-800 text-xs text-gray-500 dark:text-gray-400">
                    <th className="px-3 py-2 text-left font-medium">Parameter</th>
                    <th className="px-3 py-2 text-left font-medium">Low</th>
                    <th className="px-3 py-2 text-left font-medium">High</th>
                    <th className="px-3 py-2 text-left font-medium">Log Scale</th>
                    <th className="px-3 py-2 text-left font-medium">Type</th>
                    <th className="px-3 py-2 w-10"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                  {paramSpace.map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="px-3 py-2">
                        <input
                          value={p.name}
                          onChange={e => updateParam(i, 'name', e.target.value)}
                          disabled={isRunning}
                          className="w-full px-2 py-1 bg-transparent border border-gray-200 dark:border-gray-700 rounded text-xs focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                          placeholder="param_name"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          value={p.low}
                          onChange={e => updateParam(i, 'low', Number(e.target.value))}
                          disabled={isRunning}
                          step="any"
                          title="Low bound"
                          className="w-full px-2 py-1 bg-transparent border border-gray-200 dark:border-gray-700 rounded text-xs focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          value={p.high}
                          onChange={e => updateParam(i, 'high', Number(e.target.value))}
                          disabled={isRunning}
                          step="any"
                          title="High bound"
                          className="w-full px-2 py-1 bg-transparent border border-gray-200 dark:border-gray-700 rounded text-xs focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                        />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <input
                          type="checkbox"
                          checked={p.log_scale}
                          onChange={e => updateParam(i, 'log_scale', e.target.checked)}
                          disabled={isRunning}
                          title="Log scale"
                          className="rounded border-gray-300 text-violet-500 focus:ring-violet-500"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <select
                          value={p.dtype}
                          onChange={e => updateParam(i, 'dtype', e.target.value)}
                          disabled={isRunning}
                          title="Data type"
                          className="px-2 py-1 bg-transparent border border-gray-200 dark:border-gray-700 rounded text-xs focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                        >
                          <option value="float">float</option>
                          <option value="int">int</option>
                        </select>
                      </td>
                      <td className="px-3 py-2">
                        <button
                          onClick={() => removeParam(i)}
                          disabled={isRunning}
                          className="text-red-400 hover:text-red-500 disabled:opacity-30 text-xs"
                        >
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="px-3 py-2 bg-gray-50 dark:bg-gray-800">
                <button
                  onClick={addParam}
                  disabled={isRunning}
                  className="text-xs text-violet-500 hover:text-violet-600 font-medium disabled:opacity-50"
                >
                  + Add Parameter
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Progress ──────────────────────────────────────────────────── */}
      {isRunning && job && (
        <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="h-10 w-10 rounded-full border-4 border-violet-200 dark:border-violet-800" />
                <div
                  className="absolute inset-0 h-10 w-10 rounded-full border-4 border-violet-500 border-t-transparent animate-spin"
                />
              </div>
              <div>
                <div className="text-sm font-semibold">
                  Trial {job.current_trial} of {job.total_trials}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {STRATEGY_INFO[job.strategy]?.label ?? job.strategy} · {job.algorithm}
                  {job.active_workers > 1 && (
                    <span className="ml-1.5 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400 font-medium">
                      ⚡ {job.active_workers} workers
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-violet-600 dark:text-violet-400">{job.progress.toFixed(0)}%</div>
              {job.estimated_remaining != null && (
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  ~{formatDuration(job.estimated_remaining)} remaining
                </div>
              )}
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden mb-4">
            <div
              className={`h-full bg-gradient-to-r ${STRATEGY_INFO[job.strategy]?.color ?? 'from-violet-500 to-indigo-500'} rounded-full transition-all duration-500`}
              style={{ width: `${job.progress}%` }}
            />
          </div>

          {/* Current params being tested */}
          {job.current_params && (
            <div className="flex flex-wrap gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400">Testing:</span>
              {Object.entries(job.current_params).map(([k, v]) => (
                <span
                  key={k}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-800 text-xs font-mono"
                >
                  <span className="text-gray-500 dark:text-gray-400">{k}=</span>
                  <span className="font-semibold">{formatParamValue(v)}</span>
                </span>
              ))}
            </div>
          )}

          {/* Mini results so far */}
          {job.best_result && (
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="flex items-center gap-2 text-xs">
                <span className="text-green-600 dark:text-green-400 font-medium">🏆 Best so far:</span>
                <span className="font-semibold">{(job.best_result.accuracy * 100).toFixed(1)}% accuracy</span>
                <span className="text-gray-400">·</span>
                <span>{job.best_result.avg_reward.toFixed(3)} avg reward</span>
                <span className="text-gray-400">·</span>
                <span className="text-gray-500 dark:text-gray-400">Trial #{job.best_result.trial_id}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────────────── */}
      {job && job.results.length > 0 && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <SummaryCard
              label="Best Accuracy"
              value={job.best_result ? (job.best_result.accuracy * 100).toFixed(1) + '%' : '—'}
              sub={`Trial #${job.best_result?.trial_id ?? 0}`}
              color="green"
            />
            <SummaryCard
              label="Best Reward"
              value={job.best_result?.avg_reward.toFixed(3) ?? '—'}
              sub={`From ${job.completed_trials} trials`}
              color="violet"
            />
            <SummaryCard
              label="Trials Completed"
              value={`${job.completed_trials}/${job.total_trials}`}
              sub={job.status === 'completed' ? 'Done' : job.status}
              color="blue"
            />
            <SummaryCard
              label="Total Time"
              value={formatDuration(job.results.reduce((s, r) => s + r.training_time, 0))}
              sub={`Avg ${formatDuration(job.results.reduce((s, r) => s + r.training_time, 0) / job.results.length)}/trial`}
              color="amber"
            />
          </div>

          {/* Results table + chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Accuracy over trials chart */}
            <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
              <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Accuracy per Trial</h3>
              <TrialChart results={job.results} bestTrialId={job.best_result?.trial_id ?? 0} />
            </div>

            {/* Parallel coordinates-style param view */}
            <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
              <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Parameter Impact</h3>
              <ParamImpactChart results={job.results} bestResult={job.best_result} />
            </div>
          </div>

          {/* Results table */}
          <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
            <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
              <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">Trial Results</h3>
              {isCompleted && job.best_result && (
                <button
                  onClick={() => onTrainWithBest(50_000)}
                  className="px-4 py-2 text-sm font-medium rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-md hover:shadow-lg transition-all"
                >
                  🚀 Train with Best Params
                </button>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 dark:bg-gray-800 text-xs text-gray-500 dark:text-gray-400">
                    <th className="px-4 py-2.5 text-left font-medium">#</th>
                    <th className="px-4 py-2.5 text-left font-medium">Accuracy</th>
                    <th className="px-4 py-2.5 text-left font-medium">Avg Reward</th>
                    <th className="px-4 py-2.5 text-left font-medium">Time</th>
                    <th className="px-4 py-2.5 text-left font-medium">Parameters</th>
                    <th className="px-4 py-2.5 text-left font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                  {[...job.results]
                    .sort((a, b) => b.accuracy - a.accuracy)
                    .map(r => {
                      const isBest = r.trial_id === job.best_result?.trial_id;
                      return (
                        <tr
                          key={r.trial_id}
                          onClick={() => setSelectedTrial(selectedTrial?.trial_id === r.trial_id ? null : r)}
                          className={`cursor-pointer transition-colors ${
                            isBest
                              ? 'bg-green-50 dark:bg-green-900/10 hover:bg-green-100 dark:hover:bg-green-900/20'
                              : selectedTrial?.trial_id === r.trial_id
                              ? 'bg-violet-50 dark:bg-violet-900/10'
                              : 'hover:bg-gray-50 dark:hover:bg-gray-800/50'
                          }`}
                        >
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-1.5">
                              {isBest && <span className="text-green-500">🏆</span>}
                              <span className="font-mono text-xs">{r.trial_id}</span>
                            </div>
                          </td>
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    r.accuracy === bestAcc
                                      ? 'bg-green-500'
                                      : r.accuracy === worstAcc
                                      ? 'bg-red-400'
                                      : 'bg-violet-500'
                                  }`}
                                  style={{ width: `${r.accuracy * 100}%` }}
                                />
                              </div>
                              <span className="font-semibold">{(r.accuracy * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="px-4 py-2.5 font-mono text-xs">
                            <span className={r.avg_reward >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-500'}>
                              {r.avg_reward >= 0 ? '+' : ''}{r.avg_reward.toFixed(3)}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-xs text-gray-500 dark:text-gray-400">{r.training_time.toFixed(1)}s</td>
                          <td className="px-4 py-2.5">
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(r.params).map(([k, v]) => (
                                <span key={k} className="inline-flex px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-xs font-mono">
                                  {k}={formatParamValue(v)}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-4 py-2.5">
                            <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${
                              r.status === 'completed'
                                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                            }`}>
                              {r.status}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>

            {/* Expanded trial detail */}
            {selectedTrial && (
              <div className="p-4 border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                <h4 className="text-xs font-semibold mb-2 text-gray-600 dark:text-gray-300">
                  Trial #{selectedTrial.trial_id} — Full Parameters
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(selectedTrial.params).map(([k, v]) => (
                    <div key={k} className="bg-white dark:bg-gray-900 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">{k}</div>
                      <div className="text-sm font-semibold font-mono mt-0.5">{formatParamValue(v)}</div>
                    </div>
                  ))}
                  <div className="bg-white dark:bg-gray-900 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">Accuracy</div>
                    <div className="text-sm font-semibold mt-0.5">{(selectedTrial.accuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="bg-white dark:bg-gray-900 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Reward</div>
                    <div className={`text-sm font-semibold mt-0.5 ${selectedTrial.avg_reward >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                      {selectedTrial.avg_reward >= 0 ? '+' : ''}{selectedTrial.avg_reward.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* Empty state */}
      {(!job || job.results.length === 0) && !isRunning && (
        <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-12 text-center">
          <div className="text-4xl mb-3">🔬</div>
          <h3 className="text-lg font-semibold mb-1">No Tuning Results Yet</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            Configure your parameter search space above and start tuning to find the best hyperparameters for your model.
          </p>
        </div>
      )}
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────────────── */

function SummaryCard({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  const colors: Record<string, string> = {
    green: 'from-green-500 to-emerald-500',
    violet: 'from-violet-500 to-indigo-500',
    blue: 'from-blue-500 to-cyan-500',
    amber: 'from-amber-500 to-orange-500',
  };
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-4">
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</div>
      <div className={`text-2xl font-bold bg-gradient-to-r ${colors[color] ?? colors.violet} bg-clip-text text-transparent`}>
        {value}
      </div>
      <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{sub}</div>
    </div>
  );
}

function TrialChart({ results, bestTrialId }: { results: TuningTrialResult[]; bestTrialId: number }) {
  if (results.length === 0) return null;

  const maxAcc = Math.max(...results.map(r => r.accuracy));
  const minAcc = Math.min(...results.map(r => r.accuracy));
  const range = maxAcc - minAcc || 0.01;
  const chartH = 160;
  const barW = Math.max(4, Math.min(24, (480 / results.length) - 2));

  return (
    <div className="overflow-x-auto">
      <svg width={Math.max(480, results.length * (barW + 2) + 40)} height={chartH + 30} className="text-xs">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(pct => {
          const y = chartH - pct * chartH;
          const val = minAcc + pct * range;
          return (
            <g key={pct}>
              <line x1={30} y1={y} x2="100%" y2={y} stroke="currentColor" strokeOpacity={0.1} />
              <text x={28} y={y + 3} textAnchor="end" fill="currentColor" opacity={0.4} fontSize={11}>
                {(val * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {/* Bars */}
        {results.map((r, i) => {
          const h = ((r.accuracy - minAcc) / range) * chartH;
          const y = chartH - h;
          const isBest = r.trial_id === bestTrialId;
          return (
            <g key={r.trial_id}>
              <rect
                x={34 + i * (barW + 2)}
                y={y}
                width={barW}
                height={Math.max(2, h)}
                rx={2}
                fill={isBest ? '#22c55e' : r.status === 'error' ? '#ef4444' : '#8b5cf6'}
                opacity={isBest ? 1 : 0.7}
              />
              {results.length <= 30 && (
                <text
                  x={34 + i * (barW + 2) + barW / 2}
                  y={chartH + 14}
                  textAnchor="middle"
                  fill="currentColor"
                  opacity={0.4}
                  fontSize={11}
                >
                  {r.trial_id}
                </text>
              )}
            </g>
          );
        })}
        {/* Best line */}
        {maxAcc > 0 && (
          <line
            x1={30}
            y1={chartH - chartH}
            x2="100%"
            y2={chartH - chartH}
            stroke="#22c55e"
            strokeDasharray="4 2"
            strokeOpacity={0.5}
          />
        )}
      </svg>
    </div>
  );
}

function ParamImpactChart({ results, bestResult }: { results: TuningTrialResult[]; bestResult: TuningTrialResult | null }) {
  if (results.length < 2 || !bestResult) return <div className="text-xs text-gray-400 text-center py-8">Need more trials</div>;

  // Get all param keys
  const paramKeys = Object.keys(results[0].params);

  // For each param, compute correlation with accuracy
  const correlations = paramKeys.map(key => {
    const vals = results.map(r => r.params[key] ?? 0);
    const accs = results.map(r => r.accuracy);
    const corr = pearsonCorrelation(vals, accs);
    return { key, corr, bestVal: bestResult.params[key] ?? 0 };
  });

  correlations.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));

  const maxAbsCorr = Math.max(...correlations.map(c => Math.abs(c.corr)), 0.01);

  return (
    <div className="space-y-3">
      {correlations.map(({ key, corr, bestVal }) => {
        const pct = (Math.abs(corr) / maxAbsCorr) * 100;
        const isPositive = corr >= 0;
        return (
          <div key={key}>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-mono text-gray-600 dark:text-gray-400">{key}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">best: {formatParamValue(bestVal)}</span>
                <span className={`font-semibold ${isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-500'}`}>
                  {isPositive ? '+' : ''}{corr.toFixed(3)}
                </span>
              </div>
            </div>
            <div className="w-full h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  isPositive ? 'bg-green-500' : 'bg-red-400'
                }`}
                style={{ width: `${Math.max(2, pct)}%` }}
              />
            </div>
          </div>
        );
      })}
      <div className="text-xs text-gray-400 dark:text-gray-500 mt-2">
        Pearson correlation between parameter value and accuracy
      </div>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatParamValue(v: number): string {
  if (v === 0) return '0';
  if (Math.abs(v) < 0.001) return v.toExponential(2);
  if (Math.abs(v) < 1) return v.toPrecision(3);
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(4);
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] - mx;
    const yi = y[i] - my;
    num += xi * yi;
    dx += xi * xi;
    dy += yi * yi;
  }
  const denom = Math.sqrt(dx * dy);
  return denom === 0 ? 0 : num / denom;
}
