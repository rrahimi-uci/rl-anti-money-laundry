import { useState } from 'react';
import type { RLConfig, TrainConfig, TrainStatus } from '../types';

interface Props {
  trainStatus: TrainStatus | null;
  config: RLConfig | null;
  algorithm: string;
  onAlgorithmChange: (algorithm: string) => void;
  onStart: (cfg: Partial<TrainConfig>) => void;
  onEvaluate: (n: number, algo: string) => void;
  evaluating: boolean;
  modelExists: boolean;
}

export default function TrainingPanel({
  trainStatus,
  config,
  algorithm,
  onAlgorithmChange,
  onStart,
  onEvaluate,
  evaluating,
  modelExists,
}: Props) {
  const defaults = config?.default_config;
  const algorithms = config?.algorithms ?? ['PPO', 'A2C', 'DQN'];
  const [timesteps, setTimesteps] = useState(defaults?.timesteps ?? 50_000);
  const [lr, setLr] = useState(defaults?.learning_rate ?? 3e-4);
  const [nSteps, setNSteps] = useState(defaults?.n_steps ?? 128);
  const [batchSize, setBatchSize] = useState(defaults?.batch_size ?? 64);
  const [nEpochs, setNEpochs] = useState(defaults?.n_epochs ?? 10);
  const [entCoef, setEntCoef] = useState(defaults?.ent_coef ?? 0.01);
  const [weightStep, setWeightStep] = useState(defaults?.weight_step ?? 2.5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const isTraining = trainStatus?.status === 'training';
  const isCompleted = trainStatus?.status === 'completed';

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
      {/* Progress bar */}
      {isTraining && (
        <div className="h-1 bg-gray-200 dark:bg-gray-800">
          <div
            className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-500"
            style={{ width: `${trainStatus.progress}%` }}
          />
        </div>
      )}

      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">Training Controls</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">Configure and run RL training</p>
          </div>
          <div className="flex gap-2">
            {modelExists && (
              <button
                onClick={() => onEvaluate(200, algorithm)}
                disabled={evaluating || isTraining}
                className="px-4 py-2 text-sm font-medium rounded-xl border border-violet-300 dark:border-violet-700 text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/30 disabled:opacity-50 transition-colors"
              >
                {evaluating ? 'Evaluating…' : '📊 Evaluate'}
              </button>
            )}
            <button
              onClick={() => onStart({
                algorithm,
                timesteps,
                learning_rate: lr,
                n_steps: nSteps,
                batch_size: batchSize,
                n_epochs: nEpochs,
                ent_coef: entCoef,
                weight_step: weightStep,
              })}
              disabled={isTraining}
              className="px-5 py-2 text-sm font-medium rounded-lg bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-md hover:shadow-lg disabled:opacity-50 transition-all"
            >
              {isTraining ? '⏳ Training…' : '🚀 Start Training'}
            </button>
          </div>
        </div>

        {/* Main config */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Algorithm</label>
            <div className="flex gap-1 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-1">
              {algorithms.map(a => (
                <button
                  key={a}
                  onClick={() => onAlgorithmChange(a)}
                  disabled={isTraining}
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
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Total Timesteps</label>
            <input
              type="number"
              value={timesteps}
              onChange={e => setTimesteps(Number(e.target.value))}
              title="Total timesteps"
              step={10000}
              min={1000}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Learning Rate</label>
            <input
              type="number"
              value={lr}
              onChange={e => setLr(Number(e.target.value))}
              title="Learning rate"
              step={0.0001}
              min={0.00001}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Weight Step</label>
            <input
              type="number"
              value={weightStep}
              onChange={e => setWeightStep(Number(e.target.value))}
              title="Weight step"
              step={0.5}
              min={0.5}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
        </div>

        {/* Advanced toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-violet-500 hover:text-violet-600 font-medium mb-3"
        >
          {showAdvanced ? '▾ Hide Advanced' : '▸ Show Advanced'}
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2 pt-4 border-t border-gray-100 dark:border-gray-800">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">N Steps</label>
              <input type="number" value={nSteps} onChange={e => setNSteps(Number(e.target.value))} title="N steps" disabled={isTraining}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Batch Size</label>
              <input type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} title="Batch size" disabled={isTraining}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">N Epochs</label>
              <input type="number" value={nEpochs} onChange={e => setNEpochs(Number(e.target.value))} title="N epochs" disabled={isTraining}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Entropy Coef</label>
              <input type="number" value={entCoef} onChange={e => setEntCoef(Number(e.target.value))} title="Entropy coefficient" step={0.001} disabled={isTraining}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:ring-2 focus:ring-violet-500 disabled:opacity-50" />
            </div>
          </div>
        )}

        {/* Live training stats */}
        {isTraining && trainStatus && (
          <div className="mt-6 p-4 bg-violet-50 dark:bg-violet-900/20 rounded-xl border border-violet-200 dark:border-violet-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-violet-700 dark:text-violet-300">Training in progress</span>
              <span className="text-sm font-mono text-violet-600 dark:text-violet-400">
                {trainStatus.current_step.toLocaleString()} / {trainStatus.total_timesteps.toLocaleString()}
              </span>
            </div>
            <div className="w-full h-2 bg-violet-200 dark:bg-violet-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 rounded-full transition-all duration-500"
                style={{ width: `${trainStatus.progress}%` }}
              />
            </div>
            <div className="flex gap-6 mt-3 text-xs text-violet-600 dark:text-violet-400">
              <span>Episodes: <b>{trainStatus.episode_count.toLocaleString()}</b></span>
              {trainStatus.metrics.accuracy_history.length > 0 && (
                <>
                  <span>
                    Rolling Accuracy:{' '}
                    <b>{(trainStatus.metrics.accuracy_history.at(-1)!.accuracy * 100).toFixed(1)}%</b>
                  </span>
                  <span>
                    Avg Reward:{' '}
                    <b>{trainStatus.metrics.accuracy_history.at(-1)!.avg_reward.toFixed(3)}</b>
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Completed summary */}
        {isCompleted && trainStatus && (
          <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-green-600 dark:text-green-400 text-lg">✓</span>
              <span className="text-sm font-medium text-green-700 dark:text-green-300">Training Complete</span>
            </div>
            <div className="flex gap-6 text-xs text-green-600 dark:text-green-400">
              <span>Final Accuracy: <b>{trainStatus.metrics.final_accuracy != null ? (trainStatus.metrics.final_accuracy * 100).toFixed(1) + '%' : '—'}</b></span>
              <span>Avg Reward: <b>{trainStatus.metrics.final_avg_reward?.toFixed(3) ?? '—'}</b></span>
              <span>Episodes: <b>{trainStatus.episode_count.toLocaleString()}</b></span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
