import type { RLConfig } from '../types';

interface Props {
  config: RLConfig | null;
}

export default function ConfigPanel({ config }: Props) {
  if (!config) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-10 shadow-sm border border-gray-200 dark:border-gray-800 text-center text-gray-400">
        <div className="text-5xl mb-4">⚙️</div>
        <p className="text-sm font-medium">Loading configuration…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Architecture */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h3 className="text-base font-bold mb-5 text-gray-900 dark:text-gray-100">Architecture</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InfoBox label="Algorithm" value="PPO (Stable-Baselines3)" />
          <InfoBox label="Feature Dim" value={String(config.feature_dim)} />
          <InfoBox label="Actions" value={String(config.num_actions)} />
          <InfoBox label="Env Type" value="Contextual Bandit" />
        </div>
      </div>

      {/* Action space */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h3 className="text-base font-bold mb-5 text-gray-900 dark:text-gray-100">Action Space <span className="text-sm font-normal text-gray-400 dark:text-gray-500">({config.num_actions} actions)</span></h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {config.action_names.map((name, i) => {
            const lower = name.toLowerCase();
            const color = lower.includes('lower') || lower.includes('reduce')
              ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800'
              : lower.includes('raise') || lower.includes('boost')
              ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800'
              : lower.includes('no_change')
              ? 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800'
              : 'bg-violet-50 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400 border-violet-200 dark:border-violet-800';
            return (
              <div key={name} className={`px-3 py-2.5 rounded-xl text-xs font-mono font-medium border ${color}`}>
                <span className="opacity-50 mr-1">{i}.</span>{name}
              </div>
            );
          })}
        </div>
      </div>

      {/* Default weights */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h3 className="text-base font-bold mb-5 text-gray-900 dark:text-gray-100">Default Scoring Weights</h3>
        <div className="space-y-4">
          {Object.entries(config.default_weights).map(([key, value]) => (
            <div key={key}>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium text-gray-700 dark:text-gray-300 capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <span className="font-bold font-mono text-violet-600 dark:text-violet-400">{value}</span>
              </div>
              <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 rounded-full transition-all"
                  style={{ width: `${value}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* PPO Hyperparameters */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h3 className="text-base font-bold mb-5 text-gray-900 dark:text-gray-100">PPO Hyperparameters</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InfoBox label="Learning Rate" value={String(config.default_config.learning_rate)} />
          <InfoBox label="N Steps" value={String(config.default_config.n_steps)} />
          <InfoBox label="Batch Size" value={String(config.default_config.batch_size)} />
          <InfoBox label="N Epochs" value={String(config.default_config.n_epochs)} />
          <InfoBox label="Entropy Coef" value={String(config.default_config.ent_coef)} />
          <InfoBox label="Weight Step" value={String(config.default_config.weight_step)} />
          <InfoBox label="Seed" value={String(config.default_config.seed)} />
          <InfoBox label="Timesteps" value={config.default_config.timesteps.toLocaleString()} />
        </div>
      </div>
    </div>
  );
}

function InfoBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 border border-gray-100 dark:border-gray-700">
      <div className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-1">{label}</div>
      <div className="text-sm font-bold text-gray-800 dark:text-gray-200 leading-tight">{value}</div>
    </div>
  );
}
