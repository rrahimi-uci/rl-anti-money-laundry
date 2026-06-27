export interface TrainConfig {
  algorithm: string;
  timesteps: number;
  learning_rate: number;
  n_steps: number;
  batch_size: number;
  n_epochs: number;
  ent_coef: number;
  weight_step: number;
  seed: number;
}

export interface AccuracyPoint {
  step: number;
  accuracy: number;
  avg_reward: number;
  episode_count: number;
}

export interface TrainStatus {
  status: 'idle' | 'training' | 'completed' | 'error';
  progress: number;
  total_timesteps: number;
  current_step: number;
  episode_count: number;
  metrics: {
    accuracy_history: AccuracyPoint[];
    final_accuracy: number | null;
    final_avg_reward: number | null;
  };
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  config: TrainConfig | null;
}

export interface DataStats {
  total_episodes: number;
  scenarios: Record<string, number>;
  verdicts: Record<string, number>;
}

export interface ModelInfo {
  algorithm: string;
  path: string;
  size_kb: number;
  modified: string;
}

export interface ModelStatus {
  exists: boolean;
  models: ModelInfo[];
  path: string;
  size_kb?: number;
  modified?: string;
}

export interface ActionDist {
  action: string;
  count: number;
  pct: number;
}

export interface EpisodeResult {
  episode: number;
  action: string;
  predicted: string;
  ground_truth: string;
  reward: number;
  correct: boolean;
}

export interface EvalResult {
  accuracy: number;
  avg_reward: number;
  baseline_accuracy: number;
  improvement: number;
  total_episodes: number;
  correct: number;
  action_distribution: ActionDist[];
  reward_distribution: { values: number[]; mean: number };
  per_episode: EpisodeResult[];
}

export interface RLConfig {
  feature_dim: number;
  num_actions: number;
  action_names: string[];
  algorithms: string[];
  default_weights: Record<string, number>;
  default_config: TrainConfig;
}

export interface EpisodeSummary {
  index: number;
  alert_id: string;
  customer_cis: string;
  lob: string;
  reason_code: string;
  scenario_key: string;
  ground_truth: string;
  risk_rating: string;
  turnover_12m: number;
  red_flag_count: number;
}

export interface EpisodesPage {
  total: number;
  page: number;
  limit: number;
  episodes: EpisodeSummary[];
}
