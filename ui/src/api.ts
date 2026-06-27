import type { DataStats, EpisodesPage, EvalResult, ModelStatus, RLConfig, TrainConfig, TrainStatus } from './types';

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PUT ${path} → ${res.status}`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get<{ status: string }>('/health'),
  config: () => get<RLConfig>('/config'),
  dataStats: () => get<DataStats>('/data/stats'),
  modelStatus: () => get<ModelStatus>('/model/status'),
  trainStart: (cfg: Partial<TrainConfig>) => post<{ status: string; algorithm: string }>('/train/start', cfg),
  trainStatus: () => get<TrainStatus>('/train/status'),
  evaluate: (n_eval = 200, algorithm = 'PPO') => post<EvalResult>('/evaluate', { n_eval, algorithm }),
  generatePlots: () => post<{ status: string }>('/plots/generate'),
  feedbackStats: () => get<{ total: number; entries: unknown[] }>('/feedback/stats'),

  // Data CRUD
  listEpisodes: (page = 1, limit = 50) => get<EpisodesPage>(`/data/episodes?page=${page}&limit=${limit}`),
  getEpisode: (index: number) => get<{ index: number; episode: any }>(`/data/episodes/${index}`),
  updateEpisode: (index: number, episode: any) => put<{ status: string }>(`/data/episodes/${index}`, episode),
  deleteEpisode: (index: number) => del<{ status: string; remaining: number }>(`/data/episodes/${index}`),
  addEpisode: (episode: any) => post<{ status: string; index: number; total: number }>('/data/episodes', episode),
  downloadEpisodes: () => `${BASE}/data/download`,
  uploadEpisodes: async (file: File, mode: 'replace' | 'append' = 'replace') => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/data/upload?mode=${mode}`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json() as Promise<{ status: string; uploaded: number; total: number; mode: string }>;
  },
};
