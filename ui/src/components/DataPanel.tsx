import { useCallback, useEffect, useRef, useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { api } from '../api';
import type { DataStats, EpisodeSummary, EpisodesPage } from '../types';

interface Props {
  dataStats: DataStats | null;
  onDataChanged: () => void;
}

const COLORS = [
  '#8b5cf6', '#6366f1', '#a855f7', '#7c3aed', '#4f46e5',
  '#c084fc', '#818cf8', '#a78bfa', '#6d28d9', '#4338ca',
];

export default function DataPanel({ dataStats, onDataChanged }: Props) {
  const [subTab, setSubTab] = useState<'overview' | 'browse' | 'editor'>('overview');
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadMode, setUploadMode] = useState<'replace' | 'append'>('replace');

  // Browse state
  const [epsPage, setEpsPage] = useState<EpisodesPage | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);

  // Editor state
  const [editIndex, setEditIndex] = useState<number | null>(null);
  const [editJson, setEditJson] = useState('');
  const [editError, setEditError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const fetchPage = useCallback(async (p: number) => {
    setLoading(true);
    try {
      const data = await api.listEpisodes(p, 50);
      setEpsPage(data);
      setPage(p);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (subTab === 'browse') fetchPage(page);
  }, [subTab]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);
    try {
      const result = await api.uploadEpisodes(file, uploadMode);
      setUploadMsg(`${uploadMode === 'append' ? 'Appended' : 'Replaced'} ${result.uploaded} episodes (total: ${result.total})`);
      onDataChanged();
      if (subTab === 'browse') fetchPage(1);
    } catch (err: unknown) {
      setUploadMsg(err instanceof Error ? err.message : 'Upload failed');
    }
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDelete = async (index: number) => {
    if (!confirm(`Delete episode #${index}?`)) return;
    try {
      await api.deleteEpisode(index);
      onDataChanged();
      fetchPage(page);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const openEditor = async (index: number) => {
    try {
      const data = await api.getEpisode(index);
      setEditIndex(index);
      setEditJson(JSON.stringify(data.episode, null, 2));
      setEditError(null);
      setSubTab('editor');
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to load episode');
    }
  };

  const handleSave = async () => {
    if (editIndex === null) return;
    setEditError(null);
    setSaving(true);
    try {
      const parsed = JSON.parse(editJson);
      if (!parsed.state || !parsed.ground_truth) {
        setEditError("Episode must have 'state' and 'ground_truth' fields");
        setSaving(false);
        return;
      }
      await api.updateEpisode(editIndex, parsed);
      onDataChanged();
      setSubTab('browse');
      fetchPage(page);
    } catch (err: unknown) {
      setEditError(err instanceof Error ? err.message : 'Save failed');
    }
    setSaving(false);
  };

  const handleAddNew = () => {
    setEditIndex(-1);
    setEditJson(JSON.stringify({
      state: {
        alert_id: "NEW-ALERT-001",
        customer_cis: "CIS00000001",
        lob: "RETAIL",
        reason_code: "",
        customer_profile: {},
        pep_status: "CLEAR",
        worldcheck_result: {},
        turnover_12m: 0,
        transaction_summary: {},
        red_flags: [],
        alert_history: [],
        sar_history: []
      },
      ground_truth: "NON_SUSPICIOUS",
      scenario_key: "custom"
    }, null, 2));
    setEditError(null);
    setSubTab('editor');
  };

  const handleSaveNew = async () => {
    setEditError(null);
    setSaving(true);
    try {
      const parsed = JSON.parse(editJson);
      if (!parsed.state || !parsed.ground_truth) {
        setEditError("Episode must have 'state' and 'ground_truth' fields");
        setSaving(false);
        return;
      }
      await api.addEpisode(parsed);
      onDataChanged();
      setSubTab('browse');
      fetchPage(1);
    } catch (err: unknown) {
      setEditError(err instanceof Error ? err.message : 'Save failed');
    }
    setSaving(false);
  };

  if (!dataStats) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-10 shadow-sm border border-gray-200 dark:border-gray-800 text-center text-gray-400">
        <div className="text-5xl mb-4">📂</div>
        <p className="text-sm font-medium">Loading data statistics…</p>
      </div>
    );
  }

  const scenarioData = Object.entries(dataStats.scenarios)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  const verdictData = Object.entries(dataStats.verdicts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">Training Data</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {dataStats.total_episodes.toLocaleString()} episodes · {scenarioData.length} scenarios
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <a
              href={api.downloadEpisodes()}
              download="episodes.jsonl"
              className="px-3 py-2 text-xs font-medium rounded-lg border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              ⬇ Download
            </a>
            <select
              value={uploadMode}
              onChange={e => setUploadMode(e.target.value as 'replace' | 'append')}
              title="Upload mode"
              className="px-2 py-2 text-xs bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-700 dark:text-gray-300"
            >
              <option value="replace">Replace</option>
              <option value="append">Append</option>
            </select>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="px-3 py-2 text-xs font-medium rounded-lg bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400 border border-violet-300 dark:border-violet-700 hover:bg-violet-200 dark:hover:bg-violet-900/50 disabled:opacity-50 transition-colors"
            >
              {uploading ? '⏳ Uploading…' : '⬆ Upload .jsonl'}
            </button>
            <input ref={fileInputRef} type="file" accept=".jsonl,.json" onChange={handleUpload} className="hidden" title="Upload episodes file" />
            <button
              onClick={handleAddNew}
              className="px-3 py-2 text-xs font-medium rounded-lg bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-700 hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
            >
              + Add Episode
            </button>
          </div>
        </div>
        {uploadMsg && (
          <div className={`mt-3 text-xs px-3 py-2 rounded-lg ${
            uploadMsg.includes('failed') || uploadMsg.includes('error')
              ? 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400'
              : 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400'
          }`}>
            {uploadMsg}
          </div>
        )}
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 bg-white dark:bg-gray-900 rounded-lg p-1 shadow-sm border border-gray-200 dark:border-gray-800 w-fit">
        {([
          { key: 'overview' as const, label: 'Overview', icon: '📊' },
          { key: 'browse' as const, label: 'Browse & Edit', icon: '📋' },
        ]).map(t => (
          <button
            key={t.key}
            onClick={() => setSubTab(t.key)}
            className={`py-1.5 px-4 rounded-md text-xs font-medium transition-all ${
              subTab === t.key
                ? 'bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
          >
            <span className="mr-1">{t.icon}</span>{t.label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {subTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
            <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Scenarios</h3>
            <div className="flex items-center gap-6">
              <div className="w-48 h-48 flex-shrink-0">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={scenarioData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} innerRadius={40} stroke="none">
                      {scenarioData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '8px', fontSize: 12 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex-1 space-y-1.5 max-h-48 overflow-auto">
                {scenarioData.map((s, i) => (
                  <div key={s.name} className="flex items-center gap-2 text-xs">
                    <span className="w-3 h-3 rounded-sm flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    <span className="flex-1 truncate text-gray-700 dark:text-gray-300">{s.name}</span>
                    <span className="font-mono text-gray-500">{s.count}</span>
                    <span className="text-gray-400 w-10 text-right">{((s.count / dataStats.total_episodes) * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 shadow-sm border border-gray-200 dark:border-gray-800">
            <h3 className="text-base font-bold mb-4 text-gray-900 dark:text-gray-100">Ground Truth Verdicts</h3>
            <div className="space-y-3">
              {verdictData.map(v => {
                const pct = (v.count / dataStats.total_episodes) * 100;
                const colorMap: Record<string, string> = { CLOSE_NO_SAR: 'bg-green-500', FILE_SAR: 'bg-red-500', ESCALATE: 'bg-amber-500' };
                const barColor = colorMap[v.name] ?? 'bg-violet-500';
                return (
                  <div key={v.name}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-gray-700 dark:text-gray-300">{v.name}</span>
                      <span className="text-gray-500">{v.count} ({pct.toFixed(0)}%)</span>
                    </div>
                    <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                      <div className={`h-full ${barColor} rounded-full`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Browse & Edit */}
      {subTab === 'browse' && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-gray-400 text-sm">Loading episodes…</div>
          ) : epsPage ? (
            <>
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">#</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Alert ID</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">LOB</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Reason</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Scenario</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Verdict</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Risk</th>
                      <th className="px-3 py-2.5 text-left text-gray-500 dark:text-gray-400 font-medium">Flags</th>
                      <th className="px-3 py-2.5 text-right text-gray-500 dark:text-gray-400 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                    {epsPage.episodes.map(ep => (
                      <EpisodeRow key={ep.index} ep={ep} onEdit={() => openEditor(ep.index)} onDelete={() => handleDelete(ep.index)} />
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100 dark:border-gray-800">
                <span className="text-xs text-gray-500">
                  Showing {(epsPage.page - 1) * epsPage.limit + 1}–{Math.min(epsPage.page * epsPage.limit, epsPage.total)} of {epsPage.total}
                </span>
                <div className="flex gap-1">
                  <button onClick={() => fetchPage(page - 1)} disabled={page <= 1} className="px-3 py-1 text-xs rounded-md border border-gray-300 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">← Prev</button>
                  <span className="px-3 py-1 text-xs text-gray-500">Page {page}</span>
                  <button onClick={() => fetchPage(page + 1)} disabled={page * epsPage.limit >= epsPage.total} className="px-3 py-1 text-xs rounded-md border border-gray-300 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">Next →</button>
                </div>
              </div>
            </>
          ) : (
            <div className="p-8 text-center text-gray-400 text-sm">No episodes loaded</div>
          )}
        </div>
      )}

      {/* JSON Editor */}
      {subTab === 'editor' && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">{editIndex === -1 ? 'New Episode' : `Episode #${editIndex}`}</h3>
            <div className="flex gap-2">
              <button onClick={() => setSubTab('browse')} className="px-3 py-1.5 text-xs rounded-lg border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">Cancel</button>
              <button
                onClick={editIndex === -1 ? handleSaveNew : handleSave}
                disabled={saving}
                className="px-4 py-1.5 text-xs font-medium rounded-lg bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-sm hover:shadow-md disabled:opacity-50 transition-all"
              >
                {saving ? '⏳ Saving…' : '💾 Save'}
              </button>
            </div>
          </div>
          {editError && (
            <div className="mb-3 text-xs px-3 py-2 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">{editError}</div>
          )}
          <textarea
            value={editJson}
            onChange={e => setEditJson(e.target.value)}
            spellCheck={false}
            title="Episode JSON editor"
            className="w-full h-[500px] font-mono text-xs bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 resize-y focus:ring-2 focus:ring-violet-500 focus:border-transparent"
          />
          <p className="mt-2 text-[10px] text-gray-400">
            JSON must include <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">state</code>,{' '}
            <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">ground_truth</code>, and optionally{' '}
            <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">scenario_key</code>
          </p>
        </div>
      )}
    </div>
  );
}

function EpisodeRow({ ep, onEdit, onDelete }: { ep: EpisodeSummary; onEdit: () => void; onDelete: () => void }) {
  const verdictColor: Record<string, string> = {
    SUSPICIOUS: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400',
    NON_SUSPICIOUS: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400',
    ESCALATED_FIU: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
  };
  const riskColor: Record<string, string> = { HIGH: 'text-red-500', MEDIUM: 'text-amber-500', LOW: 'text-green-500' };

  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
      <td className="px-3 py-2 font-mono text-gray-400">{ep.index}</td>
      <td className="px-3 py-2 font-mono">{ep.alert_id}</td>
      <td className="px-3 py-2">{ep.lob}</td>
      <td className="px-3 py-2 truncate max-w-[120px]">{ep.reason_code}</td>
      <td className="px-3 py-2 truncate max-w-[120px]">{ep.scenario_key}</td>
      <td className="px-3 py-2">
        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${verdictColor[ep.ground_truth] ?? 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{ep.ground_truth}</span>
      </td>
      <td className={`px-3 py-2 font-medium ${riskColor[ep.risk_rating] ?? 'text-gray-500'}`}>{ep.risk_rating}</td>
      <td className="px-3 py-2 text-center">{ep.red_flag_count}</td>
      <td className="px-3 py-2 text-right">
        <div className="flex justify-end gap-1">
          <button onClick={onEdit} className="px-2 py-1 text-xs rounded-md border border-violet-300 dark:border-violet-700 text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/30 transition-colors" title="Edit">✏️</button>
          <button onClick={onDelete} className="px-2 py-1 text-xs rounded-md border border-red-300 dark:border-red-700 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors" title="Delete">🗑</button>
        </div>
      </td>
    </tr>
  );
}
