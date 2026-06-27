import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import type { DataStats, EvalResult, ModelStatus, RLConfig, TrainConfig, TrainStatus } from './types';
import Header from './components/Header';
import KPICards from './components/KPICards';
import TrainingPanel from './components/TrainingPanel';
import ChartsPanel from './components/ChartsPanel';
import EvaluationPanel from './components/EvaluationPanel';
import DataPanel from './components/DataPanel';
import ConfigPanel from './components/ConfigPanel';

type Tab = 'train' | 'evaluate' | 'data' | 'config';

export default function App() {
  const [dark, setDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [tab, setTab] = useState<Tab>('train');
  const [connected, setConnected] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('PPO');

  // State
  const [config, setConfig] = useState<RLConfig | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoSelectedAlgorithmRef = useRef(false);

  // Dark mode
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  // Initial fetch
  useEffect(() => {
    (async () => {
      try {
        await api.health();
        setConnected(true);
        const [cfg, data, model, status] = await Promise.allSettled([
          api.config(),
          api.dataStats(),
          api.modelStatus(),
          api.trainStatus(),
        ]);
        if (cfg.status === 'fulfilled') {
          setConfig(cfg.value);
        }
        if (data.status === 'fulfilled') {
          setDataStats(data.value);
        }
        if (model.status === 'fulfilled') {
          setModelStatus(model.value);
        }
        if (status.status === 'fulfilled') {
          setTrainStatus(status.value);
        }
      } catch {
        setConnected(false);
      }
    })();
  }, []);

  // Poll training status
  useEffect(() => {
    if (trainStatus?.status === 'training') {
      pollRef.current = setInterval(async () => {
        try {
          const s = await api.trainStatus();
          setTrainStatus(s);
          if (s.status !== 'training') {
            if (pollRef.current) clearInterval(pollRef.current);
            const m = await api.modelStatus();
            setModelStatus(m);
          }
        } catch { /* ignore */ }
      }, 1000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [trainStatus?.status]);

  useEffect(() => {
    if (autoSelectedAlgorithmRef.current) {
      return;
    }

    const latestAvailableAlgorithm = modelStatus?.models?.[0]?.algorithm ?? config?.default_config.algorithm;
    if (!latestAvailableAlgorithm) {
      return;
    }

    setSelectedAlgorithm(latestAvailableAlgorithm);
    autoSelectedAlgorithmRef.current = true;
  }, [config, modelStatus]);

  const handleStartTraining = useCallback(async (cfg: Partial<TrainConfig>) => {
    try {
      await api.trainStart(cfg);
      const s = await api.trainStatus();
      setTrainStatus(s);
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to start training');
    }
  }, []);

  const handleEvaluate = useCallback(async (nEval: number, algorithm = selectedAlgorithm) => {
    setEvaluating(true);
    try {
      const result = await api.evaluate(nEval, algorithm);
      setEvalResult(result);
      setTab('evaluate');
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Evaluation failed');
    } finally {
      setEvaluating(false);
    }
  }, [selectedAlgorithm]);

  const handleDataChanged = useCallback(async () => {
    try {
      const data = await api.dataStats();
      setDataStats(data);
    } catch { /* ignore */ }
  }, []);

  const tabs: { key: Tab; label: string; icon: string }[] = [
    { key: 'train', label: 'Training', icon: '🏋️' },
    { key: 'evaluate', label: 'Evaluation', icon: '📊' },
    { key: 'data', label: 'Data', icon: '📂' },
    { key: 'config', label: 'Config', icon: '⚙️' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors">
      <Header dark={dark} onToggleDark={() => setDark(!dark)} connected={connected} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        <KPICards trainStatus={trainStatus} modelStatus={modelStatus} dataStats={dataStats} evalResult={evalResult} />

        {/* Tab bar */}
        <div className="flex gap-1.5 bg-white dark:bg-gray-900 rounded-2xl p-1.5 shadow-sm border border-gray-200 dark:border-gray-800">
          {tabs.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 py-2.5 px-4 rounded-xl text-sm font-semibold transition-all ${tab === t.key
                  ? 'bg-gradient-to-r from-violet-500 to-indigo-500 text-white shadow-md'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
            >
              <span className="mr-2">{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === 'train' && (
          <div className="space-y-6">
            <TrainingPanel
              trainStatus={trainStatus}
              config={config}
              algorithm={selectedAlgorithm}
              onAlgorithmChange={setSelectedAlgorithm}
              onStart={handleStartTraining}
              onEvaluate={handleEvaluate}
              evaluating={evaluating}
              modelExists={!!modelStatus?.exists}
            />
            <ChartsPanel trainStatus={trainStatus} evalResult={evalResult} />
          </div>
        )}
        {tab === 'evaluate' && (
          <EvaluationPanel
            evalResult={evalResult}
            algorithm={selectedAlgorithm}
            onEvaluate={handleEvaluate}
            evaluating={evaluating}
            modelExists={!!modelStatus?.exists}
          />
        )}
        {tab === 'data' && <DataPanel dataStats={dataStats} onDataChanged={handleDataChanged} />}
        {tab === 'config' && <ConfigPanel config={config} />}
      </main>
    </div>
  );
}
