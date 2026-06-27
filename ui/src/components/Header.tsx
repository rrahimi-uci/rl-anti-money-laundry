interface Props {
  dark: boolean;
  onToggleDark: () => void;
  connected: boolean;
}

export default function Header({ dark, onToggleDark, connected }: Props) {
  return (
    <header className="bg-gradient-to-r from-violet-700 via-indigo-600 to-purple-700 shadow-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 bg-white/20 rounded-2xl flex items-center justify-center text-2xl backdrop-blur-sm shadow-inner border border-white/10">
            🧠
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight leading-tight">RL Training Dashboard</h1>
            <p className="text-sm text-violet-200 font-medium mt-0.5">AML Scoring Weight Optimisation &nbsp;·&nbsp; PPO Agent</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-white/10 rounded-xl px-3 py-2 backdrop-blur-sm border border-white/10">
            <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${connected ? 'bg-emerald-400 animate-pulse shadow-[0_0_6px_#34d399]' : 'bg-red-400'}`} />
            <span className="text-sm font-medium text-white">{connected ? 'Connected' : 'Offline'}</span>
          </div>
          <button
            onClick={onToggleDark}
            className="w-10 h-10 rounded-xl bg-white/10 hover:bg-white/25 flex items-center justify-center text-lg text-white transition-all border border-white/10 hover:border-white/25"
            title="Toggle dark mode"
          >
            {dark ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </header>
  );
}
