# RL-Anti-Money-Laundry — Reinforcement Learning for AML Risk Scoring

> A proof-of-concept that uses reinforcement learning (PPO · A2C · DQN) to
> **dynamically tune the risk-weighting parameters** of an Anti-Money-Laundering
> (AML) decision pipeline — instead of fixed, hand-tuned weights, an RL agent
> learns per-case weight adjustments from ground-truth labels or analyst
> human-in-the-loop (HITL) feedback.

📖 **Project site:** <https://rrahimi-uci.github.io/rl-anti-money-laundry/> &nbsp;·&nbsp;
🏗️ **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

**Release:** `aml-v0.0.1` — an educational, reproducible AML risk-scoring prototype
with PPO/A2C/DQN training, paired evaluation, a FastAPI backend, React dashboard, and
99-test suite.

> ⚠️ **Disclaimer:** This is an educational proof of concept. It is **not** a
> compliant, audited AML control and must not be used to make real
> financial-crime decisions.

---

## Problem

Traditional AML scoring relies on **static weights** for risk factors (PEP
status, suspicious-activity flags, turnover, adverse media, …) compared against a
fixed threshold. Tuning those weights is slow and manual. This project frames
weight tuning as a **single-step contextual bandit** and trains an RL agent to
nudge the weights and threshold per case, learning from feedback.

## Architecture at a glance

```text
┌──────────────┐     REST /api/*    ┌──────────────────┐
│  React UI    │ ◀────────────────▶ │  FastAPI Backend  │
│  (Vite :4003)│  proxy → :8200     │  server.py :8200  │
└──────────────┘                    └────────┬─────────┘
                                             │
              ┌──────────────┬───────────────┼───────────────┬───────────────┐
              ▼              ▼               ▼               ▼               ▼
        ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ aml_rl/  │  │ training/  │  │ data/      │  │integration/│
        │ env,     │  │ train,     │  │ generate   │  │ adaptive   │
        │ config,  │  │ evaluate,  │  │ episodes   │  │ decision   │
        │ features,│  │ tuning,    │  │ (JSONL)    │  │ node       │
        │ reward   │  │ visualize  │  │            │  │            │
        └──────────┘  └────────────┘  └────────────┘  └────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

### Core modules

| Module | Purpose |
| ------ | ------- |
| `aml_rl/` | Gymnasium environment, 24-dim feature extraction, reward calculation, config |
| `training/` | Training (PPO/A2C/DQN), evaluation, hyperparameter tuning, visualization |
| `data/` | Synthetic episode generation + the committed `episodes.jsonl` dataset |
| `integration/` | Drop-in decision node for the AML LangGraph pipeline |
| `server.py` | FastAPI backend serving the dashboard and training APIs |
| `ui/` | React + TypeScript dashboard (Vite, Tailwind, Recharts) |

### RL formulation

- **State** — 24-dimensional feature vector (LOB one-hot, reason codes, flag
  counts by severity, PEP/sanctions flags, normalised numerics).
- **Actions** — **19** discrete actions: nudge one weight/threshold up or down,
  a "BIG" double-step variant, or no-op.
- **Reward** — offline: `+1.0` correct suspicious/escalated, `+0.5` correct
  (efficient) non-suspicious, `-0.5` missed escalation, `-1.0` wrong; online:
  HITL gate feedback aggregated to `[-1, +1]`.
- **Episode** — single-step contextual bandit (one action per case).

## Quick start

```bash
bash start.sh
```

This creates a Python venv, installs dependencies, and launches:

- **Backend** → <http://localhost:8200>
- **Frontend** → <http://localhost:4003>

Stop with `Ctrl-C` or `bash stop.sh`.

## Manual setup

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8200 --reload
```

### Frontend

```bash
cd ui
npm install
npm run dev
```

## CLI usage

```bash
# Generate synthetic episodes (self-contained — no external project needed)
python -m data.generate_episodes --out data/episodes.jsonl --count 2000

# Train an agent
python -m training.train --episodes data/episodes.jsonl --timesteps 50000

# Evaluate (paired RL-vs-baseline comparison)
python -m training.evaluate --model models/aml_ppo --episodes data/episodes.jsonl --n-eval 200

# Generate plots
python -m training.visualize --episodes data/episodes.jsonl --model models/aml_ppo --out plots/
```

## API endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/health` | GET | Health check |
| `/api/config` | GET | RL config & constants |
| `/api/data/stats` | GET | Episode dataset statistics |
| `/api/data/episodes` | GET / POST | List (paginated) / append episodes |
| `/api/data/episodes/{i}` | GET / PUT / DELETE | Single-episode CRUD |
| `/api/data/upload` · `/download` | POST / GET | Bulk import / export JSONL |
| `/api/model/status` | GET | Trained models & metadata |
| `/api/train/start` | POST | Start a background training job |
| `/api/train/status` | GET | Training progress & live metrics |
| `/api/evaluate` | POST | Evaluate a trained model |
| `/api/plots/{name}` · `/generate` | GET / POST | Serve / regenerate plots |
| `/api/feedback/stats` | GET | HITL feedback statistics |

## Dashboard

KPI cards · training panel · live accuracy/reward charts · evaluation panel ·
episode browser · config panel · hyperparameter tuning · dark mode.

## Tech stack

**Backend** — Python · FastAPI · Gymnasium · Stable-Baselines3 · PyTorch · NumPy · Pandas · Matplotlib

**Frontend** — React 19 · TypeScript · Vite · Tailwind CSS · Recharts

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

99 tests cover the environment, feature extraction, reward calculation, episode
generation, hyperparameter tuning, the integration node, and the FastAPI API.

## Project structure

```text
rl-anti-money-laundry/
├── server.py                 # FastAPI backend
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Test dependencies
├── start.sh / stop.sh        # Launch & teardown
├── ARCHITECTURE.md           # Full design document
├── aml_rl/                   # Core RL module
│   ├── config.py             # Hyperparameters, action/reason vocabularies
│   ├── env.py                # Gymnasium environment (contextual bandit)
│   ├── feature_extractor.py  # State → 24-dim feature vector
│   └── reward.py             # Reward calculation
├── training/                 # Training, evaluation, tuning, visualization
├── integration/              # LangGraph pipeline bridge (adaptive_decision)
├── data/                     # Episode generation + datasets (JSONL)
│   ├── generate_episodes.py  # Synthetic episode generator
│   └── fallback_scenarios.py # Built-in base scenarios (stand-alone fallback)
├── docs/                     # GitHub Pages site (SEO-optimized)
├── tests/                    # pytest suite (99 tests)
└── ui/                       # React + TypeScript dashboard
```

## License

Released for educational and research purposes. See repository for details.
