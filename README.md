# RL-POC — Reinforcement Learning for AML Risk Scoring

A proof-of-concept that uses reinforcement learning (PPO, A2C, DQN) to dynamically optimize risk-weighting parameters in an Anti-Money Laundering decision pipeline. Instead of fixed, hand-tuned weights, an RL agent learns per-case weight adjustments from analyst feedback or ground-truth labels.

## Problem

Traditional AML scoring relies on **static weights** for risk factors (PEP status, suspicious flags, turnover, etc.). This RL agent learns to **dynamically adjust weights per case** based on case features and human-in-the-loop (HITL) feedback, formulated as a single-step contextual bandit.

## Architecture

```text
┌──────────────┐     REST API      ┌──────────────────┐
│  React UI    │ ◄──────────────► │  FastAPI Backend  │
│  (Vite)      │   :4003 ← :8200  │  server.py        │
└──────────────┘                   └────────┬─────────┘
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     │                      │                      │
              ┌──────▼──────┐   ┌───────────▼──────┐   ┌──────────▼──────┐
              │  aml_rl/    │   │  training/       │   │  integration/   │
              │  env, config│   │  train, evaluate │   │  adaptive       │
              │  features   │   │  tuning, viz     │   │  decision node  │
              │  reward     │   │                  │   │                 │
              └─────────────┘   └──────────────────┘   └─────────────────┘
```

### Core Modules

| Module | Purpose |
| ------ | ------- |
| `aml_rl/` | Gymnasium environment, 24-dim feature extraction, reward calculation, config |
| `training/` | Training scripts (PPO/A2C/DQN), evaluation, hyperparameter tuning, visualization |
| `integration/` | Drop-in replacement node for the AML LangGraph pipeline |
| `server.py` | FastAPI backend serving the dashboard and training APIs |
| `ui/` | React + TypeScript dashboard (Vite, Tailwind, Recharts) |

### RL Formulation

- **State**: 24-dimensional feature vector (LOB one-hot, reason codes, flag counts, PEP/sanctions flags, normalized numerics)
- **Actions**: 18 discrete actions — nudge one weight parameter up/down or no-op
- **Reward**: +1.0 correct decision, +0.8 efficient non-suspicious, −1.0 wrong; or HITL gate feedback in live mode
- **Episode**: Single-step (contextual bandit) — one action per AML case

## Quick Start

```bash
bash start.sh
```

This creates a Python venv, installs dependencies, and launches:

- **Backend**: `http://localhost:8200`
- **Frontend**: `http://localhost:4003`

Stop with `Ctrl-C` or:

```bash
bash stop.sh
```

## Manual Setup

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

## CLI Usage

### Train a model

```bash
python -m training.train --episodes data/episodes.jsonl --timesteps 50000
```

### Evaluate a model

```bash
python -m training.evaluate --model models/aml_ppo --episodes data/episodes.jsonl --n_eval 200
```

### Generate plots

```bash
python -m training.visualize --log-dir logs --out plots/
```

### Generate synthetic episodes

```bash
python data/generate_episodes.py
```

## API Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/health` | GET | Health check |
| `/api/config` | GET | RL config & constants |
| `/api/data/stats` | GET | Episode dataset statistics |
| `/api/data/episodes` | GET | Paginated episode list |
| `/api/data/episodes/{idx}` | GET | Single episode details |
| `/api/model/status` | GET | Trained models & metadata |
| `/api/train/start` | POST | Start a training job (background thread) |
| `/api/train/status` | GET | Training progress & live metrics |
| `/api/evaluate` | POST | Run evaluation on a trained model |
| `/api/plots/generate` | POST | Generate visualization PNGs |
| `/api/feedback/stats` | GET | HITL feedback statistics |

## Dashboard

The React dashboard includes:

- **KPI Cards** — accuracy, reward, model info
- **Training Panel** — configure algorithm, timesteps, and hyperparameters; start training
- **Charts Panel** — real-time accuracy and reward graphs (1s polling)
- **Evaluation Panel** — run evaluation and view metrics
- **Data Panel** — browse episodes and case details
- **Config Panel** — view current system configuration and weights
- **Tuning Panel** — hyperparameter search utilities
- **Dark mode** toggle

## Tech Stack

**Backend**: Python · FastAPI · Gymnasium · Stable-Baselines3 · NumPy · Pandas · Matplotlib

**Frontend**: React 19 · TypeScript · Vite · Tailwind CSS · Recharts

## Tests

```bash
pytest tests/
```

Covers environment behavior, feature extraction, and reward calculation.

## Project Structure

```text
rl-poc/
├── server.py                 # FastAPI backend
├── requirements.txt          # Python dependencies
├── start.sh / stop.sh        # Launch & teardown scripts
├── aml_rl/                   # Core RL module
│   ├── config.py             # Hyperparameters, action/reason vocabularies
│   ├── env.py                # Gymnasium environment (contextual bandit)
│   ├── feature_extractor.py  # State → 24-dim feature vector
│   └── reward.py             # Reward calculation
├── training/                 # Training & evaluation
│   ├── train.py              # PPO/A2C/DQN training
│   ├── evaluate.py           # Model evaluation
│   ├── tuning.py             # Hyperparameter tuning
│   └── visualize.py          # Plot generation
├── integration/              # LangGraph pipeline bridge
│   └── adaptive_decision.py  # Drop-in node for dynamic weights
├── data/                     # Episode datasets (JSONL)
├── models/                   # Trained model artifacts (.zip)
├── plots/                    # Generated visualizations
├── tests/                    # Unit tests
└── ui/                       # React + TypeScript dashboard
```
