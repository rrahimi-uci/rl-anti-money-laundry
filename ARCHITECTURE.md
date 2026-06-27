# Architecture

This document describes the design of **RL-Anti-Money-Laundry**, a proof-of-concept that applies
reinforcement learning (RL) to dynamically tune the risk-weighting parameters of
an Anti-Money-Laundering (AML) decision pipeline.

- [1. System overview](#1-system-overview)
- [2. The RL problem formulation](#2-the-rl-problem-formulation)
- [3. Core RL module (`aml_rl/`)](#3-core-rl-module-aml_rl)
- [4. Data layer (`data/`)](#4-data-layer-data)
- [5. Training & evaluation (`training/`)](#5-training--evaluation-training)
- [6. Hyperparameter tuning](#6-hyperparameter-tuning)
- [7. Pipeline integration (`integration/`)](#7-pipeline-integration-integration)
- [8. Backend API (`server.py`)](#8-backend-api-serverpy)
- [9. Frontend dashboard (`ui/`)](#9-frontend-dashboard-ui)
- [10. End-to-end data flow](#10-end-to-end-data-flow)
- [11. Design decisions & trade-offs](#11-design-decisions--trade-offs)
- [12. Limitations & future work](#12-limitations--future-work)

---

## 1. System overview

Traditional AML scoring engines apply **fixed, hand-tuned weights** to risk
factors (PEP status, suspicious-activity flags, turnover, adverse media, …) and
compare the resulting score to a fixed threshold. Tuning these weights is a slow,
manual, and largely static process.

RL-Anti-Money-Laundry replaces that static step with an **RL agent that nudges the weights and
threshold per case**, learning from ground-truth labels (offline) or analyst
human-in-the-loop (HITL) feedback (online). The whole thing ships with a training
dashboard so the learning process is observable and reproducible.

```text
┌───────────────────────┐        REST /api/*         ┌──────────────────────────┐
│   React + TS UI        │ ◀────────────────────────▶ │   FastAPI backend         │
│   (Vite, :4003)        │   Vite proxy → :8200       │   server.py (:8200)       │
└───────────────────────┘                            └────────────┬─────────────┘
                                                                   │ imports
        ┌──────────────────────────────┬───────────────────────────┼───────────────────────────┐
        ▼                              ▼                           ▼                           ▼
┌───────────────┐          ┌────────────────────┐       ┌────────────────────┐     ┌────────────────────┐
│  aml_rl/      │          │  training/         │       │  data/             │     │  integration/      │
│  env, config  │  ◀─uses─ │  train, evaluate   │ ◀──── │  generate_episodes │     │  adaptive_decision │
│  features,    │          │  tuning, visualize │       │  episodes.jsonl    │     │  (drop-in node)    │
│  reward       │          └────────────────────┘       └────────────────────┘     └────────────────────┘
└───────────────┘
        ▲
        └── Gymnasium env consumed by Stable-Baselines3 (PPO / A2C / DQN)
```

**Tech stack**

| Layer    | Technologies |
| -------- | ------------ |
| RL core  | Gymnasium, Stable-Baselines3 (PPO/A2C/DQN), NumPy, PyTorch |
| Backend  | FastAPI, Uvicorn, Pandas, Matplotlib |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts |
| Tests    | pytest, FastAPI TestClient (httpx) |

---

## 2. The RL problem formulation

The problem is modelled as a **single-step contextual bandit** (a one-step MDP):

| Element        | Definition |
| -------------- | ---------- |
| **State**      | A 24-dimensional feature vector extracted from one AML investigation case. |
| **Action**     | One of 19 discrete weight adjustments (nudge a single weight/threshold up or down, a "big" variant, or no-op). |
| **Reward**     | Scalar in `[-1, +1]` comparing the resulting decision to ground truth, or derived from analyst HITL gate decisions in live mode. |
| **Episode**    | Exactly one case: observe → act → score → reward → terminate. |

### Why a bandit and not a full MDP?

Each AML case is scored independently — there is no temporal "trajectory" where
one decision changes the next case's state. Framing it as a one-step bandit:

- keeps the credit-assignment problem trivial (reward is immediate),
- is tractable with limited labelled data, and
- maps cleanly onto the existing per-case decision node.

A multi-step variant (weights evolving across a batch of cases) is a natural
future extension; the environment is structured so this can be added without
reworking the feature/reward code.

### Action space (19 discrete actions)

Defined in [`aml_rl/config.py`](aml_rl/config.py) as `ACTION_NAMES`. Each action
either nudges one weight by `±weight_step` (default 2.5 pts), nudges it by
`±2·weight_step` ("BIG" variants), or makes no change:

```
LOWER_THRESHOLD / RAISE_THRESHOLD            (+ BIG variants)
BOOST_PEP / REDUCE_PEP                        (+ BIG variants)
BOOST_FLAG_HIGH / REDUCE_FLAG_HIGH            (+ BIG variants)
BOOST_FLAG_MED / REDUCE_FLAG_MED
BOOST_TURNOVER / BOOST_WORLDCHECK
BOOST_LOW_FLAG / REDUCE_LOW_FLAG
NO_CHANGE
```

Weights are always clamped to the `WeightBounds` ranges after each action.

---

## 3. Core RL module (`aml_rl/`)

### `config.py` — single source of truth

Holds all vocabularies, hyperparameters, and constants so every other module
stays in sync:

- `LOB_VOCAB`, `REASON_CODE_VOCAB` — one-hot encoding order.
- `FEATURE_DIM = 24` — derived from the vocab sizes + numeric features.
- `WeightBounds` — min/max for each adjustable weight.
- `DefaultWeights` — starting weights matching the legacy fixed decision node.
- `RLConfig` — PPO/A2C/DQN hyperparameters and `weight_step`.
- `ACTION_NAMES` / `NUM_ACTIONS` — the 19-action discrete space.
- `REWARD_*` — reward constants.

### `feature_extractor.py` — state → vector

`extract_features(state) -> np.ndarray(24,)` converts a raw investigation state
dict into a fixed-length `float32` vector. It is **defensive by design**: every
field is read with `.get(...)` and sensible defaults, so partial or malformed
states never crash (verified by `test_empty_state_does_not_crash`).

Feature layout:

| Index | Feature | Encoding |
| ----- | ------- | -------- |
| 0–3   | Line of business | one-hot (BUSINESS, PB, RETAIL, CORPORATE) |
| 4–10  | Reason code | one-hot (7 codes) |
| 11–13 | Red-flag counts | HIGH / MEDIUM / LOW |
| 14–15 | PEP / sanctioned | binary |
| 16    | Care marker | binary |
| 17    | Turnover (12m) | normalised `/1e6`, capped at 5.0 |
| 18    | Adverse-media hits | count, capped at 5 |
| 19    | Cash-deposit ratio | `cash / credits`, 0–1 |
| 20    | Intl-transfer ratio | `intl / debits`, 0–1 |
| 21    | Account age | years, capped at 50 |
| 22    | Prior alerts | count, capped at 10 |
| 23    | Prior SARs | count, capped at 5 |

### `env.py` — the Gymnasium environment

`AMLScoringEnv(gym.Env)` is a contextual bandit:

- `observation_space`: `Box(low=-1.0, high=50.0, shape=(24,), float32)`
- `action_space`: `Discrete(19)`
- `reset()` — samples a random case (via a seeded `np.random.Generator`),
  resets weights to defaults, returns `(obs, info)`.
- `step(action)` — applies the weight nudge, re-scores the case with the
  adjusted weights, computes the reward against ground truth, and **always
  terminates** (`terminated=True`).

`_score_case()` reproduces the production scoring rule with the *current*
weights:

```
if care_marker:            → ESCALATED_FIU            (hard override)
score  = pep_weight           if PEP / SANCTIONED
       + Σ flag_weight(sev)   per red flag
       + turnover_weight      if turnover > £500k
       + min(adverse, 3) · worldcheck_weight
decision = SUSPICIOUS if score ≥ threshold else NON_SUSPICIOUS
```

### `reward.py` — reward shaping

`compute_reward(predicted, analyst_decisions, ground_truth=None)` has two modes:

- **Offline (ground truth present):** `+1.0` correct SUSPICIOUS/ESCALATED,
  `+0.5` correct (efficient) NON_SUSPICIOUS, `-0.5` for a missed escalation,
  `-1.0` for a wrong call.
- **Online (HITL gates):** aggregates analyst gate decisions (triage, phase-1,
  phase-2, SAR review) into a normalised reward in `[-1, +1]`.

---

## 4. Data layer (`data/`)

### `generate_episodes.py`

Generates synthetic, labelled episodes by:

1. **Perturbing** four canonical base scenarios (structuring, PEP/sanctions,
   dormant reactivation, clean retail) — jittering turnover, income, and
   sampling red flags.
2. **Synthesising** six additional templated scenarios (shell-company layering,
   sudden wealth, third-party payments, etc.) for diversity.
3. Injecting ~8% label noise to make the learning problem non-trivial.

The base scenarios are loaded from the sibling AML LangGraph project when
present, and otherwise from a **self-contained fallback**
([`data/fallback_scenarios.py`](data/fallback_scenarios.py)) so the generator
works out-of-the-box in this stand-alone repo. Generation is fully deterministic
under a fixed `--seed`.

### `episodes.jsonl`

The committed dataset (2,000 episodes). Each line:

```json
{"state": { ...investigation fields... }, "ground_truth": "SUSPICIOUS", "scenario_key": "pep_sanctions_hit"}
```

### `hitl_feedback.jsonl`

Append-only log of live analyst feedback events, surfaced by `/api/feedback/stats`.

---

## 5. Training & evaluation (`training/`)

### `train.py`

Wraps the env in SB3's `Monitor`, builds a PPO agent from `RLConfig`, and trains
with a `MetricsCallback` that tracks rolling accuracy and reward. Saves the model
to `models/aml_<algo>.zip`.

### `evaluate.py`

Runs the trained policy deterministically over `n_eval` sampled cases and reports
accuracy, average reward, an action-distribution histogram, and a confusion
matrix. Crucially, it **re-seeds the env RNG before both the RL and baseline
loops** so the RL-vs-fixed-weights comparison is *paired* (same cases), making
the reported improvement (`RL Δ`) meaningful rather than noisy.

### `visualize.py`

Generates four PNGs (`matplotlib`, Agg backend): rolling accuracy, reward
distribution, action-selection frequency, and weight evolution.

---

## 6. Hyperparameter tuning

[`training/tuning.py`](training/tuning.py) implements three search strategies
over a configurable parameter space (`ParamRange` supports linear/log scales and
int/float types):

| Strategy   | Mechanism |
| ---------- | --------- |
| **Grid**   | Full Cartesian product, points-per-param derived from the trial budget. |
| **Random** | Seeded uniform/log-uniform sampling. |
| **Bayesian** | A TPE-inspired optimizer (`BayesianOptimizer`) that splits observations into "good"/"bad" quantiles and samples candidates by KDE log-likelihood ratio; falls back to random sampling during cold start. |

`run_tuning_job` orchestrates trials with a `ProcessPoolExecutor` (grid/random
run fully parallel; Bayesian runs in batches so observations feed back), tracks
progress/ETA, and supports cancellation.

---

## 7. Pipeline integration (`integration/`)

[`adaptive_decision.py`](integration/adaptive_decision.py) is a **drop-in
replacement** for the fixed-weight `decision_node` in the AML LangGraph pipeline.

- Toggled by the `USE_RL_POLICY` environment variable (default: `false` → fixed
  weights, identical to the legacy node).
- When enabled, it lazy-loads the trained PPO model, predicts the best weight
  adjustment for the case, applies it, then scores using the same rule as the
  env.
- It returns the exact output contract the legacy node produced (`outcome`,
  `decision_rationale`, `decision_details`, `stage_history`, `messages`).
- The `langchain_core` dependency is **optional**: when absent, a lightweight
  message stand-in is used so the node stays importable and testable
  stand-alone.

---

## 8. Backend API (`server.py`)

A FastAPI app exposing the training/evaluation/data surface consumed by the
dashboard. Training runs in a **background thread** with a shared, lock-guarded
state dict that the UI polls.

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/health` | GET | Liveness check |
| `/api/config` | GET | Feature dim, actions, default weights & config |
| `/api/data/stats` | GET | Dataset statistics |
| `/api/data/episodes` | GET | Paginated episode list |
| `/api/data/episodes/{i}` | GET/PUT/DELETE | CRUD on a single episode |
| `/api/data/episodes` | POST | Append an episode |
| `/api/data/upload` / `/download` | POST/GET | Bulk replace/append / export JSONL |
| `/api/model/status` | GET | Trained models + metadata |
| `/api/train/start` | POST | Start a background training job |
| `/api/train/status` | GET | Live progress & metrics |
| `/api/evaluate` | POST | Evaluate a trained model (paired vs baseline) |
| `/api/plots/{name}` / `/api/plots/generate` | GET/POST | Serve / regenerate plots |
| `/api/feedback/stats` | GET | HITL feedback log summary |

Logs are emitted as structured JSON via a custom `JSONFormatter`.

---

## 9. Frontend dashboard (`ui/`)

A Vite-powered React 19 + TypeScript SPA (served on `:4003`, proxying `/api` to
the backend on `:8200`). Components:

- **KPICards** — headline accuracy / reward / model info.
- **TrainingPanel** — algorithm & hyperparameter configuration; start training.
- **ChartsPanel** — live accuracy/reward curves (1s polling) via Recharts.
- **EvaluationPanel** — run evaluation, view confusion & action distribution.
- **DataPanel** — browse / edit episodes.
- **ConfigPanel** — current weights & system configuration.
- **TuningPanel** — hyperparameter search controls.
- **Header** — dark-mode toggle.

`src/api.ts` is the typed client; `src/types.ts` mirrors the backend schemas.

---

## 10. End-to-end data flow

```text
1. data/generate_episodes.py ──▶ data/episodes.jsonl        (labelled cases)
2. AMLScoringEnv.reset()      ──▶ extract_features(state)    (state → 24-d vector)
3. PPO.predict(obs)           ──▶ action (weight nudge)
4. env.step(action)           ──▶ _apply_action → _score_case → compute_reward
5. SB3 PPO.learn              ──▶ policy gradient update
6. evaluate / visualize       ──▶ metrics + plots
7. integration node          ──▶ production decision using the learned policy
```

---

## 11. Design decisions & trade-offs

- **Bandit over full RL** — immediate, well-defined reward; tractable on small
  data; clean mapping to the per-case node. Cost: cannot model cross-case
  dynamics (yet).
- **Discrete weight nudges over continuous control** — keeps the action space
  small and interpretable ("the agent raised the PEP weight"), at the cost of
  granularity (mitigated by `weight_step` + BIG variants).
- **Config as single source of truth** — `FEATURE_DIM`, `NUM_ACTIONS`, and the
  vocabularies are all derived in one place, preventing drift between the
  extractor, env, and server.
- **Defensive feature extraction** — production AML states are messy; the
  extractor never assumes a key exists.
- **Optional heavy deps** — `langchain_core` (integration) and the sibling
  scenarios project (data gen) are optional, so the core trains and tests with
  only the `requirements.txt` stack.
- **Paired evaluation** — re-seeding before the baseline loop turns a noisy
  unpaired comparison into a fair one.

---

## 12. Limitations & future work

- **Escalation is rule-driven.** `_score_case` only emits `ESCALATED_FIU` via the
  care-marker override; the policy cannot *learn* to escalate from score alone,
  capping accuracy on escalation-labelled cases. A third decision band (escalate
  when score ≥ high-threshold) is the obvious next step.
- **Synthetic data.** Episodes are generated, not real; absolute accuracy numbers
  are illustrative. Real deployment needs validated, labelled production data.
- **Single-step horizon.** A multi-step formulation could let the agent tune a
  *policy of weights* across a portfolio rather than per isolated case.
- **No model registry / versioning.** Models are saved as flat `.zip` files; a
  real system would track lineage, metrics, and approvals.
- **Proof of concept only — not for production AML use.** It demonstrates a
  technique; it is not a compliant, audited AML control.
