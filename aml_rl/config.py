"""Hyperparameters and configuration for the AML RL agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── LOB / Reason-code vocabularies (one-hot encoding order) ──────────────
LOB_VOCAB: List[str] = ["BUSINESS", "PB", "RETAIL", "CORPORATE"]
REASON_CODE_VOCAB: List[str] = [
    "STRUCTURING",
    "PEP_ACTIVITY",
    "DORMANT_REACTIVATION",
    "UNUSUAL_PATTERN",
    "LARGE_CASH_DEPOSIT",
    "INT_WIRE",
    "SANCTIONS_HIT",
]

# Feature vector length:
#   4 (lob) + 7 (reason) + 3 (flag counts) + 2 (pep/sanctioned)
#   + 1 (care) + 1 (turnover_norm) + 1 (adverse_media) + 1 (cash_ratio)
#   + 1 (intl_ratio) + 1 (account_age) + 1 (prior_alerts) + 1 (prior_sars)
#   = 24
FEATURE_DIM: int = len(LOB_VOCAB) + len(REASON_CODE_VOCAB) + 13


@dataclass
class WeightBounds:
    """Min/max bounds for each scoring weight the RL agent can adjust."""

    pep_weight: tuple = (20, 60)
    high_flag_weight: tuple = (15, 50)
    med_flag_weight: tuple = (5, 30)
    low_flag_weight: tuple = (1, 15)
    turnover_weight: tuple = (0, 20)
    worldcheck_weight: tuple = (0, 20)
    threshold: tuple = (20, 70)


@dataclass
class DefaultWeights:
    """Starting weights matching the current fixed decision_node."""

    pep: float = 40.0
    high_flag: float = 30.0
    med_flag: float = 15.0
    low_flag: float = 5.0
    turnover: float = 10.0
    worldcheck: float = 10.0
    threshold: float = 40.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "pep": self.pep,
            "high_flag": self.high_flag,
            "med_flag": self.med_flag,
            "low_flag": self.low_flag,
            "turnover": self.turnover,
            "worldcheck": self.worldcheck,
            "threshold": self.threshold,
        }


# ── RL Hyperparameters ───────────────────────────────────────────────────

@dataclass
class RLConfig:
    """Training hyperparameters for PPO."""

    algorithm: str = "PPO"
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    seed: int = 42
    weight_step: float = 2.5  # pts added/removed per action (finer granularity)
    model_dir: str = "models"
    log_dir: str = "logs"


# ── Action space ─────────────────────────────────────────────────────────
# Discrete actions: each nudges a single weight dimension or does nothing.

ACTION_NAMES: List[str] = [
    "LOWER_THRESHOLD",        # 0  threshold -= step
    "RAISE_THRESHOLD",        # 1  threshold += step
    "BOOST_PEP",              # 2  pep_weight += step
    "REDUCE_PEP",             # 3  pep_weight -= step
    "BOOST_FLAG_HIGH",        # 4  high_flag_weight += step
    "REDUCE_FLAG_HIGH",       # 5  high_flag_weight -= step
    "BOOST_FLAG_MED",         # 6  med_flag_weight += step
    "REDUCE_FLAG_MED",        # 7  med_flag_weight -= step
    "BOOST_TURNOVER",         # 8  turnover_weight += step
    "BOOST_WORLDCHECK",       # 9  worldcheck_weight += step
    "NO_CHANGE",              # 10 keep current weights
    "LOWER_THRESHOLD_BIG",    # 11 threshold -= 2*step
    "RAISE_THRESHOLD_BIG",    # 12 threshold += 2*step
    "BOOST_PEP_BIG",          # 13 pep_weight += 2*step
    "REDUCE_PEP_BIG",         # 14 pep_weight -= 2*step
    "BOOST_FLAG_HIGH_BIG",    # 15 high_flag_weight += 2*step
    "REDUCE_FLAG_HIGH_BIG",   # 16 high_flag_weight -= 2*step
    "REDUCE_LOW_FLAG",        # 17 low_flag_weight -= step
    "BOOST_LOW_FLAG",         # 18 low_flag_weight += step
]

NUM_ACTIONS: int = len(ACTION_NAMES)

# ── Reward constants ─────────────────────────────────────────────────────

REWARD_CONFIRM: float = 1.0       # analyst confirms automated decision
REWARD_OVERRIDE: float = -1.0     # analyst overrides (wrong decision)
REWARD_ESCALATE: float = -0.5     # we were too conservative
REWARD_EFFICIENT_CLOSE: float = 0.5  # non-suspicious confirmed fast
