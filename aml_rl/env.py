"""Gymnasium environment for AML adaptive risk scoring.

The agent observes investigation features, selects a weight-adjustment action,
and receives reward from analyst feedback (or simulated ground truth).

Each episode is ONE investigation:
    1. Environment samples a case from the episode dataset.
    2. Agent observes the feature vector.
    3. Agent picks an action (weight nudge).
    4. Environment applies the adjusted weights, computes the decision,
       compares with ground truth, and returns a reward.
    5. Episode ends (single-step MDP — one action per case).

This is intentionally a *contextual bandit* formulation: the agent sees one
state per episode and takes one action.  This keeps the problem tractable
with limited data.  Multi-step variants (weight evolution across a batch of
cases) can be added later.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import (
    ACTION_NAMES,
    FEATURE_DIM,
    NUM_ACTIONS,
    DefaultWeights,
    RLConfig,
    WeightBounds,
)
from .feature_extractor import extract_features
from .reward import compute_reward


class AMLScoringEnv(gym.Env):
    """Contextual-bandit environment for AML risk-weight tuning.

    Parameters
    ----------
    episodes_path:
        Path to a JSONL file where each line is a JSON object with keys:
          - ``state``: full AML investigation state dict
          - ``ground_truth``: expected outcome (SUSPICIOUS / NON_SUSPICIOUS / ESCALATED_FIU)
    config:
        RL hyperparameter config.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episodes_path: str | pathlib.Path,
        config: Optional[RLConfig] = None,
    ):
        super().__init__()
        self.config = config or RLConfig()
        self.bounds = WeightBounds()

        # Load episode dataset
        self.episodes: List[Dict[str, Any]] = []
        ep_path = pathlib.Path(episodes_path)
        if ep_path.exists():
            with open(ep_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.episodes.append(json.loads(line))

        if not self.episodes:
            raise ValueError(f"No episodes found at {ep_path}")

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=50.0, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Mutable weights (reset each episode to defaults)
        self._weights = DefaultWeights()
        self._current_idx = 0
        self._current_state: Dict[str, Any] = {}
        self._current_gt: str = ""
        self._rng = np.random.default_rng(self.config.seed)

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample a random episode
        idx = int(self._rng.integers(0, len(self.episodes)))
        episode = self.episodes[idx]
        self._current_state = episode["state"]
        self._current_gt = episode["ground_truth"]
        self._current_idx = idx

        # Reset weights to default
        self._weights = DefaultWeights()

        obs = extract_features(self._current_state)
        info = {
            "episode_idx": idx,
            "ground_truth": self._current_gt,
            "weights": self._weights.to_dict(),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply weight adjustment, score the case, and return reward."""

        # 1. Apply the action to weights
        self._apply_action(action)

        # 2. Score the current case with adjusted weights
        predicted = self._score_case(self._current_state)

        # 3. Compute reward against ground truth
        reward = compute_reward(
            predicted_outcome=predicted,
            analyst_decisions={},
            ground_truth_outcome=self._current_gt,
        )

        # 4. Episode always terminates after one step (bandit)
        obs = extract_features(self._current_state)
        info = {
            "predicted": predicted,
            "ground_truth": self._current_gt,
            "correct": predicted == self._current_gt,
            "weights": self._weights.to_dict(),
            "action_name": ACTION_NAMES[action],
        }

        return obs, reward, True, False, info  # terminated=True, truncated=False

    # ── Internal helpers ─────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """Nudge one weight dimension by ±step, clamped to bounds."""
        step = self.config.weight_step
        b = self.bounds

        if action == 0:  # LOWER_THRESHOLD
            self._weights.threshold = max(
                b.threshold[0], self._weights.threshold - step
            )
        elif action == 1:  # RAISE_THRESHOLD
            self._weights.threshold = min(
                b.threshold[1], self._weights.threshold + step
            )
        elif action == 2:  # BOOST_PEP
            self._weights.pep = min(b.pep_weight[1], self._weights.pep + step)
        elif action == 3:  # REDUCE_PEP
            self._weights.pep = max(b.pep_weight[0], self._weights.pep - step)
        elif action == 4:  # BOOST_FLAG_HIGH
            self._weights.high_flag = min(
                b.high_flag_weight[1], self._weights.high_flag + step
            )
        elif action == 5:  # REDUCE_FLAG_HIGH
            self._weights.high_flag = max(
                b.high_flag_weight[0], self._weights.high_flag - step
            )
        elif action == 6:  # BOOST_FLAG_MED
            self._weights.med_flag = min(
                b.med_flag_weight[1], self._weights.med_flag + step
            )
        elif action == 7:  # REDUCE_FLAG_MED
            self._weights.med_flag = max(
                b.med_flag_weight[0], self._weights.med_flag - step
            )
        elif action == 8:  # BOOST_TURNOVER
            self._weights.turnover = min(
                b.turnover_weight[1], self._weights.turnover + step
            )
        elif action == 9:  # BOOST_WORLDCHECK
            self._weights.worldcheck = min(
                b.worldcheck_weight[1], self._weights.worldcheck + step
            )
        # action == 10 → NO_CHANGE
        elif action == 11:  # LOWER_THRESHOLD_BIG
            self._weights.threshold = max(
                b.threshold[0], self._weights.threshold - 2 * step
            )
        elif action == 12:  # RAISE_THRESHOLD_BIG
            self._weights.threshold = min(
                b.threshold[1], self._weights.threshold + 2 * step
            )
        elif action == 13:  # BOOST_PEP_BIG
            self._weights.pep = min(b.pep_weight[1], self._weights.pep + 2 * step)
        elif action == 14:  # REDUCE_PEP_BIG
            self._weights.pep = max(b.pep_weight[0], self._weights.pep - 2 * step)
        elif action == 15:  # BOOST_FLAG_HIGH_BIG
            self._weights.high_flag = min(
                b.high_flag_weight[1], self._weights.high_flag + 2 * step
            )
        elif action == 16:  # REDUCE_FLAG_HIGH_BIG
            self._weights.high_flag = max(
                b.high_flag_weight[0], self._weights.high_flag - 2 * step
            )
        elif action == 17:  # REDUCE_LOW_FLAG
            self._weights.low_flag = max(
                b.low_flag_weight[0], self._weights.low_flag - step
            )
        elif action == 18:  # BOOST_LOW_FLAG
            self._weights.low_flag = min(
                b.low_flag_weight[1], self._weights.low_flag + step
            )

    def _score_case(self, state: Dict[str, Any]) -> str:
        """Reproduce the decision_node scoring with current adaptive weights."""

        red_flags = state.get("red_flags", [])
        pep = state.get("pep_status", "CLEAR")
        care = state.get("care_marker_detected", False)
        turnover = float(state.get("turnover_12m", 0))
        wc = state.get("worldcheck_result", {})
        adverse = wc.get("adverse_media", wc.get("adverse_media_hits", 0))
        if isinstance(adverse, list):
            adverse = len(adverse)

        # Care marker → override
        if care:
            return "ESCALATED_FIU"

        # Weighted score
        score = 0.0
        if pep in ("PEP", "SANCTIONED"):
            score += self._weights.pep

        for rf in red_flags:
            sev = rf.get("severity", "LOW")
            if sev == "HIGH":
                score += self._weights.high_flag
            elif sev == "MEDIUM":
                score += self._weights.med_flag
            else:
                score += self._weights.low_flag

        if turnover > 500_000:
            score += self._weights.turnover

        score += min(adverse, 3) * self._weights.worldcheck

        if score >= self._weights.threshold:
            return "SUSPICIOUS"
        return "NON_SUSPICIOUS"
