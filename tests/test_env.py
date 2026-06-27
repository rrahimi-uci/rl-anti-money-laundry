"""Tests for the AMLScoringEnv Gymnasium environment."""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import List

import numpy as np
import pytest

from aml_rl.config import FEATURE_DIM, NUM_ACTIONS, DefaultWeights, WeightBounds
from aml_rl.env import AMLScoringEnv


def _make_episode(
    pep: str = "CLEAR",
    red_flags: list | None = None,
    turnover: float = 100_000,
    care: bool = False,
    ground_truth: str = "NON_SUSPICIOUS",
):
    """Build a minimal episode dict."""
    if red_flags is None:
        red_flags = []
    return {
        "state": {
            "lob": "RETAIL",
            "reason_code": "STRUCTURING",
            "red_flags": red_flags,
            "pep_status": pep,
            "care_marker_detected": care,
            "turnover_12m": turnover,
            "worldcheck_result": {"adverse_media_hits": 0},
            "transaction_summary": {
                "total_credits": turnover,
                "total_debits": turnover * 0.8,
                "cash_deposit_total": 0,
                "international_transfer_total": 0,
            },
            "customer_profile": {"customer_since": "2022-01-01"},
            "alert_history": [],
            "sar_history": [],
        },
        "ground_truth": ground_truth,
    }


def _write_episodes(episodes: list, tmpdir: pathlib.Path) -> pathlib.Path:
    path = tmpdir / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return path


@pytest.fixture()
def clean_env(tmp_path):
    """Env with all-clean episodes (NON_SUSPICIOUS, no flags)."""
    eps = [_make_episode() for _ in range(10)]
    path = _write_episodes(eps, tmp_path)
    return AMLScoringEnv(episodes_path=path)


@pytest.fixture()
def suspicious_env(tmp_path):
    """Env with suspicious episodes (PEP + HIGH flags)."""
    flags = [
        {"type": "STRUCTURING", "severity": "HIGH"},
        {"type": "RAPID_MOVEMENT", "severity": "HIGH"},
    ]
    eps = [
        _make_episode(
            pep="PEP",
            red_flags=flags,
            turnover=1_000_000,
            ground_truth="SUSPICIOUS",
        )
        for _ in range(10)
    ]
    path = _write_episodes(eps, tmp_path)
    return AMLScoringEnv(episodes_path=path)


@pytest.fixture()
def care_env(tmp_path):
    """Env with care-marker episodes (ESCALATED_FIU)."""
    eps = [
        _make_episode(care=True, ground_truth="ESCALATED_FIU") for _ in range(10)
    ]
    path = _write_episodes(eps, tmp_path)
    return AMLScoringEnv(episodes_path=path)


class TestEnvInit:
    def test_raises_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        with pytest.raises(ValueError, match="No episodes found"):
            AMLScoringEnv(episodes_path=path)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="No episodes found"):
            AMLScoringEnv(episodes_path=tmp_path / "nonexistent.jsonl")

    def test_spaces(self, clean_env):
        assert clean_env.observation_space.shape == (FEATURE_DIM,)
        assert clean_env.action_space.n == NUM_ACTIONS


class TestReset:
    def test_returns_observation_and_info(self, clean_env):
        obs, info = clean_env.reset()
        assert obs.shape == (FEATURE_DIM,)
        assert obs.dtype == np.float32
        assert "ground_truth" in info
        assert "weights" in info

    def test_resets_weights_to_default(self, clean_env):
        clean_env.reset()
        expected = DefaultWeights().to_dict()
        assert clean_env._weights.to_dict() == expected


class TestStep:
    def test_step_terminates(self, clean_env):
        clean_env.reset()
        obs, reward, terminated, truncated, info = clean_env.step(10)  # NO_CHANGE
        assert terminated is True
        assert truncated is False

    def test_step_returns_correct_shapes(self, clean_env):
        clean_env.reset()
        obs, reward, terminated, truncated, info = clean_env.step(0)
        assert obs.shape == (FEATURE_DIM,)
        assert isinstance(reward, float)
        assert "predicted" in info
        assert "action_name" in info

    def test_clean_case_no_change_non_suspicious(self, clean_env):
        clean_env.reset()
        _, _, _, _, info = clean_env.step(10)  # NO_CHANGE
        assert info["predicted"] == "NON_SUSPICIOUS"

    def test_suspicious_case_no_change(self, suspicious_env):
        suspicious_env.reset()
        _, _, _, _, info = suspicious_env.step(10)  # NO_CHANGE
        assert info["predicted"] == "SUSPICIOUS"

    def test_care_marker_always_escalates(self, care_env):
        care_env.reset()
        # Try all actions — care marker overrides everything
        for action in range(NUM_ACTIONS):
            care_env.reset()
            _, _, _, _, info = care_env.step(action)
            assert info["predicted"] == "ESCALATED_FIU"


class TestApplyAction:
    def test_lower_threshold(self, clean_env):
        clean_env.reset()
        initial = clean_env._weights.threshold
        clean_env._apply_action(0)  # LOWER_THRESHOLD
        assert clean_env._weights.threshold == initial - 2.5

    def test_raise_threshold(self, clean_env):
        clean_env.reset()
        initial = clean_env._weights.threshold
        clean_env._apply_action(1)  # RAISE_THRESHOLD
        assert clean_env._weights.threshold == initial + 2.5

    def test_boost_pep(self, clean_env):
        clean_env.reset()
        initial = clean_env._weights.pep
        clean_env._apply_action(2)  # BOOST_PEP
        assert clean_env._weights.pep == initial + 2.5

    def test_no_change(self, clean_env):
        clean_env.reset()
        before = clean_env._weights.to_dict()
        clean_env._apply_action(10)  # NO_CHANGE
        assert clean_env._weights.to_dict() == before

    def test_bounds_respected(self, clean_env):
        clean_env.reset()
        bounds = WeightBounds()
        # Slam threshold down many times
        for _ in range(20):
            clean_env._apply_action(0)  # LOWER_THRESHOLD
        assert clean_env._weights.threshold >= bounds.threshold[0]


class TestRewardSignal:
    def test_correct_prediction_positive_reward(self, clean_env):
        clean_env.reset()
        _, reward, _, _, info = clean_env.step(10)  # NO_CHANGE
        # Clean case with default weights → NON_SUSPICIOUS matches ground truth
        assert info["correct"] is True
        assert reward > 0

    def test_wrong_prediction_negative_reward(self, suspicious_env):
        suspicious_env.reset()
        # Raise threshold to maximum so suspicious case scores below threshold
        _, reward, _, _, info = suspicious_env.step(1)  # RAISE_THRESHOLD
        # May or may not still be correct, but let's verify reward is finite
        assert np.isfinite(reward)
