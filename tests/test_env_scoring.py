"""Edge-case tests for AMLScoringEnv internal scoring and action application."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aml_rl.config import ACTION_NAMES, NUM_ACTIONS, DefaultWeights, WeightBounds
from aml_rl.env import AMLScoringEnv


def _env_with(states, tmp_path):
    path = tmp_path / "eps.jsonl"
    with open(path, "w") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    return AMLScoringEnv(episodes_path=path)


def _episode(**state):
    base = {
        "lob": "RETAIL",
        "reason_code": "STRUCTURING",
        "red_flags": [],
        "pep_status": "CLEAR",
        "care_marker_detected": False,
        "turnover_12m": 0,
        "worldcheck_result": {"adverse_media_hits": 0},
        "transaction_summary": {},
        "customer_profile": {},
        "alert_history": [],
        "sar_history": [],
    }
    base.update(state)
    return {"state": base, "ground_truth": "NON_SUSPICIOUS"}


class TestScoreCase:
    def test_pep_adds_pep_weight(self, tmp_path):
        env = _env_with([_episode(pep_status="PEP")], tmp_path)
        env.reset()
        # default pep weight 40 == default threshold 40 → SUSPICIOUS
        _, _, _, _, info = env.step(10)
        assert info["predicted"] == "SUSPICIOUS"

    def test_sanctioned_treated_like_pep(self, tmp_path):
        env = _env_with([_episode(pep_status="SANCTIONED")], tmp_path)
        env.reset()
        _, _, _, _, info = env.step(10)
        assert info["predicted"] == "SUSPICIOUS"

    def test_turnover_above_500k_contributes(self, tmp_path):
        # turnover weight (10) alone is below threshold (40) → NON_SUSPICIOUS
        env = _env_with([_episode(turnover_12m=600_000)], tmp_path)
        env.reset()
        _, _, _, _, info = env.step(10)
        assert info["predicted"] == "NON_SUSPICIOUS"

    def test_adverse_media_capped_at_three(self, tmp_path):
        # 10 adverse hits capped at 3 → 3 * 10 = 30 < 40 threshold
        env = _env_with(
            [_episode(worldcheck_result={"adverse_media_hits": 10})], tmp_path
        )
        env.reset()
        _, _, _, _, info = env.step(10)
        assert info["predicted"] == "NON_SUSPICIOUS"

    def test_lower_threshold_can_flip_to_suspicious(self, tmp_path):
        # score = worldcheck 3*10 = 30; default threshold 40 → NON_SUSPICIOUS.
        env = _env_with(
            [_episode(worldcheck_result={"adverse_media_hits": 3})], tmp_path
        )
        env.reset()
        # LOWER_THRESHOLD_BIG twice would reach 30; one big = 40-5=35 ≤ 30? no.
        # Apply big-lower threshold (action 11): 40 - 5 = 35; still > 30.
        # Apply again: 35 - 5 = 30 → score(30) >= threshold(30) → SUSPICIOUS.
        env._apply_action(11)
        env._apply_action(11)
        predicted = env._score_case(env._current_state)
        assert predicted == "SUSPICIOUS"


class TestApplyActionBounds:
    def test_all_actions_keep_weights_in_bounds(self, tmp_path):
        env = _env_with([_episode()], tmp_path)
        b = WeightBounds()
        for action in range(NUM_ACTIONS):
            env.reset()
            for _ in range(50):  # hammer the action
                env._apply_action(action)
            w = env._weights
            assert b.threshold[0] <= w.threshold <= b.threshold[1]
            assert b.pep_weight[0] <= w.pep <= b.pep_weight[1]
            assert b.high_flag_weight[0] <= w.high_flag <= b.high_flag_weight[1]
            assert b.med_flag_weight[0] <= w.med_flag <= b.med_flag_weight[1]
            assert b.low_flag_weight[0] <= w.low_flag <= b.low_flag_weight[1]
            assert b.turnover_weight[0] <= w.turnover <= b.turnover_weight[1]
            assert b.worldcheck_weight[0] <= w.worldcheck <= b.worldcheck_weight[1]

    def test_big_actions_double_step(self, tmp_path):
        env = _env_with([_episode()], tmp_path)
        env.reset()
        before = env._weights.pep
        env._apply_action(13)  # BOOST_PEP_BIG = +2*step
        assert env._weights.pep == before + 2 * env.config.weight_step


class TestPairedEvaluationReproducibility:
    def test_reseed_reproduces_episode_sequence(self, tmp_path):
        eps = [_episode(pep_status="PEP" if i % 2 else "CLEAR") for i in range(20)]
        env = _env_with(eps, tmp_path)

        env._rng = np.random.default_rng(env.config.seed)
        seq_a = [env.reset()[1]["episode_idx"] for _ in range(20)]

        env._rng = np.random.default_rng(env.config.seed)
        seq_b = [env.reset()[1]["episode_idx"] for _ in range(20)]

        assert seq_a == seq_b


def test_action_names_match_space_size(tmp_path):
    env = _env_with([_episode()], tmp_path)
    assert env.action_space.n == NUM_ACTIONS == len(ACTION_NAMES)
