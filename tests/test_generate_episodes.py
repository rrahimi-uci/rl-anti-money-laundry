"""Tests for synthetic episode generation (data.generate_episodes).

Validates that generation is self-contained (no external project required),
deterministic under a fixed seed, and produces states the feature extractor
and environment can consume.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from aml_rl.config import FEATURE_DIM
from aml_rl.feature_extractor import extract_features
from data.generate_episodes import (
    GROUND_TRUTH,
    SYNTHETIC_TEMPLATES,
    _build_synthetic_episode,
    generate_episodes,
)

VALID_OUTCOMES = {"SUSPICIOUS", "NON_SUSPICIOUS", "ESCALATED_FIU"}


class TestSyntheticEpisode:
    def test_build_has_required_keys(self):
        rng = random.Random(0)
        key = "shell_company_layering"
        state = _build_synthetic_episode(key, SYNTHETIC_TEMPLATES[key], rng)
        for k in ("lob", "reason_code", "pep_status", "red_flags", "turnover_12m"):
            assert k in state

    def test_build_features_extractable(self):
        rng = random.Random(1)
        for key, tmpl in SYNTHETIC_TEMPLATES.items():
            state = _build_synthetic_episode(key, tmpl, rng)
            obs = extract_features(state)
            assert obs.shape == (FEATURE_DIM,)
            assert np.isfinite(obs).all()


class TestGenerateEpisodes:
    def test_count_matches(self):
        eps = generate_episodes(count=100, seed=42)
        assert len(eps) == 100

    def test_all_outcomes_valid(self):
        eps = generate_episodes(count=200, seed=42)
        assert all(e["ground_truth"] in VALID_OUTCOMES for e in eps)

    def test_every_episode_has_state_and_label(self):
        eps = generate_episodes(count=50, seed=1)
        assert all("state" in e and "ground_truth" in e for e in eps)
        assert all("scenario_key" in e for e in eps)

    def test_seed_determinism(self):
        a = generate_episodes(count=60, seed=123)
        b = generate_episodes(count=60, seed=123)
        assert json.dumps(a) == json.dumps(b)

    def test_different_seeds_differ(self):
        a = generate_episodes(count=60, seed=1)
        b = generate_episodes(count=60, seed=2)
        assert json.dumps(a) != json.dumps(b)

    def test_covers_both_base_and_synthetic(self):
        eps = generate_episodes(count=300, seed=42)
        keys = {e["scenario_key"] for e in eps}
        # at least one base scenario and one synthetic-template scenario present
        assert keys & set(GROUND_TRUTH.keys())
        assert keys & set(SYNTHETIC_TEMPLATES.keys())

    def test_generated_states_drive_environment(self, tmp_path):
        from aml_rl.env import AMLScoringEnv

        eps = generate_episodes(count=40, seed=5)
        path = tmp_path / "eps.jsonl"
        with open(path, "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        env = AMLScoringEnv(episodes_path=path)
        obs, info = env.reset()
        assert obs.shape == (FEATURE_DIM,)
        _, reward, terminated, _, step_info = env.step(10)
        assert terminated is True
        assert np.isfinite(reward)
