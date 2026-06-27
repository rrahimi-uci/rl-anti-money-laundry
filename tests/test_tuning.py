"""Tests for the hyperparameter tuning utilities (training.tuning).

These cover the pure search-space machinery — parameter sampling, grid/random
config generation, and the TPE-inspired Bayesian optimizer — without launching
any actual (expensive) RL training runs.
"""

from __future__ import annotations

import random

import pytest

from training.tuning import (
    BayesianOptimizer,
    ParamRange,
    TuningJob,
    _default_max_parallel,
    generate_grid_configs,
    generate_random_configs,
)


class TestParamRange:
    def test_sample_random_within_bounds(self):
        pr = ParamRange("learning_rate", 1e-5, 1e-2, log_scale=True)
        rng = random.Random(0)
        for _ in range(100):
            val = pr.sample_random(rng)
            assert 1e-5 <= val <= 1e-2

    def test_sample_random_int_dtype(self):
        pr = ParamRange("n_steps", 64, 512, dtype="int")
        rng = random.Random(0)
        val = pr.sample_random(rng)
        assert isinstance(val, int)
        assert 64 <= val <= 512

    def test_grid_values_linear(self):
        pr = ParamRange("ent_coef", 0.0, 0.1)
        vals = pr.grid_values(3)
        assert vals == [0.0, 0.05, 0.1]

    def test_grid_values_log_scale(self):
        pr = ParamRange("learning_rate", 1e-4, 1e-2, log_scale=True)
        vals = pr.grid_values(3)
        assert vals[0] == pytest.approx(1e-4)
        assert vals[-1] == pytest.approx(1e-2)
        # monotonically increasing
        assert vals == sorted(vals)

    def test_grid_values_int_dedup(self):
        # A tiny integer range with many points must not produce duplicates.
        pr = ParamRange("n_epochs", 1, 3, dtype="int")
        vals = pr.grid_values(10)
        assert vals == [1, 2, 3]
        assert len(vals) == len(set(vals))


class TestConfigGeneration:
    def test_grid_is_cartesian_product(self):
        ranges = [
            ParamRange("a", 0.0, 1.0),
            ParamRange("b", 0.0, 1.0),
        ]
        configs = generate_grid_configs(ranges, points_per_param=3)
        assert len(configs) == 9  # 3 x 3
        assert all(set(c.keys()) == {"a", "b"} for c in configs)

    def test_random_count_and_keys(self):
        ranges = [ParamRange("a", 0.0, 1.0), ParamRange("b", 1, 10, dtype="int")]
        configs = generate_random_configs(ranges, n_trials=20, seed=1)
        assert len(configs) == 20
        assert all(set(c.keys()) == {"a", "b"} for c in configs)

    def test_random_is_seed_deterministic(self):
        ranges = [ParamRange("a", 0.0, 1.0)]
        c1 = generate_random_configs(ranges, n_trials=10, seed=7)
        c2 = generate_random_configs(ranges, n_trials=10, seed=7)
        assert c1 == c2


class TestBayesianOptimizer:
    def test_cold_start_is_random(self):
        ranges = [ParamRange("a", 0.0, 1.0)]
        opt = BayesianOptimizer(ranges, n_trials=20, seed=3)
        # With <5 observations, suggestions are plain random samples in-bounds.
        for _ in range(4):
            s = opt.suggest()
            assert 0.0 <= s["a"] <= 1.0
            opt.observe(s, score=0.5)

    def test_suggestions_stay_in_bounds_after_warmup(self):
        ranges = [
            ParamRange("a", 0.0, 1.0),
            ParamRange("lr", 1e-4, 1e-1, log_scale=True),
        ]
        opt = BayesianOptimizer(ranges, n_trials=40, seed=11)
        # Feed observations where higher 'a' is better, then keep sampling.
        for _ in range(30):
            s = opt.suggest()
            assert 0.0 <= s["a"] <= 1.0
            assert 1e-4 <= s["lr"] <= 1e-1
            opt.observe(s, score=s["a"])  # reward proportional to 'a'

    def test_log_likelihood_handles_empty_group(self):
        ranges = [ParamRange("a", 0.0, 1.0)]
        opt = BayesianOptimizer(ranges, n_trials=10)
        assert opt._log_likelihood({"a": 0.5}, []) == 0.0


class TestTuningJob:
    def test_to_dict_roundtrips_core_fields(self):
        job = TuningJob(
            job_id="job-1",
            strategy="random",
            algorithm="PPO",
            param_space=[{"name": "a", "low": 0.0, "high": 1.0}],
            total_trials=5,
        )
        d = job.to_dict()
        assert d["job_id"] == "job-1"
        assert d["strategy"] == "random"
        assert d["total_trials"] == 5
        assert d["status"] == "pending"


def test_default_max_parallel_in_range():
    mp = _default_max_parallel()
    assert 1 <= mp <= 4
