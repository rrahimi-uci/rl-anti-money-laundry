"""Hyperparameter tuning strategies: Grid Search, Random Search, Bayesian Optimization."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Parameter space definitions ──────────────────────────────────────────────


@dataclass
class ParamRange:
    """Defines a single hyperparameter search range."""

    name: str
    low: float
    high: float
    log_scale: bool = False  # sample in log space (useful for learning rate)
    dtype: str = "float"  # float | int

    def sample_random(self, rng: random.Random) -> float:
        if self.log_scale:
            val = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            val = rng.uniform(self.low, self.high)
        return int(round(val)) if self.dtype == "int" else round(val, 8)

    def grid_values(self, n_points: int) -> List[float]:
        if self.log_scale:
            vals = np.logspace(math.log10(self.low), math.log10(self.high), n_points).tolist()
        else:
            vals = np.linspace(self.low, self.high, n_points).tolist()
        if self.dtype == "int":
            seen = set()
            result = []
            for v in vals:
                iv = int(round(v))
                if iv not in seen:
                    seen.add(iv)
                    result.append(iv)
            return result
        return [round(v, 8) for v in vals]


@dataclass
class TuningResult:
    """Result of evaluating one hyperparameter configuration."""

    trial_id: int
    params: Dict[str, float]
    accuracy: float
    avg_reward: float
    training_time: float  # seconds
    status: str = "completed"  # completed | error
    error: Optional[str] = None


@dataclass
class TuningJob:
    """Full tuning job state."""

    job_id: str
    strategy: str  # grid | random | bayesian
    algorithm: str
    param_space: List[Dict[str, Any]]
    total_trials: int
    completed_trials: int = 0
    current_trial: int = 0
    status: str = "pending"  # pending | running | completed | cancelled | error
    results: List[Dict[str, Any]] = field(default_factory=list)
    best_result: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0
    current_params: Optional[Dict[str, float]] = None
    estimated_remaining: Optional[float] = None
    max_parallel: int = 1
    active_workers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "strategy": self.strategy,
            "algorithm": self.algorithm,
            "param_space": self.param_space,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "current_trial": self.current_trial,
            "status": self.status,
            "results": self.results,
            "best_result": self.best_result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "progress": self.progress,
            "current_params": self.current_params,
            "estimated_remaining": self.estimated_remaining,
            "max_parallel": self.max_parallel,
            "active_workers": self.active_workers,
        }


# ── Trial runner ─────────────────────────────────────────────────────────────


def _run_single_trial(
    algorithm: str,
    params: Dict[str, float],
    episodes_path: str,
    eval_episodes: int = 200,
    timesteps: int = 20_000,
) -> TuningResult:
    """Train a model with given params and evaluate it. Returns TuningResult."""
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.monitor import Monitor
    from aml_rl.config import RLConfig
    from aml_rl.env import AMLScoringEnv

    ALGO_MAP = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    algo_name = algorithm.upper()
    AlgoClass = ALGO_MAP[algo_name]

    start = time.time()

    config = RLConfig(
        total_timesteps=timesteps,
        learning_rate=params.get("learning_rate", 3e-4),
        n_steps=int(params.get("n_steps", 128)),
        batch_size=int(params.get("batch_size", 64)),
        n_epochs=int(params.get("n_epochs", 10)),
        ent_coef=params.get("ent_coef", 0.01),
        weight_step=params.get("weight_step", 2.5),
        gamma=params.get("gamma", 0.99),
        seed=42,
    )

    env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    env = Monitor(env)

    common_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        verbose=0,
    )
    if algo_name == "PPO":
        common_kwargs.update(
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
        )
    elif algo_name == "A2C":
        common_kwargs.update(
            n_steps=config.n_steps,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
        )
    elif algo_name == "DQN":
        common_kwargs.update(
            batch_size=config.batch_size,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
        )

    model = AlgoClass(**common_kwargs)
    model.learn(total_timesteps=timesteps)

    # Evaluate
    correct = 0
    rewards = []
    eval_env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = eval_env.step(int(action))
        correct += int(info["correct"])
        rewards.append(float(reward))

    elapsed = time.time() - start
    accuracy = correct / eval_episodes if eval_episodes else 0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0

    return TuningResult(
        trial_id=0,
        params=params,
        accuracy=round(accuracy, 4),
        avg_reward=round(avg_reward, 4),
        training_time=round(elapsed, 1),
    )


# ── Grid Search ──────────────────────────────────────────────────────────────


def generate_grid_configs(
    param_ranges: List[ParamRange],
    points_per_param: int = 3,
) -> List[Dict[str, float]]:
    """Generate all combinations for grid search."""
    grids = {}
    for pr in param_ranges:
        grids[pr.name] = pr.grid_values(points_per_param)

    keys = list(grids.keys())
    combos = list(itertools.product(*(grids[k] for k in keys)))
    return [{k: v for k, v in zip(keys, combo)} for combo in combos]


# ── Random Search ────────────────────────────────────────────────────────────


def generate_random_configs(
    param_ranges: List[ParamRange],
    n_trials: int,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Generate random parameter configs."""
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        cfg = {pr.name: pr.sample_random(rng) for pr in param_ranges}
        configs.append(cfg)
    return configs


# ── Bayesian Optimization (TPE-inspired) ────────────────────────────────────


class BayesianOptimizer:
    """Simple Tree-structured Parzen Estimator (TPE) inspired Bayesian optimizer.

    Uses a split of observations into "good" (top quantile) and "bad" to guide
    sampling towards promising regions. Falls back to random sampling when
    insufficient data is available.
    """

    def __init__(
        self,
        param_ranges: List[ParamRange],
        n_trials: int,
        gamma: float = 0.25,  # quantile split (top 25% = "good")
        n_candidates: int = 24,
        seed: int = 42,
    ):
        self.param_ranges = param_ranges
        self.n_trials = n_trials
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.rng = random.Random(seed)
        self.observations: List[Tuple[Dict[str, float], float]] = []  # (params, score)

    def suggest(self) -> Dict[str, float]:
        """Suggest next parameter configuration."""
        # Need at least a few observations before doing informed sampling
        if len(self.observations) < 5:
            return {pr.name: pr.sample_random(self.rng) for pr in self.param_ranges}

        # Split observations into good/bad by accuracy
        sorted_obs = sorted(self.observations, key=lambda x: x[1], reverse=True)
        n_good = max(1, int(len(sorted_obs) * self.gamma))
        good = [o[0] for o in sorted_obs[:n_good]]
        bad = [o[0] for o in sorted_obs[n_good:]]

        # Generate candidates and pick the one most likely under "good" distribution
        best_candidate = None
        best_score = -float("inf")

        for _ in range(self.n_candidates):
            candidate = {}
            for pr in self.param_ranges:
                # Sample from a mixture: 75% guided by good observations, 25% random
                if self.rng.random() < 0.75 and good:
                    # Pick a random good observation and perturb it
                    base = self.rng.choice(good)[pr.name]
                    span = pr.high - pr.low
                    if pr.log_scale:
                        log_base = math.log(max(base, 1e-10))
                        log_span = math.log(pr.high) - math.log(pr.low)
                        perturbed = math.exp(
                            log_base + self.rng.gauss(0, log_span * 0.1)
                        )
                    else:
                        perturbed = base + self.rng.gauss(0, span * 0.1)
                    perturbed = max(pr.low, min(pr.high, perturbed))
                    if pr.dtype == "int":
                        perturbed = int(round(perturbed))
                    else:
                        perturbed = round(perturbed, 8)
                    candidate[pr.name] = perturbed
                else:
                    candidate[pr.name] = pr.sample_random(self.rng)

            # Score: higher density under good, lower under bad
            score = self._log_likelihood(candidate, good) - self._log_likelihood(candidate, bad)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate or {pr.name: pr.sample_random(self.rng) for pr in self.param_ranges}

    def observe(self, params: Dict[str, float], score: float) -> None:
        self.observations.append((params, score))

    def _log_likelihood(self, candidate: Dict[str, float], group: List[Dict[str, float]]) -> float:
        """Estimate log-likelihood of candidate under a group using KDE."""
        if not group:
            return 0.0
        ll = 0.0
        for pr in self.param_ranges:
            val = candidate[pr.name]
            span = pr.high - pr.low
            bw = span * 0.15  # bandwidth
            if pr.log_scale:
                val = math.log(max(val, 1e-10))
                bw = (math.log(pr.high) - math.log(pr.low)) * 0.15
            densities = []
            for obs in group:
                obs_val = obs[pr.name]
                if pr.log_scale:
                    obs_val = math.log(max(obs_val, 1e-10))
                diff = (val - obs_val) / max(bw, 1e-10)
                densities.append(math.exp(-0.5 * diff * diff))
            avg_density = sum(densities) / len(densities)
            ll += math.log(max(avg_density, 1e-10))
        return ll


# ── Top-level helper for ProcessPoolExecutor (must be picklable) ─────────────


def _run_trial_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel trial execution. Runs in a subprocess."""
    trial_id, algorithm, params, episodes_path, timesteps = args
    try:
        result = _run_single_trial(
            algorithm=algorithm,
            params=params,
            episodes_path=episodes_path,
            timesteps=timesteps,
        )
        return {
            "trial_id": trial_id,
            "params": result.params,
            "accuracy": result.accuracy,
            "avg_reward": result.avg_reward,
            "training_time": result.training_time,
            "status": result.status,
        }
    except Exception as e:
        return {
            "trial_id": trial_id,
            "params": params,
            "accuracy": 0,
            "avg_reward": 0,
            "training_time": 0,
            "status": "error",
            "error": str(e),
        }


# ── Orchestrator ─────────────────────────────────────────────────────────────


def _default_max_parallel() -> int:
    """Sensible default: half CPU count, capped between 1 and 4."""
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus // 2))


def run_tuning_job(
    job: TuningJob,
    episodes_path: str,
    on_trial_complete: Optional[Callable[[TuningJob], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> TuningJob:
    """Execute a full tuning job with parallel trial execution.

    Grid and Random search run all trials in parallel batches.
    Bayesian search runs in batches of max_parallel to feed observations back
    between batches.
    """
    param_ranges = [
        ParamRange(
            name=p["name"],
            low=p["low"],
            high=p["high"],
            log_scale=p.get("log_scale", False),
            dtype=p.get("dtype", "float"),
        )
        for p in job.param_space
    ]

    job.status = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()
    max_workers = max(1, job.max_parallel)

    trial_timesteps = 20_000

    # Generate configs based on strategy
    if job.strategy == "grid":
        n_params = len(param_ranges)
        points = max(2, int(round(job.total_trials ** (1.0 / max(n_params, 1)))))
        configs = generate_grid_configs(param_ranges, points)
        job.total_trials = len(configs)
    elif job.strategy == "random":
        configs = generate_random_configs(param_ranges, job.total_trials)
    elif job.strategy == "bayesian":
        configs = None
    else:
        job.status = "error"
        job.error = f"Unknown strategy: {job.strategy}"
        return job

    bayesian_opt = None
    if job.strategy == "bayesian":
        bayesian_opt = BayesianOptimizer(param_ranges, job.total_trials)

    trial_times: List[float] = []
    next_trial_id = 1

    def _update_job_from_result(trial_dict: Dict[str, Any]) -> None:
        job.results.append(trial_dict)
        trial_times.append(trial_dict["training_time"])
        job.completed_trials = len(job.results)
        job.progress = round(job.completed_trials / job.total_trials * 100, 1)

        if trial_dict["status"] == "completed":
            if job.best_result is None or trial_dict["accuracy"] > job.best_result["accuracy"]:
                job.best_result = trial_dict

        if trial_times:
            avg_time = sum(trial_times) / len(trial_times)
            remaining_trials = job.total_trials - job.completed_trials
            # With parallelism the wall-clock time per batch is ~avg_time
            remaining_batches = math.ceil(remaining_trials / max_workers)
            job.estimated_remaining = round(remaining_batches * avg_time, 0)

    if job.strategy == "bayesian":
        # Bayesian: run in batches of max_workers so observations feed back
        assert bayesian_opt is not None
        while next_trial_id <= job.total_trials:
            if cancel_check and cancel_check():
                job.status = "cancelled"
                break

            batch_size = min(max_workers, job.total_trials - next_trial_id + 1)
            batch_configs = [bayesian_opt.suggest() for _ in range(batch_size)]

            job.current_trial = next_trial_id
            job.current_params = batch_configs[0] if batch_configs else None
            job.active_workers = batch_size
            if on_trial_complete:
                on_trial_complete(job)

            work_items = [
                (next_trial_id + j, job.algorithm, cfg, episodes_path, trial_timesteps)
                for j, cfg in enumerate(batch_configs)
            ]

            if max_workers > 1 and batch_size > 1:
                with ProcessPoolExecutor(max_workers=min(max_workers, batch_size)) as pool:
                    futures = {pool.submit(_run_trial_worker, item): item for item in work_items}
                    for future in as_completed(futures):
                        trial_dict = future.result()
                        _update_job_from_result(trial_dict)
                        if trial_dict["status"] == "completed":
                            bayesian_opt.observe(trial_dict["params"], trial_dict["accuracy"])
                        if on_trial_complete:
                            on_trial_complete(job)
            else:
                for item in work_items:
                    trial_dict = _run_trial_worker(item)
                    _update_job_from_result(trial_dict)
                    if trial_dict["status"] == "completed":
                        bayesian_opt.observe(trial_dict["params"], trial_dict["accuracy"])
                    if on_trial_complete:
                        on_trial_complete(job)

            next_trial_id += batch_size
            job.active_workers = 0
    else:
        # Grid / Random: fully parallel
        assert configs is not None
        all_work = [
            (i + 1, job.algorithm, configs[i], episodes_path, trial_timesteps)
            for i in range(job.total_trials)
        ]

        job.current_trial = 1
        job.active_workers = min(max_workers, len(all_work))
        if on_trial_complete:
            on_trial_complete(job)

        if max_workers > 1 and len(all_work) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_trial_worker, item): item for item in all_work}
                for future in as_completed(futures):
                    if cancel_check and cancel_check():
                        job.status = "cancelled"
                        pool.shutdown(wait=False, cancel_futures=True)
                        break

                    trial_dict = future.result()
                    _update_job_from_result(trial_dict)
                    job.current_trial = job.completed_trials
                    job.current_params = trial_dict["params"]
                    job.active_workers = min(
                        max_workers, job.total_trials - job.completed_trials
                    )
                    if on_trial_complete:
                        on_trial_complete(job)
        else:
            for item in all_work:
                if cancel_check and cancel_check():
                    job.status = "cancelled"
                    break
                trial_dict = _run_trial_worker(item)
                _update_job_from_result(trial_dict)
                job.current_trial = job.completed_trials
                job.current_params = trial_dict["params"]
                if on_trial_complete:
                    on_trial_complete(job)

        job.active_workers = 0

    if job.status == "running":
        job.status = "completed"
    job.progress = 100.0
    job.completed_at = datetime.now(timezone.utc).isoformat()

    return job
