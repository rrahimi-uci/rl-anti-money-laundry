"""Evaluate a trained PPO model against a held-out episode set.

Usage:
    python -m training.evaluate --model models/aml_ppo --episodes data/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO

from aml_rl.config import ACTION_NAMES, DefaultWeights, RLConfig
from aml_rl.env import AMLScoringEnv


def evaluate(
    model_path: str,
    episodes_path: str,
    n_eval: int = 200,
) -> Dict[str, float]:
    """Run the trained policy on *n_eval* episodes and report metrics."""

    config = RLConfig()
    env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    model = PPO.load(model_path)

    correct = 0
    total = 0
    rewards: List[float] = []
    action_counts: Counter = Counter()
    confusion: Counter = Counter()  # (predicted, ground_truth)

    for _ in range(n_eval):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, step_info = env.step(int(action))

        total += 1
        correct += int(step_info["correct"])
        rewards.append(reward)
        action_counts[step_info["action_name"]] += 1
        confusion[(step_info["predicted"], step_info["ground_truth"])] += 1

    accuracy = correct / total if total else 0
    avg_reward = float(np.mean(rewards))

    print(f"\n{'='*60}")
    print(f"Evaluation: {n_eval} episodes")
    print(f"{'='*60}")
    print(f"  Accuracy:    {accuracy:.2%} ({correct}/{total})")
    print(f"  Avg Reward:  {avg_reward:+.3f}")
    print(f"\n  Action distribution:")
    for action_name in ACTION_NAMES:
        count = action_counts.get(action_name, 0)
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {action_name:<20s} {count:>4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Confusion matrix (predicted → ground_truth):")
    for (pred, gt), count in sorted(confusion.items()):
        status = "✓" if pred == gt else "✗"
        print(f"    {status} {pred:<20s} → {gt:<20s}  {count:>3d}")

    # ── Baseline comparison ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Baseline (fixed weights, no RL):")
    baseline_correct = 0
    for _ in range(n_eval):
        obs, info = env.reset()
        # Action 10 = NO_CHANGE (use default weights)
        _, _, _, _, step_info = env.step(10)
        baseline_correct += int(step_info["correct"])
    baseline_acc = baseline_correct / n_eval if n_eval else 0
    print(f"  Accuracy:  {baseline_acc:.2%}")
    improvement = accuracy - baseline_acc
    print(f"  RL Δ:      {improvement:+.2%}")
    print(f"{'='*60}\n")

    return {
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "baseline_accuracy": baseline_acc,
        "improvement": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained AML RL policy")
    parser.add_argument(
        "--model", type=str, default="models/aml_ppo", help="Model path"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/episodes.jsonl", help="Episodes JSONL"
    )
    parser.add_argument("--n-eval", type=int, default=200, help="Episodes to evaluate")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.n_eval)


if __name__ == "__main__":
    main()
