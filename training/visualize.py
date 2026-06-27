"""Visualize training results — reward curves and weight evolution.

Usage:
    python -m training.visualize --log-dir logs --out plots/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from aml_rl.config import ACTION_NAMES, RLConfig
from aml_rl.env import AMLScoringEnv


def plot_training_curves(episodes_path: str, model_path: str, out_dir: str) -> None:
    """Run the trained model through all episodes and plot metrics."""

    config = RLConfig()
    env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    model = PPO.load(model_path)

    rewards = []
    accuracies = []
    actions_taken = []
    weight_history = []

    n_episodes = len(env.episodes)
    window = 50

    for i in range(min(n_episodes, 1000)):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, step_info = env.step(int(action))

        rewards.append(reward)
        accuracies.append(float(step_info["correct"]))
        actions_taken.append(int(action))
        weight_history.append(step_info["weights"].copy())

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Rolling accuracy ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_acc = [
        np.mean(accuracies[max(0, i - window) : i + 1])
        for i in range(len(accuracies))
    ]
    ax.plot(rolling_acc, color="#3b82f6", linewidth=1.5, label=f"Rolling {window}")
    ax.axhline(y=np.mean(accuracies), color="#ef4444", linestyle="--", label="Overall")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Accuracy (rolling window)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    # ── 2. Reward distribution ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=20, color="#6366f1", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(rewards), color="#ef4444", linestyle="--", label=f"Mean: {np.mean(rewards):+.2f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "reward_distribution.png", dpi=150)
    plt.close(fig)

    # ── 3. Action frequency ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    action_labels = [ACTION_NAMES[a] for a in range(len(ACTION_NAMES))]
    counts = [actions_taken.count(a) for a in range(len(ACTION_NAMES))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(ACTION_NAMES)))
    bars = ax.barh(action_labels, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title("Action Selection Frequency")
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    str(c), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "action_frequency.png", dpi=150)
    plt.close(fig)

    # ── 4. Weight evolution ──────────────────────────────────────────────
    if weight_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        keys = list(weight_history[0].keys())
        for key in keys:
            values = [w[key] for w in weight_history]
            ax.plot(values, label=key, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Weight Value")
        ax.set_title("Weight Values per Episode (after action applied)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "weight_evolution.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {out}/")
    print(f"  - accuracy_curve.png")
    print(f"  - reward_distribution.png")
    print(f"  - action_frequency.png")
    print(f"  - weight_evolution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AML RL training results")
    parser.add_argument(
        "--episodes", type=str, default="data/episodes.jsonl", help="Episodes JSONL"
    )
    parser.add_argument(
        "--model", type=str, default="models/aml_ppo", help="Trained model path"
    )
    parser.add_argument("--out", type=str, default="plots", help="Output directory")
    args = parser.parse_args()
    plot_training_curves(args.episodes, args.model, args.out)


if __name__ == "__main__":
    main()
