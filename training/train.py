"""Train PPO agent on AML risk-scoring environment.

Usage:
    python -m training.train --episodes data/episodes.jsonl --timesteps 50000
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from aml_rl.config import RLConfig
from aml_rl.env import AMLScoringEnv


class MetricsCallback(BaseCallback):
    """Track accuracy and reward across training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.episode_correct: list = []
        self.window = 100

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "correct" in info:
                self.episode_correct.append(float(info["correct"]))
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % 1000 == 0 and self.episode_correct:
            recent = self.episode_correct[-self.window :]
            acc = sum(recent) / len(recent)
            avg_r = (
                sum(self.episode_rewards[-self.window :])
                / max(1, len(self.episode_rewards[-self.window :]))
            )
            print(
                f"  step {self.num_timesteps:>6d} | "
                f"accuracy={acc:.2%} | "
                f"avg_reward={avg_r:+.3f} | "
                f"episodes={len(self.episode_correct)}"
            )
        return True


def train(
    episodes_path: str,
    timesteps: int = 50_000,
    model_out: str = "models/aml_ppo",
    seed: int = 42,
) -> PPO:
    """Train PPO and save the model."""

    config = RLConfig(total_timesteps=timesteps, seed=seed)
    env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        seed=config.seed,
        verbose=0,
    )

    print(f"Training PPO for {timesteps} timesteps...")
    print(f"  Episodes: {len(env.unwrapped.episodes)}")
    print(f"  Actions:  {env.unwrapped.action_space.n}")
    print(f"  Features: {env.unwrapped.observation_space.shape[0]}")
    print()

    callback = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save
    model_path = pathlib.Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Final accuracy
    if callback.episode_correct:
        final_acc = sum(callback.episode_correct[-200:]) / len(
            callback.episode_correct[-200:]
        )
        print(f"Final accuracy (last 200): {final_acc:.2%}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on AML scoring env")
    parser.add_argument(
        "--episodes",
        type=str,
        default="data/episodes.jsonl",
        help="Path to episode JSONL",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--model-out", type=str, default="models/aml_ppo", help="Output model path"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        episodes_path=args.episodes,
        timesteps=args.timesteps,
        model_out=args.model_out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
