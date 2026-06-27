"""FastAPI server for RL Training Dashboard."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Ensure project root on path
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aml_rl.config import (
    ACTION_NAMES,
    FEATURE_DIM,
    NUM_ACTIONS,
    DefaultWeights,
    RLConfig,
)

# ── Logging ──────────────────────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z",
            "severity": record.levelname,
            "message": record.getMessage(),
            "error": None,
            "stack_trace": None,
        }
        if record.exc_info and record.exc_info[1]:
            import traceback

            entry["error"] = type(record.exc_info[1]).__name__
            entry["stack_trace"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).replace("\n", "|")
        return json.dumps(entry, ensure_ascii=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="RL Training Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:4000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Training state ───────────────────────────────────────────────────────────

_training_state: Dict[str, Any] = {
    "status": "idle",  # idle | training | completed | error
    "progress": 0,
    "total_timesteps": 0,
    "current_step": 0,
    "episode_count": 0,
    "metrics": {
        "accuracy_history": [],  # [{step, accuracy, avg_reward, episode_count}]
        "final_accuracy": None,
        "final_avg_reward": None,
    },
    "started_at": None,
    "completed_at": None,
    "error": None,
    "config": None,
}
_training_lock = threading.Lock()
_training_thread: Optional[threading.Thread] = None


# ── Models ───────────────────────────────────────────────────────────────────

ALGORITHMS = ["PPO", "A2C", "DQN"]


class TrainRequest(BaseModel):
    algorithm: str = "PPO"
    timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    weight_step: float = 2.5
    seed: int = 42


class EvalRequest(BaseModel):
    algorithm: str = "PPO"
    n_eval: int = 200


# ── Background training ─────────────────────────────────────────────────────


def _run_training(req: TrainRequest) -> None:
    """Run RL training in a background thread."""
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from aml_rl.env import AMLScoringEnv

    algo_name = req.algorithm.upper()
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    ALGO_MAP = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgoClass = ALGO_MAP[algo_name]

    episodes_path = str(ROOT / "data" / "episodes.jsonl")
    model_out = str(ROOT / "models" / f"aml_{algo_name.lower()}")

    class DashboardCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
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

            with _training_lock:
                _training_state["current_step"] = self.num_timesteps
                _training_state["progress"] = round(
                    self.num_timesteps / _training_state["total_timesteps"] * 100, 1
                )
                _training_state["episode_count"] = len(self.episode_correct)

                if self.num_timesteps % 500 == 0 and self.episode_correct:
                    recent_correct = self.episode_correct[-self.window :]
                    recent_rewards = self.episode_rewards[-self.window :]
                    acc = sum(recent_correct) / len(recent_correct)
                    avg_r = sum(recent_rewards) / max(1, len(recent_rewards))
                    _training_state["metrics"]["accuracy_history"].append(
                        {
                            "step": self.num_timesteps,
                            "accuracy": round(acc, 4),
                            "avg_reward": round(avg_r, 4),
                            "episode_count": len(self.episode_correct),
                        }
                    )

            return True

    try:
        config = RLConfig(
            total_timesteps=req.timesteps,
            learning_rate=req.learning_rate,
            n_steps=req.n_steps,
            batch_size=req.batch_size,
            n_epochs=req.n_epochs,
            ent_coef=req.ent_coef,
            weight_step=req.weight_step,
            seed=req.seed,
        )

        env = AMLScoringEnv(episodes_path=episodes_path, config=config)
        env = Monitor(env)

        # Build algorithm-specific kwargs
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

        callback = DashboardCallback()
        model.learn(total_timesteps=req.timesteps, callback=callback)

        # Save model
        model_path = pathlib.Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        # Final metrics
        final_acc = 0.0
        final_reward = 0.0
        if callback.episode_correct:
            last_n = callback.episode_correct[-200:]
            final_acc = sum(last_n) / len(last_n)
        if callback.episode_rewards:
            last_n_r = callback.episode_rewards[-200:]
            final_reward = sum(last_n_r) / len(last_n_r)

        with _training_lock:
            _training_state["status"] = "completed"
            _training_state["progress"] = 100
            _training_state["completed_at"] = datetime.now(timezone.utc).isoformat()
            _training_state["metrics"]["final_accuracy"] = round(final_acc, 4)
            _training_state["metrics"]["final_avg_reward"] = round(final_reward, 4)

        logger.info(
            f"Training completed ({algo_name}): accuracy={final_acc:.2%} avg_reward={final_reward:+.3f}"
        )

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        with _training_lock:
            _training_state["status"] = "error"
            _training_state["error"] = str(e)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    """Return current RL configuration info."""
    return {
        "feature_dim": FEATURE_DIM,
        "num_actions": NUM_ACTIONS,
        "action_names": ACTION_NAMES,
        "algorithms": ALGORITHMS,
        "default_weights": DefaultWeights().to_dict(),
        "default_config": {
            "algorithm": "PPO",
            "timesteps": 50_000,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 10,
            "ent_coef": 0.01,
            "weight_step": 2.5,
            "seed": 42,
        },
    }


@app.get("/api/data/stats")
def data_stats():
    """Return training data statistics."""
    episodes_path = ROOT / "data" / "episodes.jsonl"
    if not episodes_path.exists():
        return {"total_episodes": 0, "scenarios": {}}

    scenarios: Dict[str, int] = {}
    verdicts: Dict[str, int] = {}
    total = 0
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            total += 1
            key = ep.get("scenario_key", "unknown")
            scenarios[key] = scenarios.get(key, 0) + 1
            gt = ep.get("ground_truth", "unknown")
            verdicts[gt] = verdicts.get(gt, 0) + 1

    return {
        "total_episodes": total,
        "scenarios": scenarios,
        "verdicts": verdicts,
        "file": str(episodes_path),
    }


# ── Helpers for episodes file ────────────────────────────────────────────────

EPISODES_PATH = ROOT / "data" / "episodes.jsonl"


def _read_all_episodes() -> List[Dict[str, Any]]:
    if not EPISODES_PATH.exists():
        return []
    with open(EPISODES_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_all_episodes(episodes: List[Dict[str, Any]]) -> None:
    EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EPISODES_PATH, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


# ── Data CRUD endpoints ─────────────────────────────────────────────────────


@app.get("/api/data/episodes")
def list_episodes(page: int = Query(1, ge=1), limit: int = Query(50, ge=1, le=500)):
    """Return paginated episode list with summary fields."""
    all_eps = _read_all_episodes()
    total = len(all_eps)
    start = (page - 1) * limit
    end = start + limit
    items = []
    for i, ep in enumerate(all_eps[start:end], start=start):
        state = ep.get("state", {})
        items.append(
            {
                "index": i,
                "alert_id": state.get("alert_id", ""),
                "customer_cis": state.get("customer_cis", ""),
                "lob": state.get("lob", ""),
                "reason_code": state.get("reason_code", ""),
                "scenario_key": ep.get("scenario_key", ""),
                "ground_truth": ep.get("ground_truth", ""),
                "risk_rating": state.get("customer_profile", {}).get("risk_rating", ""),
                "turnover_12m": state.get("turnover_12m", 0),
                "red_flag_count": len(state.get("red_flags", [])),
            }
        )
    return {"total": total, "page": page, "limit": limit, "episodes": items}


@app.get("/api/data/episodes/{index}")
def get_episode(index: int):
    """Return full episode JSON by index."""
    all_eps = _read_all_episodes()
    if index < 0 or index >= len(all_eps):
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"index": index, "episode": all_eps[index]}


@app.put("/api/data/episodes/{index}")
def update_episode(index: int, body: Dict[str, Any]):
    """Update a single episode by index. Body should be the full episode object."""
    all_eps = _read_all_episodes()
    if index < 0 or index >= len(all_eps):
        raise HTTPException(status_code=404, detail="Episode not found")
    # Validate required fields
    if "state" not in body or "ground_truth" not in body:
        raise HTTPException(
            status_code=400, detail="Episode must have 'state' and 'ground_truth'"
        )
    all_eps[index] = body
    _write_all_episodes(all_eps)
    logger.info(f"Episode {index} updated")
    return {"status": "ok", "index": index}


@app.delete("/api/data/episodes/{index}")
def delete_episode(index: int):
    """Delete a single episode by index."""
    all_eps = _read_all_episodes()
    if index < 0 or index >= len(all_eps):
        raise HTTPException(status_code=404, detail="Episode not found")
    all_eps.pop(index)
    _write_all_episodes(all_eps)
    logger.info(f"Episode {index} deleted, {len(all_eps)} remaining")
    return {"status": "ok", "remaining": len(all_eps)}


@app.post("/api/data/episodes")
def add_episode(body: Dict[str, Any]):
    """Append a new episode."""
    if "state" not in body or "ground_truth" not in body:
        raise HTTPException(
            status_code=400, detail="Episode must have 'state' and 'ground_truth'"
        )
    all_eps = _read_all_episodes()
    all_eps.append(body)
    _write_all_episodes(all_eps)
    logger.info(f"Episode added, total={len(all_eps)}")
    return {"status": "ok", "index": len(all_eps) - 1, "total": len(all_eps)}


@app.get("/api/data/download")
def download_episodes():
    """Download the full episodes.jsonl file."""
    if not EPISODES_PATH.exists():
        raise HTTPException(status_code=404, detail="No episodes file")
    return FileResponse(
        str(EPISODES_PATH),
        media_type="application/jsonl",
        filename="episodes.jsonl",
    )


@app.post("/api/data/upload")
async def upload_episodes(
    file: UploadFile = File(...),
    mode: str = Query("replace", pattern="^(replace|append)$"),
):
    """Upload episodes.jsonl file. mode=replace overwrites, mode=append adds to existing."""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    new_episodes: List[Dict[str, Any]] = []
    for i, line in enumerate(text.strip().split("\n")):
        if not line.strip():
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Invalid JSON on line {i + 1}")
        if "state" not in ep or "ground_truth" not in ep:
            raise HTTPException(
                status_code=400,
                detail=f"Line {i + 1} missing 'state' or 'ground_truth'",
            )
        new_episodes.append(ep)

    if not new_episodes:
        raise HTTPException(status_code=400, detail="No valid episodes found in file")

    if mode == "append":
        existing = _read_all_episodes()
        existing.extend(new_episodes)
        _write_all_episodes(existing)
        total = len(existing)
    else:
        _write_all_episodes(new_episodes)
        total = len(new_episodes)

    logger.info(f"Episodes uploaded: mode={mode} new={len(new_episodes)} total={total}")
    return {"status": "ok", "uploaded": len(new_episodes), "total": total, "mode": mode}


@app.get("/api/model/status")
def model_status():
    """Check which trained models exist and return info."""
    models_dir = ROOT / "models"
    found = []
    for algo in ALGORITHMS:
        p = models_dir / f"aml_{algo.lower()}.zip"
        if p.exists():
            stat = p.stat()
            found.append(
                {
                    "algorithm": algo,
                    "path": str(p),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "modified_ts": stat.st_mtime,
                }
            )
    found.sort(key=lambda item: item["modified_ts"], reverse=True)
    for item in found:
        item.pop("modified_ts", None)
    # Legacy check
    legacy = models_dir / "aml_ppo.zip"
    return {
        "exists": len(found) > 0,
        "models": found,
        "path": str(found[0]["path"]) if found else str(legacy),
        "size_kb": found[0]["size_kb"] if found else None,
        "modified": found[0]["modified"] if found else None,
    }


@app.post("/api/train/start")
def start_training(req: TrainRequest):
    """Start PPO training in background."""
    global _training_thread
    with _training_lock:
        if _training_state["status"] == "training":
            raise HTTPException(status_code=409, detail="Training already in progress")

        _training_state.update(
            {
                "status": "training",
                "progress": 0,
                "total_timesteps": req.timesteps,
                "current_step": 0,
                "episode_count": 0,
                "metrics": {
                    "accuracy_history": [],
                    "final_accuracy": None,
                    "final_avg_reward": None,
                },
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
                "config": req.dict(),
            }
        )

    _training_thread = threading.Thread(target=_run_training, args=(req,), daemon=True)
    _training_thread.start()
    logger.info(f"Training started: {req.algorithm} {req.timesteps} timesteps")
    return {"status": "started", "algorithm": req.algorithm, "timesteps": req.timesteps}


@app.get("/api/train/status")
def training_status():
    """Get current training status and metrics."""
    with _training_lock:
        return {**_training_state}


@app.post("/api/evaluate")
def run_evaluation(req: EvalRequest):
    """Run evaluation of trained model."""
    from stable_baselines3 import PPO, A2C, DQN
    from aml_rl.env import AMLScoringEnv
    import numpy as np

    algo_name = req.algorithm.upper()
    ALGO_MAP = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    if algo_name not in ALGO_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo_name}")

    model_file = ROOT / "models" / f"aml_{algo_name.lower()}.zip"
    if not model_file.exists():
        raise HTTPException(
            status_code=404, detail=f"No trained {algo_name} model found"
        )

    episodes_path = str(ROOT / "data" / "episodes.jsonl")
    config = RLConfig()
    env = AMLScoringEnv(episodes_path=episodes_path, config=config)
    model = ALGO_MAP[algo_name].load(str(model_file.with_suffix("")))

    correct = 0
    total = 0
    rewards: List[float] = []
    action_counts: Dict[str, int] = {}
    confusion: List[Dict[str, Any]] = []
    per_episode: List[Dict[str, Any]] = []

    for _ in range(req.n_eval):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, step_info = env.step(int(action))

        total += 1
        is_correct = bool(step_info["correct"])
        correct += int(is_correct)
        rewards.append(float(reward))
        act_name = step_info["action_name"]
        action_counts[act_name] = action_counts.get(act_name, 0) + 1

        per_episode.append(
            {
                "episode": total,
                "action": act_name,
                "predicted": step_info["predicted"],
                "ground_truth": step_info["ground_truth"],
                "reward": float(reward),
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total else 0

    # Baseline
    baseline_correct = 0
    for _ in range(req.n_eval):
        obs, info = env.reset()
        _, _, _, _, step_info = env.step(10)  # NO_CHANGE
        baseline_correct += int(step_info["correct"])
    baseline_acc = baseline_correct / req.n_eval if req.n_eval else 0

    # Action distribution for chart
    action_dist = []
    for name in ACTION_NAMES:
        count = action_counts.get(name, 0)
        action_dist.append(
            {
                "action": name,
                "count": count,
                "pct": round(count / total * 100, 1) if total else 0,
            }
        )

    return {
        "accuracy": round(accuracy, 4),
        "avg_reward": round(float(sum(rewards) / len(rewards)), 4) if rewards else 0,
        "baseline_accuracy": round(baseline_acc, 4),
        "improvement": round(accuracy - baseline_acc, 4),
        "total_episodes": total,
        "correct": correct,
        "action_distribution": action_dist,
        "reward_distribution": {
            "values": [round(r, 3) for r in rewards],
            "mean": round(float(sum(rewards) / len(rewards)), 3) if rewards else 0,
        },
        "per_episode": per_episode[:50],  # first 50 for detail view
    }


@app.get("/api/plots/{name}")
def get_plot(name: str):
    """Serve pre-generated plot images."""
    valid = [
        "accuracy_curve",
        "reward_distribution",
        "action_frequency",
        "weight_evolution",
    ]
    if name not in valid:
        raise HTTPException(status_code=404, detail="Unknown plot")
    path = ROOT / "plots" / f"{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not generated yet")
    return FileResponse(str(path), media_type="image/png")


@app.post("/api/plots/generate")
def generate_plots():
    """Re-generate all visualization plots."""
    from training.visualize import plot_training_curves

    model_path = ROOT / "models" / "aml_ppo"
    if not (ROOT / "models" / "aml_ppo.zip").exists():
        raise HTTPException(status_code=404, detail="No trained model found")

    episodes_path = str(ROOT / "data" / "episodes.jsonl")
    out_dir = str(ROOT / "plots")
    plot_training_curves(episodes_path, str(model_path), out_dir)
    return {"status": "ok", "plots_dir": out_dir}


@app.get("/api/feedback/stats")
def feedback_stats():
    """Return live HITL feedback statistics."""
    feedback_path = ROOT / "data" / "hitl_feedback.jsonl"
    if not feedback_path.exists():
        return {"total": 0, "entries": []}

    entries = []
    with open(feedback_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    return {
        "total": len(entries),
        "entries": entries[-20:],  # last 20
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)
