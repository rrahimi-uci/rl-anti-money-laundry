"""API tests for the FastAPI backend (server.py).

Read-only endpoints are tested against the repo's real episodes file. Mutating
endpoints (add/update/delete/upload) are redirected to a temporary file via
monkeypatching so the committed dataset is never modified.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import server
from aml_rl.config import NUM_ACTIONS

client = TestClient(server.app)


# ── Read-only endpoints ───────────────────────────────────────────────────


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_shape():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["num_actions"] == NUM_ACTIONS
    assert len(body["action_names"]) == NUM_ACTIONS
    assert body["feature_dim"] == 24
    assert set(body["default_weights"]) == {
        "pep",
        "high_flag",
        "med_flag",
        "low_flag",
        "turnover",
        "worldcheck",
        "threshold",
    }
    assert "PPO" in body["algorithms"]


def test_data_stats():
    r = client.get("/api/data/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_episodes"] >= 0
    assert isinstance(body["scenarios"], dict)


def test_list_episodes_pagination():
    r = client.get("/api/data/episodes?page=1&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["page"] == 1
    assert body["limit"] == 5
    assert len(body["episodes"]) <= 5
    if body["episodes"]:
        assert body["episodes"][0]["index"] == 0


def test_list_episodes_rejects_bad_limit():
    r = client.get("/api/data/episodes?limit=99999")
    assert r.status_code == 422  # exceeds le=500


def test_get_episode_out_of_range_404():
    r = client.get("/api/data/episodes/100000000")
    assert r.status_code == 404


def test_model_status_shape():
    r = client.get("/api/model/status")
    assert r.status_code == 200
    body = r.json()
    assert "exists" in body
    assert isinstance(body["models"], list)


def test_evaluate_unknown_algorithm():
    r = client.post("/api/evaluate", json={"algorithm": "NOPE", "n_eval": 1})
    assert r.status_code == 400


def test_evaluate_missing_model_404():
    # A valid algorithm that almost certainly has no trained model in CI.
    r = client.post("/api/evaluate", json={"algorithm": "A2C", "n_eval": 1})
    assert r.status_code in (404, 200)  # 404 if absent; 200 if a model exists


def test_train_status_idle_or_known():
    r = client.get("/api/train/status")
    assert r.status_code == 200
    assert r.json()["status"] in {"idle", "training", "completed", "error"}


def test_feedback_stats():
    r = client.get("/api/feedback/stats")
    assert r.status_code == 200
    assert "total" in r.json()


# ── Mutating endpoints (redirected to a temp file) ─────────────────────────


@pytest.fixture()
def temp_episodes(tmp_path, monkeypatch):
    path = tmp_path / "episodes.jsonl"
    seed = [
        {"state": {"lob": "RETAIL", "alert_id": "A1"}, "ground_truth": "NON_SUSPICIOUS"},
        {"state": {"lob": "PB", "alert_id": "A2"}, "ground_truth": "SUSPICIOUS"},
    ]
    with open(path, "w") as f:
        for ep in seed:
            f.write(json.dumps(ep) + "\n")
    monkeypatch.setattr(server, "EPISODES_PATH", path)
    return path


def test_add_episode(temp_episodes):
    new = {"state": {"lob": "BUSINESS"}, "ground_truth": "ESCALATED_FIU"}
    r = client.post("/api/data/episodes", json=new)
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_add_episode_rejects_invalid(temp_episodes):
    r = client.post("/api/data/episodes", json={"foo": "bar"})
    assert r.status_code == 400


def test_update_episode(temp_episodes):
    updated = {"state": {"lob": "CORPORATE"}, "ground_truth": "SUSPICIOUS"}
    r = client.put("/api/data/episodes/0", json=updated)
    assert r.status_code == 200
    # confirm persisted
    r2 = client.get("/api/data/episodes/0")
    assert r2.json()["episode"]["state"]["lob"] == "CORPORATE"


def test_delete_episode(temp_episodes):
    r = client.delete("/api/data/episodes/0")
    assert r.status_code == 200
    assert r.json()["remaining"] == 1


def test_upload_replace(temp_episodes):
    payload = "\n".join(
        json.dumps({"state": {"lob": "RETAIL"}, "ground_truth": "NON_SUSPICIOUS"})
        for _ in range(4)
    )
    r = client.post(
        "/api/data/upload?mode=replace",
        files={"file": ("episodes.jsonl", payload, "application/jsonl")},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 4


def test_upload_rejects_bad_json(temp_episodes):
    r = client.post(
        "/api/data/upload?mode=replace",
        files={"file": ("bad.jsonl", "{not json}", "application/jsonl")},
    )
    assert r.status_code == 400
