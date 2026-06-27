"""Adaptive decision node — drop-in replacement for the fixed-weight
``decision_node()`` in the AML LangGraph pipeline.

Loads a trained PPO model and uses it to adjust scoring weights per-case
before applying the same scoring logic as the original node.

Toggle via env var: ``USE_RL_POLICY=true`` (default: false → uses fixed weights).

Usage in graph.py:
    from integration.adaptive_decision import adaptive_decision_node
    # Replace:  graph.add_node("decision_node", decision_node)
    # With:     graph.add_node("decision_node", adaptive_decision_node)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict

# Ensure aml_rl is importable
RL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_ROOT))

from aml_rl.config import ACTION_NAMES, DefaultWeights, WeightBounds
from aml_rl.feature_extractor import extract_features

# Lazy-load model to avoid import cost when RL is disabled
_MODEL = None
_MODEL_PATH = str(RL_ROOT / "models" / "aml_ppo")


def _load_model():
    global _MODEL
    if _MODEL is None:
        from stable_baselines3 import PPO

        _MODEL = PPO.load(_MODEL_PATH)
    return _MODEL


def _use_rl() -> bool:
    return os.environ.get("USE_RL_POLICY", "false").lower() in ("true", "1", "yes")


def _make_message(content: str):
    """Build a LangChain ``HumanMessage`` when available, else a light stand-in.

    The node is designed as a drop-in for a LangGraph pipeline, where messages
    are ``langchain_core`` objects.  When ``langchain_core`` is not installed
    (e.g. running this repo stand-alone), we fall back to a minimal object that
    exposes the same ``content`` attribute so the node stays importable and
    testable without the heavy dependency.
    """
    try:
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)
    except Exception:  # pragma: no cover - depends on optional dependency
        from types import SimpleNamespace

        return SimpleNamespace(type="human", content=content)


def adaptive_decision_node(state: dict) -> dict:
    """Drop-in replacement for ``decision_node`` with RL-adaptive weights.

    When ``USE_RL_POLICY=true``, loads the trained PPO model, predicts the
    best weight adjustment action for this case, and applies it before scoring.
    Otherwise falls back to the fixed ``DefaultWeights``.
    """

    red_flags = state.get("red_flags", [])
    pep = state.get("pep_status", "CLEAR")
    care = state.get("care_marker_detected", False)
    turnover = float(state.get("turnover_12m", 0))
    wc = state.get("worldcheck_result", {})
    customer = state.get("customer_profile", {})

    high_severity = [rf for rf in red_flags if rf.get("severity") == "HIGH"]
    med_severity = [rf for rf in red_flags if rf.get("severity") == "MEDIUM"]
    low_severity = [rf for rf in red_flags if rf.get("severity") == "LOW"]

    # ── Determine weights ────────────────────────────────────────────────
    weights = DefaultWeights()
    action_taken = "FIXED_WEIGHTS"

    if _use_rl():
        try:
            model = _load_model()
            obs = extract_features(state)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_taken = ACTION_NAMES[action]

            # Apply action
            step = 2.5
            b = WeightBounds()
            if action == 0:
                weights.threshold = max(b.threshold[0], weights.threshold - step)
            elif action == 1:
                weights.threshold = min(b.threshold[1], weights.threshold + step)
            elif action == 2:
                weights.pep = min(b.pep_weight[1], weights.pep + step)
            elif action == 3:
                weights.pep = max(b.pep_weight[0], weights.pep - step)
            elif action == 4:
                weights.high_flag = min(b.high_flag_weight[1], weights.high_flag + step)
            elif action == 5:
                weights.high_flag = max(b.high_flag_weight[0], weights.high_flag - step)
            elif action == 6:
                weights.med_flag = min(b.med_flag_weight[1], weights.med_flag + step)
            elif action == 7:
                weights.med_flag = max(b.med_flag_weight[0], weights.med_flag - step)
            elif action == 8:
                weights.turnover = min(b.turnover_weight[1], weights.turnover + step)
            elif action == 9:
                weights.worldcheck = min(b.worldcheck_weight[1], weights.worldcheck + step)
            # 10 = NO_CHANGE
            elif action == 11:
                weights.threshold = max(b.threshold[0], weights.threshold - 2 * step)
            elif action == 12:
                weights.threshold = min(b.threshold[1], weights.threshold + 2 * step)
            elif action == 13:
                weights.pep = min(b.pep_weight[1], weights.pep + 2 * step)
            elif action == 14:
                weights.pep = max(b.pep_weight[0], weights.pep - 2 * step)
            elif action == 15:
                weights.high_flag = min(b.high_flag_weight[1], weights.high_flag + 2 * step)
            elif action == 16:
                weights.high_flag = max(b.high_flag_weight[0], weights.high_flag - 2 * step)
            elif action == 17:
                weights.low_flag = max(b.low_flag_weight[0], weights.low_flag - step)
            elif action == 18:
                weights.low_flag = min(b.low_flag_weight[1], weights.low_flag + step)
        except Exception:
            action_taken = "FALLBACK_FIXED"

    # ── Care marker override ─────────────────────────────────────────────
    if care:
        outcome = "ESCALATED_FIU"
        trigger_rule = "Care marker override — immediate FIU escalation."
        risk_score = 0
    else:
        # ── Weighted scoring ─────────────────────────────────────────────
        risk_score = 0.0
        if pep in ("PEP", "SANCTIONED"):
            risk_score += weights.pep
        for rf in red_flags:
            sev = rf.get("severity", "LOW")
            if sev == "HIGH":
                risk_score += weights.high_flag
            elif sev == "MEDIUM":
                risk_score += weights.med_flag
            else:
                risk_score += weights.low_flag
        if turnover > 500_000:
            risk_score += weights.turnover
        adverse = wc.get("adverse_media", wc.get("adverse_media_hits", 0))
        if isinstance(adverse, list):
            adverse = len(adverse)
        risk_score += min(adverse, 3) * weights.worldcheck
        risk_score = int(risk_score)

        if risk_score >= weights.threshold:
            outcome = "SUSPICIOUS"
            trigger_rule = f"Risk score {risk_score} ≥ threshold {weights.threshold} → SUSPICIOUS."
        else:
            outcome = "NON_SUSPICIOUS"
            trigger_rule = f"Risk score {risk_score} < threshold {weights.threshold} → NON_SUSPICIOUS."

    # ── Risk level ───────────────────────────────────────────────────────
    if care:
        risk_level = "CRITICAL"
    elif risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ── Build output (matching original decision_node contract) ──────────
    factors = []
    evidence_lines = [
        f"Customer: {customer.get('name', state.get('customer_cis', 'N/A'))}",
        f"PEP Status: {pep}",
        f"Care Marker: {'Yes' if care else 'No'}",
        f"Red Flags: {len(high_severity)} HIGH, {len(med_severity)} MEDIUM, {len(low_severity)} LOW",
        f"12-Month Turnover: £{turnover:,.2f}",
        f"RL Action: {action_taken}",
        f"Weights: {weights.to_dict()}",
    ]

    decision_details = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "threshold": weights.threshold,
        "trigger_rule": trigger_rule,
        "factors": factors,
        "evidence_summary": evidence_lines,
        "rl_action": action_taken,
        "rl_weights": weights.to_dict(),
    }

    rationale = f"Risk Score: {risk_score}/100 ({risk_level}). {trigger_rule} [RL: {action_taken}]"

    log_msg = (
        f"Decision: {outcome} | Score: {risk_score} ({risk_level}) | "
        f"RL: {action_taken} | Weights: {weights.to_dict()}"
    )

    return {
        "outcome": outcome,
        "decision_rationale": rationale,
        "decision_details": decision_details,
        "current_stage": "decision_node",
        "stage_history": state.get("stage_history", []) + ["decision_node"],
        "messages": [_make_message(log_msg)],
    }
