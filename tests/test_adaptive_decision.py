"""Tests for the adaptive decision node (integration.adaptive_decision).

These exercise the fixed-weight path (USE_RL_POLICY disabled), which does not
require a trained model, and verify the node's output contract is compatible
with the LangGraph ``decision_node`` it replaces.
"""

from __future__ import annotations

import pytest

from integration.adaptive_decision import _make_message, adaptive_decision_node

REQUIRED_KEYS = {
    "outcome",
    "decision_rationale",
    "decision_details",
    "current_stage",
    "stage_history",
    "messages",
}


@pytest.fixture(autouse=True)
def _disable_rl(monkeypatch):
    """Ensure the fixed-weight path is used (no trained model needed)."""
    monkeypatch.setenv("USE_RL_POLICY", "false")


class TestOutputContract:
    def test_returns_required_keys(self):
        out = adaptive_decision_node({"care_marker_detected": False})
        assert REQUIRED_KEYS <= set(out.keys())

    def test_current_stage_and_history(self):
        out = adaptive_decision_node({"stage_history": ["intake"]})
        assert out["current_stage"] == "decision_node"
        assert out["stage_history"] == ["intake", "decision_node"]

    def test_message_has_content(self):
        out = adaptive_decision_node({})
        assert len(out["messages"]) == 1
        assert "Decision:" in out["messages"][0].content


class TestScoringLogic:
    def test_care_marker_escalates(self):
        out = adaptive_decision_node({"care_marker_detected": True})
        assert out["outcome"] == "ESCALATED_FIU"
        assert out["decision_details"]["risk_level"] == "CRITICAL"

    def test_clean_case_non_suspicious(self):
        out = adaptive_decision_node(
            {
                "red_flags": [],
                "pep_status": "CLEAR",
                "care_marker_detected": False,
                "turnover_12m": 50_000,
                "worldcheck_result": {"adverse_media_hits": 0},
            }
        )
        assert out["outcome"] == "NON_SUSPICIOUS"

    def test_high_risk_case_suspicious(self):
        out = adaptive_decision_node(
            {
                "red_flags": [
                    {"severity": "HIGH"},
                    {"severity": "HIGH"},
                ],
                "pep_status": "PEP",
                "care_marker_detected": False,
                "turnover_12m": 1_000_000,
                "worldcheck_result": {"adverse_media_hits": 3},
            }
        )
        assert out["outcome"] == "SUSPICIOUS"
        assert out["decision_details"]["risk_score"] >= 40

    def test_fixed_weights_action_label(self):
        out = adaptive_decision_node({"care_marker_detected": False})
        assert out["decision_details"]["rl_action"] == "FIXED_WEIGHTS"

    def test_adverse_media_as_list_is_counted(self):
        out = adaptive_decision_node(
            {
                "red_flags": [],
                "pep_status": "CLEAR",
                "care_marker_detected": False,
                "turnover_12m": 0,
                "worldcheck_result": {"adverse_media": ["hit1", "hit2"]},
            }
        )
        # worldcheck weight (10) * min(2,3) = 20 contribution, below default
        # threshold (40) on its own → NON_SUSPICIOUS, but score must reflect it.
        assert out["decision_details"]["risk_score"] == 20


def test_make_message_fallback_shape():
    msg = _make_message("hello")
    assert msg.content == "hello"
