"""Tests for reward.compute_reward."""

from __future__ import annotations

import pytest

from aml_rl.config import (
    REWARD_CONFIRM,
    REWARD_EFFICIENT_CLOSE,
    REWARD_ESCALATE,
    REWARD_OVERRIDE,
)
from aml_rl.reward import compute_reward


class TestGroundTruthReward:
    """Reward computed from synthetic ground truth labels."""

    def test_correct_suspicious(self):
        r = compute_reward("SUSPICIOUS", {}, ground_truth_outcome="SUSPICIOUS")
        assert r == REWARD_CONFIRM

    def test_correct_non_suspicious_efficient(self):
        r = compute_reward("NON_SUSPICIOUS", {}, ground_truth_outcome="NON_SUSPICIOUS")
        assert r == REWARD_EFFICIENT_CLOSE

    def test_correct_escalated_fiu(self):
        r = compute_reward("ESCALATED_FIU", {}, ground_truth_outcome="ESCALATED_FIU")
        assert r == REWARD_CONFIRM

    def test_wrong_missed_escalation(self):
        r = compute_reward("NON_SUSPICIOUS", {}, ground_truth_outcome="ESCALATED_FIU")
        assert r == REWARD_ESCALATE

    def test_wrong_override(self):
        r = compute_reward("SUSPICIOUS", {}, ground_truth_outcome="NON_SUSPICIOUS")
        assert r == REWARD_OVERRIDE

    def test_wrong_suspicious_vs_escalated(self):
        r = compute_reward("SUSPICIOUS", {}, ground_truth_outcome="ESCALATED_FIU")
        assert r == REWARD_ESCALATE


class TestHITLReward:
    """Reward computed from analyst HITL gate decisions."""

    def test_all_positive_gates(self):
        decisions = {
            "hitl1_decision": "CONFIRM",
            "hitl2_decision": "APPROVE",
            "hitl3_decision": "PROCEED",
            "hitl4_decision": "SUBMIT",
        }
        r = compute_reward("SUSPICIOUS", decisions)
        assert -1.0 <= r <= 1.0
        assert r > 0  # All positive → positive reward

    def test_all_negative_gates(self):
        decisions = {
            "hitl1_decision": "OVERRIDE",
            "hitl2_decision": "REJECT_WITH_CORRECTIONS",
            "hitl3_decision": "FIU",
        }
        r = compute_reward("NON_SUSPICIOUS", decisions)
        assert r < 0  # All negative → negative reward

    def test_empty_decisions_zero(self):
        r = compute_reward("SUSPICIOUS", {})
        assert r == 0.0

    def test_none_decision_skipped(self):
        decisions = {"hitl1_decision": None, "hitl2_decision": "APPROVE"}
        r = compute_reward("SUSPICIOUS", decisions)
        # Only 1 gate effective → reward = 0.25 / 1 * 2 = 0.5
        assert r == pytest.approx(0.5, abs=0.01)

    def test_reward_in_range(self):
        """Reward should always be in [-1, 1]."""
        decisions = {
            "hitl1_decision": "OVERRIDE",
            "hitl2_decision": "ESCALATE",
            "hitl3_decision": "FIU",
            "hitl4_decision": "AMEND",
        }
        r = compute_reward("SUSPICIOUS", decisions)
        assert -1.0 <= r <= 1.0
