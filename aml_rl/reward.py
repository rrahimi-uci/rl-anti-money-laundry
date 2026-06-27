"""Reward calculator — translates HITL analyst feedback into scalar rewards."""

from __future__ import annotations

from typing import Any, Dict

from .config import (
    REWARD_CONFIRM,
    REWARD_EFFICIENT_CLOSE,
    REWARD_ESCALATE,
    REWARD_OVERRIDE,
)


def compute_reward(
    predicted_outcome: str,
    analyst_decisions: Dict[str, str],
    ground_truth_outcome: str | None = None,
) -> float:
    """Compute scalar reward from a completed investigation.

    Parameters
    ----------
    predicted_outcome:
        The automated decision node's recommendation
        (``SUSPICIOUS``, ``NON_SUSPICIOUS``, ``ESCALATED_FIU``).
    analyst_decisions:
        Dict of HITL gate decisions keyed by gate name,
        e.g. ``{"hitl3_decision": "PROCEED", "hitl4_decision": "SUBMIT"}``.
    ground_truth_outcome:
        Optional known-correct label for synthetic training
        (``SUSPICIOUS`` / ``NON_SUSPICIOUS`` / ``ESCALATED_FIU``).

    Returns
    -------
    float  in range [-1.0, +1.0]
    """

    # ── If we have ground truth (synthetic training) ─────────────────────
    if ground_truth_outcome is not None:
        if predicted_outcome == ground_truth_outcome:
            # Bonus for correct non-suspicious (efficient)
            if predicted_outcome == "NON_SUSPICIOUS":
                return REWARD_EFFICIENT_CLOSE
            return REWARD_CONFIRM
        else:
            # Wrong direction
            if ground_truth_outcome == "ESCALATED_FIU":
                return REWARD_ESCALATE  # we underestimated risk
            return REWARD_OVERRIDE

    # ── Reward from HITL decisions (live mode) ───────────────────────────
    reward = 0.0
    n_gates = 0

    for gate, decision in analyst_decisions.items():
        if not decision:
            continue
        n_gates += 1
        decision = decision.upper()

        # Gate 1 (triage): CONFIRM is good, OVERRIDE is bad
        if gate == "hitl1_decision":
            if decision == "CONFIRM":
                reward += 0.25
            elif decision == "OVERRIDE":
                reward -= 0.5

        # Gate 2 (phase 1 review): APPROVE vs REJECT
        elif gate == "hitl2_decision":
            if decision == "APPROVE":
                reward += 0.25
            elif decision in ("REJECT_WITH_CORRECTIONS", "ESCALATE"):
                reward -= 0.5

        # Gate 3 (phase 2 review — closest to the decision node)
        elif gate == "hitl3_decision":
            if decision == "PROCEED":
                reward += 0.5
            elif decision == "FIU":
                reward -= 0.5  # we missed escalation need
            elif decision in ("CC", "NOMO"):
                reward -= 0.25  # too aggressive

        # Gate 4 (SAR review): SUBMIT confirms SAR was warranted
        elif gate == "hitl4_decision":
            if decision == "SUBMIT":
                reward += 0.5
            elif decision in ("AMEND", "REDRAFT"):
                reward -= 0.25  # directionally right but quality issue

    # Normalise to [-1, 1] range
    if n_gates > 0:
        reward = max(-1.0, min(1.0, reward / n_gates * 2))

    return reward
