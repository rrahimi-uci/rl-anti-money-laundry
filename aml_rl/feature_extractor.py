"""Feature extractor — converts raw AML investigation state to a fixed-length
numeric vector suitable for the RL policy network."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import LOB_VOCAB, REASON_CODE_VOCAB, FEATURE_DIM


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    """One-hot encode *value* against *vocab*.  Unknown values → all-zero."""
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def extract_features(state: Dict[str, Any]) -> np.ndarray:
    """Return a (FEATURE_DIM,) float32 numpy array from an AML state dict.

    The state dict mirrors ``AMLState`` from the LangGraph pipeline plus the
    scenario data injected by mock APIs.  We pull whichever keys are available.

    Feature layout (24 dims):
        [0:4]   LOB one-hot         (BUSINESS, PB, RETAIL, CORPORATE)
        [4:11]  Reason-code one-hot (7 codes)
        [11]    num red flags HIGH severity
        [12]    num red flags MEDIUM severity
        [13]    num red flags LOW severity
        [14]    is_pep              (binary)
        [15]    is_sanctioned       (binary)
        [16]    has_care_marker     (binary)
        [17]    turnover_12m_norm   (float, /1e6 for scaling)
        [18]    adverse_media_hits  (count, capped at 5)
        [19]    cash_deposit_ratio  (float 0-1)
        [20]    intl_transfer_ratio (float 0-1)
        [21]    account_age_years   (float, capped at 50)
        [22]    prior_alerts_count  (count, capped at 10)
        [23]    prior_sars_count    (count, capped at 5)
    """

    features: List[float] = []

    # ── Categorical one-hots ─────────────────────────────────────────────
    features.extend(_one_hot(state.get("lob", ""), LOB_VOCAB))
    features.extend(_one_hot(state.get("reason_code", ""), REASON_CODE_VOCAB))

    # ── Red-flag counts by severity ──────────────────────────────────────
    red_flags = state.get("red_flags", [])
    features.append(float(sum(1 for rf in red_flags if rf.get("severity") == "HIGH")))
    features.append(float(sum(1 for rf in red_flags if rf.get("severity") == "MEDIUM")))
    features.append(float(sum(1 for rf in red_flags if rf.get("severity") == "LOW")))

    # ── PEP / sanctions / care ───────────────────────────────────────────
    pep = state.get("pep_status", "CLEAR")
    features.append(1.0 if pep == "PEP" else 0.0)
    features.append(1.0 if pep == "SANCTIONED" else 0.0)
    features.append(1.0 if state.get("care_marker_detected", False) else 0.0)

    # ── Turnover (normalised) ────────────────────────────────────────────
    turnover = float(state.get("turnover_12m", 0))
    features.append(min(turnover / 1_000_000.0, 5.0))

    # ── WorldCheck adverse media ─────────────────────────────────────────
    wc = state.get("worldcheck_result", {})
    adverse = wc.get("adverse_media", wc.get("adverse_media_hits", 0))
    if isinstance(adverse, list):
        adverse = len(adverse)
    features.append(min(float(adverse), 5.0))

    # ── Transaction ratios ───────────────────────────────────────────────
    txn_summary = state.get("transaction_summary", {})
    total_credits = txn_summary.get("total_credits", 0) or 1  # avoid div/0
    cash_deposits = txn_summary.get("cash_deposit_total", 0)
    total_debits = txn_summary.get("total_debits", 0) or 1
    intl_transfers = txn_summary.get("international_transfer_total", 0)

    features.append(min(cash_deposits / total_credits, 1.0))
    features.append(min(intl_transfers / total_debits, 1.0))

    # ── Account age ──────────────────────────────────────────────────────
    customer_since = state.get("customer_profile", {}).get("customer_since", "")
    if customer_since:
        try:
            from datetime import date

            y, m, d = customer_since.split("-")
            opened = date(int(y), int(m), int(d))
            age_years = (date.today() - opened).days / 365.25
            features.append(min(float(age_years), 50.0))
        except (ValueError, TypeError):
            features.append(0.0)
    else:
        features.append(0.0)

    # ── Prior alerts / SARs ──────────────────────────────────────────────
    features.append(min(float(len(state.get("alert_history", []))), 10.0))
    features.append(min(float(len(state.get("sar_history", []))), 5.0))

    arr = np.array(features, dtype=np.float32)

    assert arr.shape == (FEATURE_DIM,), (
        f"Expected {FEATURE_DIM} features, got {arr.shape[0]}"
    )
    return arr
