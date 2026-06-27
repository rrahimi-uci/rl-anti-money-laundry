"""Tests for feature_extractor.extract_features."""

from __future__ import annotations

import numpy as np
import pytest

from aml_rl.config import FEATURE_DIM
from aml_rl.feature_extractor import extract_features


def _base_state(**overrides):
    """Minimal AML state dict with sensible defaults."""
    state = {
        "lob": "RETAIL",
        "reason_code": "STRUCTURING",
        "red_flags": [
            {"type": "STRUCTURING", "severity": "HIGH"},
            {"type": "ROUND_AMOUNTS", "severity": "LOW"},
        ],
        "pep_status": "CLEAR",
        "care_marker_detected": False,
        "turnover_12m": 250_000,
        "worldcheck_result": {"adverse_media_hits": 1},
        "transaction_summary": {
            "total_credits": 100_000,
            "total_debits": 80_000,
            "cash_deposit_total": 30_000,
            "international_transfer_total": 10_000,
        },
        "customer_profile": {"customer_since": "2020-01-15"},
        "alert_history": [{"id": "A1"}],
        "sar_history": [],
    }
    state.update(overrides)
    return state


class TestExtractFeatures:
    def test_output_shape(self):
        obs = extract_features(_base_state())
        assert obs.shape == (FEATURE_DIM,)
        assert obs.dtype == np.float32

    def test_lob_one_hot(self):
        obs = extract_features(_base_state(lob="BUSINESS"))
        assert obs[0] == 1.0
        assert obs[1] == 0.0

        obs2 = extract_features(_base_state(lob="CORPORATE"))
        assert obs2[3] == 1.0
        assert obs2[0] == 0.0

    def test_unknown_lob_all_zeros(self):
        obs = extract_features(_base_state(lob="UNKNOWN_LOB"))
        assert obs[0:4].sum() == 0.0

    def test_reason_code_one_hot(self):
        obs = extract_features(_base_state(reason_code="PEP_ACTIVITY"))
        # PEP_ACTIVITY is index 1 in REASON_CODE_VOCAB
        assert obs[4 + 1] == 1.0
        assert obs[4] == 0.0

    def test_red_flag_counts(self):
        flags = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
        ]
        obs = extract_features(_base_state(red_flags=flags))
        assert obs[11] == 2.0  # HIGH
        assert obs[12] == 1.0  # MEDIUM
        assert obs[13] == 0.0  # LOW

    def test_pep_status(self):
        obs_pep = extract_features(_base_state(pep_status="PEP"))
        assert obs_pep[14] == 1.0  # is_pep
        assert obs_pep[15] == 0.0  # is_sanctioned

        obs_sanc = extract_features(_base_state(pep_status="SANCTIONED"))
        assert obs_sanc[14] == 0.0
        assert obs_sanc[15] == 1.0

    def test_care_marker(self):
        obs = extract_features(_base_state(care_marker_detected=True))
        assert obs[16] == 1.0

        obs2 = extract_features(_base_state(care_marker_detected=False))
        assert obs2[16] == 0.0

    def test_turnover_normalised(self):
        obs = extract_features(_base_state(turnover_12m=2_000_000))
        assert obs[17] == pytest.approx(2.0, abs=0.01)

    def test_turnover_capped(self):
        obs = extract_features(_base_state(turnover_12m=10_000_000))
        assert obs[17] == 5.0

    def test_adverse_media_capped(self):
        obs = extract_features(
            _base_state(worldcheck_result={"adverse_media_hits": 20})
        )
        assert obs[18] == 5.0

    def test_adverse_media_list(self):
        obs = extract_features(
            _base_state(worldcheck_result={"adverse_media": ["a", "b"]})
        )
        assert obs[18] == 2.0

    def test_cash_deposit_ratio(self):
        obs = extract_features(
            _base_state(
                transaction_summary={
                    "total_credits": 100_000,
                    "total_debits": 50_000,
                    "cash_deposit_total": 50_000,
                    "international_transfer_total": 0,
                }
            )
        )
        assert obs[19] == pytest.approx(0.5, abs=0.01)

    def test_prior_alerts_capped(self):
        obs = extract_features(
            _base_state(alert_history=[{"id": f"A{i}"} for i in range(15)])
        )
        assert obs[22] == 10.0

    def test_empty_state_does_not_crash(self):
        obs = extract_features({})
        assert obs.shape == (FEATURE_DIM,)
        assert np.isfinite(obs).all()
