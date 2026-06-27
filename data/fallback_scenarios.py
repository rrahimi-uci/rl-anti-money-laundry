"""Built-in copy of the four base AML scenarios.

This is a self-contained fallback used by ``generate_episodes.py`` when the
sibling AML LangGraph project (``../../aml-poc/aml-langgraph/scenarios.py``) is
not available.  Each scenario exposes exactly the fields that
``_perturb_scenario`` reads:

    alert               → alert_id, customer_cis, lob, reason_code
    customer_profile    → annual_income, care_marker, customer_since, name, ...
    worldcheck          → pep_result.is_pep, sanctions_result.is_sanctioned,
                          adverse_media_hits
    transaction_summary → monthly_turnover (list), total_credits, total_debits
    alert_history / sar_history (optional)

The values mirror the four canonical scenarios shipped with the upstream
project so that stand-alone regeneration produces statistically comparable
training data.
"""

from __future__ import annotations

from typing import Any, Dict

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ── 1. Cash-intensive business structuring (SUSPICIOUS) ───────────────
    "structuring_cash_business": {
        "alert": {
            "alert_id": "SAM8-ALERT-78421",
            "customer_cis": "CIS40287651",
            "lob": "BUSINESS",
            "reason_code": "STRUCTURING",
        },
        "customer_profile": {
            "cis": "CIS40287651",
            "name": "Ahmed Hassan",
            "occupation": "Restaurant Owner",
            "annual_income": 50_000,
            "source_of_funds": "BUSINESS",
            "customer_since": "2014-06-20",
            "risk_rating": "MEDIUM",
            "care_marker": False,
            "pep_flag": False,
        },
        "worldcheck": {
            "screening_id": "WC-784211",
            "pep_result": {"is_pep": False},
            "sanctions_result": {"is_sanctioned": False},
            "adverse_media_hits": 0,
            "status": "COMPLETED",
        },
        "transaction_summary": {
            "monthly_turnover": [70_000, 82_000, 65_000, 90_000, 78_000, 88_000,
                                 71_000, 95_000, 69_000, 84_000, 77_000, 92_000],
            "total_credits": 961_000,
            "total_debits": 880_000,
            "cash_deposit_total": 540_000,
            "international_transfer_total": 20_000,
        },
        "alert_history": [],
        "sar_history": [],
    },

    # ── 2. PEP with sanctions/adverse-media hit (SUSPICIOUS) ──────────────
    "pep_sanctions_hit": {
        "alert": {
            "alert_id": "SAM8-ALERT-91034",
            "customer_cis": "CIS72849103",
            "lob": "PB",
            "reason_code": "PEP_ACTIVITY",
        },
        "customer_profile": {
            "cis": "CIS72849103",
            "name": "Oluwaseun Adeyemi",
            "occupation": "Consultant",
            "annual_income": 458_000,
            "source_of_funds": "BUSINESS",
            "customer_since": "2019-03-15",
            "risk_rating": "HIGH",
            "care_marker": False,
            "pep_flag": True,
        },
        "worldcheck": {
            "screening_id": "WC-910341",
            "pep_result": {"is_pep": True},
            "sanctions_result": {"is_sanctioned": False},
            "adverse_media_hits": 2,
            "status": "COMPLETED",
        },
        "transaction_summary": {
            "monthly_turnover": [120_000, 95_000, 140_000, 110_000, 130_000,
                                 88_000, 150_000, 102_000, 118_000, 135_000,
                                 99_000, 145_000],
            "total_credits": 1_432_000,
            "total_debits": 1_300_000,
            "cash_deposit_total": 60_000,
            "international_transfer_total": 480_000,
        },
        "alert_history": [{"id": "H-PEP-1"}],
        "sar_history": [],
    },

    # ── 3. Dormant account reactivation, vulnerable customer (ESCALATED) ──
    "dormant_reactivation": {
        "alert": {
            "alert_id": "SAM8-ALERT-55218",
            "customer_cis": "CIS10438726",
            "lob": "RETAIL",
            "reason_code": "DORMANT_REACTIVATION",
        },
        "customer_profile": {
            "cis": "CIS10438726",
            "name": "Margaret Thornton",
            "occupation": "Retired",
            "annual_income": 11_500,
            "source_of_funds": "SAVINGS",
            "customer_since": "1985-04-10",
            "risk_rating": "LOW",
            "care_marker": True,
            "pep_flag": False,
        },
        "worldcheck": {
            "screening_id": "WC-552181",
            "pep_result": {"is_pep": False},
            "sanctions_result": {"is_sanctioned": False},
            "adverse_media_hits": 0,
            "status": "COMPLETED",
        },
        "transaction_summary": {
            "monthly_turnover": [500, 0, 0, 0, 18_000, 22_000, 19_500, 0, 0,
                                 0, 0, 0],
            "total_credits": 41_500,
            "total_debits": 40_000,
            "cash_deposit_total": 2_000,
            "international_transfer_total": 35_000,
        },
        "alert_history": [],
        "sar_history": [],
    },

    # ── 4. Clean retail customer (NON_SUSPICIOUS) ─────────────────────────
    "clean_retail": {
        "alert": {
            "alert_id": "SAM8-ALERT-33150",
            "customer_cis": "CIS85120934",
            "lob": "RETAIL",
            "reason_code": "UNUSUAL_PATTERN",
        },
        "customer_profile": {
            "cis": "CIS85120934",
            "name": "Sarah Mitchell",
            "occupation": "Software Engineer",
            "annual_income": 88_000,
            "source_of_funds": "EMPLOYMENT",
            "customer_since": "2015-09-01",
            "risk_rating": "LOW",
            "care_marker": False,
            "pep_flag": False,
        },
        "worldcheck": {
            "screening_id": "WC-331501",
            "pep_result": {"is_pep": False},
            "sanctions_result": {"is_sanctioned": False},
            "adverse_media_hits": 0,
            "status": "COMPLETED",
        },
        "transaction_summary": {
            "monthly_turnover": [7_300, 7_300, 7_400, 7_300, 7_350, 7_300,
                                 7_300, 7_400, 7_300, 7_300, 7_350, 7_300],
            "total_credits": 88_300,
            "total_debits": 84_000,
            "cash_deposit_total": 1_200,
            "international_transfer_total": 0,
        },
        "alert_history": [],
        "sar_history": [],
    },
}
