"""Generate synthetic training episodes from AML scenarios.

Reads the 4 predefined scenarios from the AML LangGraph project, applies
random perturbations (turnover jitter, flag count variation, etc.), and
writes JSONL episode files suitable for the RL environment.

Usage:
    python -m data.generate_episodes --out data/episodes.jsonl --count 500
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys
from typing import Any, Dict, List

# ── Import scenarios from sibling project ────────────────────────────────
SCENARIOS_DIR = pathlib.Path(__file__).resolve().parents[2] / "aml-poc" / "aml-langgraph"
sys.path.insert(0, str(SCENARIOS_DIR))

from scenarios import SCENARIOS  # noqa: E402

# ── Ground-truth outcomes per scenario ───────────────────────────────────
GROUND_TRUTH: Dict[str, str] = {
    "structuring_cash_business": "SUSPICIOUS",
    "pep_sanctions_hit": "SUSPICIOUS",
    "dormant_reactivation": "ESCALATED_FIU",
    "clean_retail": "NON_SUSPICIOUS",
}

# ── Synthetic-only scenario templates (no SCENARIOS entry needed) ────────
# These create more diverse training data beyond the 4 base scenarios.

SYNTHETIC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "shell_company_layering": {
        "ground_truth": "SUSPICIOUS",
        "lob": "CORPORATE",
        "reason_code": "UNUSUAL_PATTERN",
        "pep_status": "CLEAR",
        "care_marker": False,
        "turnover_range": (800_000, 3_000_000),
        "income_range": (200_000, 800_000),
        "adverse_media_range": (0, 2),
        "high_flags": (2, 4),
        "med_flags": (1, 3),
        "low_flags": (0, 2),
        "customer_since": "2023-01-01",
    },
    "sudden_wealth_pb": {
        "ground_truth": "SUSPICIOUS",
        "lob": "PB",
        "reason_code": "UNUSUAL_PATTERN",
        "pep_status": "CLEAR",
        "care_marker": False,
        "turnover_range": (500_000, 2_000_000),
        "income_range": (100_000, 300_000),
        "adverse_media_range": (0, 1),
        "high_flags": (1, 3),
        "med_flags": (1, 2),
        "low_flags": (0, 2),
        "customer_since": "2020-06-15",
    },
    "third_party_payments": {
        "ground_truth": "SUSPICIOUS",
        "lob": "RETAIL",
        "reason_code": "UNUSUAL_PATTERN",
        "pep_status": "CLEAR",
        "care_marker": False,
        "turnover_range": (50_000, 200_000),
        "income_range": (25_000, 60_000),
        "adverse_media_range": (0, 1),
        "high_flags": (1, 2),
        "med_flags": (1, 3),
        "low_flags": (0, 2),
        "customer_since": "2018-03-10",
    },
    "low_risk_business": {
        "ground_truth": "NON_SUSPICIOUS",
        "lob": "BUSINESS",
        "reason_code": "LARGE_CASH_DEPOSIT",
        "pep_status": "CLEAR",
        "care_marker": False,
        "turnover_range": (100_000, 400_000),
        "income_range": (40_000, 120_000),
        "adverse_media_range": (0, 0),
        "high_flags": (0, 0),
        "med_flags": (0, 1),
        "low_flags": (0, 2),
        "customer_since": "2010-01-01",
    },
    "pep_clean_activity": {
        "ground_truth": "NON_SUSPICIOUS",
        "lob": "PB",
        "reason_code": "PEP_ACTIVITY",
        "pep_status": "PEP",
        "care_marker": False,
        "turnover_range": (200_000, 500_000),
        "income_range": (300_000, 600_000),
        "adverse_media_range": (0, 0),
        "high_flags": (0, 0),
        "med_flags": (0, 1),
        "low_flags": (0, 1),
        "customer_since": "2015-09-20",
    },
    "elderly_care_marker": {
        "ground_truth": "ESCALATED_FIU",
        "lob": "RETAIL",
        "reason_code": "UNUSUAL_PATTERN",
        "pep_status": "CLEAR",
        "care_marker": True,
        "turnover_range": (10_000, 50_000),
        "income_range": (12_000, 25_000),
        "adverse_media_range": (0, 0),
        "high_flags": (0, 1),
        "med_flags": (0, 2),
        "low_flags": (0, 1),
        "customer_since": "2000-01-01",
    },
}

# ── Red-flag templates ───────────────────────────────────────────────────
RED_FLAG_TEMPLATES: List[Dict[str, str]] = [
    {"type": "STRUCTURING", "description": "Multiple deposits just under reporting threshold", "severity": "HIGH"},
    {"type": "RAPID_MOVEMENT", "description": "Funds moved within 24h of receipt", "severity": "HIGH"},
    {"type": "HIGH_RISK_JURISDICTION", "description": "Transfers to/from high-risk jurisdiction", "severity": "HIGH"},
    {"type": "TURNOVER_VS_INCOME", "description": "Cash turnover exceeds declared income", "severity": "MEDIUM"},
    {"type": "DORMANT_REACTIVATION", "description": "Account reactivated after extended dormancy", "severity": "MEDIUM"},
    {"type": "THIRD_PARTY_CREDITS", "description": "Credits from unknown third parties", "severity": "MEDIUM"},
    {"type": "ROUND_AMOUNTS", "description": "Multiple round-number transactions", "severity": "LOW"},
    {"type": "PEAK_VARIANCE", "description": "Monthly peak significantly exceeds average", "severity": "LOW"},
    {"type": "MISSING_NARRATIVE", "description": "Transactions with vague or missing narratives", "severity": "LOW"},
]


def _perturb_scenario(
    scenario_key: str,
    scenario: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Create a randomised variant of a base scenario.

    Returns a flat dict matching AMLState + extra fields the feature
    extractor expects (transaction_summary, customer_profile, etc.).
    """

    state: Dict[str, Any] = {}

    # ── Alert-level fields ───────────────────────────────────────────────
    alert = scenario["alert"]
    state["alert_id"] = alert["alert_id"]
    state["customer_cis"] = alert["customer_cis"]
    state["lob"] = alert["lob"]
    state["reason_code"] = alert["reason_code"]

    # ── Customer profile (pass through with minor age jitter) ────────────
    cp = copy.deepcopy(scenario["customer_profile"])
    cp["annual_income"] = max(0, cp["annual_income"] + rng.randint(-10000, 10000))
    state["customer_profile"] = cp

    # ── PEP status ───────────────────────────────────────────────────────
    wc = copy.deepcopy(scenario["worldcheck"])
    is_pep = wc["pep_result"]["is_pep"]
    is_sanctioned = wc["sanctions_result"]["is_sanctioned"]
    if is_pep:
        state["pep_status"] = "PEP"
    elif is_sanctioned:
        state["pep_status"] = "SANCTIONED"
    else:
        state["pep_status"] = "CLEAR"
    state["worldcheck_result"] = wc

    # ── Care marker ──────────────────────────────────────────────────────
    state["care_marker_detected"] = cp.get("care_marker", False)

    # ── Turnover (jitter ±20%) ───────────────────────────────────────────
    ts = copy.deepcopy(scenario["transaction_summary"])
    base_turnover = sum(ts.get("monthly_turnover", [0]))
    jitter = rng.uniform(0.8, 1.2)
    state["turnover_12m"] = round(base_turnover * jitter, 2)
    ts["total_credits"] = round(ts.get("total_credits", 0) * jitter, 2)
    ts["total_debits"] = round(ts.get("total_debits", 0) * jitter, 2)
    state["transaction_summary"] = ts

    # ── Red flags (randomised subset) ────────────────────────────────────
    if scenario_key == "clean_retail":
        # Clean scenario: 0-1 low flags
        num_flags = rng.choices([0, 1], weights=[0.7, 0.3])[0]
        pool = [rf for rf in RED_FLAG_TEMPLATES if rf["severity"] == "LOW"]
    elif scenario_key == "dormant_reactivation":
        # Care marker case: 1-3 flags
        num_flags = rng.randint(1, 3)
        pool = RED_FLAG_TEMPLATES
    elif scenario_key == "pep_sanctions_hit":
        # PEP case: 2-4 flags, biased towards HIGH
        num_flags = rng.randint(2, 4)
        pool = [rf for rf in RED_FLAG_TEMPLATES if rf["severity"] in ("HIGH", "MEDIUM")]
    else:
        # Structuring: 2-5 flags
        num_flags = rng.randint(2, 5)
        pool = RED_FLAG_TEMPLATES

    state["red_flags"] = rng.sample(pool, min(num_flags, len(pool)))

    # ── Prior history ────────────────────────────────────────────────────
    state["alert_history"] = scenario.get("alert_history", [])
    state["sar_history"] = scenario.get("sar_history", [])

    return state


def _build_synthetic_episode(
    key: str,
    template: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Build a fully synthetic episode from a template (no base scenario needed)."""

    turnover = rng.uniform(*template["turnover_range"])
    income = rng.uniform(*template["income_range"])

    state: Dict[str, Any] = {
        "alert_id": f"SYN-{rng.randint(10000, 99999)}",
        "customer_cis": f"CIS-SYN-{rng.randint(10000, 99999)}",
        "lob": template["lob"],
        "reason_code": template["reason_code"],
        "pep_status": template["pep_status"],
        "care_marker_detected": template["care_marker"],
        "turnover_12m": round(turnover, 2),
        "customer_profile": {
            "name": f"Synthetic Customer {rng.randint(1, 9999)}",
            "annual_income": round(income, 2),
            "customer_since": template["customer_since"],
            "care_marker": template["care_marker"],
        },
        "worldcheck_result": {
            "adverse_media_hits": rng.randint(*template["adverse_media_range"]),
            "status": "COMPLETED",
        },
        "transaction_summary": {
            "total_credits": round(turnover * rng.uniform(0.9, 1.1), 2),
            "total_debits": round(turnover * rng.uniform(0.6, 0.9), 2),
            "cash_deposit_total": round(turnover * rng.uniform(0.1, 0.5), 2),
            "international_transfer_total": round(turnover * rng.uniform(0.0, 0.3), 2),
        },
        "alert_history": [{"id": f"H-{i}"} for i in range(rng.randint(0, 3))],
        "sar_history": [{"id": f"S-{i}"} for i in range(rng.randint(0, 1))],
    }

    # Red flags by severity
    high_pool = [rf for rf in RED_FLAG_TEMPLATES if rf["severity"] == "HIGH"]
    med_pool = [rf for rf in RED_FLAG_TEMPLATES if rf["severity"] == "MEDIUM"]
    low_pool = [rf for rf in RED_FLAG_TEMPLATES if rf["severity"] == "LOW"]

    flags = []
    nh = rng.randint(*template["high_flags"])
    nm = rng.randint(*template["med_flags"])
    nl = rng.randint(*template["low_flags"])
    flags.extend(rng.sample(high_pool, min(nh, len(high_pool))))
    flags.extend(rng.sample(med_pool, min(nm, len(med_pool))))
    flags.extend(rng.sample(low_pool, min(nl, len(low_pool))))
    state["red_flags"] = flags

    return state


def generate_episodes(
    count: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate *count* randomised episodes across base + synthetic scenarios.

    Splits roughly 50/50 between base (4 SCENARIOS) and synthetic (6 templates).
    """

    rng = random.Random(seed)
    episodes: List[Dict[str, Any]] = []

    scenario_keys = list(SCENARIOS.keys())
    synthetic_keys = list(SYNTHETIC_TEMPLATES.keys())
    base_count = count // 2
    synth_count = count - base_count

    # ── Base scenarios ───────────────────────────────────────────────────
    for i in range(base_count):
        key = scenario_keys[i % len(scenario_keys)]
        scenario = SCENARIOS[key]
        state = _perturb_scenario(key, scenario, rng)
        gt = GROUND_TRUTH[key]

        # 8% noise injection
        if rng.random() < 0.08 and gt != "ESCALATED_FIU":
            gt = "NON_SUSPICIOUS" if gt == "SUSPICIOUS" else "SUSPICIOUS"

        episodes.append({"state": state, "ground_truth": gt, "scenario_key": key})

    # ── Synthetic scenarios ──────────────────────────────────────────────
    for i in range(synth_count):
        key = synthetic_keys[i % len(synthetic_keys)]
        template = SYNTHETIC_TEMPLATES[key]
        state = _build_synthetic_episode(key, template, rng)
        gt = template["ground_truth"]

        # 8% noise injection
        if rng.random() < 0.08 and gt != "ESCALATED_FIU":
            gt = "NON_SUSPICIOUS" if gt == "SUSPICIOUS" else "SUSPICIOUS"

        episodes.append({"state": state, "ground_truth": gt, "scenario_key": key})

    rng.shuffle(episodes)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AML RL training episodes")
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "episodes.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--count", type=int, default=2000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    episodes = generate_episodes(count=args.count, seed=args.seed)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # Stats
    from collections import Counter

    gt_counts = Counter(ep["ground_truth"] for ep in episodes)
    sc_counts = Counter(ep["scenario_key"] for ep in episodes)
    print(f"Generated {len(episodes)} episodes → {out_path}")
    print(f"  Ground truth: {dict(gt_counts)}")
    print(f"  Scenarios:    {dict(sc_counts)}")


if __name__ == "__main__":
    main()
