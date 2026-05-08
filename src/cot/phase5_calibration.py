"""
Phase 05 Calibration Harness

Tunes Phase 03 factor weights and scoring penalties against labeled diagnosis cases.

Inputs:
- data/COT/canonical_disease_schema_phase1.json
- data/COT/phase5_calibration_cases.json

Outputs:
- data/COT/phase5_calibration_report.json
- data/COT/phase5_weight_profile.json

Usage:
    python -m src.cot.phase5_calibration
"""

from __future__ import annotations

import itertools
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.cot.phase2_questionnaire_engine import (
    FACTOR_SEQUENCE,
    build_phase2_initial_state,
    load_phase1_canonical_schema,
    submit_phase2_answer,
)
from src.cot.phase3_candidate_narrowing import compute_phase3_candidate_scores


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CASES_PATH = os.path.join(PROJECT_ROOT, "data", "COT", "phase5_calibration_cases.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "COT", "phase5_calibration_report.json")
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "COT", "phase5_weight_profile.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0.0:
        n = len(weights)
        return {k: round(1.0 / n, 6) for k in weights}
    return {k: round(max(0.0, float(v)) / total, 6) for k, v in weights.items()}


def _apply_multipliers(base: Dict[str, float], multipliers: Dict[str, float]) -> Dict[str, float]:
    w = deepcopy(base)
    for k, m in multipliers.items():
        w[k] = float(w.get(k, 0.0)) * float(m)
    return _normalize_weights(w)


def _load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_case(
    schema: Dict[str, Any],
    case: Dict[str, Any],
    tuned_weights: Dict[str, float],
    scoring_cfg: Dict[str, float],
) -> Dict[str, Any]:
    state = build_phase2_initial_state(chat_id=f"calib-{case['case_id']}", schema=schema)

    # Build normalized state answers via phase2 parser.
    for factor in FACTOR_SEQUENCE:
        answer = case.get("answers", {}).get(factor, "not sure")
        result = submit_phase2_answer(state, answer, schema, use_phase4_llm=False)
        state = result["state"]

    phase3 = compute_phase3_candidate_scores(
        state,
        schema,
        factor_weights_override=tuned_weights,
        scoring_config=scoring_cfg,
    )

    ranked = phase3.get("ranked_candidates", [])
    top1 = ranked[0]["disease_id"] if ranked else None
    top3_ids = [x.get("disease_id") for x in ranked[:3]]

    expected = str(case.get("expected_disease_id"))

    return {
        "case_id": case.get("case_id"),
        "expected_disease_id": expected,
        "top1_disease_id": top1,
        "top3_disease_ids": top3_ids,
        "top1_hit": bool(top1 == expected),
        "top3_hit": bool(expected in top3_ids),
        "score_spread": phase3.get("score_spread", 0.0),
    }


def _score_trial(case_results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(case_results))
    top1 = sum(1 for c in case_results if c["top1_hit"]) / n
    top3 = sum(1 for c in case_results if c["top3_hit"]) / n
    spread_avg = sum(float(c.get("score_spread", 0.0)) for c in case_results) / n

    # weighted objective prioritizes top-1 accuracy
    objective = (0.70 * top1) + (0.25 * top3) + (0.05 * min(1.0, spread_avg))

    return {
        "top1_accuracy": round(top1, 4),
        "top3_recall": round(top3, 4),
        "avg_score_spread": round(spread_avg, 4),
        "objective": round(objective, 6),
    }


def run_phase5_calibration() -> Dict[str, Any]:
    schema = load_phase1_canonical_schema()
    base_weights = schema.get("factor_weights", {})
    cases = _load_cases()

    if not cases:
        raise ValueError("Calibration cases file is empty.")

    # Search spaces (small grid for quick iteration in development)
    multiplier_grid = [
        {
            "wilting": wm,
            "stems": sm,
            "leaves": lm,
            "tubers": tm,
            "roots": rm,
        }
        for wm, sm, lm, tm, rm in itertools.product(
            [1.0, 1.15],
            [1.0, 1.20],
            [1.0],
            [1.0, 1.20],
            [1.0],
        )
    ]

    scoring_grid = [
        {
            "match_gain": mg,
            "conflict_penalty": cp,
            "weak_mismatch_penalty": wmp,
            "free_text_base": ftb,
            "free_text_lexical_gain": ftl,
            "elimination_conflict_threshold": ect,
        }
        for mg, cp, wmp, ftb, ftl, ect in itertools.product(
            [1.0, 1.15],
            [1.1, 1.25],
            [-0.2, -0.3],
            [0.15],
            [0.35],
            [2, 3],
        )
    ]

    trials: List[Dict[str, Any]] = []
    trial_index = 0

    for multipliers in multiplier_grid:
        tuned_weights = _apply_multipliers(base_weights, multipliers)
        for scoring_cfg in scoring_grid:
            trial_index += 1
            if trial_index % 25 == 0:
                print(f"Calibration progress: trial {trial_index}/{len(multiplier_grid) * len(scoring_grid)}")
            case_results = [
                _evaluate_case(schema, case, tuned_weights, scoring_cfg)
                for case in cases
            ]
            metrics = _score_trial(case_results)

            trials.append(
                {
                    "trial_id": trial_index,
                    "weight_multipliers": multipliers,
                    "tuned_weights": tuned_weights,
                    "scoring_config": scoring_cfg,
                    "metrics": metrics,
                    "case_results": case_results,
                }
            )

    trials.sort(key=lambda t: t["metrics"]["objective"], reverse=True)
    best = trials[0]

    report = {
        "phase": "phase05_calibration",
        "generated_at_utc": _now_iso(),
        "cases_path": os.path.relpath(CASES_PATH, PROJECT_ROOT).replace("\\", "/"),
        "cases_count": len(cases),
        "trials_count": len(trials),
        "best_trial": {
            "trial_id": best["trial_id"],
            "weight_multipliers": best["weight_multipliers"],
            "tuned_weights": best["tuned_weights"],
            "scoring_config": best["scoring_config"],
            "metrics": best["metrics"],
        },
        "top_trials": [
            {
                "trial_id": t["trial_id"],
                "metrics": t["metrics"],
                "weight_multipliers": t["weight_multipliers"],
                "scoring_config": t["scoring_config"],
            }
            for t in trials[:5]
        ],
    }

    profile = {
        "profile_name": "phase5_tuned_v1",
        "generated_at_utc": report["generated_at_utc"],
        "factor_weights": best["tuned_weights"],
        "scoring_config": best["scoring_config"],
        "source": {
            "cases": os.path.relpath(CASES_PATH, PROJECT_ROOT).replace("\\", "/"),
            "metrics": best["metrics"],
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return {
        "report_path": REPORT_PATH,
        "profile_path": PROFILE_PATH,
        "best_metrics": best["metrics"],
        "best_profile": profile,
    }


def main() -> None:
    result = run_phase5_calibration()
    print("=" * 72)
    print("Phase 05 calibration completed")
    print(f"Report:  {result['report_path']}")
    print(f"Profile: {result['profile_path']}")
    print(f"Best metrics: {result['best_metrics']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
