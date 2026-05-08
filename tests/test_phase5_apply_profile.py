"""
Quick test to apply Phase 05 tuned profile to a single case.

Usage:
    python -m tests.test_phase5_apply_profile
"""

import json
from pathlib import Path

from src.cot.phase2_questionnaire_engine import (
    FACTOR_SEQUENCE,
    build_phase2_initial_state,
    load_phase1_canonical_schema,
    submit_phase2_answer,
)
from src.cot.phase3_candidate_narrowing import compute_phase3_candidate_scores


def main() -> None:
    schema = load_phase1_canonical_schema()
    profile_path = Path("data/COT/phase5_weight_profile.json")
    if not profile_path.exists():
        print("Profile not found. Run: python -m src.cot.phase5_calibration")
        return

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    tuned_weights = profile.get("factor_weights", {})
    scoring_cfg = profile.get("scoring_config", {})

    state = build_phase2_initial_state(chat_id="phase5-apply", schema=schema)
    answers = {
        "general_appearance": "stunted plants in patches",
        "wilting": "wilting on one side",
        "growth_habit": "upright",
        "stems": "vascular discoloration and ooze",
        "leaves": "chlorosis",
        "tubers": "stolon end discoloured",
        "roots": "normal",
        "period": "mid season",
    }

    for factor in FACTOR_SEQUENCE:
        submit_phase2_answer(state, answers.get(factor, "not sure"), schema, use_phase4_llm=False)

    phase3 = compute_phase3_candidate_scores(
        state,
        schema,
        factor_weights_override=tuned_weights,
        scoring_config=scoring_cfg,
    )

    top = phase3.get("ranked_candidates", [])[:3]
    print("Top-3 with tuned profile:")
    for row in top:
        print(row["disease_id"], row["disease_name"], row["score"])


if __name__ == "__main__":
    main()
