"""
Quick local smoke test for Phase 03 scoring + narrowing.

Usage:
    python -m tests.test_phase3_candidate_narrowing
"""

from src.cot.phase2_questionnaire_engine import (
    build_phase2_initial_state,
    load_phase1_canonical_schema,
    submit_phase2_answer,
)


def main() -> None:
    schema = load_phase1_canonical_schema()
    state = build_phase2_initial_state(chat_id="phase3-demo", schema=schema)

    answers = [
        "stunted in patches with yellowing",
        "wilting on one side with no full collapse yet",
        "upright growth with mild curling",
        "stem has vascular discoloration and bacterial ooze",
    ]

    for idx, answer in enumerate(answers, start=1):
        result = submit_phase2_answer(state, answer, schema)
        state = result["state"]
        phase3 = state.get("phase3", {})
        top = phase3.get("active_candidates", [])[:3]

        print(f"\nStep {idx}: {answer}")
        print("uncertainty:", phase3.get("uncertainty"))
        print("active_count:", len(phase3.get("active_candidate_ids", [])))
        print("top3:", [(x.get("disease_id"), x.get("disease_name"), x.get("score")) for x in top])


if __name__ == "__main__":
    main()
