"""
Quick local smoke test for Phase 04 final decision layer.

Usage:
    python -m tests.test_phase4_final_decision
"""

from src.cot.phase2_questionnaire_engine import (
    build_phase2_initial_state,
    build_phase2_summary,
    load_phase1_canonical_schema,
    submit_phase2_answer,
)


def main() -> None:
    schema = load_phase1_canonical_schema()
    state = build_phase2_initial_state(chat_id="phase4-demo", schema=schema)

    # Scenario leaning toward Brown Rot (one-sided wilt + ooze + vascular discoloration)
    answers = [
        "crop is stunted in patches and yellow",
        "wilting on one side of plant",
        "mostly upright plants",
        "stem has vascular discoloration and bacterial ooze",
        "leaves show chlorosis",
        "tuber has enlarged lenticels and stolon end discoloration",
        "roots look weak",
        "mostly after mid season",
    ]

    for answer in answers:
        result = submit_phase2_answer(state, answer, schema, use_phase4_llm=False)
        state = result["state"]

    summary = build_phase2_summary(state)
    phase4 = summary.get("phase4", {})

    print("completed:", summary.get("completed"))
    print("phase4_method:", phase4.get("method"))
    print("selected:", phase4.get("selected_disease_id"), phase4.get("selected_disease_name"))
    print("confidence:", phase4.get("confidence"))
    print("alternates:", phase4.get("alternate_candidates"))


if __name__ == "__main__":
    main()
