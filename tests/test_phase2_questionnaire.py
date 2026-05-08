"""
Quick local smoke test for Phase 02 questionnaire engine.

Usage:
    python -m tests.test_phase2_questionnaire
"""

from src.cot.phase2_questionnaire_engine import (
    build_phase2_initial_state,
    build_phase2_summary,
    get_current_question,
    load_phase1_canonical_schema,
    submit_phase2_answer,
)


def main() -> None:
    schema = load_phase1_canonical_schema()
    state = build_phase2_initial_state(chat_id="phase2-demo", schema=schema)

    answers = [
        "stunted in patches with yellowing",
        "wilting on one side",
        "upright with slight curling",
        "vascular discoloration and ooze in stem",
        "leaf chlorosis and upward rolling",
        "stolon-end lesions with eye discoloration",
        "root galls visible",
        "mostly mid season after wet weather",
    ]

    print("Initial question:", get_current_question(state))

    for idx, answer in enumerate(answers, start=1):
        result = submit_phase2_answer(state, answer, schema)
        state = result["state"]
        progress = result["progress"]

        print(f"\nStep {idx}")
        print("Answer:", answer)
        print("Progress:", f"{progress['completed_steps']}/{progress['total_steps']} ({progress['percent']}%)")
        print("Next question:", result["next_question"])

    summary = build_phase2_summary(state)
    print("\nCompleted:", summary["completed"])
    print("Captured answers:", len(summary["answers"]))


if __name__ == "__main__":
    main()
