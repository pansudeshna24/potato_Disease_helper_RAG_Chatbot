"""
Phase 02 Questionnaire Engine

Implements a deterministic 8-step diagnosis questionnaire workflow based on
`data/COT/canonical_disease_schema_phase1.json`.

Phase 02 scope:
- fixed factor order
- fixed question prompts
- structured answer capture
- normalization against Phase 01 synonym dictionary
 - phase-aware integration point for Phase 03 candidate narrowing
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.cot.phase3_candidate_narrowing import update_state_with_phase3
from src.cot.phase4_final_decision import update_state_with_phase4


FACTOR_SEQUENCE = [
    "general_appearance",
    "wilting",
    "growth_habit",
    "stems",
    "leaves",
    "tubers",
    "roots",
    "period",
]

PHASE2_VERSION = "phase2.v1"

QUESTION_BANK = {
    "general_appearance": "What is the general appearance of the crop (normal, stunted in patches, stunted overall, poor vigor, yellowing)?",
    "wilting": "Do you observe wilting symptoms (none, one-sided, general wilting, mostly late-season wilting)?",
    "growth_habit": "How would you describe growth habit (normal, upright/erect, rosette/bushy, spindly, deformed/curling)?",
    "stems": "What stem symptoms are present (normal, vascular discoloration, black lesions, ooze, rot, galls, necrosis)?",
    "leaves": "What leaf symptoms are present (normal, chlorosis, interveinal chlorosis, upward rolling, necrotic spots, concentric rings, mosaic/crinkling)?",
    "tubers": "What tuber symptoms are present (normal, vascular discoloration, stolon-end lesions, cracks/pits, warts/galls, active rot, eye discoloration, sclerotia)?",
    "roots": "What root symptoms are present (normal, galls, rot, poor development, necrosis, cysts)?",
    "period": "At what period or condition did symptoms appear (early, mid-season, late-season, temperature/moisture trigger)?",
}

UNKNOWN_PATTERNS = {
    "not sure",
    "unsure",
    "unknown",
    "dont know",
    "don't know",
    "na",
    "n/a",
}


def _normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_schema_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", "COT", "canonical_disease_schema_phase1.json")


def load_phase1_canonical_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    path = schema_path or _default_schema_path()
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    required = ["factor_sequence", "synonym_dictionary", "diseases"]
    for key in required:
        if key not in schema:
            raise ValueError(f"Invalid Phase 01 schema: missing '{key}'")

    return schema


def _extract_tokens_for_factor(
    factor: str,
    answer_text: str,
    synonym_dictionary: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    normalized = _normalize_text(answer_text)
    token_map = synonym_dictionary.get(factor, {})

    found: List[str] = []
    for token, keywords in token_map.items():
        for keyword in keywords:
            if _normalize_text(keyword) and _normalize_text(keyword) in normalized:
                found.append(token)
                break

    return sorted(set(found))


def _infer_answer_mode(answer_text: str, tokens: List[str]) -> str:
    n = _normalize_text(answer_text)

    if n in {"yes", "y"}:
        return "yes"
    if n in {"no", "n"}:
        return "no"
    if n in UNKNOWN_PATTERNS:
        return "unknown"
    if not tokens:
        return "free_text"
    return "coded"


def _confidence_score(tokens: List[str], answer_mode: str) -> float:
    if answer_mode == "unknown":
        return 0.0
    if answer_mode in {"yes", "no"}:
        return 0.6
    if not tokens:
        return 0.25
    base = 0.45
    boost = min(0.45, 0.15 * len(tokens))
    return round(min(0.95, base + boost), 3)


def build_phase2_initial_state(
    chat_id: str,
    schema: Dict[str, Any],
    existing_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if existing_state and not existing_state.get("completed", False):
        return existing_state

    sequence = schema.get("factor_sequence", FACTOR_SEQUENCE)
    if sequence != FACTOR_SEQUENCE:
        # Keep strict Phase 02 ordering even if schema order changes unexpectedly.
        sequence = FACTOR_SEQUENCE

    return {
        "version": PHASE2_VERSION,
        "chat_id": chat_id,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed": False,
        "step_index": 0,
        "factor_sequence": sequence,
        "answers": {},
        "history": [],
        "phase3": {
            "ranked_candidates": [],
            "active_candidate_ids": [],
            "active_candidates": [],
            "uncertainty": "high",
            "score_spread": 0.0,
            "answered_factors": 0,
            "keep_count": 0,
        },
        "phase4": {
            "method": "pending",
            "selected_disease_id": None,
            "selected_disease_name": None,
            "confidence": 0.0,
            "alternate_candidates": [],
            "rationale": "Final decision not computed yet.",
            "missing_evidence": [],
            "created_at": None,
        },
    }


def get_current_factor(state: Dict[str, Any]) -> Optional[str]:
    if state.get("completed", False):
        return None
    idx = int(state.get("step_index", 0))
    seq = state.get("factor_sequence", FACTOR_SEQUENCE)
    if idx < 0 or idx >= len(seq):
        return None
    return seq[idx]


def get_current_question(state: Dict[str, Any]) -> Optional[str]:
    factor = get_current_factor(state)
    if factor is None:
        return None
    return QUESTION_BANK.get(factor, f"Provide observations for factor: {factor}")


def get_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    seq = state.get("factor_sequence", FACTOR_SEQUENCE)
    total = len(seq)
    done = min(int(state.get("step_index", 0)), total)
    percent = int((done / total) * 100) if total else 0

    return {
        "completed_steps": done,
        "total_steps": total,
        "percent": percent,
        "current_factor": get_current_factor(state),
    }


def submit_phase2_answer(
    state: Dict[str, Any],
    answer_text: str,
    schema: Dict[str, Any],
    use_phase4_llm: bool = True,
) -> Dict[str, Any]:
    if state.get("completed", False):
        return {
            "handled": False,
            "message": "Questionnaire already completed.",
            "state": state,
            "next_question": None,
            "progress": get_progress(state),
        }

    factor = get_current_factor(state)
    if factor is None:
        state["completed"] = True
        state["updated_at"] = _now_iso()
        return {
            "handled": False,
            "message": "No active factor available.",
            "state": state,
            "next_question": None,
            "progress": get_progress(state),
        }

    synonyms = schema.get("synonym_dictionary", {})
    tokens = _extract_tokens_for_factor(factor, answer_text, synonyms)
    answer_mode = _infer_answer_mode(answer_text, tokens)

    structured_answer = {
        "factor": factor,
        "question": QUESTION_BANK.get(factor),
        "raw_answer": answer_text,
        "normalized_tokens": tokens,
        "answer_mode": answer_mode,
        "confidence": _confidence_score(tokens, answer_mode),
        "answered_at": _now_iso(),
    }

    state.setdefault("answers", {})[factor] = structured_answer
    state.setdefault("history", []).append(structured_answer)
    state["step_index"] = int(state.get("step_index", 0)) + 1

    seq = state.get("factor_sequence", FACTOR_SEQUENCE)
    if int(state["step_index"]) >= len(seq):
        state["completed"] = True

    # Phase 03 update: recompute candidate ranking after each captured factor.
    update_state_with_phase3(state, schema)

    # Phase 04 trigger:
    # - always on full completion, or
    # - early when uncertainty already low with tight shortlist after >=4 factors.
    phase3 = state.get("phase3", {})
    early_finalize = (
        int(phase3.get("answered_factors", 0)) >= 4
        and str(phase3.get("uncertainty", "high")) == "low"
        and int(phase3.get("keep_count", 99)) <= 3
    )
    if state.get("completed", False) or early_finalize:
        update_state_with_phase4(state, use_llm=use_phase4_llm)

    state["updated_at"] = _now_iso()

    return {
        "handled": True,
        "message": "Answer recorded.",
        "state": state,
        "next_question": get_current_question(state),
        "progress": get_progress(state),
    }


def build_phase2_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    answers = state.get("answers", {})
    ordered_answers = []
    for factor in FACTOR_SEQUENCE:
        if factor in answers:
            ordered_answers.append(answers[factor])

    return {
        "version": state.get("version", PHASE2_VERSION),
        "chat_id": state.get("chat_id"),
        "completed": bool(state.get("completed", False)),
        "progress": get_progress(state),
        "answers": ordered_answers,
        "phase3": state.get("phase3", {}),
        "phase4": state.get("phase4", {}),
    }
