"""
Phase 04 Final Decision Layer

Uses Phase 03 narrowed candidates and Phase 02 structured observations to produce
final diagnosis output.

Behavior:
- Preferred path: constrained LLM decision over shortlisted candidates.
- Safe fallback: deterministic top-score chooser when LLM/config unavailable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_confidence(top_score: float, second_score: float, answered_factors: int) -> float:
    spread = max(0.0, float(top_score) - float(second_score))
    factor_bonus = min(0.2, 0.02 * max(0, answered_factors))
    # map spread into [0, 1] smoothly with clamp
    spread_component = min(0.75, spread * 3.0)
    confidence = 0.2 + spread_component + factor_bonus
    return round(min(0.97, max(0.05, confidence)), 3)


def _deterministic_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    phase3 = state.get("phase3", {})
    active = phase3.get("active_candidates", [])
    ranked = phase3.get("ranked_candidates", [])

    if not active:
        return {
            "method": "deterministic_fallback",
            "selected_disease_id": None,
            "selected_disease_name": None,
            "confidence": 0.0,
            "alternate_candidates": [],
            "rationale": "No active candidates available from Phase 03.",
            "missing_evidence": [
                "Insufficient normalized evidence to shortlist diseases."
            ],
            "created_at": _now_iso(),
        }

    selected = active[0]
    second_score = active[1]["score"] if len(active) > 1 else active[0]["score"]
    answered = int(phase3.get("answered_factors", 0))

    rationale = (
        "Selected highest-scoring candidate from deterministic Phase 03 ranking. "
        "Use LLM-constrained mode for richer justification when available."
    )

    return {
        "method": "deterministic_fallback",
        "selected_disease_id": selected.get("disease_id"),
        "selected_disease_name": selected.get("disease_name"),
        "confidence": _compute_confidence(selected.get("score", 0.0), second_score, answered),
        "alternate_candidates": [
            {
                "disease_id": c.get("disease_id"),
                "disease_name": c.get("disease_name"),
                "score": c.get("score"),
            }
            for c in active[1:3]
        ],
        "rationale": rationale,
        "missing_evidence": _infer_missing_evidence(state, ranked),
        "created_at": _now_iso(),
    }


def _infer_missing_evidence(state: Dict[str, Any], ranked: List[Dict[str, Any]]) -> List[str]:
    answers = state.get("answers", {})
    missing = []

    for factor in ["stems", "leaves", "tubers", "roots"]:
        ans = answers.get(factor)
        if not ans:
            missing.append(f"No answer recorded for factor: {factor}")
            continue
        mode = ans.get("answer_mode")
        tokens = ans.get("normalized_tokens", [])
        if mode == "unknown" or (mode in {"free_text", "yes", "no"} and not tokens):
            missing.append(f"Low-specificity evidence for factor: {factor}")

    if ranked and len(ranked) > 1:
        top = ranked[0].get("score", 0.0)
        second = ranked[1].get("score", 0.0)
        if (top - second) < 0.08:
            missing.append("Top candidate separation is low; more discriminative symptom detail is needed.")

    return missing[:5]


def _build_llm_prompt(state: Dict[str, Any]) -> str:
    phase3 = state.get("phase3", {})
    active = phase3.get("active_candidates", [])
    answers = state.get("answers", {})

    ordered_answers = []
    for factor in [
        "general_appearance",
        "wilting",
        "growth_habit",
        "stems",
        "leaves",
        "tubers",
        "roots",
        "period",
    ]:
        a = answers.get(factor)
        if not a:
            continue
        ordered_answers.append(
            {
                "factor": factor,
                "raw_answer": a.get("raw_answer"),
                "normalized_tokens": a.get("normalized_tokens", []),
                "answer_mode": a.get("answer_mode"),
                "confidence": a.get("confidence"),
            }
        )

    payload = {
        "active_candidates": active,
        "observations": ordered_answers,
        "instructions": [
            "Choose exactly one disease only from active_candidates.",
            "Do not invent any new disease name or ID.",
            "Use observations and candidate scores together.",
            "If confidence is low, report missing evidence clearly.",
            "Return valid JSON only with the exact keys requested.",
        ],
        "required_output_keys": [
            "selected_disease_id",
            "selected_disease_name",
            "confidence",
            "alternate_candidates",
            "rationale",
            "missing_evidence",
        ],
    }

    return (
        "You are a potato disease diagnosis reasoner. "
        "Select the final diagnosis strictly from provided active candidates.\n\n"
        f"DATA:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return only JSON object."
    )


def _llm_constrained_decision(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq
    except Exception:
        return None

    try:
        llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            streaming=False,
            max_tokens=900,
            timeout=45.0,
            max_retries=2,
        )

        prompt = _build_llm_prompt(state)
        response = llm.invoke(prompt)
        text = (response.content or "").strip()

        # Try direct parse, then fenced-code extraction fallback.
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            parsed = json.loads(text[start : end + 1])

        required = {
            "selected_disease_id",
            "selected_disease_name",
            "confidence",
            "alternate_candidates",
            "rationale",
            "missing_evidence",
        }
        if not required.issubset(parsed.keys()):
            return None

        parsed["method"] = "llm_constrained"
        parsed["created_at"] = _now_iso()

        # confidence hard clamp for stability
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.0))
        except Exception:
            parsed["confidence"] = 0.0
        parsed["confidence"] = round(min(0.99, max(0.01, parsed["confidence"])), 3)

        return parsed

    except Exception:
        return None


def compute_phase4_final_decision(
    state: Dict[str, Any],
    use_llm: bool = True,
) -> Dict[str, Any]:
    # Gate: only decide when there are active candidates.
    phase3 = state.get("phase3", {})
    if not phase3.get("active_candidates"):
        return _deterministic_decision(state)

    if use_llm:
        llm_result = _llm_constrained_decision(state)
        if llm_result is not None:
            return llm_result

    return _deterministic_decision(state)


def update_state_with_phase4(state: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    state["phase4"] = compute_phase4_final_decision(state, use_llm=use_llm)
    return state
