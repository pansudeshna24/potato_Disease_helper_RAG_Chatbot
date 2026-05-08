"""
Phase 03 Candidate Narrowing

Deterministic scoring and narrowing over Phase 01 disease profiles using
captured Phase 02 factor answers.

Design goals:
- transparent factor-wise score trace
- weighted scoring by factor importance
- conflict-aware penalties
- uncertainty-aware shortlist sizing
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


DEFAULT_MIN_KEEP = 3
DEFAULT_MAX_KEEP = 8
DEFAULT_SCORING_CONFIG = {
    "match_gain": 1.0,
    "conflict_penalty": 1.1,
    "weak_mismatch_penalty": -0.2,
    "free_text_base": 0.15,
    "free_text_lexical_gain": 0.35,
    "elimination_conflict_threshold": 2,
}


def _as_set(values: List[str]) -> Set[str]:
    return {str(v) for v in (values or []) if str(v)}


def _lexical_overlap_score(raw_answer: str, evidence: List[str]) -> float:
    words = set((raw_answer or "").lower().split())
    if not words:
        return 0.0

    best = 0.0
    for line in (evidence or []):
        e_words = set(str(line).lower().split())
        if not e_words:
            continue
        overlap = words.intersection(e_words)
        if overlap:
            score = len(overlap) / max(1.0, len(words))
            if score > best:
                best = score
    return min(1.0, best)


def _score_factor(
    factor_weight: float,
    user_answer: Dict[str, Any],
    disease_factor: Dict[str, Any],
    scoring_config: Dict[str, float],
) -> Dict[str, Any]:
    answer_mode = user_answer.get("answer_mode", "unknown")
    user_tokens = _as_set(user_answer.get("normalized_tokens", []))
    disease_tokens = _as_set(disease_factor.get("normalized_tokens", []))
    conflict_tokens = _as_set(disease_factor.get("conflicting_tokens", []))
    confidence = float(user_answer.get("confidence", 0.0))

    if answer_mode == "unknown":
        return {
            "status": "unknown_skipped",
            "weight": factor_weight,
            "score": 0.0,
            "matched_tokens": [],
            "conflicting_tokens": [],
        }

    if not user_tokens:
        lexical = _lexical_overlap_score(user_answer.get("raw_answer", ""), disease_factor.get("evidence", []))
        # weak signal fallback when no normalized token is detected.
        contribution = factor_weight * confidence * (
            float(scoring_config["free_text_base"]) + float(scoring_config["free_text_lexical_gain"]) * lexical
        )
        return {
            "status": "free_text_weak",
            "weight": factor_weight,
            "score": round(contribution, 6),
            "matched_tokens": [],
            "conflicting_tokens": [],
            "lexical_overlap": round(lexical, 4),
        }

    matched = sorted(user_tokens.intersection(disease_tokens))
    conflicts = sorted(user_tokens.intersection(conflict_tokens))

    # Match ratio among provided tokens.
    match_ratio = len(matched) / max(1.0, len(user_tokens))
    conflict_ratio = len(conflicts) / max(1.0, len(user_tokens))

    # Base evidence score in [-1, 1]
    evidence_score = (
        float(scoring_config["match_gain"]) * match_ratio
        - float(scoring_config["conflict_penalty"]) * conflict_ratio
    )

    # If we have neither match nor conflict, treat as weak mismatch.
    if not matched and not conflicts:
        evidence_score = float(scoring_config["weak_mismatch_penalty"])

    contribution = factor_weight * confidence * evidence_score

    if conflicts and not matched:
        status = "conflict"
    elif matched and conflicts:
        status = "mixed"
    elif matched:
        status = "match"
    else:
        status = "weak_mismatch"

    return {
        "status": status,
        "weight": factor_weight,
        "score": round(contribution, 6),
        "matched_tokens": matched,
        "conflicting_tokens": conflicts,
        "match_ratio": round(match_ratio, 4),
        "conflict_ratio": round(conflict_ratio, 4),
    }


def compute_phase3_candidate_scores(
    state: Dict[str, Any],
    schema: Dict[str, Any],
    min_keep: int = DEFAULT_MIN_KEEP,
    max_keep: int = DEFAULT_MAX_KEEP,
    factor_weights_override: Dict[str, float] | None = None,
    scoring_config: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    answers: Dict[str, Dict[str, Any]] = state.get("answers", {})
    if not answers:
        return {
            "ranked_candidates": [],
            "active_candidate_ids": [],
            "uncertainty": "high",
            "score_spread": 0.0,
            "answered_factors": 0,
        }

    factor_weights: Dict[str, float] = dict(factor_weights_override or schema.get("factor_weights", {}))
    config = dict(DEFAULT_SCORING_CONFIG)
    if scoring_config:
        config.update(scoring_config)
    diseases = schema.get("diseases", [])

    ranked: List[Dict[str, Any]] = []
    answered_factor_count = len(answers)

    for disease in diseases:
        disease_id = disease.get("disease_id")
        disease_name = disease.get("disease_name")
        factor_data = disease.get("factors", {})

        total_score = 0.0
        trace: Dict[str, Any] = {}
        conflict_only_count = 0
        strong_match_count = 0

        for factor, user_answer in answers.items():
            weight = float(factor_weights.get(factor, 0.1))
            d_factor = factor_data.get(factor, {})
            f_res = _score_factor(weight, user_answer, d_factor, config)
            trace[factor] = f_res
            total_score += float(f_res["score"])

            if f_res["status"] == "conflict":
                conflict_only_count += 1
            if f_res["status"] == "match" and f_res.get("match_ratio", 0.0) >= 0.5:
                strong_match_count += 1

        # Eliminate only on repeated hard conflicts without compensating strong matches.
        eliminated = conflict_only_count >= int(config["elimination_conflict_threshold"]) and strong_match_count == 0

        ranked.append(
            {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "score": round(total_score, 6),
                "eliminated": eliminated,
                "conflict_only_count": conflict_only_count,
                "strong_match_count": strong_match_count,
                "trace": trace,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)

    not_eliminated = [r for r in ranked if not r["eliminated"]]
    pool = not_eliminated if not_eliminated else ranked

    top1 = pool[0]["score"] if pool else 0.0
    top3 = pool[min(2, len(pool) - 1)]["score"] if pool else 0.0
    spread = float(top1 - top3)

    # uncertainty and shortlist sizing
    if answered_factor_count <= 2 or spread < 0.05:
        uncertainty = "high"
        keep = min(max_keep, max(min_keep + 3, 6))
    elif spread < 0.12:
        uncertainty = "medium"
        keep = min(max_keep, max(min_keep + 1, 4))
    else:
        uncertainty = "low"
        keep = min(max_keep, max(min_keep, 3))

    active = pool[:keep]

    return {
        "ranked_candidates": ranked,
        "active_candidate_ids": [c["disease_id"] for c in active],
        "active_candidates": [
            {
                "disease_id": c["disease_id"],
                "disease_name": c["disease_name"],
                "score": c["score"],
            }
            for c in active
        ],
        "uncertainty": uncertainty,
        "score_spread": round(spread, 6),
        "answered_factors": answered_factor_count,
        "keep_count": keep,
        "scoring_config_used": config,
    }


def update_state_with_phase3(
    state: Dict[str, Any],
    schema: Dict[str, Any],
    min_keep: int = DEFAULT_MIN_KEEP,
    max_keep: int = DEFAULT_MAX_KEEP,
    factor_weights_override: Dict[str, float] | None = None,
    scoring_config: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    result = compute_phase3_candidate_scores(
        state,
        schema,
        min_keep=min_keep,
        max_keep=max_keep,
        factor_weights_override=factor_weights_override,
        scoring_config=scoring_config,
    )
    state["phase3"] = result
    return state
