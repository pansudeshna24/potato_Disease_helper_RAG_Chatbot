"""
Phase 01 Canonical Disease Schema Builder

Builds a normalized, machine-readable disease schema from data/COT/potato_disease_symptoms.txt.

Output:
    data/COT/canonical_disease_schema_phase1.json

Usage:
    python -m src.cot.build_phase1_canonical_schema
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

INPUT_PATH = os.path.join(project_root, "data", "COT", "potato_disease_symptoms.txt")
OUTPUT_PATH = os.path.join(project_root, "data", "COT", "canonical_disease_schema_phase1.json")

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

# Weights sum to 1.0 and can be tuned later with evaluation data.
FACTOR_WEIGHTS = {
    "general_appearance": 0.14,
    "wilting": 0.18,
    "growth_habit": 0.12,
    "stems": 0.14,
    "leaves": 0.14,
    "tubers": 0.14,
    "roots": 0.10,
    "period": 0.04,
}

RAW_FIELD_TO_FACTOR = {
    "stunted poor growth": "general_appearance",
    "bronzing": "general_appearance",
    "deformed": "growth_habit",
    "upright growth": "growth_habit",
    "wilt": "wilting",
    "stems": "stems",
    "with chlorosis": "leaves",
    "chlorosis absent to mild": "leaves",
    "normal tuber shape surface unblemished": "tubers",
    "normal tuber shape surface blemished without actve rot": "tubers",
    "tubers with active rot": "tubers",
    "seed tubers": "tubers",
    "stolons": "tubers",
    "roots": "roots",
    "period": "period",
}

CONTROLLED_VOCAB = {
    "general_appearance": [
        "stunted_patchy",
        "stunted_general",
        "normal_growth",
        "poor_vigor",
        "bronzing",
        "chlorosis_general",
    ],
    "wilting": [
        "wilting_present",
        "wilting_one_sided",
        "no_wilting",
        "wilting_late_season",
    ],
    "growth_habit": [
        "upright_erect",
        "rosette_bushy",
        "spindly",
        "leaf_curling",
        "deformed_growth",
        "normal_habit",
    ],
    "stems": [
        "vascular_discoloration",
        "black_lesions_stem",
        "ooze_present",
        "stem_rot",
        "stem_galls",
        "stem_necrosis",
        "stem_normal",
    ],
    "leaves": [
        "chlorosis",
        "interveinal_chlorosis",
        "leaf_roll_upward",
        "necrotic_spots",
        "concentric_rings",
        "rugosity_crinkling",
        "leaf_bronzing",
        "leaf_mosaic",
        "leaf_normal",
    ],
    "tubers": [
        "vascular_discoloration",
        "stolon_end_lesions",
        "cracks_pitted_lesions",
        "warts_galls",
        "active_rot",
        "eyes_discolored",
        "sclerotia_present",
        "tuber_normal",
    ],
    "roots": [
        "root_galls",
        "root_rot",
        "root_poor_development",
        "root_necrosis",
        "root_cysts",
        "root_normal",
    ],
    "period": [
        "early_stage",
        "mid_season",
        "late_season",
        "temperature_moisture_trigger",
    ],
}

# keyword list maps to normalized tokens by factor
TOKEN_PATTERNS: Dict[str, List[Tuple[str, List[str]]]] = {
    "general_appearance": [
        ("stunted_patchy", ["patch", "localized"]),
        ("stunted_general", ["stunted", "stunting"]),
        ("normal_growth", ["normal"]),
        ("poor_vigor", ["poor growth", "weak", "spindly"]),
        ("bronzing", ["bronzing", "bronzed"]),
        ("chlorosis_general", ["chlorosis", "pale", "yellow"]),
    ],
    "wilting": [
        ("wilting_one_sided", ["one side", "one sided"]),
        ("no_wilting", ["no wilting"]),
        ("wilting_late_season", ["late in the season", "after mid season"]),
        ("wilting_present", ["wilt", "wilting"]),
    ],
    "growth_habit": [
        ("upright_erect", ["upright", "erect"]),
        ("rosette_bushy", ["rosette", "bushy"]),
        ("spindly", ["spindly"]),
        ("leaf_curling", ["rolling", "rolled", "curl"]),
        ("deformed_growth", ["deformed", "distortion", "shortened internodes"]),
        ("normal_habit", ["normal"]),
    ],
    "stems": [
        ("vascular_discoloration", ["vascular discolouration", "vascular discoloration"]),
        ("black_lesions_stem", ["black", "dark", "lesion"]),
        ("ooze_present", ["ooze"]),
        ("stem_rot", ["rot", "rotting", "decay"]),
        ("stem_galls", ["galls", "warty"]),
        ("stem_necrosis", ["necrosis", "necrotic"]),
        ("stem_normal", ["normal"]),
    ],
    "leaves": [
        ("leaf_mosaic", ["mosaic", "mottle"]),
        ("interveinal_chlorosis", ["interveinal chlorosis"]),
        ("chlorosis", ["chlorosis", "pale", "yellow"]),
        ("leaf_roll_upward", ["upward rolling", "rolled upward", "leaf roll"]),
        ("necrotic_spots", ["necrotic", "spot", "lesion"]),
        ("concentric_rings", ["concentric", "target"]),
        ("rugosity_crinkling", ["rugosity", "crinkled"]),
        ("leaf_bronzing", ["bronzing", "bronzed"]),
        ("leaf_normal", ["normal"]),
    ],
    "tubers": [
        ("vascular_discoloration", ["vascular discolouration", "vascular discoloration"]),
        ("stolon_end_lesions", ["stolon", "stolon end"]),
        ("cracks_pitted_lesions", ["crack", "pitted", "sunken", "lesion"]),
        ("warts_galls", ["warty", "galls"]),
        ("active_rot", ["active rot", "soft rot", "watery"]),
        ("eyes_discolored", ["eyes discoloured", "eyes discolored"]),
        ("sclerotia_present", ["sclerotia"]),
        ("tuber_normal", ["normal"]),
    ],
    "roots": [
        ("root_galls", ["galls"]),
        ("root_rot", ["rot", "rotting"]),
        ("root_poor_development", ["poorly developed", "stunted", "poor development"]),
        ("root_necrosis", ["necrosis", "necrotic"]),
        ("root_cysts", ["cysts"]),
        ("root_normal", ["normal"]),
    ],
    "period": [
        ("early_stage", ["early", "at planting", "early stage"]),
        ("mid_season", ["mid season"]),
        ("late_season", ["late season", "late in the season"]),
        ("temperature_moisture_trigger", ["temperature", "humidity", "wet", "dry", "soil"]),
    ],
}

TOKEN_CONFLICTS = {
    "wilting_present": ["no_wilting"],
    "no_wilting": ["wilting_present", "wilting_one_sided", "wilting_late_season"],
    "normal_growth": ["stunted_general", "stunted_patchy", "poor_vigor"],
    "stunted_general": ["normal_growth"],
    "stunted_patchy": ["normal_growth"],
    "normal_habit": ["deformed_growth", "spindly", "rosette_bushy"],
    "stem_normal": ["stem_rot", "stem_necrosis", "black_lesions_stem", "ooze_present", "stem_galls"],
    "leaf_normal": ["chlorosis", "interveinal_chlorosis", "necrotic_spots", "leaf_roll_upward", "rugosity_crinkling", "leaf_mosaic"],
    "tuber_normal": ["active_rot", "cracks_pitted_lesions", "warts_galls", "stolon_end_lesions", "eyes_discolored"],
    "root_normal": ["root_rot", "root_galls", "root_necrosis", "root_cysts", "root_poor_development"],
}

SPECIFICITY_HINTS = {
    "wilting": ["one side", "no wilting", "wilt", "wilting"],
    "stems": ["ooze", "vascular", "galls", "warty", "black"],
    "leaves": ["concentric", "mosaic", "rugosity", "interveinal"],
    "tubers": ["active rot", "sclerotia", "stolon", "eyes discoloured", "eyes discolored"],
    "roots": ["galls", "cysts", "root rot", "necrosis"],
    "general_appearance": ["stunted", "patch"],
    "growth_habit": ["rosette", "bushy", "spindly", "upright"],
    "period": ["early", "mid", "late", "temperature", "humidity"],
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def to_key(text: str) -> str:
    return normalize_spaces(text).lower()


def parse_disease_symptom_file(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("--------------------------------------------------") if c.strip()]
    diseases: List[Dict] = []

    for chunk in chunks:
        lines = [normalize_spaces(line) for line in chunk.splitlines() if normalize_spaces(line)]
        if not lines:
            continue

        disease_name = None
        raw_fields: Dict[str, List[str]] = {}

        for line in lines:
            if line.lower().startswith("disease:"):
                disease_name = normalize_spaces(line.split(":", 1)[1])
                continue

            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            field_name = to_key(left)
            value = normalize_spaces(right)
            if not value:
                continue
            raw_fields.setdefault(field_name, []).append(value)

        if not disease_name:
            continue

        diseases.append({
            "disease_name": disease_name,
            "raw_fields": raw_fields,
        })

    return diseases


def extract_tokens(factor: str, texts: List[str]) -> List[str]:
    joined = " ".join([to_key(t) for t in texts])
    tokens: List[str] = []
    for token, keywords in TOKEN_PATTERNS.get(factor, []):
        if any(k in joined for k in keywords):
            tokens.append(token)
    return sorted(set(tokens))


def infer_clue_strength(factor: str, texts: List[str], tokens: List[str]) -> str:
    if not texts:
        return "unknown"

    joined = " ".join([to_key(t) for t in texts])
    hints = SPECIFICITY_HINTS.get(factor, [])

    if len(tokens) >= 2:
        return "required"
    if any(h in joined for h in hints):
        return "required"
    return "supportive"


def build_disease_profile(index: int, disease_name: str, raw_fields: Dict[str, List[str]]) -> Dict:
    factor_evidence: Dict[str, List[str]] = {f: [] for f in FACTOR_SEQUENCE}
    abiotic_evidence: List[str] = []

    for raw_field, values in raw_fields.items():
        if raw_field == "abiotic factors":
            abiotic_evidence.extend(values)
            # abiotic context contributes to period clues as weak evidence.
            factor_evidence["period"].extend(values)
            continue

        factor = RAW_FIELD_TO_FACTOR.get(raw_field)
        if factor:
            factor_evidence[factor].extend(values)

    factors: Dict[str, Dict] = {}
    for factor in FACTOR_SEQUENCE:
        evidence = sorted(set([normalize_spaces(v) for v in factor_evidence[factor] if normalize_spaces(v)]))
        tokens = extract_tokens(factor, evidence)
        conflicts = sorted(set([c for t in tokens for c in TOKEN_CONFLICTS.get(t, [])]))

        factors[factor] = {
            "evidence": evidence,
            "normalized_tokens": tokens,
            "clue_strength": infer_clue_strength(factor, evidence, tokens),
            "conflicting_tokens": conflicts,
        }

    return {
        "disease_id": str(index),
        "disease_name": disease_name,
        "normalized_name": to_key(disease_name),
        "factors": factors,
        "abiotic_context": sorted(set(abiotic_evidence)),
    }


def build_synonym_dictionary() -> Dict[str, Dict[str, List[str]]]:
    synonym_dictionary: Dict[str, Dict[str, List[str]]] = {}
    for factor, token_patterns in TOKEN_PATTERNS.items():
        synonym_dictionary[factor] = {}
        for token, keywords in token_patterns:
            synonym_dictionary[factor][token] = sorted(set(keywords))
    return synonym_dictionary


def build_phase1_schema() -> Dict:
    parsed = parse_disease_symptom_file(INPUT_PATH)

    disease_profiles = []
    for idx, item in enumerate(parsed, start=1):
        profile = build_disease_profile(idx, item["disease_name"], item["raw_fields"])
        disease_profiles.append(profile)

    return {
        "schema_version": "phase1.v1",
        "phase": "Phase 01 - Canonical schema foundation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "primary": "data/COT/potato_disease_symptoms.txt",
            "notes": [
                "Eight-factor canonical structure generated for deterministic diagnosis flow.",
                "Controlled vocabulary and synonym dictionary are included for normalization.",
                "Clue strength is heuristic and should be calibrated with evaluation data in later phases.",
            ],
        },
        "factor_sequence": FACTOR_SEQUENCE,
        "factor_weights": FACTOR_WEIGHTS,
        "controlled_vocab": CONTROLLED_VOCAB,
        "synonym_dictionary": build_synonym_dictionary(),
        "diseases": disease_profiles,
        "quality_flags": {
            "disease_count": len(disease_profiles),
            "factors_per_disease": len(FACTOR_SEQUENCE),
            "contains_conflicting_clues": True,
        },
    }


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    schema = build_phase1_schema()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print("=" * 68)
    print("Phase 01 canonical schema generated")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Diseases: {schema['quality_flags']['disease_count']}")
    print(f"Factors per disease: {schema['quality_flags']['factors_per_disease']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
