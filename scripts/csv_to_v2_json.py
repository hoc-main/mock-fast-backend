"""
csv_to_v2_json.py
=================
Convert researcher-friendly CSV → v2 dataset JSON

Usage:
    python csv_to_v2_json.py --input data.csv --output output.json --module ml_interview
"""

import argparse
import csv
import json
import re


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def detect_stt(text: str) -> bool:
    text = (text or "").lower()
    fillers = ["um", "uh", "like", "you know", "basically"]
    return any(f in text for f in fillers)


def split_keywords(keyword_str: str):
    words = [k.strip() for k in (keyword_str or "").split(",") if k.strip()]
    return {
        "critical": words[:2],
        "supporting": words[2:4],
        "bonus": words[4:]
    }


def compute_labels(quality: str, answer_type: str):
    base_map = {
        "strong": 0.9,
        "good": 0.75,
        "partial": 0.55,
        "weak": 0.2,
        "off_topic": 0.3
    }

    base = base_map.get((quality or "").lower(), 0.3)

    labels = {
        "semantic_similarity": base,
        "question_relevance": base - 0.05,
        "keyword_coverage": base - 0.15,
        "length_appropriateness": base,
        "discourse_quality": base - 0.1,
        "lexical_diversity": 0.7,
        "factual_accuracy": base + 0.1,
        "answer_completeness": base - 0.1
    }

    if (answer_type or "").lower() == "stt":
        labels["discourse_quality"] -= 0.2

    # clamp 0–1
    return {k: round(max(0, min(1, v)), 3) for k, v in labels.items()}


def compute_score(labels):
    return round(
        labels["semantic_similarity"] * 0.5 +
        labels["question_relevance"] * 0.15 +
        labels["keyword_coverage"] * 0.18 +
        labels["length_appropriateness"] * 0.07 +
        labels["discourse_quality"] * 0.05 +
        labels["lexical_diversity"] * 0.05,
        3
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CSV → v2 JSON converter")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--module", "-m", required=True, help="Module name (e.g., ml_interview)")

    args = parser.parse_args()

    output = []

    with open(args.input, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            labels = compute_labels(row.get("answer_quality"), row.get("answer_type"))
            score = compute_score(labels)

            item = {
                "sample_id": f"{args.module}_{i+1:04d}",
                "schema_version": "2.0",
                "module": args.module,
                "split": "train",

                "question_data": {
                    "id": f"q_{i+1}",
                    "topic": row.get("topic"),
                    "question": row.get("question"),
                    "primary_answer": row.get("ideal_answer"),
                    "answer_variants": [
                        row.get("answer_variant_1"),
                        row.get("answer_variant_2")
                    ],
                    "expected_keywords": split_keywords(row.get("keywords")),
                    "scoring_weights": {
                        "semantic": 0.5,
                        "question_relevance": 0.15,
                        "keyword": 0.18,
                        "length": 0.07,
                        "discourse": 0.05,
                        "lexical": 0.05
                    },
                    "ideal_length_words": len((row.get("ideal_answer") or "").split())
                },

                "candidate": row.get("candidate_answer"),
                "stt_realistic": detect_stt(row.get("candidate_answer")),
                "labels": labels,
                "score": score,

                "scoring_rationale": f"Auto-generated from {row.get('answer_quality')} quality and {row.get('answer_type')} type."
            }

            output.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(output)} samples → {args.output}")


if __name__ == "__main__":
    main()


'''
python csv_to_v2_json.py --input data.csv --output output.json --module ml_interview
'''