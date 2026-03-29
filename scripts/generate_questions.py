"""
generate_questions.py
======================
Extract unique questions from v2 dataset JSON and generate a clean questions JSON
ready for DB insertion.

Usage:
    python generate_questions.py input.json --module_id 1 --output questions.json
"""

import argparse
import json
import hashlib


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def hash_question(text: str) -> str:
    return hashlib.md5(normalize(text).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────

def extract_questions(data, module_id):
    seen = set()
    questions = []

    order = 1

    for idx, item in enumerate(data):
        qd = item.get("question_data", {})

        question_text = (qd.get("question") or "").strip()
        expected_answer = (qd.get("primary_answer") or "").strip()
        topic = qd.get("topic", None)

        # Skip invalid rows
        if not question_text or not expected_answer:
            continue

        # Deduplicate
        q_hash = hash_question(question_text)
        if q_hash in seen:
            continue

        seen.add(q_hash)

        questions.append({
            "module_id": module_id,
            "topic": topic,
            "question_text": question_text,
            "expected_answer": expected_answer,
            "order": order
        })

        order += 1

    return questions


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate unique questions JSON from v2 dataset"
    )

    parser.add_argument(
        "input",
        help="Path to v2 dataset JSON file"
    )

    parser.add_argument(
        "--module_id",
        "-m",
        type=int,
        required=True,
        help="Module ID to assign to all questions"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="questions.json",
        help="Output JSON file (default: questions.json)"
    )

    args = parser.parse_args()

    # Load dataset
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ ERROR: Input must be a JSON array")
        return

    questions = extract_questions(data, args.module_id)

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(questions)} unique questions")
    print(f"📁 Output saved to: {args.output}")


if __name__ == "__main__":
    main()