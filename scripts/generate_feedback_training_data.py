#!/usr/bin/env python3
"""
generate_feedback_training_data.py
====================================
Two functions in one script:

1. generate_feedback_samples()
   - Reads the existing training datasets (upsc_gs_training_v2.json,
     css_training_v2.json, ai-mock-dataset.json)
   - Runs each candidate answer through feedback_generator.py
   - Writes a feedback training dataset: feedback_training_dataset.json
   - Each record pairs (transcript, metrics stub, question_data) → rich_feedback

2. test_feedback_on_sample()
   - Runs a quick end-to-end test against 3 representative samples
     (strong / partial / weak) and prints the output to stdout.

Usage
-----
    # Generate full dataset
    python scripts/generate_feedback_training_data.py --generate

    # Quick smoke-test only
    python scripts/generate_feedback_training_data.py --test

    # Both
    python scripts/generate_feedback_training_data.py --generate --test
"""

import argparse
import json
import os
import sys

# ── path setup (works when run from project root OR scripts/ folder) ───────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

DATA_DIR   = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR)

# ── source datasets ────────────────────────────────────────────────────────────
SOURCE_FILES = [
    os.path.join(DATA_DIR, "upsc_gs_training_v2.json"),
    os.path.join(DATA_DIR, "css_training_v2.json"),
    os.path.join(DATA_DIR, "ai-mocks", "ai-mock-dataset.json"),
]


# ── metric stub builder ────────────────────────────────────────────────────────
def _build_metrics_from_labels(item: dict) -> dict:
    """
    Construct a metrics dict that mirrors evaluate_answer() output,
    sourced from the pre-labelled dataset fields instead of running
    the full NLP pipeline (avoids sentence-transformer dependency during
    dataset generation).
    """
    labels = item.get("labels", {})
    score  = float(item.get("score", 0.5))

    return {
        "final_score":        score,
        "semantic_score":     labels.get("semantic_similarity",    0.5),
        "question_relevance": labels.get("question_relevance",     0.5),
        "keyword_score":      labels.get("keyword_coverage",       0.5),
        "overlap_score":      labels.get("keyword_coverage",       0.5),  # proxy
        "length_score":       labels.get("length_appropriateness", 0.5),
        "discourse_score":    labels.get("discourse_quality",      0.4),
        "lexical_diversity":  labels.get("lexical_diversity",      0.5),
        "penalty":            0.0,
        "matched_keywords":   [],
        "missing_keywords":   [],
        "model_score":        score,
        "heuristic_score":    score,
        "scoring_mode":       "label_stub",
        "feedback":           "",
        "tip":                "",
    }


def _normalise_question_data(raw_qd: dict) -> dict:
    """
    Ensure question_data has the fields feedback_generator.py expects,
    regardless of which dataset version it came from.
    """
    qd = dict(raw_qd)

    # Alias older field names
    if "answer" in qd and "primary_answer" not in qd:
        qd["primary_answer"] = qd["answer"]
    if "ideal_answer" in qd and "primary_answer" not in qd:
        qd["primary_answer"] = qd["ideal_answer"]
    if "expected_answer" in qd and "primary_answer" not in qd:
        qd["primary_answer"] = qd["expected_answer"]

    # Flatten answer_variants list → variant_1, variant_2
    variants = qd.get("answer_variants", [])
    if isinstance(variants, list):
        if len(variants) >= 1 and "answer_variant_1" not in qd:
            qd["answer_variant_1"] = variants[0]
        if len(variants) >= 2 and "answer_variant_2" not in qd:
            qd["answer_variant_2"] = variants[1]

    # Ensure question field
    if "question_text" in qd and "question" not in qd:
        qd["question"] = qd["question_text"]

    return qd


def load_all_samples() -> list:
    """Load and normalise samples from all source files."""
    all_samples = []
    for path in SOURCE_FILES:
        if not os.path.exists(path):
            print(f"  [SKIP] Not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [SKIP] Unexpected format in {path}")
            continue
        valid = [
            item for item in data
            if item.get("candidate") and item.get("question_data")
        ]
        print(f"  Loaded {len(valid)} valid samples from {os.path.basename(path)}")
        all_samples.extend(valid)
    return all_samples


# ── Main generators ────────────────────────────────────────────────────────────

def generate_feedback_samples(output_path: str = None) -> str:
    """
    Generate feedback_training_dataset.json.
    Returns the path written.
    """
    from services.feedback_generator import generate_question_feedback

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "feedback_training_dataset.json")

    print("\n" + "=" * 64)
    print("  Feedback Training Dataset Generator")
    print("=" * 64)

    samples = load_all_samples()
    if not samples:
        print("ERROR: No samples loaded. Check source file paths.")
        return ""

    records = []
    skipped = 0

    for i, item in enumerate(samples):
        try:
            transcript    = item["candidate"]
            raw_qd        = item.get("question_data", {})
            question_data = _normalise_question_data(raw_qd)
            metrics       = _build_metrics_from_labels(item)

            feedback = generate_question_feedback(
                transcript    = transcript,
                metrics       = metrics,
                question_data = question_data,
                question_id   = item.get("sample_id", str(i)),
                seed          = i,
            )

            records.append({
                "sample_id":    item.get("sample_id", f"fb_{i:05d}"),
                "module":       item.get("module", "unknown"),
                "score":        item.get("score"),
                "score_tier":   feedback["score_tier"],
                "transcript":   transcript,
                "question_data": question_data,
                "metrics_stub": metrics,
                "feedback":     feedback,
            })

        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  [WARN] Sample {i}: {exc}")

    print(f"\n  Generated : {len(records)} feedback records")
    print(f"  Skipped   : {skipped}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Saved  ->  {output_path}  ({os.path.getsize(output_path) // 1024} KB)")
    print("=" * 64 + "\n")
    return output_path


def test_feedback_on_sample():
    """
    Smoke-test: runs 3 hand-crafted examples through the generator and prints output.
    Tests strong / partial / weak tiers and STT-realistic input.
    """
    from services.feedback_generator import (
        generate_question_feedback,
        generate_session_summary,
    )

    TEST_CASES = [
        {
            "label": "STRONG — Clean written answer",
            "transcript": (
                "The Constitution is the supreme law of the land. "
                "Firstly, it defines the structure of government by separating powers "
                "among the legislature, executive, and judiciary. "
                "Secondly, it protects the fundamental rights of every citizen, such as "
                "the right to equality and freedom of speech. "
                "Finally, it distributes powers between the central government and the states "
                "through the federal structure. Without the Constitution, there would be no "
                "legal basis for governance or individual rights."
            ),
            "metrics": {
                "final_score": 0.91, "semantic_score": 0.88, "question_relevance": 0.90,
                "keyword_score": 0.82, "overlap_score": 0.79, "length_score": 0.85,
                "discourse_score": 0.72, "lexical_diversity": 0.76, "penalty": 0.0,
                "matched_keywords": ["constitution", "supreme", "law", "government"],
                "missing_keywords": [],
            },
            "question_data": {
                "id": "polity_q001",
                "question": "If you had to explain to a school student what the Constitution actually does, how would you put it?",
                "primary_answer": "The Constitution is the supreme law of the land that defines the structure of government, distributes powers, and protects the rights of citizens.",
                "answer_variants": [
                    "The Constitution is the supreme law that governs how the country is run.",
                    "It is the highest legal document establishing rights and government structure.",
                ],
                "expected_keywords": {
                    "critical":   ["constitution", "supreme law"],
                    "supporting": ["government structure", "rights", "powers"],
                    "bonus":      ["federal", "fundamental rights", "legislature"],
                },
            },
        },
        {
            "label": "PARTIAL — STT-realistic, incomplete",
            "transcript": (
                "um so the constitution is basically like the main law right "
                "and it uh it defines how the government works and um "
                "yeah it also has something to do with rights i think "
                "like uh citizens rights and stuff like that so yeah"
            ),
            "metrics": {
                "final_score": 0.48, "semantic_score": 0.52, "question_relevance": 0.58,
                "keyword_score": 0.30, "overlap_score": 0.28, "length_score": 0.42,
                "discourse_score": 0.15, "lexical_diversity": 0.48, "penalty": 0.0,
                "matched_keywords": ["constitution", "law", "government"],
                "missing_keywords": ["supreme", "powers", "structure"],
            },
            "question_data": {
                "id": "polity_q001",
                "question": "If you had to explain to a school student what the Constitution actually does, how would you put it?",
                "primary_answer": "The Constitution is the supreme law of the land that defines the structure of government, distributes powers, and protects the rights of citizens.",
                "answer_variants": [
                    "The Constitution is the supreme law that governs how the country is run.",
                    "It is the highest legal document establishing rights and government structure.",
                ],
                "expected_keywords": {
                    "critical":   ["constitution", "supreme law"],
                    "supporting": ["government structure", "rights", "powers"],
                    "bonus":      ["federal", "fundamental rights"],
                },
            },
        },
        {
            "label": "WEAK — Vague / off-topic",
            "transcript": "I think it is something about the government. I am not really sure about it.",
            "metrics": {
                "final_score": 0.15, "semantic_score": 0.22, "question_relevance": 0.28,
                "keyword_score": 0.08, "overlap_score": 0.06, "length_score": 0.18,
                "discourse_score": 0.00, "lexical_diversity": 0.55, "penalty": 0.10,
                "matched_keywords": ["government"],
                "missing_keywords": ["constitution", "supreme law", "rights", "powers", "structure"],
            },
            "question_data": {
                "id": "polity_q001",
                "question": "If you had to explain to a school student what the Constitution actually does, how would you put it?",
                "primary_answer": "The Constitution is the supreme law of the land that defines the structure of government, distributes powers, and protects the rights of citizens.",
                "answer_variants": [
                    "The Constitution is the supreme law that governs how the country is run.",
                ],
                "expected_keywords": {
                    "critical":   ["constitution", "supreme law"],
                    "supporting": ["government structure", "rights", "powers"],
                    "bonus":      ["federal", "fundamental rights"],
                },
            },
        },
    ]

    divider = "-" * 64
    all_feedbacks = []

    for idx, tc in enumerate(TEST_CASES):
        print(f"\n{'=' * 64}")
        print(f"  TEST CASE {idx + 1}: {tc['label']}")
        print("=" * 64)

        fb = generate_question_feedback(
            transcript    = tc["transcript"],
            metrics       = tc["metrics"],
            question_data = tc["question_data"],
            seed          = idx,
        )
        all_feedbacks.append(fb)

        print(f"\n  Score      : {fb['score']:.2f}  [{fb['score_tier'].upper()}]")
        print(f"\n  Narrative")
        print(f"  {divider}")
        # Wrap at ~70 chars for terminal readability
        words = fb["narrative"].split()
        line, lines = [], []
        for w in words:
            if sum(len(x)+1 for x in line) + len(w) > 68:
                lines.append("  " + " ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append("  " + " ".join(line))
        print("\n".join(lines))

        print(f"\n  Improvement Tips")
        print(f"  {divider}")
        for i, tip in enumerate(fb["improvement_tips"], 1):
            print(f"  {i}. {tip}")

        print(f"\n  Grammar Notes  [{fb['grammar_notes']['severity'].upper()}]")
        print(f"  {divider}")
        g = fb["grammar_notes"]
        print(f"  Filler words   : {g['filler_count']} ({g['filler_rate']}%)  → {g['filler_words_found']}")
        print(f"  Repetitions    : {g['word_repetitions']}")
        print(f"  Sentences      : {g['sentence_count']}  avg {g['avg_words_per_sent']} words/sent")
        print(f"  Run-on         : {g['run_on_detected']}")

        print(f"\n  Content Analysis")
        print(f"  {divider}")
        ca = fb["content_analysis"]
        print(f"  Critical present   : {ca['present_keywords']['critical']}")
        print(f"  Critical MISSING   : {ca['missing_keywords']['critical']}")
        print(f"  Supporting present : {ca['present_keywords']['supporting']}")
        print(f"  Supporting MISSING : {ca['missing_keywords']['supporting']}")
        print(f"  Concepts covered   : {ca['concepts_covered']}")
        print(f"  Concepts MISSING   : {ca['concepts_missing']}")

        if fb["stt_flags"]:
            print(f"\n  STT Flags")
            print(f"  {divider}")
            for flag in fb["stt_flags"]:
                print(f"  ⚠  {flag}")

    # Session summary
    print(f"\n{'=' * 64}")
    print("  SESSION SUMMARY")
    print("=" * 64)
    summary = generate_session_summary(all_feedbacks)

    print(f"\n  Session Score  : {summary['session_score']:.2f}  [{summary['score_tier'].upper()}]")
    print(f"  Trend          : {summary['trend_direction']}  {summary['score_trend']}")

    print(f"\n  Summary Paragraph")
    print(f"  {divider}")
    words = summary["summary_paragraph"].split()
    line, lines = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > 68:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    print(f"\n  Strengths")
    for s in summary["strengths"]:
        print(f"  ✓ {s}")

    print(f"\n  Improvement Areas")
    for a in summary["improvement_areas"]:
        print(f"  → {a}")

    print(f"\n  Grammar Summary  [{summary['grammar_summary'].get('grammar_tier', 'n/a').upper()}]")
    gs = summary["grammar_summary"]
    print(f"  Total fillers : {gs.get('total_filler_words')}  ({gs.get('overall_filler_rate')}% rate)")
    print(f"  Top fillers   : {gs.get('most_used_fillers')}")
    print(f"  Run-ons       : {gs.get('run_on_sentences')}")

    print(f"\n  Per-Question Scores")
    for pq in summary["per_question_scores"]:
        print(f"  [{pq['score_tier']:7s}] {pq['score']:.2f}  {pq['question_text'][:60]}")

    print("\n" + "=" * 64)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedback generator tools")
    parser.add_argument("--generate", action="store_true", help="Generate feedback training dataset")
    parser.add_argument("--test",     action="store_true", help="Run smoke-test on 3 sample answers")
    parser.add_argument("--output",   default=None,        help="Override output path for --generate")
    args = parser.parse_args()

    if not args.generate and not args.test:
        parser.print_help()
        sys.exit(0)

    if args.test:
        test_feedback_on_sample()

    if args.generate:
        generate_feedback_samples(args.output)
