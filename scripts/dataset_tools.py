"""
dataset_tools.py
=================
Two tools in one file:

  1. VALIDATOR  — checks a dataset file against the v2 schema and reports all issues
  2. MIGRATOR   — upgrades a v1 dataset (fixed 4-bin scores) to v2 format

Usage
-----
# Validate a dataset:
    python dataset_tools.py validate data/training_dataset_v2_example.json

# Migrate old v1 dataset to v2 (generates programmatic labels, flags manual ones):
    python dataset_tools.py migrate data/upsc_gs_training_dataset.json --output data/upsc_gs_training_v2.json

# Full report with per-field stats:
    python dataset_tools.py validate data/upsc_gs_training_v2.json --verbose
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ── schema constants ───────────────────────────────────────────────────────────
REQUIRED_TOP_LEVEL = {"sample_id", "schema_version", "module", "split", "question_data", "candidate", "stt_realistic", "labels", "score"}
REQUIRED_QUESTION_DATA = {"id", "question", "primary_answer", "answer_variants", "expected_keywords", "scoring_weights", "ideal_length_words"}
REQUIRED_LABELS = {"semantic_similarity", "question_relevance", "keyword_coverage", "length_appropriateness", "discourse_quality", "lexical_diversity", "factual_accuracy", "answer_completeness"}
VALID_SPLITS = {"train", "val", "test"}
VALID_KEYWORD_TIERS = {"critical", "supporting", "bonus"}
VALID_WEIGHT_KEYS = {"semantic", "question_relevance", "keyword", "length", "discourse", "lexical"}
LABEL_RANGE = (0.0, 1.0)
SCORE_RANGE = (0.0, 1.0)
MIN_ANSWER_VARIANTS = 2
MAX_DIFFICULTY = 5
MIN_DIFFICULTY = 1

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "that", "this", "it", "as", "by", "be", "at",
    "from", "can", "into", "about", "what", "which", "who", "whom", "when",
    "where", "why", "how", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "its", "will",
    "would", "should", "may", "might", "could", "do", "did", "does",
    "have", "has", "had", "been", "being", "also", "very", "just", "but",
    "if", "so", "than", "then", "not", "no",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return [w for w in _normalize(text).split() if len(w) > 2 and w not in STOPWORDS]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class Issue:
    def __init__(self, sample_id: str, severity: str, field: str, message: str):
        self.sample_id = sample_id
        self.severity  = severity   # "error" | "warning" | "info"
        self.field     = field
        self.message   = message

    def __str__(self):
        sid = str(self.sample_id)
        return f"[{self.severity.upper():7s}] {sid:40s} {self.field:35s} {self.message}"


def validate_sample(item: Dict[str, Any], idx: int) -> List[Issue]:
    issues = []
    sid = item.get("sample_id", f"index_{idx}")

    def err(field, msg):  issues.append(Issue(sid, "error",   field, msg))
    def warn(field, msg): issues.append(Issue(sid, "warning", field, msg))
    def info(field, msg): issues.append(Issue(sid, "info",    field, msg))

    # ── top-level required fields ──
    for key in REQUIRED_TOP_LEVEL:
        if key not in item:
            err(key, f"Missing required field '{key}'")

    # ── schema_version ──
    if item.get("schema_version") != "2.0":
        warn("schema_version", f"Expected '2.0', got '{item.get('schema_version')}'. Run migrate to upgrade.")

    # ── split ──
    if item.get("split") not in VALID_SPLITS:
        err("split", f"Must be one of {VALID_SPLITS}, got '{item.get('split')}'")

    # ── score ──
    score = item.get("score")
    if score is not None:
        if not isinstance(score, (int, float)):
            err("score", "Must be a number")
        elif not (SCORE_RANGE[0] <= float(score) <= SCORE_RANGE[1]):
            err("score", f"Out of range [0,1]: {score}")
        # Warn if score looks like a v1 hard-bin value
        if float(score) in {0.12, 0.46, 0.78, 0.95}:
            warn("score", f"Score {score} matches a v1 hard-bin value. Consider using a continuous score.")

    # ── stt_realistic ──
    if not isinstance(item.get("stt_realistic"), bool):
        warn("stt_realistic", "Should be a boolean (true/false)")

    # ── scoring_rationale ──
    rationale = item.get("scoring_rationale", "")
    if not rationale:
        warn("scoring_rationale", "Missing scoring_rationale — harder to audit or fine-tune later")
    elif len(rationale.split()) < 10:
        warn("scoring_rationale", f"Rationale is very short ({len(rationale.split())} words). Aim for 1-2 full sentences.")

    # ── candidate ──
    candidate = item.get("candidate", "")
    if not candidate or not candidate.strip():
        err("candidate", "Empty candidate answer")
    elif len(candidate.split()) < 3:
        warn("candidate", f"Very short candidate ({len(candidate.split())} words) — intentional?")

    # ── labels ──
    labels = item.get("labels", {})
    if not isinstance(labels, dict):
        err("labels", "Must be a JSON object")
    else:
        for key in REQUIRED_LABELS:
            if key not in labels:
                err(f"labels.{key}", f"Missing required label '{key}'")
            else:
                v = labels[key]
                if not isinstance(v, (int, float)):
                    err(f"labels.{key}", f"Must be a number, got {type(v).__name__}")
                elif not (LABEL_RANGE[0] <= float(v) <= LABEL_RANGE[1]):
                    err(f"labels.{key}", f"Out of range [0,1]: {v}")

        # Consistency checks
        if "semantic_similarity" in labels and "question_relevance" in labels:
            ss = float(labels.get("semantic_similarity", 0))
            qr = float(labels.get("question_relevance", 0))
            if ss > 0.85 and qr < 0.40:
                warn("labels", f"High semantic_similarity ({ss}) but low question_relevance ({qr}). Is this an on-topic but wrong-question answer?")

        if score is not None and "semantic_similarity" in labels:
            ss = float(labels.get("semantic_similarity", 0))
            if ss > 0.90 and float(score) < 0.50:
                warn("labels+score", f"Very high semantic_similarity ({ss}) but score {score} < 0.5. Likely inconsistent.")
            if ss < 0.25 and float(score) > 0.70:
                warn("labels+score", f"Very low semantic_similarity ({ss}) but score {score} > 0.7. Likely inconsistent.")

    # ── question_data ──
    qd = item.get("question_data", {})
    if not isinstance(qd, dict):
        err("question_data", "Must be a JSON object")
        return issues

    for key in REQUIRED_QUESTION_DATA:
        if key not in qd:
            err(f"question_data.{key}", f"Missing required field '{key}'")

    # answer_variants
    variants = qd.get("answer_variants", [])
    if not isinstance(variants, list):
        err("question_data.answer_variants", "Must be a JSON array")
    elif len(variants) < MIN_ANSWER_VARIANTS:
        warn("question_data.answer_variants", f"Only {len(variants)} variant(s). Recommend ≥ {MIN_ANSWER_VARIANTS} for robust semantic matching.")

    # primary_answer
    pa = qd.get("primary_answer", "")
    if len((pa or "").split()) < 10:
        warn("question_data.primary_answer", f"Very short primary answer ({len(pa.split())} words). Semantic score will be less reliable.")

    # expected_keywords
    kw = qd.get("expected_keywords", {})
    if isinstance(kw, list):
        warn("question_data.expected_keywords", "Still using v1 flat list. Upgrade to tiered object with 'critical', 'supporting', 'bonus' keys.")
    elif isinstance(kw, dict):
        for tier in VALID_KEYWORD_TIERS:
            if tier not in kw:
                warn(f"question_data.expected_keywords.{tier}", f"Missing '{tier}' tier. Recommend specifying all three tiers.")
        if "critical" in kw and len(kw["critical"]) == 0:
            warn("question_data.expected_keywords.critical", "No critical keywords defined. Keyword scoring will not penalise missing concepts.")
    else:
        err("question_data.expected_keywords", "Must be an object with 'critical', 'supporting', 'bonus' keys")

    # scoring_weights
    sw = qd.get("scoring_weights", {})
    if isinstance(sw, dict):
        for key in VALID_WEIGHT_KEYS:
            if key not in sw:
                warn(f"question_data.scoring_weights.{key}", f"Missing weight '{key}'. Will use global default.")
        total = sum(v for v in sw.values() if isinstance(v, (int, float)))
        if abs(total - 1.0) > 0.05:
            warn("question_data.scoring_weights", f"Weights sum to {total:.3f}, expected 1.0 ± 0.05")

    # ideal_length_words
    ilw = qd.get("ideal_length_words")
    if ilw is not None:
        if not isinstance(ilw, int) or ilw < 5:
            warn("question_data.ideal_length_words", f"Should be a positive integer ≥ 5, got {ilw}")

    # difficulty
    if "difficulty" in qd:
        d = qd.get("difficulty")
        if not isinstance(d, int) or not (MIN_DIFFICULTY <= d <= MAX_DIFFICULTY):
            warn("question_data.difficulty", f"Should be int in [{MIN_DIFFICULTY},{MAX_DIFFICULTY}], got {d}")

    # STT consistency
    if item.get("stt_realistic") is True:
        norm_cand = _normalize(candidate)
        filler_words = ["um", "uh", "like", "basically", "you know", "right", "so so", "and and"]
        has_filler = any(f in norm_cand for f in filler_words)
        if not has_filler:
            info("candidate+stt_realistic", "stt_realistic=true but no filler words detected. Add spoken artifacts or set stt_realistic=false.")

    return issues


def validate_dataset(data: List[Dict], verbose: bool = False) -> Tuple[int, int, int]:
    """Validate all samples. Returns (errors, warnings, infos)."""
    all_issues = []
    sample_ids = []

    for idx, item in enumerate(data):
        issues = validate_sample(item, idx)
        all_issues.extend(issues)
        sample_ids.append(item.get("sample_id", f"index_{idx}"))

    # Dataset-level checks
    sid_counts = defaultdict(int)
    for sid in sample_ids:
        sid_counts[sid] += 1
    for sid, count in sid_counts.items():
        if count > 1:
            all_issues.append(Issue(sid, "error", "sample_id", f"Duplicate sample_id appears {count} times"))

    splits = defaultdict(int)
    for item in data:
        splits[item.get("split", "?")] += 1

    if not all_issues:
        print(f"✓  All {len(data)} samples passed validation.")
    else:
        errors   = [i for i in all_issues if i.severity == "error"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        infos    = [i for i in all_issues if i.severity == "info"]

        print(f"\n{'═'*100}")
        print(f"  Validation report: {len(data)} samples  |  {len(errors)} errors  |  {len(warnings)} warnings  |  {len(infos)} info")
        print(f"{'═'*100}")

        if errors:
            print("\n── ERRORS (must fix before training) ──")
            for issue in errors:
                print(f"  {issue}")

        if warnings:
            print("\n── WARNINGS (recommended to fix) ──")
            for issue in warnings:
                print(f"  {issue}")

        if infos and verbose:
            print("\n── INFO ──")
            for issue in infos:
                print(f"  {issue}")

    if verbose:
        print(f"\n── Dataset stats ──")
        print(f"  Total samples:  {len(data)}")
        for split, count in splits.items():
            print(f"  {split}: {count} ({100*count//len(data)}%)")

        scores = [item.get("score", 0) for item in data if isinstance(item.get("score"), (int, float))]
        if scores:
            import statistics
            print(f"  Score range:    {min(scores):.2f} – {max(scores):.2f}")
            print(f"  Score mean:     {statistics.mean(scores):.3f}")
            print(f"  Score stdev:    {statistics.stdev(scores):.3f}")

        stt_count = sum(1 for item in data if item.get("stt_realistic") is True)
        print(f"  STT-realistic:  {stt_count} samples ({100*stt_count//len(data)}%)")

        topics = defaultdict(int)
        for item in data:
            t = item.get("question_data", {}).get("topic", "?")
            topics[t] += 1
        print(f"  Topics: {dict(sorted(topics.items(), key=lambda x: -x[1]))}")

    return (
        sum(1 for i in all_issues if i.severity == "error"),
        sum(1 for i in all_issues if i.severity == "warning"),
        sum(1 for i in all_issues if i.severity == "info"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MIGRATOR  (v1 → v2)
# ══════════════════════════════════════════════════════════════════════════════

V1_SCORE_TO_QUALITY = {0.95: "strong_complete", 0.78: "good_partial_keywords", 0.46: "partial_correct", 0.12: "weak_vague"}

V1_QUALITY_LABELS = {
    "strong": {
        "semantic_similarity": 0.88, "question_relevance": 0.90, "keyword_coverage": 0.82,
        "length_appropriateness": 0.88, "discourse_quality": 0.72, "lexical_diversity": 0.76,
        "factual_accuracy": 0.95, "answer_completeness": 0.85,
    },
    "good": {
        "semantic_similarity": 0.72, "question_relevance": 0.78, "keyword_coverage": 0.60,
        "length_appropriateness": 0.75, "discourse_quality": 0.48, "lexical_diversity": 0.65,
        "factual_accuracy": 0.85, "answer_completeness": 0.62,
    },
    "partial": {
        "semantic_similarity": 0.52, "question_relevance": 0.62, "keyword_coverage": 0.35,
        "length_appropriateness": 0.60, "discourse_quality": 0.28, "lexical_diversity": 0.58,
        "factual_accuracy": 0.75, "answer_completeness": 0.40,
    },
    "weak": {
        "semantic_similarity": 0.12, "question_relevance": 0.18, "keyword_coverage": 0.05,
        "length_appropriateness": 0.20, "discourse_quality": 0.05, "lexical_diversity": 0.42,
        "factual_accuracy": 0.80, "answer_completeness": 0.05,
    },
}

def _upgrade_keywords(flat_list: list, answer_text: str) -> Dict[str, list]:
    """Convert a flat keyword list to tiered. Critical = first 40%, supporting = rest, bonus = empty."""
    if not flat_list:
        toks = _tokenize(answer_text)
        seen = []
        for w in toks:
            if w not in seen:
                seen.append(w)
        flat_list = seen[:12]

    split_at = max(2, len(flat_list) // 3)
    return {
        "critical": flat_list[:split_at],
        "supporting": flat_list[split_at:split_at * 2],
        "bonus": flat_list[split_at * 2:],
    }


def _upgrade_weights(v1_weights: Dict) -> Dict:
    """Add question_relevance and lexical to v1 weights object."""
    if not v1_weights:
        return {"semantic": 0.50, "question_relevance": 0.15, "keyword": 0.18, "length": 0.07, "discourse": 0.05, "lexical": 0.05}
    w = dict(v1_weights)
    # v1 had: semantic, keyword, length (sums to 1.0 usually)
    # Redistribute to add question_relevance + discourse + lexical
    total = sum(w.get(k, 0) for k in ["semantic", "keyword", "length"])
    if total > 0:
        scale = 0.80 / total   # free up 20% for new dims
        w["semantic"]           = round(w.get("semantic", 0.6) * scale, 2)
        w["keyword"]            = round(w.get("keyword", 0.3)  * scale, 2)
        w["length"]             = round(w.get("length", 0.1)   * scale, 2)
    w["question_relevance"] = 0.15
    w["discourse"]          = 0.03
    w["lexical"]            = 0.02
    return w


def migrate_v1_to_v2(data: List[Dict], module: str = "unknown") -> List[Dict]:
    """
    Convert v1 training samples to v2 format.
    Labels are approximated from quality tier — human review of question_relevance
    and discourse_quality labels is strongly recommended afterward.
    """
    v2 = []
    for idx, item in enumerate(data):
        quality   = item.get("quality", "partial")
        v1_score  = float(item.get("score", 0.5))
        candidate = item.get("candidate", "")
        qd_raw    = item.get("question_data", {})

        # Upgrade question_data
        v1_kw = qd_raw.get("expected_keywords", [])
        v1_weights = qd_raw.get("weights", {})
        answer = qd_raw.get("answer", "")

        question_data = {
            "id":                 str(qd_raw.get("id", f"q{idx+1}")),
            "module":             qd_raw.get("module", module),
            "topic":              qd_raw.get("topic", "general"),
            "subtopic":           qd_raw.get("subtopic", ""),
            "difficulty":         qd_raw.get("difficulty", 2),
            "question":           qd_raw.get("question", ""),
            "primary_answer":     answer,
            "answer_variants":    qd_raw.get("answer_variants", [answer]),
            "expected_keywords":  _upgrade_keywords(v1_kw, answer),
            "scoring_weights":    _upgrade_weights(v1_weights),
            "ideal_length_words": qd_raw.get("ideal_length_words", 40),
            "domain_tags":        qd_raw.get("domain_tags", [qd_raw.get("topic", "general")]),
        }

        # Approximate labels from quality tier
        labels = dict(V1_QUALITY_LABELS.get(quality, V1_QUALITY_LABELS["partial"]))

        # Detect filler words for STT flag
        norm = _normalize(candidate)
        filler = ["um", "uh", "basically", "you know", "sort of"]
        stt_realistic = any(f in norm for f in filler)

        # Adjust score: v1 bins → keep value but flag it
        is_binned = v1_score in {0.12, 0.46, 0.78, 0.95}

        v2_item = {
            "sample_id":         item.get("sample_id", f"{module}_{idx+1:04d}_migrated"),
            "schema_version":    "2.0",
            "module":            item.get("module", module),
            "split":             "train",   # reassign properly if you have a split strategy
            "question_data":     question_data,
            "candidate":         candidate,
            "stt_realistic":     stt_realistic,
            "labels":            labels,
            "score":             v1_score,
            "scoring_rationale": (
                f"[MIGRATED from v1 — {quality} tier] "
                f"Labels approximated from quality label. "
                f"{'Score is a v1 hard-bin value — replace with continuous label. ' if is_binned else ''}"
                f"question_relevance and discourse_quality labels require human review."
            ),
            "_migration_notes": {
                "original_quality":  quality,
                "labels_are_approximate": True,
                "needs_review":      ["question_relevance", "discourse_quality", "score"],
            }
        }
        v2.append(v2_item)

    return v2


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dataset tools: validate or migrate training data.")
    sub = parser.add_subparsers(dest="command")

    val_p = sub.add_parser("validate", help="Validate a v2 dataset file")
    val_p.add_argument("file", help="Path to JSON dataset file")
    val_p.add_argument("--verbose", "-v", action="store_true", help="Show dataset statistics and info-level messages")

    mig_p = sub.add_parser("migrate", help="Migrate a v1 dataset to v2 format")
    mig_p.add_argument("file", help="Path to v1 JSON dataset file")
    mig_p.add_argument("--output", "-o", required=True, help="Output path for v2 file")
    mig_p.add_argument("--module", default="unknown", help="Module slug to use for migrated items")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: Dataset must be a JSON array at the top level.")
        sys.exit(1)

    if args.command == "validate":
        errors, warnings, infos = validate_dataset(data, verbose=args.verbose)
        print(f"\n{'─'*60}")
        print(f"  Result: {errors} error(s), {warnings} warning(s), {infos} info message(s)")
        sys.exit(1 if errors > 0 else 0)

    elif args.command == "migrate":
        print(f"Migrating {len(data)} samples from v1 → v2 …")
        v2_data = migrate_v1_to_v2(data, module=args.module)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(v2_data, f, indent=2, ensure_ascii=False)
        print(f"✓  Written {len(v2_data)} samples to {args.output}")
        print()
        print("Next steps:")
        print("  1. python dataset_tools.py validate", args.output, "--verbose")
        print("  2. Open the output file and manually review fields marked with 'needs_review':")
        print("     - score          (replace hard-bin values with continuous labels)")
        print("     - question_relevance  (was the candidate actually answering THIS question?)")
        print("     - discourse_quality   (was the answer well-structured?)")
        print("  3. Add 2+ answer_variants to each question_data if missing")
        print("  4. Remove _migration_notes before final training")
        print(f"  5. python manage.py train_enhanced_model --dataset {args.output}")


if __name__ == "__main__":
    main()
