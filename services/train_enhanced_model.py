"""
train_enhanced_model.py
========================
Django management command — trains the 17-feature NLP evaluation model.

Accepts both dataset formats with a single --dataset flag:
  • CSV  (.csv)  — the human-friendly spreadsheet format
  • JSON (.json) — the old upsc_gs_training_dataset.json format

CSV columns (exact spelling, as defined in DATASET_SCHEMA):
    topic, question, ideal_answer, answer_variant_1, answer_variant_2,
    candidate_answer, answer_type, answer_quality, keywords

JSON items must have:
    candidate, question_data {question, ideal_answer|answer, …}, score

Usage
-----
    python manage.py train_enhanced_model
    python manage.py train_enhanced_model --dataset data/my_sheet.csv
    python manage.py train_enhanced_model --dataset data/upsc_gs_training_dataset.json
    python manage.py train_enhanced_model --dataset data/my_sheet.csv \
        --output interview/services/my_module_model.pkl
    python manage.py train_enhanced_model --dataset data/my_sheet.csv --no-cv
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
import argparse
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
SERVICES_DIR = "services"
MODEL_DIR = "pickles"
DATA_DIR     = "data"

DEFAULT_DATASET = str(DATA_DIR + "/upsc_gs_training_dataset.json")
DEFAULT_OUTPUT  = str(MODEL_DIR + "/enhanced_eval_model.pkl")

# ── score derivation matrix (CSV only) ────────────────────────────────────────
# Keys: (answer_quality, answer_type)  ->  float score in [0, 1]
#
# answer_quality: strong | good | partial | weak | off_topic
# answer_type:    correct | concise | verbose | off_topic | refusal | stt
#
# answer_quality drives the base score (correctness / completeness).
# answer_type is a modifier:
#   correct   no penalty  (clean written answer)
#   stt       small penalty for disfluency, content intact
#   concise   penalised for incompleteness even if correct
#   verbose   penalised for low lexical density and repetition
#   off_topic heavy penalty regardless of quality label
#   refusal   always very low regardless of quality label

SCORE_MATRIX = {
    ("strong",    "correct"):    0.92,
    ("strong",    "stt"):        0.85,
    ("strong",    "concise"):    0.80,
    ("strong",    "verbose"):    0.75,
    ("strong",    "off_topic"):  0.30,
    ("strong",    "refusal"):    0.08,

    ("good",      "correct"):    0.76,
    ("good",      "stt"):        0.70,
    ("good",      "concise"):    0.65,
    ("good",      "verbose"):    0.62,
    ("good",      "off_topic"):  0.28,
    ("good",      "refusal"):    0.08,

    ("partial",   "correct"):    0.50,
    ("partial",   "stt"):        0.46,
    ("partial",   "concise"):    0.40,
    ("partial",   "verbose"):    0.42,
    ("partial",   "off_topic"):  0.24,
    ("partial",   "refusal"):    0.10,

    ("weak",      "correct"):    0.20,
    ("weak",      "stt"):        0.18,
    ("weak",      "concise"):    0.14,
    ("weak",      "verbose"):    0.16,
    ("weak",      "off_topic"):  0.12,
    ("weak",      "refusal"):    0.08,

    ("off_topic", "correct"):    0.28,
    ("off_topic", "stt"):        0.25,
    ("off_topic", "concise"):    0.22,
    ("off_topic", "verbose"):    0.24,
    ("off_topic", "off_topic"):  0.20,
    ("off_topic", "refusal"):    0.08,
}

QUALITY_DEFAULTS = {
    "strong": 0.85, "good": 0.70, "partial": 0.46,
    "weak": 0.16, "off_topic": 0.25,
}


def _derive_score(quality: str, ans_type: str) -> float:
    q = (quality  or "").strip().lower()
    t = (ans_type or "").strip().lower()
    if (q, t) in SCORE_MATRIX:
        return SCORE_MATRIX[(q, t)]
    return QUALITY_DEFAULTS.get(q, 0.40)


# ── CSV loader ─────────────────────────────────────────────────────────────────
REQUIRED_CSV_COLS = {
    "topic", "question", "ideal_answer",
    "answer_variant_1", "answer_variant_2",
    "candidate_answer", "answer_type", "answer_quality", "keywords",
}


def _load_csv(path: str) -> list:
    """
    Load a CSV dataset and return internal training dicts.
    question_data uses field names that nlp_features.py already handles.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        actual  = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLS - actual
        if missing:
            raise ValueError(
                f"CSV missing columns: {sorted(missing)}\n"
                f"  Found: {sorted(actual)}"
            )

        for row in reader:
            candidate = (row.get("candidate_answer") or "").strip()
            if not candidate:
                continue

            quality  = (row.get("answer_quality") or "").strip().lower()
            ans_type = (row.get("answer_type")    or "").strip().lower()
            score    = _derive_score(quality, ans_type)

            question_data = {
                # primary + variants — resolved by nlp_features._resolve_*
                "ideal_answer":     (row.get("ideal_answer")     or "").strip(),
                "answer_variant_1": (row.get("answer_variant_1") or "").strip(),
                "answer_variant_2": (row.get("answer_variant_2") or "").strip(),
                # comma-separated string — parsed by nlp_features._parse_keywords
                "keywords":         (row.get("keywords")         or "").strip(),
                # question text for question_relevance feature
                "question":         (row.get("question")         or "").strip(),
                # metadata (not used by feature extractor)
                "topic":            (row.get("topic")            or "").strip(),
                "_answer_quality":  quality,
                "_answer_type":     ans_type,
            }

            rows.append({
                "candidate":     candidate,
                "question_data": question_data,
                "score":         score,
            })
    return rows


# ── JSON loader ────────────────────────────────────────────────────────────────
def _load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        candidate     = item.get("candidate", "")
        score         = item.get("score")
        question_data = item.get("question_data", {})
        if not candidate or score is None:
            continue
        # normalise legacy field name
        if "answer" in question_data and "ideal_answer" not in question_data:
            question_data = dict(question_data, ideal_answer=question_data["answer"])
        rows.append({
            "candidate":     candidate,
            "question_data": question_data,
            "score":         float(score),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Train the 17-feature NLP evaluation model"
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to training dataset (.csv or .json)"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to save the trained .pkl model"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip 5-fold cross-validation"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Validation split (default: 0.20)"
    )

    args = parser.parse_args()

    dataset_path = args.dataset
    output_path  = args.output
    run_cv       = not args.no_cv
    val_split    = args.val_split

    # Replace Django stdout with print
    def W(msg=""):
        print(msg)

    W("\n" + "=" * 64)
    W("  Enhanced NLP Evaluation Model - Training")
    W("=" * 64)


    W("\n" + "=" * 64)
    W("  Enhanced NLP Evaluation Model - Training")
    W("=" * 64)

    # ── 1. load ────────────────────────────────────────────────────────────
    W(f"\n> Loading: {dataset_path}")
    if not os.path.exists(dataset_path):
        W(f"  ERROR: File not found: {dataset_path}")
        sys.exit(1)

    ext = Path(dataset_path).suffix.lower()
    try:
        if ext == ".csv":
            W("  Format: CSV")
            samples = _load_csv(dataset_path)
            W(f"  Rows loaded: {len(samples)}")
            from collections import Counter
            combos = Counter(
                (s["question_data"].get("_answer_quality", "?"),
                    s["question_data"].get("_answer_type",    "?"))
                for s in samples
            )
            W("  Quality x Type -> score:")
            for (q, t), n in sorted(combos.items()):
                W(f"    {q:10s} + {t:10s}  n={n:4d}  -> {_derive_score(q,t):.2f}")
        elif ext == ".json":
            W("  Format: JSON")
            samples = _load_json(dataset_path)
            W(f"  Items loaded: {len(samples)}")
        else:
            W(f"  ERROR: Unsupported type '{ext}'. Use .csv or .json")
            sys.exit(1)
    except ValueError as e:
        W(f"  ERROR: {e}")
        sys.exit(1)

    if len(samples) < 20:
        W("  ERROR: Need at least 20 samples.")
        sys.exit(1)

    # ── 2. feature extraction ──────────────────────────────────────────────
    W("\n> Extracting features (encodes sentences - takes ~60s for 800 rows)...")
    from nlp_features import (
        extract_features_enhanced, FEATURE_NAMES, N_FEATURES,
    )

    X, y, skipped = [], [], 0
    t0 = time.time()

    for i, item in enumerate(samples):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (len(samples) - i)
            W(f"  {i}/{len(samples)}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")
        try:
            feats, _ = extract_features_enhanced(
                item["candidate"], item["question_data"]
            )
            if len(feats) != N_FEATURES:
                skipped += 1
                continue
            X.append(feats)
            y.append(item["score"])
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                W(f"  Warning sample {i}: {e}")

    X = np.array(X)
    y = np.array(y)
    W(f"  Done in {time.time()-t0:.1f}s  |  {len(X)} usable, {skipped} skipped")
    W(f"  Feature shape: {X.shape}")
    W(f"  Score range:   {y.min():.3f} - {y.max():.3f}   mean: {y.mean():.3f}   stdev: {y.std():.3f}")

    if len(X) < 20:
        W("  ERROR: Too few usable samples.")
        sys.exit(1)

    # ── 3. split ───────────────────────────────────────────────────────────
    W(f"\n> Split: {int((1-val_split)*100)}% train / {int(val_split*100)}% val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )
    W(f"  Train: {len(X_train)}   Val: {len(X_val)}")

    # ── 4. model ───────────────────────────────────────────────────────────
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3,
            validation_fraction=0.1, n_iter_no_change=30,
            random_state=42,
        )),
    ])

    # ── 5. train ───────────────────────────────────────────────────────────
    W("\n> Training GradientBoostingRegressor...")
    t1 = time.time()
    model.fit(X_train, y_train)
    n_trees = model.named_steps["gbr"].n_estimators_
    W(f"  Done in {time.time()-t1:.1f}s  |  Trees: {n_trees} (early-stop from 400)")

    # ── 6. metrics ─────────────────────────────────────────────────────────
    W("\n> Validation metrics")
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    W(f"  Train  MAE: {mean_absolute_error(y_train, train_preds):.4f}   R2: {r2_score(y_train, train_preds):.4f}")
    W(f"  Val    MAE: {mean_absolute_error(y_val,   val_preds):.4f}   R2: {r2_score(y_val,   val_preds):.4f}")

    gap = mean_absolute_error(y_val, val_preds) - mean_absolute_error(y_train, train_preds)
    W(f"  {'WARNING: overfit gap' if gap > 0.05 else 'OK: overfit gap'} {gap:.3f}")

    W("\n  Per-tier breakdown (val):")
    for lo, hi, label in [(0,0.3,"weak"),(0.3,0.6,"partial"),(0.6,0.85,"good"),(0.85,1.01,"strong")]:
        mask = (y_val >= lo) & (y_val < hi)
        if mask.sum() > 0:
            W(f"    {label:8s}  n={mask.sum():3d}   MAE={mean_absolute_error(y_val[mask], val_preds[mask]):.4f}")

    # ── 7. cross-validation ────────────────────────────────────────────────
    if run_cv:
        W("\n> 5-fold cross-validation")
        cv_maes, cv_r2s = [], []
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            cv_m = Pipeline([
                ("scaler", StandardScaler()),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=n_trees, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=3, random_state=42,
                )),
            ])
            cv_m.fit(X[tr_idx], y[tr_idx])
            preds = cv_m.predict(X[va_idx])
            fold_mae = mean_absolute_error(y[va_idx], preds)
            fold_r2  = r2_score(y[va_idx], preds)
            cv_maes.append(fold_mae)
            cv_r2s.append(fold_r2)
            W(f"  Fold {fold+1}: MAE {fold_mae:.4f}   R2 {fold_r2:.4f}")
        W(f"  Mean MAE: {np.mean(cv_maes):.4f} +/- {np.std(cv_maes):.4f}")
        W(f"  Mean R2:  {np.mean(cv_r2s):.4f} +/- {np.std(cv_r2s):.4f}")

    # ── 8. feature importances ─────────────────────────────────────────────
    W("\n> Feature importances")
    for imp, name in sorted(zip(model.named_steps["gbr"].feature_importances_, FEATURE_NAMES), reverse=True):
        W(f"  {name:30s} {imp:.4f}  {'|' * int(imp * 40)}")

    # ── 9. save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)
    W(f"\n> Saved -> {output_path}  ({os.path.getsize(output_path)/1024:.1f} KB)")
    W("=" * 64 + "\n")

    

if __name__ == "__main__":
    main()