"""
train_enhanced_model.py
========================
Standalone script to train the enhanced 16-feature NLP evaluation model.

Usage:
    python train_enhanced_model.py
    python train_enhanced_model.py --dataset data/upsc_gs_training_dataset.json
    python train_enhanced_model.py --output models/model.pkl
    python train_enhanced_model.py --no-cv
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_DATASET = str(DATA_DIR / "upsc_gs_training_dataset.json")
DEFAULT_OUTPUT  = str(BASE_DIR / "enhanced_eval_model.pkl")


def log(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train enhanced NLP model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--module-id", type=int, help="Train for a specific module from DB")
    parser.add_argument("--val-split", type=float, default=0.2)

    args = parser.parse_args()

    module_id    = args.module_id
    dataset_path = args.dataset
    output_path  = args.output
    run_cv       = not args.no_cv
    val_split    = args.val_split

    if module_id:
        log(f"▸ Fetching config for module_id={module_id} from database...")
        import asyncio
        from sqlalchemy import select
        from db.database import AsyncSessionLocal
        from db.models import Module
        
        async def fetch_module_paths(mid):
            async with AsyncSessionLocal() as db:
                res = await db.execute(select(Module).where(Module.id == mid))
                return res.scalar_one_or_none()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            module = loop.run_until_complete(fetch_module_paths(module_id))
        finally:
            loop.close()
        
        if not module:
            log(f"ERROR: Module {module_id} not found in database.")
            sys.exit(1)
            
        if module.dataset_json_path:
            dataset_path = module.dataset_json_path
        if module.model_pkl_path:
            output_path = module.model_pkl_path
            
        log(f"  Dataset path: {dataset_path}")
        log(f"  Model output: {output_path}")

    log("\n" + "═" * 60)
    log("  Enhanced NLP Evaluation Model — Training")
    log("═" * 60)

    # ── 1. load data ───────────────────────────────────────────────
    log(f"\n▸ Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        log(f"ERROR: File not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    log(f"Loaded {len(raw)} samples")

    # ── 2. feature extraction ──────────────────────────────────────
    log("\n▸ Extracting features...")

    # IMPORTANT: adjust import path if needed
    from nlp_features import extract_features_enhanced, FEATURE_NAMES, N_FEATURES

    X, y, skipped = [], [], 0
    t0 = time.time()

    for i, item in enumerate(raw):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(raw) - i)
            log(f"{i}/{len(raw)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        candidate     = item.get("candidate", "")
        question_data = item.get("question_data", {})
        score         = item.get("score")

        if not candidate or score is None:
            skipped += 1
            continue

        if "question" not in question_data and "Interview Question" in question_data:
            question_data = {
                "question": question_data.get("Interview Question", ""),
                "answer":   question_data.get("Answer", ""),
            }

        try:
            features, _ = extract_features_enhanced(candidate, question_data)
            if len(features) != N_FEATURES:
                skipped += 1
                continue
            X.append(features)
            y.append(float(score))
        except Exception as e:
            skipped += 1
            log(f"Warning: skipped sample {i} — {e}")

    X = np.array(X)
    y = np.array(y)

    log(f"\nDone. {len(X)} usable samples, {skipped} skipped")
    log(f"Shape: {X.shape} | Score range: {y.min():.2f}–{y.max():.2f}")

    if len(X) < 20:
        log("ERROR: Need at least 20 samples")
        sys.exit(1)

    # ── 3. split ───────────────────────────────────────────────────
    log(f"\n▸ Train/Val split ({int((1-val_split)*100)}/{int(val_split*100)})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )

    # ── 4. model ───────────────────────────────────────────────────
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])

    # ── 5. train ───────────────────────────────────────────────────
    log("\n▸ Training...")
    t1 = time.time()
    model.fit(X_train, y_train)
    log(f"Training done in {time.time() - t1:.1f}s")

    # ── 6. eval ────────────────────────────────────────────────────
    log("\n▸ Evaluation")

    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae   = mean_absolute_error(y_val, val_preds)
    train_r2  = r2_score(y_train, train_preds)
    val_r2    = r2_score(y_val, val_preds)

    log(f"Train MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    log(f"Val   MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    # ── 7. CV ──────────────────────────────────────────────────────
    if run_cv:
        log("\n▸ 5-Fold CV")

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        maes, r2s = [], []

        for i, (tr, va) in enumerate(kf.split(X)):
            model.fit(X[tr], y[tr])
            preds = model.predict(X[va])

            mae = mean_absolute_error(y[va], preds)
            r2  = r2_score(y[va], preds)

            maes.append(mae)
            r2s.append(r2)

            log(f"Fold {i+1}: MAE={mae:.4f}, R²={r2:.4f}")

        log(f"Mean MAE: {np.mean(maes):.4f}")
        log(f"Mean R² : {np.mean(r2s):.4f}")

    # ── 8. feature importance ──────────────────────────────────────
    log("\n▸ Feature Importance")

    importances = model.named_steps["gbr"].feature_importances_
    ranked = sorted(zip(importances, FEATURE_NAMES), reverse=True)

    for imp, name in ranked:
        bar = "█" * int(imp * 40)
        log(f"{name:25s} {imp:.4f} {bar}")

    # ── 9. save ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)

    log(f"\nModel saved → {output_path}")


if __name__ == "__main__":
    main()