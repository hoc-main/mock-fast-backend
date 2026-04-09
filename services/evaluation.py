"""
evaluation.py
=============
Answer evaluation service. Produces numeric metrics via nlp_features
and rich textual feedback via feedback_generator.

Public API (unchanged for views/transcription compatibility):
    evaluate_answer(transcript, question_data, model_path=None) -> dict
    classify_confirmation_intent(text) -> str
    train_evaluation_model(...)       -> dict
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .nlp_features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features_enhanced,
    soft_rule_score_enhanced,
    tokenize,
    normalize,
    WEAK_PATTERNS,
)
from .feedback_generator import generate_question_feedback

from ..db.models import Module

logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR    = os.path.join(BACKEND_DIR, "data")

MODEL_PATH             = os.path.join(BASE_DIR, "enhanced_eval_model.pkl")
LEGACY_MODEL_PATH      = os.path.join(BASE_DIR, "upsc_gs_eval_model.pkl")
DEFAULT_TRAINING_DATASET = os.path.join(DATA_DIR, "upsc_gs_training_dataset.json")


# ── model loader ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=10)
def get_trained_model(model_path: str = None):
    candidates = []
    if model_path:
        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            candidates.append(os.path.join(BACKEND_DIR, model_path.lstrip("/")))
            candidates.append(os.path.join(BASE_DIR, os.path.basename(model_path)))
    candidates.append(MODEL_PATH)
    candidates.append(LEGACY_MODEL_PATH)

    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Loaded evaluation model from: {path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
    return None


def _detect_legacy_model(model) -> bool:
    if model is None:
        return False
    if hasattr(model, "named_steps"):
        return False
    if hasattr(model, "n_features_in_") and model.n_features_in_ == 9:
        return True
    return False


# ── backwards-compat alias ─────────────────────────────────────────────────────

def extract_features(transcript: str, question_data: Dict[str, Any]):
    return extract_features_enhanced(transcript, question_data)


# ── main evaluation ────────────────────────────────────────────────────────────

async def evaluate_answer_for_module(
    transcript: str,
    question_data: Dict[str, Any],
    module_id: int,
    db: Any,
) -> Dict[str, Any]:
    from sqlalchemy import select
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()
    model_path = module.model_pkl_path if module else None
    return evaluate_answer(transcript, question_data, model_path)


def evaluate_answer(
    transcript: str,
    question_data: Dict[str, Any],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a candidate answer.

    Returns a dict with numeric metrics AND rich textual feedback.
    The feedback/tip fields come from feedback_generator.generate_question_feedback().
    """
    # Auto-extract keywords if absent
    if "expected_keywords" not in question_data:
        truth = question_data.get("answer", "")
        tokens = tokenize(truth)
        seen: list = []
        for w in tokens:
            if w not in seen:
                seen.append(w)
        question_data = dict(question_data, expected_keywords=seen[:12])

    # ── feature extraction ─────────────────────────────────────────────────────
    features, meta = extract_features_enhanced(transcript, question_data)

    # ── heuristic score ────────────────────────────────────────────────────────
    heuristic_pred = soft_rule_score_enhanced(
        s_score=meta["semantic_score"],
        q_relevance=meta["question_relevance"],
        k_score=meta["keyword_score"],
        overlap=meta["overlap_score"],
        l_score=meta["length_score"],
        penalty=meta["penalty"],
        disc=meta["discourse_score"],
        lex_div=meta["lexical_diversity"],
    )

    # ── model score ────────────────────────────────────────────────────────────
    trained_model = get_trained_model(model_path)
    model_pred    = 0.0
    scoring_mode  = "heuristic_only"

    if trained_model is not None:
        if _detect_legacy_model(trained_model):
            model_pred   = float(np.clip(trained_model.predict([features[:9]])[0], 0.0, 1.0))
            scoring_mode = "legacy_rf_blended"
        else:
            model_pred   = float(np.clip(trained_model.predict([features])[0], 0.0, 1.0))
            scoring_mode = "enhanced_gbr_blended"
        final_score = 0.65 * model_pred + 0.35 * heuristic_pred
    else:
        final_score = heuristic_pred

    # Floor boosts for clearly good answers
    if meta["semantic_score"] >= 0.80 and meta["length_score"] >= 0.45:
        final_score = max(final_score, 0.68)
    if meta["semantic_score"] >= 0.86 and meta["length_score"] >= 0.55:
        final_score = max(final_score, 0.78)

    final_score = float(np.clip(final_score, 0.0, 1.0))

    # ── rich feedback via feedback_generator ───────────────────────────────────
    # Build the metrics dict that generate_question_feedback expects
    feedback_metrics = {
        "final_score":        final_score,
        "semantic_score":     meta["semantic_score"],
        "keyword_score":      meta["keyword_score"],
        "question_relevance": meta["question_relevance"],
        "lexical_diversity":  meta["lexical_diversity"],
        "discourse_score":    meta["discourse_score"],
        "length_score":       meta["length_score"],
        "missing_keywords":   meta["missing_keywords"],
        "matched_keywords":   meta["matched_keywords"],
    }

    # Build question_data dict — supports both answer and ideal_answer field names
    fb_question_data = {
        "id":             question_data.get("id", ""),
        "question":       question_data.get("question", ""),
        "answer":         question_data.get("answer", meta["truth"]),
        "primary_answer": question_data.get("answer", meta["truth"]),
        "expected_keywords": question_data.get("expected_keywords", []),
    }

    fb_payload = generate_question_feedback(
        transcript=transcript,
        metrics=feedback_metrics,
        question_data=fb_question_data,
        question_id=str(question_data.get("id", "")),
    )

    # Map new feedback_generator keys → fields expected by transcription.py + DB
    # feedback_generator now returns: narrative, improvement_tips, score_tier, ...
    feedback = fb_payload.get("narrative", "")
    tips_list = fb_payload.get("improvement_tips", [])
    tip = "\n".join(f"- {item}" for item in tips_list)

    return {
        "scoring_mode":        scoring_mode,
        "semantic_score":      meta["semantic_score"],
        "question_relevance":  meta["question_relevance"],
        "keyword_score":       meta["keyword_score"],
        "overlap_score":       meta["overlap_score"],
        "length_score":        meta["length_score"],
        "lexical_diversity":   meta["lexical_diversity"],
        "discourse_score":     meta["discourse_score"],
        "penalty":             meta["penalty"],
        "model_score":         round(model_pred,     4),
        "heuristic_score":     round(heuristic_pred, 4),
        "final_score":         round(final_score,    4),
        "feedback":            feedback,
        "tip":                 tip,
        # Extra fields for frontend / summary endpoints
        "score_tier":          fb_payload.get("score_tier", ""),
        "improvement_tips":    tips_list,
        "grammar_notes":       fb_payload.get("grammar_notes", {}),
        "stt_flags":           fb_payload.get("stt_flags", []),
        "matched_keywords":    meta["matched_keywords"],
        "missing_keywords":    meta["missing_keywords"],
    }


# ── intent classifier (kept for backward compat — voice_intent is preferred) ──

def classify_confirmation_intent(text: str) -> str:
    norm = normalize(text)
    if any(t in norm for t in ["end", "stop", "finish interview", "terminate"]):
        return "end"
    if any(t in norm for t in ["wait", "continue", "not done", "more", "hold on", "one more", "let me continue"]):
        return "continue"
    if any(t in norm for t in ["repeat", "again", "say again", "what was the question"]):
        return "repeat"
    if any(t in norm for t in ["done", "next", "move", "move on", "yes", "continue to next", "finished"]):
        return "next"
    return "next"


# ── training (run standalone, not from request path) ──────────────────────────

def train_evaluation_model(
    training_samples=None,
    dataset_path=DEFAULT_TRAINING_DATASET,
    target_model_path=MODEL_PATH,
) -> Dict[str, Any]:
    """Kept for backwards compatibility. Prefer: python services/train_enhanced_model.py"""
    if training_samples is None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            training_samples = json.load(f)
    if not training_samples:
        raise ValueError("No training samples provided.")

    X, y = [], []
    for item in training_samples:
        try:
            feats, _ = extract_features_enhanced(item["candidate"], item["question_data"])
            X.append(feats)
            y.append(float(item["score"]))
        except Exception:
            pass

    X, y = np.array(X), np.array(y)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3,
            validation_fraction=0.1, n_iter_no_change=30,
            random_state=42,
        )),
    ])
    model.fit(X, y)
    joblib.dump(model, target_model_path)
    get_trained_model.cache_clear()

    mae = float(np.mean(np.abs(model.predict(X) - y)))
    return {
        "status":       "trained",
        "samples_used": len(X),
        "model_path":   target_model_path,
        "train_mae":    round(mae, 4),
        "features":     N_FEATURES,
    }


if __name__ == "__main__":
    print(train_evaluation_model())
