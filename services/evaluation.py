"""
evaluation.py (updated)
========================
Evaluation service for mock-interview answers.

Changes vs original
-------------------
- extract_features()      → delegates to nlp_features.extract_features_enhanced() (16 features)
- soft_rule_score()       → delegates to nlp_features.soft_rule_score_enhanced()
- get_trained_model()     → per-module path support preserved, cache cleared after training
- evaluate_answer()       → cold-start training removed from request path (startup only)
- No other public API changes — views.py requires zero edits.
"""

import logging
import json
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

BASE_DIR    = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR    = os.path.join(BACKEND_DIR, "data")

from ..db.models import Module

logger = logging.getLogger(__name__)

MODEL_PATH        = os.path.join(BASE_DIR, "enhanced_eval_model.pkl")
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "upsc_gs_eval_model.pkl")
DEFAULT_TRAINING_DATASET = os.path.join(DATA_DIR, "upsc_gs_training_dataset.json")


@lru_cache(maxsize=10)
def get_trained_model(model_path: str = None):
    candidates = []
    if model_path:
        # Handle both absolute and relative paths (relative to BACKEND_DIR)
        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            candidates.append(os.path.join(BACKEND_DIR, model_path.lstrip("/")))
            # Also try relative to current file if that fails
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
                continue
    return None


def _detect_legacy_model(model) -> bool:
    if model is None:
        return False
    if hasattr(model, "named_steps"):
        # Enhanced model is a Pipeline
        return False
    if hasattr(model, "n_features_in_") and model.n_features_in_ == 9:
        return True
    return False


def extract_features(transcript: str, question_data: Dict[str, Any]):
    """Backwards-compatible alias."""
    return extract_features_enhanced(transcript, question_data)


def classify_feedback(final_score: float, missing_keywords: List[str], transcript: str) -> str:
    short = len(tokenize(transcript)) < 5
    if final_score >= 0.85:
        return "Strong answer. You covered the concept clearly and well."
    if final_score >= 0.70:
        if missing_keywords:
            return f"Good answer overall. You covered the main idea, but you could still mention points like: {', '.join(missing_keywords[:3])}."
        return "Good answer overall. It is relevant and mostly clear."
    if final_score >= 0.50:
        if short:
            return "Relevant answer, but it is a bit short. Add a clearer definition and one or two supporting points."
        return "Partially correct answer. You have the right direction, but it needs more completeness."
    if short:
        return "The answer is too short or too vague to show clear understanding."
    return "The answer is weak or incomplete and misses important parts of the concept."


def generate_tip(transcript, truth, missing_keywords, semantic_score_value, length_score_value, question_relevance=0.0):
    candidate_tokens = tokenize(transcript)
    truth_tokens = tokenize(truth)
    candidate_norm = normalize(transcript)

    if not candidate_tokens:
        return "Start by giving a direct one-line definition of the concept, then add two important points."
    if candidate_norm in WEAK_PATTERNS or any(p in candidate_norm for p in WEAK_PATTERNS):
        return "Avoid saying only that you are unsure. Begin with whatever you know, then add one key point."
    if question_relevance < 0.30:
        return "Your answer does not seem to address this specific question. Re-read it carefully and anchor your answer to what is being asked."
    if length_score_value < 0.45:
        return "Your answer is too short. Start with a one-line definition, then add 2-3 important points and one short example."
    if semantic_score_value < 0.40:
        return "Your answer is not close enough to the expected concept. Focus first on defining the term correctly."
    if semantic_score_value < 0.60:
        if missing_keywords:
            return f"You are in the right area, but the core concept is incomplete. Add points such as: {', '.join(missing_keywords[:3])}."
        return "You are in the right area, but the concept is incomplete. Add the main purpose, process, and outcome."
    if 0.60 <= semantic_score_value < 0.80:
        if missing_keywords:
            return f"Your answer is partially correct. To improve it, include ideas like: {', '.join(missing_keywords[:4])}, and make the explanation more structured."
        return "Your answer is partially correct. Make it stronger by adding one clear definition, 2 supporting points, and one practical example."
    if semantic_score_value >= 0.80 and length_score_value < 0.70:
        return "Your answer is mostly correct, but it could be more complete. Add one more important point or a real-world example."
    truth_only = list(dict.fromkeys(w for w in truth_tokens if w not in candidate_tokens))[:4]
    if truth_only:
        return f"Good answer. To make it stronger, mention specific points such as: {', '.join(truth_only)}."
    return "Good answer. To improve further, make it more structured: definition first, then key points, then a short example."


async def evaluate_answer_for_module(
    transcript: str,
    question_data: Dict[str, Any],
    module_id: int,
    db: Any
) -> Dict[str, Any]:
    """
    Fetches module from DB to get the correct model path, then evaluates.
    """
    from sqlalchemy import select
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()
    model_path = module.model_pkl_path if module else None
    return evaluate_answer(transcript, question_data, model_path)


def evaluate_answer(transcript: str, question_data: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    if "expected_keywords" not in question_data:
        truth = question_data.get("answer", "")
        tokens = tokenize(truth)
        seen: list = []
        for w in tokens:
            if w not in seen:
                seen.append(w)
        question_data = dict(question_data, expected_keywords=seen[:12])

    features, meta = extract_features_enhanced(transcript, question_data)

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

    trained_model = get_trained_model(model_path)
    model_pred   = 0.0
    scoring_mode = "heuristic_only"

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

    if meta["semantic_score"] >= 0.80 and meta["length_score"] >= 0.45:
        final_score = max(final_score, 0.68)
    if meta["semantic_score"] >= 0.86 and meta["length_score"] >= 0.55:
        final_score = max(final_score, 0.78)

    final_score = float(np.clip(final_score, 0.0, 1.0))

    feedback = classify_feedback(final_score, meta["missing_keywords"], transcript)
    tip = generate_tip(
        transcript=transcript, truth=meta["truth"],
        missing_keywords=meta["missing_keywords"],
        semantic_score_value=meta["semantic_score"],
        length_score_value=meta["length_score"],
        question_relevance=meta["question_relevance"],
    )

    return {
        "scoring_mode":       scoring_mode,
        "semantic_score":     meta["semantic_score"],
        "question_relevance": meta["question_relevance"],
        "keyword_score":      meta["keyword_score"],
        "overlap_score":      meta["overlap_score"],
        "length_score":       meta["length_score"],
        "lexical_diversity":  meta["lexical_diversity"],
        "discourse_score":    meta["discourse_score"],
        "penalty":            meta["penalty"],
        "model_score":        round(model_pred,    4),
        "heuristic_score":    round(heuristic_pred, 4),
        "final_score":        round(final_score,   4),
        "feedback":           feedback,
        "tip":                tip,
        "matched_keywords":   meta["matched_keywords"],
        "missing_keywords":   meta["missing_keywords"],
    }


def classify_confirmation_intent(text: str) -> str:
    norm = normalize(text)
    if any(t in norm for t in ["end", "stop", "finish interview", "terminate"]): return "end"
    if any(t in norm for t in ["wait", "continue", "not done", "more", "hold on", "one more", "let me continue"]): return "continue"
    if any(t in norm for t in ["repeat", "again", "say again", "what was the question"]): return "repeat"
    if any(t in norm for t in ["done", "next", "move", "move on", "yes", "continue to next", "finished"]): return "next"
    return "next"


def train_evaluation_model(training_samples=None, dataset_path=DEFAULT_TRAINING_DATASET, target_model_path=MODEL_PATH):
    """Kept for backwards compatibility. Prefer: python manage.py train_enhanced_model"""
    if training_samples is None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            training_samples = json.load(f)
    if not training_samples:
        raise ValueError("No training samples provided.")

    X, y = [], []
    for item in training_samples:
        try:
            features, _ = extract_features_enhanced(item["candidate"], item["question_data"])
            X.append(features)
            y.append(float(item["score"]))
        except Exception:
            pass

    X, y = np.array(X), np.array(y)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, validation_fraction=0.1, n_iter_no_change=30, random_state=42)),
    ])
    model.fit(X, y)
    joblib.dump(model, target_model_path)
    get_trained_model.cache_clear()

    mae = float(np.mean(np.abs(model.predict(X) - y)))
    return {"status": "trained", "samples_used": len(X), "model_path": target_model_path, "train_mae": round(mae, 4), "features": N_FEATURES}


if __name__ == "__main__":
    print(train_evaluation_model())
