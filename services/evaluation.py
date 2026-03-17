"""
Mock implementations of evaluation services.
Replace the bodies of these functions with your real ML logic —
the signatures and return shapes are production-ready.
"""
import random
from typing import Optional


def evaluate_answer(transcript: str, question_data: dict, model_path: Optional[str] = None) -> dict:
    """
    Evaluate a candidate's answer against the expected answer.

    Args:
        transcript:     What the candidate actually said.
        question_data:  Dict with keys: id, question, answer (expected).
        model_path:     Optional path to a .pkl model for this module.

    Returns:
        Dict with scores and feedback — shape must stay stable.
    """
    # ── MOCK: replace with your real semantic + keyword scoring ──────────────
    if not transcript or len(transcript.strip()) < 5:
        return {
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "final_score": 0.0,
            "feedback": "No answer detected. Please speak clearly and try again.",
            "tip": "Make sure your microphone is working and speak for at least a few seconds.",
            "missing_keywords": [],
        }

    semantic_score = round(random.uniform(55.0, 95.0), 1)
    keyword_score  = round(random.uniform(50.0, 90.0), 1)
    final_score    = round(semantic_score * 0.6 + keyword_score * 0.4, 1)

    expected = question_data.get("answer", "")
    # Simulate extracting keywords from expected answer
    words         = [w.strip(".,;:").lower() for w in expected.split() if len(w) > 5]
    spoken_words  = transcript.lower()
    missing       = [w for w in words[:8] if w not in spoken_words]

    return {
        "semantic_score": semantic_score,
        "keyword_score":  keyword_score,
        "final_score":    final_score,
        "feedback": (
            "Good answer! You covered the main concepts well."
            if final_score >= 70
            else "Your answer touched on some points but missed key details."
        ),
        "tip": (
            "Try to structure your answer using the STAR method for clarity."
            if final_score < 70
            else "Great work! Consider adding a concrete example next time."
        ),
        "missing_keywords": missing[:5],
    }
    # ─────────────────────────────────────────────────────────────────────────


def classify_confirmation_intent(transcript: str) -> str:
    """
    Classify whether the candidate wants to proceed or repeat.

    Args:
        transcript: What the candidate said after being asked "next or repeat?".

    Returns:
        "next" | "repeat" | "skip"
    """
    # ── MOCK: replace with your NLP classifier ────────────────────────────────
    if not transcript:
        return "next"

    text = transcript.lower().strip()

    repeat_signals = {"repeat", "again", "redo", "retry", "no", "nope", "wait", "hold", "continue"}
    skip_signals   = {"skip", "pass", "next question", "move on"}

    if any(s in text for s in repeat_signals):
        return "repeat"
    if any(s in text for s in skip_signals):
        return "skip"
    return "next"
    # ─────────────────────────────────────────────────────────────────────────