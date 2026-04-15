"""
routers/feedback.py
====================
Rich feedback endpoints — per-question and session-level summary.

POST /api/feedback/question
    Body : QuestionFeedbackRequest
    Returns : QuestionFeedbackResponse

POST /api/feedback/session
    Body : SessionFeedbackRequest
    Returns : SessionFeedbackResponse

POST /api/feedback/session/from-db
    Query : session_id=<int>
    Returns : SessionFeedbackResponse   (pulls answers directly from DB)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db.models import InterviewAnswer, InterviewSession, Question

# FIX [10]: removed unused `from ..services.evaluation import evaluate_answer`

from ..services.feedback_generator import (
    generate_question_feedback,
    generate_session_summary,
)
from ..services import llm_feedback as _llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


# ── Request / Response schemas ────────────────────────────────────────────────

class MetricSnapshot(BaseModel):
    """Mirrors evaluate_answer() output. All fields optional for flexibility."""
    final_score:        float = 0.5
    semantic_score:     float = 0.5
    question_relevance: float = 0.5
    keyword_score:      float = 0.5
    overlap_score:      float = 0.5
    length_score:       float = 0.5
    discourse_score:    float = 0.4
    lexical_diversity:  float = 0.5
    penalty:            float = 0.0
    matched_keywords:   List[str] = Field(default_factory=list)
    missing_keywords:   List[str] = Field(default_factory=list)
    model_score:        float = 0.5
    heuristic_score:    float = 0.5
    scoring_mode:       str = "unknown"
    feedback:           str = ""
    tip:                str = ""


class QuestionDataIn(BaseModel):
    """Flexible question_data — accepts both dataset formats."""
    id:               Optional[str]       = None
    question:         Optional[str]       = None
    question_text:    Optional[str]       = None   # alias used in some datasets
    primary_answer:   Optional[str]       = None
    ideal_answer:     Optional[str]       = None   # alias
    answer:           Optional[str]       = None   # legacy alias
    answer_variant_1: Optional[str]       = None
    answer_variant_2: Optional[str]       = None
    answer_variants:  Optional[List[str]] = None
    expected_keywords: Optional[Any]     = None    # dict or list
    topic:            Optional[str]       = None


class QuestionFeedbackRequest(BaseModel):
    transcript:     str
    metrics:        MetricSnapshot
    question_data:  QuestionDataIn
    question_id:    Optional[str] = None
    question_index: int = 0   # used for deterministic template selection


class GrammarNotes(BaseModel):
    filler_count:       int
    filler_words_found: List[str]
    filler_rate:        float
    word_repetitions:   List[str]
    phrase_repetitions: List[str]
    sentence_count:     int
    avg_words_per_sent: float
    run_on_detected:    bool
    capitalization_ok:  bool
    severity:           str


class ContentAnalysis(BaseModel):
    present_keywords:  Dict[str, List[str]]
    missing_keywords:  Dict[str, List[str]]
    variant_scores:    List[float]
    best_variant_idx:  int
    concepts_covered:  List[str]
    concepts_missing:  List[str]


class QuestionFeedbackResponse(BaseModel):
    question_id:      str
    question_text:    str
    score:            float
    score_tier:       str
    narrative:        str
    improvement_tips: List[str]
    grammar_notes:    GrammarNotes
    content_analysis: ContentAnalysis
    stt_flags:        List[str]
    metric_snapshot:  Dict[str, float]


class PerQuestionInput(BaseModel):
    """One answered question's data for session-level aggregation."""
    transcript:     str
    metrics:        MetricSnapshot
    question_data:  QuestionDataIn
    question_id:    Optional[str] = None
    question_index: int = 0


class SessionFeedbackRequest(BaseModel):
    session_id: Optional[int] = None
    questions:  List[PerQuestionInput]


class GrammarSummary(BaseModel):
    total_filler_words:  int
    overall_filler_rate: float
    run_on_sentences:    int
    most_used_fillers:   List[str]
    grammar_tier:        str


class SessionFeedbackResponse(BaseModel):
    session_score:       float
    score_tier:          str
    score_trend:         List[float]
    trend_direction:     str
    strengths:           List[str]
    improvement_areas:   List[str]
    grammar_summary:     GrammarSummary
    summary_paragraph:   str
    per_question_scores: List[Dict[str, Any]]
    metric_averages:     Dict[str, float]
    question_feedbacks:  List[QuestionFeedbackResponse]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_question_data_dict(qd: QuestionDataIn) -> dict:
    """Convert Pydantic model to the plain dict feedback_generator expects."""
    d = {}
    for field in [
        "question", "question_text", "primary_answer", "ideal_answer",
        "answer", "answer_variant_1", "answer_variant_2",
        "answer_variants", "expected_keywords", "topic", "id",
    ]:
        val = getattr(qd, field, None)
        if val is not None:
            d[field] = val
    return d


def _build_metrics_dict(m: MetricSnapshot) -> dict:
    return m.model_dump()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/question", response_model=QuestionFeedbackResponse)
async def get_question_feedback(req: QuestionFeedbackRequest):
    """
    Generate rich feedback for a single answered question.
    Tries LLM (Groq) first for narrative + tips.
    Falls back to rule-based feedback_generator if LLM fails.
    """
    try:
        fb = generate_question_feedback(
            transcript=req.transcript,
            metrics=_build_metrics_dict(req.metrics),
            question_data=_build_question_data_dict(req.question_data),
            question_id=req.question_id,
            seed=req.question_index,
        )

        # Try LLM first — override narrative + tips if it succeeds
        if _llm.llm_available:
            try:
                qd = _build_question_data_dict(req.question_data)
                llm_result = await _llm.generate_llm_feedback(
                    question=qd.get("question", qd.get("question_text", "")),
                    candidate_answer=req.transcript,
                    expected_answer=qd.get("primary_answer", qd.get("answer", "")),
                    metrics=_build_metrics_dict(req.metrics),
                    missing_keywords=req.metrics.missing_keywords,
                )
                if llm_result and llm_result.get("feedback"):
                    fb["narrative"] = llm_result["feedback"]
                    if llm_result.get("tip"):
                        fb["improvement_tips"] = [llm_result["tip"]]
                    logger.info("Using LLM feedback for /question")
                else:
                    logger.info("LLM returned empty for /question, using rule-based")
            except Exception as exc:
                logger.warning(f"LLM failed for /question, using rule-based: {exc}")

        return QuestionFeedbackResponse(**fb)
    except Exception as exc:
        logger.exception("Error generating question feedback")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/session", response_model=SessionFeedbackResponse)
async def get_session_feedback(req: SessionFeedbackRequest):
    """
    Generate rich feedback for an entire interview session.
    Tries LLM first per question, falls back to rule-based.
    """
    if not req.questions:
        raise HTTPException(status_code=400, detail="No questions provided.")

    try:
        all_feedbacks = []
        for item in req.questions:
            fb = generate_question_feedback(
                transcript=item.transcript,
                metrics=_build_metrics_dict(item.metrics),
                question_data=_build_question_data_dict(item.question_data),
                question_id=item.question_id,
                seed=item.question_index,
            )

            # Try LLM first per question
            if _llm.llm_available:
                try:
                    qd = _build_question_data_dict(item.question_data)
                    llm_result = await _llm.generate_llm_feedback(
                        question=qd.get("question", qd.get("question_text", "")),
                        candidate_answer=item.transcript,
                        expected_answer=qd.get("primary_answer", qd.get("answer", "")),
                        metrics=_build_metrics_dict(item.metrics),
                        missing_keywords=item.metrics.missing_keywords,
                    )
                    if llm_result and llm_result.get("feedback"):
                        fb["narrative"] = llm_result["feedback"]
                        if llm_result.get("tip"):
                            fb["improvement_tips"] = [llm_result["tip"]]
                except Exception as exc:
                    logger.warning(f"LLM failed for session question, using rule-based: {exc}")

            all_feedbacks.append(fb)

        summary = generate_session_summary(all_feedbacks)

        return SessionFeedbackResponse(
            **summary,
            question_feedbacks=[QuestionFeedbackResponse(**fb) for fb in all_feedbacks],
        )
    except Exception as exc:
        logger.exception("Error generating session feedback")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/session/from-db", response_model=SessionFeedbackResponse)
async def get_session_feedback_from_db(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Pull all answers for a completed session directly from the DB
    and generate full rich feedback without the caller needing to
    re-send transcripts and metrics.
    """
    result = await db.execute(
        select(InterviewAnswer)
        .where(InterviewAnswer.session_id == session_id)
        .order_by(InterviewAnswer.id)
    )
    answers = result.scalars().all()

    if not answers:
        raise HTTPException(
            status_code=404,
            detail=f"No answers found for session {session_id}."
        )

    all_feedbacks = []

    for idx, ans in enumerate(answers):
        # Load question from DB
        q_result = await db.execute(
            select(Question).where(Question.id == ans.question_id)
        )
        question = q_result.scalar_one_or_none()

        if question is None:
            logger.warning(f"Question {ans.question_id} not found, skipping.")
            continue

        # FIX [1]: question.question_text not question.question
        # FIX [2]: question.expected_answer not question.answer
        question_data = {
            "id":             str(question.id),
            "question":       question.question_text,
            "primary_answer": question.expected_answer,
            "answer":         question.expected_answer,
            "topic":          question.topic or "",
            # expected_keywords not stored on Question model — leave empty
            "expected_keywords": [],
        }

        # FIX [3][4][5][6][7]: columns that don't exist on InterviewAnswer.
        # length_score, model_score, heuristic_score, scoring_mode,
        # matched_keywords are NOT stored in the DB. Use safe fallbacks.
        # FIX [11]: overlap_score not stored — default 0.0, don't proxy keyword_score
        metrics = {
            "final_score":        float(ans.final_score        or 0.0),
            "semantic_score":     float(ans.semantic_score     or 0.0),
            "question_relevance": float(ans.question_relevance or 0.0),
            "keyword_score":      float(ans.keyword_score      or 0.0),
            "overlap_score":      0.0,    # not stored in DB
            "length_score":       0.0,    # not stored in DB
            "discourse_score":    float(ans.discourse_score    or 0.0),
            "lexical_diversity":  float(ans.lexical_diversity  or 0.0),
            "penalty":            float(ans.penalty            or 0.0),
            "matched_keywords":   [],     # not stored in DB
            "missing_keywords":   list(ans.missing_keywords    or []),
            "model_score":        0.0,    # not stored in DB
            "heuristic_score":    0.0,    # not stored in DB
            "scoring_mode":       "unknown",  # not stored in DB
            "feedback":           ans.feedback or "",
            "tip":                ans.tip      or "",
        }

        fb = generate_question_feedback(
            transcript=ans.transcript or "",
            metrics=metrics,
            question_data=question_data,
            question_id=str(question.id),
            seed=idx,
        )
        all_feedbacks.append(fb)

    if not all_feedbacks:
        raise HTTPException(
            status_code=422,
            detail="Could not generate feedback — questions may be missing from DB."
        )

    summary = generate_session_summary(all_feedbacks)

    return SessionFeedbackResponse(
        **summary,
        question_feedbacks=[QuestionFeedbackResponse(**fb) for fb in all_feedbacks],
    )
