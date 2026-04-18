"""
routers/sessions.py
====================
Interview session management endpoints.

POST /api/interview/start/
POST /api/interview/{id}/next/
POST /api/interview/{id}/terminate/
GET  /api/interview/{id}/summary/
GET  /api/interview/{id}/detail/
GET  /api/interview/user-sessions/?user_id=
"""
import json
import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db.models import InterviewAnswer, InterviewSession, Module, Question, User
from ..schemas import EvaluationRequest, EvaluationResponse, EvaluationOut, NextQuestionResponse, QuestionOut, StartInterviewRequest, StartInterviewResponse
from ..services.evaluation import evaluate_answer
from ..services.feedback_generator import generate_question_feedback, generate_session_summary
from ..services.nlp_features import tokenize
from ..services import llm_feedback as _llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/interview", tags=["Sessions"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ── question loading ───────────────────────────────────────────────────────────

async def _ensure_questions_loaded(module: Module, db: AsyncSession) -> None:
    count_result = await db.execute(
        select(func.count()).select_from(Question).where(Question.module_id == module.id)
    )
    if count_result.scalar() > 0:
        return

    json_path = module.module_json_path
    if not json_path:
        full_path = BASE_DIR / "data" / "questions.json"
    else:
        if json_path.startswith("/"):
            json_path = json_path[1:]
        full_path = BASE_DIR / json_path
        if not full_path.exists():
            full_path = BASE_DIR / "data" / os.path.basename(json_path)

    if not full_path.exists():
        full_path = BASE_DIR / "data" / "questions.json"

    if not full_path.exists():
        logger.warning(f"No question file found for module {module.id}")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    seen_texts = set()
    for item in data:
        q_data = item.get("question_data", item)
        text = q_data.get("question", q_data.get("Interview Question", "")).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        questions.append(Question(
            module_id=module.id,
            topic=q_data.get("topic"),
            question_text=text,
            expected_answer=q_data.get("answer", q_data.get("Answer", "")),
            order=len(questions),
        ))

    db.add_all(questions)
    await db.commit()
    logger.info(f"Loaded {len(questions)} questions for module {module.id}")


async def _get_question_at(session: InterviewSession, db: AsyncSession) -> Question | None:
    result = await db.execute(
        select(Question)
        .where(Question.module_id == session.module_id)
        .order_by(Question.order)
        .offset(session.current_index)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _question_count(module_id: int, db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count()).select_from(Question).where(Question.module_id == module_id)
    )
    return result.scalar()


# ── feedback helpers ───────────────────────────────────────────────────────────

def _get_feedback_dict(answer: InterviewAnswer, question: Question) -> dict:
    """
    Build the full feedback dict from the new feedback_generator.
    Uses DB-stored scores + re-runs grammar/content analysis on the transcript.
    """
    cand_tokens = len(tokenize(answer.transcript or ""))
    ref_tokens  = len(tokenize(question.expected_answer or "")) or 1

    metrics = {
        "final_score":        float(answer.final_score        or 0.0),
        "semantic_score":     float(answer.semantic_score     or 0.0),
        "keyword_score":      float(answer.keyword_score      or 0.0),
        "question_relevance": float(answer.question_relevance or 0.0),
        "lexical_diversity":  float(answer.lexical_diversity  or 0.0),
        "discourse_score":    float(answer.discourse_score    or 0.0),
        "length_score":       min(1.0, cand_tokens / ref_tokens),
        "missing_keywords":   list(answer.missing_keywords    or []),
        "matched_keywords":   [],
    }

    question_data = {
        "id":             str(question.id),
        "question":       question.question_text,
        "answer":         question.expected_answer,
        "primary_answer": question.expected_answer,
    }

    return generate_question_feedback(
        transcript=answer.transcript or "",
        metrics=metrics,
        question_data=question_data,
        question_id=str(question.id),
    )


def _serialize_answer(answer: InterviewAnswer, question: Question, fb: dict) -> dict:
    """Merge DB answer row + rich feedback dict into an API-ready dict."""
    return {
        "question_id":        question.id,
        "question_text":      question.question_text,
        "transcript":         answer.transcript or "",
        "final_score":        answer.final_score,
        "semantic_score":     answer.semantic_score,
        "keyword_score":      answer.keyword_score,
        "question_relevance": answer.question_relevance,
        "lexical_diversity":  answer.lexical_diversity,
        "discourse_score":    answer.discourse_score,
        "penalty":            answer.penalty,
        # Use stored feedback/tip (written at evaluation time, may be richer)
        "feedback":           answer.feedback or fb.get("narrative", ""),
        "tip":                answer.tip      or "",
        # New feedback fields from the rich generator
        "score_tier":          fb.get("score_tier", ""),
        "improvement_bullets": fb.get("improvement_tips", []),
        "grammar_notes":       fb.get("grammar_notes", {}),
        "stt_flags":           fb.get("stt_flags", []),
        "missing_keywords":    list(answer.missing_keywords or []),
    }


def _build_session_summary(feedback_dicts: List[dict]) -> dict:
    """
    Generate session-level summary from a list of feedback dicts.
    generate_session_summary expects the same dicts returned by
    generate_question_feedback (contains score, score_tier, grammar_notes, etc.)
    """
    if not feedback_dicts:
        return {
            "summary_paragraph":   "No answers were recorded for this session.",
            "strengths":           [],
            "improvement_areas":   [],
            "metric_averages":     {},
            "grammar_summary":     {},
        }
    return generate_session_summary(feedback_dicts)


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start/", response_model=StartInterviewResponse)
async def start_interview(body: StartInterviewRequest, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    module_result = await db.execute(select(Module).where(Module.id == body.module_id))
    module = module_result.scalar_one_or_none()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    await _ensure_questions_loaded(module, db)

    total = await _question_count(module.id, db)
    if total == 0:
        raise HTTPException(status_code=500, detail="No questions found for this module")

    session_result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.user_id  == body.user_id,
            InterviewSession.module_id == body.module_id,
            InterviewSession.status   == "active",
        )
    )
    session = session_result.scalars().first()

    if not session:
        session = InterviewSession(
            user_id=body.user_id,
            module_id=body.module_id,
            current_index=0,
            status="active",
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

    current_q = await _get_question_at(session, db)
    if not current_q:
        raise HTTPException(status_code=500, detail="Could not load first question")

    return StartInterviewResponse(
        session_id=session.id,
        question=QuestionOut(
            id=current_q.id,
            topic=current_q.topic,
            question=current_q.question_text,
            answer=current_q.expected_answer,
        ),
        question_index=session.current_index,
        total_questions=total,
    )


@router.post("/{session_id}/next/", response_model=NextQuestionResponse)
async def next_question(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.current_index += 1
    await db.commit()
    await db.refresh(session)

    total = await _question_count(session.module_id, db)

    if session.current_index >= total:
        session.status = "completed"
        await db.commit()

        rows_result = await db.execute(
            select(InterviewAnswer, Question)
            .join(Question, InterviewAnswer.question_id == Question.id)
            .where(InterviewAnswer.session_id == session_id)
            .order_by(Question.order)
        )
        rows = rows_result.all()

        feedback_dicts = [_get_feedback_dict(a, q) for a, q in rows]
        results        = [_serialize_answer(a, q, fb) for (a, q), fb in zip(rows, feedback_dicts)]
        session_sum    = _build_session_summary(feedback_dicts)

        return NextQuestionResponse(
            session_id=session_id,
            completed=True,
            summary={"results": results, **session_sum},
        )

    next_q = await _get_question_at(session, db)
    return NextQuestionResponse(
        session_id=session_id,
        completed=False,
        question=QuestionOut(
            id=next_q.id,
            topic=next_q.topic,
            question=next_q.question_text,
            answer=next_q.expected_answer,
        ),
        question_index=session.current_index,
        total_questions=total,
    )


@router.post("/{session_id}/terminate/")
async def terminate_session(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = "completed"
    await db.commit()

    rows_result = await db.execute(
        select(InterviewAnswer, Question)
        .join(Question, InterviewAnswer.question_id == Question.id)
        .where(InterviewAnswer.session_id == session_id)
        .order_by(Question.order)
    )
    rows = rows_result.all()

    feedback_dicts = [_get_feedback_dict(a, q) for a, q in rows]
    results        = [_serialize_answer(a, q, fb) for (a, q), fb in zip(rows, feedback_dicts)]
    session_sum    = _build_session_summary(feedback_dicts)

    return {
        "session_id": session_id,
        "terminated": True,
        "results":    results,
        **session_sum,
    }


@router.get("/{session_id}/summary/")
async def get_summary(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rows_result = await db.execute(
        select(InterviewAnswer, Question)
        .join(Question, InterviewAnswer.question_id == Question.id)
        .where(InterviewAnswer.session_id == session_id)
        .order_by(Question.order)
    )
    rows = rows_result.all()

    total_result = await db.execute(
        select(func.count()).select_from(Question).where(Question.module_id == session.module_id)
    )

    feedback_dicts = [_get_feedback_dict(a, q) for a, q in rows]
    results        = [_serialize_answer(a, q, fb) for (a, q), fb in zip(rows, feedback_dicts)]
    session_sum    = _build_session_summary(feedback_dicts)

    return {
        "session_id":          session_id,
        "completed_questions": len(results),
        "total_questions":     total_result.scalar(),
        "summary_paragraph":   session_sum.get("summary_paragraph", ""),
        "strengths":           session_sum.get("strengths", []),
        "improvement_areas":   session_sum.get("improvement_areas", []),
        "results":             results,
    }


@router.get("/{session_id}/detail/")
async def get_session_detail(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rows_result = await db.execute(
        select(InterviewAnswer, Question)
        .join(Question, InterviewAnswer.question_id == Question.id)
        .where(InterviewAnswer.session_id == session_id)
        .order_by(Question.order)
    )
    rows = rows_result.all()

    avg_result = await db.execute(
        select(func.avg(InterviewAnswer.final_score))
        .where(InterviewAnswer.session_id == session_id)
    )

    module_result = await db.execute(select(Module).where(Module.id == session.module_id))
    module = module_result.scalar_one_or_none()

    feedback_dicts = [_get_feedback_dict(a, q) for a, q in rows]
    results        = [_serialize_answer(a, q, fb) for (a, q), fb in zip(rows, feedback_dicts)]
    session_sum    = _build_session_summary(feedback_dicts)

    return {
        "session_id":        session_id,
        "created_at":        session.created_at,
        "module_name":       module.module_name if module else "Unknown",
        "total_score":       round(avg_result.scalar() or 0.0, 1),
        "summary_paragraph": session_sum.get("summary_paragraph", ""),
        "strengths":         session_sum.get("strengths", []),
        "improvement_areas": session_sum.get("improvement_areas", []),
        "results":           results,
    }


@router.get("/user-sessions/")
async def get_user_sessions(user_id: int, db: AsyncSession = Depends(get_db)):
    sessions_result = await db.execute(
        select(InterviewSession)
        .where(InterviewSession.user_id == user_id, InterviewSession.status == "completed")
        .order_by(InterviewSession.created_at.desc())
    )
    sessions = sessions_result.scalars().all()

    output = []
    for s in sessions:
        avg_result = await db.execute(
            select(func.avg(InterviewAnswer.final_score))
            .where(InterviewAnswer.session_id == s.id)
        )
        count_result = await db.execute(
            select(func.count()).select_from(InterviewAnswer)
            .where(InterviewAnswer.session_id == s.id)
        )
        module_result = await db.execute(select(Module).where(Module.id == s.module_id))
        module = module_result.scalar_one_or_none()

        output.append({
            "id":             s.id,
            "created_at":     s.created_at,
            "module_name":    module.module_name if module else "Unknown",
            "total_score":    round(avg_result.scalar() or 0.0, 1),
            "question_count": count_result.scalar(),
        })

    return {"data": output}


# ── deferred evaluation ────────────────────────────────────────────────────────

async def _upsert_answer(
    session: InterviewSession,
    question: Question,
    transcript: str,
    evaluation: dict,
    db: AsyncSession,
) -> None:
    result = await db.execute(
        select(InterviewAnswer).where(
            InterviewAnswer.session_id == session.id,
            InterviewAnswer.question_id == question.id,
        )
    )
    answer = result.scalar_one_or_none()
    fields = dict(
        transcript=transcript,
        semantic_score=evaluation["semantic_score"],
        keyword_score=evaluation["keyword_score"],
        question_relevance=evaluation.get("question_relevance", 0.0),
        lexical_diversity=evaluation.get("lexical_diversity", 0.0),
        discourse_score=evaluation.get("discourse_score", 0.0),
        penalty=evaluation.get("penalty", 0.0),
        final_score=evaluation["final_score"],
        feedback=evaluation["feedback"],
        tip=evaluation["tip"],
        missing_keywords=evaluation["missing_keywords"],
    )
    if answer:
        for k, v in fields.items():
            setattr(answer, k, v)
    else:
        db.add(InterviewAnswer(
            session_id=session.id,
            question_id=question.id,
            raw_segments=[],
            **fields,
        ))
    await db.commit()


@router.post("/{session_id}/evaluate/", response_model=EvaluationResponse)
async def evaluate_question(
    session_id: int,
    body: EvaluationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch and return a previously stored evaluation for a question.
    Evaluation is performed by the WebSocket transcription handler;
    this endpoint simply reads back the stored result.
    """
    ans_result = await db.execute(
        select(InterviewAnswer).where(
            InterviewAnswer.session_id == session_id,
            InterviewAnswer.question_id == body.question_id,
        )
    )
    ans_row = ans_result.scalar_one_or_none()
    if not ans_row:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found. Complete the question via the WebSocket first.",
        )

    return EvaluationResponse(
        evaluation=EvaluationOut(
            score=ans_row.final_score,
            semantic_score=ans_row.semantic_score,
            keyword_score=ans_row.keyword_score,
            question_relevance=ans_row.question_relevance,
            lexical_diversity=ans_row.lexical_diversity,
            discourse_score=ans_row.discourse_score,
            penalty=ans_row.penalty,
            feedback=ans_row.feedback,
            tip=ans_row.tip,
            missing_keywords=list(ans_row.missing_keywords or []),
        )
    )

