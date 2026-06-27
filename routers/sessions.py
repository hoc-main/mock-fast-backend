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
from sqlalchemy.orm import selectinload

from ..db.database import get_db
from ..db.models import InterviewAnswer, InterviewSession, Module, Question, Subdomain, User, Purchase
from ..schemas import EvaluationRequest, EvaluationResponse, EvaluationOut, NextQuestionResponse, QuestionOut, StartInterviewRequest, StartInterviewResponse
from ..services.evaluation import evaluate_answer
from ..services.feedback_generator import generate_question_feedback, generate_session_summary
from ..services.nlp_features import tokenize
from ..services import llm_feedback as _llm
from ..services.llm_feedback import rewrite_session_summary
from ..services.conversation_agent import pick_next_question

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
    score = float(answer.final_score or 0.0)
    # Convert 0-1 to 0-100 percentage
    score_pct = round(score * 100, 1) if score <= 1.0 else round(score, 1)

    # Always prefer the DB-stored feedback (same as what was spoken to the user)
    # Only fall back to regenerated template if DB has nothing
    feedback_text = answer.feedback or fb.get("narrative", "")
    tip_text = answer.tip or (fb.get("improvement_tips", [""])[0] if fb.get("improvement_tips") else "")

    return {
        "question_id":        question.id,
        "question_text":      question.question_text,
        "transcript":         answer.transcript or "",
        "final_score":        score_pct,
        "feedback":           feedback_text,
        "tip":                tip_text,
        "score_tier":          fb.get("score_tier", ""),
        "improvement_bullets": fb.get("improvement_tips", []),
        "missing_keywords":    list(answer.missing_keywords or []),
    }


async def _build_session_summary(feedback_dicts: List[dict]) -> dict:
    """
    Generate session-level summary from a list of feedback dicts.
    Uses template logic first, then rewrites with LLM for better phrasing.
    """
    if not feedback_dicts:
        return {
            "summary_paragraph":   "No answers were recorded for this session.",
            "strengths":           [],
            "improvement_areas":   [],
            "metric_averages":     {},
            "grammar_summary":     {},
        }
    summary = generate_session_summary(feedback_dicts)
    summary = await rewrite_session_summary(summary)
    return summary


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start/", response_model=StartInterviewResponse)
async def start_interview(body: StartInterviewRequest, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    module_result = await db.execute(select(Module).options(selectinload(Module.subdomain)).where(Module.id == body.module_id))
    module = module_result.scalar_one_or_none()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Check if module is free or user has purchased it
    if not module.is_free:
        purchase_result = await db.execute(
            select(Purchase)
            .where(
                Purchase.user_id == body.user_id,
                Purchase.module_id == body.module_id
            )
        )
        purchase = purchase_result.scalar_one_or_none()
        if not purchase:
            raise HTTPException(
                status_code=403,
                detail="You need to purchase this mock interview to access it."
            )

    # Check for existing sessions
    existing_sessions_result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.user_id == body.user_id,
            InterviewSession.module_id == body.module_id,
        )
    )
    existing_sessions = existing_sessions_result.scalars().all()

    # Check if there's an active session to resume
    active_session = next((s for s in existing_sessions if s.status == "active"), None)
    if active_session:
        session = active_session
    else:
        # If there's any completed session, block
        if existing_sessions:
            raise HTTPException(
                status_code=403,
                detail="You have already attempted this mock interview. Multiple attempts are not allowed."
            )
        # No sessions at all, create new
        session = InterviewSession(
            user_id=body.user_id,
            module_id=body.module_id,
            current_index=0,
            status="active",
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

    await _ensure_questions_loaded(module, db)

    total = await _question_count(module.id, db)
    if total == 0:
        raise HTTPException(status_code=500, detail="No questions found for this module")

    current_q = await _get_question_at(session, db)

    # Try to pick a question not asked in any previous session (across all modules)
    if current_q and session.user_id:
        past_sessions_result = await db.execute(
            select(InterviewSession).where(
                InterviewSession.user_id == body.user_id,
                InterviewSession.status == "completed",
                InterviewSession.id != session.id,
            )
        )
        past_asked_ids = set()
        for past_s in past_sessions_result.scalars().all():
            past_asked_ids.update(past_s.asked_question_ids or [])

        if current_q.id in past_asked_ids:
            # Find first question not asked in any past session
            all_q_result = await db.execute(
                select(Question)
                .where(Question.module_id == module.id)
                .order_by(Question.order)
            )
            for q in all_q_result.scalars().all():
                if q.id not in past_asked_ids:
                    current_q = q
                    session.current_index = q.order
                    await db.commit()
                    break

    if not current_q:
        raise HTTPException(status_code=500, detail="Could not load first question")

    # Track first question as asked
    asked = list(session.asked_question_ids or [])
    if current_q.id not in asked:
        asked.append(current_q.id)
    session.asked_question_ids = asked
    session.current_question_id = current_q.id
    await db.commit()

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
        domain_id=module.subdomain.domain_id if module and module.subdomain else None,
        subdomain_id=module.subdomain_id if module else None,
        module_id=module.id if module else None,
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

    # Use number of actually answered questions to determine completion,
    # not current_index which may drift from actual progress when LLM picks questions
    answered_count_result = await db.execute(
        select(func.count()).select_from(InterviewAnswer)
        .where(InterviewAnswer.session_id == session_id)
    )
    answered_count = answered_count_result.scalar() or 0

    if answered_count >= total:
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
        session_sum    = await _build_session_summary(feedback_dicts)

        return NextQuestionResponse(
            session_id=session_id,
            completed=True,
            summary={"results": results, **session_sum},
        )

    # ── Conversational: LLM picks next question ───────────────────────────────
    next_q = None
    transition = None

    if _llm.llm_available:
        # Get all remaining (unanswered) questions — exclude this session + past sessions
        asked_ids = list(session.asked_question_ids or [])

        # Exclude questions from ALL of this user's past sessions (any module)
        # so the same user never gets the same question across different sessions
        if session.user_id:
            past_sessions_result = await db.execute(
                select(InterviewSession).where(
                    InterviewSession.user_id == session.user_id,
                    InterviewSession.status == "completed",
                    InterviewSession.id != session.id,
                )
            )
            for past_session in past_sessions_result.scalars().all():
                asked_ids.extend(past_session.asked_question_ids or [])
            asked_ids = list(set(asked_ids))  # deduplicate

        # Get current module and its job_roles + subdomain
        module_result = await db.execute(select(Module).where(Module.id == session.module_id))
        module = module_result.scalar_one_or_none()
        module_topic = module.module_name if module else "technical"
        current_job_roles = set(module.job_roles or []) if module else set()

        # Expand question pool: all modules in same subdomain sharing job roles
        sibling_module_ids = [session.module_id]
        if module and module.subdomain_id and current_job_roles:
            sibling_result = await db.execute(
                select(Module).where(
                    Module.subdomain_id == module.subdomain_id,
                    Module.id != module.id,
                )
            )
            for sibling in sibling_result.scalars().all():
                sibling_roles = set(sibling.job_roles or [])
                if sibling_roles & current_job_roles:  # intersection
                    sibling_module_ids.append(sibling.id)
                    # Ensure sibling questions are loaded
                    await _ensure_questions_loaded(sibling, db)

        logger.info(
            f"[session={session_id}] cross-question pool: "
            f"{len(sibling_module_ids)} modules (subdomain={module.subdomain_id}, "
            f"roles={list(current_job_roles)[:3]})"
        )

        # Fetch questions from the expanded pool
        all_q_result = await db.execute(
            select(Question)
            .where(Question.module_id.in_(sibling_module_ids))
            .order_by(Question.module_id, Question.order)
        )
        all_questions = all_q_result.scalars().all()
        remaining = [
            {"id": q.id, "question": q.question_text, "topic": q.topic or ""}
            for q in all_questions if q.id not in asked_ids
        ]

        # If all questions exhausted across sessions, reset to current session only
        if not remaining:
            current_asked = list(session.asked_question_ids or [])
            remaining = [
                {"id": q.id, "question": q.question_text, "topic": q.topic or ""}
                for q in all_questions if q.id not in current_asked
            ]

        # Ask LLM to pick next question (cap at 20 to keep prompt manageable)
        MAX_QUESTIONS_FOR_LLM = 20
        if remaining:
            import random
            llm_pool = remaining if len(remaining) <= MAX_QUESTIONS_FOR_LLM else random.sample(remaining, MAX_QUESTIONS_FOR_LLM)
            decision = await pick_next_question(
                remaining_questions=llm_pool,
                conversation_history=list(session.conversation_history or []),
                module_topic=module_topic,
            )
            if decision:
                # Load the chosen question
                q_result = await db.execute(
                    select(Question).where(Question.id == decision.question_id)
                )
                next_q = q_result.scalar_one_or_none()
                transition = decision.transition
                logger.info(
                    f"[session={session_id}] LLM picked Q{decision.question_id}: "
                    f"{decision.reasoning[:60]}"
                )

    # Fallback: sequential order
    if not next_q:
        next_q = await _get_question_at(session, db)

    # Track asked question
    asked = list(session.asked_question_ids or [])
    if next_q.id not in asked:
        asked.append(next_q.id)
    session.asked_question_ids = asked
    session.current_question_id = next_q.id
    await db.commit()

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
        transition=transition,
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
    session_sum    = await _build_session_summary(feedback_dicts)

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
    session_sum    = await _build_session_summary(feedback_dicts)

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

    module_result = await db.execute(
        select(Module)
        .options(selectinload(Module.subdomain))
        .where(Module.id == session.module_id)
    )
    module = module_result.scalar_one_or_none()

    feedback_dicts = [_get_feedback_dict(a, q) for a, q in rows]
    results        = [_serialize_answer(a, q, fb) for (a, q), fb in zip(rows, feedback_dicts)]
    session_sum    = await _build_session_summary(feedback_dicts)

    avg_score = avg_result.scalar() or 0.0
    # Convert 0-1 score to 0-100 percentage for frontend display
    total_score_pct = round(avg_score * 100, 1) if avg_score <= 1.0 else round(avg_score, 1)

    return {
        "session_id":        session_id,
        "created_at":        session.created_at,
        "module_name":       module.module_name if module else "Unknown",
        "subdomain_name":    module.subdomain.name if (module and module.subdomain) else "Unknown",
        "total_score":       total_score_pct,
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
        module_result = await db.execute(
            select(Module)
            .options(selectinload(Module.subdomain))
            .where(Module.id == s.module_id)
        )
        module = module_result.scalar_one_or_none()

        avg_score = avg_result.scalar() or 0.0
        total_score_pct = round(avg_score * 100, 1) if avg_score <= 1.0 else round(avg_score, 1)

        output.append({
            "id":             s.id,
            "created_at":     s.created_at,
            "module_name":    module.module_name if module else "Unknown",
            "subdomain_name":  module.subdomain.name if (module and module.subdomain) else "Unknown",
            "total_score":    total_score_pct,
            "question_count": count_result.scalar(),
        })

    return {"data": output}


# ── delete session (public) ────────────────────────────────────────────────────

@router.delete("/session/")
async def delete_session(user_id: int, module_id: int, db: AsyncSession = Depends(get_db)):
    """Delete all sessions and answers for a user+module so the exam can be retaken."""
    sessions_result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.user_id == user_id,
            InterviewSession.module_id == module_id,
        )
    )
    sessions = sessions_result.scalars().all()
    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found for this user and module")

    for s in sessions:
        answers_result = await db.execute(
            select(InterviewAnswer).where(InterviewAnswer.session_id == s.id)
        )
        for answer in answers_result.scalars().all():
            await db.delete(answer)
        await db.delete(s)

    await db.commit()
    return {"deleted": True, "sessions_removed": len(sessions)}


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
            feedback=ans_row.feedback,
            tip=ans_row.tip,
            missing_keywords=list(ans_row.missing_keywords or []),
        )
    )

