import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db.models import InterviewAnswer, InterviewSession, Module, Question, User
from ..schemas import NextQuestionResponse, QuestionOut, StartInterviewRequest, StartInterviewResponse
from ..services.feedback_generator import generate_question_feedback, generate_session_summary
from ..services.nlp_features import tokenize

logger = logging.getLogger(__name__)

# Routes under /api/interview/ — matches frontend patterns exactly
router = APIRouter(prefix="/api/interview", tags=["Sessions"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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


def _build_evaluation_payload(answer: InterviewAnswer, question: Question) -> dict:
    cand_tokens = len(tokenize(answer.transcript))
    ref_tokens = len(tokenize(question.expected_answer)) or 1
    evaluation = {
        "final_score": answer.final_score,
        "semantic_score": answer.semantic_score,
        "keyword_score": answer.keyword_score,
        "question_relevance": answer.question_relevance,
        "lexical_diversity": answer.lexical_diversity,
        "discourse_score": answer.discourse_score,
        "length_score": min(1.0, cand_tokens / ref_tokens),
        "missing_keywords": answer.missing_keywords or [],
    }
    return generate_question_feedback(
        question_text=question.question_text,
        expected_answer=question.expected_answer,
        candidate_answer=answer.transcript,
        evaluation=evaluation,
        answer_variants=[question.expected_answer] if question.expected_answer else [],
        scoring_rationale="",
        force_template=False,
    )


def _serialize_answer(answer: InterviewAnswer, question: Question) -> dict:
    payload = _build_evaluation_payload(answer, question)
    return {
        "question_id": question.id,
        "question_text": question.question_text,
        "transcript": answer.transcript,
        "final_score": answer.final_score,
        "semantic_score": answer.semantic_score,
        "keyword_score": answer.keyword_score,
        "question_relevance": answer.question_relevance,
        "lexical_diversity": answer.lexical_diversity,
        "discourse_score": answer.discourse_score,
        "penalty": answer.penalty,
        "feedback": answer.feedback,
        "tip": answer.tip,
        "question_summary": payload.get("question_summary"),
        "improvement_bullets": payload.get("improvement_bullets", []),
        "missing_keywords": answer.missing_keywords or [],
    }


def _serialize_session_summary(results: list[dict]) -> dict:
    return generate_session_summary(results)


# POST /api/interview/start/
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
            InterviewSession.user_id == body.user_id,
            InterviewSession.module_id == body.module_id,
            InterviewSession.status == "active",
        )
    )
    # session = session_result.scalar_one_or_none()

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


# POST /api/interview/{session_id}/next/
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

        answers_result = await db.execute(
            select(InterviewAnswer, Question)
            .join(Question, InterviewAnswer.question_id == Question.id)
            .where(InterviewAnswer.session_id == session_id)
            .order_by(Question.order)
        )
        rows = answers_result.all()
        summary = [_serialize_answer(a, q) for a, q in rows]
        session_summary = _serialize_session_summary(summary)
        return NextQuestionResponse(session_id=session_id, completed=True, summary={"results": summary, **session_summary})

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


# POST /api/interview/{session_id}/terminate/
@router.post("/{session_id}/terminate/")
async def terminate_session(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = "completed"
    await db.commit()

    answers_result = await db.execute(
        select(InterviewAnswer, Question)
        .join(Question, InterviewAnswer.question_id == Question.id)
        .where(InterviewAnswer.session_id == session_id)
        .order_by(Question.order)
    )
    rows = answers_result.all()
    results = [_serialize_answer(a, q) for a, q in rows]
    session_summary = _serialize_session_summary(results)
    return {"session_id": session_id, "terminated": True, "results": results, **session_summary}


# GET /api/interview/{session_id}/summary/
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

    results = [_serialize_answer(a, q) for a, q in rows]
    session_summary = _serialize_session_summary(results)

    return {
        "session_id": session_id,
        "completed_questions": len(results),
        "total_questions": total_result.scalar(),
        "session_summary": session_summary.get("session_summary"),
        "overall_strengths": session_summary.get("overall_strengths", []),
        "overall_improvements": session_summary.get("overall_improvements", []),
        "results": results,
    }


# GET /api/interview/{session_id}/detail/
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

    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "module_name": module.module_name if module else "Unknown",
        "total_score": round(avg_result.scalar() or 0.0, 1),
        "session_summary": _serialize_session_summary([_serialize_answer(a, q) for a, q in rows]).get("session_summary"),
        "results": [_serialize_answer(a, q) for a, q in rows],
    }


# GET /api/interview/user-sessions/?user_id={id}
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
            "id": s.id,
            "created_at": s.created_at,
            "module_name": module.module_name if module else "Unknown",
            "total_score": round(avg_result.scalar() or 0.0, 1),
            "question_count": count_result.scalar(),
        })

    return {"data": output}