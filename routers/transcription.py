"""
WebSocket transcription endpoint.

Protocol (client → server):
  - binary frames : raw 16-bit PCM audio at 16kHz mono
  - text "STOP"   : flush, evaluate, save answer to DB, close

Protocol (server → client):
  - {"type": "partial",    "text": "..."}           — live words as spoken
  - {"type": "final",      "text": "..."}           — completed utterance (silence detected)
  - {"type": "evaluation", "transcript": "...", "evaluation": {...}} — after STOP
  - {"type": "error",      "message": "..."}        — on failure
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import AsyncSessionLocal
from ..db.models import InterviewAnswer, InterviewSession, Question
from ..services.evaluation import classify_confirmation_intent, evaluate_answer
from ..services.transcription import create_recognizer, flush_recognizer, process_chunk

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


async def _get_current_question(session: InterviewSession, db: AsyncSession) -> Optional[Question]:
    result = await db.execute(
        select(Question)
        .where(Question.module_id == session.module_id)
        .order_by(Question.order)
        .offset(session.current_index)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _upsert_answer(
    session: InterviewSession,
    question: Question,
    transcript: str,
    evaluation: dict,
    db: AsyncSession,
) -> None:
    """Insert or update the InterviewAnswer row — mirrors Django's update_or_create."""
    result = await db.execute(
        select(InterviewAnswer).where(
            InterviewAnswer.session_id == session.id,
            InterviewAnswer.question_id == question.id,
        )
    )
    answer = result.scalar_one_or_none()

    if answer:
        answer.transcript      = transcript
        answer.semantic_score  = evaluation["semantic_score"]
        answer.keyword_score   = evaluation["keyword_score"]
        answer.final_score     = evaluation["final_score"]
        answer.feedback        = evaluation["feedback"]
        answer.tip             = evaluation["tip"]
        answer.missing_keywords = evaluation["missing_keywords"]
    else:
        answer = InterviewAnswer(
            session_id       = session.id,
            question_id      = question.id,
            transcript       = transcript,
            semantic_score   = evaluation["semantic_score"],
            keyword_score    = evaluation["keyword_score"],
            final_score      = evaluation["final_score"],
            feedback         = evaluation["feedback"],
            tip              = evaluation["tip"],
            missing_keywords = evaluation["missing_keywords"],
            raw_segments     = [],
        )
        db.add(answer)

    await db.commit()


@router.websocket("/ws/transcribe/{session_id}")
async def transcribe_websocket(session_id: int, websocket: WebSocket):
    """
    Live transcription WebSocket for a specific interview session.

    URL params:
        session_id : InterviewSession PK — determines which question to evaluate against.
    """
    await websocket.accept()
    logger.info(f"[session={session_id}] WebSocket connected")

    recognizer = create_recognizer()
    full_transcript_parts: list[str] = []  # accumulates finals across utterances

    # Each WS connection gets its own DB session (not shared across connections)
    async with AsyncSessionLocal() as db:
        # Load the interview session
        result = await db.execute(
            select(InterviewSession)
            .options(selectinload(InterviewSession.module))
            .where(InterviewSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Session {session_id} not found"})
            )
            await websocket.close()
            return

        if session.status == "completed":
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Interview session already completed"})
            )
            await websocket.close()
            return

        current_question = await _get_current_question(session, db)
        if not current_question:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No question available at current index"})
            )
            await websocket.close()
            return

        logger.info(
            f"[session={session_id}] Q#{session.current_index}: {current_question.question_text[:60]}..."
        )

        try:
            while True:
                message = await websocket.receive()

                # ── Binary: raw PCM chunk from browser ScriptProcessorNode ────
                if "bytes" in message:
                    pcm_bytes = message["bytes"]

                    if len(pcm_bytes) < 2:
                        continue  # skip empty / malformed chunks

                    partial_text, final_text = process_chunk(recognizer, pcm_bytes)

                    if partial_text:
                        await websocket.send_text(
                            json.dumps({"type": "partial", "text": partial_text})
                        )

                    if final_text:
                        # An utterance completed (Vosk detected silence)
                        full_transcript_parts.append(final_text)
                        await websocket.send_text(
                            json.dumps({"type": "final", "text": final_text})
                        )
                        logger.debug(f"[session={session_id}] utterance: {final_text}")

                # ── Text: control messages ────────────────────────────────────
                elif "text" in message:
                    command = message["text"].strip().upper()

                    if command == "STOP":
                        # Flush any remaining audio in the recognizer buffer
                        leftover = flush_recognizer(recognizer)
                        if leftover:
                            full_transcript_parts.append(leftover)
                            await websocket.send_text(
                                json.dumps({"type": "final", "text": leftover})
                            )

                        # Build complete transcript for this question
                        full_transcript = " ".join(full_transcript_parts).strip()
                        logger.info(
                            f"[session={session_id}] STOP received. "
                            f"Full transcript ({len(full_transcript)} chars): {full_transcript[:80]}"
                        )

                        # Evaluate against the current question
                        question_data = {
                            "id": current_question.id,
                            "question": current_question.question_text,
                            "answer": current_question.expected_answer,
                        }
                        evaluation = evaluate_answer(
                            transcript=full_transcript,
                            question_data=question_data,
                            model_path=session.module.model_pkl_path if session.module else None,
                        )

                        # Persist to DB
                        await _upsert_answer(session, current_question, full_transcript, evaluation, db)
                        logger.info(
                            f"[session={session_id}] Answer saved. "
                            f"score={evaluation['final_score']}"
                        )

                        # Send evaluation back to client
                        await websocket.send_text(json.dumps({
                            "type": "evaluation",
                            "transcript": full_transcript,
                            "evaluation": {
                                "score":           evaluation["final_score"],
                                "semantic_score":  evaluation["semantic_score"],
                                "keyword_score":   evaluation["keyword_score"],
                                "feedback":        evaluation["feedback"],
                                "tip":             evaluation["tip"],
                                "missing_keywords": evaluation["missing_keywords"],
                            },
                        }))

                        await websocket.close()
                        break

        except WebSocketDisconnect:
            logger.info(f"[session={session_id}] Client disconnected mid-session")
        except Exception as exc:
            logger.exception(f"[session={session_id}] Unhandled error: {exc}")
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": str(exc)})
                )
            except Exception:
                pass


@router.websocket("/ws/intent/{session_id}")
async def intent_websocket(session_id: int, websocket: WebSocket):
    """
    Classify candidate's spoken intent ("next" / "repeat" / "skip").
    Expects the same raw PCM stream — send "STOP" when done speaking.
    """
    await websocket.accept()
    logger.info(f"[intent session={session_id}] Connected")

    recognizer = create_recognizer()
    parts: list[str] = []

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                _, final_text = process_chunk(recognizer, message["bytes"])
                if final_text:
                    parts.append(final_text)

            elif "text" in message and message["text"].strip().upper() == "STOP":
                leftover = flush_recognizer(recognizer)
                if leftover:
                    parts.append(leftover)

                transcript = " ".join(parts).strip()
                intent = classify_confirmation_intent(transcript)
                logger.info(f"[intent session={session_id}] transcript='{transcript}' intent={intent}")

                await websocket.send_text(json.dumps({
                    "session_id": session_id,
                    "intent": intent,
                    "transcript": transcript,
                }))
                await websocket.close()
                break

    except WebSocketDisconnect:
        logger.info(f"[intent session={session_id}] Disconnected")