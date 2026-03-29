"""
WebSocket transcription endpoint — Deepgram Flux backend.

Flux handles end-of-turn detection natively via EndOfTurn events.
No more manual silence timers or VAD on the backend.

Protocol (client → server):  UNCHANGED
  - binary frames : raw 16-bit PCM at 16kHz mono
  - text "STOP"   : flush, evaluate, save, close

Protocol (server → client):  UNCHANGED
  - {"type": "partial",    "text": "..."}
  - {"type": "final",      "text": "..."}
  - {"type": "evaluation", "transcript": "...", "evaluation": {...}}
  - {"type": "error",      "message": "..."}
"""
import asyncio
import json
import logging
import os
from typing import Optional

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import AsyncSessionLocal
from ..db.models import InterviewAnswer, InterviewSession, Question
from ..services.evaluation import classify_confirmation_intent, evaluate_answer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])

# ── Deepgram client ───────────────────────────────────────────────────────────
# AsyncDeepgramClient reads DEEPGRAM_API_KEY from env automatically
_dg_client = AsyncDeepgramClient()


# ── DB helpers (unchanged) ────────────────────────────────────────────────────

async def _get_current_question(
    session: InterviewSession, db: AsyncSession
) -> Optional[Question]:
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


# ─────────────────────────────────────────────────────────────────────────────
# /ws/transcribe/{session_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/transcribe/{session_id}")
async def transcribe_websocket(session_id: int, websocket: WebSocket, prior: str = Query(default="")):
    await websocket.accept()
    logger.info(f"[session={session_id}] WebSocket connected")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(InterviewSession)
            .options(selectinload(InterviewSession.module))
            .where(InterviewSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Session {session_id} not found"}))
            await websocket.close()
            return

        if session.status == "completed":
            await websocket.send_text(json.dumps({"type": "error", "message": "Session already completed"}))
            await websocket.close()
            return

        current_question = await _get_current_question(session, db)
        if not current_question:
            await websocket.send_text(json.dumps({"type": "error", "message": "No question at current index"}))
            await websocket.close()
            return

        logger.info(f"[session={session_id}] Q#{session.current_index}: {current_question.question_text[:60]}...")

        final_parts: list[str] = []
        last_partial: str = ""
        audio_paused = False

        try:
            async with _dg_client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate="16000",
            ) as dg_conn:

                def on_message(message) -> None:
                    nonlocal last_partial
                    if getattr(message, "type", "") != "TurnInfo":
                        return

                    transcript = getattr(message, "transcript", "").strip()
                    if not transcript:
                        return

                    event = getattr(message, "event", "")
                    print(f"[session={session_id}] event={event} transcript={transcript[:60]}")

                    last_partial = transcript

                def on_error(error) -> None:
                    print(f"[session={session_id}] Deepgram error: {error}")
                    asyncio.create_task(
                        websocket.send_text(json.dumps({"type": "error", "message": str(error)}))
                    )

                dg_conn.on(EventType.MESSAGE, on_message)
                dg_conn.on(EventType.ERROR, on_error)

                listen_task = asyncio.create_task(dg_conn.start_listening())

                try:
                    while True:
                        message = await websocket.receive()

                        if "bytes" in message:
                            pcm = message["bytes"]
                            if len(pcm) >= 2 and not audio_paused:
                                await dg_conn._send(pcm)

                        elif "text" in message:
                            text = message["text"].strip().upper()

                            if text == "PAUSE":
                                audio_paused = True
                                logger.info(f"[session={session_id}] PAUSE — stopping audio forward, waiting for EndOfTurn")
                                continue

                            if text != "STOP":
                                continue

                            # Give Flux time to fire EndOfTurn after the silence gap
                            await asyncio.sleep(0.8)
                            listen_task.cancel()

                            # Merge committed turns + any uncommitted partial
                            parts_to_join = last_partial

                            current_transcript = " ".join(parts_to_join).strip()

                            # Prepend prior continuation if this is a "continue" chunk
                            full_transcript = (
                                f"{prior.strip()} {current_transcript}".strip()
                                if prior.strip()
                                else current_transcript
                            )

                            logger.info(
                                f"[session={session_id}] STOP. "
                                f"transcript ({len(full_transcript)} chars): {full_transcript[:80]}"
                            )

                            question_data = {
                                "id": current_question.id,
                                "question": current_question.question_text,
                                "answer": current_question.expected_answer,
                            }
                            evaluation = evaluate_answer(
                                transcript=full_transcript,
                                question_data=question_data,
                                model_path=(
                                    session.module.model_pkl_path if session.module else None
                                ),
                            )

                            await _upsert_answer(session, current_question, full_transcript, evaluation, db)
                            logger.info(f"[session={session_id}] saved. score={evaluation['final_score']}")

                            await websocket.send_text(json.dumps({
                                "type": "evaluation",
                                "transcript": full_transcript,
                                "evaluation": {
                                    "score":            evaluation["final_score"],
                                    "semantic_score":   evaluation["semantic_score"],
                                    "keyword_score":    evaluation["keyword_score"],
                                    "feedback":         evaluation["feedback"],
                                    "tip":              evaluation["tip"],
                                    "missing_keywords": evaluation["missing_keywords"],
                                },
                            }))

                            await websocket.close()
                            break

                except WebSocketDisconnect:
                    logger.info(f"[session={session_id}] client disconnected")
                    listen_task.cancel()
                except Exception as exc:
                    logger.exception(f"[session={session_id}] error: {exc}")
                    listen_task.cancel()
                    try:
                        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
                    except Exception:
                        pass

        except Exception as exc:
            logger.exception(f"[session={session_id}] Flux connection failed: {exc}")
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
                await websocket.close()
            except Exception:
                pass
# ─────────────────────────────────────────────────────────────────────────────
# /ws/intent/{session_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/intent/{session_id}")
async def intent_websocket(session_id: int, websocket: WebSocket):
    await websocket.accept()
    logger.info(f"[intent session={session_id}] connected")

    parts: list[str] = []

    try:
        # Intent answers are short — use tighter eot_timeout
        async with _dg_client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate="16000",
            eot_timeout_ms="800",   # short: "yes / no / continue" answers
        ) as dg_conn:

            def on_message(message) -> None:
                msg_type = getattr(message, "type", "")
                if msg_type != "TurnInfo":
                    return
                transcript = getattr(message, "transcript", "").strip()
                event = getattr(message, "event", "")
                if transcript and event == "EndOfTurn":
                    parts.append(transcript)

            def on_error(error) -> None:
                logger.error(f"[intent session={session_id}] Deepgram error: {error}")

            dg_conn.on(EventType.MESSAGE, on_message)
            dg_conn.on(EventType.ERROR,   on_error)

            listen_task = asyncio.create_task(dg_conn.start_listening())

            while True:
                message = await websocket.receive()

                if "bytes" in message:
                    pcm = message["bytes"]
                    if len(pcm) >= 2:
                        await dg_conn._send(pcm)

                elif "text" in message and message["text"].strip().upper() == "STOP":
                    await asyncio.sleep(0.3)
                    listen_task.cancel()

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
        logger.info(f"[intent session={session_id}] disconnected")
    except Exception as exc:
        logger.exception(f"[intent session={session_id}] error: {exc}")
        try:
            await websocket.send_text(json.dumps({"intent": "next"}))
        except Exception:
            pass