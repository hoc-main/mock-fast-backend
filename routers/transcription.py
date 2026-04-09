"""
WebSocket transcription endpoint — Deepgram Flux backend.

Protocol (client → server):
  - binary frames : raw 16-bit PCM at 16 kHz mono
  - text "STOP"   : flush, evaluate, save, close

Protocol (server → client):
  Instant messages (always available):
    {"type": "partial",         "text": "..."}
    {"type": "silence",         "message": "..."}
    {"type": "still_listening", "partial_transcript": "...", "message": "..."}
    {"type": "repeat_question", "question": "...", "topic": "..."}
    {"type": "skipped",         "completed": bool, ...}
    {"type": "session_ended",   "message": "..."}
    {"type": "evaluation",      "transcript": "...", "evaluation": {...}}
    {"type": "error",           "message": "..."}
"""

import asyncio
import json
import logging
import time
from typing import Optional

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import AsyncSessionLocal
from ..db.models import InterviewAnswer, InterviewSession, Question
from ..services.evaluation import evaluate_answer
from ..services.voice_intent import classify_voice_intent, classify_nav_intent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])

_dg_client = AsyncDeepgramClient()


# ── DB helpers ─────────────────────────────────────────────────────────────────

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


async def _question_count(module_id: int, db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(Question)
        .where(Question.module_id == module_id)
    )
    return result.scalar()


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


# ── transcription WebSocket ────────────────────────────────────────────────────

@router.websocket("/ws/transcribe/{session_id}")
async def transcribe_websocket(
    session_id: int,
    websocket: WebSocket,
    prior: str = Query(default=""),
):
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
            await websocket.send_text(json.dumps(
                {"type": "error", "message": f"Session {session_id} not found"}
            ))
            await websocket.close()
            return

        if session.status == "completed":
            await websocket.send_text(json.dumps(
                {"type": "error", "message": "Session already completed"}
            ))
            await websocket.close()
            return

        current_question = await _get_current_question(session, db)
        if not current_question:
            await websocket.send_text(json.dumps(
                {"type": "error", "message": "No question at current index"}
            ))
            await websocket.close()
            return

        logger.info(
            f"[session={session_id}] Q#{session.current_index}: "
            f"{current_question.question_text[:60]}..."
        )

        final_transcript: str = ""
        last_partial:     str = ""
        state = {
            "first_audio_at":  None,
            "evaluation_sent": False,
            # Incremented on every Deepgram Update event.
            # > 0 means user was actively speaking before this EndOfTurn —
            # used to detect premature EndOfTurn on a mid-sentence pause.
            "update_count": 0,
        }

        async def handle_end_of_turn(raw_transcript: str) -> None:
            """
            Called on Deepgram EndOfTurn or STOP command.

            Step 1: classify intent (silence / repeat / skip / end / answer)
            Step 2: act — only "answer" passes through to NLP evaluation.
            """
            if state["evaluation_sent"]:
                return
            state["evaluation_sent"] = True

            current = (raw_transcript or last_partial).strip()

            # ── classify ───────────────────────────────────────────────────────
            intent_result = classify_voice_intent(current)
            logger.info(
                f"[session={session_id}] intent={intent_result.intent} "
                f"({intent_result.confidence:.2f}) — {intent_result.reason}"
            )

            # ── silence ────────────────────────────────────────────────────────
            if intent_result.intent == "silence":
                await websocket.send_text(json.dumps({
                    "type":    "silence",
                    "message": "We didn't catch anything. Please go ahead and answer when ready.",
                }))
                state["evaluation_sent"] = False
                return

            # ── repeat question ────────────────────────────────────────────────
            if intent_result.intent == "repeat":
                logger.info(f"[session={session_id}] sending repeat_question")
                await websocket.send_text(json.dumps({
                    "type":     "repeat_question",
                    "question": current_question.question_text,
                    "topic":    current_question.topic or "",
                }))
                state["evaluation_sent"] = False
                return

            # ── skip question ──────────────────────────────────────────────────
            if intent_result.intent == "skip":
                logger.info(f"[session={session_id}] skipping question")
                total = await _question_count(session.module_id, db)
                session.current_index += 1
                await db.commit()
                await db.refresh(session)

                if session.current_index >= total:
                    session.status = "completed"
                    await db.commit()
                    await websocket.send_text(json.dumps({
                        "type":      "skipped",
                        "completed": True,
                        "message":   "Last question skipped. Interview complete.",
                    }))
                else:
                    next_q = await _get_current_question(session, db)
                    await websocket.send_text(json.dumps({
                        "type":            "skipped",
                        "completed":       False,
                        "question_index":  session.current_index,
                        "total_questions": total,
                        "next_question": {
                            "id":       next_q.id,
                            "topic":    next_q.topic or "",
                            "question": next_q.question_text,
                        },
                    }))
                return

            # ── end interview ──────────────────────────────────────────────────
            if intent_result.intent == "end":
                logger.info(f"[session={session_id}] ending interview on user request")
                session.status = "completed"
                await db.commit()
                await websocket.send_text(json.dumps({
                    "type":    "session_ended",
                    "message": "Interview ended at your request.",
                }))
                await websocket.close()
                return


            # ── answer → non-blocking turn finish ──────────────────────────────
            full_transcript = (
                f"{prior.strip()} {current}".strip()
                if prior.strip() else current
            )
            logger.info(f"[session={session_id}] turn finished: {full_transcript[:80]}")

            # Note: evaluate_answer and _upsert_answer are DEFERRED 
            # until the user confirms they are finished via the frontend.

            await websocket.send_text(json.dumps({
                "type":       "turn_finished",
                "transcript": full_transcript,
            }))

        # ── Deepgram connection ────────────────────────────────────────────────
        try:
            async with _dg_client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate="16000",
                eot_timeout_ms="8000",
                eot_threshold="0.9",
            ) as dg_conn:

                def on_message(message) -> None:
                    nonlocal final_transcript, last_partial
                    if getattr(message, "type", "") != "TurnInfo":
                        return
                    transcript = getattr(message, "transcript", "").strip()
                    event      = getattr(message, "event", "")

                    if event == "Update":
                        last_partial = transcript
                        state["update_count"] += 1   # user is actively speaking
                        if transcript:
                            asyncio.create_task(websocket.send_text(json.dumps({
                                "type": "partial", "text": transcript,
                            })))
                    elif event == "EndOfTurn":
                        final_transcript = transcript
                        last_partial     = ""
                        logger.debug(
                            f"[session={session_id}] EndOfTurn: '{transcript[:60]}'"
                        )
                        asyncio.create_task(handle_end_of_turn(transcript))

                def on_error(error) -> None:
                    logger.error(f"[session={session_id}] Deepgram error: {error}")
                    asyncio.create_task(websocket.send_text(json.dumps({
                        "type": "error", "message": str(error),
                    })))

                dg_conn.on(EventType.MESSAGE, on_message)
                dg_conn.on(EventType.ERROR,   on_error)
                listen_task = asyncio.create_task(dg_conn.start_listening())

                try:
                    while True:
                        message = await websocket.receive()
                        if "bytes" in message:
                            pcm = message["bytes"]
                            if len(pcm) >= 2:
                                if state["first_audio_at"] is None:
                                    state["first_audio_at"] = time.time()
                                await dg_conn._send(pcm)
                        elif "text" in message:
                            if message["text"].strip().upper() == "STOP":
                                listen_task.cancel()
                                if not state["evaluation_sent"]:
                                    await handle_end_of_turn(final_transcript or last_partial)
                                await websocket.close()
                                break

                except (WebSocketDisconnect, RuntimeError) as exc:
                    if (
                        isinstance(exc, RuntimeError)
                        and "disconnect" not in str(exc).lower()
                    ):
                        raise
                    logger.info(f"[session={session_id}] client disconnected (normal)")
                    listen_task.cancel()

                except Exception as exc:
                    logger.exception(f"[session={session_id}] error: {exc}")
                    listen_task.cancel()
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error", "message": str(exc),
                        }))
                    except Exception:
                        pass

        except Exception as exc:
            logger.exception(
                f"[session={session_id}] Deepgram connection failed: {exc}"
            )
            try:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": str(exc),
                }))
                await websocket.close()
            except Exception:
                pass


# ── intent WebSocket (between questions) ──────────────────────────────────────

@router.websocket("/ws/intent/{session_id}")
async def intent_websocket(session_id: int, websocket: WebSocket):
    """
    Between-question navigation intent.
    Returns: {"session_id": N, "intent": "next|continue|repeat|skip|end",
              "transcript": "...", "confidence": 0.9}
    """
    await websocket.accept()
    logger.info(f"[intent session={session_id}] connected")
    last_partial: str = ""
    intent_sent = False

    async def handle_intent_end_of_turn(transcript: str) -> None:
        nonlocal intent_sent
        if intent_sent:
            return
        intent_sent = True
        result = classify_nav_intent(transcript)
        logger.info(
            f"[intent session={session_id}] intent={result.intent} "
            f"({result.confidence:.2f}) transcript='{transcript}'"
        )
        try:
            await websocket.send_text(json.dumps({
                "session_id": session_id,
                "intent":     result.intent,
                "transcript": transcript,
                "confidence": round(result.confidence, 2),
            }))
        except Exception:
            pass

    try:
        async with _dg_client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate="16000",
            eot_timeout_ms="1800",
        ) as dg_conn:

            def on_message(message) -> None:
                nonlocal last_partial
                if getattr(message, "type", "") != "TurnInfo":
                    return
                transcript = getattr(message, "transcript", "").strip()
                if not transcript:
                    return
                event = getattr(message, "event", "")
                if event == "Update":
                    last_partial = transcript
                elif event == "EndOfTurn":
                    asyncio.create_task(handle_intent_end_of_turn(transcript))

            def on_error(error) -> None:
                logger.error(f"[intent session={session_id}] Deepgram error: {error}")

            dg_conn.on(EventType.MESSAGE, on_message)
            dg_conn.on(EventType.ERROR,   on_error)
            listen_task = asyncio.create_task(dg_conn.start_listening())

            try:
                while True:
                    message = await websocket.receive()
                    if "bytes" in message:
                        pcm = message["bytes"]
                        if len(pcm) >= 2:
                            await dg_conn._send(pcm)
                    elif "text" in message:
                        if message["text"].strip().upper() == "STOP":
                            listen_task.cancel()
                            if not intent_sent:
                                await handle_intent_end_of_turn(last_partial)
                            await websocket.close()
                            break

            except (WebSocketDisconnect, RuntimeError) as exc:
                if (
                    isinstance(exc, RuntimeError)
                    and "disconnect" not in str(exc).lower()
                ):
                    raise
                logger.info(f"[intent session={session_id}] client disconnected")
                listen_task.cancel()

    except Exception as exc:
        logger.exception(f"[intent session={session_id}] error: {exc}")
        try:
            await websocket.send_text(json.dumps({
                "session_id": session_id,
                "intent":     "next",
                "transcript": "",
                "confidence": 0.0,
            }))
        except Exception:
            pass
