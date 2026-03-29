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
import time

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

        final_transcript: str = ""
        last_partial: str = ""
        # Mutable state accessible from nested async functions
        state = {
            "first_audio_at": None,   # when first PCM chunk arrived
            "evaluation_sent": False,
        }

        async def handle_end_of_turn(transcript: str) -> None:
            if state["evaluation_sent"]:
                return

            # ✅ Guard: reject suspiciously short transcripts
            # Mid-sentence interruptions produce short incomplete text
            word_count = len(transcript.split())
            if word_count < 5:
                logger.info(
                    f"[session={session_id}] EndOfTurn ignored — only {word_count} words: '{transcript}'"
                )
                return

            state["evaluation_sent"] = True

            current = transcript or last_partial
            full_transcript = (
                f"{prior.strip()} {current}".strip()
                if prior.strip() else current
            )

            logger.info(f"[session={session_id}] EndOfTurn → evaluating: {full_transcript[:80]}")

            question_data = {
                "id": current_question.id,
                "question": current_question.question_text,
                "answer": current_question.expected_answer,
            }
            evaluation = evaluate_answer(
                transcript=full_transcript,
                question_data=question_data,
                model_path=(session.module.model_pkl_path if session.module else None),
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

        try:
            async with _dg_client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate="16000",
                eot_timeout_ms="5000",   # ✅ generous pause tolerance
            ) as dg_conn:

                def on_message(message) -> None:
                    nonlocal final_transcript, last_partial
                    if getattr(message, "type", "") != "TurnInfo":
                        return

                    transcript = getattr(message, "transcript", "").strip()
                    if not transcript:
                        return

                    event = getattr(message, "event", "")
                    logger.debug(f"[session={session_id}] {event}: {transcript[:60]}")

                    if event == "Update":
                        last_partial = transcript
                        print("update: ", transcript)
                    elif event == "EndOfTurn":
                        final_transcript = transcript
                        last_partial = ""
                        print("end of turn: ", transcript)
                        asyncio.create_task(handle_end_of_turn(transcript))

                def on_error(error) -> None:
                    logger.error(f"[session={session_id}] Deepgram error: {error}")
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
                            if len(pcm) >= 2:
                                # ✅ Track when first audio arrives
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

@router.websocket("/ws/intent/{session_id}")
async def intent_websocket(session_id: int, websocket: WebSocket):
    await websocket.accept()
    logger.info(f"[intent session={session_id}] connected")

    last_partial: str = ""
    intent_sent = False

    async def handle_intent_end_of_turn(transcript: str) -> None:
        nonlocal intent_sent
        if intent_sent:
            return
        intent_sent = True
        intent = classify_confirmation_intent(transcript)
        logger.info(f"[intent session={session_id}] transcript='{transcript}' intent={intent}")
        await websocket.send_text(json.dumps({
            "session_id": session_id,
            "intent": intent,
            "transcript": transcript,
        }))

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
            dg_conn.on(EventType.ERROR, on_error)

            listen_task = asyncio.create_task(dg_conn.start_listening())

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

    except WebSocketDisconnect:
        logger.info(f"[intent session={session_id}] disconnected")
    except Exception as exc:
        logger.exception(f"[intent session={session_id}] error: {exc}")
        try:
            await websocket.send_text(json.dumps({"intent": "next"}))
        except Exception:
            pass