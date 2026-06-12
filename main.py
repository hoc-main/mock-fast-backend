"""
Interview FastAPI Server
========================
Start:
    uvicorn mock_fast_backend.main:app --host 0.0.0.0 --port 8001 --reload

Routes:
    GET  /api/domains/
    GET  /api/domains/{id}/subdomains/
    GET  /api/subdomains/{id}/modules/
    GET  /api/modules/{id}/
    POST /api/interview/start/
    POST /api/interview/{id}/next/
    POST /api/interview/{id}/terminate/
    GET  /api/interview/{id}/summary/
    GET  /api/interview/{id}/detail/
    GET  /api/interview/user-sessions/?user_id=
    GET  /api/performance-stats/?user_id=
    POST /api/feedback/question
    POST /api/feedback/session
    POST /api/feedback/session/from-db
    WS   /ws/transcribe/{session_id}
    WS   /ws/intent/{session_id}
    GET  /health
"""
import logging
import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import hierarchy, sessions, stats, transcription, feedback, tts, jobs
from .services.llm_feedback import check_llm_available
from .db.database import engine, Base
from .db import models  # noqa: ensure models are registered

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Interview Transcription API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hierarchy.router)      # /api/domains/, /api/subdomains/, /api/modules/
app.include_router(sessions.router)       # /api/interview/...
app.include_router(stats.router)          # /api/performance-stats/
app.include_router(transcription.router)  # /ws/transcribe/{id}, /ws/intent/{id}
app.include_router(feedback.router)       # /api/feedback/...
app.include_router(tts.router)            # /api/tts/
app.include_router(jobs.router)           # /api/jobs/


@app.on_event("startup")
async def startup_event():
    # Create tables (SQLite) or add missing columns (PostgreSQL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Auto-add new columns that don't exist yet (create_all won't ALTER existing tables)
        await _auto_add_missing_columns(conn)
    check_llm_available()


async def _auto_add_missing_columns(conn):
    """Add columns introduced after initial table creation. Safe to re-run."""
    from sqlalchemy import text
    
    # Detect if PostgreSQL or SQLite
    is_pg = "postgresql" in str(conn.engine.url)
    
    migrations = [
        ("interview_interviewsession", "current_question_id", "INTEGER NULL"),
        ("interview_interviewsession", "conversation_history", "JSONB DEFAULT '[]'::jsonb" if is_pg else "TEXT DEFAULT '[]'"),
        ("interview_interviewsession", "asked_question_ids", "JSONB DEFAULT '[]'::jsonb" if is_pg else "TEXT DEFAULT '[]'"),
    ]
    for table, column, col_def in migrations:
        try:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
            logger.info(f"Auto-added column: {table}.{column}")
        except Exception:
            pass  # Column already exists


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/test/conversation-agent")
async def test_conversation_agent(body: dict):
    """Test endpoint for conversation agent — feed dummy data to verify Groq picks next question."""
    from .services.conversation_agent import pick_next_question
    result = await pick_next_question(
        remaining_questions=body.get("remaining_questions", []),
        conversation_history=body.get("conversation_history", []),
        module_topic=body.get("module_topic", "technical"),
    )
    if result:
        return {"status": "success", "question_id": result.question_id, "transition": result.transition, "reasoning": result.reasoning}
    return {"status": "failed", "message": "LLM unavailable or parse error"}
