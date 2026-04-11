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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import hierarchy, sessions, stats, transcription, feedback, tts

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


@app.get("/health")
async def health():
    return {"status": "ok"}
