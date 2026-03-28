"""
Interview FastAPI Server
========================
Start:
    uvicorn interview_fastapi.main:app --host 0.0.0.0 --port 8001 --reload

Route alignment with frontend API_BASE = "http://localhost:8001/api":
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
    WS   /ws/transcribe/{session_id}
    WS   /ws/intent/{session_id}
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import hierarchy, sessions, stats, transcription
from .services.transcription import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Starting up — loading Vosk model...")
#     load_model()
#     logger.info("Ready to accept connections.")
#     yield
#     logger.info("Shutting down.")


app = FastAPI(
    title="Interview Transcription API",
    version="1.0.0",
    # lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hierarchy.router)   # /api/domains/, /api/subdomains/, /api/modules/
app.include_router(sessions.router)   # /api/interview/...
app.include_router(stats.router)      # /api/performance-stats/
app.include_router(transcription.router)  # /ws/transcribe/{id}, /ws/intent/{id}


@app.get("/health")
async def health():
    return {"status": "ok"}