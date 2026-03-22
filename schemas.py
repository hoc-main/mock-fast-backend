from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel


# ── Shared ────────────────────────────────────────────────────────────────────

class QuestionOut(BaseModel):
    id: int
    topic: Optional[str]
    question: str
    answer: str


class EvaluationOut(BaseModel):
    score: float
    semantic_score: float
    keyword_score: float
    question_relevance: float
    lexical_diversity: float
    discourse_score: float
    penalty: float
    feedback: str
    tip: str
    missing_keywords: List[str]


# ── Session ───────────────────────────────────────────────────────────────────

class StartInterviewRequest(BaseModel):
    user_id: int
    module_id: int


class StartInterviewResponse(BaseModel):
    session_id: int
    question: QuestionOut
    question_index: int
    total_questions: int


class NextQuestionResponse(BaseModel):
    session_id: int
    completed: bool
    question: Optional[QuestionOut] = None
    question_index: Optional[int] = None
    total_questions: Optional[int] = None
    summary: Optional[List[Any]] = None


# ── Transcription WebSocket messages ─────────────────────────────────────────
# These are JSON-encoded and sent over the WS.

class WSPartialMessage(BaseModel):
    type: str = "partial"
    text: str


class WSFinalMessage(BaseModel):
    type: str = "final"
    text: str


class WSEvaluationMessage(BaseModel):
    type: str = "evaluation"
    transcript: str
    evaluation: EvaluationOut


class WSErrorMessage(BaseModel):
    type: str = "error"
    message: str


# ── Intent ────────────────────────────────────────────────────────────────────

class IntentResponse(BaseModel):
    session_id: int
    intent: str       # "next" | "repeat" | "skip"
    transcript: str


# ── Summary & Stats ───────────────────────────────────────────────────────────

class AnswerResult(BaseModel):
    question_id: Optional[int]
    question_text: str
    transcript: str
    final_score: float
    semantic_score: float
    keyword_score: float
    question_relevance: float = 0.0
    lexical_diversity: float = 0.0
    discourse_score: float = 0.0
    penalty: float = 0.0
    feedback: str
    tip: str
    missing_keywords: List[Any]


class SummaryResponse(BaseModel):
    session_id: int
    completed_questions: int
    total_questions: int
    results: List[AnswerResult]


class SessionListItem(BaseModel):
    id: int
    created_at: datetime
    module_name: str
    total_score: float
    question_count: int


class SessionDetailResponse(BaseModel):
    session_id: int
    created_at: datetime
    module_name: str
    total_score: float
    results: List[AnswerResult]


class PerformanceStatsResponse(BaseModel):
    attempts: int
    latest_score: float
    avg_score: float
    confidence_score: float
    technical_score: float
    global_rank: Optional[int]


# ── Hierarchy ─────────────────────────────────────────────────────────────────

class ModuleOut(BaseModel):
    id: int
    module_name: str
    slug: str
    is_free: bool = True
    question_count: int = 0

class SubdomainOut(BaseModel):
    id: int
    name: str
    slug: str
    modules: List[ModuleOut] = []

class DomainOut(BaseModel):
    id: int
    name: str
    slug: str
    subdomains: List[SubdomainOut] = []

class DomainListOut(BaseModel):
    data: List[DomainOut]

class ModuleDetailOut(ModuleOut):
    domain_name: Optional[str] = None
    subdomain_name: Optional[str] = None