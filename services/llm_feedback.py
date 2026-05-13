"""
llm_feedback.py
================
LLM-powered feedback via LangChain + Groq with structured output
and conversation buffer memory to eliminate chain-of-thought leaks.

Environment variables (add to .env):
    GROQ_API_KEY     your groq api key
    GROQ_MODEL       openai/gpt-oss-20b
    GROQ_TIMEOUT     15
"""

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Optional, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, asc, func

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = ""
GROQ_MODEL:   str = "openai/gpt-oss-20b"
GROQ_TIMEOUT: int = 15
llm_available: bool = False
_chain = None

# ── LRU cache ─────────────────────────────────────────────────────────────────
_CACHE_MAX = 50
_response_cache: OrderedDict[str, dict] = OrderedDict()


def _cache_key(question: str, score: float) -> str:
    return f"{question.strip().lower()[:80]}|{round(score, 1)}"


def _cache_put(key: str, result: dict) -> None:
    _response_cache[key] = result
    _response_cache.move_to_end(key)
    if len(_response_cache) > _CACHE_MAX:
        _response_cache.popitem(last=False)


def _cache_get(key: str) -> Optional[dict]:
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None


# ── structured output schema ──────────────────────────────────────────────────
class FeedbackOutput(BaseModel):
    feedback: str = Field(description="2-3 sentences of detailed feedback, 40-60 words. Address candidate as 'you'.")
    tip: str = Field(description="2-3 sentences of professional improvement tip with concrete example, 30-50 words.")
    tts_feedback: str = Field(description="One short natural sentence under 15 words summarizing how the answer went.")


# ── few-shot examples (pre-loaded into memory) ───────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "human": """QUESTION: What is polymorphism in OOP?
CANDIDATE'S ANSWER: Polymorphism is when objects can take different forms.
EVALUATION SUMMARY:
- Overall quality: weak (score 0.25/1.00)
- Key issues identified: the answer is too short; key concepts missing: method overriding, method overloading, interfaces.""",
        "ai": FeedbackOutput(
            feedback="You correctly identified that polymorphism involves objects taking different forms, which shows basic awareness. However, your answer lacks depth — you didn't explain method overriding, overloading, or how interfaces enable polymorphic behavior, which are essential to a complete answer.",
            tip="Structure your response by first defining polymorphism, then distinguishing compile-time (overloading) from runtime (overriding) polymorphism. For example: 'A Dog and Cat class both override an Animal.speak() method, producing different outputs.'",
            tts_feedback="Your answer was too brief and missed key polymorphism concepts.",
        ),
    },
    {
        "human": """QUESTION: Explain the event loop in Node.js.
CANDIDATE'S ANSWER: The event loop is what makes Node.js non-blocking. It handles callbacks by checking the callback queue after the call stack is empty. This allows Node to handle many connections with a single thread using phases like timers, poll, and check.
EVALUATION SUMMARY:
- Overall quality: good (score 0.78/1.00)
- The answer is mostly correct but could be more complete.""",
        "ai": FeedbackOutput(
            feedback="You gave a solid explanation of the event loop's role in non-blocking I/O and correctly mentioned the callback queue and call stack relationship. You also referenced specific phases, which shows deeper understanding. To strengthen this further, mention the microtask queue and how promises are prioritized.",
            tip="Add a brief example showing how setTimeout and Promise.resolve callbacks execute in different order due to the microtask queue having higher priority than the timer phase.",
            tts_feedback="Strong answer with good technical depth on the event loop.",
        ),
    },
]

# ── system prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert interview coach. You evaluate mock interview answers and provide structured feedback.

RULES:
- Never think out loud, count words, or repeat instructions.
- Never wrap output in quotes or markdown.
- Address the candidate as "you".
- Do not copy the reference answer.
- Do not mention scores or metrics.
- Be specific to the question asked."""


def check_llm_available() -> bool:
    global llm_available, _chain, GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "15"))
    if not GROQ_API_KEY:
        llm_available = False
        logger.warning("GROQ_API_KEY not set — LLM feedback disabled, rule-based fallback active")
        return False
    try:
        is_reasoning = "oss" in GROQ_MODEL or "reasoning" in GROQ_MODEL
        kwargs = dict(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.35,
            max_tokens=800 if is_reasoning else 450,
        )
        if is_reasoning:
            kwargs["reasoning_effort"] = "medium"

        llm = ChatGroq(**kwargs)

        # Build few-shot messages for memory buffer
        few_shot_messages = []
        for ex in _FEW_SHOT_EXAMPLES:
            few_shot_messages.append(HumanMessage(content=ex["human"]))
            ai_obj = ex["ai"]
            few_shot_messages.append(AIMessage(content=(
                f"FEEDBACK: {ai_obj.feedback}\n"
                f"TIP: {ai_obj.tip}\n"
                f"TTS: {ai_obj.tts_feedback}"
            )))

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_SYSTEM_PROMPT),
            *few_shot_messages,
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

        _chain = prompt | llm.with_structured_output(FeedbackOutput)
        llm_available = True
        logger.info(f"LangChain+Groq initialized — LLM feedback enabled ({GROQ_MODEL})")
        return True
    except Exception as exc:
        llm_available = False
        logger.warning(f"LangChain init failed — LLM feedback disabled. ({exc})")
        return False


# ── conversation history buffer (last N good exchanges) ──────────────────────
_MAX_HISTORY = 6
_history: list = []
_kb_loaded: bool = False


def _add_to_history(human_msg: str, result: dict) -> None:
    _history.append(HumanMessage(content=human_msg))
    _history.append(AIMessage(content=(
        f"FEEDBACK: {result['feedback']}\n"
        f"TIP: {result['tip']}\n"
        f"TTS: {result['tts_feedback']}"
    )))
    while len(_history) > _MAX_HISTORY * 2:
        _history.pop(0)
        _history.pop(0)


# ── DB knowledge base loader ─────────────────────────────────────────────────
async def load_knowledge_from_db() -> None:
    """Load best past feedback from DB into history buffer as few-shot examples."""
    global _kb_loaded
    if _kb_loaded:
        return
    try:
        from ..db.database import AsyncSessionLocal
        from ..db.models import InterviewAnswer, Question

        async with AsyncSessionLocal() as db:
            # Only load LLM-quality feedback (longer than 80 chars = not rule-based template)
            min_feedback_len = 80

            # Get top-rated answers (good examples)
            good_result = await db.execute(
                select(InterviewAnswer, Question)
                .join(Question, InterviewAnswer.question_id == Question.id)
                .where(
                    InterviewAnswer.feedback != "",
                    InterviewAnswer.feedback.isnot(None),
                    InterviewAnswer.final_score >= 0.7,
                    func.length(InterviewAnswer.feedback) >= min_feedback_len,
                )
                .order_by(desc(InterviewAnswer.final_score))
                .limit(3)
            )
            good_rows = good_result.all()

            # Get low-rated answers (weak examples)
            weak_result = await db.execute(
                select(InterviewAnswer, Question)
                .join(Question, InterviewAnswer.question_id == Question.id)
                .where(
                    InterviewAnswer.feedback != "",
                    InterviewAnswer.feedback.isnot(None),
                    InterviewAnswer.final_score < 0.4,
                    func.length(InterviewAnswer.feedback) >= min_feedback_len,
                )
                .order_by(asc(InterviewAnswer.final_score))
                .limit(2)
            )
            weak_rows = weak_result.all()

            rows = weak_rows + good_rows
            if not rows:
                logger.info("No past feedback in DB — using hardcoded few-shot only")
                _kb_loaded = True
                return

            for ans, q in rows:
                quality = _quality_label(float(ans.final_score or 0))
                # Derive TTS from first sentence of feedback
                tts_line = (ans.feedback or "").split(".")[0].strip() + "."
                human_msg = (
                    f"QUESTION:\n{q.question_text}\n\n"
                    f"CANDIDATE'S ANSWER:\n{ans.transcript}\n\n"
                    f"EVALUATION SUMMARY:\n"
                    f"- Overall quality: {quality} (score {ans.final_score:.2f}/1.00)"
                )
                ai_msg = (
                    f"FEEDBACK: {ans.feedback}\n"
                    f"TIP: {ans.tip or 'No specific tip provided.'}\n"
                    f"TTS: {tts_line}"
                )
                _history.append(HumanMessage(content=human_msg))
                _history.append(AIMessage(content=ai_msg))

            # Trim if too many
            while len(_history) > _MAX_HISTORY * 2:
                _history.pop(0)
                _history.pop(0)

            _kb_loaded = True
            logger.info(f"Loaded {len(rows)} past feedback examples from DB into LLM history")
    except Exception as exc:
        logger.warning(f"Could not load knowledge from DB: {exc}")
        _kb_loaded = True  # Don't retry on every call


# ── score → quality label ─────────────────────────────────────────────────────
def _quality_label(score: float) -> str:
    if score >= 0.85: return "strong"
    if score >= 0.70: return "good"
    if score >= 0.50: return "partial"
    return "weak"


# ── build human message ───────────────────────────────────────────────────────
def _build_input(
    question: str,
    candidate_answer: str,
    expected_answer: str,
    metrics: dict,
    missing_keywords: list,
) -> str:
    score     = metrics.get("final_score", 0.0)
    semantic  = metrics.get("semantic_score", 0.0)
    relevance = metrics.get("question_relevance", 0.0)
    kw_score  = metrics.get("keyword_score", 0.0)
    length_sc = metrics.get("length_score", 0.0)
    discourse = metrics.get("discourse_score", 0.0)
    lex_div   = metrics.get("lexical_diversity", 0.0)
    quality   = _quality_label(score)
    missing_str = ", ".join(missing_keywords[:5]) if missing_keywords else "none"

    diagnostics = []
    if relevance < 0.40:
        diagnostics.append("the answer did not address the specific question asked")
    if semantic < 0.50:
        diagnostics.append("the meaning is far from the expected concept")
    elif semantic < 0.70:
        diagnostics.append("the meaning is partially correct but incomplete")
    if kw_score < 0.40 and missing_keywords:
        diagnostics.append(f"key concepts missing: {missing_str}")
    if length_sc < 0.45:
        diagnostics.append("the answer is too short")
    if discourse < 0.33:
        diagnostics.append("no structured reasoning or examples provided")
    if lex_div < 0.45:
        diagnostics.append("the answer is repetitive")

    diagnostic_line = (
        "Key issues identified: " + "; ".join(diagnostics) + "."
        if diagnostics else
        "The answer is mostly correct but could be more complete."
    )

    return f"""QUESTION:
{question}

CANDIDATE'S ANSWER:
{candidate_answer}

REFERENCE ANSWER (for context only — do not quote or copy from it):
{expected_answer}

EVALUATION SUMMARY:
- Overall quality: {quality} (score {score:.2f}/1.00)
- Semantic closeness: {semantic:.2f}/1.00
- Question relevance: {relevance:.2f}/1.00
- Keyword coverage: {kw_score:.2f}/1.00
- {diagnostic_line}"""


# ── sync call (runs in thread pool) ──────────────────────────────────────────
def _call_chain_sync(human_input: str, cache_key: str) -> Optional[dict]:
    if not _chain:
        return None
    try:
        output: FeedbackOutput = _chain.invoke({
            "input": human_input,
            "history": list(_history),
        })
        result = {
            "feedback": output.feedback.strip().strip('"'),
            "tip": output.tip.strip().strip('"'),
            "tts_feedback": output.tts_feedback.strip().strip('"'),
        }
        _add_to_history(human_input, result)
        _cache_put(cache_key, result)
        logger.info(f"LLM feedback OK ({len(result['feedback'])} chars)")
        return result
    except Exception as exc:
        logger.warning(f"LangChain call failed: {exc}")
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f"Serving cached response for: {cache_key[:60]}")
            return cached
        return None


# ── async public API ──────────────────────────────────────────────────────────
async def generate_llm_feedback(
    question:         str,
    candidate_answer: str,
    expected_answer:  str,
    metrics:          dict,
    missing_keywords: list,
) -> Optional[dict]:
    """
    Returns {"feedback": str, "tip": str, "tts_feedback": str} or None on failure.
    """
    if not llm_available:
        return None

    # Lazy-load DB knowledge on first call
    if not _kb_loaded:
        await load_knowledge_from_db()

    key = _cache_key(question, metrics.get("final_score", 0.0))
    cached = _cache_get(key)
    if cached:
        logger.info(f"Cache hit for: {key[:60]}")
        return cached

    human_input = _build_input(
        question, candidate_answer, expected_answer, metrics, missing_keywords
    )

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _call_chain_sync, human_input, key)
    except Exception as exc:
        logger.warning(f"LLM feedback executor error: {exc}")
        return None
