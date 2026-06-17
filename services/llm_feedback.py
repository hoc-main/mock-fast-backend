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
    feedback: str = Field(description="3-5 sentences explaining what the candidate got right, what key points they missed, and what a strong answer should include. Be specific — name the exact concepts, terms, and reasoning gaps. 70-120 words.")
    tip: str = Field(description="A concrete actionable tip: tell them exactly what to say or include next time, with a mini example. 30-50 words.")
    tts_feedback: str = Field(description="A natural spoken sentence (20-35 words) that a human interviewer would say as verbal feedback. Acknowledge what they got right and briefly mention the main gap. Must sound conversational, not robotic.")


# ── few-shot examples (pre-loaded into memory) ───────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "human": """QUESTION: What is polymorphism in OOP?

CANDIDATE'S ANSWER: Polymorphism is when objects can take different forms.

REFERENCE ANSWER: Polymorphism allows objects of different classes to be treated through the same interface. There are two types: compile-time (method overloading — same method name, different parameters) and runtime (method overriding — subclass provides specific implementation of a parent method). Interfaces and abstract classes enable polymorphic behavior by defining contracts that multiple classes implement differently.""",
        "ai": FeedbackOutput(
            feedback="You got the core idea right — polymorphism is about objects taking different forms. But you missed the two types: compile-time polymorphism through method overloading, and runtime polymorphism through method overriding. You also didn't mention how interfaces enable it.",
            tip="Next time, structure it as: 'There are two types — overloading where same method has different parameters, and overriding where a subclass redefines a parent method. For example, Dog and Cat both override Animal.speak() differently.'",
            tts_feedback="You got the basic idea right, but you need to cover the two types — overloading and overriding — and mention how interfaces enable polymorphic behavior.",
        ),
    },
    {
        "human": """QUESTION: Explain the event loop in Node.js.

CANDIDATE'S ANSWER: The event loop is what makes Node.js non-blocking. It handles callbacks by checking the callback queue after the call stack is empty. This allows Node to handle many connections with a single thread using phases like timers, poll, and check.

REFERENCE ANSWER: The event loop is Node.js's mechanism for non-blocking I/O. It continuously checks the call stack and callback queue. When the stack is empty, it picks callbacks from the queue. It runs in phases: timers, pending callbacks, poll (I/O), check (setImmediate), and close callbacks. The microtask queue (Promises, process.nextTick) has priority over the regular callback queue.""",
        "ai": FeedbackOutput(
            feedback="You nailed the fundamentals — non-blocking I/O, single thread, callback queue, and call stack relationship. You even named specific phases which shows solid understanding. The one gap: you didn't mention the microtask queue and how Promises get priority over setTimeout callbacks.",
            tip="Add this point: 'The microtask queue handles Promises and process.nextTick with higher priority — so Promise.resolve() executes before setTimeout(fn, 0) even though both are async.'",
            tts_feedback="Strong answer — you covered non-blocking I/O and the phases well. The one thing missing is the microtask queue and how Promises get priority over regular callbacks.",
        ),
    },
]

# ── system prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a senior technical interviewer giving verbal feedback after each answer.

Your job:
1. Compare the candidate's answer against the reference answer.
2. Identify SPECIFIC concepts they covered correctly and SPECIFIC concepts they missed or got wrong.
3. Give a concrete tip with an example of what a better answer would include.

RULES:
- Be SPECIFIC — name exact concepts, terms, or ideas. Never say vague things like "your answer lacks depth" or "could be more complete" without saying WHAT is missing.
- Address the candidate as "you".
- Never mention scores, metrics, semantic similarity, or keywords.
- Never copy the reference answer verbatim — paraphrase the missing points.
- Sound like a real interviewer talking to a candidate, not a robot generating a report.
- The tts_feedback should sound like what an interviewer would naturally say out loud right after hearing the answer (e.g. "Good start, but you missed the key distinction between X and Y")."""


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
            SystemMessage(content=_SYSTEM_PROMPT + "\n\nRespond in EXACTLY this format:\nFEEDBACK: <your feedback>\nTIP: <your tip>\nTTS: <one natural spoken sentence, 20-35 words, acknowledging what they got right and the main gap>"),
            *few_shot_messages,
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

        _chain = prompt | llm
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
    return f"""QUESTION:
{question}

CANDIDATE'S ANSWER:
{candidate_answer}

REFERENCE ANSWER:
{expected_answer}"""


# ── sync call (runs in thread pool) ──────────────────────────────────────────
def _parse_llm_response(raw: str) -> Optional[dict]:
    """Parse FEEDBACK/TIP/TTS from plain text LLM response."""
    feedback = ""
    tip = ""
    tts = ""
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FEEDBACK:"):
            feedback = stripped.split(":", 1)[1].strip()
        elif upper.startswith("TIP:"):
            tip = stripped.split(":", 1)[1].strip()
        elif upper.startswith("TTS:"):
            tts = stripped.split(":", 1)[1].strip()
    if feedback:
        return {
            "feedback": feedback.strip('"'),
            "tip": tip.strip('"'),
            "tts_feedback": tts.strip('"'),
        }
    return None


def _call_chain_sync(human_input: str, cache_key: str) -> Optional[dict]:
    if not _chain:
        return None
    try:
        output = _chain.invoke({
            "input": human_input,
            "history": list(_history),
        })
        raw = output.content if hasattr(output, 'content') else str(output)
        result = _parse_llm_response(raw)
        if not result:
            logger.warning(f"LLM response parse failed: {raw[:200]}")
            return None
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


# ── LLM Transcript Cleanup ───────────────────────────────────────────────────

def _clean_transcript_sync(raw_transcript: str, question: str) -> Optional[str]:
    """Clean up garbled STT transcript using LLM."""
    if not GROQ_API_KEY or len(raw_transcript.split()) < 5:
        return None
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=os.getenv("GROQ_MODEL_FAST", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=300,
        )
        messages = [
            SystemMessage(content="""You are a transcript cleaner. Fix obvious speech-to-text errors, remove filler words (um, uh, like, basically), fix broken grammar, and make it readable. STRICT RULES: Do NOT add any new words, concepts, or information. Do NOT rephrase or improve the answer. Only fix clear STT mistakes (e.g. "there" -> "their" if obvious, repeated words, broken sentences). If the transcript is already clean, return it unchanged. Return ONLY the cleaned transcript."""),
            HumanMessage(content=raw_transcript),
        ]
        output = llm.invoke(messages)
        cleaned = (output.content or "").strip()
        if cleaned and len(cleaned) > 10:
            return cleaned
        return None
    except Exception as exc:
        logger.warning(f"Transcript cleanup failed: {exc}")
        return None


async def clean_transcript(raw_transcript: str, question: str) -> str:
    """Clean a garbled STT transcript. Returns original if cleanup fails."""
    if not llm_available or not raw_transcript.strip():
        return raw_transcript
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _clean_transcript_sync, raw_transcript, question),
            timeout=5.0,
        )
        if result:
            logger.info(f"Transcript cleaned: '{raw_transcript[:50]}...' -> '{result[:50]}...'")
            return result
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(f"Transcript cleanup timeout/error: {exc}")
    return raw_transcript


# ── LLM Session Summary Rewriter ─────────────────────────────────────────────

_SESSION_SUMMARY_SYSTEM = """You are a senior interview coach writing a performance summary for a candidate after a mock interview session.

You will receive raw metrics and template-generated strengths/improvements. Your job is to rewrite them into polished, human-sounding language.

RULES:
- Write a 3-4 sentence summary paragraph that feels like a real coach talking to a candidate.
- Rewrite strengths as concise, encouraging bullet points (keep the meaning, improve the phrasing).
- Rewrite improvement areas as actionable, specific advice (not generic platitudes).
- Reference actual numbers (score %, question count) naturally.
- Don't invent new information — only rephrase what's provided.
- Address the candidate as "you".
- Be direct, professional, and motivating.

Respond in EXACTLY this format:
SUMMARY: <your 3-4 sentence paragraph>
STRENGTHS: <bullet1> | <bullet2> | <bullet3>
IMPROVEMENTS: <bullet1> | <bullet2> | <bullet3>"""


def _call_summary_rewrite_sync(human_input: str) -> Optional[dict]:
    """Sync LLM call for session summary rewriting."""
    if not GROQ_API_KEY:
        return None
    try:
        is_reasoning = "oss" in GROQ_MODEL or "reasoning" in GROQ_MODEL
        kwargs = dict(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.4,
            max_tokens=2000 if is_reasoning else 400,
        )
        if is_reasoning:
            kwargs["reasoning_effort"] = "medium"

        llm = ChatGroq(**kwargs)
        messages = [
            SystemMessage(content=_SESSION_SUMMARY_SYSTEM),
            HumanMessage(content=human_input),
        ]
        output = llm.invoke(messages)
        raw = output.content if hasattr(output, 'content') else str(output)

        # For reasoning models, content may be empty — check additional_kwargs
        if not raw and hasattr(output, 'additional_kwargs'):
            raw = output.additional_kwargs.get('reasoning_content', '')

        # Parse response
        summary = ""
        strengths = []
        improvements = []
        for line in raw.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("SUMMARY:"):
                summary = stripped.split(":", 1)[1].strip()
            elif upper.startswith("STRENGTHS:"):
                strengths = [s.strip() for s in stripped.split(":", 1)[1].split("|") if s.strip()]
            elif upper.startswith("IMPROVEMENTS:"):
                improvements = [s.strip() for s in stripped.split(":", 1)[1].split("|") if s.strip()]

        if summary:
            return {
                "summary_paragraph": summary,
                "strengths": strengths or None,
                "improvement_areas": improvements or None,
            }
        return None
    except Exception as exc:
        logger.warning(f"LLM session summary rewrite failed: {exc}")
        return None


async def rewrite_session_summary(template_summary: dict) -> dict:
    """
    Takes the template-generated session summary and rewrites the narrative parts using LLM.
    Returns the same dict with improved summary_paragraph, strengths, and improvement_areas.
    Falls back to template if LLM fails.
    """
    if not llm_available:
        return template_summary

    human_input = (
        f"SESSION METRICS:\n"
        f"- Questions answered: {len(template_summary.get('score_trend', []))}\n"
        f"- Overall score: {template_summary.get('session_score', 0):.0%}\n"
        f"- Score tier: {template_summary.get('score_tier', 'unknown')}\n"
        f"- Trend: {template_summary.get('trend_direction', 'consistent')}\n"
        f"- Metric averages: {template_summary.get('metric_averages', {})}\n\n"
        f"TEMPLATE STRENGTHS:\n"
        f"{chr(10).join('- ' + s for s in template_summary.get('strengths', []))}\n\n"
        f"TEMPLATE IMPROVEMENTS:\n"
        f"{chr(10).join('- ' + s for s in template_summary.get('improvement_areas', []))}\n\n"
        f"TEMPLATE SUMMARY:\n"
        f"{template_summary.get('summary_paragraph', '')}\n"
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _call_summary_rewrite_sync, human_input)
        if result:
            template_summary["summary_paragraph"] = result["summary_paragraph"]
            if result.get("strengths"):
                template_summary["strengths"] = result["strengths"]
            if result.get("improvement_areas"):
                template_summary["improvement_areas"] = result["improvement_areas"]
            logger.info("Session summary rewritten by LLM")
    except Exception as exc:
        logger.warning(f"LLM summary rewrite executor error: {exc}")

    return template_summary
