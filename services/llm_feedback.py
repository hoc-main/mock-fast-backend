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
    feedback: str = Field(description="8-12 sentences giving a thorough conversational evaluation addressing EVERY part of the question. For each part: state whether the candidate covered it or missed it, explain what was expected, and guide them on what a strong answer looks like. Be specific with concepts, terms, and reasoning. Include what they said that was correct, what was wrong or missing, and exactly what they should have said instead. Speak naturally like a real interviewer talking face-to-face. 250-300 words. Use plain ASCII text only, no special unicode dashes.")
    tip: str = Field(description="A concrete actionable tip: tell them exactly what to say or include next time, with a mini example structure. 40-60 words. Use plain ASCII text only.")


# ── few-shot examples (pre-loaded into memory) ───────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "human": """QUESTION: What is polymorphism in OOP?

CANDIDATE'S ANSWER: Polymorphism is when objects can take different forms.

REFERENCE ANSWER: Polymorphism allows objects of different classes to be treated through the same interface. There are two types: compile-time (method overloading — same method name, different parameters) and runtime (method overriding — subclass provides specific implementation of a parent method). Interfaces and abstract classes enable polymorphic behavior by defining contracts that multiple classes implement differently.""",
        "ai": FeedbackOutput(
            feedback="Your answer is too vague to demonstrate real understanding. Saying 'objects take different forms' is just restating the dictionary definition without explaining HOW polymorphism actually works. The question expected you to distinguish between compile-time polymorphism (method overloading — same method name with different parameters) and runtime polymorphism (method overriding — subclass redefines parent method). You also completely missed how interfaces and abstract classes enable polymorphic behavior by defining contracts. This level of surface-only answer would not pass in a real interview.",
            tip="Structure it as: 'Polymorphism has two types — compile-time through overloading (same method, different params) and runtime through overriding (subclass redefines parent behavior). Interfaces enable this by defining contracts multiple classes implement differently. Example: Dog and Cat both override Animal.speak().'",
        ),
    },
    {
        "human": """QUESTION: Explain the event loop in Node.js.

CANDIDATE'S ANSWER: The event loop is what makes Node.js non-blocking. It handles callbacks by checking the callback queue after the call stack is empty. This allows Node to handle many connections with a single thread using phases like timers, poll, and check.

REFERENCE ANSWER: The event loop is Node.js's mechanism for non-blocking I/O. It continuously checks the call stack and callback queue. When the stack is empty, it picks callbacks from the queue. It runs in phases: timers, pending callbacks, poll (I/O), check (setImmediate), and close callbacks. The microtask queue (Promises, process.nextTick) has priority over the regular callback queue.""",
        "ai": FeedbackOutput(
            feedback="You covered the basics well — non-blocking nature, single thread, call stack and callback queue relationship, and even named some phases. That shows you understand the core mechanism. However, you missed a critical piece: the microtask queue. In Node.js, Promises and process.nextTick get priority over the regular callback queue, meaning Promise.resolve() executes before setTimeout(fn, 0). This is a common interview follow-up and distinguishes someone who has worked with async code from someone who just read about it.",
            tip="Add: 'The microtask queue handles Promises and process.nextTick with higher priority than the callback queue — so Promise.resolve() runs before setTimeout(fn, 0) even though both are async. This is why mixing Promises with timers can produce unexpected ordering.'",
        ),
    },
]

# ── system prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a tough but fair senior technical interviewer giving verbal feedback face-to-face after each answer. You speak naturally like a real person in a conversation - not like a written report.

Your approach:
1. Be TRANSPARENT about gaps - don't sugarcoat. If they missed the point, say so clearly.
2. GUIDE them toward what the question wanted - explain what a complete answer should have covered.
3. Name SPECIFIC concepts, terms, and reasoning steps they missed.
4. If their answer was vague or surface-level, call it out and explain what depth was expected.
5. Only acknowledge what they got right if they actually demonstrated understanding, not just surface keywords.

TONE & STYLE:
- Sound like you're TALKING to someone, not writing an essay. Use conversational phrasing.
- Use natural speech patterns: "Look, you got the basic idea, but here's the problem..." or "Okay so you mentioned X, which is correct, but the question was really asking you to..."
- Don't use numbered lists like "First... Second... Third..." - weave your points naturally.
- Don't start with "Your answer is..." - be more human. Start with what you observed.
- Be the kind of interviewer who makes candidates think "that was tough but I learned exactly what I need to improve"

RULES:
- Be direct and honest. Don't blindly praise or motivate. If the answer was weak, say it's weak and explain why.
- If they only answered part of a multi-part question, explicitly point out which parts they skipped.
- Guide them: "The question was asking you to explain X AND Y. You only touched on X at a surface level."
- Name exact missing concepts, don't say vague things like "you could go deeper".
- Address the candidate as "you".
- Never mention scores, metrics, or keyword matching.
- Never copy the reference answer verbatim - paraphrase the key points they should have made.
- Use plain ASCII text only, no special unicode characters or dashes."""


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
            max_tokens=2000 if is_reasoning else 450,
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
                f"TIP: {ai_obj.tip}"
            )))

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_SYSTEM_PROMPT + "\n\nRespond in EXACTLY this format:\nFEEDBACK: <8-12 sentences, 250-300 words, conversational tone, covering EVERY part of the question thoroughly, plain ASCII only>\nTIP: <your actionable tip, 40-60 words, plain ASCII>"),
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
        f"TIP: {result['tip']}"
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
                human_msg = (
                    f"QUESTION:\n{q.question_text}\n\n"
                    f"CANDIDATE'S ANSWER:\n{ans.transcript}\n\n"
                    f"EVALUATION SUMMARY:\n"
                    f"- Overall quality: {quality} (score {ans.final_score:.2f}/1.00)"
                )
                ai_msg = (
                    f"FEEDBACK: {ans.feedback}\n"
                    f"TIP: {ans.tip or 'No specific tip provided.'}"
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
    """Parse FEEDBACK/TIP from plain text LLM response."""
    feedback = ""
    tip = ""
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FEEDBACK:"):
            feedback = stripped.split(":", 1)[1].strip()
        elif upper.startswith("TIP:"):
            tip = stripped.split(":", 1)[1].strip()
    if feedback:
        return {
            "feedback": feedback.strip('"'),
            "tip": tip.strip('"'),
            "tts_feedback": "",
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

        # For reasoning models, content may be empty — check additional_kwargs
        if not raw and hasattr(output, 'additional_kwargs'):
            raw = output.additional_kwargs.get('reasoning_content', '')

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
            SystemMessage(content="""You are fixing speech-to-text errors in a recorded answer. The topic context is provided ONLY so you can identify misheard technical terms.

Fix:
- Misheard technical terms (use topic context to infer correct terms)
- Filler words (um, uh, like, basically, so, you know)
- Repeated/stuttered words
- Broken grammar from STT

DO NOT:
- Add any new information, explanations, or points
- Expand or improve the answer in any way
- Generate content from the topic — only fix what's already there
- Change the meaning or add words the speaker didn't say

If a word is unrecognizable and you can't confidently fix it, leave it. Return ONLY the cleaned transcript."""),
            HumanMessage(content=f"TOPIC (for reference only): {question}\n\nTRANSCRIPT: {raw_transcript}"),
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
