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
GROQ_API_KEYS: list = []  # Multiple keys for rotation
_current_key_index: int = 0
GROQ_MODEL:   str = "openai/gpt-oss-20b"
GROQ_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
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
    feedback: str = Field(description="IMPORTANT: Write 250-300 words. Give 12-15 sentences of detailed conversational evaluation covering: (1) what they said correctly and why it matters, (2) each specific concept they missed with explanation of why it matters, (3) what the complete answer should include with specific examples, (4) why the missing pieces matter in interviews, (5) how their answer compares to what an interviewer expects. Be thorough, specific, and conversational. Plain ASCII only.")
    tip: str = Field(description="A concrete actionable tip showing exactly how to structure a better answer next time, with a specific example sentence they could use. 60-100 words. Plain ASCII only.")


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

CRITICAL - TRANSCRIPT AWARENESS:
The candidate's answer was captured via speech-to-text (STT). The transcript may be INCOMPLETE or contain errors:
- Words may be misheard, garbled, or missing
- Technical terms are often transcribed incorrectly
- The candidate may have said MORE than what appears in the transcript
- If the transcript seems unusually short, choppy, or incoherent, the candidate likely said more that wasn't captured
- NEVER penalize the candidate for what might be an STT capture issue
- If you notice the transcript seems incomplete (very short, ends abruptly, missing logical flow), explicitly acknowledge this: "Based on what was captured, it seems like you may have said more that wasn't picked up..."
- Focus your evaluation on the SUBSTANCE of what IS present, not on gaps that could be STT failures
- When in doubt about whether something was missed by the candidate or by the microphone, give the candidate the benefit of the doubt

Your approach:
1. Evaluate the CONTENT that was captured fairly - acknowledge what they got right.
2. Be TRANSPARENT about gaps - but distinguish between "you didn't mention X" vs "the transcript doesn't show X (you may have covered it)".
3. GUIDE them toward what the question wanted - explain what a complete answer should cover.
4. Name SPECIFIC concepts, terms, and reasoning steps that appear to be missing from the captured response.
5. If their captured answer was vague or surface-level, suggest how to make it more precise.

TONE & STYLE:
- Sound like you're TALKING to someone, not writing an essay. Use conversational phrasing.
- The candidate SPOKE their answer verbally. Always refer to what they "said" or "mentioned" - never "wrote" or "written".
- Use natural speech patterns: "Look, you got the basic idea, but here's the problem..." or "Okay so you mentioned X, which is correct, but the question was really asking you to..."
- Don't use numbered lists like "First... Second... Third..." - weave your points naturally.
- Don't start with "Your answer is..." - be more human. Start with what you observed.
- Be the kind of interviewer who makes candidates think "that was tough but I learned exactly what I need to improve"

RULES:
- Be direct and honest, but FAIR. Acknowledge what they got right before addressing gaps.
- If they only answered part of a multi-part question, point out which parts seem missing from the transcript.
- Guide them: "The question was asking you to explain X AND Y. Based on what was captured, you touched on X but Y seems to be missing."
- Name exact missing concepts, don't say vague things like "you could go deeper".
- Address the candidate as "you".
- Never mention scores, metrics, keyword matching, or STT/transcription technology directly.
- Never copy the reference answer verbatim - paraphrase the key points they should have made.
- Never use words like "wrote", "written", "text" - always use "said", "mentioned", "spoke about", "told me".
- Use plain ASCII text only, no special unicode characters or dashes."""


def _get_current_key() -> str:
    """Get current API key from rotation pool."""
    if not GROQ_API_KEYS:
        return ""
    return GROQ_API_KEYS[_current_key_index % len(GROQ_API_KEYS)]


def _rotate_key() -> str:
    """Rotate to next API key when rate limited. Returns new key."""
    global _current_key_index
    if len(GROQ_API_KEYS) <= 1:
        return _get_current_key()
    _current_key_index = (_current_key_index + 1) % len(GROQ_API_KEYS)
    logger.info(f"Rotated to API key #{_current_key_index + 1}/{len(GROQ_API_KEYS)}")
    return GROQ_API_KEYS[_current_key_index]


def check_llm_available() -> bool:
    global llm_available, _chain, GROQ_API_KEYS, GROQ_MODEL, GROQ_FALLBACK_MODEL, GROQ_TIMEOUT
    # Support multiple keys: GROQ_API_KEY (single) or GROQ_API_KEYS (comma-separated)
    keys_str = os.getenv("GROQ_API_KEYS", "")
    single_key = os.getenv("GROQ_API_KEY", "")
    if keys_str:
        GROQ_API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
    elif single_key:
        GROQ_API_KEYS = [single_key]
    else:
        GROQ_API_KEYS = []

    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "15"))

    if not GROQ_API_KEYS:
        llm_available = False
        logger.warning("GROQ_API_KEY(S) not set — LLM feedback disabled, rule-based fallback active")
        return False
    try:
        _build_chain(_get_current_key(), GROQ_MODEL)
        llm_available = True
        logger.info(f"LangChain+Groq initialized — LLM feedback enabled ({GROQ_MODEL}, {len(GROQ_API_KEYS)} keys)")
        return True
    except Exception as exc:
        llm_available = False
        logger.warning(f"LangChain init failed — LLM feedback disabled. ({exc})")
        return False


def _build_chain(api_key: str, model: str):
    """Build/rebuild the LangChain chain with given key and model."""
    global _chain
    is_reasoning = "oss" in model or "reasoning" in model
    kwargs = dict(
        api_key=api_key,
        model=model,
        temperature=0.35,
        max_tokens=1500 if is_reasoning else 700,
    )
    if is_reasoning:
        kwargs["reasoning_effort"] = "medium"

    llm = ChatGroq(**kwargs)

    few_shot_messages = []
    for ex in _FEW_SHOT_EXAMPLES:
        few_shot_messages.append(HumanMessage(content=ex["human"]))
        ai_obj = ex["ai"]
        few_shot_messages.append(AIMessage(content=(
            f"FEEDBACK: {ai_obj.feedback}\n"
            f"TIP: {ai_obj.tip}"
        )))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_SYSTEM_PROMPT + "\n\nRespond in EXACTLY this format:\nFEEDBACK: <12-15 sentences, 250-300 words, detailed and conversational, plain ASCII only>\nTIP: <actionable tip with example sentence, 60-100 words, plain ASCII>"),
        *few_shot_messages,
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{input}"),
    ])

    _chain = prompt | llm


# ── conversation history buffers (separated: DB knowledge vs live session) ───
_MAX_KB_EXAMPLES = 8       # permanent DB-sourced few-shot examples
_MAX_SESSION_HISTORY = 4   # live conversation exchanges this session
_kb_examples: list = []    # permanent, only replaced on module change
_session_history: list = [] # rolling window of current session
_kb_loaded_module: Optional[int] = None  # track which module's KB is loaded


def _add_to_history(human_msg: str, result: dict) -> None:
    _session_history.append(HumanMessage(content=human_msg))
    _session_history.append(AIMessage(content=(
        f"FEEDBACK: {result['feedback']}\n"
        f"TIP: {result['tip']}"
    )))
    while len(_session_history) > _MAX_SESSION_HISTORY * 2:
        _session_history.pop(0)
        _session_history.pop(0)


def _get_full_history() -> list:
    """Combine permanent KB examples + live session history for LLM context."""
    return _kb_examples + _session_history


# ── DB knowledge base loader (topic-aware) ───────────────────────────────────
async def load_knowledge_from_db(module_id: Optional[int] = None) -> None:
    """Load best past feedback from DB into KB buffer as few-shot examples.
    
    Prioritizes examples from the same module (topic-aware), then fills
    remaining slots with diverse examples from other modules.
    """
    global _kb_loaded_module, _kb_examples
    
    # Skip if already loaded for this module
    if _kb_loaded_module == (module_id or -1):
        return
    
    try:
        from ..db.database import AsyncSessionLocal
        from ..db.models import InterviewAnswer, Question, InterviewSession

        async with AsyncSessionLocal() as db:
            min_feedback_len = 80
            _kb_examples.clear()
            
            # --- Phase 1: Topic-matched examples (same module) ---
            topic_rows = []
            if module_id:
                topic_good = await db.execute(
                    select(InterviewAnswer, Question)
                    .join(Question, InterviewAnswer.question_id == Question.id)
                    .join(InterviewSession, InterviewAnswer.session_id == InterviewSession.id)
                    .where(
                        InterviewSession.module_id == module_id,
                        InterviewAnswer.feedback != "",
                        InterviewAnswer.feedback.isnot(None),
                        InterviewAnswer.final_score >= 0.7,
                        func.length(InterviewAnswer.feedback) >= min_feedback_len,
                    )
                    .order_by(desc(InterviewAnswer.final_score))
                    .limit(3)
                )
                topic_weak = await db.execute(
                    select(InterviewAnswer, Question)
                    .join(Question, InterviewAnswer.question_id == Question.id)
                    .join(InterviewSession, InterviewAnswer.session_id == InterviewSession.id)
                    .where(
                        InterviewSession.module_id == module_id,
                        InterviewAnswer.feedback != "",
                        InterviewAnswer.feedback.isnot(None),
                        InterviewAnswer.final_score < 0.4,
                        func.length(InterviewAnswer.feedback) >= min_feedback_len,
                    )
                    .order_by(asc(InterviewAnswer.final_score))
                    .limit(2)
                )
                topic_rows = topic_weak.all() + topic_good.all()

            # --- Phase 2: Fill remaining slots with global diverse examples ---
            topic_ids = {ans.id for ans, _ in topic_rows}
            remaining = _MAX_KB_EXAMPLES - len(topic_rows)
            
            global_rows = []
            if remaining > 0:
                good_query = (
                    select(InterviewAnswer, Question)
                    .join(Question, InterviewAnswer.question_id == Question.id)
                    .where(
                        InterviewAnswer.feedback != "",
                        InterviewAnswer.feedback.isnot(None),
                        InterviewAnswer.final_score >= 0.7,
                        func.length(InterviewAnswer.feedback) >= min_feedback_len,
                    )
                    .order_by(desc(InterviewAnswer.final_score))
                    .limit(remaining // 2 + 1)
                )
                if topic_ids:
                    good_query = good_query.where(InterviewAnswer.id.notin_(topic_ids))
                global_good = await db.execute(good_query)

                weak_query = (
                    select(InterviewAnswer, Question)
                    .join(Question, InterviewAnswer.question_id == Question.id)
                    .where(
                        InterviewAnswer.feedback != "",
                        InterviewAnswer.feedback.isnot(None),
                        InterviewAnswer.final_score < 0.4,
                        func.length(InterviewAnswer.feedback) >= min_feedback_len,
                    )
                    .order_by(asc(InterviewAnswer.final_score))
                    .limit(remaining // 2)
                )
                if topic_ids:
                    weak_query = weak_query.where(InterviewAnswer.id.notin_(topic_ids))
                global_weak = await db.execute(weak_query)

                global_rows = global_weak.all() + global_good.all()

            # --- Build KB examples (topic first, then global) ---
            all_rows = topic_rows + global_rows
            if not all_rows:
                logger.info("No past feedback in DB — using hardcoded few-shot only")
                _kb_loaded_module = module_id or -1
                return

            for ans, q in all_rows[:_MAX_KB_EXAMPLES]:
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
                _kb_examples.append(HumanMessage(content=human_msg))
                _kb_examples.append(AIMessage(content=ai_msg))

            _kb_loaded_module = module_id or -1
            logger.info(
                f"KB loaded: {len(topic_rows)} topic-matched + "
                f"{len(all_rows) - len(topic_rows)} global = "
                f"{len(all_rows)} examples (module_id={module_id})"
            )
    except Exception as exc:
        logger.warning(f"Could not load knowledge from DB: {exc}")
        _kb_loaded_module = module_id or -1  # Don't retry on every call


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
def _strip_markdown(text: str) -> str:
    """Remove markdown formatting artifacts from LLM output for clean plain-text display."""
    import re as _re
    # Remove bold/italic markers
    text = _re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)
    text = _re.sub(r'_{1,3}([^_]+?)_{1,3}', r'\1', text)
    # Remove bullet markers at start of lines
    text = _re.sub(r'^\s*[-*+]\s+', '', text, flags=_re.MULTILINE)
    # Remove numbered list markers
    text = _re.sub(r'^\s*\d+\.\s+', '', text, flags=_re.MULTILINE)
    # Remove headers
    text = _re.sub(r'^#{1,4}\s+', '', text, flags=_re.MULTILINE)
    # Remove backtick code formatting
    text = _re.sub(r'`([^`]+?)`', r'\1', text)
    # Remove code blocks
    text = _re.sub(r'```[\s\S]*?```', '', text)
    # Collapse multiple newlines into single space
    text = _re.sub(r'\n+', ' ', text)
    # Collapse multiple spaces
    text = _re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _parse_llm_response(raw: str) -> Optional[dict]:
    """Parse FEEDBACK/TIP from plain text LLM response. Handles multi-line feedback."""
    lines = raw.splitlines()
    feedback_lines = []
    tip_lines = []
    current = None

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FEEDBACK:"):
            current = "feedback"
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                feedback_lines.append(rest)
        elif upper.startswith("TIP:"):
            current = "tip"
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                tip_lines.append(rest)
        elif current == "feedback" and stripped:
            feedback_lines.append(stripped)
        elif current == "tip" and stripped:
            tip_lines.append(stripped)

    feedback = " ".join(feedback_lines).strip().strip('"')
    tip = " ".join(tip_lines).strip().strip('"')

    # Strip markdown artifacts and enforce length limits
    feedback = _strip_markdown(feedback)
    tip = _strip_markdown(tip)

    # Cap tip at ~150 words to prevent excessively long tips
    tip_words = tip.split()
    if len(tip_words) > 150:
        tip = " ".join(tip_words[:150]).rstrip(".,;:") + "."

    if feedback:
        return {
            "feedback": feedback,
            "tip": tip,
            "tts_feedback": "",
        }
    return None


def _call_chain_sync(human_input: str, cache_key: str) -> Optional[dict]:
    if not _chain:
        return None
    try:
        output = _chain.invoke({
            "input": human_input,
            "history": _get_full_history(),
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
        exc_str = str(exc).lower()
        # Rate limit hit — rotate key and retry once
        if "rate_limit" in exc_str or "429" in exc_str or "too many" in exc_str:
            logger.warning(f"Rate limited on key #{_current_key_index + 1}, rotating...")
            new_key = _rotate_key()
            if new_key:
                try:
                    _build_chain(new_key, GROQ_MODEL)
                    output = _chain.invoke({"input": human_input, "history": _get_full_history()})
                    raw = output.content if hasattr(output, 'content') else str(output)
                    if not raw and hasattr(output, 'additional_kwargs'):
                        raw = output.additional_kwargs.get('reasoning_content', '')
                    result = _parse_llm_response(raw)
                    if result:
                        _add_to_history(human_input, result)
                        _cache_put(cache_key, result)
                        logger.info(f"LLM feedback OK after key rotation ({len(result['feedback'])} chars)")
                        return result
                except Exception as retry_exc:
                    logger.warning(f"Retry after rotation also failed: {retry_exc}")
            # Try fallback model as last resort
            if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
                try:
                    logger.info(f"Trying fallback model: {GROQ_FALLBACK_MODEL}")
                    _build_chain(_get_current_key(), GROQ_FALLBACK_MODEL)
                    output = _chain.invoke({"input": human_input, "history": _get_full_history()})
                    raw = output.content if hasattr(output, 'content') else str(output)
                    if not raw and hasattr(output, 'additional_kwargs'):
                        raw = output.additional_kwargs.get('reasoning_content', '')
                    result = _parse_llm_response(raw)
                    if result:
                        _add_to_history(human_input, result)
                        _cache_put(cache_key, result)
                        logger.info(f"LLM feedback OK via fallback model ({len(result['feedback'])} chars)")
                        return result
                except Exception as fb_exc:
                    logger.warning(f"Fallback model also failed: {fb_exc}")
                finally:
                    # Restore primary model for next call
                    try:
                        _build_chain(_get_current_key(), GROQ_MODEL)
                    except Exception:
                        pass
        else:
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
    module_id:        Optional[int] = None,
) -> Optional[dict]:
    """
    Returns {"feedback": str, "tip": str, "tts_feedback": str} or None on failure.
    Pass module_id to get topic-aware few-shot examples from the DB.
    """
    if not llm_available:
        return None

    # Lazy-load or refresh DB knowledge when module changes
    if _kb_loaded_module != (module_id or -1):
        await load_knowledge_from_db(module_id)

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
    """Clean up garbled STT transcript using LLM. Rotates key on 429."""
    if not GROQ_API_KEYS or len(raw_transcript.split()) < 5:
        return None
    messages = [
        SystemMessage(content="Fix speech-to-text errors: misheard terms, fillers, stutters, broken grammar. Do NOT add new info. Return ONLY cleaned transcript."),
        HumanMessage(content=f"TOPIC: {question}\nTRANSCRIPT: {raw_transcript}"),
    ]
    for attempt in range(2):
        try:
            llm = ChatGroq(api_key=_get_current_key(), model=GROQ_FALLBACK_MODEL, temperature=0.1, max_tokens=300)
            output = llm.invoke(messages)
            cleaned = (output.content or "").strip()
            if cleaned and len(cleaned) > 10:
                return cleaned
            return None
        except Exception as exc:
            if ("429" in str(exc) or "rate_limit" in str(exc).lower()) and attempt == 0:
                logger.warning("Transcript cleanup rate limited, rotating key...")
                _rotate_key()
                continue
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
    """Sync LLM call for session summary rewriting. Rotates key on 429."""
    if not GROQ_API_KEYS:
        return None
    for attempt in range(2):
        try:
            is_reasoning = "oss" in GROQ_MODEL or "reasoning" in GROQ_MODEL
            kwargs = dict(api_key=_get_current_key(), model=GROQ_MODEL, temperature=0.4, max_tokens=1000 if is_reasoning else 400)
            if is_reasoning:
                kwargs["reasoning_effort"] = "low"
            llm = ChatGroq(**kwargs)
            messages = [SystemMessage(content=_SESSION_SUMMARY_SYSTEM), HumanMessage(content=human_input)]
            output = llm.invoke(messages)
            raw = output.content if hasattr(output, 'content') else str(output)
            if not raw and hasattr(output, 'additional_kwargs'):
                raw = output.additional_kwargs.get('reasoning_content', '')
            summary, strengths, improvements = "", [], []
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
                return {"summary_paragraph": summary, "strengths": strengths or None, "improvement_areas": improvements or None}
            return None
        except Exception as exc:
            if ("429" in str(exc) or "rate_limit" in str(exc).lower()) and attempt == 0:
                logger.warning("Summary rewrite rate limited, rotating key...")
                _rotate_key()
                continue
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
