"""
llm_feedback.py
================
LLM-powered feedback and tip generation via Groq API.

This is an ENRICHMENT layer — it runs after the NLP scorer and the
rule-based feedback have already been produced.
The rule-based result is always available instantly as fallback.

Environment variables (add to .env):
    GROQ_API_KEY     your groq api key
    GROQ_MODEL       openai/gpt-oss-120b
    GROQ_TIMEOUT     15
"""

import asyncio
import logging
import os
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

# ── availability flag ─────────────────────────────────────────────────────────
GROQ_API_KEY: str = ""
GROQ_MODEL:   str = "openai/gpt-oss-120b"
GROQ_TIMEOUT: int = 15
llm_available: bool = False
_groq_client: Optional[Groq] = None


def check_llm_available() -> bool:
    """Read env vars at call time (after load_dotenv) and init client."""
    global llm_available, _groq_client, GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "15"))
    if not GROQ_API_KEY:
        llm_available = False
        logger.warning("GROQ_API_KEY not set — LLM feedback disabled, rule-based fallback active")
        return False
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        llm_available = True
        logger.info(f"Groq client initialized — LLM feedback enabled ({GROQ_MODEL})")
        return True
    except Exception as exc:
        llm_available = False
        logger.warning(f"Groq client init failed — LLM feedback disabled. ({exc})")
        return False


# ── score → quality label for the prompt ──────────────────────────────────────
def _quality_label(score: float) -> str:
    if score >= 0.85: return "strong"
    if score >= 0.70: return "good"
    if score >= 0.50: return "partial"
    return "weak"


# ── prompt builder ─────────────────────────────────────────────────────────────
def _build_prompt(
    question:         str,
    candidate_answer: str,
    expected_answer:  str,
    metrics:          dict,
    missing_keywords: list,
) -> str:
    score       = metrics.get("final_score",        0.0)
    semantic    = metrics.get("semantic_score",      0.0)
    relevance   = metrics.get("question_relevance",  0.0)
    kw_score    = metrics.get("keyword_score",       0.0)
    length_sc   = metrics.get("length_score",        0.0)
    discourse   = metrics.get("discourse_score",     0.0)
    lex_div     = metrics.get("lexical_diversity",   0.0)
    quality     = _quality_label(score)
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

    return f"""You are an expert interview coach. A candidate just answered a mock interview question.
Your job is to write personalised feedback and a concrete improvement tip.

QUESTION:
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
- {diagnostic_line}

INSTRUCTIONS:
Write exactly two lines in this format — nothing else before or after:

FEEDBACK: <one sentence — what the candidate did well or the primary gap in their answer>
TIP: <one or two sentences — the single most actionable thing they should do differently next time>

Rules you must follow:
- Address the candidate directly ("your answer", "you covered", "you missed").
- Be specific to this answer and question — not generic interview advice.
- Do not copy phrases from the reference answer.
- Do not mention scores, numbers, or metric names.
- FEEDBACK must be under 25 words.
- TIP must be under 40 words.
- Output only the two FEEDBACK and TIP lines, nothing else."""


# ── synchronous Groq call (runs in thread pool) ──────────────────────────────
def _call_groq_sync(prompt: str) -> Optional[dict]:
    if not _groq_client:
        return None
    try:
        is_reasoning = "oss" in GROQ_MODEL or "reasoning" in GROQ_MODEL
        call_kwargs = dict(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_completion_tokens=400 if is_reasoning else 200,
            top_p=0.9,
            stream=False,
        )
        if is_reasoning:
            call_kwargs["reasoning_effort"] = "medium"
        else:
            call_kwargs["stop"] = ["\n\n", "---", "==="]
        completion = _groq_client.chat.completions.create(**call_kwargs)
        msg = completion.choices[0].message
        # Try content first, then reasoning field for oss models
        raw = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning", None) or "").strip()
        # Parse from both — content takes priority
        result = _parse_response(raw) if raw else None
        if not result and reasoning:
            result = _parse_response(reasoning)
        logger.debug(f"Groq content: {raw[:100]}, reasoning: {reasoning[:100]}")
        return result
    except Exception as exc:
        logger.warning(f"Groq call failed: {exc}")
        return None


def _parse_response(raw: str) -> Optional[dict]:
    feedback: Optional[str] = None
    tip:      Optional[str] = None

    for line in raw.splitlines():
        stripped = line.strip()
        upper    = stripped.upper()
        if upper.startswith("FEEDBACK:"):
            feedback = stripped.split(":", 1)[1].strip()
        elif upper.startswith("TIP:"):
            tip = stripped.split(":", 1)[1].strip()

    if feedback and tip:
        return {"feedback": feedback, "tip": tip}
    if feedback and not tip:
        logger.debug("Groq response missing TIP line — using feedback only")
        return {"feedback": feedback, "tip": ""}
    logger.debug(f"Groq response could not be parsed: {raw[:200]}")
    return None


# ── async public API ───────────────────────────────────────────────────────────
async def generate_llm_feedback(
    question:         str,
    candidate_answer: str,
    expected_answer:  str,
    metrics:          dict,
    missing_keywords: list,
) -> Optional[dict]:
    """
    Returns {"feedback": str, "tip": str} or None on failure.
    The caller must handle None gracefully — rule-based feedback is the fallback.
    """
    if not llm_available:
        return None

    prompt = _build_prompt(
        question, candidate_answer, expected_answer, metrics, missing_keywords
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _call_groq_sync, prompt)
        return result
    except Exception as exc:
        logger.warning(f"LLM feedback executor error: {exc}")
        return None
