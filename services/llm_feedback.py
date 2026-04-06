"""
llm_feedback.py
================
LLM-powered feedback and tip generation via local Ollama (llama3.2:3b).

This is an ENRICHMENT layer — it runs after the NLP scorer and the
rule-based feedback have already been sent to the client.
The rule-based result is always available instantly.
This improves the feedback quality with a personalised, context-aware message
that arrives 3-5 seconds later as a second WebSocket message.

No new pip dependencies — uses only stdlib urllib.request for the HTTP call.

Environment variables (add to .env):
    OLLAMA_URL      http://localhost:11434/api/generate
    OLLAMA_MODEL    llama3.2:3b
    OLLAMA_TIMEOUT  12
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# ── config from env ────────────────────────────────────────────────────────────
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "12"))

# ── availability flag — set at startup, read at call time ─────────────────────
# Set to False if the startup health-check in main.py found Ollama unreachable.
# Avoids a 12-second timeout on every request when the daemon is not running.
ollama_available: bool = True


def check_ollama_available() -> bool:
    """
    Ping the Ollama tags endpoint. Called once at startup.
    Sets the module-level flag so every subsequent call fast-fails if needed.
    """
    global ollama_available
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3):
            pass
        ollama_available = True
        logger.info(f"Ollama reachable — LLM feedback enabled ({OLLAMA_MODEL})")
        return True
    except Exception as exc:
        ollama_available = False
        logger.warning(f"Ollama not reachable — LLM feedback disabled. Rule-based fallback active. ({exc})")
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

    # Build a brief diagnostic summary so the model knows WHY the score is what it is
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


# ── synchronous Ollama HTTP call (runs in thread pool) ────────────────────────
def _call_ollama_sync(prompt: str) -> Optional[dict]:
    """
    Blocking HTTP POST to Ollama. Always returns {"feedback": str, "tip": str}
    or None if the call fails or times out.
    """

    print("sending request to ollama")

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    0.35,   # low — consistent, not creative
            "num_predict":    140,    # enough for feedback + tip, stops rambling
            "top_p":          0.90,
            "repeat_penalty": 1.15,
            "stop":           ["\n\n", "---", "==="],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("request sent to ollama")

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = resp.read()
        result = json.loads(body)
        raw    = result.get("response", "").strip()
        logger.debug(f"Ollama raw response: {raw[:200]}")
        return _parse_response(raw)
    except urllib.error.URLError as exc:
        logger.warning(f"Ollama connection error: {exc}")
        return None
    except TimeoutError:
        logger.warning(f"Ollama timed out after {OLLAMA_TIMEOUT}s")
        return None
    except Exception as exc:
        logger.warning(f"Ollama call failed: {exc}")
        return None


def _parse_response(raw: str) -> Optional[dict]:
    """
    Extract FEEDBACK and TIP from the model output.
    Handles minor variations in capitalisation and spacing.
    """
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

    # Fallback: if model ignored the format, try to split on double-newline
    if feedback and not tip:
        logger.debug("Ollama response missing TIP line — using feedback only")
        return {"feedback": feedback, "tip": ""}

    logger.debug(f"Ollama response could not be parsed: {raw[:200]}")
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
    Async wrapper that runs the blocking Ollama call in a thread pool executor.

    Returns {"feedback": str, "tip": str} or None if:
      - Ollama daemon is not running (ollama_available == False)
      - The call times out
      - The response cannot be parsed

    The caller (transcription.py) must handle None gracefully — rule-based
    feedback is already in the DB and sent to the client by this point.
    """
    if not ollama_available:
        return None

    prompt = _build_prompt(
        question, candidate_answer, expected_answer, metrics, missing_keywords
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _call_ollama_sync, prompt)
        return result
    except Exception as exc:
        logger.warning(f"LLM feedback executor error: {exc}")
        return None
