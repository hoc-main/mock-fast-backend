"""
services/intro_interview_agent.py
==================================
Generates dynamic cross-questions based on the user's self-introduction
and their subsequent answers. No pre-loaded questions — everything is
generated on-the-fly by the LLM.

Flow:
1. User introduces themselves (skills, projects, experience)
2. LLM generates a cross-question referencing their intro
3. After each answer, LLM generates the next cross-question based on conversation
"""

import asyncio
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_groq_client():
    from .llm_feedback import _get_current_key, GROQ_API_KEYS
    from groq import Groq
    api_key = _get_current_key() if GROQ_API_KEYS else os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _call_llm(prompt: str) -> Optional[str]:
    client = _get_groq_client()
    if not client:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    fallback = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
    use_model = fallback if "oss" in model else model

    try:
        completion = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=300,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(f"Intro interview agent LLM call failed: {exc}")
        from .llm_feedback import _rotate_key, GROQ_API_KEYS
        if len(GROQ_API_KEYS) > 1:
            _rotate_key()
            try:
                client = _get_groq_client()
                completion = client.chat.completions.create(
                    model=use_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_completion_tokens=300,
                    stream=False,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception:
                pass
        return None


def _parse_question(raw: str) -> Optional[str]:
    """Extract the question from LLM response."""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("QUESTION:"):
            return stripped.split(":", 1)[1].strip()
    # If no QUESTION: prefix, return the whole thing if it looks like a question
    clean = raw.strip().strip('"')
    if clean:
        return clean
    return None


async def generate_first_question(introduction: str, module_topic: str = "technical") -> Optional[str]:
    """
    Generate the first cross-question based on user's self-introduction.
    
    Example:
      Input: "I am a software engineer specialising in nodejs development"
      Output: "As you mentioned you work with Node.js, can you explain the event loop?"
    """
    from .llm_feedback import GROQ_API_KEYS
    if not (GROQ_API_KEYS or os.getenv("GROQ_API_KEY")):
        return None

    prompt = f"""You are a senior technical interviewer. The candidate just introduced themselves. Generate ONE targeted technical question that directly references something they mentioned.

CANDIDATE'S INTRODUCTION:
"{introduction}"

RULES:
- Start by referencing what they said (e.g. "As you mentioned you work with X..." or "Since you have experience with Y...")
- Ask a specific technical question that tests real understanding of what they claimed
- The question should be conversational, like a real interviewer naturally following up
- Ask about internals, tradeoffs, or real-world scenarios — not surface-level definitions
- ONE question only, keep it concise

Respond in this format:
QUESTION: <your cross-question>"""

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call_llm, prompt)
    if raw:
        return _parse_question(raw)
    return None


async def generate_followup_question(
    introduction: str,
    conversation: List[dict],
    module_topic: str = "technical",
) -> Optional[str]:
    """
    Generate the next cross-question based on the user's last answer.
    
    Example:
      Last Q: "Can you explain the event loop?"
      Last A: "It handles callbacks using a queue..."
      Output: "You mentioned the callback queue — can you explain the difference between microtasks and macrotasks?"
    """
    from .llm_feedback import GROQ_API_KEYS
    if not (GROQ_API_KEYS or os.getenv("GROQ_API_KEY")):
        return None

    history = ""
    for entry in conversation:
        history += f"  Interviewer: {entry['question']}\n"
        answer_snippet = entry['answer'][:300]
        history += f"  Candidate: {answer_snippet}\n\n"

    prompt = f"""You are a senior technical interviewer conducting a cross-questioning session. Based on the candidate's introduction and their latest answer, generate the next question.

CANDIDATE'S INTRODUCTION:
"{introduction}"

CONVERSATION SO FAR:
{history}

RULES:
- Reference something specific from their LAST answer (e.g. "You mentioned X, can you explain...")
- If their answer was vague or incomplete, dig deeper into the same topic
- If their answer was solid, escalate to a harder related concept or ask about tradeoffs
- Keep it conversational — sound like a real interviewer naturally following up
- Do NOT repeat any question already asked
- ONE question only

Respond in this format:
QUESTION: <your cross-question based on their last answer>"""

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _call_llm, prompt)
    if raw:
        return _parse_question(raw)
    return None
