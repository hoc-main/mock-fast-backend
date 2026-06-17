"""
services/conversation_agent.py
===============================
Conversational interview agent — uses LLM to:
1. Pick the next question dynamically based on candidate performance
2. Generate natural conversational transitions between questions
3. Maintain interview flow context

The LLM sees: all remaining questions, past Q&A with scores, and decides
what to ask next + how to phrase the transition naturally.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Cached Groq client — reuse across calls to avoid re-init overhead
_groq_client = None
_groq_client_key = None


def _get_groq_client():
    global _groq_client, _groq_client_key
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    if _groq_client and _groq_client_key == api_key:
        return _groq_client
    _groq_client = Groq(api_key=api_key)
    _groq_client_key = api_key
    return _groq_client


class NextQuestionDecision(BaseModel):
    question_id: int = Field(description="ID of the next question to ask")
    transition: str = Field(description="Natural conversational transition to the next question, 1-2 sentences")
    reasoning: str = Field(description="Brief internal reasoning for why this question was chosen")


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_next_question_prompt(
    remaining_questions: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    module_topic: str,
) -> str:
    # Format conversation history with actual answer content
    history_text = ""
    if conversation_history:
        for i, entry in enumerate(conversation_history, 1):
            score_label = "strong" if entry["score"] >= 0.7 else "partial" if entry["score"] >= 0.45 else "weak"
            history_text += f"  Q{i}: {entry['question']}\n"
            # Include candidate's actual answer so LLM can probe deeper
            if entry.get("answer"):
                # Truncate long answers to keep prompt manageable
                answer_snippet = entry["answer"][:300]
                if len(entry["answer"]) > 300:
                    answer_snippet += "..."
                history_text += f"  Candidate said: \"{answer_snippet}\"\n"
            history_text += (
                f"  Answer quality: {score_label} ({entry['score']:.0%})\n"
                f"  Key gaps: {entry.get('gaps', 'none')}\n\n"
            )
    else:
        history_text = "  (This is the first question)\n"

    # Format remaining questions
    questions_text = ""
    for q in remaining_questions:
        questions_text += f"  [{q['id']}] {q['question']}\n"

    return f"""You are an expert interview conductor for a {module_topic} interview.
The candidate just answered a question and has ALREADY received detailed feedback about their gaps. Now you need to pick the next question and write a conversational transition to it.

INTERVIEW HISTORY:
{history_text}
REMAINING QUESTIONS (pick one by ID):
{questions_text}
RULES FOR PICKING THE QUESTION:
- If candidate was weak on a topic, pick a question that probes deeper into what they missed
- If candidate was strong, escalate to a harder or related topic
- Do NOT repeat questions already asked

RULES FOR THE TRANSITION:
- Write a natural, conversational transition (2-4 sentences) that connects the feedback they just received to the next question
- Explain WHY you're asking the next question — connect it to their gaps or strengths
- Make it sound like a real interviewer guiding the conversation, not reading from a script
- Good examples:
  "So we talked about how overfitting happens when your model is too complex. One of the most common tools we use to fight that is regularization — it basically penalizes the model for being too complex. So tell me, what do you know about regularization and why we use it in machine learning?"
  "Alright, so you clearly understand the basics of how trees work, but what I want to dig into now is how we actually validate that our model generalizes. This is where cross-validation comes in. Can you walk me through how cross-validation works and why its important?"
- Do NOT start with generic praise like "Good job" or "Great answer"
- Make it feel like the interviewer is thinking out loud and naturally arriving at the next question
- Use plain ASCII text only

Respond in EXACTLY this format:
QUESTION_ID: <number>
TRANSITION: <your conversational transition that explains why this question is next and naturally asks it>
REASONING: <why you picked this question>"""


# ── Sync call (runs in thread pool) ──────────────────────────────────────────

def _pick_next_question_sync(prompt: str) -> Optional[NextQuestionDecision]:
    client = _get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if not client:
        return None

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            is_reasoning = "oss" in model or "reasoning" in model
            call_kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_completion_tokens=500 if is_reasoning else 250,
                top_p=0.9,
                stream=False,
            )
            if is_reasoning:
                call_kwargs["reasoning_effort"] = "low"
            completion = client.chat.completions.create(**call_kwargs)
            msg = completion.choices[0].message
            raw = (msg.content or "").strip()
            reasoning_text = (getattr(msg, "reasoning", None) or "").strip()
            result = _parse_decision(raw)
            if not result and reasoning_text:
                result = _parse_decision(reasoning_text)
            return result
        except Exception as exc:
            if attempt < max_retries:
                wait = 1.5 * (attempt + 1)
                logger.info(f"Conversation agent retry {attempt + 1}/{max_retries} after {wait}s: {exc}")
                time.sleep(wait)
            else:
                logger.warning(f"Conversation agent LLM call failed after {max_retries + 1} attempts: {exc}")
                return None


def _parse_decision(raw: str) -> Optional[NextQuestionDecision]:
    question_id = None
    transition = None
    reasoning = None

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("QUESTION_ID:"):
            try:
                question_id = int(stripped.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif upper.startswith("TRANSITION:"):
            transition = stripped.split(":", 1)[1].strip()
        elif upper.startswith("REASONING:"):
            reasoning = stripped.split(":", 1)[1].strip()

    if question_id and transition:
        return NextQuestionDecision(
            question_id=question_id,
            transition=transition,
            reasoning=reasoning or "",
        )
    return None


# ── Public async API ──────────────────────────────────────────────────────────

async def pick_next_question(
    remaining_questions: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    module_topic: str = "technical",
) -> Optional[NextQuestionDecision]:
    """
    Ask the LLM to pick the next question and generate a transition.

    Args:
        remaining_questions: List of {id, question, topic} dicts for unanswered questions
        conversation_history: List of {question, answer, score, gaps} for answered questions
        module_topic: The module/domain name for context

    Returns:
        NextQuestionDecision or None (caller falls back to sequential)
    """
    if not os.getenv("GROQ_API_KEY") or not remaining_questions:
        return None

    prompt = _build_next_question_prompt(
        remaining_questions, conversation_history, module_topic
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _pick_next_question_sync, prompt)
        # Validate the chosen ID exists in remaining questions
        if result:
            valid_ids = {q["id"] for q in remaining_questions}
            if result.question_id not in valid_ids:
                logger.warning(f"LLM picked invalid question_id {result.question_id}, falling back")
                return None
        return result
    except Exception as exc:
        logger.warning(f"Conversation agent error: {exc}")
        return None
