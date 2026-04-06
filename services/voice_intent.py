"""
voice_intent.py
================
Voice intent classifier for the mock-interview pipeline.

Two distinct use cases, one classifier:

  1. DURING ANSWER  (called from /ws/transcribe before evaluation fires)
     Possible intents: silence | repeat | skip | end | answer
     "answer" is the only intent that proceeds to NLP evaluation.

  2. BETWEEN QUESTIONS  (called from /ws/intent after evaluation completes)
     Possible intents: next | continue | repeat | skip | end
     Replaces the old classify_confirmation_intent() stub.

Design choices
--------------
- Zero ML at inference time — pure text pattern matching + heuristics.
  Runs in < 1ms, no model loading, safe to call on every EndOfTurn event.
- STT-aware: patterns account for Deepgram transcription noise, missing
  punctuation, filler words, and common mishearings.
- Filler-stripping happens before any length check so "um uh" doesn't
  count as a two-word answer.
- Intent priority order (checked top-down): end > skip > repeat > silence > answer.
  This means a command like "skip I don't know" is classified as skip, not answer.

Integration
-----------
  from services.voice_intent import classify_voice_intent, classify_nav_intent

  # During answer turn
  result = classify_voice_intent(transcript)
  if result.intent != "answer":
      # handle non-answer (see routers/transcription.py)
      ...

  # Between questions
  result = classify_nav_intent(transcript)
  # result.intent is: next | continue | repeat | skip | end
"""

import re
from dataclasses import dataclass
from typing import Literal

# ── types ──────────────────────────────────────────────────────────────────────

AnswerIntent = Literal["silence", "repeat", "skip", "end", "answer"]
NavIntent    = Literal["next", "continue", "repeat", "skip", "end"]


@dataclass
class IntentResult:
    intent:       str    # one of the intent literals above
    confidence:   float  # 0.0–1.0 — how certain the classifier is
    reason:       str    # short log-friendly explanation
    cleaned_text: str    # filler-stripped normalised text (for downstream use)


# ── filler words that Deepgram reliably transcribes ───────────────────────────
_FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm", "erm", "er",
    "ah", "ahh", "oh", "okay", "ok", "like", "you know",
    "i mean", "so", "well", "right", "yeah", "yep", "ya",
    "basically", "literally", "actually",
}

# ── pattern groups ────────────────────────────────────────────────────────────
# Each group is a list of substrings.  A match fires if ANY substring is found
# in the normalised (lowercase, punctuation-stripped) transcript.
# Longer / more-specific phrases come first so they shadow shorter ones.

_END_PATTERNS = [
    "end the interview",
    "stop the interview",
    "finish the interview",
    "terminate the interview",
    "end interview",
    "stop interview",
    "i want to stop",
    "i want to end",
    "let's end",
    "lets end",
    "quit the interview",
    "quit interview",
    "i'm done with the interview",
    "im done with the interview",
    "end session",
    "stop session",
]

_SKIP_PATTERNS = [
    "skip this question",
    "skip the question",
    "next question please",
    "can we move on",
    "can you move on",
    "move to the next",
    "move on to next",
    "go to next",
    "go to the next",
    "i want to skip",
    "i'll skip",
    "ill skip",
    "skip this one",
    "skip this",
    "pass this",
    "pass on this",
    "i'll pass",
    "ill pass",
    "just skip",
    "please skip",
    "skip",       # bare "skip" — low specificity, checked last in this group
    "pass",       # bare "pass"
    "next",       # bare "next" — also used in nav, keep here for answer-phase
]

_REPEAT_PATTERNS = [
    "can you repeat the question",
    "could you repeat the question",
    "please repeat the question",
    "repeat the question",
    "say the question again",
    "what was the question",
    "what is the question",
    "what's the question",
    "whats the question",
    "say that again",
    "can you say that again",
    "could you say that again",
    "can you repeat that",
    "could you repeat that",
    "please repeat that",
    "repeat that",
    "come again",
    "pardon",
    "i didn't hear",
    "i did not hear",
    "i couldn't hear",
    "i could not hear",
    "didn't catch that",
    "did not catch",
    "didn't get that",
    "did not get that",
    "what did you say",
    "what did you ask",
    "can you ask again",
    "ask again",
    "repeat",     # bare "repeat"
]

# ── nav-only patterns (between-question phase) ────────────────────────────────

_NEXT_PATTERNS = [
    "continue to next",
    "go to next question",
    "move to next question",
    "next question",
    "i'm ready for the next",
    "im ready for the next",
    "ready for next",
    "yes next",
    "yes move on",
    "yes go ahead",
    "done with this",
    "done next",
    "finished",
    "i'm done",
    "im done",
    "move on",
    "move forward",
    "yes",
    "done",
    "next",
    "okay next",
    "ok next",
    "alright next",
    "sure",
    "go ahead",
    "proceed",
    "yep",
    "yeah",
]

_CONTINUE_PATTERNS = [
    "let me continue",
    "let me add",
    "let me finish",
    "i have more to say",
    "i'm not done",
    "im not done",
    "i am not done",
    "not done yet",
    "wait let me",
    "hold on",
    "one moment",
    "one more thing",
    "one more point",
    "wait",
    "more",
    "continue",
    "i want to add",
]


# ── utilities ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation, keeping spaces."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_fillers(text: str) -> str:
    """Remove filler words from a normalised string."""
    words = text.split()
    cleaned = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(cleaned)


def _meaningful_word_count(norm_text: str) -> int:
    """Count words after filler removal."""
    return len(_strip_fillers(norm_text).split())


def _matches_any(norm_text: str, patterns: list) -> tuple[bool, str]:
    """
    Return (matched, matched_pattern).
    Checks each pattern as a substring of the normalised text.
    """
    for pattern in patterns:
        if pattern in norm_text:
            return True, pattern
    return False, ""


# ── main classifiers ──────────────────────────────────────────────────────────

def classify_voice_intent(transcript: str) -> IntentResult:
    """
    Classify a transcript that arrived during the ANSWER phase.

    Returns an IntentResult with intent in:
        "silence"  — nothing meaningful was said; prompt user to answer
        "repeat"   — user asked to hear the question again
        "skip"     — user wants to skip this question
        "end"      — user wants to end the interview
        "answer"   — treat as a real answer, proceed to NLP evaluation

    Call this BEFORE evaluate_answer(). Only proceed to evaluation when
    result.intent == "answer".
    """
    norm         = _normalize(transcript)
    cleaned      = _strip_fillers(norm)
    word_count   = _meaningful_word_count(norm)

    # ── 1. silence: empty transcript ──────────────────────────────────────────
    if not norm:
        return IntentResult(
            intent="silence",
            confidence=1.0,
            reason="empty transcript",
            cleaned_text="",
        )

    # ── 2. silence: filler-only (um, uh, hmm, …) ──────────────────────────────
    if not cleaned:
        return IntentResult(
            intent="silence",
            confidence=0.95,
            reason=f"filler-only transcript: '{norm}'",
            cleaned_text="",
        )

    # ── 3. end interview command ───────────────────────────────────────────────
    matched, pattern = _matches_any(norm, _END_PATTERNS)
    if matched:
        return IntentResult(
            intent="end",
            confidence=0.95,
            reason=f"end pattern matched: '{pattern}'",
            cleaned_text=cleaned,
        )

    # ── 4. skip question command ───────────────────────────────────────────────
    matched, pattern = _matches_any(norm, _SKIP_PATTERNS)
    if matched:
        # "next" and "pass" are ambiguous when combined with answer text.
        # Only classify as skip if the whole utterance is short (≤ 4 words)
        # or the match is a multi-word skip phrase.
        is_bare = pattern in ("skip", "pass", "next")
        if not is_bare or word_count <= 4:
            return IntentResult(
                intent="skip",
                confidence=0.90 if not is_bare else 0.75,
                reason=f"skip pattern matched: '{pattern}'",
                cleaned_text=cleaned,
            )

    # ── 5. repeat question command ─────────────────────────────────────────────
    matched, pattern = _matches_any(norm, _REPEAT_PATTERNS)
    if matched:
        is_bare = pattern in ("repeat", "pardon")
        if not is_bare or word_count <= 4:
            return IntentResult(
                intent="repeat",
                confidence=0.90 if not is_bare else 0.75,
                reason=f"repeat pattern matched: '{pattern}'",
                cleaned_text=cleaned,
            )

    # ── 6. silence: too short to be a real answer ─────────────────────────────
    # After removing fillers, if fewer than 3 meaningful words remain AND
    # no intent was matched above, treat as silence.
    if word_count < 3:
        return IntentResult(
            intent="silence",
            confidence=0.80,
            reason=f"too short ({word_count} meaningful word(s)) and no intent matched",
            cleaned_text=cleaned,
        )

    # ── 7. default: real answer ────────────────────────────────────────────────
    return IntentResult(
        intent="answer",
        confidence=0.90,
        reason=f"no intent pattern matched, {word_count} meaningful words",
        cleaned_text=cleaned,
    )


def classify_nav_intent(transcript: str) -> IntentResult:
    """
    Classify a transcript from the BETWEEN-QUESTIONS phase (/ws/intent).
    Replaces the old classify_confirmation_intent() stub in evaluation.py.

    Returns an IntentResult with intent in:
        "next"      — move to the next question  (default)
        "continue"  — user has more to add to their answer
        "repeat"    — user wants to hear the question again
        "skip"      — user wants to skip without answering
        "end"       — user wants to end the interview

    The default fallback is "next" — a navigation WebSocket that can't
    classify defaults to advancing, which is the least-disruptive action.
    """
    norm    = _normalize(transcript)
    cleaned = _strip_fillers(norm)

    # Empty → assume "next" (they may have just pressed the button with no audio)
    if not norm or not cleaned:
        return IntentResult(
            intent="next",
            confidence=0.60,
            reason="empty/filler transcript — defaulting to next",
            cleaned_text="",
        )

    # Priority order: end > repeat > continue > next > skip
    # "next" is checked BEFORE skip so bare "next" advances rather than skips.
    matched, pattern = _matches_any(norm, _END_PATTERNS)
    if matched:
        return IntentResult(intent="end", confidence=0.95,
                            reason=f"end: '{pattern}'", cleaned_text=cleaned)

    matched, pattern = _matches_any(norm, _REPEAT_PATTERNS)
    if matched:
        return IntentResult(intent="repeat", confidence=0.90,
                            reason=f"repeat: '{pattern}'", cleaned_text=cleaned)

    matched, pattern = _matches_any(norm, _CONTINUE_PATTERNS)
    if matched:
        return IntentResult(intent="continue", confidence=0.88,
                            reason=f"continue: '{pattern}'", cleaned_text=cleaned)

    matched, pattern = _matches_any(norm, _NEXT_PATTERNS)
    if matched:
        return IntentResult(intent="next", confidence=0.88,
                            reason=f"next: '{pattern}'", cleaned_text=cleaned)

    matched, pattern = _matches_any(norm, _SKIP_PATTERNS)
    if matched:
        return IntentResult(intent="skip", confidence=0.88,
                            reason=f"skip: '{pattern}'", cleaned_text=cleaned)

    # Fallback
    return IntentResult(
        intent="next",
        confidence=0.55,
        reason="no nav pattern matched — defaulting to next",
        cleaned_text=cleaned,
    )
