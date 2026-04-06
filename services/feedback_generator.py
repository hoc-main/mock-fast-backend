"""Feedback generation utilities.

This module keeps the backend backward compatible while adding a richer
question-level feedback payload and a session-level summary.

Design:
- If a fine-tuned seq2seq model exists in services/feedback_model/, it is used.
- Otherwise the generator falls back to a deterministic rubric-driven template.
- The same heuristic generator is also used to build synthetic training targets.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .nlp_features import normalize, split_sentences, tokenize

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.getenv("FEEDBACK_MODEL_DIR", BASE_DIR / "feedback_model"))
DEFAULT_MODEL_NAME = os.getenv("FEEDBACK_BASE_MODEL", "google/flan-t5-small")

FILLER_WORDS = {
    "um", "uh", "erm", "like", "you know", "actually", "basically",
    "sort of", "kind of", "i mean", "right", "well",
}


def _unique_nonempty(values: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        value = (value or "").strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _merge_texts(*parts: Optional[str]) -> str:
    return " ".join(_unique_nonempty(p for p in parts if p))


def _safe_percent(value: float) -> int:
    return max(0, min(100, int(round((value or 0.0) * 100))))


def _score_band(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "partial"
    return "weak"


def _keyword_preview(items: Sequence[str], limit: int = 3) -> str:
    items = [str(x).strip() for x in items if str(x).strip()]
    if not items:
        return ""
    items = items[:limit]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _detect_repetition(text: str) -> Dict[str, Any]:
    tokens = tokenize(text)
    repeated_consecutive = re.findall(r"\b(\w+)(?:\s+\1\b)+", normalize(text))
    repeated_ratio = 0.0
    if tokens:
        repeated_ratio = 1.0 - (len(set(tokens)) / len(tokens))
    filler_hits = []
    lower = normalize(text)
    for filler in FILLER_WORDS:
        if filler in lower:
            filler_hits.append(filler)
    return {
        "repeated_consecutive": _unique_nonempty(repeated_consecutive),
        "repeated_ratio": round(repeated_ratio, 3),
        "filler_hits": filler_hits,
    }


def _detect_structure(text: str) -> Dict[str, Any]:
    sentences = split_sentences(text)
    tokens = tokenize(text)
    punctuation = len(re.findall(r"[.,;:!?]", text or ""))
    long_response = len(tokens) >= 30
    no_punctuation = punctuation == 0 and long_response
    too_short = len(tokens) < 8
    avg_sentence_length = mean([len(s.split()) for s in sentences]) if sentences else 0.0
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "no_punctuation": no_punctuation,
        "too_short": too_short,
    }


def _compose_question_feedback(
    *,
    question_text: str,
    expected_answer: str,
    candidate_answer: str,
    evaluation: Dict[str, Any],
    answer_variants: Optional[Sequence[str]] = None,
    scoring_rationale: str = "",
) -> Dict[str, Any]:
    score = float(evaluation.get("final_score", evaluation.get("score", 0.0)) or 0.0)
    semantic_score = float(evaluation.get("semantic_score", 0.0) or 0.0)
    keyword_score = float(evaluation.get("keyword_score", 0.0) or 0.0)
    question_relevance = float(evaluation.get("question_relevance", 0.0) or 0.0)
    lexical_diversity = float(evaluation.get("lexical_diversity", 0.0) or 0.0)
    discourse_score = float(evaluation.get("discourse_score", 0.0) or 0.0)
    length_score = float(evaluation.get("length_score", 0.0) or evaluation.get("answer_length_ratio", 0.0) or 0.0)
    missing_keywords = _unique_nonempty(evaluation.get("missing_keywords", []) or [])
    matched_keywords = _unique_nonempty(evaluation.get("matched_keywords", []) or [])

    repetition = _detect_repetition(candidate_answer)
    structure = _detect_structure(candidate_answer)
    band = _score_band(score)

    positives: List[str] = []
    negatives: List[str] = []
    bullets: List[str] = []

    if semantic_score >= 0.75:
        positives.append("the core idea is clearly related to the expected answer")
    elif semantic_score >= 0.5:
        positives.append("the response is in the right topic area")
    else:
        negatives.append("the answer is only loosely connected to the target concept")

    if keyword_score >= 0.65 or matched_keywords:
        preview = _keyword_preview(matched_keywords, 3)
        if preview:
            positives.append(f"it includes useful terminology such as {preview}")
        else:
            positives.append("it uses some of the expected subject language")

    if question_relevance >= 0.70:
        positives.append("it stays focused on the asked question")
    elif question_relevance < 0.40:
        negatives.append("parts of the response drift away from the question")

    if discourse_score >= 0.40:
        positives.append("the explanation shows some structure and flow")
    else:
        negatives.append("the response needs a clearer structure")

    if lexical_diversity >= 0.60:
        positives.append("the vocabulary is varied enough to keep the answer natural")
    elif lexical_diversity < 0.40:
        negatives.append("the wording feels limited or repetitive")

    if length_score < 0.45:
        negatives.append("the answer is too short to fully develop the idea")
    elif length_score > 0.90:
        positives.append("the answer gives enough detail to be informative")

    if repetition["repeated_consecutive"] or repetition["repeated_ratio"] > 0.35:
        negatives.append("there are repeated words or phrases that reduce clarity")

    if repetition["filler_hits"]:
        negatives.append("filler expressions make the answer sound less confident")

    if structure["no_punctuation"]:
        negatives.append("the response would read better with shorter, punctuated sentences")

    if structure["too_short"]:
        negatives.append("the answer needs at least one more supporting point")

    if not positives:
        positives.append("the answer shows some awareness of the topic")
    if not negatives:
        negatives.append("the response can still be sharpened with a more polished structure")

    if band in {"strong", "good"}:
        first_sentence = (
            f"You gave a relevant answer and showed a solid grasp of the concept, especially where you used {' and '.join(positives[:2]).rstrip('.')}"
        )
        second_sentence = (
            f"The main gap is that {'; '.join(negatives[:2]).rstrip('.')}."
        )
        third_sentence = (
            "A stronger version would start with a direct definition, add one or two supporting points, and finish with a short example or consequence."
        )
    else:
        first_sentence = (
            f"Your answer shows partial understanding, but it does not yet fully capture the expected concept."
        )
        second_sentence = (
            f"It does better on {' and '.join(positives[:1]).rstrip('.')} while still being limited because {'; '.join(negatives[:2]).rstrip('.')}."
        )
        third_sentence = (
            "Focus on the core definition first, then add the missing facts, and keep the structure more direct and concise."
        )

    feedback_paragraph = " ".join([first_sentence, second_sentence, third_sentence]).strip()

    if missing_keywords:
        bullets.append(f"Add the missing points: {_keyword_preview(missing_keywords, 3)}.")
    elif answer_variants:
        bullets.append("Compare your response against the reference answer and include the most important missing idea.")
    else:
        bullets.append("Include one more concrete point to complete the explanation.")

    if structure["too_short"] or length_score < 0.55:
        bullets.append("Expand the answer with a definition, one supporting detail, and a brief example.")
    else:
        bullets.append("Keep the answer focused and remove any unnecessary drift.")

    if repetition["repeated_consecutive"] or repetition["repeated_ratio"] > 0.25:
        bullets.append("Remove repeated words or repeated phrases to make the answer cleaner.")

    if repetition["filler_hits"]:
        bullets.append("Reduce filler words such as um, like, or basically so the delivery sounds more confident.")

    if structure["no_punctuation"]:
        bullets.append("Break the answer into shorter sentences so the grammar and flow are easier to follow.")

    if discourse_score < 0.35:
        bullets.append("Use a clearer sequence: definition → explanation → example.")
    elif discourse_score < 0.55:
        bullets.append("Add a linking phrase such as ‘for example’ or ‘therefore’ to improve flow.")

    bullets = _unique_nonempty(bullets)[:4]

    if score >= 0.80:
        question_summary = "Strong answer overall, with only minor improvements needed for polish and completeness."
    elif score >= 0.60:
        question_summary = "Mostly on track, but the answer needs more detail and a cleaner structure."
    elif score >= 0.40:
        question_summary = "Partially correct, though the explanation is still incomplete and needs clearer support."
    else:
        question_summary = "The response is too weak or too vague to show solid understanding yet."

    if scoring_rationale:
        session_summary_hint = "The answer fits the current quality band, but the synthetic dataset rationale still suggests human review for edge cases."
    else:
        session_summary_hint = "Across this answer, focus on clarity, completeness, and a tighter structure."

    return {
        "feedback_paragraph": feedback_paragraph,
        "improvement_bullets": bullets,
        "question_summary": question_summary,
        "session_summary_hint": session_summary_hint,
        "pros": positives[:3],
        "cons": negatives[:3],
        "bands": {
            "score_band": band,
            "score_percent": _safe_percent(score),
        },
        "signals": {
            "repetition": repetition,
            "structure": structure,
        },
    }


class FeedbackGenerator:
    """Optional transformer-backed feedback generator with template fallback."""

    def __init__(self, model_dir: Optional[Path | str] = None, base_model: Optional[str] = None):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.base_model = base_model or DEFAULT_MODEL_NAME
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    @property
    def available(self) -> bool:
        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            return True
        return False

    def _load(self) -> None:
        if self._model is not None or self._tokenizer is not None:
            return
        if not self.available:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            self._pipeline = pipeline(
                "text2text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
        except Exception:
            self._tokenizer = None
            self._model = None
            self._pipeline = None

    def build_prompt(
        self,
        *,
        question_text: str,
        expected_answer: str,
        candidate_answer: str,
        evaluation: Dict[str, Any],
        answer_variants: Optional[Sequence[str]] = None,
        scoring_rationale: str = "",
    ) -> str:
        variants_text = "\n".join(f"- {v}" for v in _unique_nonempty(answer_variants or [])) or "-"
        payload = {
            "question": question_text,
            "reference_answer": expected_answer,
            "answer_variants": variants_text,
            "candidate_answer": candidate_answer,
            "evaluation": evaluation,
            "scoring_rationale": scoring_rationale,
        }
        return (
            "Write a coaching response in strict JSON with the keys "
            "feedback_paragraph, improvement_bullets, question_summary, session_summary_hint, pros, cons. "
            "The feedback_paragraph must be 3 to 4 sentences long and include both positives and negatives. "
            "The improvement_bullets must contain 3 to 4 short bullet points. "
            f"INPUT: {json.dumps(payload, ensure_ascii=False)}"
        )

    def generate(
        self,
        *,
        question_text: str,
        expected_answer: str,
        candidate_answer: str,
        evaluation: Dict[str, Any],
        answer_variants: Optional[Sequence[str]] = None,
        scoring_rationale: str = "",
        force_template: bool = False,
    ) -> Dict[str, Any]:
        fallback = _compose_question_feedback(
            question_text=question_text,
            expected_answer=expected_answer,
            candidate_answer=candidate_answer,
            evaluation=evaluation,
            answer_variants=answer_variants,
            scoring_rationale=scoring_rationale,
        )

        if force_template:
            fallback["model_used"] = False
            return fallback

        self._load()
        if not self._pipeline:
            fallback["model_used"] = False
            return fallback

        prompt = self.build_prompt(
            question_text=question_text,
            expected_answer=expected_answer,
            candidate_answer=candidate_answer,
            evaluation=evaluation,
            answer_variants=answer_variants,
            scoring_rationale=scoring_rationale,
        )
        try:
            result = self._pipeline(
                prompt,
                max_new_tokens=220,
                do_sample=False,
                temperature=0.0,
                truncation=True,
            )[0]["generated_text"]
            parsed = _parse_generated_json(result)
            if parsed:
                parsed.setdefault("model_used", True)
                return parsed
        except Exception:
            pass

        fallback["model_used"] = False
        return fallback


@staticmethod
def _parse_generated_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def generate_question_feedback(
    *,
    question_text: str,
    expected_answer: str,
    candidate_answer: str,
    evaluation: Dict[str, Any],
    answer_variants: Optional[Sequence[str]] = None,
    scoring_rationale: str = "",
    force_template: bool = False,
) -> Dict[str, Any]:
    generator = FeedbackGenerator()
    return generator.generate(
        question_text=question_text,
        expected_answer=expected_answer,
        candidate_answer=candidate_answer,
        evaluation=evaluation,
        answer_variants=answer_variants,
        scoring_rationale=scoring_rationale,
        force_template=force_template,
    )


def generate_session_summary(question_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the results from one session into a session-level summary."""
    if not question_results:
        return {
            "session_summary": "No answers were recorded for this session.",
            "overall_strengths": [],
            "overall_improvements": ["Record at least one answer so the session summary can be generated."],
        }

    scores = [float(item.get("final_score", 0.0) or 0.0) for item in question_results]
    semantic_scores = [float(item.get("semantic_score", 0.0) or 0.0) for item in question_results]
    keyword_scores = [float(item.get("keyword_score", 0.0) or 0.0) for item in question_results]
    question_relevance_scores = [float(item.get("question_relevance", 0.0) or 0.0) for item in question_results]

    all_missing_keywords = []
    all_feedback = []
    all_tips = []
    for item in question_results:
        all_missing_keywords.extend(item.get("missing_keywords", []) or [])
        if item.get("feedback"):
            all_feedback.append(item["feedback"])
        if item.get("tip"):
            all_tips.append(item["tip"])

    counter = Counter([normalize(x) for x in all_missing_keywords if x])
    most_common_missing = [item for item, _count in counter.most_common(4)]

    avg_score = mean(scores)
    avg_semantic = mean(semantic_scores)
    avg_keyword = mean(keyword_scores)
    avg_relevance = mean(question_relevance_scores)

    strong_points: List[str] = []
    improvement_points: List[str] = []

    if avg_semantic >= 0.7:
        strong_points.append("the answers stayed reasonably close to the expected concepts")
    if avg_keyword >= 0.6:
        strong_points.append("several answers used the right domain-specific terminology")
    if avg_relevance >= 0.65:
        strong_points.append("most responses stayed on topic")
    if any("definition" in normalize(x) for x in all_feedback):
        strong_points.append("some responses already had a clear definition-first structure")

    if avg_score < 0.8:
        improvement_points.append("answers need more completeness and stronger supporting detail")
    if avg_relevance < 0.65:
        improvement_points.append("some responses should stay closer to the exact question")
    if most_common_missing:
        improvement_points.append(f"repeat the missing concepts more consistently, especially {_keyword_preview(most_common_missing, 3)}")
    if any("repeat" in normalize(x) for x in all_tips):
        improvement_points.append("reduce repetition and filler words across answers")

    if not strong_points:
        strong_points.append("the session showed partial understanding of the material")
    if not improvement_points:
        improvement_points.append("keep refining structure, clarity, and answer depth")

    session_summary = (
        f"Across the session, the average score was {_safe_percent(avg_score)}%, which suggests a { _score_band(avg_score) } overall performance. "
        f"The strongest area was the conceptual alignment with the reference answers, while the biggest opportunity is to make answers more complete and better structured. "
        f"To improve further, keep the response focused, add missing key terms, and finish each answer with a clear concluding line."
    )

    return {
        "session_summary": session_summary,
        "overall_strengths": strong_points[:4],
        "overall_improvements": improvement_points[:4],
        "average_score": round(avg_score, 4),
        "average_semantic_score": round(avg_semantic, 4),
        "average_keyword_score": round(avg_keyword, 4),
        "average_relevance_score": round(avg_relevance, 4),
        "common_missing_keywords": most_common_missing,
    }


def build_training_example(
    sample: Dict[str, Any],
    *,
    force_template: bool = True,
) -> Dict[str, Any]:
    """Convert a raw dataset row into a supervised generation example."""
    q = sample.get("question_data", {})
    evaluation = {
        "final_score": sample.get("score", 0.0),
        "semantic_score": sample.get("labels", {}).get("semantic_similarity", 0.0),
        "keyword_score": sample.get("labels", {}).get("keyword_coverage", 0.0),
        "question_relevance": sample.get("labels", {}).get("question_relevance", 0.0),
        "lexical_diversity": sample.get("labels", {}).get("lexical_diversity", 0.0),
        "discourse_score": sample.get("labels", {}).get("discourse_quality", 0.0),
        "length_score": sample.get("labels", {}).get("answer_completeness", 0.0),
        "missing_keywords": _extract_missing_keywords(q, sample.get("candidate", "")),
        "matched_keywords": _extract_matched_keywords(q, sample.get("candidate", "")),
    }
    payload = generate_question_feedback(
        question_text=q.get("question", ""),
        expected_answer=q.get("primary_answer", q.get("answer", "")),
        candidate_answer=sample.get("candidate", ""),
        evaluation=evaluation,
        answer_variants=q.get("answer_variants", []),
        scoring_rationale=sample.get("scoring_rationale", ""),
        force_template=force_template,
    )

    input_text = json.dumps(
        {
            "question": q.get("question", ""),
            "reference_answers": q.get("answer_variants", []),
            "candidate_answer": sample.get("candidate", ""),
            "metrics": evaluation,
            "original_quality": sample.get("_migration_notes", {}).get("original_quality"),
        },
        ensure_ascii=False,
    )

    target_text = json.dumps(
        {
            "feedback_paragraph": payload["feedback_paragraph"],
            "improvement_bullets": payload["improvement_bullets"],
            "question_summary": payload["question_summary"],
            "session_summary_hint": payload["session_summary_hint"],
        },
        ensure_ascii=False,
    )

    return {"input_text": input_text, "target_text": target_text, "payload": payload}


def _extract_missing_keywords(question_data: Dict[str, Any], candidate: str) -> List[str]:
    expected = question_data.get("expected_keywords", {}) or {}
    if isinstance(expected, dict):
        keywords = []
        for group in ["critical", "supporting", "bonus"]:
            keywords.extend(expected.get(group, []) or [])
    else:
        keywords = list(expected)
    cand = normalize(candidate)
    missing = []
    for kw in keywords:
        if normalize(str(kw)) not in cand:
            missing.append(str(kw))
    return _unique_nonempty(missing)


def _extract_matched_keywords(question_data: Dict[str, Any], candidate: str) -> List[str]:
    expected = question_data.get("expected_keywords", {}) or {}
    if isinstance(expected, dict):
        keywords = []
        for group in ["critical", "supporting", "bonus"]:
            keywords.extend(expected.get(group, []) or [])
    else:
        keywords = list(expected)
    cand = normalize(candidate)
    matched = []
    for kw in keywords:
        if normalize(str(kw)) in cand:
            matched.append(str(kw))
    return _unique_nonempty(matched)
