"""
feedback_generator.py
======================
Rich, per-question feedback and session-summary generator for mock interviews.

Inputs per question
-------------------
- transcript     : str   — raw STT candidate answer
- metrics        : dict  — output of evaluation.evaluate_answer()
- question_data  : dict  — question record from dataset (with answer_variants, keywords, etc.)

Per-question output
-------------------
{
  "question_id"      : str,
  "question_text"    : str,
  "score"            : float,
  "score_tier"       : "strong" | "good" | "partial" | "weak",
  "narrative"        : str,          # 3-4 sentences: balanced pros + cons
  "improvement_tips" : list[str],    # 3-4 actionable bullet points
  "grammar_notes"    : dict,         # filler counts, repetitions, sentence stats
  "content_analysis" : dict,         # present / missing keywords per tier
  "stt_flags"        : list[str],    # spoken artifacts detected
}

Session summary output
-----------------------
{
  "session_score"       : float,      # weighted average across questions
  "score_trend"         : list[float],
  "strengths"           : list[str],
  "improvement_areas"   : list[str],
  "grammar_summary"     : dict,
  "summary_paragraph"   : str,        # 3-4 sentence holistic summary
  "per_question_scores" : list[dict],
}
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

FILLER_WORDS = {
    "um", "uh", "like", "basically", "right", "okay", "so", "you know",
    "kind of", "sort of", "actually", "literally", "honestly", "just",
    "i mean", "i guess", "well", "yeah", "yep", "hmm",
}

# STT self-correction patterns  e.g. "the the", "it is it is"
_REPEATED_WORD_RE = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)

# Run-on detector: sentences joined by 3+ "and/so/but" connectives without punctuation
_RUNON_RE = re.compile(
    r"(?<![.!?])\s+(?:and|so|but|then|also)\s+(?:and|so|but|then|also)\s+(?:and|so|but|then|also)",
    re.IGNORECASE,
)

SCORE_TIERS = [
    (0.85, "strong"),
    (0.68, "good"),
    (0.45, "partial"),
    (0.00, "weak"),
]

# Narrative sentence templates keyed by (tier, dimension)
# Each slot has multiple variants so text is less robotic.
_NARRATIVE_TEMPLATES = {
    # Opening sentence — overall impression
    "open_strong":   [
        "Your answer demonstrated a solid grasp of the topic and addressed the question clearly.",
        "You provided a well-structured and accurate response that covered the core idea effectively.",
        "The answer reflected genuine understanding and was largely on point for what was asked.",
    ],
    "open_good":     [
        "Your answer showed a reasonable understanding of the topic, though there is room to sharpen it further.",
        "The response was generally relevant and correct in its direction, but fell short in a few key areas.",
        "You covered the main idea adequately, even if some important details were left out.",
    ],
    "open_partial":  [
        "Your answer touched on the topic but lacked the depth and accuracy needed for a complete response.",
        "The response was partially on track, but it missed several important aspects of the question.",
        "You demonstrated some familiarity with the subject, yet the answer felt incomplete overall.",
    ],
    "open_weak":     [
        "The answer did not adequately address what was being asked and requires significant improvement.",
        "Your response was quite vague or off-topic, making it difficult to assess your understanding.",
        "The answer lacked substance and did not reflect a clear command of the subject matter.",
    ],

    # Positive observation (pros sentence)
    "pro_semantic":  "Your explanation aligned well with the expected answer on a conceptual level.",
    "pro_keywords":  "You successfully incorporated several key terms that are central to this topic.",
    "pro_discourse": "The response had good structural flow, with clear sequencing and logical connectives.",
    "pro_length":    "The length of your answer was appropriate and proportional to the question's scope.",
    "pro_lexical":   "You used a varied vocabulary, which added depth and precision to your response.",
    "pro_relevance": "You stayed focused on the question and did not wander into unrelated territory.",

    # Constructive observation (cons sentence)
    "con_semantic":  "However, the core concept was not explained with sufficient accuracy or completeness.",
    "con_keywords":  "Key terms and concepts that define this topic were either missing or only partially mentioned.",
    "con_discourse": "The answer lacked structured reasoning — it would benefit from clearer sequencing (e.g., definition → explanation → example).",
    "con_length_short": "The response was too brief to demonstrate full understanding; more elaboration was needed.",
    "con_length_long":  "The answer was somewhat verbose, with ideas repeated rather than developed further.",
    "con_relevance": "Portions of the answer drifted away from the specific question, reducing its overall relevance.",
    "con_grammar":   "There were noticeable spoken artifacts — filler words and repetitions — that reduced clarity.",

    # Closing observation
    "close_strong":  "Overall, this was a commendable response that would leave a positive impression in an interview.",
    "close_good":    "With a bit more precision and completeness, this answer could easily reach a strong level.",
    "close_partial": "Focusing on the missing keywords and adding a concrete example would significantly improve this answer.",
    "close_weak":    "Revisiting the fundamentals of this topic and practising a structured 'define it, explain it, then give an example' approach will help greatly.",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_tier(score: float) -> str:
    for threshold, label in SCORE_TIERS:
        if score >= threshold:
            return label
    return "weak"


def _pick(templates: List[str], seed: int = 0) -> str:
    """Deterministic pick from a list to avoid random variation on retries."""
    return templates[seed % len(templates)]


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    stopwords = {
        "a","an","the","is","are","was","were","to","of","and","or","in","on",
        "for","with","that","this","it","as","by","be","at","from","can","i",
        "me","my","we","our","you","your","he","she","they","them","will","would",
        "should","may","might","could","do","did","does","have","has","had","been",
        "also","very","just","but","if","so","than","then","not","no","am","about",
    }
    return [
        w for w in _normalize(text).split()
        if len(w) > 2 and w not in stopwords
    ]


# ── Grammar & STT Analysis ─────────────────────────────────────────────────────

def analyse_grammar_and_stt(transcript: str) -> Dict[str, Any]:
    """
    Returns a structured grammar/STT analysis dict:
    {
      "filler_count"       : int,
      "filler_words_found" : list[str],
      "filler_rate"        : float,        # fillers per 100 words
      "word_repetitions"   : list[str],    # "the the", "it is it is"
      "phrase_repetitions" : list[str],    # repeated 3-4 word phrases
      "sentence_count"     : int,
      "avg_words_per_sent" : float,
      "run_on_detected"    : bool,
      "capitalization_ok"  : bool,         # False if text is all lower (common STT)
      "severity"           : "clean"|"minor"|"moderate"|"heavy",
    }
    """
    if not transcript:
        return {
            "filler_count": 0, "filler_words_found": [], "filler_rate": 0.0,
            "word_repetitions": [], "phrase_repetitions": [],
            "sentence_count": 0, "avg_words_per_sent": 0.0,
            "run_on_detected": False, "capitalization_ok": True,
            "severity": "clean",
        }

    norm = transcript.lower()
    words = norm.split()
    total_words = max(len(words), 1)

    # Filler detection
    found_fillers: List[str] = []
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, norm)
        found_fillers.extend(matches)
    filler_count = len(found_fillers)
    filler_types = list(dict.fromkeys(found_fillers))  # unique, order preserved
    filler_rate = (filler_count / total_words) * 100

    # Word-level repetitions ("the the", "was was")
    word_reps = list(set(_REPEATED_WORD_RE.findall(norm)))

    # Phrase-level repetitions (3-gram repeated)
    phrase_reps: List[str] = []
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    seen_ngrams: set = set()
    for ng in trigrams:
        if trigrams.count(ng) > 1 and ng not in seen_ngrams:
            phrase_reps.append(ng)
            seen_ngrams.add(ng)

    # Sentence stats
    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_words = total_words / sentence_count

    run_on = bool(_RUNON_RE.search(norm))
    cap_ok = transcript[0].isupper() if transcript else True

    # Severity
    issues = (
        (1 if filler_rate > 15 else 0) +
        (1 if filler_rate > 8 else 0) +
        (1 if word_reps else 0) +
        (1 if len(phrase_reps) > 1 else 0) +
        (1 if run_on else 0)
    )
    severity = {0: "clean", 1: "minor", 2: "moderate"}.get(issues, "heavy")

    return {
        "filler_count":       filler_count,
        "filler_words_found": filler_types[:8],
        "filler_rate":        round(filler_rate, 1),
        "word_repetitions":   word_reps[:5],
        "phrase_repetitions": phrase_reps[:3],
        "sentence_count":     sentence_count,
        "avg_words_per_sent": round(avg_words, 1),
        "run_on_detected":    run_on,
        "capitalization_ok":  cap_ok,
        "severity":           severity,
    }


# ── Content Analysis ───────────────────────────────────────────────────────────

def analyse_content(transcript: str, question_data: Dict[str, Any],
                    metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares transcript against all answer variants and keyword tiers.
    Returns:
    {
      "present_keywords"  : {"critical": [...], "supporting": [...], "bonus": [...]},
      "missing_keywords"  : {"critical": [...], "supporting": [...], "bonus": [...]},
      "variant_scores"    : [float, float, float],  # similarity to each variant
      "best_variant_idx"  : int,
      "concepts_covered"  : list[str],   # high-level points matched
      "concepts_missing"  : list[str],   # high-level points absent
    }
    """
    norm_transcript = _normalize(transcript)
    cand_tokens = set(_tokenize(transcript))

    # Keyword tiers — support both dict format and flat list format
    raw_kw = question_data.get("expected_keywords", {})
    if isinstance(raw_kw, dict):
        critical   = [k.lower() for k in raw_kw.get("critical", [])]
        supporting = [k.lower() for k in raw_kw.get("supporting", [])]
        bonus      = [k.lower() for k in raw_kw.get("bonus", [])]
    else:
        # flat list — treat all as supporting
        critical   = []
        supporting = [k.lower() for k in (raw_kw or [])]
        bonus      = []

    def _check_kw(kw_list: List[str]) -> Tuple[List[str], List[str]]:
        present, absent = [], []
        for kw in kw_list:
            # Check either as a phrase or as individual tokens
            if kw in norm_transcript or all(t in cand_tokens for t in kw.split()):
                present.append(kw)
            else:
                absent.append(kw)
        return present, absent

    crit_present,   crit_missing   = _check_kw(critical)
    supp_present,   supp_missing   = _check_kw(supporting)
    bonus_present,  bonus_missing  = _check_kw(bonus)

    # Variant similarity (embedding-free: Jaccard on token sets as proxy)
    variants: List[str] = []
    if question_data.get("primary_answer"):
        variants.append(question_data["primary_answer"])
    for key in ("answer_variant_1", "answer_variant_2"):
        if question_data.get(key):
            variants.append(question_data[key])
    for v in question_data.get("answer_variants", []):
        if v and v not in variants:
            variants.append(v)

    variant_scores: List[float] = []
    for var in variants:
        var_tokens = set(_tokenize(var))
        if not var_tokens:
            variant_scores.append(0.0)
            continue
        intersection = cand_tokens & var_tokens
        union = cand_tokens | var_tokens
        variant_scores.append(len(intersection) / len(union) if union else 0.0)

    best_idx = variant_scores.index(max(variant_scores)) if variant_scores else 0

    # High-level concepts: sentences from the best variant not matched
    best_variant = variants[best_idx] if variants else ""
    best_sentences = [s.strip() for s in re.split(r"[.!?]", best_variant) if s.strip()]
    concepts_covered, concepts_missing = [], []
    for sent in best_sentences:
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            continue
        overlap = len(cand_tokens & sent_tokens) / len(sent_tokens)
        if overlap >= 0.40:
            # Summarise as the first 6 words
            concepts_covered.append(" ".join(sent.split()[:7]).rstrip(",;"))
        else:
            concepts_missing.append(" ".join(sent.split()[:7]).rstrip(",;"))

    return {
        "present_keywords":  {"critical": crit_present, "supporting": supp_present, "bonus": bonus_present},
        "missing_keywords":  {"critical": crit_missing,  "supporting": supp_missing,  "bonus": bonus_missing},
        "variant_scores":    [round(s, 3) for s in variant_scores],
        "best_variant_idx":  best_idx,
        "concepts_covered":  concepts_covered[:5],
        "concepts_missing":  concepts_missing[:5],
    }


# ── Narrative Builder ──────────────────────────────────────────────────────────

def _build_narrative(
    score: float,
    tier: str,
    metrics: Dict[str, Any],
    grammar: Dict[str, Any],
    content: Dict[str, Any],
    seed: int = 0,
) -> str:
    """
    Builds a 3-4 sentence narrative paragraph covering pros and cons.
    """
    T = _NARRATIVE_TEMPLATES
    sentences: List[str] = []

    # 1. Opening sentence
    sentences.append(_pick(T[f"open_{tier}"], seed))

    # 2. Positive observation (pick the strongest metric as the 'pro')
    semantic  = metrics.get("semantic_score", 0.0)
    kw_score  = metrics.get("keyword_score", 0.0)
    discourse = metrics.get("discourse_score", 0.0)
    length    = metrics.get("length_score", 0.0)
    lexical   = metrics.get("lexical_diversity", 0.0)
    relevance = metrics.get("question_relevance", 0.0)

    pro_scores = [
        (semantic,  T["pro_semantic"]),
        (kw_score,  T["pro_keywords"]),
        (discourse, T["pro_discourse"]),
        (length,    T["pro_length"]),
        (lexical,   T["pro_lexical"]),
        (relevance, T["pro_relevance"]),
    ]
    # Only add a pro sentence if the best metric is genuinely good (>= 0.55)
    pro_scores_good = [(v, s) for v, s in pro_scores if v >= 0.55]
    if pro_scores_good:
        pro_scores_good.sort(reverse=True)
        sentences.append(pro_scores_good[0][1])

    # 3. Constructive/con observation (pick the weakest metric)
    con_scores = [
        (semantic,  T["con_semantic"]),
        (kw_score,  T["con_keywords"]),
        (discourse, T["con_discourse"]),
        (relevance, T["con_relevance"]),
    ]
    # Length-specific cons
    # word count unused after refactor — kept as placeholder
    if length < 0.45:
        con_scores.append((length, T["con_length_short"]))
    elif lexical < 0.45:
        con_scores.append((lexical, T["con_length_long"]))

    con_scores_bad = [(v, s) for v, s in con_scores if v < 0.60]
    if con_scores_bad:
        con_scores_bad.sort()   # ascending — weakest first
        sentences.append(con_scores_bad[0][1])
    elif grammar["severity"] in ("moderate", "heavy"):
        sentences.append(T["con_grammar"])

    # 4. Closing sentence
    sentences.append(T[f"close_{tier}"])

    # Ensure 3-4 sentences
    return " ".join(sentences[:4])


# ── Improvement Tips Builder ───────────────────────────────────────────────────

def _build_improvement_tips(
    score: float,
    tier: str,
    metrics: Dict[str, Any],
    grammar: Dict[str, Any],
    content: Dict[str, Any],
) -> List[str]:
    """
    Returns 3-4 actionable bullet-point tips.
    """
    tips: List[str] = []

    semantic  = metrics.get("semantic_score", 0.0)
    kw_score  = metrics.get("keyword_score", 0.0)
    discourse = metrics.get("discourse_score", 0.0)
    length    = metrics.get("length_score", 0.0)
    relevance = metrics.get("question_relevance", 0.0)
    lexical   = metrics.get("lexical_diversity", 0.0)

    missing_critical   = content["missing_keywords"]["critical"]
    missing_supporting = content["missing_keywords"]["supporting"]
    concepts_missing   = content["concepts_missing"]

    # Tip 1 — critical missing keywords
    if missing_critical:
        kws = ", ".join(f'"{k}"' for k in missing_critical[:3])
        tips.append(
            f"Make sure to include critical keywords: {kws}. "
            "These are the core terms evaluators expect in a strong answer."
        )
    elif missing_supporting:
        kws = ", ".join(f'"{k}"' for k in missing_supporting[:3])
        tips.append(
            f"Strengthen your answer by weaving in supporting terms like {kws} "
            "to show breadth of knowledge."
        )

    # Tip 2 — content depth
    if concepts_missing:
        point = concepts_missing[0]
        tips.append(
            f"Your answer missed the idea around '{point}...'. "
            "Covering this point would make the response more complete."
        )
    elif semantic < 0.60:
        tips.append(
            "Focus on explaining the concept more precisely — "
            "start with a clear one-line definition, then add the mechanism or process, "
            "and finish with a real-world example."
        )

    # Tip 3 — structure / discourse
    if discourse < 0.40:
        tips.append(
            "Improve structure by using signposting phrases like "
            "'Firstly...', 'This is because...', 'For example...', and 'In summary...'. "
            "Structured answers score higher on clarity."
        )

    # Tip 4 — grammar / STT
    if grammar["severity"] in ("moderate", "heavy"):
        fillers = grammar["filler_words_found"]
        if fillers:
            top_fillers = ", ".join(f'"{f}"' for f in fillers[:3])
            tips.append(
                f"Reduce spoken filler words such as {top_fillers}. "
                "Pausing silently for 1-2 seconds is far more effective than filling with verbal crutches."
            )
        if grammar["word_repetitions"]:
            reps = ", ".join(f'"{r}"' for r in grammar["word_repetitions"][:2])
            tips.append(
                f"Avoid word repetitions like {reps}. "
                "Speak more slowly and deliberately to prevent inadvertent doubling."
            )
    elif grammar["severity"] == "minor" and len(tips) < 3:
        tips.append(
            "Minor filler words were detected. Practising answers out loud with a timer "
            "will naturally reduce hesitation sounds over time."
        )

    # Tip 5 — length
    if length < 0.40 and len(tips) < 4:
        tips.append(
            "The answer was too short. Aim for 40-60 words that include: "
            "(1) a one-line definition, (2) two key points, (3) one short real-world example."
        )
    elif lexical < 0.40 and len(tips) < 4:
        tips.append(
            "The answer felt repetitive in its vocabulary. Try using synonyms and related concepts "
            "to demonstrate a richer command of the subject."
        )

    # Ensure at least 3 tips, at most 4
    if len(tips) == 0:
        tips.append("Review the ideal answer and identify which structural elements made it effective.")
    if len(tips) < 3:
        tips.append(
            "Practice articulating your answer in 3 layers: "
            "what it is → how it works → why it matters."
        )
    if len(tips) < 3:
        tips.append(
            "After answering, ask yourself: "
            "'Did I define the term, explain the mechanism, and give one example?' "
            "If any is missing, add it."
        )

    return tips[:4]


# ── STT Flags ──────────────────────────────────────────────────────────────────

def _build_stt_flags(grammar: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if grammar["filler_rate"] > 8:
        flags.append(
            f"High filler-word density ({grammar['filler_rate']}% of words): "
            + ", ".join(grammar["filler_words_found"][:4])
        )
    if grammar["word_repetitions"]:
        flags.append(
            "Immediate word repetitions detected (typical STT artifact): "
            + ", ".join(f'"{r}"' for r in grammar["word_repetitions"][:3])
        )
    if grammar["phrase_repetitions"]:
        flags.append(
            "Repeated phrases found: "
            + "; ".join(f'"{p}"' for p in grammar["phrase_repetitions"][:2])
        )
    if grammar["run_on_detected"]:
        flags.append(
            "Run-on sentence detected — multiple clauses joined without punctuation. "
            "Break into shorter, clearer sentences."
        )
    if not grammar["capitalization_ok"]:
        flags.append(
            "Text appears to be all lowercase (common in raw STT output). "
            "Verify punctuation and capitalisation before final review."
        )
    return flags


# ── Main per-question entry point ──────────────────────────────────────────────

def generate_question_feedback(
    transcript: str,
    metrics: Dict[str, Any],
    question_data: Dict[str, Any],
    question_id: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Generate rich feedback for a single interview question.

    Parameters
    ----------
    transcript    : The raw STT text of the candidate's answer.
    metrics       : Output dict from evaluation.evaluate_answer().
    question_data : The question record (from dataset JSON).
    question_id   : Optional explicit ID override.
    seed          : Deterministic template selection seed (e.g. question index).

    Returns
    -------
    Full feedback dict (see module docstring).
    """
    score = float(metrics.get("final_score", 0.0))
    tier  = _score_tier(score)

    # Question metadata
    q_id   = question_id or question_data.get("id", "unknown")
    q_text = (
        question_data.get("question")
        or question_data.get("question_text")
        or question_data.get("text", "")
    )

    # Sub-analyses
    grammar = analyse_grammar_and_stt(transcript)
    content = analyse_content(transcript, question_data, metrics)

    # Narrative and tips
    narrative = _build_narrative(score, tier, metrics, grammar, content, seed)
    tips      = _build_improvement_tips(score, tier, metrics, grammar, content)
    stt_flags = _build_stt_flags(grammar)

    return {
        "question_id":      q_id,
        "question_text":    q_text,
        "score":            round(score, 4),
        "score_tier":       tier,
        "narrative":        narrative,
        "improvement_tips": tips,
        "grammar_notes":    grammar,
        "content_analysis": content,
        "stt_flags":        stt_flags,
        # Raw metric snapshot for frontend display
        "metric_snapshot":  {
            "semantic_similarity": round(metrics.get("semantic_score", 0.0), 3),
            "keyword_coverage":    round(metrics.get("keyword_score", 0.0), 3),
            "question_relevance":  round(metrics.get("question_relevance", 0.0), 3),
            "discourse_quality":   round(metrics.get("discourse_score", 0.0), 3),
            "lexical_diversity":   round(metrics.get("lexical_diversity", 0.0), 3),
            "length_score":        round(metrics.get("length_score", 0.0), 3),
        },
    }


# ── Session Summary ────────────────────────────────────────────────────────────

def generate_session_summary(
    question_feedbacks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate feedback across all questions in a session.

    Parameters
    ----------
    question_feedbacks : List of dicts returned by generate_question_feedback().

    Returns
    -------
    Session-level summary dict.
    """
    if not question_feedbacks:
        return {
            "session_score": 0.0,
            "score_trend": [],
            "strengths": [],
            "improvement_areas": [],
            "grammar_summary": {},
            "summary_paragraph": "No questions were answered in this session.",
            "per_question_scores": [],
        }

    scores     = [fb["score"] for fb in question_feedbacks]
    tiers      = [fb["score_tier"] for fb in question_feedbacks]
    n          = len(scores)
    avg_score  = sum(scores) / n
    session_tier = _score_tier(avg_score)

    # Per-question score summary for display
    per_q_scores = [
        {
            "question_id":   fb["question_id"],
            "question_text": fb["question_text"][:80] + "..." if len(fb["question_text"]) > 80 else fb["question_text"],
            "score":         fb["score"],
            "score_tier":    fb["score_tier"],
        }
        for fb in question_feedbacks
    ]

    # Score trend (is the candidate improving across the session?)
    trend_direction = "improving" if (n > 1 and scores[-1] > scores[0]) else \
                      "declining"  if (n > 1 and scores[-1] < scores[0]) else \
                      "consistent"

    # Grammar aggregation
    total_fillers    = sum(fb["grammar_notes"]["filler_count"] for fb in question_feedbacks)
    total_words_est  = sum(
        fb["grammar_notes"]["sentence_count"] * max(fb["grammar_notes"]["avg_words_per_sent"], 1)
        for fb in question_feedbacks
    )
    overall_filler_rate = round((total_fillers / max(total_words_est, 1)) * 100, 1)
    run_ons_detected    = sum(1 for fb in question_feedbacks if fb["grammar_notes"]["run_on_detected"])
    all_fillers = []
    for fb in question_feedbacks:
        all_fillers.extend(fb["grammar_notes"]["filler_words_found"])
    top_fillers = [w for w, _ in Counter(all_fillers).most_common(5)]

    grammar_summary = {
        "total_filler_words":   total_fillers,
        "overall_filler_rate":  overall_filler_rate,
        "run_on_sentences":     run_ons_detected,
        "most_used_fillers":    top_fillers,
        "grammar_tier":         "clean" if overall_filler_rate < 4 else
                                 "minor" if overall_filler_rate < 9 else
                                 "moderate" if overall_filler_rate < 16 else "heavy",
    }

    # Recurring strengths and improvement areas
    all_metrics = [fb["metric_snapshot"] for fb in question_feedbacks]

    def avg_metric(key: str) -> float:
        vals = [m.get(key, 0.0) for m in all_metrics]
        return sum(vals) / len(vals)

    strengths: List[str] = []
    improvements: List[str] = []

    avg_semantic  = avg_metric("semantic_similarity")
    avg_kw        = avg_metric("keyword_coverage")
    avg_relevance = avg_metric("question_relevance")
    avg_discourse = avg_metric("discourse_quality")
    avg_lexical   = avg_metric("lexical_diversity")
    avg_length    = avg_metric("length_score")

    if avg_semantic  >= 0.65: strengths.append("Strong conceptual understanding — answers aligned well with expected content.")
    if avg_kw        >= 0.60: strengths.append("Good keyword coverage — you consistently used domain-specific terminology.")
    if avg_relevance >= 0.70: strengths.append("Excellent question focus — answers were relevant to what was actually asked.")
    if avg_discourse >= 0.50: strengths.append("Structured reasoning — you used logical connectives and organised your thoughts well.")
    if avg_lexical   >= 0.65: strengths.append("Rich vocabulary — varied word choice added precision to your answers.")

    if avg_semantic  < 0.55:  improvements.append("Conceptual accuracy — focus on understanding core definitions and mechanisms, not just topic area.")
    if avg_kw        < 0.45:  improvements.append("Keyword recall — practise identifying and including the 3-5 essential terms for each topic.")
    if avg_relevance < 0.55:  improvements.append("Question focus — ensure each answer directly addresses the question before expanding.")
    if avg_discourse < 0.35:  improvements.append("Answer structure — adopt a consistent format: define → explain → example → conclude.")
    if avg_length    < 0.40:  improvements.append("Answer depth — most answers were too brief; aim for 40-70 words covering key sub-points.")
    if grammar_summary["grammar_tier"] in ("moderate", "heavy"):
        improvements.append(
            f"Spoken delivery — reduce filler words (rate: {overall_filler_rate}% of words). "
            "Practise pausing instead of filling."
        )

    # Fallbacks if nothing qualifies
    if not strengths:
        strengths.append("You completed the session and engaged with each question.")
    if not improvements:
        improvements.append("Continue practising to reinforce and refine your already solid foundations.")

    # Summary paragraph (3-4 sentences)
    tier_phrases = {
        "strong":  ("excellent", "highly competitive"),
        "good":    ("solid", "promising"),
        "partial": ("developing", "needs further work"),
        "weak":    ("foundational", "requires significant preparation"),
    }
    adj, readiness = tier_phrases.get(session_tier, ("mixed", "variable"))

    sent1 = (
        f"Across {n} question{'s' if n != 1 else ''}, your overall session score was "
        f"{avg_score:.0%}, placing you in the {session_tier!r} tier."
    )
    sent2 = (
        f"Your strongest dimension was "
        f"{'semantic accuracy' if avg_semantic == max(avg_semantic, avg_kw, avg_relevance) else 'question relevance' if avg_relevance == max(avg_semantic, avg_kw, avg_relevance) else 'keyword coverage'}, "
        f"while "
        f"{'answer structure' if avg_discourse < min(avg_semantic, avg_kw) else 'keyword recall' if avg_kw < avg_semantic else 'conceptual depth'} "
        f"was the area that dragged your score down most."
    )
    sent3_trend = {
        "improving":  "Your scores improved as the session progressed, suggesting you warm up well — try to bring that energy earlier.",
        "declining":  "Your scores trended downward across the session, which may indicate fatigue or unfamiliar later questions — pacing and preparation are key.",
        "consistent": "Your performance was broadly consistent across questions, which shows stable but uniform preparation.",
    }[trend_direction]
    sent4 = (
        f"To move to the next tier, prioritise: "
        + "; ".join(improvements[:2]) + "."
        if improvements else
        "Continue building on your strengths with increasingly complex questions and timed practice."
    )

    summary_paragraph = " ".join([sent1, sent2, sent3_trend, sent4])

    return {
        "session_score":       round(avg_score, 4),
        "score_tier":          session_tier,
        "score_trend":         scores,
        "trend_direction":     trend_direction,
        "strengths":           strengths[:4],
        "improvement_areas":   improvements[:4],
        "grammar_summary":     grammar_summary,
        "summary_paragraph":   summary_paragraph,
        "per_question_scores": per_q_scores,
        "metric_averages":     {
            "semantic_similarity": round(avg_semantic,  3),
            "keyword_coverage":    round(avg_kw,        3),
            "question_relevance":  round(avg_relevance, 3),
            "discourse_quality":   round(avg_discourse, 3),
            "lexical_diversity":   round(avg_lexical,   3),
            "length_score":        round(avg_length,    3),
        },
    }
