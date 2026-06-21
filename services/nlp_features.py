"""
nlp_features.py
================
Enhanced NLP feature extractor for mock-interview answer evaluation.

Produces 16 features (vs 9 in evaluation.py) including:
  - Question-relevance score  (context awareness: candidate vs *question*, not just expected answer)
  - Lexical diversity          (vocabulary richness)
  - Discourse structure score  (connectives, examples, causals)
  - Sentence-level stats       (count, avg length)
  - Answer length ratio        (proportional completeness vs expected answer)

Drop-in: extract_features_enhanced() returns the same (np.ndarray, meta_dict) signature
as the original extract_features() so evaluation.py can swap it in with one line change.
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# ── stopwords (expanded vs original) ──────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "that", "this", "it", "as", "by", "be", "at",
    "from", "can", "into", "about", "what", "which", "who", "whom", "when",
    "where", "why", "how",
    # expanded
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "its", "will", "would", "should", "may", "might",
    "could", "shall", "do", "did", "does", "have", "has", "had", "been",
    "being", "also", "very", "just", "but", "if", "so", "than", "then",
    "not", "no", "nor", "yet", "both", "either", "neither", "once", "here",
    "there", "when", "while", "am",
}

WEAK_PATTERNS = {
    "i don't know", "dont know", "not sure", "no idea",
    "can't say", "cannot say", "i have no idea", "i am not sure",
}

# Discourse markers that indicate structured reasoning
DISCOURSE_MARKERS = {
    # sequencing
    "first", "firstly", "second", "secondly", "third", "thirdly",
    "finally", "lastly", "next", "then",
    # causation
    "because", "therefore", "thus", "hence", "consequently", "as a result",
    # elaboration
    "for example", "for instance", "such as", "namely", "specifically",
    "in other words", "that is",
    # contrast
    "however", "although", "despite", "whereas", "on the other hand",
    # addition
    "moreover", "furthermore", "additionally", "in addition", "also",
    # conclusion
    "in conclusion", "to summarise", "to summarize", "overall", "in summary",
}


# ── lazy model loader (same pattern as evaluation.py) ─────────────────────────
@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ── text utilities ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return [w for w in normalize(text).split() if len(w) > 2 and w not in STOPWORDS]


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter — good enough for spoken/transcribed text."""
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in raw if s.strip()]


# ── individual feature functions ───────────────────────────────────────────────
def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    return model.encode(texts, normalize_embeddings=True)


def semantic_score(candidate: str, reference: str) -> float:
    """Cosine similarity between two texts via sentence embeddings."""
    if not candidate.strip() or not reference.strip():
        return 0.0
    emb = _embed([candidate, reference])
    return float(np.clip(cosine_similarity([emb[0]], [emb[1]])[0][0], 0.0, 1.0))


def semantic_scores_batch(candidate: str, reference_a: str, reference_b: str) -> Tuple[float, float]:
    """
    Compute two cosine similarities in a single encode call.
    Returns (sim_a, sim_b).
    Used to get both answer-similarity AND question-relevance without 2x encoding.
    """
    if not candidate.strip():
        return 0.0, 0.0
    texts = [candidate, reference_a or "", reference_b or ""]
    emb = _embed(texts)
    sim_a = float(np.clip(cosine_similarity([emb[0]], [emb[1]])[0][0], 0.0, 1.0)) if reference_a.strip() else 0.0
    sim_b = float(np.clip(cosine_similarity([emb[0]], [emb[2]])[0][0], 0.0, 1.0)) if reference_b.strip() else 0.0
    return sim_a, sim_b


def _stem_simple(word: str) -> str:
    """Very simple suffix stripping for fuzzy comparison."""
    for suffix in ("tion", "sion", "ing", "ment", "ness", "ity", "ies", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _fuzzy_kw_match(kw_norm: str, candidate_norm: str, candidate_tokens: set) -> bool:
    """
    Check if a keyword is present in the candidate text using multiple strategies:
    1. Exact substring match (with hyphen normalization)
    2. Stem-based token overlap (handles plurals, tense changes)
    3. Fuzzy ratio for multi-word keywords (handles minor STT garbling)
    4. Abbreviation/acronym detection
    """
    # Strategy 1: exact substring (also check with hyphens collapsed)
    if kw_norm in candidate_norm:
        return True
    kw_dehyphen = kw_norm.replace("-", " ")
    cand_dehyphen = candidate_norm.replace("-", " ")
    if kw_dehyphen in cand_dehyphen:
        return True

    kw_tokens = [t for t in kw_norm.split() if len(t) > 2 and t not in STOPWORDS]

    # Strategy 2: stem-based token containment
    # If all meaningful tokens of the keyword (stemmed) appear in candidate (stemmed)
    if kw_tokens:
        cand_stems = {_stem_simple(t) for t in candidate_tokens}
        kw_stems = {_stem_simple(t) for t in kw_tokens}
        if kw_stems and kw_stems.issubset(cand_stems):
            return True
        # Allow 1 missing stem for multi-word keywords (>=3 tokens)
        if len(kw_stems) >= 3:
            hits = sum(1 for s in kw_stems if s in cand_stems)
            if hits >= len(kw_stems) - 1:
                return True

    # Strategy 3: fuzzy ratio for multi-word keywords
    # Slide a window over candidate and check similarity
    if len(kw_tokens) >= 2:
        kw_len = len(kw_norm)
        # Check all substrings of similar length
        for i in range(max(0, len(candidate_norm) - kw_len - 5)):
            window = candidate_norm[i:i + kw_len + 5]
            ratio = SequenceMatcher(None, kw_norm, window[:kw_len + 2]).ratio()
            if ratio >= 0.82:
                return True

    # Strategy 4: abbreviation/acronym check
    # e.g., "PSI" matches "population stability index"
    if len(kw_norm) <= 5 and kw_norm.replace(" ", "").isalpha():
        # kw might be an acronym — check if candidate has the expanded form
        # or kw might be expanded — check if candidate has the acronym
        pass  # handled by stem match above for expansions
    elif len(kw_tokens) >= 3:
        # Build acronym from keyword and check if it appears in candidate
        acronym = "".join(t[0] for t in kw_tokens if t)
        if len(acronym) >= 2 and acronym in candidate_norm.split():
            return True

    return False


def keyword_score(candidate: str, expected_keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    if not expected_keywords:
        return 0.0, [], []
    candidate_norm = normalize(candidate)
    candidate_tokens = set(tokenize(candidate))
    matched, missing = [], []
    for kw in expected_keywords:
        kw_norm = normalize(kw)
        if kw_norm and _fuzzy_kw_match(kw_norm, candidate_norm, candidate_tokens):
            matched.append(kw)
        else:
            missing.append(kw)
    score = len(matched) / max(len(expected_keywords), 1)
    return score, matched, missing


def overlap_score(candidate: str, truth: str) -> float:
    c_words = set(tokenize(candidate))
    t_words = set(tokenize(truth))
    if not c_words or not t_words:
        return 0.0
    return len(c_words & t_words) / len(t_words)


def length_score(candidate: str, min_words: int = 6, ideal_words: int = 25) -> float:
    """
    Ideal raised to 25 tokens (was 18) — better calibrated for domain questions.
    """
    words = tokenize(candidate)
    n = len(words)
    if n == 0:
        return 0.0
    if n < min_words:
        return max(0.35, n / max(min_words, 1))
    if n >= ideal_words:
        return 1.0
    return n / ideal_words


def weak_answer_penalty(candidate: str) -> float:
    norm = normalize(candidate)
    if not norm:
        return 0.25
    if norm in WEAK_PATTERNS or any(p in norm for p in WEAK_PATTERNS):
        return 0.20
    if len(tokenize(candidate)) <= 2:
        return 0.10
    return 0.0


def lexical_diversity(candidate: str) -> float:
    """Type-token ratio on meaningful tokens. Penalises repetitive answers."""
    tokens = tokenize(candidate)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def discourse_score(candidate: str) -> float:
    """
    Fraction of discourse marker categories present (not raw count).
    Caps at 1.0 so a single very marker-heavy answer doesn't dominate.
    """
    norm = normalize(candidate)
    # Group markers by function so using 5 sequencers counts as 1 category, not 5
    category_hits = {
        "sequence": any(m in norm for m in ["first", "second", "third", "finally", "next", "then", "lastly"]),
        "cause":    any(m in norm for m in ["because", "therefore", "thus", "hence", "consequently", "as a result"]),
        "example":  any(m in norm for m in ["for example", "for instance", "such as", "namely"]),
        "contrast": any(m in norm for m in ["however", "although", "despite", "whereas", "on the other hand"]),
        "addition": any(m in norm for m in ["moreover", "furthermore", "additionally", "in addition"]),
        "summary":  any(m in norm for m in ["in conclusion", "to summarise", "to summarize", "overall"]),
    }
    hits = sum(category_hits.values())
    return min(1.0, hits / 3)  # 3+ categories → full score


def sentence_features(candidate: str) -> Tuple[float, float]:
    """Returns (sentence_count_norm, avg_sentence_length_norm)."""
    sentences = split_sentences(candidate)
    n_sent = len(sentences)
    if n_sent == 0:
        return 0.0, 0.0
    sent_count_norm = min(1.0, n_sent / 6)          # 6+ sentences → full score
    avg_len = np.mean([len(s.split()) for s in sentences])
    avg_len_norm = min(1.0, avg_len / 20)            # avg 20 words/sentence → full score
    return float(sent_count_norm), float(avg_len_norm)


def answer_length_ratio(candidate: str, truth: str) -> float:
    """
    How long the candidate answer is relative to the expected answer.
    Capped at 1.0 (being longer than expected is not better).
    """
    c_tok = len(tokenize(candidate))
    t_tok = len(tokenize(truth))
    if t_tok == 0:
        return 0.0
    return min(1.0, c_tok / t_tok)


# ── main interface ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # original 9
    "semantic_score",         # 0  candidate vs expected answer
    "keyword_score",          # 1
    "overlap_score",          # 2
    "length_score",           # 3
    "weak_penalty",           # 4
    "candidate_token_count",  # 5
    "truth_token_count",      # 6
    "expected_keyword_count", # 7
    "matched_keyword_count",  # 8
    # new 7
    "question_relevance",     # 9  candidate vs question text  ← context awareness
    "lexical_diversity",      # 10 vocabulary richness
    "discourse_score",        # 11 structured reasoning markers
    "sentence_count_norm",    # 12
    "avg_sentence_len_norm",  # 13
    "answer_length_ratio",    # 14
    "kw_density",             # 15 matched_kws / candidate_token_count
]

N_FEATURES = len(FEATURE_NAMES)   # 16


def extract_features_enhanced(
    candidate: str,
    question_data: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Same signature as evaluation.py::extract_features().
    Returns (feature_vector: np.ndarray[16], meta: dict).

    question_data must contain at minimum:
        "question"  — the interview question text
        "answer"    — the expected / model answer

    Optional:
        "expected_keywords" — list[str]; auto-extracted from answer if absent.
    """
    truth = question_data.get("answer", "")
    question_text = question_data.get("question", "")

    # Auto-extract keywords if missing (better than first-10 tokens: use all unique tokens)
    expected_keywords = question_data.get("expected_keywords", [])
    if not expected_keywords:
        truth_tokens = tokenize(truth)
        seen: list = []
        for w in truth_tokens:
            if w not in seen:
                seen.append(w)
        expected_keywords = seen[:12]   # slightly more than original 10

    # ── embedding-based scores (single encode call for efficiency) ──
    s_score_ans, s_score_q = semantic_scores_batch(candidate, truth, question_text)

    # ── traditional scores ──
    k_score, matched, missing = keyword_score(candidate, expected_keywords)
    o_score = overlap_score(candidate, truth)
    l_score = length_score(candidate)
    penalty = weak_answer_penalty(candidate)

    # ── new NLP features ──
    lex_div = lexical_diversity(candidate)
    disc    = discourse_score(candidate)
    s_cnt, avg_sl = sentence_features(candidate)
    al_ratio = answer_length_ratio(candidate, truth)

    cand_tok_count = len(tokenize(candidate))
    kw_density = (len(matched) / cand_tok_count) if cand_tok_count > 0 else 0.0

    features = np.array([
        s_score_ans,            # 0
        k_score,                # 1
        o_score,                # 2
        l_score,                # 3
        penalty,                # 4
        cand_tok_count,         # 5
        len(tokenize(truth)),   # 6
        len(expected_keywords), # 7
        len(matched),           # 8
        s_score_q,              # 9  ← context awareness
        lex_div,                # 10
        disc,                   # 11
        s_cnt,                  # 12
        avg_sl,                 # 13
        al_ratio,               # 14
        kw_density,             # 15
    ], dtype=float)

    meta = {
        "semantic_score":       round(s_score_ans, 4),
        "question_relevance":   round(s_score_q,   4),
        "keyword_score":        round(k_score,      4),
        "overlap_score":        round(o_score,      4),
        "length_score":         round(l_score,      4),
        "penalty":              round(penalty,      4),
        "lexical_diversity":    round(lex_div,      4),
        "discourse_score":      round(disc,         4),
        "sentence_count_norm":  round(s_cnt,        4),
        "avg_sentence_len":     round(avg_sl,       4),
        "answer_length_ratio":  round(al_ratio,     4),
        "matched_keywords":     matched,
        "missing_keywords":     missing,
        "truth":                truth,
    }

    return features, meta


# ── heuristic fallback (mirrors evaluation.py, updated weights) ────────────────
def soft_rule_score_enhanced(
    s_score: float,
    q_relevance: float,
    k_score: float,
    overlap: float,
    l_score: float,
    penalty: float,
    disc: float,
    lex_div: float,
) -> float:
    """
    Heuristic score using the full feature set.
    Weights tuned to match the score tier boundaries in the training data.
    """
    blended_kw = max(k_score, overlap)
    score = (
        0.40 * s_score
        + 0.10 * q_relevance         # context: is the answer actually about the question?
        + 0.30 * blended_kw           # keyword/overlap heavily weighted — forces specificity
        + 0.08 * l_score
        + 0.06 * disc
        + 0.06 * lex_div
        - min(penalty, 0.15)
    )

    # Penalize high semantic but low keyword (topic-adjacent waffle)
    if s_score >= 0.60 and blended_kw < 0.20:
        score *= 0.75  # can't get a good score by just talking about the topic

    # Floor boosts ONLY for genuinely strong answers (high semantic AND keywords)
    if s_score >= 0.82 and blended_kw >= 0.50 and l_score >= 0.50:
        score = max(score, 0.65)
    if s_score >= 0.88 and blended_kw >= 0.60 and l_score >= 0.55:
        score = max(score, 0.78)

    return float(np.clip(score, 0.0, 1.0))
