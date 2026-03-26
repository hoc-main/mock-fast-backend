# Training Dataset Schema — Mock Interview Evaluator
# Version 2.0

## Overview

Each dataset file is a JSON array of **training samples**.
Every sample pairs a candidate answer with its full question context and
a set of explicit numeric labels — one per evaluation dimension.

The scorer learns from these labels independently, which lets you later
adjust per-dimension weights without retraining.

---

## Top-level sample structure

```json
{
  "sample_id":          "upsc_gs_polity_001_v3",
  "schema_version":     "2.0",
  "module":             "upsc_gs",
  "split":              "train",
  "question_data":      { ... },
  "candidate":          "...",
  "stt_realistic":      true,
  "labels":             { ... },
  "score":              0.74,
  "scoring_rationale":  "..."
}
```

| Field              | Type    | Required | Notes |
|--------------------|---------|----------|-------|
| sample_id          | string  | ✓        | Unique. Convention: `{module}_{topic}_{qid}_{variant}` |
| schema_version     | string  | ✓        | Always "2.0" for this format |
| module             | string  | ✓        | Matches Module.slug in DB |
| split              | string  | ✓        | "train" \| "val" \| "test". Pre-assigned, never random. |
| question_data      | object  | ✓        | See below |
| candidate          | string  | ✓        | The answer being evaluated |
| stt_realistic      | boolean | ✓        | true = contains spoken artifacts (filler, repetition, transcription noise) |
| labels             | object  | ✓        | Per-dimension numeric scores. See below. |
| score              | float   | ✓        | Final holistic score 0.0–1.0. This is the regression target. |
| scoring_rationale  | string  | ✓        | 1–2 sentences explaining why score was assigned. For audit + future LLM labelling. |

---

## question_data structure

```json
{
  "id":                 "polity_q001",
  "module":             "upsc_gs",
  "topic":              "polity",
  "subtopic":           "constitution_basics",
  "difficulty":         2,
  "question":           "If you had to explain to a school student what the Constitution actually does, how would you put it?",
  "answer_variants":    [ "...", "...", "..." ],
  "primary_answer":     "...",
  "expected_keywords":  {
    "critical":         ["constitution", "supreme law", "government structure"],
    "supporting":       ["rights", "powers", "citizens", "framers"],
    "bonus":            ["preamble", "fundamental rights", "directive principles"]
  },
  "scoring_weights":    {
    "semantic":         0.55,
    "question_relevance": 0.15,
    "keyword":          0.20,
    "length":           0.05,
    "discourse":        0.05
  },
  "ideal_length_words": 40,
  "domain_tags":        ["civics", "constitutional_law"]
}
```

| Field              | Type         | Notes |
|--------------------|--------------|-------|
| id                 | string       | Stable across dataset versions |
| module             | string       | |
| topic              | string       | Broad topic (polity, economy, …) |
| subtopic           | string       | Narrower concept within the topic |
| difficulty         | int 1–5      | 1=recall, 3=application, 5=synthesis+evaluation |
| question           | string       | The verbatim interview question |
| answer_variants    | list[string] | 3–5 paraphrases of a correct answer. Model learns similarity to ANY of these, not just one. |
| primary_answer     | string       | The clearest / most complete variant. Used for keyword extraction. |
| expected_keywords  | object       | Three tiers (see below) |
| scoring_weights    | object       | Per-dimension weights for this question. Harder questions weight semantic more. |
| ideal_length_words | int          | Domain-calibrated ideal answer length in words (not tokens). |
| domain_tags        | list[string] | Free-form tags for analytics and filtering |

### Keyword tiers

- **critical** — concepts the answer MUST mention to score above 0.5. Missing all critical keywords → score capped at 0.55.
- **supporting** — flesh out the answer; each adds ~0.05 to keyword_score.
- **bonus** — advanced points that push score from good → strong.

---

## labels structure

These are the direct training signals for each dimension.
The model learns all of them; `score` is the weighted aggregate.

```json
{
  "semantic_similarity":   0.82,
  "question_relevance":    0.91,
  "keyword_coverage":      0.67,
  "length_appropriateness": 0.75,
  "discourse_quality":     0.50,
  "lexical_diversity":     0.68,
  "factual_accuracy":      0.90,
  "answer_completeness":   0.70
}
```

| Label                  | Range  | How to assign |
|------------------------|--------|---------------|
| semantic_similarity    | 0–1    | Cosine similarity of answer to primary_answer embedding. Compute programmatically. |
| question_relevance     | 0–1    | Does the answer address THIS question or just the topic? Human or LLM label. |
| keyword_coverage       | 0–1    | Fraction of critical+supporting keywords present. Compute programmatically. |
| length_appropriateness | 0–1    | How close is length to ideal_length_words? Compute programmatically. |
| discourse_quality      | 0–1    | Does the answer have structure (definition → explanation → example)? Human/LLM label. |
| lexical_diversity      | 0–1    | Type-token ratio of meaningful tokens. Compute programmatically. |
| factual_accuracy       | 0–1    | Is the content correct (no wrong facts)? Human label. Use 1.0 if not labelled. |
| answer_completeness    | 0–1    | Covers all required sub-points? Human/LLM label. |

---

## Candidate answer quality types

Every question should have at least one sample of each type:

| type                  | Description | Example score range |
|-----------------------|-------------|---------------------|
| strong_complete       | Correct, structured, covers all key points | 0.88–0.98 |
| strong_concise        | Correct and complete but shorter than ideal | 0.78–0.88 |
| good_partial_keywords | Right direction, misses 1–2 critical keywords | 0.62–0.76 |
| good_verbose          | Correct but rambling, repetitive | 0.60–0.72 |
| partial_correct       | Correct facts, but incomplete | 0.42–0.58 |
| partial_off_topic     | Answers the topic area but not this specific question | 0.28–0.42 |
| weak_vague            | Generic, hand-wavy, no specifics | 0.15–0.28 |
| weak_wrong            | Factually incorrect | 0.05–0.18 |
| weak_refusal          | "I don't know", "not sure" | 0.05–0.14 |
| stt_artifact_good     | Correct answer with spoken filler words and transcription noise | 0.60–0.78 |
| stt_artifact_partial  | Partial answer with spoken artifacts | 0.35–0.55 |

Minimum recommended: 6 samples per question (strong_complete, good_partial_keywords,
partial_correct, partial_off_topic, weak_vague, stt_artifact_good).

---

## STT-realistic text conventions

When `stt_realistic: true`, the candidate text should include:
- Filler words: "um", "uh", "you know", "like", "basically", "right"
- Self-corrections: "the constitution is — well, it defines the structure"
- Repetition: "it is the it is the supreme law"
- Transcription plausible errors: lowercase throughout, missing punctuation
- Run-on sentences joined by "and" or "so"

Example:
```
"um the constitution is basically uh the highest law in the country right 
and it like defines how the government works and protects citizens and um 
yeah it also distributes powers between different branches"
```

---

## Score computation guide

The holistic `score` should be computed as:

```
score = (
    w_semantic   * labels.semantic_similarity
  + w_relevance  * labels.question_relevance
  + w_keyword    * labels.keyword_coverage
  + w_length     * labels.length_appropriateness
  + w_discourse  * labels.discourse_quality
  + w_lexical    * labels.lexical_diversity
  + w_factual    * labels.factual_accuracy
  + w_complete   * labels.answer_completeness
  - penalty_for_refusal_or_empty
)
```

Use `question_data.scoring_weights` for per-question weights.
Global fallback weights: semantic=0.45, question_relevance=0.15, keyword=0.20,
length=0.07, discourse=0.06, lexical=0.04, factual=0.02, completeness=0.01.

Score is clamped to [0.0, 1.0].

---

## File naming convention

```
data/
  {module}_training.json     ← training split (70%)
  {module}_val.json          ← validation split (15%)
  {module}_test.json         ← held-out test split (15%), never used during training
  {module}_questions.json    ← question bank (no candidates)
```

---

## Minimum viable dataset sizes

| Use case                      | Min samples | Recommended |
|-------------------------------|-------------|-------------|
| Single module, rough baseline | 200         | 600         |
| Single module, production     | 600         | 1500        |
| Multi-topic, generalisation   | 1500        | 4000+       |
| Fine-tune per topic           | 80/topic    | 200/topic   |
