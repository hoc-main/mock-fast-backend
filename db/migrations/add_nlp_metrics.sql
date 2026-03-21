-- Migration: Add new NLP metrics to interview_interviewanswer table
-- Description: Adds question_relevance, lexical_diversity, discourse_score, and penalty columns.

ALTER TABLE interview_interviewanswer
ADD COLUMN IF NOT EXISTS question_relevance DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS lexical_diversity DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS discourse_score DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS penalty DOUBLE PRECISION DEFAULT 0.0;
