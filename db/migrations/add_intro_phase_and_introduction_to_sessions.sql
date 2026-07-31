ALTER TABLE interview_interviewsession ADD COLUMN IF NOT EXISTS intro_phase BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE interview_interviewsession ADD COLUMN IF NOT EXISTS introduction TEXT DEFAULT '';
