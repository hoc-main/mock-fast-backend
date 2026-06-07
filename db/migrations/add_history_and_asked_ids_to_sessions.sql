-- Migration to add conversation_history and asked_question_ids to interview_interviewsession table
ALTER TABLE interview_interviewsession 
ADD COLUMN IF NOT EXISTS conversation_history JSONB DEFAULT '[]';

ALTER TABLE interview_interviewsession 
ADD COLUMN IF NOT EXISTS asked_question_ids JSONB DEFAULT '[]';
