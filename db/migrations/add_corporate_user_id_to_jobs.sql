-- Migration to add corporate_user_id to jobs table and set default for existing records

-- Add the column with a default value of 31 and set up the foreign key reference
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS corporate_user_id INT DEFAULT 31 REFERENCES users(user_id);

-- Ensure all existing records are updated to 31 just in case the default didn't apply to existing rows in some SQL dialects
UPDATE jobs 
SET corporate_user_id = 31 
WHERE corporate_user_id IS NULL;
