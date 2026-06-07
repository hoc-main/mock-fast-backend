-- Migration: Add job_roles column to modules table
ALTER TABLE modules ADD COLUMN IF NOT EXISTS job_roles JSONB DEFAULT '[]'::jsonb;
