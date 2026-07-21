-- Migration: Create corporate_otps table
-- Description: Table to store temporary OTPs for corporate login

CREATE TABLE IF NOT EXISTS corporate_otps (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    otp VARCHAR(10) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corporate_otps_email ON corporate_otps(email);
