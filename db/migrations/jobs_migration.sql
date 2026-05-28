-- Migration to create jobs and job_applications tables (PostgreSQL Compatible)

-- 1. Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    corporate_user_id INT, -- Can reference users table if needed
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    work_mode VARCHAR(50) NOT NULL,
    start_date VARCHAR(50) NOT NULL,
    duration VARCHAR(50) NOT NULL,
    stipend VARCHAR(100) NOT NULL,
    apply_by VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    schedule VARCHAR(50) NOT NULL,
    posted VARCHAR(50) NOT NULL,
    skills JSONB NOT NULL DEFAULT '[]', 
    description TEXT NOT NULL,
    responsibilities JSONB NOT NULL DEFAULT '[]', 
    additional_note TEXT,
    who_can_apply JSONB NOT NULL DEFAULT '[]', 
    other_requirements JSONB NOT NULL DEFAULT '[]', 
    perks JSONB NOT NULL DEFAULT '[]', 
    company_about TEXT NOT NULL,
    opps_posted INT DEFAULT 0,
    candidates_hired INT DEFAULT 0,
    logo VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Create job_applications table
CREATE TABLE IF NOT EXISTS job_applications (
    id SERIAL PRIMARY KEY,
    job_id INT NOT NULL REFERENCES jobs(id),
    user_id INT NOT NULL, -- Assuming foreign key to users table if needed
    phone_number VARCHAR(15) NOT NULL,
    resume_url VARCHAR(255) NOT NULL,
    certificates_url VARCHAR(255),
    cover_letter TEXT,
    status VARCHAR(20) DEFAULT 'pending', -- PostgreSQL doesn't use ENUM like MySQL by default without type creation
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX_job_id INT, -- PostgreSQL uses separate CREATE INDEX commands
    INDEX_user_id INT
);

CREATE INDEX IF NOT EXISTS idx_job_app_job_id ON job_applications(job_id);
CREATE INDEX IF NOT EXISTS idx_job_app_user_id ON job_applications(user_id);

-- 3. Update existing modules table
ALTER TABLE modules ADD COLUMN IF NOT EXISTS companies JSONB NOT NULL DEFAULT '[]';

-- 4. Initial data for jobs
INSERT INTO jobs (
    title, company, work_mode, start_date, duration, stipend, apply_by, type, schedule, posted, 
    skills, description, responsibilities, additional_note, who_can_apply, other_requirements, perks, 
    company_about, opps_posted, candidates_hired
) VALUES (
    'Subject Matter Expert (SME) - AI/ML', 'House Of Couton Private Limited', 'Work from home', 'Immediately', '3 Months', '₹12,000 - ₹35,000/month', '23 Mar 2026', 'Internship', 'Part time', '3 weeks ago',
    '["Artificial Intelligence", "Data Science", "Data Structures", "Deep Learning", "Machine Learning", "Natural Language Processing (NLP)", "Neural Networks", "Python"]',
    'We are looking for an AI/ML Subject Matter Expert to join our team and help us build cutting-edge solutions.',
    '["Review and analyze research papers, ensuring quality checks and providing improvement suggestions", "Work on coding and conducting research in AI/ML, assisting as needed", "Define and refine project scope, ensuring clarity and feasibility", "Write reports and code, co-authoring as per requirement"]',
    'This involves a large portion of report preparation and academic paper preparation.',
    '["Are available for the work from home job/internship", "Can start the work from home job/internship between 7th May''24 and 11th Jun''24", "Are available for duration of 3 months", "Have relevant skills and interests"]',
    '["Academic/report writing requirement", "Published papers preference"]',
    '["Certificate", "Letter of recommendation", "Flexible work hours"]',
    'House Of Couton is a tech-driven company focused on innovative solutions in AI and Machine Learning.',
    15, 120
),
(
    'Frontend Developer (React)', 'AI Hire Studio', 'Remote', 'Immediately', '6 Months', '₹20,000 - ₹40,000/month', '15 Jun 2026', 'Internship', 'Full time', '1 week ago',
    '["React.js", "Next.js", "Tailwind CSS", "TypeScript", "Framer Motion"]',
    'Join our frontend team to build beautiful and interactive user interfaces for our AI-powered platforms.',
    '["Develop new user-facing features using React.js", "Build reusable components and front-end libraries for future use", "Translate designs and wireframes into high quality code", "Optimize components for maximum performance across a vast array of web-capable devices and browsers"]',
    NULL,
    '["Are available for the remote internship", "Can start between 1st Jun''24 and 1st Jul''24", "Are available for 6 months", "Strong proficiency in JavaScript, including DOM manipulation and the JavaScript object model"]',
    '["Experience with popular React.js workflows (such as Redux or Context API)", "Familiarity with newer specifications of EcmaScript"]',
    '["Certificate", "Flexible work hours", "Pre-placement offer (PPO)"]',
    'AI Hire Studio is a cutting edge application designed to auto evaluate candidates steadiness through auto interviews.',
    50, 450
);
