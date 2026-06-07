"""
import_csv_questions.py
======================
Script to parse "AIML INTERVIEW QUES. WITH ROLES - Sheet1.csv" and import them into the DB.
Matches questions to existing modules by comparing the CSV's "Question" or keywords to module names.
If no matching module is found, it finds or creates a default module under a general subdomain.

Usage:
    python import_csv_questions.py --db_url postgresql+psycopg2://user:pass@localhost:5432/db
"""

import os
import csv
import json
import argparse
import re
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def clean_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def parse_list(cell):
    if not cell:
        return []
    # split by comma, strip whitespace and quotes
    items = []
    for item in re.split(r',', cell):
        val = item.strip().strip('"').strip("'").strip()
        if val:
            items.append(val)
    return items

def main():
    parser = argparse.ArgumentParser(description="Feed AIML CSV questions into PostgreSQL database")
    parser.add_argument("--csv", default="AIML INTERVIEW QUES. WITH ROLES - Sheet1.csv", help="Path to the CSV file")
    parser.add_argument("--module_id", type=int, required=True, help="Module ID to assign all questions to")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable is not defined in the .env file")
        return

    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Check in scripts relative folder
        script_dir_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(script_dir_path):
            csv_path = script_dir_path
        else:
            print(f"❌ File not found: {csv_path}")
            return

    # Initialize SQLAlchemy connection
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load existing modules
    print("Fetching modules from database...")
    modules_res = session.execute(text("SELECT id, module_name, slug, subdomain_id, companies, job_roles FROM modules")).fetchall()
    
    # We will parse CSV rows
    print(f"Reading CSV file from {csv_path}...")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Found {len(rows)} questions in CSV.")

    inserted_count = 0
    updated_module_count = 0

    for idx, row in enumerate(rows):
        q_text = row.get("Question", "").strip()
        companies = parse_list(row.get("Company", ""))
        roles = parse_list(row.get("Role", ""))
        expected_answer = row.get("ans_ver1", "").strip()

        if not q_text or not expected_answer:
            print(f"⚠️ Row {idx + 2} skipped due to missing Question or Answer text.")
            continue

        # Assign to the explicit module ID provided in arguments
        matched_module_id = args.module_id

        # Verify that this module actually exists in database
        module_exists = any(m[0] == matched_module_id for m in modules_res)
        if not module_exists:
            print(f"❌ ERROR: The specified --module_id {matched_module_id} does not exist in the database.")
            return

        # Check if question already exists in the database for this module to avoid duplicates
        existing = session.execute(
            text("SELECT id FROM interview_question WHERE module_id = :m_id AND LOWER(TRIM(question_text)) = :q_text"),
            {"m_id": matched_module_id, "q_text": q_text.lower()}
        ).fetchone()

        if existing:
            print(f"⏭️ Question already exists: '{q_text[:40]}...'")
        else:
            # Insert question
            session.execute(
                text("""
                    INSERT INTO interview_question (module_id, topic, question_text, expected_answer, "order", created_at)
                    VALUES (:m_id, :topic, :q_text, :expected, :order, NOW())
                """),
                {
                    "m_id": matched_module_id,
                    "topic": roles[0] if roles else "AI/ML",
                    "q_text": q_text,
                    "expected": expected_answer,
                    "order": idx
                }
            )
            inserted_count += 1

        # Also update module's companies & job_roles if they are missing
        for m in modules_res:
            m_id, m_name, m_slug, sd_id, m_companies, m_job_roles = m
            if m_id == matched_module_id:
                # Load current JSONs safely
                try:
                    curr_companies = m_companies if isinstance(m_companies, list) else json.loads(m_companies or "[]")
                except:
                    curr_companies = []
                try:
                    curr_roles = m_job_roles if isinstance(m_job_roles, list) else json.loads(m_job_roles or "[]")
                except:
                    curr_roles = []

                updated = False
                for c in companies:
                    if c not in curr_companies:
                        curr_companies.append(c)
                        updated = True
                for r in roles:
                    if r not in curr_roles:
                        curr_roles.append(r)
                        updated = True

                if updated:
                    session.execute(
                        text("UPDATE modules SET companies = :companies, job_roles = :roles WHERE id = :m_id"),
                        {
                            "companies": json.dumps(curr_companies),
                            "roles": json.dumps(curr_roles),
                            "m_id": m_id
                        }
                    )
                    updated_module_count += 1
                break

    session.commit()
    session.close()

    print(f"🎉 Complete! Inserted {inserted_count} new questions.")
    print(f"💼 Updated {updated_module_count} modules with new companies/job roles metadata.")

if __name__ == "__main__":
    main()
