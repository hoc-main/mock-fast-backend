"""
insert_questions.py
===================
Insert questions JSON into PostgreSQL using SQLAlchemy (raw SQL)

- Handles duplicates in Python (no ON CONFLICT needed)
- Safe for current schema

Usage:
    python insert_questions.py --input questions.json --db_url postgresql+psycopg2://user:pass@localhost:5432/db
"""

import argparse
import json
from sqlalchemy import create_engine, text


INSERT_QUERY = text("""
INSERT INTO interview_question (
    module_id,
    topic,
    question_text,
    expected_answer,
    "order",
    created_at
)
VALUES (
    :module_id,
    :topic,
    :question_text,
    :expected_answer,
    :order,
    NOW()
);
""")


def deduplicate(questions):
    seen = set()
    filtered = []

    for q in questions:
        key = (q["module_id"], q["question_text"].strip().lower())

        if key not in seen:
            seen.add(key)
            filtered.append(q)

    return filtered


def insert_questions(engine, questions):
    # remove duplicates BEFORE insert
    questions = deduplicate(questions)

    data = [
        {
            "module_id": q["module_id"],
            "topic": q.get("topic"),
            "question_text": q["question_text"],
            "expected_answer": q["expected_answer"],
            "order": q.get("order", 0),
        }
        for q in questions
    ]

    with engine.begin() as conn:
        conn.execute(INSERT_QUERY, data)

    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Insert questions using SQLAlchemy raw queries")

    parser.add_argument("--input", "-i", required=True, help="questions.json file")
    parser.add_argument("--db_url", "-d", required=True, help="SQLAlchemy DB URL")

    args = parser.parse_args()

    # Load JSON
    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        print("❌ ERROR: JSON must be a list")
        return

    engine = create_engine(args.db_url)

    try:
        inserted = insert_questions(engine, questions)
        print(f"✅ Inserted {inserted} unique questions")

    except Exception as e:
        print("❌ ERROR:", str(e))


if __name__ == "__main__":
    main()