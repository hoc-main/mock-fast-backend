"""
SQLAlchemy models that mirror Django's existing DB tables.
- managed=False tables (users, domains, subdomains, modules) → reflected as read-only
- Django-managed tables (questions, interview_sessions, interview_answers) → full access
- Table names match Django's db_table / auto-generated names exactly
- No migrations run from here — Django owns schema changes
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, func,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .database import Base


# ── Read-only mirrors of Django managed=False tables ─────────────────────────

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    first_name: Mapped[str] = mapped_column(String(255))
    last_name: Mapped[str] = mapped_column(String(255))

    sessions: Mapped[List["InterviewSession"]] = relationship(back_populates="user")


class Domain(Base):
    __tablename__ = "domains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(255), unique=True)

    subdomains: Mapped[List["Subdomain"]] = relationship(back_populates="domain")


class Subdomain(Base):
    __tablename__ = "subdomains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"))
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(255), unique=True)

    domain: Mapped["Domain"] = relationship(back_populates="subdomains")
    modules: Mapped[List["Module"]] = relationship(back_populates="subdomain")


class Module(Base):
    __tablename__ = "modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subdomain_id: Mapped[int] = mapped_column(ForeignKey("subdomains.id"))
    module_name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(255), unique=True)
    module_json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    model_pkl_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    dataset_json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    subdomain: Mapped["Subdomain"] = relationship(back_populates="modules")
    questions: Mapped[List["Question"]] = relationship(back_populates="module")
    sessions: Mapped[List["InterviewSession"]] = relationship(back_populates="module")


# ── Django-managed tables — FastAPI reads and writes ─────────────────────────

class Question(Base):
    # Django auto-generates: <app_label>_question
    # Update app_label prefix to match your Django app name e.g. "interview_question"
    __tablename__ = "interview_question"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"))
    topic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    question_text: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), server_default=func.now())

    module: Mapped["Module"] = relationship(back_populates="questions")
    user_answers: Mapped[List["InterviewAnswer"]] = relationship(back_populates="question")


class InterviewSession(Base):
    __tablename__ = "interview_interviewsession"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    module_id: Mapped[Optional[int]] = mapped_column(ForeignKey("modules.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), server_default=func.now())
    current_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="active")

    user: Mapped[Optional["User"]] = relationship(back_populates="sessions")
    module: Mapped[Optional["Module"]] = relationship(back_populates="sessions")
    answers: Mapped[List["InterviewAnswer"]] = relationship(
        back_populates="session",
        order_by="InterviewAnswer.id",
    )


class InterviewAnswer(Base):
    __tablename__ = "interview_interviewanswer"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("interview_interviewsession.id"))
    question_id: Mapped[Optional[int]] = mapped_column(ForeignKey("interview_question.id"), nullable=True)
    transcript: Mapped[str] = mapped_column(Text, default="")
    semantic_score: Mapped[float] = mapped_column(Float, default=0.0)
    keyword_score: Mapped[float] = mapped_column(Float, default=0.0)
    question_relevance: Mapped[float] = mapped_column(Float, default=0.0)
    lexical_diversity: Mapped[float] = mapped_column(Float, default=0.0)
    discourse_score: Mapped[float] = mapped_column(Float, default=0.0)
    penalty: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[str] = mapped_column(Text, default="")
    tip: Mapped[str] = mapped_column(Text, default="")
    missing_keywords: Mapped[list] = mapped_column(JSON, default=list)
    raw_segments: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), server_default=func.now())

    session: Mapped["InterviewSession"] = relationship(back_populates="answers")
    question: Mapped[Optional["Question"]] = relationship(back_populates="user_answers")