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
    Integer, JSON, String, Text, func, Boolean, UniqueConstraint
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    sessions: Mapped[List["InterviewSession"]] = relationship(back_populates="user")


class CorporateUser(Base):
    __tablename__ = "corporate_users"

    corp_user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corp_email: Mapped[str] = mapped_column(String(255), unique=True)
    corp_first_name: Mapped[str] = mapped_column(String(255))
    corp_last_name: Mapped[str] = mapped_column(String(255))
    password: Mapped[str] = mapped_column(String(255))
    corp_job_tile: Mapped[str] = mapped_column(String(255))
    corp_corp_name: Mapped[str] = mapped_column(String(255))
    corp_emp_size: Mapped[str] = mapped_column(String(255))
    corp_phone_no: Mapped[str] = mapped_column(String(255))
    corp_region: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())



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
    is_free: Mapped[bool] = mapped_column(Boolean, default=True)
    companies: Mapped[list] = mapped_column(JSON, default=list) # List of company names
    job_roles: Mapped[list] = mapped_column(JSON, default=list) # List of job roles this module covers

    subdomain: Mapped["Subdomain"] = relationship(back_populates="modules")
    questions: Mapped[List["Question"]] = relationship(back_populates="module")
    sessions: Mapped[List["InterviewSession"]] = relationship(back_populates="module")

    @property
    def question_count(self) -> int:
        return len(self.questions)


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
    current_question_id: Mapped[Optional[int]] = mapped_column(ForeignKey("interview_question.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    conversation_history: Mapped[list] = mapped_column(JSON, default=list)  # [{question, score, gaps}]
    asked_question_ids: Mapped[list] = mapped_column(JSON, default=list)    # [int] IDs already asked

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


# ── Jobs & Applications ──────────────────────────────────────────────────────

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    corporate_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    work_mode: Mapped[str] = mapped_column(String(50))
    start_date: Mapped[str] = mapped_column(String(50))
    duration: Mapped[str] = mapped_column(String(50))
    stipend: Mapped[str] = mapped_column(String(100))
    apply_by: Mapped[str] = mapped_column(String(50))
    type: Mapped[str] = mapped_column(String(50))
    schedule: Mapped[str] = mapped_column(String(50))
    posted: Mapped[str] = mapped_column(String(50))
    skills: Mapped[list] = mapped_column(JSON, default=list) # JSON array
    description: Mapped[str] = mapped_column(Text)
    responsibilities: Mapped[list] = mapped_column(JSON, default=list) # JSON array
    additional_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    who_can_apply: Mapped[list] = mapped_column(JSON, default=list) # JSON array
    other_requirements: Mapped[list] = mapped_column(JSON, default=list) # JSON array
    perks: Mapped[list] = mapped_column(JSON, default=list) # JSON array
    company_about: Mapped[str] = mapped_column(Text)
    opps_posted: Mapped[int] = mapped_column(Integer, default=0)
    candidates_hired: Mapped[int] = mapped_column(Integer, default=0)
    logo: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    applications: Mapped[List["JobApplication"]] = relationship(back_populates="job")


class JobApplication(Base):
    __tablename__ = "job_applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    phone_number: Mapped[str] = mapped_column(String(15))
    resume_url: Mapped[str] = mapped_column(String(255))
    certificates_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cover_letter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    job: Mapped["Job"] = relationship(back_populates="applications")
    user: Mapped["User"] = relationship()


# ── Payments & Purchases ──────────────────────────────────────────────────

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"))
    razorpay_order_id: Mapped[str] = mapped_column(String(100))
    amount: Mapped[int] = mapped_column(Integer)  # Amount in paise
    currency: Mapped[str] = mapped_column(String(10), default="INR")
    status: Mapped[str] = mapped_column(String(50), default="created")  # created, paid, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        UniqueConstraint('razorpay_order_id', name='_razorpay_order_id_uc'),
    )


class Purchase(Base):
    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"))
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    razorpay_payment_id: Mapped[str] = mapped_column(String(100))
    razorpay_order_id: Mapped[str] = mapped_column(String(100))
    razorpay_signature: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    webhook_signature: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    amount: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(10), default="INR")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        UniqueConstraint('order_id', name='_order_id_uc'),
        UniqueConstraint('razorpay_payment_id', name='_razorpay_payment_id_uc'),
    )


# ── OTP ───────────────────────────────────────────────────────────────────────

class CorporateOTP(Base):
    __tablename__ = "corporate_otps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    otp: Mapped[str] = mapped_column(String(10))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())