import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db.database import get_db
from ..db.models import Job, JobApplication, User
from ..schemas import JobOut, JobApplicationOut

router = APIRouter(prefix="/api", tags=["Jobs"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/jobs", response_model=List[JobOut])
async def get_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).order_by(Job.created_at.desc()))
    return result.scalars().all()

@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job_detail(job_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/jobs/{job_id}/apply")
async def apply_for_job(
    job_id: int,
    phone_number: str = Form(...),
    cover_letter: Optional[str] = Form(None),
    resume: UploadFile = File(...),
    certificate: Optional[UploadFile] = File(None),
    user_id: int = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Validate Job
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 2. Validate Phone
    if not phone_number.isdigit() or len(phone_number) != 10:
        raise HTTPException(status_code=400, detail="Invalid phone number. Must be 10 digits.")

    # 3. Save Files
    resume_ext = os.path.splitext(resume.filename)[1]
    if resume_ext.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    
    resume_filename = f"{uuid.uuid4()}{resume_ext}"
    # Physical path for saving
    resume_physical_path = os.path.join(UPLOAD_DIR, resume_filename)
    with open(resume_physical_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
    
    # Logical path for DB (always forward slashes)
    resume_stored_path = f"uploads/{resume_filename}"

    cert_stored_path = None
    if certificate:
        cert_ext = os.path.splitext(certificate.filename)[1]
        if cert_ext.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Certificates must be a PDF file")
        
        cert_filename = f"{uuid.uuid4()}{cert_ext}"
        cert_physical_path = os.path.join(UPLOAD_DIR, cert_filename)
        with open(cert_physical_path, "wb") as buffer:
            shutil.copyfileobj(certificate.file, buffer)
        cert_stored_path = f"uploads/{cert_filename}"

    # 4. Create Application
    application = JobApplication(
        job_id=job_id,
        user_id=user_id,
        phone_number=phone_number,
        resume_url=resume_stored_path,
        certificates_url=cert_stored_path,
        cover_letter=cover_letter,
        status="pending"
    )
    db.add(application)
    await db.commit()
    await db.refresh(application)

    return {"message": "Application submitted successfully", "application_id": application.id}
