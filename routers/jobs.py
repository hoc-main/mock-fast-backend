import os
import shutil
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse


from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db.database import get_db
from ..db.models import Job, JobApplication, User
from ..schemas import JobOut, JobApplicationOut
from ..services.email_service import send_application_confirmation_email


router = APIRouter(prefix="/api", tags=["Jobs"])
logger = logging.getLogger(__name__)


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
    background_tasks: BackgroundTasks = BackgroundTasks(),
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

    # 5. Fetch user and job details for email
    user_result = await db.execute(select(User).where(User.user_id == user_id))
    user = user_result.scalar_one_or_none()
    
    if user and user.email:
        logger.info(f"Adding background task to send confirmation email to {user.email}")
        background_tasks.add_task(
            send_application_confirmation_email,
            recipient_email=user.email,
            user_name=f"{user.first_name} {user.last_name}",
            job_title=job.title,
            company_name=job.company
        )
    else:
        logger.warning(f"Could not send confirmation email for user_id {user_id}: User not found or email missing.")

    logger.info(f"Application {application.id} submitted successfully for job {job_id} by user {user_id}")
    return {"message": "Application submitted successfully", "application_id": application.id}

@router.get("/corporate/{corporate_user_id}/applicants")
async def get_corporate_applicants(corporate_user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(JobApplication, Job, User)
        .join(Job, JobApplication.job_id == Job.id)
        .join(User, JobApplication.user_id == User.user_id)
        .where(Job.corporate_user_id == corporate_user_id)
        .order_by(JobApplication.applied_at.desc())
    )
    
    applicants = []
    for app, job, user in result.all():
        applicants.append({
            "id": app.id,
            "job_id": job.id,
            "job_title": job.title,
            "applicant_name": f"{user.first_name} {user.last_name}",
            "applicant_email": user.email,
            "phone_number": app.phone_number,
            "resume_url": app.resume_url,
            "certificates_url": app.certificates_url,
            "cover_letter": app.cover_letter,
            "status": app.status,
            "applied_at": app.applied_at.isoformat() if app.applied_at else None
        })
    return applicants


@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    # The filename parameter is a path suffix after "uploads/"
    # For security, let's make sure they are not doing directory traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path, media_type="application/pdf")

