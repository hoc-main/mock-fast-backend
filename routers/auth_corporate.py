import random
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
import jwt
import os

from ..db.database import get_db
from ..db.models import CorporateUser, CorporateOTP
from ..services.email_service import send_otp_email

router = APIRouter(
    prefix="/api/corporate",
    tags=["Corporate Auth"]
)

class SendOtpRequest(BaseModel):
    email: EmailStr

class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str

@router.post("/send-otp")
async def send_otp(req: SendOtpRequest, db: AsyncSession = Depends(get_db)):
    # Verify if user exists
    result = await db.execute(select(CorporateUser).filter(CorporateUser.corp_email == req.email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found. Please register first."
        )

    # Rate Limiting / Cooldown Check (1 minute)
    recent_otp_result = await db.execute(
        select(CorporateOTP)
        .filter(CorporateOTP.email == req.email)
        .order_by(CorporateOTP.created_at.desc())
    )
    recent_otp = recent_otp_result.scalars().first()
    
    if recent_otp and (datetime.utcnow() - recent_otp.created_at) < timedelta(minutes=1):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please wait at least 1 minute before requesting a new OTP."
        )

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    # Clean up older OTPs for this email to avoid clutter
    await db.execute(delete(CorporateOTP).where(CorporateOTP.email == req.email))
    await db.commit()

    # Save new OTP
    new_otp = CorporateOTP(
        email=req.email,
        otp=otp,
        expires_at=expires_at
    )
    db.add(new_otp)
    await db.commit()

    # Send email
    success = send_otp_email(req.email, otp)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email."
        )

    return {"success": True, "message": "OTP sent successfully."}


@router.post("/verify-otp")
async def verify_otp(req: VerifyOtpRequest, db: AsyncSession = Depends(get_db)):
    # Find OTP
    result = await db.execute(
        select(CorporateOTP).filter(
            CorporateOTP.email == req.email,
            CorporateOTP.otp == req.otp
        )
    )
    otp_record = result.scalars().first()

    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OTP."
        )

    if otp_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OTP has expired."
        )

    # Fetch user
    user_result = await db.execute(select(CorporateUser).filter(CorporateUser.corp_email == req.email))
    user = user_result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    # Delete OTP record after successful verification
    await db.execute(delete(CorporateOTP).where(CorporateOTP.id == otp_record.id))
    await db.commit()

    # Generate session token (mimicking flask_jwt_extended access token)
    secret_key = os.getenv("JWT_SECRET_KEY", "super-secret-key")
    payload = {
        "iat": datetime.utcnow(),
        "nbf": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=1),
        "jti": str(uuid.uuid4()),
        "sub": user.corp_user_id,
        "type": "access"
    }
    access_token = jwt.encode(payload, secret_key, algorithm="HS256")

    user_data = {
        "id": user.corp_user_id,
        "email": user.corp_email,
        "firstName": user.corp_first_name,
        "lastName": user.corp_last_name,
        "jobTitle": user.corp_job_tile,
        "corpName": user.corp_corp_name,
        "empSize": user.corp_emp_size,
        "phoneNo": user.corp_phone_no,
        "region": user.corp_region,
    }

    return {
        "message": "Login successful",
        "corp_user_data": user_data,
        "token": access_token
    }
