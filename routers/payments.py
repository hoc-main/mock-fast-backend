"""
routers/payments.py
====================
Payment management endpoints for purchasing mock interviews using Razorpay.

POST /api/payments/create-order/
POST /api/payments/verify-payment/
POST /api/payments/webhook/
GET  /api/payments/user-purchases/
"""
import logging
import os
import razorpay
import hmac
import hashlib
import datetime
from typing import Optional
from razorpay.errors import SignatureVerificationError

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from ..db.database import get_db
from ..db.models import Order, Purchase, User, Module
from ..schemas import CreateOrderRequest, CreateOrderResponse, VerifyPaymentRequest, PurchaseOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["Payments"])

# Initialize Razorpay client
razorpay_client = razorpay.Client(auth=(
    os.getenv("RAZORPAY_KEY_ID"),
    os.getenv("RAZORPAY_KEY_SECRET")
))

# Razorpay webhook secret from env
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

@router.get("/price/")
async def get_price(user_id: int, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.user_id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.created_at:
        now = datetime.datetime.now()
        if user.created_at.tzinfo is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
        time_diff = now - user.created_at
        if time_diff.total_seconds() <= 48 * 3600:
            return {"rupees": 20, "paise": 2000, "display": "₹20.00"}
            
    return {"rupees": 50, "paise": 5000, "display": "₹50.00"}



@router.post("/create-order/", response_model=CreateOrderResponse)
async def create_order(body: CreateOrderRequest, db: AsyncSession = Depends(get_db)):
    # Check if user and module exist
    user_result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    module_result = await db.execute(select(Module).where(Module.id == body.module_id))
    module = module_result.scalar_one_or_none()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Calculate amount dynamically
    actual_amount = 5000
    if user.created_at:
        now = datetime.datetime.now()
        if user.created_at.tzinfo is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
        time_diff = now - user.created_at
        if time_diff.total_seconds() <= 48 * 3600:
            actual_amount = 2000
            
    # Create Razorpay order
    razorpay_order = razorpay_client.order.create({
        "amount": actual_amount,
        "currency": "INR",
        "payment_capture": 1
    })
    
    # Save order to DB
    new_order = Order(
        user_id=body.user_id,
        module_id=body.module_id,
        razorpay_order_id=razorpay_order["id"],
        amount=actual_amount,
        currency="INR",
        status="created"
    )
    db.add(new_order)
    await db.commit()
    await db.refresh(new_order)
    
    return CreateOrderResponse(
        order_id=new_order.id,
        razorpay_order_id=razorpay_order["id"],
        amount=actual_amount,
        currency="INR"
    )


@router.post("/verify-payment/", response_model=PurchaseOut)
async def verify_payment(body: VerifyPaymentRequest, db: AsyncSession = Depends(get_db)):
    # Retrieve the order
    order_result = await db.execute(select(Order).where(Order.id == body.order_id))
    order = order_result.scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Check that order ID from request matches the order
    if body.razorpay_order_id != order.razorpay_order_id:
        raise HTTPException(status_code=400, detail="Order ID mismatch")
    
    # Check if purchase already exists
    existing_purchase_result = await db.execute(
        select(Purchase).where(Purchase.order_id == order.id)
    )
    existing_purchase = existing_purchase_result.scalar_one_or_none()
    if existing_purchase:
        return existing_purchase
    
    # Verify Razorpay signature
    try:
        razorpay_client.utility.verify_payment_signature({
            "razorpay_order_id": body.razorpay_order_id,
            "razorpay_payment_id": body.razorpay_payment_id,
            "razorpay_signature": body.razorpay_signature
        })
    except SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Payment verification failed")
    
    # Fetch payment from Razorpay and verify amount, status, and order ID
    try:
        payment = razorpay_client.payment.fetch(body.razorpay_payment_id)
        
        # Verify amount matches the order amount
        if payment["amount"] != order.amount:
            raise HTTPException(status_code=400, detail="Amount mismatch")
        
        # Verify payment is captured
        if payment["status"] != "captured":
            raise HTTPException(status_code=400, detail="Payment not captured")
        
        # Verify payment belongs to this order
        if payment["order_id"] != order.razorpay_order_id:
            raise HTTPException(status_code=400, detail="Payment belongs to different order")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to communicate with Razorpay")
        raise HTTPException(status_code=500, detail="Failed to communicate with Razorpay")
    
    # Create purchase and update order status (safer, no nested transactions)
    try:
        purchase = Purchase(
            user_id=order.user_id,
            module_id=order.module_id,
            order_id=order.id,
            razorpay_payment_id=body.razorpay_payment_id,
            razorpay_order_id=body.razorpay_order_id,
            razorpay_signature=body.razorpay_signature,
            amount=order.amount,
            currency=order.currency
        )
        db.add(purchase)
        
        order.status = "paid"
        
        await db.commit()
        await db.refresh(purchase)
        return purchase
    except IntegrityError:
        await db.rollback()
        # Race condition - purchase already created, return it
        existing_purchase_result = await db.execute(
            select(Purchase).where(Purchase.order_id == order.id)
        )
        existing_purchase = existing_purchase_result.scalar_one_or_none()
        return existing_purchase


@router.get("/user-purchases/", response_model=list[PurchaseOut])
async def get_user_purchases(user_id: int, db: AsyncSession = Depends(get_db)):
    # TODO: Replace with proper authentication (JWT/session) and use current_user.id instead
    # For now, keep it as is, but this is insecure in production!
    result = await db.execute(
        select(Purchase)
        .where(Purchase.user_id == user_id)
        .order_by(Purchase.created_at.desc())
    )
    return result.scalars().all()


@router.post("/webhook/")
async def razorpay_webhook(request: Request, x_razorpay_signature: Optional[str] = Header(None), db: AsyncSession = Depends(get_db)):
    """
    Handle Razorpay webhook events (payment success, failure, etc.)
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        logger.warning("Razorpay webhook secret not set - ignoring webhook")
        return {"status": "ok"}
    
    # Get request body
    body = await request.body()
    body_str = body.decode("utf-8")
    
    # Verify webhook signature
    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_signature, x_razorpay_signature or ""):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Parse webhook payload
    import json
    payload = json.loads(body_str)
    event = payload.get("event")
    logger.info(f"Received Razorpay webhook event: {event}")
    
    if event == "payment.captured":
        # Handle successful payment
        payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        razorpay_order_id = payment_entity.get("order_id")
        razorpay_payment_id = payment_entity.get("id")
        
        if razorpay_order_id:
            # Find the order (order is our source of truth for amount and currency)
            order_result = await db.execute(
                select(Order).where(Order.razorpay_order_id == razorpay_order_id)
            )
            order = order_result.scalar_one_or_none()
            
            if order and order.status == "created":
                # Check if purchase already exists (in case verify-payment was already called)
                existing_purchase_result = await db.execute(
                    select(Purchase).where(Purchase.order_id == order.id)
                )
                existing_purchase = existing_purchase_result.scalar_one_or_none()
                
                if not existing_purchase:
                    try:
                        purchase = Purchase(
                            user_id=order.user_id,
                            module_id=order.module_id,
                            order_id=order.id,
                            razorpay_payment_id=razorpay_payment_id,
                            razorpay_order_id=razorpay_order_id,
                            webhook_signature=x_razorpay_signature,
                            amount=order.amount,
                            currency=order.currency
                        )
                        db.add(purchase)
                        order.status = "paid"
                        
                        await db.commit()
                        logger.info(f"Created purchase via webhook for order {order.id}")
                    except IntegrityError:
                        await db.rollback()
                        logger.warning(f"Purchase already exists for order {order.id} (race condition)")
    
    elif event == "payment.failed":
        # Handle failed payment
        payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        razorpay_order_id = payment_entity.get("order_id")
        
        if razorpay_order_id:
            order_result = await db.execute(
                select(Order).where(Order.razorpay_order_id == razorpay_order_id)
            )
            order = order_result.scalar_one_or_none()
            
            if order and order.status == "created":
                order.status = "failed"
                await db.commit()
                logger.info(f"Marked order {order.id} as failed via webhook")
    
    return {"status": "ok"}
