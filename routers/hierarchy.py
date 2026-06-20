from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from typing import List, Optional

from ..db.database import get_db
from ..db.models import Domain, Module, Question, Subdomain, InterviewSession, Purchase
from ..schemas import DomainOut, ModuleOut, SubdomainOut, ModuleDetailOut, DomainListOut

# All routes under /api/ — matches frontend API_BASE = "http://localhost:8001/api"
router = APIRouter(prefix="/api", tags=["Hierarchy"])



@router.get("/domains/")
@router.get("/domains", response_model=DomainListOut)
async def get_domains(user_id: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Domain)
        .options(
            selectinload(Domain.subdomains)
            .selectinload(Subdomain.modules)
            .selectinload(Module.questions)
        )
        .order_by(Domain.name)
    )
    domains = result.scalars().all()

    attempted_module_ids = set()
    active_module_ids = set()
    purchased_module_ids = set()
    if user_id:
        sessions_result = await db.execute(
            select(InterviewSession.module_id, InterviewSession.status)
            .where(InterviewSession.user_id == user_id)
        )
        for mid, status in sessions_result.all():
            if mid is not None:
                if status == "completed":
                    attempted_module_ids.add(mid)
                elif status == "active":
                    active_module_ids.add(mid)
        
        purchases_result = await db.execute(
            select(Purchase.module_id)
            .where(Purchase.user_id == user_id)
        )
        purchased_module_ids = {mid for mid in purchases_result.scalars().all() if mid is not None}

    for domain in domains:
        for subdomain in domain.subdomains:
            for module in subdomain.modules:
                module.is_attempted = module.id in attempted_module_ids
                module.is_active = module.id in active_module_ids
                module.is_purchased = module.id in purchased_module_ids

    return {"data": domains}


@router.get("/domains/{domain_id}/subdomains/", response_model=List[SubdomainOut])
async def get_subdomains(domain_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Subdomain)
        .where(Subdomain.domain_id == domain_id)
        .order_by(Subdomain.name)
    )
    return result.scalars().all()


@router.get("/subdomains/{subdomain_id}/modules/", response_model=List[ModuleOut])
async def get_modules(subdomain_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Module)
        .options(selectinload(Module.questions))
        .where(Module.subdomain_id == subdomain_id)
        .order_by(Module.module_name)
    )
    return result.scalars().all()


@router.get("/modules/{module_id}/", response_model=ModuleDetailOut)
async def get_module_detail(module_id: int, user_id: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    query = (
        select(
            Module,
            Subdomain.name.label("subdomain_name"),
            Domain.name.label("domain_name"),
            func.count(Question.id).label("question_count")
        )
        .join(Subdomain, Module.subdomain_id == Subdomain.id)
        .join(Domain, Subdomain.domain_id == Domain.id)
        .outerjoin(Question, Module.id == Question.module_id)
        .where(Module.id == module_id)
        .group_by(Module.id, Subdomain.name, Domain.name)
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Module not found")
        
    module_obj, subdomain_name, domain_name, question_count = row
    
    is_attempted = False
    is_active = False
    is_purchased = False
    
    if user_id:
        # Check if attempted or active
        sessions_result = await db.execute(
            select(InterviewSession.status)
            .where(
                InterviewSession.user_id == user_id,
                InterviewSession.module_id == module_id
            )
        )
        for status in sessions_result.scalars().all():
            if status == "completed":
                is_attempted = True
            elif status == "active":
                is_active = True
        
        # Check if purchased
        purchase_result = await db.execute(
            select(Purchase.id)
            .where(Purchase.user_id == user_id, Purchase.module_id == module_id)
            .limit(1)
        )
        is_purchased = purchase_result.scalar_one_or_none() is not None
        
    return ModuleDetailOut(
        id=module_obj.id,
        module_name=module_obj.module_name,
        slug=module_obj.slug,
        is_free=module_obj.is_free,
        companies=module_obj.companies or [],
        job_roles=module_obj.job_roles or [],
        domain_name=domain_name,
        subdomain_name=subdomain_name,
        question_count=question_count,
        is_attempted=is_attempted,
        is_active=is_active,
        is_purchased=is_purchased
    )