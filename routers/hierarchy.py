from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from typing import List, Optional

from ..db.database import get_db
from ..db.models import Domain, Module, Question, Subdomain
from ..schemas import DomainOut, ModuleOut, SubdomainOut, ModuleDetailOut, DomainListOut

# All routes under /api/ — matches frontend API_BASE = "http://localhost:8001/api"
router = APIRouter(prefix="/api", tags=["Hierarchy"])



@router.get("/domains/")
@router.get("/domains", response_model=DomainListOut)
async def get_domains(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Domain)
        .options(
            selectinload(Domain.subdomains)
            .selectinload(Subdomain.modules)
            .selectinload(Module.questions)
        )
        .order_by(Domain.name)
    )
    return {"data": result.scalars().all()}


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
async def get_module_detail(module_id: int, db: AsyncSession = Depends(get_db)):
    query = (
        select(
            Module.id,
            Module.module_name,
            Module.slug,
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
    return row