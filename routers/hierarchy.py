from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ..db.database import get_db
from ..db.models import Domain, Module, Subdomain
from ..schemas import DomainOut, ModuleOut, SubdomainOut

# All routes under /api/ — matches frontend API_BASE = "http://localhost:8001/api"
router = APIRouter(prefix="/api", tags=["Hierarchy"])


@router.get("/domains/", response_model=List[DomainOut])
async def get_domains(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Domain).order_by(Domain.name))
    return result.scalars().all()


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
        .where(Module.subdomain_id == subdomain_id)
        .order_by(Module.module_name)
    )
    return result.scalars().all()


@router.get("/modules/{module_id}/")
async def get_module_detail(module_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return {
        "id": module.id,
        "module_name": module.module_name,
        "slug": module.slug,
        "subdomain_id": module.subdomain_id,
    }