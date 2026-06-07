import asyncio
import os
import sys

# Ensure parent directory is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import AsyncSessionLocal
from db.models import Module
from sqlalchemy import select

# Helper to determine job roles based on module name
def get_roles_for_module(name: str) -> list:
    n = name.lower()
    if "data science" in n or "data science core" in n:
        return ["Data Scientist", "ML Engineer", "Data Analyst"]
    if "machine learning" in n or "ml core" in n or "deep learning" in n:
        return ["ML Engineer", "AI Researcher", "Data Scientist"]
    if "ai" in n or "artificial intelligence" in n:
        return ["AI Engineer", "ML Specialist", "Research Scientist"]
    if "system design" in n or "architecture" in n:
        return ["System Architect", "Senior Backend Engineer", "Technical Lead"]
    if "web" in n or "frontend" in n or "react" in n or "javascript" in n:
        return ["Frontend Developer", "Full Stack Developer", "Web Engineer"]
    if "backend" in n or "node" in n or "python" in n:
        return ["Backend Developer", "Software Engineer", "Full Stack Developer"]
    if "devops" in n or "cloud" in n or "kubernetes" in n:
        return ["DevOps Engineer", "Cloud Solutions Architect", "SRE"]
    if "cyber" in n or "security" in n:
        return ["Security Engineer", "Security Analyst", "Penetration Tester"]
    return ["Software Engineer", "Full Stack Developer"]

async def seed_job_roles():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Module))
        modules = result.scalars().all()
        print(f"Found {len(modules)} modules in the database.")
        
        for m in modules:
            roles = get_roles_for_module(m.module_name)
            m.job_roles = roles
            print(f"Assigned roles {roles} to module '{m.module_name}'")
            
        await db.commit()
        print("✅ Successfully seeded job roles for all modules!")

if __name__ == "__main__":
    asyncio.run(seed_job_roles())
