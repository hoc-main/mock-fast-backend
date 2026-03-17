from fastapi import APIRouter, Depends
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db.models import InterviewAnswer, InterviewSession

# GET /api/performance-stats/?user_id={id}
router = APIRouter(prefix="/api", tags=["Stats"])


@router.get("/performance-stats/")
async def get_performance_stats(user_id: int, db: AsyncSession = Depends(get_db)):
    sessions_result = await db.execute(
        select(InterviewSession)
        .where(InterviewSession.user_id == user_id, InterviewSession.status == "completed")
    )
    sessions = sessions_result.scalars().all()
    attempts = len(sessions)

    if attempts == 0:
        return {
            "attempts": 0,
            "latest_score": 0,
            "avg_score": 0,
            "confidence_score": 0,
            "technical_score": 0,
            "global_rank": None,
        }

    # Latest session score
    latest = sorted(sessions, key=lambda s: s.created_at, reverse=True)[0]
    latest_avg = await db.execute(
        select(func.avg(InterviewAnswer.final_score))
        .where(InterviewAnswer.session_id == latest.id)
    )
    latest_score = round(latest_avg.scalar() or 0.0, 1)

    # Overall avg + stddev
    session_ids = [s.id for s in sessions]
    overall = await db.execute(
        select(
            func.avg(InterviewAnswer.final_score),
            func.stddev(InterviewAnswer.final_score),
        ).where(InterviewAnswer.session_id.in_(session_ids))
    )
    avg_val, std_val = overall.one()
    avg_score        = round(avg_val or 0.0, 1)
    confidence_score = round(max(0, 100 - (std_val or 0)), 1)

    # Technical score: avg of last 5 sessions
    last_5_ids = [s.id for s in sorted(sessions, key=lambda s: s.created_at, reverse=True)[:5]]
    tech_result = await db.execute(
        select(func.avg(InterviewAnswer.final_score))
        .where(InterviewAnswer.session_id.in_(last_5_ids))
    )
    technical_score = round(tech_result.scalar() or 0.0, 1)

    # Global rank
    rank_rows = await db.execute(
        select(
            InterviewSession.user_id,
            func.avg(InterviewAnswer.final_score).label("user_avg"),
        )
        .join(InterviewAnswer, InterviewAnswer.session_id == InterviewSession.id)
        .where(InterviewSession.status == "completed")
        .group_by(InterviewSession.user_id)
        .order_by(desc("user_avg"))
    )
    rank = 1
    for row in rank_rows.all():
        if row.user_id == user_id:
            break
        rank += 1

    return {
        "attempts": attempts,
        "latest_score": latest_score,
        "avg_score": avg_score,
        "confidence_score": confidence_score,
        "technical_score": technical_score,
        "global_rank": rank,
    }