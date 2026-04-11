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

from datetime import datetime, timedelta
from itertools import groupby
from pydantic import BaseModel
from typing import List
from ..db.models import User

class LeaderboardEntry(BaseModel):
    user_id: int
    name: str
    avg_score: float
    streak: int
    rank: int

@router.get("/mock/leaderboard/", response_model=List[LeaderboardEntry])
async def get_mock_leaderboard(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """
    Returns the global leaderboard of users based on their average score and active streak.
    """
    # 1. Fetch dates of all completed sessions grouped by user, ordered descending
    dates_query = await db.execute(
        select(InterviewSession.user_id, func.date(InterviewSession.created_at).label("session_date"))
        .where(InterviewSession.status == "completed")
        .group_by(InterviewSession.user_id, func.date(InterviewSession.created_at))
        .order_by(InterviewSession.user_id, desc("session_date"))
    )
    user_dates = dates_query.all()

    # 2. Compute dynamic streak per user
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    
    streaks = {}
    for user_id, items in groupby(user_dates, key=lambda x: x.user_id):
        dates_list = [i.session_date for i in items]
        streak = 0
        # Streak is active if they did a session today or yesterday
        if dates_list and dates_list[0] >= yesterday:
            streak = 1
            for i in range(1, len(dates_list)):
                if (dates_list[i-1] - dates_list[i]).days == 1:
                    streak += 1
                else:
                    break
        streaks[user_id] = streak

    # 3. Fetch all average scores per user
    scores_query = await db.execute(
        select(
            InterviewSession.user_id,
            func.avg(InterviewAnswer.final_score).label("avg_score")
        )
        .join(InterviewAnswer, InterviewAnswer.session_id == InterviewSession.id)
        .where(InterviewSession.status == "completed")
        .group_by(InterviewSession.user_id)
    )
    user_scores = {r.user_id: round(r.avg_score or 0.0, 1) for r in scores_query.all()}
    
    # 4. Fetch user names
    users_query = await db.execute(select(User.user_id, User.first_name, User.last_name))
    user_names = {u.user_id: f"{u.first_name} {u.last_name}".strip() for u in users_query.all()}
    
    # 5. Assemble data
    leaderboard = []
    for uid, score in user_scores.items():
        name = user_names.get(uid)
        if not name:
            name = f"User {uid}"
        leaderboard.append({
            "user_id": uid,
            "name": name,
            "avg_score": score,
            "streak": streaks.get(uid, 0)
        })
        
    # Sort primarily by avg_score DESC, tie-breaker streak DESC
    leaderboard.sort(key=lambda x: (x["avg_score"], x["streak"]), reverse=True)
    
    # Assign ranks and enforce limit
    final_result = []
    for rank, entry in enumerate(leaderboard[:limit], start=1):
        entry["rank"] = rank
        final_result.append(LeaderboardEntry(**entry))
        
    return final_result