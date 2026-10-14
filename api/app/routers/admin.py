import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_admin_user, get_db
from ..models.db import CreditTransaction, Job, SchedulerHeartbeat, User

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Response schemas ──────────────────────────────────────────────────────────

class AdminStatus(BaseModel):
    is_admin: bool


class AdminJobItem(BaseModel):
    job_id: uuid.UUID
    user_id: str
    user_email: str
    status: str
    config_name: str | None
    instance_type: str | None
    instance_id: str | None
    estimated_cost: float | None
    actual_cost: float | None
    created_at: str
    provisioned_at: str | None
    started_at: str | None
    completed_at: str | None
    error_message: str | None


class AdminUserItem(BaseModel):
    id: str
    email: str
    credit_balance_cents: int
    job_count: int


class AdjustCreditsRequest(BaseModel):
    amount_cents: int = Field(..., ge=-10_000_000, le=10_000_000)  # ±$100,000
    description: str = Field(..., min_length=1, max_length=500)


class AdjustCreditsResponse(BaseModel):
    new_balance_cents: int


class SchedulerStatus(BaseModel):
    alive: bool
    ticked_at: str | None
    pending_count: int
    active_count: int
    dispatched_count: int


class DailyCount(BaseModel):
    date: str
    count: int


class DailyRevenue(BaseModel):
    date: str
    revenue_cents: int


class AdminMetricsResponse(BaseModel):
    total_jobs: int
    jobs_by_status: dict[str, int]
    total_users: int
    total_revenue_cents: int
    outstanding_credits_cents: int
    jobs_per_day: list[DailyCount]
    revenue_per_day: list[DailyRevenue]
    new_users_per_day: list[DailyCount]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/scheduler", response_model=SchedulerStatus)
async def scheduler_status(
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(SchedulerHeartbeat))
    hb = result.scalar_one_or_none()
    if not hb:
        return SchedulerStatus(alive=False, ticked_at=None, pending_count=0, active_count=0, dispatched_count=0)
    age_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - hb.ticked_at).total_seconds()
    return SchedulerStatus(
        alive=age_seconds < 30,
        ticked_at=hb.ticked_at.isoformat() + 'Z',
        pending_count=hb.pending_count,
        active_count=hb.active_count,
        dispatched_count=hb.dispatched_count,
    )


@router.get("/status", response_model=AdminStatus)
async def admin_status(_: User = Depends(get_admin_user)):
    return AdminStatus(is_admin=True)


@router.get("/jobs", response_model=list[AdminJobItem])
async def list_all_jobs(
    status: str | None = None,
    user_id: str | None = None,
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Job, User.email).join(User, Job.session_id == User.id)
    if status:
        query = query.where(Job.status == status)
    if user_id:
        query = query.where(Job.session_id == user_id)
    query = query.order_by(Job.created_at.desc()).limit(500)

    result = await db.execute(query)
    return [
        AdminJobItem(
            job_id=job.id,
            user_id=job.session_id,
            user_email=email,
            status=job.status,
            config_name=job.config_name,
            instance_type=job.instance_type,
            instance_id=job.instance_id if job.instance_id and job.instance_id != 'provisioning' else None,
            estimated_cost=float(job.estimated_cost) if job.estimated_cost is not None else None,
            actual_cost=float(job.actual_cost) if job.actual_cost is not None else None,
            created_at=job.created_at.isoformat() + 'Z',
            provisioned_at=job.provisioned_at.isoformat() + 'Z' if job.provisioned_at else None,
            started_at=job.started_at.isoformat() + 'Z' if job.started_at else None,
            completed_at=job.completed_at.isoformat() + 'Z' if job.completed_at else None,
            error_message=job.error_message,
        )
        for job, email in result.all()
    ]


@router.get("/users", response_model=list[AdminUserItem])
async def list_all_users(
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User, func.count(Job.id).label("job_count"))
        .outerjoin(Job, Job.session_id == User.id)
        .group_by(User.id)
        .order_by(User.email)
    )
    return [
        AdminUserItem(
            id=user.id,
            email=user.email,
            credit_balance_cents=user.credit_balance_cents,
            job_count=job_count,
        )
        for user, job_count in result.all()
    ]


@router.post("/users/{user_id}/credits", response_model=AdjustCreditsResponse)
async def adjust_credits(
    user_id: str,
    body: AdjustCreditsRequest,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id).with_for_update())
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.credit_balance_cents += body.amount_cents
    if target.credit_balance_cents < 0:
        target.credit_balance_cents = 0

    db.add(CreditTransaction(
        user_id=user_id,
        amount_cents=body.amount_cents,
        description=f"[Admin: {admin.email}] {body.description}",
    ))
    await db.commit()
    return AdjustCreditsResponse(new_balance_cents=target.credit_balance_cents)


@router.get("/metrics", response_model=AdminMetricsResponse)
async def get_metrics(
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)

    # Jobs by status
    result = await db.execute(
        select(Job.status, func.count(Job.id).label("cnt")).group_by(Job.status)
    )
    jobs_by_status: dict[str, int] = {row.status: row.cnt for row in result.all()}
    total_jobs = sum(jobs_by_status.values())

    # Jobs per day (last 30 days)
    result = await db.execute(
        select(cast(Job.created_at, Date).label("date"), func.count(Job.id).label("cnt"))
        .where(Job.created_at >= cutoff)
        .group_by(cast(Job.created_at, Date))
        .order_by(cast(Job.created_at, Date))
    )
    jobs_per_day = [DailyCount(date=str(row.date), count=row.cnt) for row in result.all()]

    # Revenue per day (last 30 days) — completed jobs only
    result = await db.execute(
        select(cast(Job.completed_at, Date).label("date"), func.sum(Job.actual_cost).label("revenue"))
        .where(Job.status == "completed", Job.completed_at >= cutoff, Job.actual_cost.is_not(None))
        .group_by(cast(Job.completed_at, Date))
        .order_by(cast(Job.completed_at, Date))
    )
    revenue_per_day = [
        DailyRevenue(date=str(row.date), revenue_cents=int(float(row.revenue) * 100))
        for row in result.all()
    ]

    # New users per day (last 30 days)
    result = await db.execute(
        select(cast(User.created_at, Date).label("date"), func.count(User.id).label("cnt"))
        .where(User.created_at >= cutoff)
        .group_by(cast(User.created_at, Date))
        .order_by(cast(User.created_at, Date))
    )
    new_users_per_day = [DailyCount(date=str(row.date), count=row.cnt) for row in result.all()]

    result = await db.execute(select(func.count(User.id)))
    total_users = result.scalar_one()

    result = await db.execute(
        select(func.sum(Job.actual_cost)).where(Job.status == "completed", Job.actual_cost.is_not(None))
    )
    total_revenue_cents = int(float(result.scalar_one() or 0) * 100)

    result = await db.execute(select(func.sum(User.credit_balance_cents)))
    outstanding_credits_cents = int(result.scalar_one() or 0)

    return AdminMetricsResponse(
        total_jobs=total_jobs,
        jobs_by_status=jobs_by_status,
        total_users=total_users,
        total_revenue_cents=total_revenue_cents,
        outstanding_credits_cents=outstanding_credits_cents,
        jobs_per_day=jobs_per_day,
        revenue_per_day=revenue_per_day,
        new_users_per_day=new_users_per_day,
    )
