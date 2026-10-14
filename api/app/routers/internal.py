import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db
from ..models.db import CreditTransaction, Job, Log, Metric, User
from ..schemas.jobs import (
    CheckpointRequest,
    CompleteRequest,
    EvaluationRequest,
    HeartbeatRequest,
    LogsBatch,
    MetricsRequest,
)
from ..services import secrets as secrets_service
from ..services.cost import MARKUP
from ..services.stripe_service import charge_overage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["internal"])


async def _get_job(
    db: AsyncSession,
    job_id: uuid.UUID,
    token: str,
) -> Job:
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")
    presented_hash = hashlib.sha256(token.encode()).hexdigest()
    if not hmac.compare_digest(presented_hash, job.internal_token):
        raise HTTPException(status_code=401, detail="Invalid internal token")
    return job


@router.post("/{job_id}/heartbeat", status_code=204)
async def heartbeat(
    job_id: uuid.UUID,
    req: HeartbeatRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job(db, job_id, x_internal_token)
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot heartbeat a job in '{job.status}' state",
        )
    job.last_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None)
    if job.started_at is None:
        job.status = "running"
        job.instance_id = req.instance_id
        job.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    elif job.status == "checkpointing":
        job.status = "running"
    await db.commit()


@router.post("/{job_id}/logs", status_code=204)
async def append_logs(
    job_id: uuid.UUID,
    req: LogsBatch,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    await _get_job(db, job_id, x_internal_token)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db.add_all([Log(job_id=job_id, line=line, timestamp=now) for line in req.lines])
    await db.commit()


@router.post("/{job_id}/metrics", status_code=204)
async def append_metrics(
    job_id: uuid.UUID,
    req: MetricsRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    await _get_job(db, job_id, x_internal_token)
    db.add_all(
        [
            Metric(
                job_id=job_id,
                step=m.step,
                epoch=m.epoch,
                train_loss=m.train_loss,
                learning_rate=m.learning_rate,
                gpu_utilization=m.gpu_utilization,
                gpu_memory_used_mb=m.gpu_memory_used_mb,
                tokens_per_second=m.tokens_per_second,
            )
            for m in req.metrics
        ]
    )
    await db.commit()


@router.patch("/{job_id}/checkpoint", status_code=204)
async def record_checkpoint(
    job_id: uuid.UUID,
    req: CheckpointRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job(db, job_id, x_internal_token)
    job.last_checkpoint_s3_path = req.checkpoint_s3_path
    job.last_checkpoint_step = req.step
    job.checkpointed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    job.status = "checkpointing"
    await db.commit()


@router.post("/{job_id}/evaluation", status_code=204)
async def record_evaluation(
    job_id: uuid.UUID,
    req: EvaluationRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    """The before/after comparison the trainer runs once training is done."""
    job = await _get_job(db, job_id, x_internal_token)
    job.evaluation = req.report
    await db.commit()


@router.post("/{job_id}/complete", status_code=204)
async def complete_job(
    job_id: uuid.UUID,
    req: CompleteRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job(db, job_id, x_internal_token)
    if job.status in ("completed", "failed", "cancelled"):
        return  # idempotent — training instance may retry; cancelled jobs are terminal
    job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    if req.error_message:
        job.status = "failed"
        job.error_message = req.error_message
    else:
        job.status = "completed"
        job.model_s3_path = req.model_s3_path
        if req.modelfile:
            job.modelfile = req.modelfile
        if req.actual_cost is not None:
            job.actual_cost = (Decimal(str(req.actual_cost)) * MARKUP).quantize(Decimal("0.01"))

    # Settle reservation: refund or charge the difference between estimated and actual cost
    estimated_cents = int(float(job.estimated_cost or 0) * 100)
    actual_cents = int(float(job.actual_cost or 0) * 100)
    diff = estimated_cents - actual_cents  # positive = refund unused, negative = overage

    db_user = None
    if estimated_cents > 0 and diff != 0:
        user_result = await db.execute(
            select(User).where(User.id == job.session_id).with_for_update()
        )
        db_user = user_result.scalar_one_or_none()
        if db_user:
            db_user.credit_balance_cents = db_user.credit_balance_cents + diff
            label = "refund" if diff > 0 else "overage"
            db.add(CreditTransaction(
                user_id=job.session_id,
                amount_cents=diff,
                description=f"Job settled — {label}",
                job_id=job.id,
            ))

    await db.commit()

    # If the balance went negative, immediately charge the saved card for the difference
    if db_user is not None and db_user.credit_balance_cents < 0:
        overage_cents = abs(db_user.credit_balance_cents)
        if db_user.stripe_customer_id and db_user.stripe_payment_method_id:
            try:
                charge_overage(
                    db_user.stripe_customer_id,
                    db_user.stripe_payment_method_id,
                    overage_cents,
                    str(job.id),
                )
                db_user.credit_balance_cents = 0
                db.add(CreditTransaction(
                    user_id=job.session_id,
                    amount_cents=overage_cents,
                    description="Overage charge — balance recovered",
                    job_id=job.id,
                ))
                await db.commit()
            except Exception:
                logger.exception(
                    "Overage charge failed for user %s job %s — balance remains negative",
                    job.session_id, job.id,
                )

    if job.hf_token_secret_arn:
        secrets_service.delete_secret(job.hf_token_secret_arn)
