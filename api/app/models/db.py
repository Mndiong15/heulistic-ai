import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # Cognito sub
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    credit_balance_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_payment_method_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payment_past_due: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")


class CreditTransaction(Base):
    __tablename__ = "credit_transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)  # positive=add, negative=deduct
    description: Mapped[str | None] = mapped_column(String(255))
    stripe_payment_intent_id: Mapped[str | None] = mapped_column(String(255))
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("idx_jobs_last_heartbeat", "status", "last_heartbeat"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    config_yaml: Mapped[str] = mapped_column(Text, nullable=False)
    config_name: Mapped[str | None] = mapped_column(String(255))
    config_s3_path: Mapped[str | None] = mapped_column(String(500))
    dataset_s3_path: Mapped[str | None] = mapped_column(String(500))
    dataset_size_mb: Mapped[int | None] = mapped_column(Integer)
    model_s3_path: Mapped[str | None] = mapped_column(String(500))
    instance_id: Mapped[str | None] = mapped_column(String(50))
    instance_type: Mapped[str | None] = mapped_column(String(20))
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    actual_cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    estimated_duration_minutes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime)
    error_message: Mapped[str | None] = mapped_column(Text)
    hf_token_secret_arn: Mapped[str | None] = mapped_column(String(500))
    provisioned_at: Mapped[datetime | None] = mapped_column(DateTime)
    checkpointed_at: Mapped[datetime | None] = mapped_column(DateTime)
    queue_position: Mapped[int | None] = mapped_column(Integer)
    last_checkpoint_s3_path: Mapped[str | None] = mapped_column(String(500))
    last_checkpoint_step: Mapped[int | None] = mapped_column(Integer)
    resume_from_checkpoint_path: Mapped[str | None] = mapped_column(String(500))
    internal_token: Mapped[str | None] = mapped_column(String(64))

    # "train", or "export": turning a finished training job's model into a
    # GGUF for Ollama. An export points back at the job it exports.
    kind: Mapped[str] = mapped_column(String(10), nullable=False, default="train", server_default="train")
    source_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=True, index=True
    )
    export_quant: Mapped[str | None] = mapped_column(String(16))
    modelfile: Mapped[str | None] = mapped_column(Text)
    # Root volume for the instance; None means the provisioner's default.
    disk_gb: Mapped[int | None] = mapped_column(Integer)
    # The before/after report written at the end of training.
    evaluation: Mapped[dict | None] = mapped_column(JSONB)

    metrics: Mapped[list["Metric"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    logs: Mapped[list["Log"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class SchedulerHeartbeat(Base):
    """Singleton row (id=1) written by the scheduler on every tick."""
    __tablename__ = "scheduler_heartbeat"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    ticked_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    pending_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dispatched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[float | None] = mapped_column(Float)
    train_loss: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    learning_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 8))
    gpu_utilization: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    gpu_memory_used_mb: Mapped[int | None] = mapped_column(Integer)
    tokens_per_second: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)

    job: Mapped["Job"] = relationship(back_populates="metrics")


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    line: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    job: Mapped["Job"] = relationship(back_populates="logs")
