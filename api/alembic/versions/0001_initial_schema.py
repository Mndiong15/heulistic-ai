"""initial schema

Revision ID: 0001
Create Date: 2024-01-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("config_yaml", sa.Text, nullable=False),
        sa.Column("config_name", sa.String(255)),
        sa.Column("config_s3_path", sa.String(500)),
        sa.Column("dataset_s3_path", sa.String(500)),
        sa.Column("dataset_size_mb", sa.Integer),
        sa.Column("model_s3_path", sa.String(500)),
        sa.Column("instance_id", sa.String(50)),
        sa.Column("instance_type", sa.String(20)),
        sa.Column("estimated_cost", sa.Numeric(10, 2)),
        sa.Column("actual_cost", sa.Numeric(10, 2)),
        sa.Column("estimated_duration_minutes", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("cancelled_at", sa.DateTime),
        sa.Column("last_heartbeat", sa.DateTime),
        sa.Column("error_message", sa.Text),
    )
    op.create_index("idx_jobs_session_id", "jobs", ["session_id"])
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_created_at", "jobs", ["created_at"])
    op.create_index("idx_jobs_last_heartbeat", "jobs", ["status", "last_heartbeat"])

    op.create_table(
        "metrics",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer, nullable=False),
        sa.Column("epoch", sa.Integer),
        sa.Column("train_loss", sa.Numeric(10, 6)),
        sa.Column("learning_rate", sa.Numeric(10, 8)),
        sa.Column("gpu_utilization", sa.Numeric(5, 2)),
        sa.Column("gpu_memory_used_mb", sa.Integer),
        sa.Column("tokens_per_second", sa.Integer),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_metrics_job_id", "metrics", ["job_id"])
    op.create_index("idx_metrics_timestamp", "metrics", ["timestamp"])

    op.create_table(
        "logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("line", sa.Text, nullable=False),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_logs_job_id", "logs", ["job_id"])


def downgrade() -> None:
    op.drop_table("logs")
    op.drop_table("metrics")
    op.drop_table("jobs")
