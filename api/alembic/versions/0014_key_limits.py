"""API key expiry and spend tracking, and disabling accounts

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("expires_at", sa.DateTime(), nullable=True))
    op.execute("UPDATE api_keys SET expires_at = created_at + INTERVAL '365 days'")
    op.alter_column("api_keys", "expires_at", nullable=False)

    op.add_column("jobs", sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_jobs_api_key_id", "jobs", "api_keys", ["api_key_id"], ["id"])
    op.create_index("ix_jobs_api_key_id", "jobs", ["api_key_id"])

    op.add_column("users", sa.Column("disabled_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "disabled_at")
    op.drop_index("ix_jobs_api_key_id", table_name="jobs")
    op.drop_constraint("fk_jobs_api_key_id", "jobs", type_="foreignkey")
    op.drop_column("jobs", "api_key_id")
    op.drop_column("api_keys", "expires_at")
