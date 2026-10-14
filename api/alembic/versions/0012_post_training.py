"""Post-training: export jobs and the before/after evaluation report

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-12
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("kind", sa.String(10), nullable=False, server_default="train"))
    op.add_column("jobs", sa.Column("source_job_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_jobs_source_job_id", "jobs", "jobs", ["source_job_id"], ["id"])
    op.create_index("ix_jobs_source_job_id", "jobs", ["source_job_id"])
    op.add_column("jobs", sa.Column("export_quant", sa.String(16), nullable=True))
    op.add_column("jobs", sa.Column("modelfile", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("disk_gb", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("evaluation", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "evaluation")
    op.drop_column("jobs", "disk_gb")
    op.drop_column("jobs", "modelfile")
    op.drop_column("jobs", "export_quant")
    op.drop_index("ix_jobs_source_job_id", table_name="jobs")
    op.drop_constraint("fk_jobs_source_job_id", "jobs", type_="foreignkey")
    op.drop_column("jobs", "source_job_id")
    op.drop_column("jobs", "kind")
