"""Add queue position, checkpoint fields, and provisioned/checkpointed timestamps

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-15
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("provisioned_at", sa.DateTime(), nullable=True))
    op.add_column("jobs", sa.Column("checkpointed_at", sa.DateTime(), nullable=True))
    op.add_column("jobs", sa.Column("queue_position", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("last_checkpoint_s3_path", sa.String(500), nullable=True))
    op.add_column("jobs", sa.Column("last_checkpoint_step", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("resume_from_checkpoint_path", sa.String(500), nullable=True))


def downgrade():
    op.drop_column("jobs", "resume_from_checkpoint_path")
    op.drop_column("jobs", "last_checkpoint_step")
    op.drop_column("jobs", "last_checkpoint_s3_path")
    op.drop_column("jobs", "queue_position")
    op.drop_column("jobs", "checkpointed_at")
    op.drop_column("jobs", "provisioned_at")
