"""add hf_token_secret_arn to jobs

Revision ID: 0002
Down revision: 0001
Create Date: 2026-05-14
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("hf_token_secret_arn", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "hf_token_secret_arn")
