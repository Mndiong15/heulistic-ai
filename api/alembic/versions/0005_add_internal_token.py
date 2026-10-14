"""add internal_token to jobs

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-17
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS internal_token VARCHAR(64)"
    ))


def downgrade() -> None:
    op.execute(sa.text(
        "ALTER TABLE jobs DROP COLUMN IF EXISTS internal_token"
    ))
