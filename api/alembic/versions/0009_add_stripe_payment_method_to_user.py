"""add stripe_payment_method_id to users

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("stripe_payment_method_id", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "stripe_payment_method_id")
