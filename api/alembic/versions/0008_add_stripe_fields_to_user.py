"""add stripe_customer_id and stripe_subscription_id to users

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-21
"""
import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("stripe_customer_id", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("stripe_subscription_id", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "stripe_subscription_id")
    op.drop_column("users", "stripe_customer_id")
