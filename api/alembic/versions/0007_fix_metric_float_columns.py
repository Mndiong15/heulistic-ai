"""fix metric epoch and tokens_per_second to float

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-20
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("metrics", "epoch",
                    type_=sa.Float(),
                    existing_type=sa.Integer(),
                    existing_nullable=True)
    op.alter_column("metrics", "tokens_per_second",
                    type_=sa.Float(),
                    existing_type=sa.Integer(),
                    existing_nullable=True)


def downgrade() -> None:
    op.alter_column("metrics", "epoch",
                    type_=sa.Integer(),
                    existing_type=sa.Float(),
                    existing_nullable=True)
    op.alter_column("metrics", "tokens_per_second",
                    type_=sa.Integer(),
                    existing_type=sa.Float(),
                    existing_nullable=True)
