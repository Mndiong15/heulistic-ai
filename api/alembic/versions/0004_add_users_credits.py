"""Add users and credit_transactions tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-15
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) NOT NULL,
            email VARCHAR(255) NOT NULL,
            credit_balance_cents INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT users_pkey PRIMARY KEY (id),
            CONSTRAINT uq_users_email UNIQUE (email)
        )
    """))

    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS credit_transactions (
            id BIGSERIAL NOT NULL,
            user_id VARCHAR(36) NOT NULL,
            amount_cents INTEGER NOT NULL,
            description VARCHAR(255),
            stripe_payment_intent_id VARCHAR(255),
            job_id UUID,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT credit_transactions_pkey PRIMARY KEY (id),
            CONSTRAINT credit_transactions_user_id_fkey
                FOREIGN KEY (user_id) REFERENCES users(id),
            CONSTRAINT credit_transactions_job_id_fkey
                FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    """))

    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_credit_transactions_user_id
        ON credit_transactions (user_id)
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS credit_transactions"))
    op.execute(sa.text("DROP TABLE IF EXISTS users"))
