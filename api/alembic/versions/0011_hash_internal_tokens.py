"""Invalidate plaintext internal tokens — column now stores SHA-256 hashes

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing rows stored the plaintext token. Null them out — the scheduler
    # now generates tokens at provision time and stores only SHA-256(token).
    # Any in-flight jobs will fail their next heartbeat and be cleaned up by the watchdog.
    op.execute(sa.text("UPDATE jobs SET internal_token = NULL WHERE internal_token IS NOT NULL"))


def downgrade() -> None:
    pass  # Plaintext originals are unrecoverable; downgrade is a no-op
