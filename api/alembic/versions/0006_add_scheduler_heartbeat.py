"""add scheduler_heartbeat table

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-18
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS scheduler_heartbeat (
            id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            ticked_at       TIMESTAMP NOT NULL,
            pending_count   INTEGER NOT NULL DEFAULT 0,
            active_count    INTEGER NOT NULL DEFAULT 0,
            dispatched_count INTEGER NOT NULL DEFAULT 0
        )
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS scheduler_heartbeat"))
