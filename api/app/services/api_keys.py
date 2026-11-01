"""API keys: creating them and checking them.

A key is `hk_` followed by 32 random bytes. Only its SHA-256 is stored, so a
database leak does not leak keys — the same reasoning as the per-job internal
tokens. A full 256-bit key needs no slow hash: it cannot be guessed.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db import ApiKey, User

PREFIX = "hk_"
_SHOWN_CHARS = 10  # "hk_" plus 7 — enough to tell keys apart in a list
# Refreshing last_used_at on every request would write to the database on
# every request; once a minute is plenty to see whether a key is in use.
_USE_RECORD_INTERVAL = timedelta(minutes=1)


def is_api_key(token: str) -> bool:
    return token.startswith(PREFIX)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> tuple[str, str, str]:
    """A fresh key: (key, its hash, its displayable prefix)."""
    key = PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key), key[:_SHOWN_CHARS]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def user_for_key(db: AsyncSession, key: str) -> User | None:
    """The owner of an active key, or None."""
    row = (
        await db.execute(
            select(ApiKey).where(ApiKey.key_hash == hash_key(key), ApiKey.revoked_at.is_(None))
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    now = _now()
    if row.last_used_at is None or now - row.last_used_at > _USE_RECORD_INTERVAL:
        row.last_used_at = now
        await db.commit()
    return await db.get(User, row.user_id)
