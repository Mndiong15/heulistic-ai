"""API keys for the CLI (`heulistic login`), and who the caller is.

Created on the website, shown once, and revocable; keys themselves cannot manage
keys. `GET /api/me` lets the CLI confirm a key and say whose it is.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..dependencies import get_current_user, get_db, get_signed_in_user
from ..models.db import ApiKey, User
from ..schemas.api_keys import ApiKeyInfo, CreatedKey, CreateKeyRequest, Me
from ..services import api_keys

router = APIRouter(tags=["api-keys"])


def _info(row: ApiKey) -> dict:
    return {
        "id": row.id, "name": row.name, "prefix": row.prefix,
        "created_at": row.created_at, "last_used_at": row.last_used_at, "expires_at": row.expires_at,
    }


@router.get("/me", response_model=Me)
async def me(user: User = Depends(get_current_user)):
    return Me(
        email=user.email,
        credit_balance_cents=user.credit_balance_cents,
        billing_enabled=settings.billing_enabled,
    )


@router.post("/keys", response_model=CreatedKey, status_code=201)
async def create_key(
    req: CreateKeyRequest,
    user: User = Depends(get_signed_in_user),
    db: AsyncSession = Depends(get_db),
):
    key, key_hash, prefix = api_keys.new_key()
    row = ApiKey(
        user_id=user.id, name=req.name.strip(), key_hash=key_hash, prefix=prefix,
        expires_at=api_keys.expiry(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return CreatedKey(**_info(row), key=key)


@router.get("/keys", response_model=list[ApiKeyInfo])
async def list_keys(user: User = Depends(get_signed_in_user), db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
        .order_by(ApiKey.created_at)
    )
    return [ApiKeyInfo(**_info(r)) for r in rows.scalars()]


@router.delete("/keys/{key_id}", status_code=204)
async def revoke_key(
    key_id: uuid.UUID,
    user: User = Depends(get_signed_in_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.get(ApiKey, key_id)
    # Someone else's key and no key at all look the same: no hint either exists.
    if row is None or row.user_id != user.id or row.revoked_at is not None:
        raise HTTPException(status_code=404, detail="API key not found")
    row.revoked_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    return Response(status_code=204)
