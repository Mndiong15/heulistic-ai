import uuid

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import async_session_maker
from .models.db import User
from .services import api_keys


async def get_db():
    async with async_session_maker() as session:
        yield session


def _enabled(user: User) -> User:
    if user.disabled_at is not None:
        raise HTTPException(status_code=403, detail="This account is disabled")
    return user


async def get_current_user(
    authorization: str = Header(alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")

    # An API key, from `heulistic login`. Checked first so that local mode,
    # which takes any other token as a user id, still maps it to its owner.
    if api_keys.is_api_key(token):
        user = await api_keys.user_for_key(db, token)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid, revoked or expired API key")
        return _enabled(user)

    if settings.local_mode and not token.startswith("eyJ"):
        user_id = token.replace("dev-", "")
        email = f"{user_id[:8]}@local.dev"
    else:
        from .services.cognito import verify_token
        try:
            claims = await verify_token(token)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        user_id = claims["sub"]
        email = claims.get("email", "")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        user = User(id=user_id, email=email)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return _enabled(user)


async def get_api_key_id(
    authorization: str = Header(alias="Authorization", default=""),
    db: AsyncSession = Depends(get_db),
) -> uuid.UUID | None:
    """The API key behind this request, or None for a website login."""
    token = authorization.removeprefix("Bearer ")
    return await api_keys.key_id(db, token) if api_keys.is_api_key(token) else None


async def get_signed_in_user(
    authorization: str = Header(alias="Authorization", default=""),
    user: User = Depends(get_current_user),
) -> User:
    """A person at a login, not an API key. Key management and admin use this,
    so that revoking a leaked key ends the leak: it cannot mint a replacement,
    revoke other keys, or reach the admin endpoints."""
    if api_keys.is_api_key(authorization.removeprefix("Bearer ")):
        raise HTTPException(status_code=403, detail="API keys cannot do this; sign in on the website")
    return user


async def get_admin_user(user: User = Depends(get_signed_in_user)) -> User:
    if not settings.is_admin(user.email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
