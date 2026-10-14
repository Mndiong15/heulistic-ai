from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import async_session_maker
from .models.db import User


async def get_db():
    async with async_session_maker() as session:
        yield session


async def get_current_user(
    authorization: str = Header(alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")

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
    return user


async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    if not settings.is_admin(user.email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
