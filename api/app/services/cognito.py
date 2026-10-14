import time

import httpx
from jose import JWTError, jwt

from ..config import settings

_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 3600  # 1 hour


async def _get_jwks() -> dict:
    global _jwks_fetched_at
    if _jwks_cache and (time.time() - _jwks_fetched_at) < _JWKS_TTL:
        return _jwks_cache
    url = (
        f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/"
        f"{settings.cognito_user_pool_id}/.well-known/jwks.json"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        _jwks_cache.clear()
        _jwks_cache.update(resp.json())
        _jwks_fetched_at = time.time()
    return _jwks_cache


async def verify_token(token: str) -> dict:
    jwks = await _get_jwks()
    try:
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=settings.cognito_client_id,
        )
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc
    return claims  # contains 'sub', 'email', 'exp', etc.
