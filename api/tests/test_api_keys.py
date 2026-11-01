"""API keys — how the CLI signs in to Heulistic Cloud.

A user creates a key on the website, pastes it into `heulistic login`, and the
CLI sends it as a bearer token. Only a hash is stored; the key itself is shown
once. Keys work alongside Cognito logins, not instead of them.
"""

import hashlib

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user
from app.main import app
from app.models.db import ApiKey
from tests.conftest import VALID_CONFIG


@pytest.fixture
def real_auth():
    """Use the real get_current_user, which is where keys are checked — the
    autouse test override would otherwise accept any user id as a token."""
    saved = app.dependency_overrides.pop(get_current_user, None)
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved


@pytest_asyncio.fixture
async def with_key(db_session):
    """A client that authenticates with nothing but an API key."""
    clients = []

    async def make(key: str) -> AsyncClient:
        c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                        headers={"Authorization": f"Bearer {key}"})
        clients.append(c)
        return c

    yield make
    for c in clients:
        await c.aclose()


async def _create(client: AsyncClient, name: str = "laptop") -> dict:
    resp = await client.post("/api/keys", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── Creating and listing ──────────────────────────────────────────────────────


class TestKeys:
    async def test_the_key_is_shown_once_at_creation(self, client: AsyncClient):
        created = await _create(client)
        assert created["key"].startswith("hk_")
        assert len(created["key"]) > 40
        assert created["key"].startswith(created["prefix"])

        listed = (await client.get("/api/keys")).json()
        assert [k["id"] for k in listed] == [created["id"]]
        assert "key" not in listed[0]
        assert listed[0]["name"] == "laptop"

    async def test_only_a_hash_is_stored(self, client: AsyncClient, db_session: AsyncSession):
        created = await _create(client)
        row = (await db_session.execute(select(ApiKey))).scalar_one()
        assert row.key_hash == hashlib.sha256(created["key"].encode()).hexdigest()
        assert created["key"] not in (row.key_hash, row.prefix)

    async def test_a_name_is_required(self, client: AsyncClient):
        assert (await client.post("/api/keys", json={"name": ""})).status_code == 422

    async def test_you_see_only_your_own_keys(self, client: AsyncClient, other_client: AsyncClient):
        await _create(client)
        assert (await other_client.get("/api/keys")).json() == []

    async def test_you_cannot_revoke_someone_elses_key(
        self, client: AsyncClient, other_client: AsyncClient
    ):
        created = await _create(client)
        assert (await other_client.delete(f"/api/keys/{created['id']}")).status_code == 404

    async def test_revoked_keys_leave_the_list(self, client: AsyncClient):
        created = await _create(client)
        assert (await client.delete(f"/api/keys/{created['id']}")).status_code == 204
        assert (await client.get("/api/keys")).json() == []


# ── Signing in with a key ─────────────────────────────────────────────────────
# These run with real authentication, which the normal test client cannot get
# through — so their keys go straight into the database, made the same way the
# endpoint makes them.


async def _stored_key(db_session: AsyncSession, client: AsyncClient, revoked: bool = False) -> str:
    from datetime import datetime
    from app.services import api_keys
    from tests.conftest import SESSION_ID
    key, key_hash, prefix = api_keys.new_key()
    db_session.add(ApiKey(user_id=SESSION_ID, name="cli", key_hash=key_hash, prefix=prefix,
                          revoked_at=datetime(2026, 1, 1) if revoked else None))
    await db_session.commit()
    return key


class TestSigningInWithAKey:
    async def test_me_says_who_the_key_belongs_to(self, client, real_auth, with_key, db_session):
        key = await _stored_key(db_session, client)
        me = await (await with_key(key)).get("/api/me")
        assert me.status_code == 200
        assert me.json()["email"] == "test@local.dev"
        assert me.json()["credit_balance_cents"] == 1_000_000

    async def test_a_key_can_submit_a_job(self, client, real_auth, with_key, db_session):
        key = await _stored_key(db_session, client)
        resp = await (await with_key(key)).post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201, resp.text

    async def test_a_revoked_key_is_refused(self, client, real_auth, with_key, db_session):
        key = await _stored_key(db_session, client, revoked=True)
        assert (await (await with_key(key)).get("/api/me")).status_code == 401

    async def test_an_unknown_key_is_refused(self, client, real_auth, with_key):
        resp = await (await with_key("hk_not-a-real-key")).get("/api/me")
        assert resp.status_code == 401

    async def test_use_is_recorded(self, client, real_auth, with_key, db_session: AsyncSession):
        key = await _stored_key(db_session, client)
        await (await with_key(key)).get("/api/me")
        row = (await db_session.execute(select(ApiKey))).scalar_one()
        await db_session.refresh(row)
        assert row.last_used_at is not None

    async def test_keys_work_in_local_mode_too(self, client, real_auth, with_key, monkeypatch, db_session):
        # Local mode accepts any non-JWT token as a user id; a key must still
        # resolve to its owner rather than becoming a user named "hk_…".
        from app.config import settings
        monkeypatch.setattr(settings, "local_mode", True)
        key = await _stored_key(db_session, client)
        me = await (await with_key(key)).get("/api/me")
        assert me.json()["email"] == "test@local.dev"
