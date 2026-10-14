import asyncio
import hashlib
import os
import uuid

# Must be set before any app module is imported so pydantic-settings picks them up.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://heulistic_admin:password@localhost:5432/heulistic")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("ADMIN_EMAILS", "admin@test.dev")
# Match CI. With LOCAL_MODE=true the settings validator disables billing and
# every /api/credits test gets a 404. Cognito is not reached either way, since
# `get_current_user` is overridden below, and AWS is kept out by the offline
# instance-catalogue fixture.
os.environ.setdefault("LOCAL_MODE", "false")

import pytest
import pytest_asyncio
from fastapi import Depends, Header, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.dependencies import get_current_user, get_db
from app.main import app
from app.models.db import User

# ── Constants ─────────────────────────────────────────────────────────────────

TEST_DB_URL = os.environ["DATABASE_URL"]
_DB_AVAILABLE = False  # set to True below if Postgres is reachable
SESSION_ID = str(uuid.uuid4())
OTHER_SESSION_ID = str(uuid.uuid4())
ADMIN_ID = str(uuid.uuid4())
ADMIN_EMAIL = "admin@test.dev"

# Token used by test jobs — plaintext sent in X-Internal-Token, hash stored in DB.
INTERNAL_TEST_TOKEN = "heulistic-internal-test-token-1234"
INTERNAL_TEST_TOKEN_HASH = hashlib.sha256(INTERNAL_TEST_TOKEN.encode()).hexdigest()

VALID_CONFIG = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
adapter: lora
lora_r: 16
num_epochs: 1
micro_batch_size: 2
learning_rate: 2e-4
datasets:
  - path: tatsu-lab/alpaca
"""

# ── DB engine (reused across tests) ──────────────────────────────────────────

# _engine is declared but never touched outside pytest-asyncio's event loop so
# asyncpg connections are always bound to the correct loop.
_engine = create_async_engine(TEST_DB_URL, echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def _truncate_tables() -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE metrics, logs, credit_transactions, jobs, users RESTART IDENTITY CASCADE")
        )


async def _create_tables_isolated() -> None:
    """Create tables using a short-lived engine so the pool isn't bound to asyncio.run()'s loop."""
    eng = create_async_engine(TEST_DB_URL, echo=False)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await eng.dispose()


# Run table creation before pytest-asyncio sets up its event loop.
# Fails gracefully when Postgres isn't up — DB-dependent tests will be skipped,
# but pure unit tests (test_cost.py, test_instances.py) still run.
try:
    asyncio.run(_create_tables_isolated())
    _DB_AVAILABLE = True
except Exception as _db_exc:
    import warnings
    warnings.warn(
        f"PostgreSQL not reachable ({_db_exc}). "
        "Start the stack with `heulistic start` or `docker compose up -d db` "
        "to run integration tests. Pure unit tests will still run.",
        stacklevel=1,
    )


# ── Session-scoped event loop ─────────────────────────────────────────────────
# asyncpg connections are bound to the event loop that opened them.
# Pinning all tests to one loop keeps the _engine pool valid across tests.


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Per-test cleanup ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def clean_tables():
    if _DB_AVAILABLE:
        await _truncate_tables()
    yield


# ── Auth bypass ───────────────────────────────────────────────────────────────
# Accept any UUID Bearer token as a direct user_id — no Cognito calls in tests.


@pytest.fixture(autouse=True)
def _override_auth():
    async def _test_auth(
        authorization: str = Header(alias="Authorization", default=""),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        user_id = authorization.removeprefix("Bearer ").strip()
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    app.dependency_overrides[get_current_user] = _test_auth
    yield
    app.dependency_overrides.pop(get_current_user, None)


# ── DB session fixture ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    if not _DB_AVAILABLE:
        pytest.skip("PostgreSQL not reachable — run `docker compose up -d db` first")
    async with _session_factory() as session:
        yield session


# ── HTTP clients ──────────────────────────────────────────────────────────────


def _make_db_override(session: AsyncSession):
    async def _override():
        yield session
    return _override


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncClient:
    """Test client scoped to SESSION_ID with pre-seeded credits."""
    db_session.add(User(id=SESSION_ID, email="test@local.dev", credit_balance_cents=1_000_000))
    await db_session.commit()
    app.dependency_overrides[get_db] = _make_db_override(db_session)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={
            "Authorization": f"Bearer {SESSION_ID}",
            "X-Internal-Token": INTERNAL_TEST_TOKEN,
        },
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def other_client(db_session: AsyncSession) -> AsyncClient:
    """Second client with a different user (for ownership / isolation tests)."""
    db_session.add(User(id=OTHER_SESSION_ID, email="other@local.dev", credit_balance_cents=1_000_000))
    await db_session.commit()
    app.dependency_overrides[get_db] = _make_db_override(db_session)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {OTHER_SESSION_ID}"},
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def admin_client(db_session: AsyncSession) -> AsyncClient:
    """Client authenticated as the admin user."""
    db_session.add(User(id=ADMIN_ID, email=ADMIN_EMAIL, credit_balance_cents=0))
    await db_session.commit()
    app.dependency_overrides[get_db] = _make_db_override(db_session)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {ADMIN_ID}"},
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def anon_client(db_session: AsyncSession) -> AsyncClient:
    """Client with no X-Session-ID header (DB still mocked for safety)."""
    app.dependency_overrides[get_db] = _make_db_override(db_session)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def authed_client(db_session: AsyncSession) -> AsyncClient:
    """Client with Authorization: Bearer header and pre-seeded user (0 credits)."""
    db_session.add(User(id=SESSION_ID, email="test@local.dev", credit_balance_cents=0))
    await db_session.commit()
    app.dependency_overrides[get_db] = _make_db_override(db_session)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {SESSION_ID}"},
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


# ── Stripe webhook mock (applied to every test) ──────────────────────────────
# Bypass signature verification so webhook tests can send plain JSON.


@pytest.fixture(autouse=True)
def mock_stripe_webhook(monkeypatch):
    import json as _json

    monkeypatch.setattr(
        "app.routers.webhooks.construct_webhook_event",
        lambda payload, sig_header: _json.loads(payload),
    )


# ── Instance catalogue (applied to every test) ────────────────────────────────


@pytest.fixture(autouse=True)
def offline_instance_catalog(monkeypatch):
    """Pin the GPU catalogue to the hardcoded fallback.

    `get_catalog()` otherwise calls ec2:DescribeInstanceTypes and the AWS
    Pricing API. On a developer machine that happens to hold AWS credentials
    that succeeds, returning live prices — which changes which instance is
    cheapest for a given VRAM figure and makes instance-selection assertions
    fail locally while passing in CI. Forcing the fetch to fail makes the
    catalogue identical everywhere, and keeps the unit tests offline.
    """
    from app.services import instances

    def _offline(*args, **kwargs):
        raise RuntimeError("AWS is not reachable from tests")

    monkeypatch.setattr(instances, "_fetch_gpu_specs", _offline)
    monkeypatch.setattr(instances, "_fetch_on_demand_prices", _offline)
    instances.get_catalog.cache_clear()
    yield
    instances.get_catalog.cache_clear()


# ── S3 mock (applied to every test) ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_s3(monkeypatch):
    monkeypatch.setattr(
        "app.services.s3.upload_bytes",
        lambda data, key, ct="application/octet-stream": f"s3://test-bucket/{key}",
    )
    monkeypatch.setattr(
        "app.services.s3.upload_fileobj",
        lambda fobj, key: f"s3://test-bucket/{key}",
    )
    monkeypatch.setattr(
        "app.services.s3.generate_presigned_url",
        lambda key, expires_in=3600: f"https://test-bucket.s3.amazonaws.com/{key}?X-Amz-Expires={expires_in}",
    )
    monkeypatch.setattr(
        "app.services.s3.s3_key_from_path",
        lambda path: path.split("/", 3)[-1],
    )


# ── HuggingFace mock (applied to every test) ─────────────────────────────────
# Default: all HF API calls succeed (200). Tests in TestHFTokenValidation
# override this per-test via their own monkeypatch.setattr calls.


class _FakeHFResponse:
    status_code = 200


class _DefaultFakeHFClient:
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, url, **kwargs): return _FakeHFResponse()
    async def head(self, url, **kwargs): return _FakeHFResponse()


@pytest.fixture(autouse=True)
def mock_hf_client(monkeypatch):
    monkeypatch.setattr(
        "app.routers.jobs.httpx.AsyncClient",
        lambda **kw: _DefaultFakeHFClient(),
    )


# ── Convenience helper ────────────────────────────────────────────────────────


async def submit_job(client: AsyncClient, config: str = VALID_CONFIG) -> dict:
    """POST /api/jobs and return parsed JSON. Asserts 201."""
    resp = await client.post(
        "/api/jobs",
        files={"config": ("config.yaml", config.encode(), "application/x-yaml")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
