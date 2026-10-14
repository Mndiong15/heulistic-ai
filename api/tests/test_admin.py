"""Tests for admin endpoints (GET /api/admin/status|jobs|users, POST /api/admin/users/:id/credits)."""

import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import CreditTransaction, Job
from tests.conftest import ADMIN_EMAIL, ADMIN_ID, SESSION_ID, VALID_CONFIG, submit_job


# ── GET /api/admin/status ─────────────────────────────────────────────────────


class TestAdminStatus:
    async def test_admin_returns_200(self, admin_client: AsyncClient):
        resp = await admin_client.get("/api/admin/status")
        assert resp.status_code == 200
        assert resp.json() == {"is_admin": True}

    async def test_non_admin_returns_403(self, client: AsyncClient):
        resp = await client.get("/api/admin/status")
        assert resp.status_code == 403

    async def test_unauthenticated_returns_401(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/admin/status")
        assert resp.status_code == 401


# ── GET /api/admin/jobs ───────────────────────────────────────────────────────


class TestAdminListJobs:
    async def test_returns_all_users_jobs(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        # Two jobs from different users
        await submit_job(client)
        db_session.add(Job(
            id=uuid.uuid4(),
            session_id=ADMIN_ID,
            status="completed",
            config_yaml=VALID_CONFIG,
            config_name="admin-job",
        ))
        await db_session.commit()

        resp = await admin_client.get("/api/admin/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 2
        user_ids = {j["user_id"] for j in jobs}
        assert SESSION_ID in user_ids
        assert ADMIN_ID in user_ids

    async def test_response_includes_user_email(
        self, admin_client: AsyncClient, client: AsyncClient
    ):
        await submit_job(client)
        resp = await admin_client.get("/api/admin/jobs")
        assert resp.status_code == 200
        job = resp.json()[0]
        assert "user_email" in job
        assert job["user_email"] == "test@local.dev"

    async def test_status_filter(
        self, admin_client: AsyncClient, db_session: AsyncSession
    ):
        db_session.add(Job(
            id=uuid.uuid4(), session_id=ADMIN_ID, status="running", config_yaml=VALID_CONFIG
        ))
        db_session.add(Job(
            id=uuid.uuid4(), session_id=ADMIN_ID, status="completed", config_yaml=VALID_CONFIG
        ))
        await db_session.commit()

        resp = await admin_client.get("/api/admin/jobs?status=running")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "running"

    async def test_user_id_filter(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        await submit_job(client)
        db_session.add(Job(
            id=uuid.uuid4(), session_id=ADMIN_ID, status="pending", config_yaml=VALID_CONFIG
        ))
        await db_session.commit()

        resp = await admin_client.get(f"/api/admin/jobs?user_id={SESSION_ID}")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1
        assert jobs[0]["user_id"] == SESSION_ID

    async def test_non_admin_returns_403(self, client: AsyncClient):
        resp = await client.get("/api/admin/jobs")
        assert resp.status_code == 403

    async def test_empty_when_no_jobs(self, admin_client: AsyncClient):
        resp = await admin_client.get("/api/admin/jobs")
        assert resp.status_code == 200
        assert resp.json() == []


# ── GET /api/admin/users ──────────────────────────────────────────────────────


class TestAdminListUsers:
    async def test_returns_all_users(
        self, admin_client: AsyncClient, client: AsyncClient
    ):
        resp = await admin_client.get("/api/admin/users")
        assert resp.status_code == 200
        users = resp.json()
        # admin_client fixture seeds the admin user; client fixture seeds the regular user
        assert len(users) == 2
        emails = {u["email"] for u in users}
        assert "test@local.dev" in emails
        assert ADMIN_EMAIL in emails

    async def test_job_count_is_accurate(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        await submit_job(client)
        await submit_job(client)

        resp = await admin_client.get("/api/admin/users")
        assert resp.status_code == 200
        users = {u["email"]: u for u in resp.json()}
        assert users["test@local.dev"]["job_count"] == 2
        assert users[ADMIN_EMAIL]["job_count"] == 0

    async def test_includes_credit_balance(self, admin_client: AsyncClient, client: AsyncClient):
        resp = await admin_client.get("/api/admin/users")
        assert resp.status_code == 200
        users = {u["email"]: u for u in resp.json()}
        assert users["test@local.dev"]["credit_balance_cents"] == 1_000_000

    async def test_non_admin_returns_403(self, client: AsyncClient):
        resp = await client.get("/api/admin/users")
        assert resp.status_code == 403


# ── POST /api/admin/users/:id/credits ────────────────────────────────────────


class TestAdminAdjustCredits:
    async def test_add_credits(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        resp = await admin_client.post(
            f"/api/admin/users/{SESSION_ID}/credits",
            json={"amount_cents": 500, "description": "courtesy credit"},
        )
        assert resp.status_code == 200
        assert resp.json()["new_balance_cents"] == 1_000_500

    async def test_remove_credits(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        resp = await admin_client.post(
            f"/api/admin/users/{SESSION_ID}/credits",
            json={"amount_cents": -200, "description": "correction"},
        )
        assert resp.status_code == 200
        assert resp.json()["new_balance_cents"] == 999_800

    async def test_balance_floors_at_zero(
        self, admin_client: AsyncClient, client: AsyncClient
    ):
        resp = await admin_client.post(
            f"/api/admin/users/{SESSION_ID}/credits",
            json={"amount_cents": -9_999_999, "description": "massive deduction"},
        )
        assert resp.status_code == 200
        assert resp.json()["new_balance_cents"] == 0

    async def test_creates_credit_transaction(
        self, admin_client: AsyncClient, client: AsyncClient, db_session: AsyncSession
    ):
        await admin_client.post(
            f"/api/admin/users/{SESSION_ID}/credits",
            json={"amount_cents": 100, "description": "test grant"},
        )
        txns = (await db_session.execute(
            select(CreditTransaction).where(CreditTransaction.user_id == SESSION_ID)
        )).scalars().all()
        assert len(txns) == 1
        assert txns[0].amount_cents == 100
        assert "test grant" in txns[0].description
        assert ADMIN_EMAIL in txns[0].description

    async def test_unknown_user_returns_404(self, admin_client: AsyncClient):
        resp = await admin_client.post(
            f"/api/admin/users/{uuid.uuid4()}/credits",
            json={"amount_cents": 100, "description": "ghost"},
        )
        assert resp.status_code == 404

    async def test_non_admin_returns_403(self, client: AsyncClient):
        resp = await client.post(
            f"/api/admin/users/{SESSION_ID}/credits",
            json={"amount_cents": 100, "description": "hack"},
        )
        assert resp.status_code == 403


# ── GET /api/admin/metrics ────────────────────────────────────────────────────


class TestAdminMetrics:
    async def test_non_admin_returns_403(self, client: AsyncClient):
        resp = await client.get("/api/admin/metrics")
        assert resp.status_code == 403

    async def test_returns_200_with_all_fields(self, admin_client: AsyncClient):
        resp = await admin_client.get("/api/admin/metrics")
        assert resp.status_code == 200
        data = resp.json()
        for field in (
            "total_jobs", "jobs_by_status", "total_users",
            "total_revenue_cents", "outstanding_credits_cents",
            "jobs_per_day", "revenue_per_day", "new_users_per_day",
        ):
            assert field in data, f"missing field: {field}"

    async def test_total_jobs_and_by_status(self, admin_client: AsyncClient, db_session):
        db_session.add(Job(id=uuid.uuid4(), session_id=ADMIN_ID, status="completed", config_yaml=VALID_CONFIG))
        db_session.add(Job(id=uuid.uuid4(), session_id=ADMIN_ID, status="pending", config_yaml=VALID_CONFIG))
        db_session.add(Job(id=uuid.uuid4(), session_id=ADMIN_ID, status="failed", config_yaml=VALID_CONFIG))
        await db_session.commit()

        resp = await admin_client.get("/api/admin/metrics")
        data = resp.json()
        assert data["total_jobs"] == 3
        assert data["jobs_by_status"]["completed"] == 1
        assert data["jobs_by_status"]["pending"] == 1
        assert data["jobs_by_status"]["failed"] == 1

    async def test_revenue_only_from_completed_jobs(self, admin_client: AsyncClient, db_session):
        from datetime import datetime, timezone
        from decimal import Decimal as D

        db_session.add(Job(
            id=uuid.uuid4(), session_id=ADMIN_ID, status="completed",
            config_yaml=VALID_CONFIG, actual_cost=D("3.00"), completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        ))
        # failed job with actual_cost should NOT be counted
        db_session.add(Job(
            id=uuid.uuid4(), session_id=ADMIN_ID, status="failed",
            config_yaml=VALID_CONFIG, actual_cost=D("1.00"), completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        ))
        await db_session.commit()

        resp = await admin_client.get("/api/admin/metrics")
        assert resp.json()["total_revenue_cents"] == 300

    async def test_total_users_count(self, admin_client: AsyncClient, client: AsyncClient):
        # admin_client seeds admin user, client seeds SESSION_ID user → 2 total
        resp = await admin_client.get("/api/admin/metrics")
        assert resp.json()["total_users"] == 2

    async def test_outstanding_credits_sums_all_balances(
        self, admin_client: AsyncClient, client: AsyncClient
    ):
        # admin user: 0 credits, SESSION_ID user: 1_000_000 cents
        resp = await admin_client.get("/api/admin/metrics")
        assert resp.json()["outstanding_credits_cents"] == 1_000_000

    async def test_time_series_lists_are_returned(self, admin_client: AsyncClient):
        resp = await admin_client.get("/api/admin/metrics")
        data = resp.json()
        assert isinstance(data["jobs_per_day"], list)
        assert isinstance(data["revenue_per_day"], list)
        assert isinstance(data["new_users_per_day"], list)
