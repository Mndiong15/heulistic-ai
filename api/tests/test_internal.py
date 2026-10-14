"""Tests for internal endpoints used by training instances."""

import hashlib
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Job, Log, Metric, User
from tests.conftest import INTERNAL_TEST_TOKEN, INTERNAL_TEST_TOKEN_HASH, SESSION_ID, VALID_CONFIG


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _insert_job(db: AsyncSession, **kwargs) -> Job:
    kwargs.setdefault("internal_token", INTERNAL_TEST_TOKEN_HASH)
    job = Job(
        id=uuid.uuid4(),
        session_id=SESSION_ID,
        status=kwargs.pop("status", "pending"),
        config_yaml=VALID_CONFIG,
        **kwargs,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


# ── POST /internal/jobs/:id/heartbeat ────────────────────────────────────────


class TestHeartbeat:
    async def test_first_heartbeat_sets_running_and_started_at(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="pending")

        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-abc123"},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "running"
        assert job.instance_id == "i-abc123"
        assert job.started_at is not None
        assert job.last_heartbeat is not None

    async def test_subsequent_heartbeat_keeps_started_at(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        started = datetime(2024, 1, 1, 12, 0, 0)
        job = await _insert_job(
            db_session,
            status="running",
            instance_id="i-abc123",
            started_at=started,
        )

        await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-abc123"},
        )

        await db_session.refresh(job)
        # started_at must not change on subsequent heartbeat
        assert job.started_at == started
        assert job.last_heartbeat is not None

    async def test_provisioning_sentinel_replaced_on_first_heartbeat(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        # Scheduler sets instance_id='provisioning' before the instance boots
        job = await _insert_job(
            db_session, status="running", instance_id="provisioning"
        )

        await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-real123"},
        )

        await db_session.refresh(job)
        assert job.instance_id == "i-real123"
        assert job.started_at is not None

    async def test_unknown_job_returns_404(self, client: AsyncClient):
        resp = await client.post(
            f"/internal/jobs/{uuid.uuid4()}/heartbeat",
            json={"instance_id": "i-xyz"},
        )
        assert resp.status_code == 404

    async def test_heartbeat_completed_job_returns_409(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="completed")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-late"},
        )
        assert resp.status_code == 409

    async def test_heartbeat_cancelled_job_returns_409(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="cancelled")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-late"},
        )
        assert resp.status_code == 409

    async def test_heartbeat_failed_job_returns_409(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="failed")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-late"},
        )
        assert resp.status_code == 409


# ── POST /internal/jobs/:id/logs ──────────────────────────────────────────────


class TestAppendLogs:
    async def test_logs_are_stored(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")

        resp = await client.post(
            f"/internal/jobs/{job.id}/logs",
            json={"lines": ["step 1 done", "step 2 done", "epoch 1 complete"]},
        )
        assert resp.status_code == 204

        result = await db_session.execute(select(Log).where(Log.job_id == job.id))
        logs = result.scalars().all()
        assert len(logs) == 3
        assert logs[0].line == "step 1 done"

    async def test_empty_lines_accepted(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/logs",
            json={"lines": []},
        )
        assert resp.status_code == 204

    async def test_unknown_job_returns_404(self, client: AsyncClient):
        resp = await client.post(
            f"/internal/jobs/{uuid.uuid4()}/logs",
            json={"lines": ["hello"]},
        )
        assert resp.status_code == 404

    async def test_multiple_batches_accumulate(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")

        await client.post(f"/internal/jobs/{job.id}/logs", json={"lines": ["line A"]})
        await client.post(f"/internal/jobs/{job.id}/logs", json={"lines": ["line B", "line C"]})

        result = await db_session.execute(select(Log).where(Log.job_id == job.id))
        assert len(result.scalars().all()) == 3


# ── POST /internal/jobs/:id/metrics ──────────────────────────────────────────


class TestAppendMetrics:
    METRIC_BATCH = [
        {
            "step": 100,
            "epoch": 1,
            "train_loss": 0.4521,
            "learning_rate": 0.0002,
            "gpu_utilization": 95.0,
            "gpu_memory_used_mb": 20480,
            "tokens_per_second": 450,
        },
        {
            "step": 200,
            "epoch": 1,
            "train_loss": 0.3812,
            "learning_rate": 0.0001,
            "gpu_utilization": 96.0,
            "gpu_memory_used_mb": 20500,
            "tokens_per_second": 460,
        },
    ]

    async def test_metrics_are_stored(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")

        resp = await client.post(
            f"/internal/jobs/{job.id}/metrics",
            json={"metrics": self.METRIC_BATCH},
        )
        assert resp.status_code == 204

        result = await db_session.execute(select(Metric).where(Metric.job_id == job.id))
        metrics = result.scalars().all()
        assert len(metrics) == 2
        assert metrics[0].step == 100
        assert float(metrics[0].train_loss) == pytest.approx(0.4521, rel=1e-4)

    async def test_partial_fields_accepted(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/metrics",
            json={"metrics": [{"step": 50}]},
        )
        assert resp.status_code == 204

    async def test_unknown_job_returns_404(self, client: AsyncClient):
        resp = await client.post(
            f"/internal/jobs/{uuid.uuid4()}/metrics",
            json={"metrics": [{"step": 1}]},
        )
        assert resp.status_code == 404

    async def test_fractional_epoch_and_tps_accepted(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        # Training container emits fractional epoch (e.g. 2.962) and float
        # tokens_per_second (e.g. 40.14). Previously rejected by int | None schema.
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/metrics",
            json={"metrics": [{"step": 100, "epoch": 2.962, "tokens_per_second": 40.14}]},
        )
        assert resp.status_code == 204

        result = await db_session.execute(select(Metric).where(Metric.job_id == job.id))
        m = result.scalar_one()
        assert m.epoch == pytest.approx(2.962, rel=1e-4)
        assert m.tokens_per_second == pytest.approx(40.14, rel=1e-4)

    async def test_integer_epoch_still_accepted(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/metrics",
            json={"metrics": [{"step": 50, "epoch": 1, "tokens_per_second": 300}]},
        )
        assert resp.status_code == 204


# ── POST /internal/jobs/:id/complete ─────────────────────────────────────────


class TestCompleteJob:
    async def test_success_completion(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")
        model_path = f"s3://test-bucket/temp-models/{job.id}/model.tar.gz"

        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": model_path, "actual_cost": 1.23},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "completed"
        assert job.model_s3_path == model_path
        # Stored value is quantized to 2 decimal places (Decimal.quantize); allow ±$0.01
        assert float(job.actual_cost) == pytest.approx(1.23 * 2.2, abs=0.01)
        assert job.completed_at is not None

    async def test_failure_completion(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")

        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"error_message": "CUDA out of memory"},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "failed"
        assert job.error_message == "CUDA out of memory"
        assert job.completed_at is not None
        assert job.model_s3_path is None

    async def test_error_message_takes_precedence(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")

        await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={
                "model_s3_path": "s3://bucket/model.tar.gz",
                "error_message": "Something went wrong",
            },
        )

        await db_session.refresh(job)
        assert job.status == "failed"

    async def test_unknown_job_returns_404(self, client: AsyncClient):
        resp = await client.post(
            f"/internal/jobs/{uuid.uuid4()}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz"},
        )
        assert resp.status_code == 404

    async def test_complete_without_cost(self, client: AsyncClient, db_session: AsyncSession):
        job = await _insert_job(db_session, status="running")
        model_path = f"s3://test-bucket/temp-models/{job.id}/model.tar.gz"

        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": model_path},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "completed"
        assert job.actual_cost is None

    async def test_double_complete_is_idempotent(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")
        model_path = f"s3://test-bucket/temp-models/{job.id}/model.tar.gz"
        payload = {"model_s3_path": model_path, "actual_cost": 1.50}

        await client.post(f"/internal/jobs/{job.id}/complete", json=payload)
        await db_session.refresh(job)
        first_completed_at = job.completed_at

        # Second call — should be a no-op
        resp = await client.post(f"/internal/jobs/{job.id}/complete", json=payload)
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "completed"
        assert job.completed_at == first_completed_at  # unchanged

    async def test_complete_failed_job_is_idempotent(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")

        await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"error_message": "OOM"},
        )
        await db_session.refresh(job)
        assert job.status == "failed"

        # Retry with a success payload — must not overwrite the failure
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz"},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "failed"  # unchanged

    async def test_complete_cancelled_job_is_idempotent(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="cancelled")

        # Training container finishes after cancel — must not overwrite cancelled status
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz"},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "cancelled"  # unchanged


# ── POST /internal/jobs/:id/complete — Stripe metering ───────────────────────


class TestCompleteJobOverageCharge:
    """When job settlement pushes the balance negative, charge_overage fires immediately."""

    async def _set_user(self, db_session, *, balance=50, customer_id=None, payment_method_id=None):
        result = await db_session.execute(select(User).where(User.id == SESSION_ID))
        user = result.scalar_one()
        user.credit_balance_cents = balance
        user.stripe_customer_id = customer_id
        user.stripe_payment_method_id = payment_method_id
        await db_session.commit()
        return user

    async def test_charges_overage_when_balance_goes_negative(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        # User has 50 cents; estimated=1.00 ($100c), actual=1.50 × MARKUP(2.2)=3.30 ($330c)
        # diff = 100 - 330 = -230 → balance = 50 - 230 = -180 → triggers charge
        user = await self._set_user(
            db_session,
            balance=50,
            customer_id="cus_overage",
            payment_method_id="pm_saved",
        )

        charge_calls = []
        monkeypatch.setattr(
            "app.routers.internal.charge_overage",
            lambda cid, pmid, cents, jid: charge_calls.append((cid, pmid, cents)),
        )

        job = await _insert_job(db_session, status="running", estimated_cost=Decimal("1.00"))
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz", "actual_cost": 1.50},
        )
        assert resp.status_code == 204

        assert len(charge_calls) == 1
        cid, pmid, cents = charge_calls[0]
        assert cid == "cus_overage"
        assert pmid == "pm_saved"
        assert cents == 180  # abs(-180)

        await db_session.refresh(user)
        assert user.credit_balance_cents == 0  # zeroed after successful charge

    async def test_skips_charge_when_no_payment_method(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        await self._set_user(db_session, balance=50, customer_id="cus_no_pm")

        charge_calls = []
        monkeypatch.setattr(
            "app.routers.internal.charge_overage",
            lambda cid, pmid, cents, jid: charge_calls.append(cid),
        )

        job = await _insert_job(db_session, status="running", estimated_cost=Decimal("1.00"))
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz", "actual_cost": 1.50},
        )
        assert resp.status_code == 204
        assert charge_calls == []

    async def test_skips_charge_when_no_stripe_ids(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        await self._set_user(db_session, balance=50)  # no customer/payment method

        charge_calls = []
        monkeypatch.setattr(
            "app.routers.internal.charge_overage",
            lambda cid, pmid, cents, jid: charge_calls.append(cid),
        )

        job = await _insert_job(db_session, status="running", estimated_cost=Decimal("1.00"))
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz", "actual_cost": 1.50},
        )
        assert resp.status_code == 204
        assert charge_calls == []

    async def test_charge_failure_does_not_fail_completion(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        await self._set_user(
            db_session,
            balance=50,
            customer_id="cus_explodes",
            payment_method_id="pm_explodes",
        )

        monkeypatch.setattr(
            "app.routers.internal.charge_overage",
            lambda cid, pmid, cents, jid: (_ for _ in ()).throw(RuntimeError("card declined")),
        )

        job = await _insert_job(db_session, status="running", estimated_cost=Decimal("1.00"))
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz", "actual_cost": 1.50},
        )
        assert resp.status_code == 204

        await db_session.refresh(job)
        assert job.status == "completed"  # unaffected by Stripe failure

    async def test_no_charge_when_balance_stays_positive(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        # actual < estimated → refund, balance stays positive
        await self._set_user(
            db_session,
            balance=1_000_000,
            customer_id="cus_refund",
            payment_method_id="pm_refund",
        )

        charge_calls = []
        monkeypatch.setattr(
            "app.routers.internal.charge_overage",
            lambda cid, pmid, cents, jid: charge_calls.append(cid),
        )

        job = await _insert_job(db_session, status="running", estimated_cost=Decimal("5.00"))
        resp = await client.post(
            f"/internal/jobs/{job.id}/complete",
            json={"model_s3_path": "s3://bucket/model.tar.gz", "actual_cost": 1.00},
        )
        assert resp.status_code == 204
        assert charge_calls == []


# ── Token auth enforcement ────────────────────────────────────────────────────


class TestTokenAuth:
    """Verify that /internal/ endpoints enforce X-Internal-Token."""

    async def test_missing_token_returns_401(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-x"},
            headers={"X-Internal-Token": ""},
        )
        assert resp.status_code == 401

    async def test_wrong_token_returns_401(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="running")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-x"},
            headers={"X-Internal-Token": "definitely-wrong-token"},
        )
        assert resp.status_code == 401

    async def test_no_token_on_job_returns_401(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        # Job with no token stored (not yet provisioned by scheduler)
        job = await _insert_job(db_session, status="pending", internal_token=None)
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-x"},
        )
        assert resp.status_code == 401

    async def test_correct_token_returns_204(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _insert_job(db_session, status="pending")
        resp = await client.post(
            f"/internal/jobs/{job.id}/heartbeat",
            json={"instance_id": "i-x"},
        )
        assert resp.status_code == 204

    async def test_token_from_another_job_returns_401(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        # Token valid for job_b must not authenticate requests for job_a.
        other_token = "heulistic-other-job-token-9999"
        other_hash = hashlib.sha256(other_token.encode()).hexdigest()

        job_a = await _insert_job(db_session, status="running")
        await _insert_job(db_session, status="running", internal_token=other_hash)

        resp = await client.post(
            f"/internal/jobs/{job_a.id}/heartbeat",
            json={"instance_id": "i-x"},
            headers={"X-Internal-Token": other_token},
        )
        assert resp.status_code == 401
