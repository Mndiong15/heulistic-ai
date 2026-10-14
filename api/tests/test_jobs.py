"""Tests for public job endpoints (POST/GET/DELETE /api/jobs, /api/jobs/:id/*)."""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import CreditTransaction, Job, User
from tests.conftest import OTHER_SESSION_ID, SESSION_ID, VALID_CONFIG, submit_job


# ── POST /api/jobs ─────────────────────────────────────────────────────────────


class TestSubmitJob:
    async def test_success_returns_201_with_job_id(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["instance_type"] == "g6.xlarge"
        assert data["estimated_cost"] is not None

    async def test_invalid_yaml_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", b"key: [unclosed", "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "Invalid YAML" in resp.json()["detail"]

    async def test_missing_base_model_returns_422(self, client: AsyncClient):
        bad_config = b"adapter: lora\nnum_epochs: 1\n"
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", bad_config, "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "base_model" in resp.json()["detail"]

    async def test_no_auth_returns_401(self, anon_client: AsyncClient):
        resp = await anon_client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 401

    async def test_invalid_auth_scheme_returns_401(self, anon_client: AsyncClient):
        resp = await anon_client.post(
            "/api/jobs",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 401

    async def test_with_dataset_records_size(self, client: AsyncClient):
        dataset = b'{"text": "hello world"}\n' * 100
        resp = await client.post(
            "/api/jobs",
            files={
                "config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml"),
                "dataset": ("data.jsonl", dataset, "application/x-ndjson"),
            },
        )
        assert resp.status_code == 201

    async def test_jobs_past_the_running_limit_are_queued_not_rejected(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """The scheduler caps how many of a user's jobs *run* at once and holds
        the rest as pending. Submit used to apply the same cap to queued jobs
        too, so a fourth job — say, the next point of a sweep — was refused
        instead of waiting its turn."""
        for _ in range(3):
            db_session.add(Job(id=uuid.uuid4(), session_id=SESSION_ID,
                               status="running", config_yaml=VALID_CONFIG))
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "pending"

    @pytest.mark.parametrize("status", ["pending", "provisioning", "running", "checkpointing"])
    async def test_unfinished_jobs_are_capped_per_user(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch, status
    ):
        """An anti-abuse cap on everything not yet finished. It counts every
        live status — the old check counted only pending and running, so jobs
        that were provisioning or checkpointing slipped past it."""
        from app.config import settings
        monkeypatch.setattr(settings, "max_unfinished_jobs_per_user", 4)
        for _ in range(4):
            db_session.add(Job(id=uuid.uuid4(), session_id=SESSION_ID,
                               status=status, config_yaml=VALID_CONFIG))
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 429
        assert "unfinished" in resp.json()["detail"]

    async def test_finished_jobs_do_not_count_toward_the_cap(
        self, client: AsyncClient, db_session: AsyncSession, monkeypatch
    ):
        from app.config import settings
        monkeypatch.setattr(settings, "max_unfinished_jobs_per_user", 4)
        monkeypatch.setattr(settings, "max_submissions_per_hour", 100)
        for status in ("completed", "failed", "cancelled", "completed"):
            db_session.add(Job(id=uuid.uuid4(), session_id=SESSION_ID,
                               status=status, config_yaml=VALID_CONFIG))
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_rate_limit_hourly_submissions(self, client: AsyncClient, db_session: AsyncSession):
        from datetime import timedelta
        from app.config import settings

        # Fill up the hourly window with completed jobs
        for _ in range(settings.max_submissions_per_hour):
            db_session.add(
                Job(
                    id=uuid.uuid4(),
                    session_id=SESSION_ID,
                    status="completed",
                    config_yaml=VALID_CONFIG,
                )
            )
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 429

    async def test_a_busy_platform_queues_instead_of_refusing(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """How many jobs run at once is the scheduler's call — it is bounded by
        the EC2 GPU quota, and the scheduler already requeues on
        VcpuLimitExceeded. Refusing at submit only turned a wait into an error."""
        for _ in range(20):
            db_session.add(Job(id=uuid.uuid4(), session_id=str(uuid.uuid4()),
                               status="running", config_yaml=VALID_CONFIG))
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "pending"

    async def test_dataset_too_large_returns_413(self, client: AsyncClient, monkeypatch):
        """Reject a dataset over the configured cap.

        The limit is lowered for the test rather than allocating the real one:
        `max_dataset_size_mb` defaults to 5120, so building a payload just over
        it meant a 5 GB bytes object — which httpx then copies into a multipart
        body. That exhausted memory and SIGKILLed the whole session on any
        machine that had not set MAX_DATASET_SIZE_MB, as only CI does.
        """
        from app.config import settings

        monkeypatch.setattr(settings, "max_dataset_size_mb", 1)
        oversized = b"x" * (1 * 1024 * 1024 + 1)
        resp = await client.post(
            "/api/jobs",
            files={
                "config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml"),
                "dataset": ("data.jsonl", oversized, "application/x-ndjson"),
            },
        )
        assert resp.status_code == 413
        assert "limit" in resp.json()["detail"].lower()

    async def test_s3_upload_failure_returns_503(self, client: AsyncClient, monkeypatch):
        from botocore.exceptions import ClientError

        def raise_s3_error(*args, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ServiceUnavailable", "Message": "Service unavailable"}},
                "PutObject",
            )

        monkeypatch.setattr("app.services.s3.upload_bytes", raise_s3_error)

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()


# ── POST /api/jobs — HuggingFace token validation ─────────────────────────────


class TestHFTokenValidation:
    """HuggingFace token and gated-model checks run during job submission."""

    def _patch_hf(self, monkeypatch, *, whoami_status=200, model_status=200, raise_exc=None):
        class _FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, url, **kwargs):
                # only whoami uses GET
                if raise_exc is not None:
                    raise raise_exc
                r = MagicMock()
                r.status_code = whoami_status
                return r
            async def head(self, url, **kwargs):
                # model access check uses HEAD on resolve URL
                if raise_exc is not None:
                    raise raise_exc
                r = MagicMock()
                r.status_code = model_status
                return r
        monkeypatch.setattr("app.routers.jobs.httpx.AsyncClient", lambda **kw: _FakeClient())

    def _patch_secrets(self, monkeypatch):
        monkeypatch.setattr(
            "app.routers.jobs.secrets_service.store_hf_token",
            lambda job_id, token: f"arn:aws:secretsmanager:us-east-1:123:secret:hf-{job_id}",
        )

    async def test_valid_token_job_submits_successfully(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch)
        self._patch_secrets(monkeypatch)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_validtoken123"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_invalid_token_returns_422(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch, whoami_status=401)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_badtoken"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "HuggingFace token" in resp.json()["detail"]

    async def test_forbidden_token_returns_422(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch, whoami_status=403)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_badtoken"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "HuggingFace token" in resp.json()["detail"]

    async def test_model_not_found_returns_422(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch, model_status=404)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_validtoken123"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "not found" in resp.json()["detail"]

    async def test_model_access_denied_returns_422(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch, model_status=403)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_validtoken123"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "access" in resp.json()["detail"].lower()

    async def test_network_error_does_not_block_submission(self, client: AsyncClient, monkeypatch):
        self._patch_hf(monkeypatch, raise_exc=ConnectionError("timeout"))
        self._patch_secrets(monkeypatch)
        resp = await client.post(
            "/api/jobs",
            data={"hf_token": "hf_validtoken123"},
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_no_token_public_model_submits_successfully(self, client: AsyncClient, monkeypatch):
        """No token but model is publicly accessible (HEAD 200) → job created."""
        self._patch_hf(monkeypatch, model_status=200)
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_no_token_gated_model_returns_422(self, client: AsyncClient, monkeypatch):
        """No token and model is gated (HEAD 403) → 422 telling user to add a token."""
        self._patch_hf(monkeypatch, model_status=403)
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "gated" in detail.lower()
        assert "token" in detail.lower()

    async def test_no_token_model_not_found_returns_422(self, client: AsyncClient, monkeypatch):
        """No token and model returns 404 → 422 model not found."""
        self._patch_hf(monkeypatch, model_status=404)
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "not found" in resp.json()["detail"]


# ── GET /api/jobs ──────────────────────────────────────────────────────────────


# ── POST /api/jobs — Axolotl config validation ───────────────────────────────


class TestAxolotlConfigValidation:
    """Pre-flight checks on required Axolotl config fields."""

    def _post(self, client: AsyncClient, config: str):
        return client.post(
            "/api/jobs",
            files={"config": ("config.yaml", config.encode(), "application/x-yaml")},
        )

    async def test_missing_learning_rate_returns_422(self, client: AsyncClient):
        config = "\n".join(
            line for line in VALID_CONFIG.splitlines() if not line.startswith("learning_rate")
        )
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "learning_rate" in resp.json()["detail"]

    async def test_zero_num_epochs_returns_422(self, client: AsyncClient):
        config = VALID_CONFIG.replace("num_epochs: 1", "num_epochs: 0")
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "num_epochs" in resp.json()["detail"]

    async def test_negative_num_epochs_returns_422(self, client: AsyncClient):
        config = VALID_CONFIG.replace("num_epochs: 1", "num_epochs: -1")
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "num_epochs" in resp.json()["detail"]

    async def test_zero_micro_batch_size_returns_422(self, client: AsyncClient):
        config = VALID_CONFIG.replace("micro_batch_size: 2", "micro_batch_size: 0")
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "micro_batch_size" in resp.json()["detail"]

    async def test_no_dataset_source_returns_422(self, client: AsyncClient):
        config = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
adapter: lora
lora_r: 16
num_epochs: 1
micro_batch_size: 2
learning_rate: 2e-4
"""
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "dataset" in resp.json()["detail"].lower()

    async def test_pretraining_dataset_accepted_without_upload(self, client: AsyncClient):
        config = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
num_epochs: 1
micro_batch_size: 2
learning_rate: 2e-4
pretraining_dataset:
  - path: cerebras/SlimPajama-627B
    type: pretrain
"""
        resp = await self._post(client, config)
        assert resp.status_code == 201

    async def test_valid_config_submits_successfully(self, client: AsyncClient):
        resp = await self._post(client, VALID_CONFIG)
        assert resp.status_code == 201

    async def test_multiple_errors_reported_together(self, client: AsyncClient):
        config = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
num_epochs: 0
micro_batch_size: 0
"""
        resp = await self._post(client, config)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "learning_rate" in detail
        assert "num_epochs" in detail
        assert "micro_batch_size" in detail


# ── POST /api/jobs — dataset validation ───────────────────────────────────────


class TestDatasetValidation:
    """Pre-flight content checks on uploaded datasets."""

    _ALPACA_CONFIG = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
adapter: lora
lora_r: 16
num_epochs: 1
micro_batch_size: 2
learning_rate: 2e-4
datasets:
  - path: data.jsonl
    type: alpaca
"""
    _COMPLETION_CONFIG = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
adapter: lora
lora_r: 16
num_epochs: 1
micro_batch_size: 2
learning_rate: 2e-4
datasets:
  - path: data.jsonl
    type: completion
"""

    def _jsonl(self, rows: list) -> bytes:
        import json as _json
        return "\n".join(_json.dumps(r) for r in rows).encode()

    def _alpaca_rows(self, n: int = 15) -> list:
        return [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(n)]

    def _post(self, client: AsyncClient, dataset: bytes, config: str = VALID_CONFIG):
        return client.post(
            "/api/jobs",
            files={
                "config": ("config.yaml", config.encode(), "application/x-yaml"),
                "dataset": ("data.jsonl", dataset, "application/x-ndjson"),
            },
        )

    async def test_empty_file_returns_422(self, client: AsyncClient):
        resp = await self._post(client, b"")
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    async def test_whitespace_only_returns_422(self, client: AsyncClient):
        resp = await self._post(client, b"   \n\n  ")
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    async def test_too_few_examples_returns_422(self, client: AsyncClient):
        dataset = self._jsonl([{"text": f"x{i}"} for i in range(5)])
        resp = await self._post(client, dataset)
        assert resp.status_code == 422
        assert "at least 10" in resp.json()["detail"]

    async def test_invalid_json_line_returns_422(self, client: AsyncClient):
        good = [b'{"text": "ok"}'] * 10
        bad = [b"not json at all"]
        dataset = b"\n".join(good + bad)
        resp = await self._post(client, dataset)
        assert resp.status_code == 422
        assert "not valid JSON" in resp.json()["detail"]

    async def test_valid_jsonl_no_type_check_succeeds(self, client: AsyncClient):
        # VALID_CONFIG has no datasets.type → field check skipped
        dataset = self._jsonl([{"text": f"example {i}"} for i in range(15)])
        resp = await self._post(client, dataset)
        assert resp.status_code == 201

    async def test_alpaca_missing_fields_returns_422(self, client: AsyncClient):
        dataset = self._jsonl([{"text": "no required fields"} for _ in range(15)])
        resp = await self._post(client, dataset, self._ALPACA_CONFIG)
        assert resp.status_code == 422
        assert "missing" in resp.json()["detail"].lower()
        assert "instruction" in resp.json()["detail"] or "output" in resp.json()["detail"]

    async def test_alpaca_valid_dataset_succeeds(self, client: AsyncClient):
        dataset = self._jsonl(self._alpaca_rows())
        resp = await self._post(client, dataset, self._ALPACA_CONFIG)
        assert resp.status_code == 201

    async def test_completion_missing_text_returns_422(self, client: AsyncClient):
        dataset = self._jsonl([{"instruction": "x", "output": "y"} for _ in range(15)])
        resp = await self._post(client, dataset, self._COMPLETION_CONFIG)
        assert resp.status_code == 422
        assert "text" in resp.json()["detail"]

    async def test_completion_valid_dataset_succeeds(self, client: AsyncClient):
        dataset = self._jsonl([{"text": f"sample {i}"} for i in range(15)])
        resp = await self._post(client, dataset, self._COMPLETION_CONFIG)
        assert resp.status_code == 201

    async def test_only_first_100_lines_checked_for_json(self, client: AsyncClient):
        # Line 101 is invalid JSON — should not be caught (we only scan first 100)
        valid = [b'{"text": "x"}'] * 100
        invalid = [b"this is not json"]
        dataset = b"\n".join(valid + invalid)
        resp = await self._post(client, dataset)
        assert resp.status_code == 201

    async def test_unknown_dataset_type_skips_field_check(self, client: AsyncClient):
        config = VALID_CONFIG + "datasets:\n  - path: data.jsonl\n    type: custom_format\n"
        dataset = self._jsonl([{"whatever": f"x{i}"} for i in range(15)])
        resp = await self._post(client, dataset, config)
        assert resp.status_code == 201


class TestListJobs:
    async def test_empty_returns_empty_list(self, client: AsyncClient):
        resp = await client.get("/api/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_submitted_jobs(self, client: AsyncClient):
        await submit_job(client)
        await submit_job(client)

        resp = await client.get("/api/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_only_own_session_jobs(self, client: AsyncClient, other_client: AsyncClient):
        await submit_job(client)
        await submit_job(other_client)

        resp = await client.get("/api/jobs")
        assert len(resp.json()) == 1
        assert resp.json()[0]["job_id"] is not None

    async def test_ordered_newest_first(self, client: AsyncClient):
        r1 = await submit_job(client)
        r2 = await submit_job(client)

        resp = await client.get("/api/jobs")
        ids = [j["job_id"] for j in resp.json()]
        assert ids[0] == r2["job_id"]
        assert ids[1] == r1["job_id"]


# ── GET /api/jobs/:id ──────────────────────────────────────────────────────────


class TestGetJob:
    async def test_success(self, client: AsyncClient):
        job = await submit_job(client)
        resp = await client.get(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job["job_id"]
        assert data["status"] == "pending"
        assert "config_yaml" in data

    async def test_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/api/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_wrong_session_returns_403(self, client: AsyncClient, other_client: AsyncClient):
        job = await submit_job(client)
        resp = await other_client.get(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 403

    async def test_includes_cost_fields(self, client: AsyncClient):
        job = await submit_job(client)
        resp = await client.get(f"/api/jobs/{job['job_id']}")
        data = resp.json()
        assert "costs" in data
        assert data["costs"]["estimated"] is not None


# ── DELETE /api/jobs/:id ───────────────────────────────────────────────────────


class TestCancelJob:
    async def test_cancel_pending_job(self, client: AsyncClient):
        job = await submit_job(client)
        resp = await client.delete(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 204

    async def test_cancel_running_job(self, client: AsyncClient, db_session: AsyncSession):
        job = await submit_job(client)
        # Manually put it in running state
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "running"
        await db_session.commit()

        resp = await client.delete(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 204

    async def test_cancel_records_actual_cost(self, client: AsyncClient, db_session: AsyncSession):
        """Cancelling settled the ledger but left `actual_cost` NULL.

        The refund was computed from elapsed time and written to
        credit_transactions, yet the job row kept no record of what was
        actually charged — so the UI showed a blank actual cost and the ledger
        entry could not be reconciled against the job.
        """
        job = await submit_job(client)
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "running"
        db_job.started_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        await db_session.commit()

        assert (await client.delete(f"/api/jobs/{job['job_id']}")).status_code == 204

        await db_session.refresh(db_job)
        assert db_job.actual_cost is not None
        # One hour on the selected instance, at the platform markup.
        assert db_job.actual_cost > 0

    async def test_cancel_before_start_records_zero_cost(self, client: AsyncClient, db_session: AsyncSession):
        job = await submit_job(client)
        assert (await client.delete(f"/api/jobs/{job['job_id']}")).status_code == 204

        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        # Never provisioned, so nothing was spent — but it must be 0, not NULL.
        assert db_job.actual_cost == 0

    async def test_cancel_completed_job_returns_409(self, client: AsyncClient, db_session: AsyncSession):
        job = await submit_job(client)
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "completed"
        await db_session.commit()

        resp = await client.delete(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 409

    async def test_cancel_nonexistent_returns_404(self, client: AsyncClient):
        resp = await client.delete(f"/api/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_cancel_checkpointing_job(self, client: AsyncClient, db_session: AsyncSession):
        # Regression: cancel was rejected with 409 for jobs in 'checkpointing' state,
        # but the frontend showed the cancel button for that state.
        job = await submit_job(client)
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "checkpointing"
        await db_session.commit()

        resp = await client.delete(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 204

    async def test_cancel_wrong_session_returns_403(self, client: AsyncClient, other_client: AsyncClient):
        job = await submit_job(client)
        resp = await other_client.delete(f"/api/jobs/{job['job_id']}")
        assert resp.status_code == 403


# ── GET /api/jobs/:id/metrics ──────────────────────────────────────────────────


class TestGetMetrics:
    async def test_empty_metrics(self, client: AsyncClient):
        job = await submit_job(client)
        resp = await client.get(f"/api/jobs/{job['job_id']}/metrics")
        assert resp.status_code == 200
        assert resp.json() == []


# ── GET /api/jobs/:id/download ────────────────────────────────────────────────


class TestDownload:
    async def test_pending_job_returns_409(self, client: AsyncClient):
        job = await submit_job(client)
        resp = await client.get(f"/api/jobs/{job['job_id']}/download")
        assert resp.status_code == 409

    async def test_completed_no_path_returns_404(self, client: AsyncClient, db_session: AsyncSession):
        job = await submit_job(client)
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "completed"
        db_job.model_s3_path = None
        await db_session.commit()

        resp = await client.get(f"/api/jobs/{job['job_id']}/download")
        assert resp.status_code == 404

    async def test_completed_with_path_returns_presigned_url(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await submit_job(client)
        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(job["job_id"])))
        db_job = result.scalar_one()
        db_job.status = "completed"
        db_job.model_s3_path = f"s3://test-bucket/temp-models/{job['job_id']}/model.tar.gz"
        await db_session.commit()

        resp = await client.get(f"/api/jobs/{job['job_id']}/download")
        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data
        assert "s3.amazonaws.com" in data["url"]


# ── Credit buffer ──────────────────────────────────────────────────────────────


class TestCreditBuffer:
    """1.5× buffer held at submission; held amount stored as estimated_cost."""

    _KNOWN_COST = Decimal("1.00")
    _HELD_CENTS = 150  # ceil(1.00 * 100 * 1.5)

    @pytest.fixture(autouse=True)
    def fixed_estimate(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.cost.parse_and_estimate",
            lambda *a, **kw: {
                "instance_type": "g5.xlarge",
                "estimated_duration_minutes": 10,
                "estimated_cost": self._KNOWN_COST,
                "hourly_rate": Decimal("6.00"),
            },
        )

    async def test_zero_credits_returns_402(self, authed_client: AsyncClient):
        resp = await authed_client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 402

    async def test_error_message_mentions_buffer(self, authed_client: AsyncClient):
        resp = await authed_client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert "buffer" in resp.json()["detail"].lower()

    async def test_one_cent_short_returns_402(self, client: AsyncClient, db_session: AsyncSession):
        result = await db_session.execute(select(User).where(User.id == SESSION_ID))
        user = result.scalar_one()
        user.credit_balance_cents = self._HELD_CENTS - 1
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 402

    async def test_exact_buffer_amount_succeeds(self, client: AsyncClient, db_session: AsyncSession):
        result = await db_session.execute(select(User).where(User.id == SESSION_ID))
        user = result.scalar_one()
        user.credit_balance_cents = self._HELD_CENTS
        await db_session.commit()

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_held_amount_stored_as_estimated_cost(self, client: AsyncClient, db_session: AsyncSession):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

        result = await db_session.execute(select(Job).where(Job.id == uuid.UUID(resp.json()["job_id"])))
        job = result.scalar_one()
        assert float(job.estimated_cost) == pytest.approx(self._HELD_CENTS / 100, abs=0.01)

    async def test_held_amount_deducted_from_balance(self, client: AsyncClient, db_session: AsyncSession):
        result = await db_session.execute(select(User).where(User.id == SESSION_ID))
        user = result.scalar_one()
        initial = user.credit_balance_cents

        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

        await db_session.refresh(user)
        assert user.credit_balance_cents == initial - self._HELD_CENTS

    async def test_transaction_amount_and_description(self, client: AsyncClient, db_session: AsyncSession):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

        result = await db_session.execute(
            select(CreditTransaction).where(CreditTransaction.job_id == uuid.UUID(resp.json()["job_id"]))
        )
        txn = result.scalar_one()
        assert txn.amount_cents == -self._HELD_CENTS
        assert "buffer" in txn.description.lower()


# ── Single-node multi-GPU vs genuine multi-node ───────────────────────────────


def _config_with(**extra_fields) -> bytes:
    """Return VALID_CONFIG extended with the given YAML fields."""
    lines = "\n".join(f"{k}: {v}" for k, v in extra_fields.items())
    return (VALID_CONFIG + lines + "\n").encode()


@pytest.mark.asyncio
class TestSingleNodeMultiGpuIsAllowed:
    """These settings were rejected as "multi-node" and should not have been.

    Heulistic sells four-GPU instances (g4dn/g5/g6/g6e .12xlarge) and the
    training container runs `accelerate launch --num_processes=N` when it sees
    more than one GPU, so sharding across GPUs on one machine is supported.
    Axolotl's own schema marks none of these unsupported either, and the
    frontend validator has always allowed them — so the API was rejecting
    configs its own UI called valid.
    """

    async def test_world_size_greater_than_1_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(world_size=4), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_world_size_1_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(world_size=1), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_tensor_parallel_size_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(tensor_parallel_size=2), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_dp_shard_size_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(dp_shard_size=4), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_context_parallel_size_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(context_parallel_size=2), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_deepspeed_is_not_blocked(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(deepspeed="deepspeed_configs/zero2.json"), "application/x-yaml")},
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
class TestMultiNodeRejection:
    """Ray genuinely needs a cluster of machines, which is not supported."""

    async def test_use_ray_true_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(use_ray="true"), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "multi-node" in resp.json()["detail"].lower()

    async def test_use_ray_false_is_accepted(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(use_ray="false"), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_ray_num_workers_greater_than_1_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(ray_num_workers=4), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "multi-node" in resp.json()["detail"].lower()


# ── Validation delegated to heulistic-axolotl ─────────────────────────────────


@pytest.mark.asyncio
class TestSharedValidator:
    """Submission validation now comes from the shared schema package."""

    async def test_wrong_type_is_rejected(self, client: AsyncClient):
        # New capability: the old hand-written checks did no type checking.
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(micro_batch_size="not-a-number"), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "micro_batch_size" in resp.json()["detail"]

    async def test_unknown_field_is_only_a_warning(self, client: AsyncClient):
        # VALID_CONFIG itself carries `model_type`, which Axolotl no longer
        # documents. Warnings must never block a submission.
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(totally_made_up_field=1), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_placeholder_model_is_rejected(self, client: AsyncClient):
        config = VALID_CONFIG.replace("meta-llama/Llama-2-7b-hf", "YOUR_MODEL_HERE").encode()
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", config, "application/x-yaml")},
        )
        assert resp.status_code == 422

    async def test_quoted_scientific_learning_rate_is_accepted(self, client: AsyncClient):
        # YAML parses `2e-4` as a string; Axolotl types learning_rate
        # `str | float`, so this must not be read as a type error.
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", VALID_CONFIG.encode(), "application/x-yaml")},
        )
        assert resp.status_code == 201

    async def test_invalid_value_is_rejected(self, client: AsyncClient):
        resp = await client.post(
            "/api/jobs",
            files={"config": ("config.yaml", _config_with(num_epochs=0), "application/x-yaml")},
        )
        assert resp.status_code == 422
        assert "num_epochs" in resp.json()["detail"]


class TestSubmitChecksTheHardware:
    """Submit validated the config but never against the GPU it would run on,
    so a job that fit no instance was accepted, billed for the closest box,
    and failed with an OOM once the instance had booted."""

    @staticmethod
    def _post(client: AsyncClient, config: str, instance_type: str | None = None):
        data = {"instance_type": instance_type} if instance_type else {}
        return client.post(
            "/api/jobs",
            files={"config": ("config.yaml", config.encode(), "application/x-yaml")},
            data=data,
        )

    async def test_a_job_that_fits_no_instance_is_rejected(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        # 13B full fine-tune: ~250 GB, ~62 GB per card even sharded over 4 x 44.
        config = VALID_CONFIG.replace("adapter: lora\n", "").replace("7b", "13b")
        resp = await self._post(client, config)
        assert resp.status_code == 422
        assert "VRAM" in resp.json()["detail"]
        jobs = (await db_session.execute(select(Job))).scalars().all()
        assert jobs == []

    async def test_a_chosen_instance_that_is_too_small_is_rejected(self, client: AsyncClient):
        # 7B LoRA needs ~16.8 GB; a g4dn.xlarge has 16.
        resp = await self._post(client, VALID_CONFIG, instance_type="g4dn.xlarge")
        assert resp.status_code == 422
        assert "g4dn.xlarge" in resp.json()["detail"]

    async def test_a_tight_fit_is_accepted(self, client: AsyncClient):
        # 70B QLoRA is ~42 GB: 95.5% of a g6e.xlarge. Selection places it
        # there, so submit must not then refuse it.
        config = VALID_CONFIG.replace("7b", "70b").replace(
            "adapter: lora", "adapter: qlora\nload_in_4bit: true"
        )
        resp = await self._post(client, config)
        assert resp.status_code == 201
        assert resp.json()["instance_type"] == "g6e.xlarge"


class TestValidateConfigHardwareFlag:
    FULL_13B = VALID_CONFIG.replace("adapter: lora\n", "").replace("7b", "13b")

    def test_local_mode_does_not_check_against_cloud_instances(self):
        # The local GPU is invisible to the API; a cloud catalogue verdict
        # would reject jobs a big local rig can run.
        from app.routers.jobs import _validate_config
        _validate_config(self.FULL_13B, has_dataset_upload=False, check_hardware=False)

    def test_cloud_checks_against_the_instance_selection_would_rent(self):
        from fastapi import HTTPException
        from app.routers.jobs import _validate_config
        with pytest.raises(HTTPException) as exc:
            _validate_config(self.FULL_13B, has_dataset_upload=False, check_hardware=True)
        assert "No available instance" in exc.value.detail
