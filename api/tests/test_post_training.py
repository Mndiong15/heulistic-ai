"""After training: the before/after report, and exporting a model for Ollama.

An export is a job of its own — queued, billed and settled like training —
whose source is a finished training job. It runs in the training image with
JOB_KIND=export and produces a GGUF plus a Modelfile.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import CreditTransaction, Job
from tests.conftest import INTERNAL_TEST_TOKEN_HASH, OTHER_SESSION_ID, SESSION_ID, VALID_CONFIG

REPORT = {
    "examples": 12,
    "base": {"loss": 1.89, "perplexity": 6.64},
    "tuned": {"loss": 0.37, "perplexity": 1.45},
    "samples": [{"prompt": "p", "reference": "r", "base": "b", "tuned": "t"}],
}


async def _job(db: AsyncSession, **kwargs) -> Job:
    kwargs.setdefault("internal_token", INTERNAL_TEST_TOKEN_HASH)
    kwargs.setdefault("session_id", SESSION_ID)
    job = Job(id=uuid.uuid4(), status=kwargs.pop("status", "completed"),
              config_yaml=kwargs.pop("config_yaml", VALID_CONFIG), **kwargs)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _trained(db: AsyncSession, **kwargs) -> Job:
    kwargs.setdefault("model_s3_path", "s3://test-bucket/models/x/model.tar.gz")
    kwargs.setdefault("config_s3_path", "s3://test-bucket/configs/x/config.yaml")
    return await _job(db, **kwargs)


# ── The before/after report ───────────────────────────────────────────────────


class TestEvaluationReport:
    async def test_the_trainer_stores_it_and_the_job_shows_it(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _job(db_session, status="running")
        resp = await client.post(f"/internal/jobs/{job.id}/evaluation", json={"report": REPORT})
        assert resp.status_code == 204

        detail = (await client.get(f"/api/jobs/{job.id}")).json()
        assert detail["evaluation"] == REPORT

    async def test_needs_the_jobs_token(self, anon_client: AsyncClient, db_session: AsyncSession):
        job = await _job(db_session, status="running")
        resp = await anon_client.post(
            f"/internal/jobs/{job.id}/evaluation", json={"report": REPORT},
            headers={"X-Internal-Token": "wrong"},
        )
        assert resp.status_code in (401, 403)

    async def test_a_job_without_one_says_so(self, client: AsyncClient, db_session: AsyncSession):
        job = await _job(db_session)
        assert (await client.get(f"/api/jobs/{job.id}")).json()["evaluation"] is None


# ── Export options ────────────────────────────────────────────────────────────


class TestExportOptions:
    async def test_each_format_with_its_size_memory_and_cost(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _trained(db_session)
        resp = await client.get(f"/api/jobs/{job.id}/export-options")
        assert resp.status_code == 200
        options = resp.json()["options"]
        assert [o["quant"] for o in options] == ["Q4_K_M", "Q5_K_M", "Q8_0"]
        q4 = options[0]
        # 7B model: a ~4.3 GB file that needs ~5.4 GB to run.
        assert q4["file_gb"] == pytest.approx(4.3, abs=0.1)
        assert q4["memory_gb"] == pytest.approx(5.4, abs=0.1)
        assert float(q4["estimated_cost"]) > 0
        assert q4["estimated_minutes"] > 0
        assert resp.json()["recommended"] == "Q4_K_M"

    async def test_only_for_a_finished_training_job(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        job = await _job(db_session, status="running")
        assert (await client.get(f"/api/jobs/{job.id}/export-options")).status_code == 409


# ── Starting an export ────────────────────────────────────────────────────────


class TestStartExport:
    async def test_queues_an_export_job_for_the_trained_model(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={"quant": "Q5_K_M"})
        assert resp.status_code == 201, resp.text

        export = await db_session.get(Job, uuid.UUID(resp.json()["job_id"]))
        assert export.kind == "export"
        assert export.source_job_id == source.id
        assert export.export_quant == "Q5_K_M"
        assert export.status == "pending"
        assert export.config_yaml == source.config_yaml
        assert export.config_s3_path.endswith(f"configs/{export.id}/config.yaml")
        assert export.instance_type == "g4dn.xlarge"
        # Sized for the merged weights and full-precision GGUF side by side.
        assert export.disk_gb >= 200

    async def test_defaults_to_q4_k_m(self, client: AsyncClient, db_session: AsyncSession):
        source = await _trained(db_session)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={})
        export = await db_session.get(Job, uuid.UUID(resp.json()["job_id"]))
        assert export.export_quant == "Q4_K_M"

    async def test_reserves_credits_like_any_job(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={})
        tx = (await db_session.execute(
            select(CreditTransaction).where(CreditTransaction.job_id == uuid.UUID(resp.json()["job_id"]))
        )).scalar_one()
        assert tx.amount_cents < 0

    async def test_no_credits_is_402(self, authed_client: AsyncClient, db_session: AsyncSession):
        source = await _trained(db_session)
        resp = await authed_client.post(f"/api/jobs/{source.id}/export", json={})
        assert resp.status_code == 402

    async def test_unknown_format_is_422(self, client: AsyncClient, db_session: AsyncSession):
        source = await _trained(db_session)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={"quant": "Q2_K"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("state", [
        {"status": "running"},
        {"status": "failed"},
        {"model_s3_path": None},
        {"kind": "export"},
    ])
    async def test_only_a_finished_training_job_can_be_exported(
        self, client: AsyncClient, db_session: AsyncSession, state
    ):
        source = await _trained(db_session, **state)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={})
        assert resp.status_code == 409

    async def test_someone_elses_job_is_refused(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session, session_id=OTHER_SESSION_ID)
        resp = await client.post(f"/api/jobs/{source.id}/export", json={})
        assert resp.status_code in (403, 404)

    async def test_the_same_format_twice_is_refused(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """It would cost the user a second time for the same file."""
        source = await _trained(db_session)
        first = (await client.post(f"/api/jobs/{source.id}/export", json={})).json()["job_id"]
        resp = await client.post(f"/api/jobs/{source.id}/export", json={})
        assert resp.status_code == 409
        assert first in resp.json()["detail"]

    async def test_a_failed_export_can_be_retried(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        await _job(db_session, kind="export", source_job_id=source.id,
                   export_quant="Q4_K_M", status="failed")
        assert (await client.post(f"/api/jobs/{source.id}/export", json={})).status_code == 201


# ── What the user sees ────────────────────────────────────────────────────────


class TestExportResults:
    async def test_the_trainer_hands_back_the_modelfile(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        export = await _job(db_session, kind="export", source_job_id=source.id,
                            export_quant="Q4_K_M", status="running")
        resp = await client.post(f"/internal/jobs/{export.id}/complete", json={
            "model_s3_path": f"s3://test-bucket/models/{export.id}/m-Q4_K_M.gguf",
            "actual_cost": 0.05,
            "modelfile": "FROM ./m-Q4_K_M.gguf\n",
        })
        assert resp.status_code == 204
        detail = (await client.get(f"/api/jobs/{export.id}")).json()
        assert detail["kind"] == "export"
        assert detail["source_job_id"] == str(source.id)
        assert detail["export_quant"] == "Q4_K_M"
        assert detail["modelfile"] == "FROM ./m-Q4_K_M.gguf\n"

    async def test_a_training_job_lists_its_exports(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        export = await _job(db_session, kind="export", source_job_id=source.id,
                            export_quant="Q8_0", status="completed")
        detail = (await client.get(f"/api/jobs/{source.id}")).json()
        assert detail["kind"] == "train"
        assert detail["exports"] == [
            {"job_id": str(export.id), "quant": "Q8_0", "status": "completed"}
        ]

    async def test_the_job_list_says_which_jobs_are_exports(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        source = await _trained(db_session)
        await _job(db_session, kind="export", source_job_id=source.id, export_quant="Q4_K_M")
        kinds = sorted(j["kind"] for j in (await client.get("/api/jobs")).json())
        assert kinds == ["export", "train"]
