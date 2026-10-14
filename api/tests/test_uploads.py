"""Dataset uploads.

Datasets used to travel browser → load balancer → API, where the whole file
was read into memory on a 2 GB t3.small before a single S3 PUT (itself capped
at 5 GB). Anything past ~1.5 GB took the API down for every user. Now the
browser sends parts straight to S3 through presigned multipart URLs, and the
API only ever reads the first megabyte, to validate it.
"""

import json
import math
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Job
from tests.conftest import OTHER_SESSION_ID, SESSION_ID, VALID_CONFIG

MB = 1024 * 1024
GOOD_ROWS = b"".join(
    json.dumps({"instruction": f"q{i}", "output": f"a{i}"}).encode() + b"\n" for i in range(50)
)


class FakeBucket:
    """The handful of S3 operations uploads use, kept in memory."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, str] = {}  # upload_id -> key
        self.completed: list[tuple[str, str, list]] = []
        self.aborted: list[tuple[str, str]] = []

    def create_multipart_upload(self, key):
        upload_id = uuid.uuid4().hex
        self.uploads[upload_id] = key
        return upload_id

    def presign_upload_part(self, key, upload_id, part_number, expires_in):
        return f"https://bucket.s3.test/{key}?uploadId={upload_id}&partNumber={part_number}"

    def complete_multipart_upload(self, key, upload_id, parts):
        self.completed.append((key, upload_id, parts))

    def abort_multipart_upload(self, key, upload_id):
        self.aborted.append((key, upload_id))

    def object_size(self, key):
        data = self.objects.get(key)
        return None if data is None else len(data)

    def read_head(self, key, nbytes):
        return self.objects[key][:nbytes]


@pytest.fixture
def bucket(monkeypatch):
    fake = FakeBucket()
    for name in ("create_multipart_upload", "presign_upload_part",
                 "complete_multipart_upload", "abort_multipart_upload",
                 "object_size", "read_head"):
        monkeypatch.setattr(f"app.services.s3.{name}", getattr(fake, name))
    return fake


def own_key(name="data.jsonl"):
    return f"temp-datasets/{SESSION_ID}/{uuid.uuid4().hex}/{name}"


# ── Starting an upload ────────────────────────────────────────────────────────


class TestCreateUpload:
    async def test_returns_a_presigned_url_per_part(self, client: AsyncClient, bucket, monkeypatch):
        # The cap is set here, not inherited: CI runs with MAX_DATASET_SIZE_MB=10.
        from app.config import settings
        monkeypatch.setattr(settings, "max_dataset_size_mb", 1024)
        size = 300 * MB
        resp = await client.post("/api/uploads", json={"size_bytes": size})
        assert resp.status_code == 201
        body = resp.json()
        assert body["key"].startswith(f"temp-datasets/{SESSION_ID}/")
        assert body["upload_id"] in bucket.uploads
        assert len(body["part_urls"]) == math.ceil(size / body["part_size"])

    async def test_parts_stay_within_s3_limits_at_the_cap(self, client: AsyncClient, bucket, monkeypatch):
        # The default cap, set explicitly — whatever the environment loaded
        # (CI uses 10 MB) would make this check a trivially small file.
        from app.config import Settings, settings
        cap = Settings.model_fields["max_dataset_size_mb"].default
        monkeypatch.setattr(settings, "max_dataset_size_mb", cap)
        size = cap * MB
        body = (await client.post("/api/uploads", json={"size_bytes": size})).json()
        assert body["part_size"] >= 5 * MB  # S3's minimum part size
        assert len(body["part_urls"]) <= 10_000  # S3's maximum part count

    def test_the_default_cap_is_well_above_the_old_5_gb(self):
        # The default, not the loaded value: a developer's .env may pin its own.
        from app.config import Settings
        assert Settings.model_fields["max_dataset_size_mb"].default >= 50 * 1024

    async def test_over_the_cap_is_413(self, client: AsyncClient, bucket, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "max_dataset_size_mb", 10)
        resp = await client.post("/api/uploads", json={"size_bytes": 10 * MB + 1})
        assert resp.status_code == 413
        assert bucket.uploads == {}

    @pytest.mark.parametrize("size", [0, -1])
    async def test_empty_is_422(self, client: AsyncClient, bucket, size):
        resp = await client.post("/api/uploads", json={"size_bytes": size})
        assert resp.status_code == 422

    async def test_requires_auth(self, anon_client: AsyncClient, bucket):
        resp = await anon_client.post("/api/uploads", json={"size_bytes": MB})
        assert resp.status_code == 401

    async def test_not_offered_in_local_mode(self, client: AsyncClient, bucket, monkeypatch):
        """Presigned URLs would name MinIO's in-container hostname, which the
        browser cannot reach. Local mode streams uploads through the API."""
        from app.config import settings
        monkeypatch.setattr(settings, "local_mode", True)
        resp = await client.post("/api/uploads", json={"size_bytes": MB})
        assert resp.status_code == 404


# ── Finishing an upload ───────────────────────────────────────────────────────


class TestCompleteUpload:
    async def test_completes_with_parts_in_order(self, client: AsyncClient, bucket):
        key = own_key()
        resp = await client.post("/api/uploads/complete", json={
            "key": key, "upload_id": "u1",
            "parts": [{"part_number": 2, "etag": '"b"'}, {"part_number": 1, "etag": '"a"'}],
        })
        assert resp.status_code == 200
        (done_key, done_id, parts), = bucket.completed
        assert (done_key, done_id) == (key, "u1")
        assert [p["part_number"] for p in parts] == [1, 2]

    async def test_another_users_key_is_403(self, client: AsyncClient, bucket):
        key = f"temp-datasets/{OTHER_SESSION_ID}/x/data.jsonl"
        resp = await client.post("/api/uploads/complete", json={
            "key": key, "upload_id": "u1", "parts": [{"part_number": 1, "etag": '"a"'}],
        })
        assert resp.status_code == 403
        assert bucket.completed == []

    async def test_a_prefix_lookalike_is_403(self, client: AsyncClient, bucket):
        # "temp-datasets/<me>-evil/..." starts with "temp-datasets/<me>".
        key = f"temp-datasets/{SESSION_ID}-evil/x/data.jsonl"
        resp = await client.post("/api/uploads/complete", json={
            "key": key, "upload_id": "u1", "parts": [{"part_number": 1, "etag": '"a"'}],
        })
        assert resp.status_code == 403

    async def test_abort(self, client: AsyncClient, bucket):
        key = own_key()
        resp = await client.post("/api/uploads/abort", json={"key": key, "upload_id": "u1"})
        assert resp.status_code == 204
        assert bucket.aborted == [(key, "u1")]


# ── Submitting with an uploaded dataset ───────────────────────────────────────

ALPACA_CONFIG = VALID_CONFIG.replace(
    "  - path: tatsu-lab/alpaca\n", "  - path: tatsu-lab/alpaca\n    type: alpaca\n"
)


def submit(client, dataset_key, config=ALPACA_CONFIG):
    return client.post(
        "/api/jobs",
        files={"config": ("config.yaml", config.encode(), "application/x-yaml")},
        data={"dataset_key": dataset_key},
    )


class TestSubmitWithDatasetKey:
    async def test_job_points_at_the_uploaded_object(
        self, client: AsyncClient, bucket, db_session: AsyncSession
    ):
        key = own_key()
        bucket.objects[key] = GOOD_ROWS + b"x" * (3 * MB)  # size comes from S3, not the body
        resp = await submit(client, key)
        assert resp.status_code == 201, resp.text
        job = (await db_session.execute(select(Job))).scalar_one()
        assert job.dataset_s3_path.endswith(key)
        assert job.dataset_size_mb == 3

    async def test_only_the_head_is_read(self, client: AsyncClient, bucket, monkeypatch):
        key = own_key()
        bucket.objects[key] = GOOD_ROWS * 1_500  # ~3 MB
        seen = []
        real = bucket.read_head
        monkeypatch.setattr("app.services.s3.read_head",
                            lambda k, n: seen.append(n) or real(k, n))
        resp = await submit(client, key)
        assert resp.status_code == 201, resp.text
        assert seen and max(seen) <= MB

    async def test_another_users_key_is_403(self, client: AsyncClient, bucket):
        key = f"temp-datasets/{OTHER_SESSION_ID}/x/data.jsonl"
        bucket.objects[key] = GOOD_ROWS
        assert (await submit(client, key)).status_code == 403

    async def test_a_missing_object_is_422(self, client: AsyncClient, bucket):
        resp = await submit(client, own_key())
        assert resp.status_code == 422
        assert "upload" in resp.json()["detail"].lower()

    async def test_over_the_cap_is_413(self, client: AsyncClient, bucket, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "max_dataset_size_mb", 1)
        key = own_key()
        bucket.objects[key] = GOOD_ROWS + b"x" * MB
        assert (await submit(client, key)).status_code == 413

    async def test_the_head_is_validated(self, client: AsyncClient, bucket):
        key = own_key()
        bucket.objects[key] = b'{"text": "only the wrong fields"}\n' * 50
        resp = await submit(client, key)
        assert resp.status_code == 422
        assert "instruction" in resp.json()["detail"]

    async def test_too_few_examples_is_422(self, client: AsyncClient, bucket):
        key = own_key()
        bucket.objects[key] = GOOD_ROWS.split(b"\n", 3)[0] + b"\n"
        resp = await submit(client, key)
        assert resp.status_code == 422
        assert "at least" in resp.json()["detail"]


# ── Submitting with a file through the API ────────────────────────────────────


class TestStreamedUpload:
    async def test_the_file_is_never_read_whole(self, client: AsyncClient, monkeypatch):
        """Reading the upload in one call is what took the API down: every
        read must be bounded, and the body must reach S3 as a stream."""
        from starlette.datastructures import UploadFile

        sizes, streamed = [], []
        real_read = UploadFile.read

        async def bounded_read(self, size=-1):
            sizes.append(size)
            return await real_read(self, size)

        monkeypatch.setattr(UploadFile, "read", bounded_read)
        monkeypatch.setattr("app.services.s3.upload_fileobj",
                            lambda f, key: streamed.append(key) or f"s3://test-bucket/{key}")
        monkeypatch.setattr("app.services.s3.upload_bytes", lambda *a, **k: pytest.fail(
            "a dataset must not be uploaded from an in-memory copy"
        ) if "temp-datasets" in a[1] else f"s3://test-bucket/{a[1]}")

        resp = await client.post("/api/jobs", files={
            "config": ("config.yaml", ALPACA_CONFIG.encode(), "application/x-yaml"),
            "dataset": ("data.jsonl", GOOD_ROWS * 40, "application/x-ndjson"),
        })
        assert resp.status_code == 201, resp.text
        # MB + 1: the config is read one byte past its cap, to detect overflow.
        assert all(0 < s <= MB + 1 for s in sizes), sizes
        assert any(k.startswith("temp-datasets/") for k in streamed)

    async def test_an_oversized_config_is_rejected_without_reading_it_whole(
        self, client: AsyncClient
    ):
        resp = await client.post("/api/jobs", files={
            "config": ("config.yaml", ALPACA_CONFIG.encode() + b"#" * (2 * MB), "application/x-yaml"),
        })
        assert resp.status_code == 413
        assert "config" in resp.json()["detail"].lower()

    async def test_cloud_caps_uploads_through_the_api_lower(
        self, client: AsyncClient, monkeypatch
    ):
        """In the cloud a large file must go straight to S3: through the API it
        is spooled to the t3.small's 30 GB disk first."""
        from app.config import settings
        monkeypatch.setattr(settings, "max_dataset_size_mb", 100)
        monkeypatch.setattr(settings, "max_proxied_upload_mb", 1)
        resp = await client.post("/api/jobs", files={
            "config": ("config.yaml", ALPACA_CONFIG.encode(), "application/x-yaml"),
            "dataset": ("data.jsonl", GOOD_ROWS + b"x" * MB, "application/x-ndjson"),
        })
        assert resp.status_code == 413

    def test_local_mode_streams_up_to_the_full_cap(self):
        from app.config import Settings
        # Everything this depends on is passed in: _env_file=None ignores a
        # developer's .env, and an explicit cap overrides the environment's
        # (CI sets MAX_DATASET_SIZE_MB=10, below the 5 GB proxied cap).
        common = dict(_env_file=None, database_url="postgresql://x", s3_bucket="b",
                      max_dataset_size_mb=51200)
        local = Settings(**common, local_mode=True)
        cloud = Settings(**common, local_mode=False)
        assert local.max_proxied_upload_mb == local.max_dataset_size_mb
        assert cloud.max_proxied_upload_mb < cloud.max_dataset_size_mb
