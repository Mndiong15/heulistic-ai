"""Tests for local_provisioner.provision() — Docker-based training container launch."""

import hashlib

import local_provisioner


# ── Fake Docker client ────────────────────────────────────────────────────────


class _FakeContainer:
    short_id = "abc1234"


class _FakeContainers:
    def __init__(self):
        self.run_calls: list[dict] = []

    def run(self, image, **kwargs):
        self.run_calls.append({"image": image, **kwargs})
        return _FakeContainer()


class _FakeDockerClient:
    def __init__(self):
        self.containers = _FakeContainers()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_job(**overrides):
    base = {
        "id": "ffffffff-0000-0000-0000-000000000006",
        "config_s3_path": "s3://heulistic-local/configs/fff/config.yaml",
        "dataset_s3_path": None,
        "resume_from_checkpoint_path": None,
    }
    base.update(overrides)
    return base


# ── TestLocalProvision ────────────────────────────────────────────────────────


class TestLocalProvision:
    def _setup(self, monkeypatch) -> _FakeDockerClient:
        fake = _FakeDockerClient()
        monkeypatch.setattr("local_provisioner._client", lambda: fake)
        monkeypatch.setenv("LOCAL_TRAINING_IMAGE", "heulistic/training:latest")
        return fake

    def test_returns_tuple_of_two_strings(self, monkeypatch):
        self._setup(monkeypatch)
        result = local_provisioner.provision(_make_job())
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, str) for v in result)

    def test_instance_id_equals_job_id(self, monkeypatch):
        self._setup(monkeypatch)
        job = _make_job()
        instance_id, _ = local_provisioner.provision(job)
        assert instance_id == str(job["id"])

    def test_token_hash_is_64_char_hex(self, monkeypatch):
        self._setup(monkeypatch)
        _, token_hash = local_provisioner.provision(_make_job())
        assert len(token_hash) == 64
        assert all(c in "0123456789abcdef" for c in token_hash)

    def test_db_hash_validates_container_token(self, monkeypatch):
        # The hash written to DB must validate the plaintext the container sends.
        # This mirrors exactly what the API's _get_job() does at verify time.
        fake = self._setup(monkeypatch)
        _, token_hash = local_provisioner.provision(_make_job())

        env = fake.containers.run_calls[0]["environment"]
        plain_token = env["INTERNAL_TOKEN"]

        assert hashlib.sha256(plain_token.encode()).hexdigest() == token_hash

    def test_plaintext_token_differs_from_hash(self, monkeypatch):
        # Sanity check: the container must not receive the hash itself.
        fake = self._setup(monkeypatch)
        _, token_hash = local_provisioner.provision(_make_job())

        env = fake.containers.run_calls[0]["environment"]
        assert env["INTERNAL_TOKEN"] != token_hash

    def test_token_is_unique_per_call(self, monkeypatch):
        self._setup(monkeypatch)
        _, hash1 = local_provisioner.provision(_make_job())
        _, hash2 = local_provisioner.provision(_make_job())
        assert hash1 != hash2

    def test_container_labeled_with_job_id(self, monkeypatch):
        fake = self._setup(monkeypatch)
        job = _make_job()
        local_provisioner.provision(job)

        labels = fake.containers.run_calls[0]["labels"]
        assert labels.get("heulistic.job_id") == str(job["id"])


# ── GPUs per job ──────────────────────────────────────────────────────────────
# By default a local job gets every GPU, so jobs run one at a time. With
# GPUS_PER_JOB set, each job is pinned to that many free cards, and a 4-GPU
# workstation can run a sweep four jobs at a time.


class _RunningContainer:
    def __init__(self, gpus):
        self.labels = {local_provisioner._LABEL_KEY: "other", local_provisioner._GPU_LABEL_KEY: gpus}


class _ContainersWithRunning(_FakeContainers):
    def __init__(self, running):
        super().__init__()
        self._running = running

    def list(self, filters=None):
        return list(self._running)


class TestGpusPerJob:
    def _setup(self, monkeypatch, running=(), per_job="1", count="4"):
        fake = _FakeDockerClient()
        fake.containers = _ContainersWithRunning([_RunningContainer(g) for g in running])
        monkeypatch.setattr("local_provisioner._client", lambda: fake)
        monkeypatch.setenv("LOCAL_TRAINING_IMAGE", "heulistic/training:latest")
        monkeypatch.setenv("GPUS_PER_JOB", per_job)
        monkeypatch.setenv("LOCAL_GPU_COUNT", count)
        return fake

    @staticmethod
    def _request(fake):
        (request,) = fake.containers.run_calls[0]["device_requests"]
        return request

    def test_default_gives_the_job_every_gpu(self, monkeypatch):
        fake = self._setup(monkeypatch, per_job="0")
        local_provisioner.provision(_make_job())
        request = self._request(fake)
        assert request["Count"] == -1
        assert not request.get("DeviceIDs")

    def test_a_job_is_pinned_to_a_free_card(self, monkeypatch):
        fake = self._setup(monkeypatch, running=["0", "2"])
        local_provisioner.provision(_make_job())
        assert self._request(fake)["DeviceIDs"] == ["1"]

    def test_the_cards_it_holds_are_labelled_for_the_next_job(self, monkeypatch):
        fake = self._setup(monkeypatch, running=["0"])
        local_provisioner.provision(_make_job())
        labels = fake.containers.run_calls[0]["labels"]
        assert labels[local_provisioner._GPU_LABEL_KEY] == "1"

    def test_several_cards_per_job(self, monkeypatch):
        fake = self._setup(monkeypatch, running=["0,2"], per_job="2")
        local_provisioner.provision(_make_job())
        assert self._request(fake)["DeviceIDs"] == ["1", "3"]

    def test_no_free_cards_is_retryable_not_a_failure(self, monkeypatch):
        fake = self._setup(monkeypatch, running=["0,1", "2,3"], per_job="2")
        try:
            local_provisioner.provision(_make_job())
        except local_provisioner.NoFreeGpus:
            pass
        else:
            raise AssertionError("expected NoFreeGpus")
        assert fake.containers.run_calls == []


class TestLocalExport:
    def _setup(self, monkeypatch):
        fake = _FakeDockerClient()
        monkeypatch.setattr("local_provisioner._client", lambda: fake)
        monkeypatch.setenv("LOCAL_TRAINING_IMAGE", "heulistic/training:latest")
        monkeypatch.setenv("LOCAL_GPU", "true")
        return fake

    def _job(self):
        return _make_job(kind="export", export_quant="Q8_0",
                         source_model_s3_path="s3://heulistic-local/models/src/model.tar.gz")

    def test_the_container_is_told_what_to_export(self, monkeypatch):
        fake = self._setup(monkeypatch)
        local_provisioner.provision(self._job())
        env = fake.containers.run_calls[0]["environment"]
        assert env["JOB_KIND"] == "export"
        assert env["EXPORT_QUANT"] == "Q8_0"
        assert env["SOURCE_MODEL_S3_PATH"] == "s3://heulistic-local/models/src/model.tar.gz"

    def test_an_export_takes_no_gpu(self, monkeypatch):
        # Merging and quantizing run on the CPU; holding a GPU would block training.
        fake = self._setup(monkeypatch)
        local_provisioner.provision(self._job())
        assert fake.containers.run_calls[0]["device_requests"] == []
