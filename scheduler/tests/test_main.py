import botocore.exceptions
import pytest

import main


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client_error(code: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": "test"}}, "RunInstances"
    )


class _Cursor:
    def __init__(self):
        self.executed = []
        self.rowcount = 1

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def executemany(self, query, seq):
        for params in seq:
            self.executed.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Conn:
    def __init__(self):
        self.cursors: list[_Cursor] = []
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> _Cursor:
        cur = _Cursor()
        self.cursors.append(cur)
        return cur

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


def _job():
    return {"id": "job-aaa", "instance_type": "g5.xlarge"}


# ── _dispatch ─────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_successful_provision_sets_instance_id(self, monkeypatch):
        monkeypatch.setattr("main.provisioner.provision", lambda job: ("i-abc123", "hash_abc"))
        conn = _Conn()

        main._dispatch(conn, _job())

        # Claim + set_instance = 2 cursors, 2 commits
        assert conn.commit_count == 2
        set_instance_params = conn.cursors[-1].executed[0][1]
        assert set_instance_params["instance_id"] == "i-abc123"
        assert set_instance_params["token_hash"] == "hash_abc"
        assert set_instance_params["job_id"] == "job-aaa"

    def test_transient_capacity_error_requeues_job(self, monkeypatch):
        monkeypatch.setattr(
            "main.provisioner.provision",
            lambda job: (_ for _ in ()).throw(_make_client_error("VcpuLimitExceeded")),
        )
        conn = _Conn()

        main._dispatch(conn, _job())

        # Job must be requeued, not failed
        last_query = conn.cursors[-1].executed[0][0]
        assert "pending" in last_query
        assert "failed" not in last_query
        assert conn.commit_count == 2  # claim + requeue

    def test_insufficient_capacity_requeues_job(self, monkeypatch):
        monkeypatch.setattr(
            "main.provisioner.provision",
            lambda job: (_ for _ in ()).throw(_make_client_error("InsufficientInstanceCapacity")),
        )
        conn = _Conn()

        main._dispatch(conn, _job())

        last_query = conn.cursors[-1].executed[0][0]
        assert "pending" in last_query

    def test_unknown_aws_error_fails_job(self, monkeypatch):
        monkeypatch.setattr(
            "main.provisioner.provision",
            lambda job: (_ for _ in ()).throw(_make_client_error("UnauthorizedOperation")),
        )
        conn = _Conn()

        main._dispatch(conn, _job())

        last_query = conn.cursors[-1].executed[0][0]
        assert "failed" in last_query

    def test_non_aws_error_fails_job(self, monkeypatch):
        monkeypatch.setattr(
            "main.provisioner.provision",
            lambda job: (_ for _ in ()).throw(RuntimeError("SSM param missing")),
        )
        conn = _Conn()

        main._dispatch(conn, _job())

        last_query = conn.cursors[-1].executed[0][0]
        assert "failed" in last_query

    def test_already_claimed_job_is_skipped(self, monkeypatch):
        provisioned = []
        monkeypatch.setattr("main.provisioner.provision", lambda job: provisioned.append(job) or ("i-x", "hash_x"))

        conn = _Conn()
        # Simulate rowcount=0 (another scheduler claimed it first)
        conn.cursors  # not yet created
        original_cursor = conn.cursor

        def _cursor_with_zero_rowcount():
            cur = _Cursor()
            cur.rowcount = 0
            conn.cursors.append(cur)
            return cur

        conn.cursor = _cursor_with_zero_rowcount

        main._dispatch(conn, _job())

        assert provisioned == []
        assert conn.rollback_count == 1


# ── Local-mode concurrency ────────────────────────────────────────────────────


class TestLocalModeConcurrency:
    """One machine, one GPU set — the cloud default of 20 would thrash it."""

    def _settings(self, **env):
        import config as config_module

        # _env_file=None ignores the developer's scheduler/.env, which sets
        # MAX_CONCURRENT_JOBS and would otherwise mask the resolved default.
        return config_module.Settings(_env_file=None, **env)

    def test_local_mode_defaults_to_one_job(self):
        s = self._settings(database_url="postgresql://x/y", local_mode=True)
        assert s.max_concurrent_jobs == 1

    def test_cloud_capacity_is_the_vcpu_quota(self):
        # In the cloud the limit is the EC2 GPU quota, not a job count.
        s = self._settings(database_url="postgresql://x/y", local_mode=False)
        assert s.vcpu_quota == 64

    def test_a_dotenv_from_an_older_release_still_loads(self, tmp_path):
        import config as config_module
        env = tmp_path / ".env"
        env.write_text("DATABASE_URL=postgresql://x/y\nMAX_JOBS_PER_SESSION=3\n")
        assert config_module.Settings(_env_file=env).vcpu_quota == 64

    def test_an_explicit_value_wins_even_locally(self):
        # A multi-GPU workstation can opt back in.
        s = self._settings(
            database_url="postgresql://x/y", local_mode=True, max_concurrent_jobs=4
        )
        assert s.max_concurrent_jobs == 4


# ── Choosing jobs ─────────────────────────────────────────────────────────────


class TestChooseJobs:
    @staticmethod
    def _jobs(n, itype="g5.xlarge"):
        from datetime import datetime, timedelta
        t0 = datetime(2026, 1, 1)
        return [{"id": i, "session_id": f"u{i}", "instance_type": itype,
                 "queue_position": i + 1, "created_at": t0 + timedelta(minutes=i)}
                for i in range(n)]

    def test_cloud_starts_as_many_as_the_quota_holds(self, monkeypatch):
        import main
        monkeypatch.setattr(main.settings, "local_mode", False)
        monkeypatch.setattr(main.settings, "vcpu_quota", 64)
        assert len(main._choose_jobs(self._jobs(30), active=[])) == 16

    def test_cloud_is_not_capped_at_twenty_jobs(self, monkeypatch):
        import main
        monkeypatch.setattr(main.settings, "local_mode", False)
        monkeypatch.setattr(main.settings, "vcpu_quota", 256)
        assert len(main._choose_jobs(self._jobs(30), active=[])) == 30

    def test_local_mode_counts_jobs(self, monkeypatch):
        import main
        monkeypatch.setattr(main.settings, "local_mode", True)
        monkeypatch.setattr(main.settings, "max_concurrent_jobs", 1)
        jobs = self._jobs(3, itype="local-gpu")
        assert len(main._choose_jobs(jobs, active=[])) == 1
        assert main._choose_jobs(jobs, active=jobs[:1]) == []


class TestLocalGpuSharing:
    def _settings(self, **env):
        import config as config_module
        return config_module.Settings(_env_file=None, database_url="postgresql://x/y",
                                      local_mode=True, **env)

    def test_capacity_is_gpus_divided_by_gpus_per_job(self):
        assert self._settings(local_gpu_count=4, gpus_per_job=1).max_concurrent_jobs == 4
        assert self._settings(local_gpu_count=4, gpus_per_job=2).max_concurrent_jobs == 2
        assert self._settings(local_gpu_count=3, gpus_per_job=2).max_concurrent_jobs == 1

    def test_by_default_one_job_holds_every_gpu(self):
        assert self._settings(local_gpu_count=4).max_concurrent_jobs == 1

    def test_no_free_gpus_requeues_the_job(self, monkeypatch):
        import local_provisioner

        def busy(job):
            raise local_provisioner.NoFreeGpus("all cards in use")

        monkeypatch.setattr("main.provisioner.provision", busy)
        conn = _Conn()
        main._dispatch(conn, _job())
        last_query = conn.cursors[-1].executed[0][0]
        assert "pending" in last_query and "failed" not in last_query


# ── Queue positions ───────────────────────────────────────────────────────────


class TestUpdateQueuePositions:
    @staticmethod
    def _pending(n, user, position=None, resume=False):
        from datetime import datetime, timedelta
        return {"id": f"job-{n}", "session_id": user, "instance_type": "g5.xlarge",
                "queue_position": 0 if resume else position,
                "created_at": datetime(2026, 1, 1) + timedelta(minutes=n)}

    def _written(self, conn):
        return {p["id"]: p["pos"] for cur in conn.cursors for _, p in cur.executed}

    def test_writes_fair_order_positions(self):
        conn = _Conn()
        pending = [self._pending(1, "sweeper"), self._pending(2, "sweeper"),
                   self._pending(3, "b")]
        main._update_queue_positions(conn, pending, active=[])
        assert self._written(conn) == {"job-1": 1, "job-3": 2, "job-2": 3}

    def test_only_changed_rows_are_written(self):
        conn = _Conn()
        pending = [self._pending(1, "a", position=1), self._pending(2, "b", position=5)]
        main._update_queue_positions(conn, pending, active=[])
        assert self._written(conn) == {"job-2": 2}

    def test_resumes_are_left_alone(self):
        conn = _Conn()
        pending = [self._pending(1, "a", resume=True), self._pending(2, "b")]
        main._update_queue_positions(conn, pending, active=[])
        assert self._written(conn) == {"job-2": 2}

    def test_only_still_pending_jobs_are_renumbered(self):
        # A job may start or be cancelled between the read and this write.
        conn = _Conn()
        main._update_queue_positions(conn, [self._pending(1, "a")], active=[])
        query = conn.cursors[-1].executed[0][0]
        assert "status = 'pending'" in query


class TestLocalExportsDoNotTakeASlot:
    def test_an_export_runs_beside_a_training_job(self, monkeypatch):
        from datetime import datetime
        monkeypatch.setattr(main.settings, "local_mode", True)
        monkeypatch.setattr(main.settings, "max_concurrent_jobs", 1)
        t = datetime(2026, 1, 1)
        training = {"id": 1, "session_id": "u", "instance_type": "local-gpu", "kind": "train",
                    "queue_position": 1, "created_at": t}
        export = {"id": 2, "session_id": "u", "instance_type": "g4dn.xlarge", "kind": "export",
                  "queue_position": 2, "created_at": t}
        assert [j["id"] for j in main._choose_jobs([export], active=[training])] == [2]
