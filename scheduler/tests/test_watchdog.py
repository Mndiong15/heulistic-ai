import watchdog


class _Cursor:
    """Fake psycopg2 cursor that acts as a context manager."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Conn:
    """Fake psycopg2 connection.

    cursor() call sequence:
      0 — orphan SELECT  (returns orphans)
      1..N — one UPDATE per orphan
      N+1 — cancelled SELECT  (returns cancelled)
      N+2..M — one UPDATE per cancelled job
    """

    def __init__(self, orphans=None, cancelled=None):
        self._orphans = orphans or []
        self._cancelled = cancelled or []
        self._call_count = 0
        self.update_cursors: list[_Cursor] = []
        self.commit_count = 0

    def cursor(self) -> _Cursor:
        n_orphans = len(self._orphans)
        # First SELECT = orphan query
        if self._call_count == 0:
            cur = _Cursor(self._orphans)
        # One UPDATE cursor per orphan
        elif self._call_count <= n_orphans:
            cur = _Cursor([])
            self.update_cursors.append(cur)
        # Next SELECT = cancelled query
        elif self._call_count == n_orphans + 1:
            cur = _Cursor(self._cancelled)
        # One UPDATE cursor per cancelled job
        else:
            cur = _Cursor([])
            self.update_cursors.append(cur)
        self._call_count += 1
        return cur

    def commit(self):
        self.commit_count += 1


class TestWatchdogOrphans:
    def test_no_orphans_does_nothing(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        conn = _Conn(orphans=[])
        watchdog.run(conn)

        assert terminated == []
        assert conn.commit_count == 0

    def test_orphaned_job_terminated_and_marked_failed(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        orphan = {"id": "job-aaa", "instance_id": "i-123abc"}
        conn = _Conn(orphans=[orphan])
        watchdog.run(conn)

        assert terminated == ["i-123abc"]
        assert conn.commit_count == 1
        assert len(conn.update_cursors) == 1
        assert conn.update_cursors[0].executed[0][1] == ("job-aaa",)

    def test_local_mode_also_terminates(self, monkeypatch):
        """A hung local container must be stopped, not just marked failed.

        The watchdog used to guard termination behind `not local_mode`, so a
        local job that stopped heartbeating was marked failed while its
        container kept running — holding the GPU indefinitely.
        `provisioner.terminate` already dispatches to the Docker provisioner in
        local mode, so the guard only ever leaked containers.
        """
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", True)

        conn = _Conn(orphans=[{"id": "job-local", "instance_id": "job-local"}])
        watchdog.run(conn)

        assert terminated == ["job-local"]

    def test_local_mode_also_terminates_cancelled_jobs(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", True)

        conn = _Conn(orphans=[], cancelled=[{"id": "job-x", "instance_id": "job-x"}])
        watchdog.run(conn)

        assert terminated == ["job-x"]

    def test_multiple_orphans(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        orphans = [
            {"id": "job-1", "instance_id": "i-aaa"},
            {"id": "job-2", "instance_id": "i-bbb"},
        ]
        conn = _Conn(orphans=orphans)
        watchdog.run(conn)

        assert sorted(terminated) == ["i-aaa", "i-bbb"]
        assert conn.commit_count == 2
        assert len(conn.update_cursors) == 2

    def test_terminate_failure_still_marks_job_failed(self, monkeypatch):
        monkeypatch.setattr("watchdog.terminate", lambda iid: (_ for _ in ()).throw(Exception("EC2 permission denied")))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        orphan = {"id": "job-fail", "instance_id": "i-xyz"}
        conn = _Conn(orphans=[orphan])

        watchdog.run(conn)

        assert conn.commit_count == 1
        assert conn.update_cursors[0].executed[0][1] == ("job-fail",)


class TestWatchdogCancelledJobs:
    def test_cancelled_instance_is_terminated(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        cancelled = {"id": "job-cancel", "instance_id": "i-cancel1"}
        conn = _Conn(cancelled=[cancelled])
        watchdog.run(conn)

        assert terminated == ["i-cancel1"]
        assert conn.commit_count == 1

    def test_cancelled_instance_id_cleared(self, monkeypatch):
        monkeypatch.setattr("watchdog.terminate", lambda iid: None)
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        cancelled = {"id": "job-cancel", "instance_id": "i-cancel1"}
        conn = _Conn(cancelled=[cancelled])
        watchdog.run(conn)

        assert len(conn.update_cursors) == 1
        assert conn.update_cursors[0].executed[0][1] == ("job-cancel",)

    def test_multiple_cancelled_jobs(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        cancelled = [
            {"id": "job-c1", "instance_id": "i-c1"},
            {"id": "job-c2", "instance_id": "i-c2"},
        ]
        conn = _Conn(cancelled=cancelled)
        watchdog.run(conn)

        assert sorted(terminated) == ["i-c1", "i-c2"]
        assert conn.commit_count == 2

    def test_cancelled_terminate_failure_still_clears_instance(self, monkeypatch):
        monkeypatch.setattr("watchdog.terminate", lambda iid: (_ for _ in ()).throw(Exception("no perms")))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        cancelled = {"id": "job-c3", "instance_id": "i-c3"}
        conn = _Conn(cancelled=[cancelled])

        watchdog.run(conn)

        assert conn.commit_count == 1
        assert conn.update_cursors[0].executed[0][1] == ("job-c3",)

    def test_orphan_and_cancelled_both_handled(self, monkeypatch):
        terminated = []
        monkeypatch.setattr("watchdog.terminate", lambda iid: terminated.append(iid))
        monkeypatch.setattr(watchdog.settings, "local_mode", False)

        conn = _Conn(
            orphans=[{"id": "job-o", "instance_id": "i-orphan"}],
            cancelled=[{"id": "job-c", "instance_id": "i-cancelled"}],
        )
        watchdog.run(conn)

        assert sorted(terminated) == ["i-cancelled", "i-orphan"]
        assert conn.commit_count == 2
