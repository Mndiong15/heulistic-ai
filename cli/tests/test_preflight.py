"""Pre-flight checks.

The Docker check used to discard `docker info`'s own error and print a fixed
"start Docker and try again". On Windows that is often the wrong advice: the
daemon may be absent entirely because Docker was installed inside WSL, in
which case the fix is to run the CLI from WSL rather than to start anything.
"""

import subprocess

import pytest

from heulistic import preflight


@pytest.fixture
def docker(monkeypatch):
    """Fake the presence of the docker binary and the daemon's reply."""

    def _set(installed=True, returncode=0, stderr=""):
        monkeypatch.setattr(
            preflight.shutil, "which",
            lambda name: "/usr/bin/docker" if (installed and name == "docker") else None,
        )

        def _run(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

        monkeypatch.setattr(preflight.subprocess, "run", _run)

    return _set


# ── Docker availability ───────────────────────────────────────────────────────

def test_no_docker_binary(docker):
    PLATFORM = "linux"
    docker(installed=False)
    problem = preflight.docker_problem(PLATFORM)
    assert problem is not None
    assert "not installed" in problem
    assert "docs.docker.com" in problem


def test_daemon_unreachable_surfaces_its_own_error(docker):
    PLATFORM = "linux"
    docker(returncode=1, stderr=(
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
        "Is the docker daemon running?"
    ))
    problem = preflight.docker_problem(PLATFORM)
    assert "Cannot connect to the Docker daemon" in problem


def test_windows_gets_a_windows_specific_hint(docker):
    PLATFORM = "win32"
    docker(returncode=1, stderr=(
        "error during connect: ... The system cannot find the file specified."
    ))
    problem = preflight.docker_problem(PLATFORM)
    assert "Docker Desktop" in problem
    assert "WSL" in problem, "the WSL case is the likely cause and must be mentioned"


def test_linux_hint_does_not_mention_windows(docker):
    PLATFORM = "linux"
    docker(returncode=1, stderr="Cannot connect to the Docker daemon")
    problem = preflight.docker_problem(PLATFORM)
    assert "Docker Desktop" not in problem


def test_healthy_docker_is_no_problem(docker):
    PLATFORM = "linux"
    docker(returncode=0)
    assert preflight.docker_problem(PLATFORM) is None


def test_empty_stderr_still_produces_actionable_text(docker):
    PLATFORM = "linux"
    docker(returncode=1, stderr="")
    problem = preflight.docker_problem(PLATFORM)
    assert problem
    assert "running" in problem.lower()


def test_docker_binary_that_cannot_be_executed(docker, monkeypatch):
    PLATFORM = "linux"
    docker(installed=True)

    def _boom(*a, **kw):
        raise OSError("exec format error")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    problem = preflight.docker_problem(PLATFORM)
    assert problem and "exec format error" in problem


# ── Port checks ───────────────────────────────────────────────────────────────

def test_port_free_detects_a_listener():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        taken = srv.getsockname()[1]
        assert preflight._port_free(taken) is False

    # Once the listener closes the port is free again.
    assert preflight._port_free(taken) is True


# ── Automatic port selection ──────────────────────────────────────────────────

class TestFindFreePort:
    def _listener(self, port=0):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return s, s.getsockname()[1]

    def test_returns_the_preferred_port_when_free(self):
        s, port = self._listener()
        s.close()
        assert preflight.find_free_port(port) == port

    def test_steps_past_a_busy_port(self):
        s, port = self._listener()
        try:
            assert preflight.find_free_port(port) == port + 1
        finally:
            s.close()

    def test_steps_past_several_busy_ports(self):
        held = []
        try:
            s, base = self._listener()
            held.append(s)
            for offset in (1, 2):
                try:
                    s2, _ = self._listener(base + offset)
                    held.append(s2)
                except OSError:
                    pytest.skip("could not reserve a contiguous port range")
            assert preflight.find_free_port(base) == base + 3
        finally:
            for s in held:
                s.close()

    def test_gives_up_after_the_attempt_limit(self, monkeypatch):
        monkeypatch.setattr(preflight, "_port_free", lambda port: False)
        assert preflight.find_free_port(3000, attempts=5) is None

    def test_avoids_ports_already_claimed_in_this_run(self, monkeypatch):
        # The frontend is resolved first; the API must not land on top of it.
        monkeypatch.setattr(preflight, "_port_free", lambda port: True)
        assert preflight.find_free_port(3000, avoid={3000}) == 3001


class TestResolvePorts:
    def test_both_free_and_default_are_kept(self, monkeypatch):
        monkeypatch.setattr(preflight, "_port_free", lambda port: True)
        assert preflight.resolve_ports(3000, 8000, False, False) == (3000, 8000, [])

    def test_a_defaulted_busy_port_shifts_and_is_reported(self, monkeypatch):
        monkeypatch.setattr(preflight, "_port_free", lambda port: port not in (3000,))
        frontend, api, notes = preflight.resolve_ports(3000, 8000, False, False)
        assert (frontend, api) == (3001, 8000)
        assert any("3000" in n and "3001" in n for n in notes)

    def test_an_explicit_busy_port_is_left_alone(self, monkeypatch):
        # Honour the request; `run()` then reports the clash as before.
        monkeypatch.setattr(preflight, "_port_free", lambda port: port not in (3000,))
        frontend, api, notes = preflight.resolve_ports(3000, 8000, True, False)
        assert frontend == 3000
        assert notes == []

    def test_the_two_ports_never_collide(self, monkeypatch):
        monkeypatch.setattr(preflight, "_port_free", lambda port: True)
        frontend, api, _ = preflight.resolve_ports(3000, 3000, False, False)
        assert frontend != api

    def test_exhaustion_leaves_the_port_unchanged(self, monkeypatch):
        monkeypatch.setattr(preflight, "_port_free", lambda port: False)
        frontend, api, notes = preflight.resolve_ports(3000, 8000, False, False)
        assert (frontend, api) == (3000, 8000)
        assert notes == []
