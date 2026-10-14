"""Inspecting the running local stack.

`docker compose ps --format json` is the source of truth. Its shape has changed
across Compose releases — older builds emit one JSON array, newer ones emit
newline-delimited objects — so both are handled.
"""

import json

import pytest

from heulistic import stack

NDJSON = "\n".join(
    json.dumps(row)
    for row in [
        {"Service": "db", "State": "running", "Health": "healthy", "Publishers": []},
        {
            "Service": "api",
            "State": "running",
            "Health": "",
            "Publishers": [{"URL": "0.0.0.0", "PublishedPort": 8000, "TargetPort": 8000}],
        },
        {
            "Service": "frontend",
            "State": "running",
            "Health": "",
            "Publishers": [{"URL": "0.0.0.0", "PublishedPort": 3000, "TargetPort": 3000}],
        },
    ]
)

JSON_ARRAY = json.dumps(
    [
        {"Service": "db", "State": "running", "Health": "healthy", "Publishers": []},
        {"Service": "api", "State": "exited", "Health": "", "Publishers": []},
    ]
)


@pytest.fixture
def ps(monkeypatch):
    def _set(output):
        monkeypatch.setattr(stack, "_compose_ps_output", lambda: output)

    return _set


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_parses_newline_delimited_objects(ps):
    ps(NDJSON)
    services = stack.services()
    assert [s.name for s in services] == ["db", "api", "frontend"]
    assert all(s.running for s in services)


def test_parses_a_single_json_array(ps):
    ps(JSON_ARRAY)
    services = stack.services()
    assert [s.name for s in services] == ["db", "api"]
    assert services[0].running is True
    assert services[1].running is False


def test_health_is_surfaced_when_present(ps):
    ps(NDJSON)
    by_name = {s.name: s for s in stack.services()}
    assert by_name["db"].health == "healthy"
    assert by_name["api"].health == ""


def test_no_stack_running(ps):
    ps("")
    assert stack.services() == []
    assert stack.is_running() is False


def test_docker_unavailable(ps):
    ps(None)
    assert stack.services() == []
    assert stack.is_running() is False


def test_malformed_output_is_not_fatal(ps):
    ps("not json at all")
    assert stack.services() == []


def test_partial_garbage_keeps_the_good_rows(ps):
    ps('{"Service": "db", "State": "running"}\ngarbage\n')
    assert [s.name for s in stack.services()] == ["db"]


def test_is_running_requires_a_running_service(ps):
    ps(JSON_ARRAY)
    assert stack.is_running() is True
    ps(json.dumps([{"Service": "api", "State": "exited", "Publishers": []}]))
    assert stack.is_running() is False


# ── Published ports ───────────────────────────────────────────────────────────

def test_published_port_is_found(ps):
    ps(NDJSON)
    assert stack.published_port("api") == 8000
    assert stack.published_port("frontend") == 3000


def test_published_port_missing_returns_none(ps):
    ps(NDJSON)
    assert stack.published_port("db") is None
    assert stack.published_port("nonexistent") is None


def test_api_url_uses_the_published_port(ps):
    ps(NDJSON)
    assert stack.api_url() == "http://localhost:8000"


def test_api_url_is_none_when_the_api_is_not_published(ps):
    ps(json.dumps([{"Service": "db", "State": "running", "Publishers": []}]))
    assert stack.api_url() is None


# ── Job counts ────────────────────────────────────────────────────────────────

def test_job_counts_group_by_status(monkeypatch):
    monkeypatch.setattr(
        stack,
        "_get_json",
        lambda url: [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed"},
            {"status": "failed"},
        ],
    )
    assert stack.job_counts("http://localhost:8000") == {
        "running": 2,
        "completed": 1,
        "failed": 1,
    }


def test_job_counts_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setattr(stack, "_get_json", lambda url: None)
    assert stack.job_counts("http://localhost:8000") is None


def test_job_counts_tolerates_an_unexpected_payload(monkeypatch):
    monkeypatch.setattr(stack, "_get_json", lambda url: {"detail": "Unauthorized"})
    assert stack.job_counts("http://localhost:8000") is None


# ── Pointing the CLI at another stack ─────────────────────────────────────────

class TestApiUrlOverride:
    """`HEULISTIC_API_URL` lets the CLI talk to a stack it did not start.

    Without it the CLI can only ever see `~/.heulistic/local_compose.yml`, so a
    stack brought up from the repo's own docker-compose.yml — the only way to
    exercise working-tree changes — is invisible to `train` and `download`.
    """

    def test_override_wins_over_compose(self, ps, monkeypatch):
        ps(NDJSON)  # compose says port 8000
        monkeypatch.setenv("HEULISTIC_API_URL", "http://localhost:8002")
        assert stack.api_url() == "http://localhost:8002"

    def test_override_used_when_no_stack_is_running(self, ps, monkeypatch):
        ps(None)
        monkeypatch.setenv("HEULISTIC_API_URL", "http://localhost:8002")
        assert stack.api_url() == "http://localhost:8002"

    def test_trailing_slash_is_trimmed(self, ps, monkeypatch):
        ps(None)
        monkeypatch.setenv("HEULISTIC_API_URL", "http://localhost:8002/")
        assert stack.api_url() == "http://localhost:8002"

    def test_blank_override_is_ignored(self, ps, monkeypatch):
        ps(NDJSON)
        monkeypatch.setenv("HEULISTIC_API_URL", "")
        assert stack.api_url() == "http://localhost:8000"

    def test_is_overridden_reports_the_source(self, monkeypatch):
        monkeypatch.delenv("HEULISTIC_API_URL", raising=False)
        assert stack.api_url_is_overridden() is False
        monkeypatch.setenv("HEULISTIC_API_URL", "http://x:1")
        assert stack.api_url_is_overridden() is True
