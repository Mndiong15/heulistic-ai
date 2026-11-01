"""Shared CLI test setup."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_login(monkeypatch, tmp_path):
    """Keep tests away from the developer's own `heulistic login`.

    Logged in, the CLI sends commands to Heulistic Cloud; a real credentials
    file or HEULISTIC_API_KEY on the test machine would quietly reroute every
    local-stack test.
    """
    from heulistic import client, session

    monkeypatch.setattr(session, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.delenv(session.KEY_ENV, raising=False)
    # A command run by an earlier test may have switched the client to a cloud key.
    monkeypatch.setattr(client, "_token", client.DEV_TOKEN)
