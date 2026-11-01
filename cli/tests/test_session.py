"""Which backend a command talks to: the local stack or Heulistic Cloud.

Logged out (the default), everything goes to the local stack, as it always
has. After `heulistic login`, commands go to the cloud with the saved API key;
`--local` sends one command to the local stack instead.
"""

import os
import stat

import pytest

from heulistic import client, session, stack


@pytest.fixture
def local_stack(monkeypatch):
    monkeypatch.setattr(stack, "api_url", lambda: "http://localhost:8000")


def test_logged_out_uses_the_local_stack(local_stack):
    s = session.resolve()
    assert (s.api_url, s.token, s.cloud) == ("http://localhost:8000", client.DEV_TOKEN, False)


def test_logged_in_uses_the_cloud(local_stack):
    session.save("hk_secret", "https://api.heulistic.com")
    s = session.resolve()
    assert (s.api_url, s.token, s.cloud) == ("https://api.heulistic.com", "hk_secret", True)


def test_local_flag_overrides_a_login(local_stack):
    session.save("hk_secret", "https://api.heulistic.com")
    s = session.resolve(local=True)
    assert (s.token, s.cloud) == (client.DEV_TOKEN, False)


def test_an_env_key_logs_in_without_a_file(local_stack, monkeypatch):
    # For CI and servers: no `heulistic login` step, nothing written to disk.
    monkeypatch.setenv(session.KEY_ENV, "hk_from_env")
    s = session.resolve()
    assert (s.api_url, s.token, s.cloud) == (session.CLOUD_API_URL, "hk_from_env", True)


def test_the_saved_key_is_readable_only_by_you():
    path = session.save("hk_secret", "https://api.heulistic.com")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_logout_forgets_the_key(local_stack):
    session.save("hk_secret", "https://api.heulistic.com")
    assert session.clear() is True
    assert session.resolve().cloud is False
    assert session.clear() is False  # already logged out


def test_requests_carry_the_sessions_token(monkeypatch):
    client.use(session.Session("https://api.heulistic.com", "hk_secret", cloud=True))
    assert client._headers()["Authorization"] == "Bearer hk_secret"
    client.use(session.Session("http://localhost:8000", client.DEV_TOKEN, cloud=False))
    assert client._headers()["Authorization"] == f"Bearer {client.DEV_TOKEN}"
