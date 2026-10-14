"""Pinning container images to the installed CLI version.

The images used to be pinned to `:latest`, so `pip install heulistic==0.1.3`
ran whatever `latest` happened to be. A stale cached image then masked a
missing release for weeks: `_pull_images` found it, printed "Using cached", and
never checked whether anything newer existed.

Pinning to the release version turns that silent mismatch into an explicit
pull failure, which is the whole point.
"""

import pytest

from heulistic import local_runner


@pytest.fixture
def installed_version(monkeypatch):
    def _set(version):
        monkeypatch.setattr(local_runner, "_installed_version", lambda: version)

    return _set


# ── Deriving the tag ──────────────────────────────────────────────────────────

def test_tag_comes_from_the_installed_version(installed_version, monkeypatch):
    monkeypatch.delenv("HEULISTIC_IMAGE_TAG", raising=False)
    installed_version("0.1.3")
    assert local_runner.image_tag() == "v0.1.3"


def test_an_explicit_override_wins(installed_version, monkeypatch):
    monkeypatch.delenv("HEULISTIC_IMAGE_TAG", raising=False)
    installed_version("0.1.3")
    assert local_runner.image_tag("latest") == "latest"
    assert local_runner.image_tag("v0.1.2") == "v0.1.2"


def test_the_environment_variable_is_honoured(installed_version, monkeypatch):
    installed_version("0.1.3")
    monkeypatch.setenv("HEULISTIC_IMAGE_TAG", "latest")
    assert local_runner.image_tag() == "latest"


def test_an_explicit_argument_beats_the_environment(installed_version, monkeypatch):
    installed_version("0.1.3")
    monkeypatch.setenv("HEULISTIC_IMAGE_TAG", "latest")
    assert local_runner.image_tag("v0.1.1") == "v0.1.1"


def test_falls_back_to_latest_when_the_version_is_unknown(installed_version, monkeypatch):
    # Running from a source checkout with nothing installed.
    monkeypatch.delenv("HEULISTIC_IMAGE_TAG", raising=False)
    installed_version(None)
    assert local_runner.image_tag() == "latest"


def test_a_blank_environment_variable_is_ignored(installed_version, monkeypatch):
    installed_version("0.1.3")
    monkeypatch.setenv("HEULISTIC_IMAGE_TAG", "")
    assert local_runner.image_tag() == "v0.1.3"


# ── Feeding it to compose ─────────────────────────────────────────────────────

def test_env_carries_the_tag(installed_version, monkeypatch):
    monkeypatch.delenv("HEULISTIC_IMAGE_TAG", raising=False)
    installed_version("0.1.3")
    env = local_runner._env(3000, 8000, None, True, image_tag="v0.1.3")
    assert env["HEULISTIC_IMAGE_TAG"] == "v0.1.3"


def test_compose_file_substitutes_the_tag():
    """Every heulistic image in the shipped compose file must be templated.

    A literal `:latest` left behind would silently opt that one service out of
    pinning — which is exactly how frontend-local went stale unnoticed.
    """
    text = (local_runner._COMPOSE_SRC).read_text()
    literal = [
        line.strip()
        for line in text.splitlines()
        if "heulistic/" in line and ":latest" in line
    ]
    assert literal == [], literal
    for service in ("api", "scheduler", "frontend-local", "training"):
        assert f"heulistic/{service}:${{HEULISTIC_IMAGE_TAG" in text, service


# ── Pulling ───────────────────────────────────────────────────────────────────

def test_pull_list_uses_the_resolved_tag(monkeypatch):
    seen = []
    monkeypatch.setattr(local_runner, "_image_exists_locally", lambda i: True)
    monkeypatch.setattr(local_runner, "_say", lambda m: seen.append(m))
    local_runner._pull_images("v0.1.3")
    joined = " ".join(str(s) for s in seen)
    assert "heulistic/api:v0.1.3" in joined
    assert "heulistic/frontend-local:v0.1.3" in joined
    assert ":latest" not in joined


def test_a_missing_pinned_image_explains_the_version_link(monkeypatch, capsys):
    monkeypatch.setattr(local_runner, "_image_exists_locally", lambda i: False)
    monkeypatch.setattr(local_runner.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 1})())
    monkeypatch.setattr(local_runner.time, "sleep", lambda s: None)

    with pytest.raises(SystemExit):
        local_runner._pull_images("v9.9.9")

    message = capsys.readouterr().err
    assert "v9.9.9" in message
    # The likely cause is a CLI newer than the published images.
    assert "HEULISTIC_IMAGE_TAG" in message or "--image-tag" in message
