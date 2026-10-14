"""Local-mode download URLs.

In local mode there is no S3 presigning — the API serves the file itself and
hands back a plain URL. The browser opens that URL directly, so it has to
point at wherever the API is actually published.

Note: conftest installs an autouse `mock_s3` fixture that replaces
`generate_presigned_url` for every test. The real function is bound here at
import time, before any fixture runs, so these tests exercise the
implementation rather than the stub.
"""

import pytest

from app import config
from app.services.s3 import generate_presigned_url as real_generate_presigned_url


@pytest.fixture
def local_mode(monkeypatch):
    def _configure(api_public_url="http://localhost:8000"):
        monkeypatch.setattr(config.settings, "local_mode", True)
        monkeypatch.setattr(config.settings, "api_public_url", api_public_url)

    return _configure


class TestLocalDownloadUrl:
    def test_uses_the_configured_api_url(self, local_mode):
        local_mode("http://localhost:8000")
        assert real_generate_presigned_url("models/abc/model.tar.gz") == (
            "http://localhost:8000/local-download/models/abc/model.tar.gz"
        )

    def test_honours_a_non_default_port(self, local_mode):
        """`heulistic start --api-port 8001` must not yield dead links.

        The URL was hardcoded to port 8000, so any custom port produced a
        download button pointing at nothing.
        """
        local_mode("http://localhost:8001")
        assert real_generate_presigned_url("models/abc/model.tar.gz").startswith(
            "http://localhost:8001/local-download/"
        )

    def test_trailing_slash_is_not_doubled(self, local_mode):
        local_mode("http://localhost:8000/")
        assert "//local-download" not in real_generate_presigned_url("k")

    def test_works_behind_a_hostname(self, local_mode):
        local_mode("http://my-box.local:9000")
        assert real_generate_presigned_url("k") == (
            "http://my-box.local:9000/local-download/k"
        )

    def test_cloud_mode_still_presigns(self, monkeypatch):
        """Outside local mode the S3 presigning path is still taken.

        Dummy credentials are injected because signing needs them, and CI has
        none — the first version of this test passed only on a machine that
        happened to hold real AWS credentials.
        """
        from app.services import s3 as s3_module

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        # The boto client is cached module-wide; drop it so it is rebuilt with
        # the dummy credentials above.
        monkeypatch.setattr(s3_module, "_client", None)
        monkeypatch.setattr(config.settings, "local_mode", False)

        url = real_generate_presigned_url("models/abc/model.tar.gz")
        assert "local-download" not in url
        assert url.startswith("https://")
        assert "X-Amz-Signature" in url
