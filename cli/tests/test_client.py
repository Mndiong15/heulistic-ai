"""HTTP client for the local API.

Uses only the standard library so `pip install heulistic` stays light — which
means multipart bodies are built by hand, hence the encoder tests below.
"""

import json

import pytest

from heulistic import client


# ── Multipart encoding ────────────────────────────────────────────────────────

class TestMultipart:
    def test_encodes_a_single_field(self):
        content_type, body = client.encode_multipart({"hf_token": "hf_abc"}, {})
        assert content_type.startswith("multipart/form-data; boundary=")
        text = body.decode()
        assert 'name="hf_token"' in text
        assert "hf_abc" in text

    def test_encodes_a_file(self):
        _, body = client.encode_multipart({}, {"config": ("config.yaml", b"base_model: x\n")})
        text = body.decode()
        assert 'name="config"' in text
        assert 'filename="config.yaml"' in text
        assert "base_model: x" in text

    def test_omits_empty_fields(self):
        _, body = client.encode_multipart({"hf_token": None, "instance_type": ""}, {})
        text = body.decode()
        assert "hf_token" not in text
        assert "instance_type" not in text

    def test_body_is_terminated_with_the_closing_boundary(self):
        content_type, body = client.encode_multipart({"a": "b"}, {})
        boundary = content_type.split("boundary=")[1]
        assert body.endswith(f"--{boundary}--\r\n".encode())

    def test_binary_content_survives(self):
        blob = bytes(range(256))
        _, body = client.encode_multipart({}, {"dataset": ("d.jsonl", blob)})
        assert blob in body


# ── Requests ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_http(monkeypatch):
    """Capture outgoing requests and return canned responses."""
    calls = []

    def _install(response=None, error=None):
        def _fake(method, url, *, body=None, content_type=None, stream=False):
            calls.append({"method": method, "url": url, "body": body,
                          "content_type": content_type, "stream": stream})
            if error:
                raise error
            return response

        monkeypatch.setattr(client, "_request", _fake)
        return calls

    return _install


class TestSubmit:
    def test_posts_multipart_to_jobs(self, fake_http, tmp_path):
        calls = fake_http(response=json.dumps({"job_id": "abc", "status": "pending"}).encode())
        cfg = tmp_path / "c.yaml"
        cfg.write_text("base_model: x\n")

        job = client.submit("http://localhost:8000", cfg)

        assert job["job_id"] == "abc"
        assert calls[0]["method"] == "POST"
        assert calls[0]["url"] == "http://localhost:8000/api/jobs"
        assert calls[0]["content_type"].startswith("multipart/form-data")

    def test_includes_dataset_and_token(self, fake_http, tmp_path):
        calls = fake_http(response=b'{"job_id": "abc"}')
        cfg = tmp_path / "c.yaml"; cfg.write_text("base_model: x\n")
        ds = tmp_path / "d.jsonl"; ds.write_text('{"a": 1}\n')

        client.submit("http://localhost:8000", cfg, dataset=ds, hf_token="hf_z")

        body = calls[0]["body"].decode()
        assert 'filename="d.jsonl"' in body
        assert "hf_z" in body


class TestJobQueries:
    def test_get_job(self, fake_http):
        fake_http(response=b'{"job_id": "abc", "status": "completed"}')
        assert client.get_job("http://x", "abc")["status"] == "completed"

    def test_download_url(self, fake_http):
        fake_http(response=b'{"url": "http://localhost:8001/local-download/models/abc/model.tar.gz"}')
        assert client.download_url("http://x", "abc").endswith("model.tar.gz")


# ── Log streaming ─────────────────────────────────────────────────────────────

class TestStreamLogs:
    def _sse(self, *payloads):
        return [f"data: {json.dumps(p)}".encode() for p in payloads] + [b""]

    def test_yields_log_lines(self, monkeypatch):
        monkeypatch.setattr(
            client, "_stream_lines",
            lambda url: iter(self._sse({"line": "step 1"}, {"line": "step 2"})),
        )
        assert list(client.stream_logs("http://x", "abc")) == [
            ("line", "step 1"),
            ("line", "step 2"),
        ]

    def test_terminal_event_ends_the_stream(self, monkeypatch):
        monkeypatch.setattr(
            client, "_stream_lines",
            lambda url: iter(self._sse({"line": "a"}, {"event": "done", "status": "completed"})),
        )
        assert list(client.stream_logs("http://x", "abc")) == [
            ("line", "a"),
            ("done", "completed"),
        ]

    def test_ignores_keepalives_and_malformed_frames(self, monkeypatch):
        monkeypatch.setattr(
            client, "_stream_lines",
            lambda url: iter([b"", b": keepalive", b"data: {not json", b'data: {"line": "ok"}']),
        )
        assert list(client.stream_logs("http://x", "abc")) == [("line", "ok")]


# ── Errors ────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_api_error_detail_is_surfaced(self, fake_http, tmp_path):
        fake_http(error=client.ApiError("Insufficient credits"))
        cfg = tmp_path / "c.yaml"; cfg.write_text("base_model: x\n")
        with pytest.raises(client.ApiError, match="Insufficient credits"):
            client.submit("http://localhost:8000", cfg)
