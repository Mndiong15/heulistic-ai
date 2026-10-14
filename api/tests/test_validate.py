"""Tests for POST /api/validate.

The endpoint exists so both frontends can stop carrying their own copy of the
Axolotl rules. It returns every diagnostic — errors, warnings and hints — and
never rejects a request for a bad config: a 422 would make it impossible to
show the user what is wrong.
"""

import pytest
from httpx import AsyncClient

from tests.conftest import VALID_CONFIG

pytestmark = pytest.mark.asyncio


async def _validate(client: AsyncClient, config: str, **body):
    return await client.post("/api/validate", json={"config_yaml": config, **body})


class TestValidateEndpoint:
    async def test_valid_config_is_ok(self, client: AsyncClient):
        resp = await _validate(client, VALID_CONFIG)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["counts"]["error"] == 0

    async def test_a_broken_config_is_still_a_200(self, client: AsyncClient):
        # The point is to report problems, not to refuse to answer.
        resp = await _validate(client, "base_model: org/m-7b\nnum_epochs: 0\n")
        assert resp.status_code == 200
        assert resp.json()["ok"] is False

    async def test_diagnostics_carry_code_level_and_field(self, client: AsyncClient):
        resp = await _validate(client, "base_model: org/m-7b\nnum_epochs: 0\n")
        codes = {d["code"] for d in resp.json()["diagnostics"]}
        assert "epochs.invalid" in codes
        first = resp.json()["diagnostics"][0]
        assert {"level", "code", "message", "field", "line"} <= set(first)

    async def test_line_numbers_are_reported(self, client: AsyncClient):
        resp = await _validate(client, "base_model: org/m-7b\nlearning_rate: 0.1\nbogus: 1\n")
        bogus = next(d for d in resp.json()["diagnostics"] if d["field"] == "bogus")
        assert bogus["line"] == 3

    async def test_counts_are_grouped_by_level(self, client: AsyncClient):
        resp = await _validate(client, "base_model: org/m-7b\nnum_epochs: 0\nbogus: 1\n")
        counts = resp.json()["counts"]
        assert counts["error"] >= 1
        assert counts["warning"] >= 1
        assert set(counts) == {"error", "warning", "info"}

    async def test_yaml_syntax_error_is_reported_not_raised(self, client: AsyncClient):
        resp = await _validate(client, "base_model: [unclosed\n")
        assert resp.status_code == 200
        assert resp.json()["diagnostics"][0]["code"] == "yaml.syntax"

    async def test_instance_type_enables_the_vram_check(self, client: AsyncClient):
        # A 70B full fine-tune cannot fit a single 22 GB A10G.
        resp = await _validate(
            client,
            "base_model: org/m-70b\nlearning_rate: 0.0002\n",
            instance_type="g5.xlarge",
        )
        codes = {d["code"] for d in resp.json()["diagnostics"]}
        assert "vram.over" in codes

    async def test_without_an_instance_type_there_is_no_vram_check(self, client: AsyncClient):
        resp = await _validate(client, "base_model: org/m-70b\nlearning_rate: 0.0002\n")
        codes = {d["code"] for d in resp.json()["diagnostics"]}
        assert not any(c.startswith("vram.") for c in codes)

    async def test_unknown_instance_type_is_rejected(self, client: AsyncClient):
        resp = await _validate(
            client, VALID_CONFIG, instance_type="not-an-instance"
        )
        assert resp.status_code == 422

    async def test_single_node_multi_gpu_is_not_an_error(self, client: AsyncClient):
        resp = await _validate(client, VALID_CONFIG + "world_size: 4\n")
        assert resp.json()["ok"] is True
