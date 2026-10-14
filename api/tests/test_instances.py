"""Tests for GET /api/instances."""

import pytest
from httpx import AsyncClient

from app.services.cost import MARKUP

_MOCK_CATALOG = {
    "g4dn.xlarge":  {"gpus": 1, "vram_gb": 16,  "on_demand_rate": 0.526},
    "g5.xlarge":    {"gpus": 1, "vram_gb": 24,  "on_demand_rate": 1.006},
    "g5.12xlarge":  {"gpus": 4, "vram_gb": 96,  "on_demand_rate": 5.672},
    "g5.48xlarge":  {"gpus": 8, "vram_gb": 192, "on_demand_rate": 16.288},
    "p4d.24xlarge": {"gpus": 8, "vram_gb": 320, "on_demand_rate": 32.77},
}


@pytest.fixture(autouse=True)
def mock_catalog(monkeypatch):
    monkeypatch.setattr("app.routers.instances.get_catalog", lambda: dict(_MOCK_CATALOG))


class TestListInstances:
    async def test_returns_200(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        assert resp.status_code == 200

    async def test_returns_all_catalog_entries(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        data = resp.json()
        assert len(data) == len(_MOCK_CATALOG)

    async def test_response_shape(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        for item in resp.json():
            assert "instance_type" in item
            assert "gpus" in item
            assert "vram_gb" in item
            assert "our_rate" in item
            assert "on_demand_rate" not in item, "raw AWS price must not be exposed to users"

    async def test_our_rate_is_marked_up(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        by_type = {item["instance_type"]: item for item in resp.json()}
        for itype, info in _MOCK_CATALOG.items():
            expected = round(info["on_demand_rate"] * float(MARKUP), 3)
            assert abs(by_type[itype]["our_rate"] - expected) < 0.001, (
                f"{itype}: our_rate {by_type[itype]['our_rate']} != {expected}"
            )

    async def test_sorted_by_our_rate_ascending(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        rates = [item["our_rate"] for item in resp.json()]
        assert rates == sorted(rates)

    async def test_gpu_and_vram_fields_are_positive(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        for item in resp.json():
            assert item["gpus"] >= 1
            assert item["vram_gb"] > 0

    async def test_no_auth_required(self, anon_client: AsyncClient):
        # /api/instances is a public endpoint — no session header needed
        resp = await anon_client.get("/api/instances")
        assert resp.status_code == 200

    async def test_multi_gpu_instances_present(self, anon_client: AsyncClient):
        resp = await anon_client.get("/api/instances")
        multi = [i for i in resp.json() if i["gpus"] > 1]
        assert len(multi) > 0


# Bound at import, before conftest swaps the module attribute for an offline stub.
from app.services.instances import _fetch_gpu_specs  # noqa: E402


class _FakeEC2:
    """The shape ec2:DescribeInstanceTypes returns for one 1-GPU and one 4-GPU box."""

    def __init__(self, pages):
        self._pages = pages

    def get_paginator(self, _name):
        return self

    def paginate(self, **_kwargs):
        return self._pages


def _instance(itype, count, per_gpu_mib):
    return {
        "InstanceType": itype,
        "GpuInfo": {
            "Gpus": [{"Count": count, "MemoryInfo": {"SizeInMiB": per_gpu_mib}}],
            "TotalGpuMemoryInMiB": count * per_gpu_mib,
        },
    }


class TestFetchGpuSpecs:
    def test_vram_is_the_instance_total_not_one_card(self, monkeypatch):
        """`MemoryInfo.SizeInMiB` is per card. Selection plans per GPU by
        dividing the total by the GPU count, so storing one card's memory here
        made a 4 × 44 GB g6e.12xlarge look like 4 × 11 GB in production — while
        the fallback table, which the tests use, already stored totals."""
        pages = [{"InstanceTypes": [
            _instance("g6e.xlarge", 1, 46068),
            _instance("g6e.12xlarge", 4, 46068),
        ]}]
        monkeypatch.setattr("app.services.instances.boto3.client", lambda *a, **k: _FakeEC2(pages))

        specs = _fetch_gpu_specs("us-east-1")

        assert specs["g6e.xlarge"] == {"gpus": 1, "vram_gb": 44}
        assert specs["g6e.12xlarge"] == {"gpus": 4, "vram_gb": 179}

    def test_total_is_derived_when_aws_omits_it(self, monkeypatch):
        inst = _instance("g5.12xlarge", 4, 23028)
        del inst["GpuInfo"]["TotalGpuMemoryInMiB"]
        monkeypatch.setattr(
            "app.services.instances.boto3.client",
            lambda *a, **k: _FakeEC2([{"InstanceTypes": [inst]}]),
        )

        assert _fetch_gpu_specs("us-east-1")["g5.12xlarge"]["vram_gb"] == 89
