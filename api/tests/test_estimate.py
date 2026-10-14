"""Tests for POST /api/estimate."""

import pytest
from httpx import AsyncClient


LLAMA_LORA = """\
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
adapter: lora
num_epochs: 3
"""

MISTRAL_FULL = """\
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
adapter: null
num_epochs: 2
"""


class TestEstimate:
    async def test_lora_config_returns_small_instance(self, client: AsyncClient):
        resp = await client.post(
            "/api/estimate",
            json={"config_yaml": LLAMA_LORA, "dataset_size_mb": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["instance_type"] == "g6.xlarge"
        assert data["estimated_cost"] is not None
        assert data["estimated_duration_minutes"] > 0
        assert data["hourly_rate"] is not None

    async def test_full_finetune_returns_large_instance(self, client: AsyncClient):
        """A full fine-tune needs the biggest box we offer.

        Mistral-7B with no adapter is 7B x 16 bytes x 1.2 = ~134 GB of VRAM,
        so the only instance that fits is g6e.12xlarge (4x L40S, 176 GB). The
        assertion previously named g6.2xlarge, which has 22 GB — a tenth of
        what the config needs, and not a "large instance" at all.
        """
        resp = await client.post(
            "/api/estimate",
            json={"config_yaml": MISTRAL_FULL},
        )
        assert resp.status_code == 200
        assert resp.json()["instance_type"] == "g6e.12xlarge"

    async def test_larger_dataset_increases_duration(self, client: AsyncClient):
        r_small = await client.post(
            "/api/estimate",
            json={"config_yaml": LLAMA_LORA, "dataset_size_mb": 1},
        )
        r_large = await client.post(
            "/api/estimate",
            json={"config_yaml": LLAMA_LORA, "dataset_size_mb": 500},
        )
        assert r_large.json()["estimated_duration_minutes"] > r_small.json()["estimated_duration_minutes"]

    async def test_no_session_id_still_works(self, anon_client: AsyncClient):
        # /api/estimate does not require a session ID
        resp = await anon_client.post(
            "/api/estimate",
            json={"config_yaml": LLAMA_LORA},
        )
        assert resp.status_code == 200

    async def test_invalid_yaml_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/estimate",
            json={"config_yaml": "not: valid: yaml: [[["},
        )
        assert resp.status_code == 422

    async def test_non_dict_yaml_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/estimate",
            json={"config_yaml": "- item\n- another\n"},
        )
        assert resp.status_code == 422
