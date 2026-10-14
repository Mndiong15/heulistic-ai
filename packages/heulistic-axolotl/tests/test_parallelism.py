"""Planning how a job uses the GPUs it is given.

The bug this exists to fix: instance selection and the validator compared the
VRAM requirement against the *total* across all GPUs. Training ran plain DDP,
where every GPU holds a full copy of the model — so a job needing 48 GB was
judged to fit a 4x16 GB box and then OOMed on all four cards, after being
billed for the attempt.

A plan compares against VRAM *per GPU*, and shards with ZeRO-3 when a model
does not fit a single card — so the multi-GPU boxes become usable rather than
a trap.
"""

import pytest

from heulistic_axolotl import Hardware, plan_parallelism

QLORA_70B = {"base_model": "org/m-70b", "adapter": "qlora", "load_in_4bit": True}  # 42 GB
LORA_7B = {"base_model": "org/m-7b", "adapter": "lora"}                            # 16.8 GB
LORA_13B = {"base_model": "org/m-13b", "adapter": "lora"}                          # 31.2 GB
LORA_20B = {"base_model": "org/m-20b", "adapter": "lora"}                          # 48 GB
FULL_7B = {"base_model": "org/m-7b"}                                               # 134.4 GB

G5_12XL = Hardware(total_vram_gb=88, gpu_count=4, label="g5.12xlarge")    # 4x22
G4DN_12XL = Hardware(total_vram_gb=64, gpu_count=4, label="g4dn.12xlarge")  # 4x16
G6E_12XL = Hardware(total_vram_gb=176, gpu_count=4, label="g6e.12xlarge")  # 4x44


# ── Hardware ──────────────────────────────────────────────────────────────────

def test_per_gpu_vram():
    assert G5_12XL.per_gpu_vram_gb == 22
    assert Hardware(total_vram_gb=24, gpu_count=1).per_gpu_vram_gb == 24


# ── The routing bug ───────────────────────────────────────────────────────────

def test_total_vram_is_not_per_gpu_vram():
    """20B LoRA (48 GB) on 4x16 GB: the old total check said 48 <= 64, fits.
    Under DDP each 16 GB card needs all 48 GB. It only works sharded."""
    plan = plan_parallelism(LORA_20B, G4DN_12XL)
    assert plan.strategy == "zero3"
    assert plan.auto_sharded is True
    assert plan.per_gpu_gb == pytest.approx(12.0)
    assert plan.fits is True


def test_a_quantized_model_is_never_auto_sharded():
    """70B QLoRA (42 GB) on 4x22 GB: the old total check said 42 <= 88, fits.
    QLoRA with ZeRO-3 is unreliable, so rather than launch something flaky it
    must say plainly that it does not fit."""
    plan = plan_parallelism(QLORA_70B, G5_12XL)
    assert plan.auto_sharded is False
    assert plan.fits is False
    assert plan.verdict == "over"
    assert "quantiz" in plan.reason.lower()


# ── Strategy choice ───────────────────────────────────────────────────────────

def test_single_gpu_box_is_single():
    plan = plan_parallelism(LORA_7B, Hardware(total_vram_gb=22, gpu_count=1))
    assert plan.strategy == "single"
    assert plan.per_gpu_gb == pytest.approx(16.8)


def test_fits_one_gpu_so_ddp_replicates_it():
    plan = plan_parallelism(LORA_7B, G5_12XL)
    assert plan.strategy == "ddp"
    assert plan.auto_sharded is False
    assert plan.per_gpu_gb == pytest.approx(16.8)  # a full copy per card
    assert plan.fits is True


def test_too_big_for_one_gpu_is_sharded():
    plan = plan_parallelism(FULL_7B, G6E_12XL)
    assert plan.strategy == "zero3"
    assert plan.auto_sharded is True
    assert plan.per_gpu_gb == pytest.approx(33.6)
    assert plan.fits is True


def test_too_big_even_sharded_does_not_fit():
    # 7B full fine-tune sharded across 4x22 GB is still 33.6 GB per card.
    plan = plan_parallelism(FULL_7B, G5_12XL)
    assert plan.fits is False
    assert plan.verdict == "over"


# ── Respecting the user's own config ──────────────────────────────────────────

def test_user_fsdp_is_respected_and_sharded():
    plan = plan_parallelism({**LORA_20B, "fsdp": ["full_shard"]}, G4DN_12XL)
    assert plan.strategy == "fsdp"
    assert plan.auto_sharded is False
    assert plan.per_gpu_gb == pytest.approx(12.0)
    assert plan.exact is True


def test_user_zero3_is_recognised_from_the_path():
    config = {**LORA_20B, "deepspeed": "deepspeed_configs/zero3_bf16.json"}
    plan = plan_parallelism(config, G4DN_12XL)
    assert plan.strategy == "zero3"
    assert plan.auto_sharded is False
    assert plan.per_gpu_gb == pytest.approx(12.0)


def test_user_zero2_keeps_weights_replicated_and_is_not_exact():
    """ZeRO-1/2 shard optimiser state but not weights, which this estimate
    cannot separate — so it is modelled as replicated and marked inexact
    rather than given a confident number."""
    config = {**LORA_7B, "deepspeed": "deepspeed_configs/zero2.json"}
    plan = plan_parallelism(config, G5_12XL)
    assert plan.strategy == "deepspeed"
    assert plan.exact is False
    assert plan.per_gpu_gb == pytest.approx(16.8)


# ── Thresholds ────────────────────────────────────────────────────────────────

def test_fits_and_verdict_draw_the_same_line():
    """Selection reads `fits`, the validator reads `verdict`. When they used
    different lines, 70B QLoRA on a 44 GB card (95.5% full) was rented by the
    API and blocked by the UI. It fits — QLoRA's headline result is a 65B
    model on one 48 GB card — and is tight."""
    plan = plan_parallelism(QLORA_70B, Hardware(total_vram_gb=44, gpu_count=1))
    assert plan.fits is True
    assert plan.verdict == "tight"


@pytest.mark.parametrize("vram", [40, 41.9, 42, 42.1, 44, 60])
def test_fits_exactly_when_the_verdict_is_not_over(vram):
    plan = plan_parallelism(QLORA_70B, Hardware(total_vram_gb=vram, gpu_count=1))
    assert plan.fits == (plan.verdict != "over")


def test_tight_is_reported():
    plan = plan_parallelism(LORA_7B, Hardware(total_vram_gb=22, gpu_count=1))
    assert plan.verdict == "tight"  # 16.8 / 22 = 76%
