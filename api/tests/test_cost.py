"""Unit tests for the cost estimation service (no DB or HTTP needed)."""

from decimal import Decimal

import pytest

from app.services.cost import (
    _estimate_num_samples,
    _framework_tax_seconds,
    _parse_param_billions,
    _training_bits,
    _vram_required_gb,
    estimate_cost,
    estimate_duration_minutes,
    parse_and_estimate,
    select_instance,
)
from app.services.instances import get_rate


class TestParseParamBillions:
    def test_7b_llama(self):
        assert _parse_param_billions("meta-llama/Llama-2-7b-hf") == 7.0

    def test_13b_llama(self):
        assert _parse_param_billions("meta-llama/Llama-2-13b-hf") == 13.0

    def test_70b_llama(self):
        assert _parse_param_billions("meta-llama/Llama-2-70b-hf") == 70.0

    def test_uppercase_b(self):
        assert _parse_param_billions("mistralai/Mistral-7B-v0.1") == 7.0

    def test_decimal_params(self):
        assert _parse_param_billions("some/model-3.8b") == 3.8

    def test_empty_string_defaults_to_7(self):
        assert _parse_param_billions("") == 7.0

    def test_unrecognised_name_defaults_to_7(self):
        assert _parse_param_billions("some/model-without-size") == 7.0

    def test_8b_llama3(self):
        assert _parse_param_billions("meta-llama/Meta-Llama-3-8B") == 8.0


class TestTrainingBits:
    def test_qlora_adapter_implies_4bit(self):
        assert _training_bits({"adapter": "qlora"}) == 4

    def test_explicit_load_in_4bit(self):
        assert _training_bits({"adapter": "lora", "load_in_4bit": True}) == 4

    def test_load_in_8bit(self):
        assert _training_bits({"adapter": "lora", "load_in_8bit": True}) == 8

    def test_lora_defaults_to_16bit(self):
        assert _training_bits({"adapter": "lora"}) == 16

    def test_full_finetune_defaults_to_16bit(self):
        assert _training_bits({}) == 16

    def test_4bit_flag_overrides_adapter(self):
        assert _training_bits({"adapter": "lora", "load_in_4bit": True}) == 4


class TestVramRequiredGb:
    def test_qlora_7b(self):
        # 7 × 0.5 bytes × 1.2 = 4.2 GB
        result = _vram_required_gb({"adapter": "qlora", "base_model": "model-7b"})
        assert abs(result - 4.2) < 0.01

    def test_lora_bf16_7b(self):
        # 7 × 2.0 bytes × 1.2 = 16.8 GB
        result = _vram_required_gb({"adapter": "lora", "base_model": "model-7b"})
        assert abs(result - 16.8) < 0.01

    def test_lora_8bit_7b(self):
        # 7 × 1.0 bytes × 1.2 = 8.4 GB
        result = _vram_required_gb({"adapter": "lora", "load_in_8bit": True, "base_model": "model-7b"})
        assert abs(result - 8.4) < 0.01

    def test_full_finetune_7b(self):
        # 7 × 16.0 bytes × 1.2 = 134.4 GB
        result = _vram_required_gb({"base_model": "model-7b"})
        assert abs(result - 134.4) < 0.01

    def test_full_finetune_70b(self):
        # 70 × 16.0 × 1.2 = 1344 GB — far exceeds any instance
        result = _vram_required_gb({"base_model": "model-70b"})
        assert result > 1000

    def test_qlora_scales_with_param_count(self):
        small = _vram_required_gb({"adapter": "qlora", "base_model": "model-7b"})
        large = _vram_required_gb({"adapter": "qlora", "base_model": "model-70b"})
        assert large > small

    def test_full_finetune_needs_more_than_lora(self):
        lora = _vram_required_gb({"adapter": "lora", "base_model": "model-7b"})
        full = _vram_required_gb({"base_model": "model-7b"})
        assert full > lora


class TestSelectInstance:
    def test_lora_bf16_7b_picks_22gb_instance(self):
        # 7B LoRA bf16 = 16.8 GB → needs 22 GB → cheapest 22 GB is g6.xlarge
        assert select_instance({"adapter": "lora", "base_model": "meta-llama/Llama-2-7b-hf"}) == "g6.xlarge"

    def test_qlora_7b_picks_cheapest_fitting(self):
        # 7B QLoRA = 4.2 GB → 16 GB instances are sufficient → g4dn.xlarge is cheapest
        assert select_instance({"adapter": "qlora", "base_model": "mistralai/Mistral-7B-v0.1"}) == "g4dn.xlarge"

    def test_qlora_cheaper_than_lora_same_model(self):
        qlora = select_instance({"adapter": "qlora", "base_model": "model-7b"})
        lora = select_instance({"adapter": "lora", "base_model": "model-7b"})
        from app.services.instances import get_catalog
        catalog = get_catalog()
        assert catalog[qlora]["on_demand_rate"] < catalog[lora]["on_demand_rate"]

    def test_full_finetune_7b_needs_large_instance(self):
        # 7B full = 134.4 GB → only g6e.12xlarge (176 GB) fits
        assert select_instance({"adapter": None, "base_model": "mistralai/Mistral-7B-v0.1"}) == "g6e.12xlarge"

    def test_lora_13b_needs_44gb(self):
        # 13B LoRA bf16 = 31.2 GB → 22 GB is not enough → 44 GB (g6e.xlarge)
        assert select_instance({"adapter": "lora", "base_model": "meta-llama/Llama-2-13b-hf"}) == "g6e.xlarge"

    def test_qlora_70b_fits_in_44gb(self):
        # 70B QLoRA = 42 GB → fits in g6e.xlarge (44 GB), no need for 176 GB
        assert select_instance({"adapter": "qlora", "base_model": "meta-llama/Llama-2-70b-hf"}) == "g6e.xlarge"

    def test_no_adapter_treated_as_full_finetune(self):
        # Missing adapter = full fine-tune
        full = select_instance({"base_model": "mistralai/Mistral-7B-v0.1"})
        assert full == "g6e.12xlarge"

    def test_empty_base_model_still_returns_valid_instance(self):
        result = select_instance({"adapter": "lora", "base_model": ""})
        from app.services.instances import get_catalog
        assert result in get_catalog()

    def test_returns_max_vram_when_nothing_fits(self):
        # 70B full fine-tune = 1344 GB — exceeds all instances → max VRAM instance
        result = select_instance({"adapter": None, "base_model": "model-70b"})
        from app.services.instances import get_catalog
        catalog = get_catalog()
        assert catalog[result]["vram_gb"] == max(v["vram_gb"] for v in catalog.values())


class TestEstimateDuration:
    def test_minimum_floor(self):
        duration = estimate_duration_minutes({"adapter": "lora"}, dataset_size_mb=0)
        assert duration >= 10

    def test_scales_with_dataset_size(self):
        small = estimate_duration_minutes({"adapter": "lora", "num_epochs": 1}, dataset_size_mb=1)
        large = estimate_duration_minutes({"adapter": "lora", "num_epochs": 1}, dataset_size_mb=100)
        assert large > small

    def test_scales_with_epochs(self):
        one_epoch = estimate_duration_minutes({"adapter": "lora", "num_epochs": 1}, dataset_size_mb=10)
        three_epochs = estimate_duration_minutes({"adapter": "lora", "num_epochs": 3}, dataset_size_mb=10)
        assert three_epochs > one_epoch

    def test_full_finetune_slower_than_lora(self):
        lora = estimate_duration_minutes({"adapter": "lora", "num_epochs": 1}, dataset_size_mb=10)
        full = estimate_duration_minutes({"adapter": None, "num_epochs": 1}, dataset_size_mb=10)
        assert full > lora


class TestEstimateCost:
    def test_cost_scales_with_duration(self):
        c30 = estimate_cost("g5.xlarge", 30)
        c60 = estimate_cost("g5.xlarge", 60)
        c120 = estimate_cost("g5.xlarge", 120)
        assert 0 < c30 < c60 < c120

    def test_larger_instance_costs_more(self):
        small = estimate_cost("g5.xlarge", 60)
        large = estimate_cost("g5.2xlarge", 60)
        assert large > small

    def test_returns_decimal(self):
        assert isinstance(estimate_cost("g5.xlarge", 60), Decimal)

    def test_known_rate(self):
        from app.services.cost import MARKUP
        cost = estimate_cost("g5.xlarge", 60)
        expected = get_rate("g5.xlarge") * float(MARKUP)
        assert float(cost) == pytest.approx(expected, rel=0.01)


class TestEstimateNumSamples:
    def test_zero_mb_treated_as_one(self):
        assert _estimate_num_samples(0, 512) == _estimate_num_samples(1, 512)

    def test_longer_sequence_fewer_samples(self):
        short = _estimate_num_samples(100, 256)
        long = _estimate_num_samples(100, 2048)
        assert short > long

    def test_larger_dataset_more_samples(self):
        small = _estimate_num_samples(10, 512)
        large = _estimate_num_samples(100, 512)
        assert large > small

    def test_always_at_least_one(self):
        assert _estimate_num_samples(0, 99999) >= 1


class TestFrameworkTax:
    def test_no_val_set_no_eval_tax(self):
        # Only checkpoint writes (1 epoch, lora → 10 sec)
        tax = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1}, num_raw_samples=1000, total_steps=100
        )
        assert tax == 10.0

    def test_full_finetune_checkpoint_more_expensive(self):
        lora_tax = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1}, num_raw_samples=1000, total_steps=100
        )
        full_tax = _framework_tax_seconds(
            {"adapter": None, "num_epochs": 1}, num_raw_samples=1000, total_steps=100
        )
        assert full_tax > lora_tax

    def test_val_set_adds_eval_tax(self):
        without = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1}, num_raw_samples=1000, total_steps=100
        )
        with_val = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1, "val_set_size": 0.1},
            num_raw_samples=1000,
            total_steps=100,
        )
        assert with_val > without

    def test_eval_steps_overrides_per_epoch_eval(self):
        # eval_steps=25 on 100 total_steps → 4 eval runs
        # default (1 eval per epoch, 1 epoch) → 1 eval run
        few_evals = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1, "val_set_size": 0.1},
            num_raw_samples=1000,
            total_steps=100,
        )
        many_evals = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1, "val_set_size": 0.1, "eval_steps": 25},
            num_raw_samples=1000,
            total_steps=100,
        )
        assert many_evals > few_evals

    def test_saves_per_epoch_adds_checkpoint_overhead(self):
        default = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1}, num_raw_samples=1000, total_steps=100
        )
        many_saves = _framework_tax_seconds(
            {"adapter": "lora", "num_epochs": 1, "saves_per_epoch": 5},
            num_raw_samples=1000,
            total_steps=100,
        )
        assert many_saves > default


class TestEstimateDurationNewFeatures:
    BASE = {"adapter": "lora", "num_epochs": 1, "micro_batch_size": 2}

    def test_grad_accum_does_not_inflate_estimate(self):
        # Old bug: steps were computed without grad_accum, giving an 8× overcount.
        # New algorithm: total GPU iters = ceil(samples / global_batch) * grad_accum,
        # where global_batch = micro_batch * grad_accum — the terms cancel, so the
        # estimate should be approximately the same regardless of grad_accum.
        no_accum = estimate_duration_minutes({**self.BASE}, dataset_size_mb=50)
        with_accum = estimate_duration_minutes(
            {**self.BASE, "gradient_accumulation_steps": 8}, dataset_size_mb=50
        )
        assert with_accum < no_accum * 1.5  # must not be anywhere near 8× higher

    def test_max_steps_caps_estimate(self):
        """max_steps shorter than computed total → estimate is shorter."""
        full = estimate_duration_minutes({**self.BASE, "num_epochs": 3}, dataset_size_mb=100)
        capped = estimate_duration_minutes(
            {**self.BASE, "num_epochs": 3, "max_steps": 10}, dataset_size_mb=100
        )
        assert capped < full

    def test_multi_gpu_faster_than_single(self):
        """g5.12xlarge (4 GPUs) produces a shorter estimate than g5.xlarge."""
        single = estimate_duration_minutes(self.BASE, dataset_size_mb=50, instance_type="g5.xlarge")
        multi = estimate_duration_minutes(self.BASE, dataset_size_mb=50, instance_type="g5.12xlarge")
        assert multi < single

    def test_frequent_evals_add_overhead(self):
        # eval_steps=1 triggers an eval pass every optimizer step — eval overhead
        # then dominates over the training time reduction from a smaller train split.
        no_val = estimate_duration_minutes({**self.BASE, "num_epochs": 1}, dataset_size_mb=50)
        many_evals = estimate_duration_minutes(
            {**self.BASE, "num_epochs": 1, "val_set_size": 0.1, "eval_steps": 1},
            dataset_size_mb=50,
        )
        assert many_evals > no_val

    def test_cold_start_floor(self):
        """Even max_steps=1 includes 5-min cold start; minimum floor is 10 min."""
        result = estimate_duration_minutes({**self.BASE, "max_steps": 1}, dataset_size_mb=1)
        assert result >= 10

    def test_saves_per_epoch_adds_overhead(self):
        default = estimate_duration_minutes(self.BASE, dataset_size_mb=50)
        many_saves = estimate_duration_minutes(
            {**self.BASE, "saves_per_epoch": 10}, dataset_size_mb=50
        )
        assert many_saves > default

    def test_sequence_len_does_not_inflate_estimate(self):
        # training_seconds = total_samples * sequence_len / tps, but
        # total_samples = dataset_MB * 50k / sequence_len — the terms cancel,
        # so total compute is constant. Estimates must be within 1 min of each other.
        short = estimate_duration_minutes({**self.BASE, "sequence_len": 256}, dataset_size_mb=50)
        long = estimate_duration_minutes({**self.BASE, "sequence_len": 4096}, dataset_size_mb=50)
        assert abs(long - short) <= 1


class TestParseAndEstimate:
    LLAMA_LORA = """\
base_model: meta-llama/Llama-2-7b-hf
adapter: lora
num_epochs: 3
"""

    def test_returns_required_keys(self):
        result = parse_and_estimate(self.LLAMA_LORA)
        assert {"instance_type", "estimated_duration_minutes", "estimated_cost", "hourly_rate"} <= result.keys()

    def test_selects_correct_instance(self):
        result = parse_and_estimate(self.LLAMA_LORA)
        assert result["instance_type"] == "g6.xlarge"

    def test_cost_is_positive(self):
        result = parse_and_estimate(self.LLAMA_LORA)
        assert result["estimated_cost"] > 0

    def test_invalid_yaml_raises(self):
        with pytest.raises(Exception):
            parse_and_estimate("not: valid: yaml: [[[")

    def test_non_dict_yaml_raises(self):
        with pytest.raises(ValueError):
            parse_and_estimate("- item1\n- item2\n")


# ── Selection plans per GPU ───────────────────────────────────────────────────


class TestSelectionPlansPerGpu:
    """Selection compared the requirement with *total* VRAM, so it routed jobs
    to multi-GPU boxes where DDP — one full model copy per card — OOMed on every
    GPU, and billed for the attempt."""

    def _plan_for(self, config, itype):
        from heulistic_axolotl import Hardware, plan_parallelism

        from app.services.instances import get_catalog

        info = get_catalog()[itype]
        hw = Hardware(total_vram_gb=info["vram_gb"], gpu_count=info["gpus"], label=itype)
        return plan_parallelism(config, hw)

    @pytest.mark.parametrize("config", [
        {"adapter": "qlora", "base_model": "org/m-7b"},
        {"adapter": "lora", "base_model": "org/m-7b"},
        {"adapter": "lora", "base_model": "org/m-13b"},
        {"adapter": "lora", "base_model": "org/m-20b"},
        {"adapter": "lora", "base_model": "org/m-34b"},
        {"base_model": "org/m-3b"},
        {"base_model": "org/m-7b"},
    ])
    def test_the_chosen_instance_actually_fits(self, config):
        """Whatever is rented must be hardware the trainer can run the job on."""
        assert self._plan_for(config, select_instance(config)).fits is True

    def test_an_unquantized_model_too_big_for_one_card_is_sharded(self):
        # 20B LoRA at 16-bit is 48 GB: no single card holds it, but four T4s
        # do once ZeRO-3 splits it to 12 GB each.
        config = {"adapter": "lora", "base_model": "org/m-20b"}
        itype = select_instance(config)
        plan = self._plan_for(config, itype)
        assert plan.strategy == "zero3" and plan.auto_sharded

    def test_a_quantized_model_is_not_sent_to_cards_it_cannot_fit(self):
        """100B QLoRA is 60 GB. Total VRAM said a 4x16 GB box fits it; per card
        it does not, and quantized models are not auto-sharded."""
        config = {"adapter": "qlora", "load_in_4bit": True, "base_model": "org/m-100b"}
        assert select_instance(config) != "g4dn.12xlarge"

    def test_when_nothing_fits_pick_the_closest_and_cheapest(self):
        """The old fallback took the biggest box. For a quantized model that is
        a 4-GPU g6e.12xlarge at $13.56/h, whose extra cards it cannot use — the
        same 44 GB per GPU is available on a g6e.xlarge at $1.86/h."""
        config = {"adapter": "qlora", "load_in_4bit": True, "base_model": "org/m-100b"}
        assert select_instance(config) == "g6e.xlarge"

    def test_the_old_fallback_still_wins_when_more_gpus_genuinely_help(self):
        # 13B full fine-tune (250 GB) fits nowhere, but sharding over 4x44 GB
        # comes far closer than any single card — so the big box is right here.
        assert select_instance({"base_model": "org/m-13b"}) == "g6e.12xlarge"


class TestSelectionAgreesWithTheValidator:
    """Whatever selection rents, the validator must accept — the UI blocks
    submission on its errors. They once used different capacity lines, and the
    UI refused 70B jobs the API had already placed."""

    @pytest.mark.parametrize("config", [
        {"base_model": "org/m-7b", "adapter": "qlora", "load_in_4bit": True},
        {"base_model": "org/m-7b", "adapter": "lora"},
        {"base_model": "org/m-7b"},
        {"base_model": "org/m-13b", "adapter": "lora"},
        {"base_model": "org/m-20b", "adapter": "lora"},
        {"base_model": "org/m-70b", "adapter": "qlora", "load_in_4bit": True},
        {"base_model": "org/m-70b", "adapter": "lora"},
    ])
    def test_the_selected_instance_passes_validation(self, config):
        import yaml
        from heulistic_axolotl import validate
        from app.services.cost import _hardware
        from app.services.instances import get_catalog

        itype = select_instance(config)
        target = _hardware(itype, get_catalog()[itype])
        text = yaml.safe_dump({**config, "learning_rate": 0.0002,
                               "datasets": [{"path": "tatsu-lab/alpaca"}]})
        errors = [d for d in validate(text, target=target)
                  if d.level == "error" and d.code.startswith("vram.")]
        assert errors == [], f"{itype}: {errors}"


class TestMultiGpuScaling:
    """Four GPUs are not four times as fast. Gradients are all-reduced every
    step, and these instances link their GPUs over PCIe, not NVLink. ZeRO-3
    is slower still: it gathers the weights over that link in every forward
    and backward pass. Assuming perfect scaling under-reserved credits for
    every multi-GPU job and left the difference to an overage charge."""

    # 7B LoRA: ~17 GB, fits one 22 GB card, so a g5.12xlarge runs it as DDP.
    # A large dataset keeps training time well above the fixed overheads.
    LORA_7B = {"base_model": "org/m-7b", "adapter": "lora", "num_epochs": 1,
               "micro_batch_size": 2}

    def test_efficiency_by_strategy(self):
        from app.services.cost import _scaling_efficiency
        assert _scaling_efficiency("single") == 1.0
        assert 0 < _scaling_efficiency("ddp") < 1.0
        assert _scaling_efficiency("zero3") < _scaling_efficiency("ddp")
        assert _scaling_efficiency("fsdp") == _scaling_efficiency("zero3")
        assert _scaling_efficiency("zero3") < _scaling_efficiency("deepspeed") < _scaling_efficiency("ddp")

    def test_four_gpus_are_less_than_four_times_as_fast(self):
        single = estimate_duration_minutes(self.LORA_7B, dataset_size_mb=2000, instance_type="g5.xlarge")
        multi = estimate_duration_minutes(self.LORA_7B, dataset_size_mb=2000, instance_type="g5.12xlarge")
        # ~14 h of training against 5 min of fixed overhead: linear scaling
        # gives a ratio of ~3.9, so only a real efficiency loss gets under 3.8.
        assert 2 < single / multi < 3.8

    def test_sharding_is_slower_than_replication(self):
        ddp = estimate_duration_minutes(self.LORA_7B, dataset_size_mb=2000, instance_type="g5.12xlarge")
        zero3 = estimate_duration_minutes(
            {**self.LORA_7B, "deepspeed": "deepspeed_configs/zero3.json"},
            dataset_size_mb=2000, instance_type="g5.12xlarge",
        )
        assert zero3 > ddp

    def test_one_gpu_is_unchanged(self):
        # Local mode and single-GPU instances must not pick up a penalty.
        a = estimate_duration_minutes(self.LORA_7B, dataset_size_mb=2000, instance_type="g5.xlarge")
        b = estimate_duration_minutes(self.LORA_7B, dataset_size_mb=2000, instance_type="local-gpu")
        assert a == b


class TestEstimateIncludesTheComparison:
    """Training now ends with a before/after comparison on the same paid
    instance, so the estimate — and the credits held — must cover it."""

    CFG = {"base_model": "org/m-7b", "adapter": "lora", "num_epochs": 1, "micro_batch_size": 2}

    def test_comparing_adds_time(self):
        with_eval = estimate_duration_minutes(self.CFG, dataset_size_mb=50)
        without = estimate_duration_minutes({**self.CFG, "val_set_size": 0}, dataset_size_mb=50)
        assert with_eval > without

    def test_bigger_models_take_longer_to_compare(self):
        from app.services.cost import evaluation_minutes
        assert evaluation_minutes({"base_model": "org/m-70b"}) > evaluation_minutes({"base_model": "org/m-7b"})


class TestExportEstimate:
    def test_export_runs_on_the_cheapest_gpu_box(self):
        from app.services.cost import export_instance
        assert export_instance({"base_model": "org/m-7b"}) == "g4dn.xlarge"

    def test_bigger_models_take_longer(self):
        from app.services.cost import export_minutes
        assert export_minutes({"base_model": "org/m-70b"}) > export_minutes({"base_model": "org/m-7b"})

    def test_disk_covers_the_peak(self):
        from app.services.cost import export_disk_gb
        assert export_disk_gb({"base_model": "org/m-7b"}) >= 200       # never below the default
        assert export_disk_gb({"base_model": "org/m-70b"}) > 2 * 70 * 2  # merged + f16 GGUF
