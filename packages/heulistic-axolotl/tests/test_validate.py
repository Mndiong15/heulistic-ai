"""Config validation rules.

Rules come from three places:

  * the generated Axolotl schema — unknown fields, wrong types, required fields
  * `frontend/src/lib/validate-config.ts` — the coherence and sanity rules that
    were worth keeping, now schema-backed rather than hand-listed
  * Heulistic platform limits — single-node only

Levels follow the frontend's existing contract: `error` blocks submission,
`warning` is likely-wrong, `info` is a suggestion.
"""

import textwrap

import pytest

from heulistic_axolotl.validate import validate
from heulistic_axolotl.vram import Hardware

# A config that should pass cleanly — every rule below is a mutation of this.
CLEAN = """\
base_model: mistralai/Mistral-7B-v0.1
learning_rate: 0.0002
adapter: qlora
load_in_4bit: true
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 3
save_steps: 100
bf16: auto
gradient_checkpointing: true
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
"""


def check(config: str, **kwargs):
    return validate(textwrap.dedent(config), **kwargs)


def levels(diags, level):
    return [d for d in diags if d.level == level]


def fields(diags, level=None):
    return {d.field for d in diags if level is None or d.level == level}


def message_for(diags, field):
    return next((d.message for d in diags if d.field == field), None)


def codes(diags):
    return {d.code for d in diags}


def by_code(diags, code):
    return next((d for d in diags if d.code == code), None)


# ── Baseline ──────────────────────────────────────────────────────────────────

def test_clean_config_produces_no_errors_or_warnings():
    diags = check(CLEAN)
    assert levels(diags, "error") == []
    assert levels(diags, "warning") == []


def test_clean_config_is_also_clean_against_matching_hardware():
    # 7B QLoRA needs ~4.2 GB — comfortable on a 22 GB A10G.
    diags = check(CLEAN, target=Hardware(total_vram_gb=22, gpu_count=1))
    assert levels(diags, "error") == []
    assert levels(diags, "warning") == []


# ── YAML and shape ────────────────────────────────────────────────────────────

def test_syntax_error_is_reported_and_stops_there():
    diags = check("base_model: [unclosed\n")
    assert len(diags) == 1
    assert diags[0].level == "error"
    assert "yaml" in diags[0].message.lower()


def test_non_mapping_document():
    diags = check("- just\n- a\n- list\n")
    assert levels(diags, "error")
    assert "mapping" in diags[0].message.lower()


def test_empty_document():
    assert levels(check(""), "error")


# ── Required fields ───────────────────────────────────────────────────────────

def test_missing_required_fields():
    diags = check("num_epochs: 1\n")
    assert "base_model" in fields(diags, "error")
    assert "learning_rate" in fields(diags, "error")


def test_required_field_present_but_null():
    diags = check("base_model:\nlearning_rate: 0.0002\n")
    assert "base_model" in fields(diags, "error")


# ── Type checking (new — the TS validator had none) ───────────────────────────

def test_wrong_type_is_an_error():
    diags = check(CLEAN.replace("micro_batch_size: 1", "micro_batch_size: small"))
    assert "micro_batch_size" in fields(diags, "error")
    assert "int" in message_for(diags, "micro_batch_size")


def test_bool_is_not_accepted_where_an_int_belongs():
    diags = check(CLEAN.replace("micro_batch_size: 1", "micro_batch_size: true"))
    assert "micro_batch_size" in fields(diags, "error")


def test_learning_rate_accepts_the_quoted_scientific_form():
    # Axolotl types this `str | float`, and its own examples quote it.
    diags = check(CLEAN.replace("learning_rate: 0.0002", "learning_rate: '3e-4'"))
    assert levels(diags, "error") == []


def test_bf16_accepts_auto_and_bool_but_not_arbitrary_strings():
    assert levels(check(CLEAN.replace("bf16: auto", "bf16: true")), "error") == []
    diags = check(CLEAN.replace("bf16: auto", "bf16: yes-please"))
    assert "bf16" in fields(diags, "error")


def test_undocumented_enum_values_are_not_flagged():
    # `optimizer` is typed `CustomSupportedOptimizers`, whose members the
    # reference never lists — so no value can be judged wrong.
    diags = check(CLEAN + "optimizer: adamw_bnb_8bit\nrl: dpo\n")
    assert levels(diags, "error") == []


# ── Unknown fields and aliases ────────────────────────────────────────────────

def test_unknown_field_is_a_warning():
    diags = check(CLEAN + "definitely_not_a_field: 1\n")
    assert "definitely_not_a_field" in fields(diags, "warning")


@pytest.mark.parametrize(
    "alias,suggestion",
    [
        ("lr", "learning_rate"),
        ("epochs", "num_epochs"),
        ("model", "base_model"),
        ("batch", "micro_batch_size"),
        ("lora_modules", "lora_target_modules"),
    ],
)
def test_common_aliases_suggest_the_real_field(alias, suggestion):
    diags = check(CLEAN + f"{alias}: 1\n")
    assert suggestion in message_for(diags, alias)


def test_near_miss_typo_gets_a_suggestion():
    # Not in the alias table, but one edit away from a real field.
    diags = check(CLEAN + "sequence_length: 2048\n")
    assert "sequence_len" in message_for(diags, "sequence_length")


def test_unknown_field_with_no_close_match_still_warns():
    diags = check(CLEAN + "zzzzzz: 1\n")
    assert "zzzzzz" in fields(diags, "warning")


# ── base_model ────────────────────────────────────────────────────────────────

def test_placeholder_model_is_an_error():
    diags = check(CLEAN.replace("mistralai/Mistral-7B-v0.1", "YOUR_MODEL_HERE"))
    assert "base_model" in fields(diags, "error")


def test_model_with_whitespace_is_an_error():
    diags = check(CLEAN.replace("mistralai/Mistral-7B-v0.1", "'my model'"))
    assert "base_model" in fields(diags, "error")


def test_model_without_a_slash_warns():
    diags = check(CLEAN.replace("mistralai/Mistral-7B-v0.1", "mistral7b"))
    assert "base_model" in fields(diags, "warning")


# ── Adapter coherence ─────────────────────────────────────────────────────────

def test_qlora_without_4bit_warns():
    diags = check(CLEAN.replace("load_in_4bit: true", "load_in_4bit: false"))
    assert "load_in_4bit" in fields(diags, "warning")


def test_lora_fields_without_an_adapter_warn():
    config = CLEAN.replace("adapter: qlora", "").replace("load_in_4bit: true", "")
    diags = check(config + "lora_r: 16\nlora_alpha: 32\n")
    assert "adapter" in fields(diags, "warning")


def test_large_lora_r_warns():
    diags = check(CLEAN + "lora_r: 512\n")
    assert "lora_r" in fields(diags, "warning")


# ── Datasets ──────────────────────────────────────────────────────────────────

def test_no_dataset_at_all_warns():
    config = CLEAN.split("datasets:")[0]
    diags = check(config)
    assert "datasets" in fields(diags, "warning")


def test_pretraining_dataset_satisfies_the_dataset_requirement():
    config = CLEAN.split("datasets:")[0] + "pretraining_dataset: c4\n"
    assert "datasets" not in fields(diags := check(config), "warning"), diags


def test_dataset_entry_without_path_or_name_is_an_error():
    # Both lines of the entry are replaced — swapping only the first would
    # leave an orphaned `type:` and produce a YAML error instead.
    config = CLEAN.replace("  - path: tatsu-lab/alpaca\n    type: alpaca\n", "  - {}\n")
    diags = check(config)
    assert codes(diags) >= {"dataset.no_path"}


def test_dataset_entry_without_type_warns():
    config = CLEAN.replace("    type: alpaca\n", "")
    diags = check(config)
    assert any("datasets[0]" in (d.field or "") for d in levels(diags, "warning"))


def test_unrecognised_dataset_type_warns():
    diags = check(CLEAN.replace("type: alpaca", "type: not_a_format"))
    assert any("datasets[0]" in (d.field or "") for d in levels(diags, "warning"))


def test_unknown_key_inside_a_dataset_entry_warns():
    # Validated against the SFTDataset sub-schema.
    diags = check(CLEAN.replace("    type: alpaca\n", "    type: alpaca\n    pathh: x\n"))
    assert any("pathh" in (d.field or "") for d in levels(diags, "warning"))


# ── Numeric sanity ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_num_epochs_is_an_error(value):
    diags = check(CLEAN.replace("num_epochs: 3", f"num_epochs: {value}"))
    assert "num_epochs" in fields(diags, "error")


def test_absurd_num_epochs_warns():
    diags = check(CLEAN.replace("num_epochs: 3", "num_epochs: 500"))
    assert "num_epochs" in fields(diags, "warning")


def test_missing_epochs_and_max_steps_warns():
    diags = check(CLEAN.replace("num_epochs: 3\n", ""))
    assert "num_epochs" in fields(diags, "warning")


def test_max_steps_satisfies_the_epochs_requirement():
    config = CLEAN.replace("num_epochs: 3", "max_steps: 1000")
    assert "num_epochs" not in fields(check(config), "warning")


def test_non_positive_learning_rate_is_an_error():
    diags = check(CLEAN.replace("learning_rate: 0.0002", "learning_rate: 0"))
    assert "learning_rate" in fields(diags, "error")


def test_very_high_learning_rate_warns():
    diags = check(CLEAN.replace("learning_rate: 0.0002", "learning_rate: 0.5"))
    assert "learning_rate" in fields(diags, "warning")


def test_sequence_len_bounds():
    assert "sequence_len" in fields(
        check(CLEAN.replace("sequence_len: 2048", "sequence_len: 0")), "error"
    )
    assert "sequence_len" in fields(
        check(CLEAN.replace("sequence_len: 2048", "sequence_len: 999999")), "error"
    )
    assert "sequence_len" in fields(
        check(CLEAN.replace("sequence_len: 2048", "sequence_len: 16384")), "warning"
    )


def test_non_positive_gradient_accumulation_is_an_error():
    diags = check(CLEAN.replace("gradient_accumulation_steps: 4", "gradient_accumulation_steps: 0"))
    assert "gradient_accumulation_steps" in fields(diags, "error")


def test_save_steps_zero_warns_about_losing_resumability():
    diags = check(CLEAN.replace("save_steps: 100", "save_steps: 0"))
    assert "save_steps" in fields(diags, "warning")


# ── Informational hints ───────────────────────────────────────────────────────

def test_missing_optional_settings_produce_info_not_warnings():
    minimal = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: qlora\n"
    diags = check(minimal)
    assert levels(diags, "error") == []
    info_fields = fields(diags, "info")
    assert {"bf16", "gradient_checkpointing"} <= info_fields


# ── Platform limits ───────────────────────────────────────────────────────────

def test_use_ray_is_rejected():
    diags = check(CLEAN + "use_ray: true\n")
    assert "use_ray" in fields(diags, "error")


def test_multiple_ray_workers_rejected():
    diags = check(CLEAN + "ray_num_workers: 4\n")
    assert "ray_num_workers" in fields(diags, "error")


def test_single_node_multi_gpu_settings_are_allowed():
    """The bug this package exists to retire.

    The API rejects these outright; the frontend says they are fine. They are
    valid on one multi-GPU instance, which Heulistic sells (g5.12xlarge and
    friends have four GPUs), so they must not be errors.
    """
    config = CLEAN + "world_size: 4\ntensor_parallel_size: 2\ncontext_parallel_size: 2\n"
    diags = check(config)
    assert levels(diags, "error") == []
    assert "world_size" not in fields(diags, "warning")


# ── Hardware fit ──────────────────────────────────────────────────────────────

def test_no_hardware_target_means_no_vram_check():
    # A 70B full fine-tune needs ~1.3 TB; with no target we cannot know.
    # Filtered by code, not by message text — the `gradient_checkpointing`
    # hint also mentions VRAM, and matching on that substring made this pass
    # for the wrong reason.
    config = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    assert not {c for c in codes(check(config)) if c.startswith("vram.")}


def test_config_that_cannot_fit_is_an_error():
    config = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    diags = check(config, target=Hardware(total_vram_gb=22, gpu_count=1, label="g5.xlarge"))
    over = by_code(diags, "vram.over")
    assert over is not None and over.level == "error"
    assert "g5.xlarge" in over.message


def test_tight_fit_warns_and_suggests_a_remedy():
    # 7B LoRA at 16-bit needs ~16.8 GB — 76% of a 22 GB card.
    config = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: lora\n"
    diags = check(config, target=Hardware(total_vram_gb=22, gpu_count=1))
    tight = by_code(diags, "vram.tight")
    assert tight is not None and tight.level == "warning"
    assert "gradient_checkpointing" in tight.message
    assert by_code(diags, "vram.over") is None


def test_local_gpu_message_names_the_device():
    config = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    diags = check(config, target=Hardware(total_vram_gb=8, gpu_count=1, label="RTX 4060"))
    assert "RTX 4060" in by_code(diags, "vram.over").message


def test_multi_gpu_target_hints_at_sharding():
    config = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: lora\n"
    diags = check(config, target=Hardware(total_vram_gb=88, gpu_count=4))
    hint = by_code(diags, "hardware.multi_gpu")
    assert hint is not None and hint.level == "info"


def test_sharding_hint_suppressed_when_already_configured():
    config = (
        "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: lora\n"
        "deepspeed: zero2.json\n"
    )
    diags = check(config, target=Hardware(total_vram_gb=88, gpu_count=4))
    assert by_code(diags, "hardware.multi_gpu") is None


# ── Rule codes ────────────────────────────────────────────────────────────────

def test_every_diagnostic_carries_a_code():
    """Codes are the stable API — the CLI filters on them and the frontend
    will key off them, so an uncoded diagnostic is a hole in that contract."""
    messy = (
        "base_model: YOUR_MODEL_HERE\n"
        "learning_rate: 5\n"
        "num_epochs: 0\n"
        "sequence_len: 999999\n"
        "micro_batch_size: nope\n"
        "use_ray: true\n"
        "bogus_field: 1\n"
        "lr: 3\n"
        "datasets:\n"
        "  - type: not_a_format\n"
    )
    diags = check(messy, target=Hardware(total_vram_gb=8, gpu_count=2))
    assert diags
    assert all(d.code for d in diags), [d for d in diags if not d.code]
    assert all("." in d.code for d in diags)


def test_codes_are_unique_per_rule_not_per_occurrence():
    diags = check(CLEAN + "bogus_a: 1\nbogus_b: 2\n")
    unknown = [d for d in diags if d.code == "field.unknown"]
    assert len(unknown) == 2
    assert {d.field for d in unknown} == {"bogus_a", "bogus_b"}


# ── Output contract ───────────────────────────────────────────────────────────

def test_diagnostics_carry_line_numbers_for_top_level_fields():
    diags = check(CLEAN + "definitely_not_a_field: 1\n")
    unknown = next(d for d in diags if d.field == "definitely_not_a_field")
    assert unknown.line == len(CLEAN.splitlines()) + 1


def test_unlocated_findings_sort_after_located_ones():
    # `learning_rate` is missing so it has no line; `base_model` is on line 1.
    # Listing the line-less finding first would imply it came earlier in the
    # file than it does.
    diags = check("base_model: YOUR_MODEL_HERE\n")
    errors = levels(diags, "error")
    assert [d.field for d in errors] == ["base_model", "learning_rate"]
    assert errors[0].line == 1
    assert errors[1].line is None


def test_results_are_ordered_by_severity_then_line():
    diags = check(
        "base_model: YOUR_MODEL_HERE\n"
        "learning_rate: 0.0002\n"
        "bogus_field: 1\n"
    )
    order = [d.level for d in diags]
    assert order == sorted(order, key=["error", "warning", "info"].index)


def test_validation_is_deterministic():
    config = CLEAN + "bogus_a: 1\nbogus_b: 2\n"
    assert [ (d.level, d.field, d.message) for d in check(config) ] == [
        (d.level, d.field, d.message) for d in check(config)
    ]


# ── Multi-GPU fit is judged per GPU ───────────────────────────────────────────

def test_a_model_too_big_for_one_card_is_reported_as_sharded():
    """13B LoRA (31 GB) on 4x22 GB. The old total-VRAM check passed this as
    fitting under DDP — where it would OOM on every card. It fits only sharded,
    and the user should be told that is what will happen."""
    config = "base_model: org/m-13b\nlearning_rate: 0.0002\nadapter: lora\n"
    diags = check(config, target=Hardware(total_vram_gb=88, gpu_count=4, label="g5.12xlarge"))
    assert by_code(diags, "vram.over") is None
    sharded = by_code(diags, "hardware.auto_sharded")
    assert sharded is not None and sharded.level == "info"
    assert "ZeRO-3" in sharded.message


def test_a_quantized_model_too_big_for_one_card_is_an_error():
    """70B QLoRA (42 GB) on 4x22 GB. Total says 42 <= 88; per card it does not
    fit, and quantized models are not auto-sharded."""
    config = "base_model: org/m-70b\nlearning_rate: 0.0002\nadapter: qlora\nload_in_4bit: true\n"
    diags = check(config, target=Hardware(total_vram_gb=88, gpu_count=4, label="g5.12xlarge"))
    over = by_code(diags, "vram.over")
    assert over is not None and over.level == "error"
    assert "per GPU" in over.message
    assert "quantiz" in over.message.lower()


def test_an_unmodellable_user_config_warns_instead_of_blocking():
    """ZeRO-2 is modelled as unsharded, which may overstate the need. Blocking
    a submission on a number we cannot vouch for would be wrong."""
    config = (
        "base_model: org/m-20b\nlearning_rate: 0.0002\nadapter: lora\n"
        "deepspeed: deepspeed_configs/zero2.json\n"
    )
    diags = check(config, target=Hardware(total_vram_gb=64, gpu_count=4))
    assert by_code(diags, "vram.over") is None
    unverified = by_code(diags, "vram.unverified")
    assert unverified is not None and unverified.level == "warning"


def test_the_ddp_hint_no_longer_recommends_zero2():
    """ZeRO-2 shards optimiser state but not weights, so it cannot rescue a
    model that does not fit one card — recommending it was wrong."""
    config = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: lora\n"
    diags = check(config, target=Hardware(total_vram_gb=88, gpu_count=4))
    hint = by_code(diags, "hardware.multi_gpu")
    assert hint is not None
    assert "ZeRO-2" not in hint.message
    assert "copy" in hint.message.lower()


def test_a_95_percent_fit_is_tight_not_over():
    # 70B QLoRA is ~42 GB: 95.5% of a 44 GB card. Rented by instance selection,
    # so the validator must not call it impossible.
    config = ("base_model: org/m-70b\nlearning_rate: 0.0002\n"
              "adapter: qlora\nload_in_4bit: true\n")
    diags = check(config, target=Hardware(total_vram_gb=44, gpu_count=1, label="g6e.xlarge"))
    assert by_code(diags, "vram.over") is None
    assert by_code(diags, "vram.tight").level == "warning"
