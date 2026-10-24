"""Parsing the real, vendored Axolotl reference.

These tests do two jobs. They prove the parser survives the actual file rather
than only the hand-written fixtures in `test_reference.py`, and they act as a
tripwire: when the reference is refreshed for a new Axolotl release, the counts
below shift and the diff shows exactly what changed in the schema.

A failure here is normally *expected* after a reference update — read the diff,
confirm it matches the Axolotl changelog, then update the constants.
"""

import re

import pytest

from heulistic.axolotl._reference import parse_reference
from heulistic.axolotl.reference import reference_path, reference_text

# Tripwire constants — see module docstring before changing.
EXPECTED_TOP_LEVEL_FIELDS = 390
EXPECTED_SUBSCHEMAS = 23
EXPECTED_UNIQUE_FIELDS = 623
EXPECTED_SOURCE_FIELD_LINES = 752


@pytest.fixture(scope="module")
def ref():
    return parse_reference(reference_text())


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_reference_file_is_present():
    # Dev-time input only — it lives outside `src/` and is deliberately absent
    # from the built wheel, which ships the generated schema instead.
    assert reference_path().is_file()


def test_every_field_line_is_consumed(ref):
    """No field may be silently dropped.

    Counted independently off the raw text, so a parser that quietly skipped
    lines could not also fake this number.
    """
    source_lines = [
        line
        for line in reference_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(source_lines) == EXPECTED_SOURCE_FIELD_LINES
    assert ref.consumed_field_lines == len(source_lines)


def test_merge_only_removes_duplicates(ref):
    # The gap between consumed lines and unique fields is duplicate model
    # definitions (SFTDataset and friends are spelled out per referring field).
    assert ref.total_field_count == EXPECTED_UNIQUE_FIELDS
    assert ref.consumed_field_lines - ref.total_field_count == 129


def test_expected_shape(ref):
    assert ref.field_count == EXPECTED_TOP_LEVEL_FIELDS
    assert len(ref.subschemas) == EXPECTED_SUBSCHEMAS


def test_every_field_name_is_an_identifier(ref):
    pattern = re.compile(r"^[a-z_][a-z0-9_]*$")
    assert [n for n in ref.fields if not pattern.match(n)] == []


# ── Anchors: fields the validator and the platform depend on ──────────────────

def test_required_fields(ref):
    required = [name for name, f in ref.fields.items() if f.required]
    assert required == ["base_model", "learning_rate"]


def test_learning_rate_accepts_str_or_float(ref):
    # Axolotl allows "3e-4" as a string as well as a float, so a validator must
    # not reject the quoted form that appears throughout the Axolotl examples.
    lr = ref.fields["learning_rate"]
    assert lr.type.accepts(0.0003) is True
    assert lr.type.accepts("3e-4") is True
    assert lr.type.accepts(True) is False


def test_datasets_carries_the_five_entry_models(ref):
    datasets = ref.fields["datasets"]
    assert datasets.type.primitives == frozenset({"list"})
    assert set(datasets.type.named) == {
        "SFTDataset",
        "DPODataset",
        "KTODataset",
        "StepwiseSupervisedDataset",
        "SyntheticDataset",
    }
    assert "path" in ref.subschemas["SFTDataset"].fields


def test_bf16_accepts_auto_or_bool(ref):
    bf16 = ref.fields["bf16"]
    assert bf16.type.accepts("auto") is True
    assert bf16.type.accepts(True) is True
    assert bf16.type.accepts("yes") is False


def test_integer_fields_reject_booleans(ref):
    assert ref.fields["micro_batch_size"].type.accepts(4) is True
    assert ref.fields["micro_batch_size"].type.accepts(True) is False
    assert ref.fields["sequence_len"].type.accepts(2048) is True


def test_multi_gpu_fields_exist_and_are_plain_integers(ref):
    # These are valid single-node multi-GPU settings. The API currently rejects
    # them outright (jobs.py `_MULTI_NODE_INT_FIELDS`), which is the bug the
    # shared validator is meant to retire — nothing in the schema marks them
    # unsupported.
    for name in ("world_size", "tensor_parallel_size", "context_parallel_size"):
        assert name in ref.fields
        assert ref.fields[name].type.accepts(4) is True


def test_docs_are_attached(ref):
    documented = sum(1 for f in ref.fields.values() if f.doc)
    assert documented >= 300
    assert ref.fields["sample_packing"].doc != ""


def test_defaults_are_typed(ref):
    defaults = {n: f.default for n, f in ref.fields.items() if f.has_default}
    assert len(defaults) == 67
    # Nothing numeric-looking should have leaked into the wrong type.
    for name, value in defaults.items():
        assert value is not None, name
        assert isinstance(value, (bool, int, float, str)), name


# ── Sub-schemas ───────────────────────────────────────────────────────────────

def test_nested_required_fields_live_on_their_models(ref):
    # `name`/`modules`/`lr` are required on LrGroup, not at the top level.
    lr_group = ref.subschemas["LrGroup"]
    assert [n for n, f in lr_group.fields.items() if f.required] == [
        "name",
        "modules",
        "lr",
    ]


def test_deeply_nested_model_records_its_parent(ref):
    assert ref.subschemas["UserDefinedPrompterType"].parent == "SFTDataset"
    assert ref.subschemas["LoftQConfig"].parent == "PeftConfig"


# ── Named types with no model block ───────────────────────────────────────────

def test_named_types_that_the_reference_never_defines(ref):
    """Seven referenced types are enums documented only in prose.

    Pinned so that a future reference gaining a `# For RLType:` block — or
    losing one of these — is noticed, since the validator treats undefined
    named types permissively.
    """
    referenced = {t for f in ref.fields.values() for t in f.type.named}
    for model in ref.subschemas.values():
        referenced |= {t for f in model.fields.values() for t in f.type.named}

    assert referenced - set(ref.subschemas) == {
        "ChatTemplate",
        "CustomSupportedOptimizers",
        "FP8Recipe",
        "RLType",
        "Resampling",
        "RingAttnFunc",
        "TorchAOQuantDType",
    }


def test_enum_fields_accept_their_documented_values(ref):
    # Each value below is quoted from that field's own doc comment. Before the
    # permissiveness rule these all returned False.
    assert ref.fields["rl"].type.accepts("dpo") is True
    assert ref.fields["optimizer"].type.accepts("adamw_torch") is True
    assert ref.fields["chat_template"].type.accepts("chatml") is True
    assert ref.subschemas["FP8Config"].fields["recipe"].type.accepts("tensorwise") is True


def test_provenance_is_recorded(ref):
    from heulistic.axolotl.reference import SOURCE_URL

    assert SOURCE_URL == "https://docs.axolotl.ai/docs/config-reference.html"
    assert SOURCE_URL in reference_text()
