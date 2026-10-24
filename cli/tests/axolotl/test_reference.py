"""Parsing the Axolotl config reference format.

The reference (axolotl.yaml) is not YAML — it is a pseudo-YAML dump of the
Axolotl Pydantic models, shaped as:

    # doc comment
    field_name: type_expr = default

with nested models introduced by a `# For <TypeName>:` marker and indented
two spaces further than their parent.

Fixtures below are lifted from the real file so the parser is tested against
the shapes it will actually meet.
"""

import textwrap

import pytest

from heulistic.axolotl._reference import ReferenceParseError, parse_reference


def ref(src: str):
    """Dedent a fixture and parse it."""
    return parse_reference(textwrap.dedent(src).lstrip("\n"))


# ── Fields, docs, defaults ────────────────────────────────────────────────────

def test_field_with_doc_and_default():
    r = ref("""
        # Allow overwrite yml config using from cli
        strict: bool | None = False
    """)
    f = r.fields["strict"]
    assert f.doc == "Allow overwrite yml config using from cli"
    assert f.has_default is True
    assert f.default is False
    assert f.required is False
    assert f.raw_type == "bool | None"


def test_field_without_doc_or_default():
    r = ref("""
        resume_from_checkpoint: str | None
    """)
    f = r.fields["resume_from_checkpoint"]
    assert f.doc == ""
    assert f.has_default is False
    assert f.default is None


def test_multi_line_doc_is_joined():
    r = ref("""
        # Resize the model embeddings when new tokens are added to multiples of 32. This is
        # reported to improve training speed on some models
        resize_token_embeddings_to_32x: bool | None
    """)
    assert r.fields["resize_token_embeddings_to_32x"].doc == (
        "Resize the model embeddings when new tokens are added to multiples of 32. "
        "This is reported to improve training speed on some models"
    )


def test_blank_line_clears_a_dangling_doc_comment():
    # A comment separated from the next field by a blank line documents nothing.
    # (No such case exists in the reference today; this pins the behaviour so a
    # future reference edit cannot silently misattribute docs.)
    r = ref("""
        # orphaned section header

        seed: int | None
    """)
    assert r.fields["seed"].doc == ""


def test_field_order_is_preserved():
    r = ref("""
        strict: bool | None = False
        resume_from_checkpoint: str | None
        seed: int | None
    """)
    assert list(r.fields) == ["strict", "resume_from_checkpoint", "seed"]


# ── Default coercion ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "line,expected",
    [
        ("a: bool | None = False", False),
        ("a: bool | None = True", True),
        ("a: int | None = 8000", 8000),
        ("a: int | None = 0", 0),
        ("a: float | None = 1.0", 1.0),
        ("a: float | None = 0.9", 0.9),
        # Defaults that only *look* numeric must stay strings.
        ("a: str | None = 0.0.0.0", "0.0.0.0"),
        ("a: Literal['auto'] | bool | None = auto", "auto"),
        ("a: str | None = train", "train"),
        ("a: TorchAOQuantDType | None = TorchAOQuantDType.int8", "TorchAOQuantDType.int8"),
    ],
)
def test_default_coercion_follows_the_declared_type(line, expected):
    f = ref(line).fields["a"]
    assert f.has_default is True
    assert f.default == expected
    assert type(f.default) is type(expected)


def test_int_default_on_a_float_field_stays_numeric():
    assert ref("a: float | None = 1").fields["a"].default == 1.0


# ── Required ──────────────────────────────────────────────────────────────────

def test_required_marker():
    r = ref("""
        base_model: str (required)
        learning_rate: str | float (required)
    """)
    assert r.fields["base_model"].required is True
    assert r.fields["base_model"].raw_type == "str"
    assert r.fields["learning_rate"].required is True
    assert r.fields["learning_rate"].type.primitives == frozenset({"str", "float"})


def test_fields_are_optional_by_default():
    assert ref("seed: int | None").fields["seed"].required is False


# ── Nested sub-schemas ────────────────────────────────────────────────────────

def test_nested_block_becomes_a_subschema():
    r = ref("""
        trl: TRLConfig | None
          # For TRLConfig:
          # Beta parameter for the RL training. Same as `rl_beta`. Use
          beta: float | None
          max_completion_length: int | None
    """)
    assert "TRLConfig" in r.fields["trl"].type.named
    sub = r.subschemas["TRLConfig"]
    assert list(sub.fields) == ["beta", "max_completion_length"]
    assert sub.fields["beta"].doc.startswith("Beta parameter for the RL training")
    # Nested fields must not leak into the top level.
    assert "beta" not in r.fields


def test_dedent_closes_the_nested_block():
    r = ref("""
        trl: TRLConfig | None
          # For TRLConfig:
          beta: float | None
        seed: int | None
    """)
    assert list(r.fields) == ["trl", "seed"]
    assert list(r.subschemas["TRLConfig"].fields) == ["beta"]


def test_multiple_for_sections_share_one_parent():
    # `datasets` is a union of five models, each with its own `# For X:` block.
    r = ref("""
        datasets: Annotated[list[SFTDataset | DPODataset], MinLen(1)] | None
          # For SFTDataset:
          path: str | None
          # For DPODataset:
          split: str | None
    """)
    assert list(r.subschemas["SFTDataset"].fields) == ["path"]
    assert list(r.subschemas["DPODataset"].fields) == ["split"]


def test_two_levels_of_nesting():
    r = ref("""
        datasets: list[SFTDataset] | None
          # For SFTDataset:
          # The type of prompt to use for training.
          type: str | UserDefinedPrompterType | None
            # For UserDefinedPrompterType:
            # Custom user instruction prompt
            system_prompt: str | None
          path: str | None
    """)
    assert list(r.subschemas["SFTDataset"].fields) == ["type", "path"]
    assert list(r.subschemas["UserDefinedPrompterType"].fields) == ["system_prompt"]
    assert r.subschemas["UserDefinedPrompterType"].parent == "SFTDataset"


def test_prose_beginning_with_for_is_a_doc_comment_not_a_marker():
    # Real line 391: "# For `completion` datasets only, uses the provided field…"
    # A loose marker regex would turn this into a bogus sub-schema and orphan
    # every field beneath it.
    r = ref("""
        datasets: list[SFTDataset] | None
          # For SFTDataset:
          path: str | None
          # For `completion` datasets only, uses the provided field instead of `text` column
          field: str | None
    """)
    sft = r.subschemas["SFTDataset"]
    assert list(sft.fields) == ["path", "field"]
    assert sft.fields["field"].doc.startswith("For `completion` datasets only")
    assert set(r.subschemas) == {"SFTDataset"}


def test_repeated_identical_subschema_definition_is_merged():
    # SFTDataset is spelled out three times in the reference (under `datasets`,
    # `test_datasets`, and one more). Identical repeats must collapse, not clash.
    r = ref("""
        datasets: list[SFTDataset] | None
          # For SFTDataset:
          path: str | None
        test_datasets: list[SFTDataset] | None
          # For SFTDataset:
          path: str | None
    """)
    assert list(r.subschemas["SFTDataset"].fields) == ["path"]


def test_conflicting_subschema_definition_is_an_error():
    # If Axolotl ever ships two different shapes under one name, fail loudly at
    # generation time rather than silently keeping whichever came last.
    with pytest.raises(ReferenceParseError, match="SFTDataset"):
        ref("""
            datasets: list[SFTDataset] | None
              # For SFTDataset:
              path: str | None
            test_datasets: list[SFTDataset] | None
              # For SFTDataset:
              path: int | None
        """)


# ── Structural guards ─────────────────────────────────────────────────────────

def test_every_field_line_is_accounted_for():
    r = ref("""
        strict: bool | None = False
        trl: TRLConfig | None
          # For TRLConfig:
          beta: float | None
        seed: int | None
    """)
    assert r.field_count == 3  # top-level
    assert r.total_field_count == 4  # including nested


def test_unparseable_line_raises():
    with pytest.raises(ReferenceParseError):
        ref("this line has no colon\n")


def test_orphaned_indented_field_raises():
    # An indented field with no `# For X:` marker above it means the parser has
    # lost its place. The fixture needs a column-0 anchor line, or the helper's
    # dedent would strip the very indentation under test.
    with pytest.raises(ReferenceParseError, match="no enclosing block"):
        ref("""
            strict: bool | None = False
              beta: float | None
        """)


def test_field_skipping_a_nesting_level_raises():
    with pytest.raises(ReferenceParseError, match="no enclosing block"):
        ref("""
            strict: bool | None = False
                beta: float | None
        """)
