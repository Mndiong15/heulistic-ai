"""Type-expression parsing.

Every expression here is copied verbatim from the Axolotl config reference
(axolotl.yaml), so these cases mirror the real type vocabulary rather than
invented shapes.
"""

from heulistic.axolotl.spec import parse_type


# ── Primitives ────────────────────────────────────────────────────────────────

def test_optional_str():
    t = parse_type("str | None")
    assert t.optional is True
    assert t.primitives == frozenset({"str"})
    assert t.literals == ()
    assert t.named == ()


def test_non_optional_bool():
    # e.g. `use_vllm: bool = False` — no `| None`, so the field is never null
    t = parse_type("bool")
    assert t.optional is False
    assert t.primitives == frozenset({"bool"})


def test_optional_int_and_float():
    assert parse_type("int | None").primitives == frozenset({"int"})
    assert parse_type("float | None").primitives == frozenset({"float"})


def test_any_is_its_own_primitive():
    t = parse_type("Any | None")
    assert t.primitives == frozenset({"any"})
    assert t.accepts_anything is True


# ── Containers ────────────────────────────────────────────────────────────────

def test_list_of_str():
    t = parse_type("list[str] | None")
    assert t.primitives == frozenset({"list"})
    assert t.optional is True


def test_dict_types():
    assert parse_type("dict[str, Any] | None").primitives == frozenset({"dict"})
    assert parse_type("dict[str, list[str]] | None").primitives == frozenset({"dict"})
    # PositiveInt is a pydantic type in the value position — must not leak out
    # as a named sub-schema, since there is no `# For PositiveInt:` block.
    assert parse_type("dict[str, PositiveInt] | None").named == ()


# ── Literals ──────────────────────────────────────────────────────────────────

def test_literal_enum():
    t = parse_type("Literal['all', 'turn', 'last', 'none'] | None")
    assert t.literals == ("all", "turn", "last", "none")
    assert t.primitives == frozenset({"str"})


def test_literal_mixed_with_primitive():
    # `bf16: Literal['auto'] | bool | None` — accepts the string "auto" or a bool
    t = parse_type("Literal['auto'] | bool | None")
    assert t.literals == ("auto",)
    assert t.primitives == frozenset({"str", "bool"})
    assert t.optional is True


# ── Unions ────────────────────────────────────────────────────────────────────

def test_union_of_primitives():
    assert parse_type("str | list[str] | None").primitives == frozenset({"str", "list"})
    assert parse_type("int | float | None").primitives == frozenset({"int", "float"})


def test_required_marker_is_not_part_of_the_type():
    # `learning_rate: str | float (required)` — the marker is stripped upstream,
    # but parse_type must tolerate it defensively.
    t = parse_type("str | float (required)")
    assert t.primitives == frozenset({"str", "float"})


# ── Named sub-schemas ─────────────────────────────────────────────────────────

def test_named_type():
    t = parse_type("TRLConfig | None")
    assert t.named == ("TRLConfig",)
    assert t.primitives == frozenset()


def test_named_type_unioned_with_primitive():
    t = parse_type("ChatTemplate | str | None")
    assert t.named == ("ChatTemplate",)
    assert t.primitives == frozenset({"str"})


def test_annotated_list_of_union_unwraps_to_members():
    # The `datasets` field — the hairiest expression in the reference.
    t = parse_type(
        "Annotated[list[SFTDataset | DPODataset | KTODataset | "
        "StepwiseSupervisedDataset | SyntheticDataset], MinLen(1)] | None"
    )
    assert t.primitives == frozenset({"list"})
    assert t.named == (
        "SFTDataset",
        "DPODataset",
        "KTODataset",
        "StepwiseSupervisedDataset",
        "SyntheticDataset",
    )
    assert t.optional is True
    # MinLen(1) is a constraint, not a type — it must not become a named schema.
    assert "MinLen" not in t.named


def test_top_level_split_respects_brackets():
    # A naive split on "|" would tear `dict[str, A | B]` apart.
    t = parse_type("dict[str, str] | list[str] | None")
    assert t.primitives == frozenset({"dict", "list"})
    assert t.named == ()


# ── Type checking against values ──────────────────────────────────────────────

def test_accepts_value_by_primitive():
    t = parse_type("float | None")
    assert t.accepts(0.0003) is True
    assert t.accepts(None) is True
    assert t.accepts("3e-4") is False


def test_bool_is_not_an_int():
    # Python bools are ints; the checker must not let `bf16: 1` pass as a bool
    # or `micro_batch_size: true` pass as an int.
    assert parse_type("int | None").accepts(True) is False
    assert parse_type("bool | None").accepts(1) is False


def test_int_is_acceptable_where_float_is_expected():
    # `learning_rate: 1` should not be flagged — int widens to float.
    assert parse_type("float | None").accepts(1) is True
    assert parse_type("int | None").accepts(1.5) is False


def test_literal_value_must_be_in_the_allowed_set():
    t = parse_type("Literal['auto'] | bool | None")
    assert t.accepts("auto") is True
    assert t.accepts(True) is True
    assert t.accepts("automatic") is False


def test_any_accepts_everything():
    t = parse_type("Any | None")
    assert t.accepts("x") is True
    assert t.accepts(123) is True
    assert t.accepts({"a": 1}) is True


def test_non_optional_type_rejects_none():
    assert parse_type("bool").accepts(None) is False
    assert parse_type("bool | None").accepts(None) is True


# ── Named types the reference never defines ───────────────────────────────────

def test_undocumented_enum_accepts_any_value():
    # `recipe: FP8Recipe = tensorwise` — FP8Recipe has no `# For FP8Recipe:`
    # block, so its members are unknown to us. Rejecting here would flag valid
    # values on `optimizer`, `rl`, `chat_template` and the QAT/PTQ dtypes.
    t = parse_type("FP8Recipe")
    assert t.primitives == frozenset()
    assert t.named == ("FP8Recipe",)
    assert t.accepts("tensorwise") is True
    assert t.accepts("rowwise_with_gw_hp") is True


def test_optimizer_style_enum_accepts_strings():
    assert parse_type("CustomSupportedOptimizers | None").accepts("adamw_torch") is True
    assert parse_type("RLType | None").accepts("dpo") is True


def test_model_valued_field_accepts_a_mapping():
    # `datasets[].type` may be a prompt-style string *or* an inline
    # UserDefinedPrompterType mapping.
    t = parse_type("str | UserDefinedPrompterType | None")
    assert t.accepts("alpaca") is True
    assert t.accepts({"system_prompt": "You are helpful"}) is True


def test_named_types_do_not_make_the_check_useless():
    # Where the type still offers a concrete kind, that kind is enforced —
    # permissiveness applies only to the named part.
    t = parse_type("Annotated[list[SFTDataset | DPODataset], MinLen(1)] | None")
    assert t.accepts([{"path": "x"}]) is True
    assert t.accepts("not-a-list") is False
    assert t.accepts(7) is False
