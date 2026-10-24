"""Types describing the Axolotl config schema.

`TypeExpr` is a parsed form of the type expressions used in the Axolotl config
reference — `str | None`, `Literal['auto'] | bool | None`,
`Annotated[list[SFTDataset | DPODataset], MinLen(1)] | None`, and so on.

The parsed form is deliberately shallow: it captures what a config validator
needs to answer (does this value fit? what are the allowed values? which
sub-schema applies?) rather than reconstructing Python's typing algebra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

__all__ = ["TypeExpr", "parse_type", "Field", "SubSchema", "Reference"]

_SCALARS = {"str", "int", "float", "bool"}
_NAMED_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_REQUIRED_SUFFIX = "(required)"


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on `sep`, ignoring separators nested inside brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _inner(text: str) -> str:
    """Return the contents of the outermost bracket pair."""
    start = text.find("[")
    end = text.rfind("]")
    return text[start + 1 : end] if start != -1 and end > start else ""


def _literal_value(token: str) -> Any:
    token = token.strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if token in ("True", "False"):
        return token == "True"
    if token.lstrip("-").isdigit():
        return int(token)
    return token


@dataclass(frozen=True)
class TypeExpr:
    """A parsed type expression.

    `primitives` holds the scalar/container kinds the field accepts, drawn from
    ``{"str", "int", "float", "bool", "list", "dict", "any"}``. `literals` holds
    the permitted values when the type is (or includes) a ``Literal[...]``, and
    `named` holds referenced model names that have their own sub-schema.
    """

    raw: str
    optional: bool
    primitives: frozenset[str]
    literals: tuple[Any, ...] = ()
    named: tuple[str, ...] = ()
    # Primitives contributed *only* by Literal members. A string field whose
    # sole string-ness comes from `Literal['auto']` accepts "auto" and nothing
    # else, so these are excluded when falling back to a plain kind check.
    literal_primitives: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def accepts_anything(self) -> bool:
        return "any" in self.primitives

    def accepts(self, value: Any) -> bool:
        """True if `value` is a plausible instance of this type.

        Deliberately permissive around named types. Of the 30 model names the
        reference mentions, only 23 have a `# For X:` block; the rest —
        `RLType`, `CustomSupportedOptimizers`, `FP8Recipe`, `ChatTemplate`,
        `Resampling`, `RingAttnFunc`, `TorchAOQuantDType` — are enums whose
        members appear only in prose. Judging a value against a member list we
        do not have would reject correct configs, so when a named type is the
        only thing left to check against, the value is allowed through.
        """
        if value is None:
            return self.optional
        if self.accepts_anything:
            return True
        if self.literals and any(
            type(value) is type(lit) and value == lit for lit in self.literals
        ):
            return True

        kinds = self.primitives - self.literal_primitives
        if _matches_kind(value, kinds):
            return True

        if self.named:
            # A named member is either a model — a mapping, once written in
            # YAML — or an undocumented enum. Accept mappings either way, and
            # accept anything when the type offers no other kind to test.
            if isinstance(value, dict):
                return True
            if not kinds:
                return True
        return False


def _matches_kind(value: Any, kinds: frozenset[str]) -> bool:
    # `bool` is a subclass of `int` in Python; keep them distinct so neither
    # `micro_batch_size: true` nor `bf16: 1` slips through as valid.
    value_type = type(value)
    if value_type is bool:
        return "bool" in kinds
    if value_type is int:
        # An int is acceptable wherever a float is expected (learning_rate: 1).
        return "int" in kinds or "float" in kinds
    if value_type is float:
        return "float" in kinds
    if isinstance(value, str):
        return "str" in kinds
    if isinstance(value, list):
        return "list" in kinds
    if isinstance(value, dict):
        return "dict" in kinds
    return False


@lru_cache(maxsize=None)
def parse_type(expression: str) -> TypeExpr:
    """Parse an Axolotl reference type expression into a `TypeExpr`.

    Cached: the reference's 600-odd fields draw on barely a hundred distinct
    type expressions, and the schema is rebuilt from them on every import.
    """
    raw = expression.strip()
    cleaned = raw.removesuffix(_REQUIRED_SUFFIX).strip()

    optional = False
    primitives: set[str] = set()
    literal_primitives: set[str] = set()
    literals: list[Any] = []
    named: list[str] = []

    def absorb(member: str) -> None:
        nonlocal optional
        member = member.strip()
        if not member:
            return
        if member == "None":
            optional = True
        elif member == "Any":
            primitives.add("any")
        elif member in _SCALARS:
            primitives.add(member)
        elif member.startswith("Literal["):
            for token in _split_top_level(_inner(member), ","):
                value = _literal_value(token)
                literals.append(value)
                kind = {str: "str", bool: "bool", int: "int"}.get(type(value), "str")
                primitives.add(kind)
                literal_primitives.add(kind)
        elif member.startswith("list["):
            primitives.add("list")
            # Element models matter — `list[SFTDataset | DPODataset]` drives
            # per-entry validation of `datasets`.
            for element in _split_top_level(_inner(member), "|"):
                if _NAMED_RE.match(element) and element not in named:
                    named.append(element)
        elif member.startswith("dict["):
            # Value types inside dicts (`dict[str, PositiveInt]`) are pydantic
            # scalars with no sub-schema block, so they are not collected.
            primitives.add("dict")
        elif member.startswith("Annotated["):
            # Annotated[T, *constraints] — only T carries type information.
            args = _split_top_level(_inner(member), ",")
            if args:
                for part in _split_top_level(args[0], "|"):
                    absorb(part)
        elif _NAMED_RE.match(member):
            if member not in named:
                named.append(member)

    for member in _split_top_level(cleaned, "|"):
        absorb(member)

    return TypeExpr(
        raw=raw,
        optional=optional,
        primitives=frozenset(primitives),
        literals=tuple(literals),
        named=tuple(named),
        literal_primitives=frozenset(literal_primitives),
    )


@dataclass(frozen=True)
class Field:
    """One configuration field from the Axolotl reference."""

    name: str
    raw_type: str
    type: TypeExpr
    required: bool = False
    has_default: bool = False
    default: Any = None
    doc: str = ""
    lineno: int = 0

    @property
    def signature(self) -> tuple:
        """Structural identity, used to detect conflicting redefinitions.

        Doc text is excluded: the reference repeats models such as `SFTDataset`
        verbatim except for occasional comment rewording, which is not a schema
        change.
        """
        return (self.name, self.raw_type, self.required, self.has_default, self.default)


@dataclass
class SubSchema:
    """A nested model (`SFTDataset`, `TRLConfig`, …) with its own fields."""

    name: str
    parent: str | None = None
    fields: dict[str, Field] = field(default_factory=dict)

    @property
    def signature(self) -> tuple:
        return tuple(f.signature for f in self.fields.values())


@dataclass
class Reference:
    """The complete parsed Axolotl config reference."""

    fields: dict[str, Field] = field(default_factory=dict)
    subschemas: dict[str, SubSchema] = field(default_factory=dict)
    # Field lines consumed from the source, counting repeated model definitions
    # each time they appear. `total_field_count` counts them once, after
    # merging, so the gap between the two is pure duplication — and a gap that
    # is *not* explained by duplication would mean fields went missing.
    consumed_field_lines: int = 0

    @property
    def field_count(self) -> int:
        """Number of top-level fields."""
        return len(self.fields)

    @property
    def total_field_count(self) -> int:
        """Top-level fields plus every field of every sub-schema."""
        return len(self.fields) + sum(len(s.fields) for s in self.subschemas.values())
