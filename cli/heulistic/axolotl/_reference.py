"""Parser for the Axolotl config reference format.

The reference published at https://docs.axolotl.ai/docs/config-reference.html
(vendored here as `reference/axolotl.yaml`) is a pseudo-YAML dump of Axolotl's
Pydantic models:

    # doc comment, possibly spanning
    # several lines
    field_name: type_expr = default
    other_field: str (required)

    parent_field: SomeModel | None
      # For SomeModel:
      nested_field: int | None

Nested models are introduced by a `# For <ModelName>:` marker and indented two
spaces past their parent. Indentation is 0, 2, or 4 throughout.

This module is a build-time tool: its output is generated into `schema.py`, so
neither the CLI nor the API parses the reference at import time.
"""

from __future__ import annotations

import re

from .spec import Field, Reference, SubSchema, parse_type

__all__ = ["ReferenceParseError", "parse_reference"]

_INDENT_STEP = 2
_REQUIRED_SUFFIX = "(required)"
_ROOT = ""

# Deliberately strict: the reference also contains prose comments that begin
# with "For " — e.g. "# For `completion` datasets only, uses the provided
# field…" — which are documentation, not model markers. Requiring a bare
# identifier followed by a closing colon keeps those out.
_MARKER_RE = re.compile(r"^#\s*For\s+([A-Za-z_][A-Za-z0-9_]*):\s*$")
_FIELD_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<rest>.*)$")


class ReferenceParseError(Exception):
    """Raised when the reference does not match the expected format."""


def _coerce_default(raw: str, type_expr) -> object:
    """Interpret a default value in light of the field's declared type.

    Defaults are written unquoted, so `0.0.0.0` (a host), `auto` (a literal),
    and `TorchAOQuantDType.int8` (an enum member) must all survive as strings
    rather than being coerced numerically.
    """
    raw = raw.strip()
    primitives = type_expr.primitives
    if "bool" in primitives and raw in ("True", "False"):
        return raw == "True"
    if "int" in primitives and raw.lstrip("-").isdigit():
        return int(raw)
    if "float" in primitives:
        try:
            return float(raw)
        except ValueError:
            pass
    return raw


def _parse_field_line(name: str, rest: str, lineno: int, doc: str) -> Field:
    required = False
    if rest.endswith(_REQUIRED_SUFFIX):
        required = True
        rest = rest[: -len(_REQUIRED_SUFFIX)].strip()

    has_default = False
    default: object = None
    if " = " in rest:
        rest, default_raw = rest.split(" = ", 1)
        rest = rest.strip()
        has_default = True

    if not rest:
        raise ReferenceParseError(f"line {lineno}: field '{name}' has no type")

    type_expr = parse_type(rest)
    if has_default:
        default = _coerce_default(default_raw, type_expr)

    return Field(
        name=name,
        raw_type=rest,
        type=type_expr,
        required=required,
        has_default=has_default,
        default=default,
        doc=doc,
        lineno=lineno,
    )


def parse_reference(text: str) -> Reference:
    """Parse the Axolotl reference into a `Reference`."""
    root = SubSchema(name=_ROOT)
    stack: list[tuple[int, SubSchema]] = [(0, root)]
    # Every occurrence of each model, reconciled once parsing completes.
    occurrences: dict[str, list[SubSchema]] = {}
    doc_lines: list[str] = []
    consumed = 0

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            # A comment divorced from its field by a blank line documents
            # nothing — dropping it prevents misattribution.
            doc_lines.clear()
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line.startswith("#"):
            marker = _MARKER_RE.match(line)
            if not marker:
                doc_lines.append(line.lstrip("#").strip())
                continue

            model_name = marker.group(1)
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            enclosing_indent, enclosing = stack[-1]
            if indent != enclosing_indent + _INDENT_STEP:
                raise ReferenceParseError(
                    f"line {lineno}: '# For {model_name}:' is indented {indent}, "
                    f"expected {enclosing_indent + _INDENT_STEP}"
                )
            if not enclosing.fields:
                raise ReferenceParseError(
                    f"line {lineno}: '# For {model_name}:' has no parent field"
                )
            schema = SubSchema(
                name=model_name,
                parent=enclosing.name or None,
            )
            occurrences.setdefault(model_name, []).append(schema)
            stack.append((indent, schema))
            doc_lines.clear()
            continue

        match = _FIELD_RE.match(line)
        if not match:
            raise ReferenceParseError(f"line {lineno}: cannot parse {line!r}")

        while len(stack) > 1 and stack[-1][0] > indent:
            stack.pop()
        if stack[-1][0] != indent:
            raise ReferenceParseError(
                f"line {lineno}: field indented {indent} has no enclosing block"
            )

        container = stack[-1][1]
        name = match.group("name")
        container.fields[name] = _parse_field_line(
            name, match.group("rest").strip(), lineno, " ".join(doc_lines)
        )
        consumed += 1
        doc_lines.clear()

    return Reference(
        fields=root.fields,
        subschemas=_reconcile(occurrences),
        consumed_field_lines=consumed,
    )


def _reconcile(occurrences: dict[str, list[SubSchema]]) -> dict[str, SubSchema]:
    """Collapse repeated model definitions, rejecting genuine conflicts.

    `SFTDataset` and friends are spelled out once per field that references
    them. Identical repeats collapse to one entry; a repeat whose fields differ
    means the reference is ambiguous and generation must stop.
    """
    merged: dict[str, SubSchema] = {}
    for name, instances in occurrences.items():
        first = instances[0]
        for other in instances[1:]:
            if other.signature != first.signature:
                raise ReferenceParseError(
                    f"model '{name}' is defined more than once with differing fields "
                    f"(line {_first_lineno(first)} vs line {_first_lineno(other)})"
                )
        merged[name] = first
    return merged


def _first_lineno(schema: SubSchema) -> int:
    return next((f.lineno for f in schema.fields.values()), 0)
