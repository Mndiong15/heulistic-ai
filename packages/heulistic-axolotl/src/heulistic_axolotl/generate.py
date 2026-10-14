"""Generate `_schema_data.py` from the vendored Axolotl config reference.

    python -m heulistic_axolotl.generate

The generated module is committed. That is deliberate: the schema then arrives
as reviewable source, so refreshing Axolotl support produces a diff showing
exactly which fields were added, removed, or retyped — and the installed wheel
carries no dependency on the reference file itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ._reference import parse_reference
from .reference import reference_path, reference_text
from .spec import Field, Reference

_OUTPUT = Path(__file__).resolve().parent / "_schema_data.py"

_HEADER = '''\
"""Axolotl config schema — GENERATED, do not edit by hand.

Source     : reference/axolotl.yaml
Regenerate : python -m heulistic_axolotl.generate

Each entry is (name, raw_type, required, has_default, default, doc).
"""

# Number of field lines in the source reference, including the repeated model
# definitions that collapse during parsing. Guards against a truncated refresh.
REFERENCE_FIELD_LINES = {lines}

FIELDS: tuple[tuple, ...] = (
'''


def _render_field(f: Field, indent: str) -> str:
    return (
        f"{indent}({f.name!r}, {f.raw_type!r}, {f.required!r}, "
        f"{f.has_default!r}, {f.default!r}, {f.doc!r}),\n"
    )


def render(reference: Reference) -> str:
    out = [_HEADER.format(lines=reference.consumed_field_lines)]

    for field in reference.fields.values():
        out.append(_render_field(field, "    "))
    out.append(")\n\nSUBSCHEMAS: tuple[tuple, ...] = (\n")

    for schema in reference.subschemas.values():
        out.append(f"    ({schema.name!r}, {schema.parent!r}, (\n")
        for field in schema.fields.values():
            out.append(_render_field(field, "        "))
        out.append("    )),\n")
    out.append(")\n")

    return "".join(out)


def main() -> int:
    reference = parse_reference(reference_text())
    _OUTPUT.write_text(render(reference), encoding="utf-8")

    print(f"Read    {reference_path()}")
    print(f"Wrote   {_OUTPUT}")
    print(
        f"Schema  {reference.field_count} top-level fields, "
        f"{len(reference.subschemas)} models, "
        f"{reference.total_field_count} fields total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
