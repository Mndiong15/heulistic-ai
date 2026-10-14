"""The Axolotl config schema, ready to query.

This is the module consumers use. It rebuilds a `Reference` from the generated
`_schema_data` tables at import time — no YAML parsing, no reference file
needed at runtime.

    from heulistic_axolotl.schema import SCHEMA, get_field

    get_field("micro_batch_size").type.accepts(4)   # True
    get_field("micro_batch_size").type.accepts(True) # False
"""

from __future__ import annotations

from functools import lru_cache

from . import _schema_data
from .spec import Field, Reference, SubSchema, parse_type

__all__ = [
    "SCHEMA",
    "get_field",
    "known_field_names",
    "is_known_field",
    "get_model",
]


def _build_field(entry: tuple) -> Field:
    name, raw_type, required, has_default, default, doc = entry
    return Field(
        name=name,
        raw_type=raw_type,
        type=parse_type(raw_type),
        required=required,
        has_default=has_default,
        default=default,
        doc=doc,
    )


def _build() -> Reference:
    fields = {e[0]: _build_field(e) for e in _schema_data.FIELDS}
    subschemas = {
        name: SubSchema(
            name=name,
            parent=parent,
            fields={e[0]: _build_field(e) for e in entries},
        )
        for name, parent, entries in _schema_data.SUBSCHEMAS
    }
    total = len(fields) + sum(len(s.fields) for s in subschemas.values())
    return Reference(
        fields=fields,
        subschemas=subschemas,
        # Recorded from the source reference rather than recomputed, so a
        # truncated regeneration is detectable.
        consumed_field_lines=_schema_data.REFERENCE_FIELD_LINES,
    )


SCHEMA: Reference = _build()


def get_field(name: str) -> Field | None:
    """Look up a top-level field, or None if Axolotl has no such option."""
    return SCHEMA.fields.get(name)


def get_model(name: str) -> SubSchema | None:
    """Look up a nested model such as `SFTDataset` or `TRLConfig`."""
    return SCHEMA.subschemas.get(name)


@lru_cache(maxsize=1)
def known_field_names() -> frozenset[str]:
    """Every recognised top-level field name."""
    return frozenset(SCHEMA.fields)


def is_known_field(name: str) -> bool:
    return name in known_field_names()
