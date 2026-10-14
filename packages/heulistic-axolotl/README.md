# heulistic-axolotl

Axolotl config schema and validation, as a small dependency-light Python package.

It is the shared core behind `heulistic validate-config` in the CLI and config
validation in the Heulistic API, so both agree on what a valid Axolotl config
looks like.

## What it gives you

The schema is **generated from Axolotl's own config reference**, not hand-written,
so it knows every field's type, whether it is required, its default, and its
documentation — 390 top-level fields and 23 nested models at present.

```python
from heulistic_axolotl.schema import get_field, is_known_field

get_field("micro_batch_size").type.accepts(4)      # True
get_field("micro_batch_size").type.accepts(True)   # False — bool is not an int
get_field("bf16").type.accepts("auto")             # True — Literal['auto'] | bool
get_field("learning_rate").required                # True
get_field("sample_packing").doc                    # the upstream doc comment

is_known_field("lr")                               # False — it's `learning_rate`
```

## Refreshing for a new Axolotl release

1. Replace `reference/axolotl.yaml` with the current
   [config reference](https://docs.axolotl.ai/docs/config-reference.html).
2. Regenerate the schema:

   ```bash
   python -m heulistic_axolotl.generate
   ```

3. Run the tests:

   ```bash
   pytest
   ```

`tests/test_real_reference.py` holds field counts as tripwires, so a refresh
that changes the schema fails loudly and the diff of `_schema_data.py` shows
exactly which fields were added, removed, or retyped. Update the constants once
the change is confirmed against the Axolotl changelog.

## Layout

| Path | Role |
|---|---|
| `reference/axolotl.yaml` | Upstream reference, verbatim. Dev input; not shipped in the wheel. |
| `src/heulistic_axolotl/_reference.py` | Parser for the reference format. Build-time only. |
| `src/heulistic_axolotl/generate.py` | Writes `_schema_data.py`. Run by hand. |
| `src/heulistic_axolotl/_schema_data.py` | **Generated.** Committed so schema changes are reviewable. |
| `src/heulistic_axolotl/schema.py` | The runtime API — `SCHEMA`, `get_field`, `get_model`. |
| `src/heulistic_axolotl/spec.py` | `Field`, `TypeExpr`, and type-expression parsing. |

## Licence

MIT.
