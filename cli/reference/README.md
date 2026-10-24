# `heulistic.axolotl` — the Axolotl config schema

Axolotl config schema and validation, inside the `heulistic` package.

It is the shared core behind `heulistic validate-config` in the CLI, config
validation in the API and GPU planning in the trainer, so all three agree on
what a valid Axolotl config looks like.

## What it gives you

The schema is **generated from Axolotl's own config reference**, not hand-written,
so it knows every field's type, whether it is required, its default, and its
documentation — 390 top-level fields and 23 nested models at present.

```python
from heulistic.axolotl.schema import get_field, is_known_field

get_field("micro_batch_size").type.accepts(4)      # True
get_field("micro_batch_size").type.accepts(True)   # False — bool is not an int
get_field("bf16").type.accepts("auto")             # True — Literal['auto'] | bool
get_field("learning_rate").required                # True
get_field("sample_packing").doc                    # the upstream doc comment

is_known_field("lr")                               # False — it's `learning_rate`
```

## Refreshing for a new Axolotl release

1. Replace `cli/reference/axolotl.yaml` with the current
   [config reference](https://docs.axolotl.ai/docs/config-reference.html).
2. Regenerate the schema:

   ```bash
   cd cli && python -m heulistic.axolotl.generate
   ```

3. Run the tests:

   ```bash
   cd cli && pytest tests/axolotl
   ```

`cli/tests/axolotl/test_real_reference.py` holds field counts as tripwires, so a refresh
that changes the schema fails loudly and the diff of `_schema_data.py` shows
exactly which fields were added, removed, or retyped. Update the constants once
the change is confirmed against the Axolotl changelog.

## Layout

| Path | Role |
|---|---|
| `cli/reference/axolotl.yaml` | Upstream reference, verbatim. Dev input; not shipped in the wheel. |
| `cli/heulistic/axolotl/_reference.py` | Parser for the reference format. Build-time only. |
| `cli/heulistic/axolotl/generate.py` | Writes `_schema_data.py`. Run by hand. |
| `cli/heulistic/axolotl/_schema_data.py` | **Generated.** Committed so schema changes are reviewable. |
| `cli/heulistic/axolotl/schema.py` | The runtime API — `SCHEMA`, `get_field`, `get_model`. |
| `cli/heulistic/axolotl/spec.py` | `Field`, `TypeExpr`, and type-expression parsing. |
