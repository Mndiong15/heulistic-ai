"""Access to the vendored Axolotl config reference.

The reference is the upstream config documentation
(https://docs.axolotl.ai/docs/config-reference.html) saved verbatim. It is the
input to schema generation, not something read at runtime — `schema.py` is
generated from it and committed, so refreshing Axolotl support is:

    1. replace `reference/axolotl.yaml` with the new upstream reference
    2. run `python -m heulistic.axolotl.generate`
    3. run the tests — `test_real_reference.py` reports what changed
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SOURCE_URL", "reference_path", "reference_text"]

#: Where the vendored reference comes from. Refreshing means replacing the body
#: of `reference/axolotl.yaml` with the config block published here.
SOURCE_URL = "https://docs.axolotl.ai/docs/config-reference.html"

_REFERENCE = Path(__file__).resolve().parent.parent.parent / "reference" / "axolotl.yaml"


def reference_path() -> Path:
    """Filesystem path to the vendored Axolotl config reference."""
    return _REFERENCE


def reference_text() -> str:
    return reference_path().read_text(encoding="utf-8")
