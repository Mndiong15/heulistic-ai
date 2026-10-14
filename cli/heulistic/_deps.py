"""Lazy access to heulistic-axolotl.

`main.py` has to import every command module so typer can register it, so any
module-scope dependency becomes a dependency of the *whole* CLI. Importing the
shared package lazily keeps `stop`, `logs` and `status` working when the
install is broken — which is precisely when they are wanted.

It also means a split environment reports something useful instead of a
traceback. The usual cause is the CLI and its dependency landing under
different interpreters.
"""

from __future__ import annotations

import sys
from types import ModuleType

__all__ = ["require_axolotl"]

_MESSAGE = """\
  ✗ heulistic-axolotl is not installed in this environment.

    This command needs it. `heulistic start`, `stop`, `logs` and `status` do not.

    Install it:
      pip install heulistic-axolotl

    If `heulistic` itself runs but cannot see its own dependency, the two are
    probably installed under different interpreters. Check with:
      python -c "import heulistic, heulistic_axolotl"
      which heulistic
    and reinstall both into one environment:
      pip install --force-reinstall heulistic
"""


def require_axolotl() -> ModuleType:
    """Return the heulistic_axolotl module, or exit with an explanation."""
    try:
        import heulistic_axolotl
    except ModuleNotFoundError:
        print(_MESSAGE, file=sys.stderr)
        raise SystemExit(1)
    return heulistic_axolotl
