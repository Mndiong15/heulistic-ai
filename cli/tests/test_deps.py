"""Graceful degradation when heulistic-axolotl is missing.

`main.py` imports every command module to register it with typer, and four of
those modules used to import heulistic-axolotl at module scope — so a missing
or mis-placed dependency broke *every* command, including `stop`, `logs` and
`status`, which do not need it. Those are exactly the commands you want when
your install is in a bad state.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from heulistic import _deps

CLI_DIR = Path(__file__).resolve().parents[1]
PKG_SRC = CLI_DIR.parent / "packages" / "heulistic-axolotl" / "src"


# Clearing PYTHONPATH is not enough to hide the package: it is pip-installed
# (editable) in the interpreter running these tests, so it lives in
# site-packages. A sitecustomize shim, which CPython imports automatically at
# startup, blocks it at the import-system level instead.
_BLOCKER = """\
import sys


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "heulistic_axolotl" or name.startswith("heulistic_axolotl."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, _Block())
"""


@pytest.fixture(scope="module")
def blocker_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("no-axolotl")
    (path / "sitecustomize.py").write_text(_BLOCKER)
    return path


def _run(args, with_axolotl: bool, blocker_dir=None):
    """Run the CLI in a subprocess, optionally with the shared package blocked."""
    parts = [str(CLI_DIR), str(PKG_SRC)]
    if not with_axolotl:
        assert blocker_dir is not None, "pass blocker_dir to hide the package"
        parts.insert(0, str(blocker_dir))
    return subprocess.run(
        [sys.executable, "-m", "heulistic", *args],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ":".join(parts)},
    )


# ── Commands that must survive a missing dependency ───────────────────────────

def test_help_works_without_the_shared_package(blocker_dir):
    result = _run(["--help"], with_axolotl=False, blocker_dir=blocker_dir)
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    for command in ("start", "stop", "logs", "status", "train", "init"):
        assert command in result.stdout


def test_stop_does_not_crash_on_a_missing_dependency(blocker_dir):
    result = _run(["stop"], with_axolotl=False, blocker_dir=blocker_dir)
    assert "ModuleNotFoundError" not in result.stderr
    assert "heulistic_axolotl" not in result.stderr


def test_logs_does_not_crash_on_a_missing_dependency(blocker_dir):
    result = _run(["logs"], with_axolotl=False, blocker_dir=blocker_dir)
    assert "ModuleNotFoundError" not in result.stderr


# ── Commands that genuinely need it must explain themselves ───────────────────

@pytest.mark.parametrize("command", [["init"], ["validate-config", "x.yaml"]])
def test_dependent_commands_give_an_actionable_error(command, tmp_path, monkeypatch, blocker_dir):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.yaml").write_text("base_model: a/b\n")
    result = _run(command, with_axolotl=False, blocker_dir=blocker_dir)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "heulistic-axolotl" in combined
    assert "pip install" in combined
    # A bare traceback is not an explanation.
    assert "Traceback" not in combined


def test_commands_work_when_the_package_is_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run(["init"], with_axolotl=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "config.yaml").exists()


# ── The helper itself ─────────────────────────────────────────────────────────

def test_require_returns_the_module_when_present():
    module = _deps.require_axolotl()
    assert hasattr(module, "validate")
    assert hasattr(module, "Hardware")


def test_require_exits_with_a_message_when_absent(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("heulistic_axolotl"):
            raise ModuleNotFoundError("No module named 'heulistic_axolotl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.delitem(sys.modules, "heulistic_axolotl", raising=False)

    with pytest.raises(SystemExit) as exc:
        _deps.require_axolotl()
    assert exc.value.code == 1
    message = capsys.readouterr().err
    assert "heulistic-axolotl" in message
    assert "pip install" in message
