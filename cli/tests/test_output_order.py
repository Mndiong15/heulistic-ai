"""Progress output must stay in order when redirected.

Progress went to stdout and errors to stderr. Redirected to a file or a pipe,
stdout is block-buffered while stderr is not, so a failure appeared *above*
the attempts that produced it:

    ✗ Failed to pull heulistic/api:latest after 3 attempts.
      Pulling heulistic/api:latest…
      Pulling heulistic/api:latest  (retry)…

Interactively it reads fine. In a log pasted into an issue — which is how a
stranger's failed install gets debugged — it inverts cause and effect.
"""

import subprocess
import sys
from pathlib import Path

CLI_DIR = Path(__file__).resolve().parents[1]


def _run_script(body: str, tmp_path: Path) -> list[str]:
    """Run a snippet with stdout and stderr merged into one file, as a shell
    redirect or a CI log would."""
    script = tmp_path / "snippet.py"
    script.write_text(body)
    out = tmp_path / "merged.log"
    with open(out, "w") as handle:
        subprocess.run(
            [sys.executable, str(script)],
            stdout=handle,
            stderr=handle,
            env={"PATH": "", "PYTHONPATH": str(CLI_DIR)},
            check=True,
        )
    return [line for line in out.read_text().splitlines() if line.strip()]


def test_progress_keeps_its_place_around_an_error(tmp_path):
    lines = _run_script(
        """
import sys
from heulistic import local_runner as lr

lr._say("attempt 1")
lr._say("attempt 2")
print("FAILED after 2 attempts", file=sys.stderr)
""",
        tmp_path,
    )
    assert lines == ["attempt 1", "attempt 2", "FAILED after 2 attempts"], lines


def test_the_bug_reproduces_without_flushing(tmp_path):
    """Proof the test above is actually testing something.

    The same sequence through a plain unflushed print inverts, which is the
    original defect.
    """
    lines = _run_script(
        """
import sys
print("attempt 1")
print("attempt 2")
print("FAILED after 2 attempts", file=sys.stderr)
""",
        tmp_path,
    )
    assert lines[0] == "FAILED after 2 attempts", (
        "expected the unflushed version to invert; if this fails, Python's "
        "buffering changed and the regression test above proves less"
    )


def test_say_writes_to_stdout(tmp_path):
    """Progress belongs on stdout so it can be piped separately from errors."""
    script = tmp_path / "s.py"
    script.write_text("from heulistic import local_runner as lr\nlr._say('hello')\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={"PATH": "", "PYTHONPATH": str(CLI_DIR)},
    )
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
