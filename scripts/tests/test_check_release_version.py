"""Tests for the release version guard.

The guard is the only thing standing between a mistyped tag and a permanent
PyPI version that disagrees with the container images carrying the same name.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_release_version as guard  # noqa: E402


class TestCheck:
    def test_matching_tag_passes(self):
        assert guard.check("v0.1.3", "0.1.3") is None

    def test_mismatched_tag_is_rejected(self):
        problem = guard.check("v0.1.2", "0.1.3")
        assert problem is not None
        # The message must name both sides; "mismatch" alone is not actionable.
        assert "0.1.2" in problem and "0.1.3" in problem

    def test_missing_v_prefix_is_rejected(self):
        problem = guard.check("0.1.3", "0.1.3")
        assert problem is not None
        assert "v0.1.3" in problem

    def test_unrelated_tag_is_rejected(self):
        assert guard.check("v1.0.0", "0.1.3") is not None

    @pytest.mark.parametrize("version", ["0.1.3", "1.0.0", "2.11.0", "0.2.0rc1"])
    def test_any_matching_pair_passes(self, version):
        assert guard.check(f"v{version}", version) is None


class TestDeclaredVersion:
    def test_reads_the_real_cli_pyproject(self):
        version = guard.declared_version()
        assert version.count(".") >= 2, version

    def test_reads_a_given_file(self, tmp_path):
        p = tmp_path / "pyproject.toml"
        p.write_text('[project]\nname = "x"\nversion = "9.9.9"\n')
        assert guard.declared_version(p) == "9.9.9"

    def test_a_file_without_a_version_is_an_error(self, tmp_path):
        p = tmp_path / "pyproject.toml"
        p.write_text('[project]\nname = "x"\n')
        with pytest.raises(SystemExit):
            guard.declared_version(p)

    def test_the_real_repo_is_currently_consistent(self):
        """Guards against committing a version the next tag cannot match."""
        version = guard.declared_version()
        assert guard.check(f"v{version}", version) is None


class TestExitCodes:
    def test_success(self):
        assert guard.main(["prog", f"v{guard.declared_version()}"]) == 0

    def test_failure(self):
        assert guard.main(["prog", "v0.0.0-not-the-version"]) == 1

    def test_usage_error(self):
        assert guard.main(["prog"]) == 2
