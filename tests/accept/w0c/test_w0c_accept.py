"""W0c: `scripts/laned/accept <WP> [--evidence <file>]` (§3 conventions):
pytest over tests/accept/<wp>/, the stub scan over PATHS, `make laned-e2e`
when the repo defines it, the evidence verifier, and invariant 10."""

from __future__ import annotations

import json
import shutil

import pytest

from .helpers import commit, init_repo, output, run_tool, write

IMPL = "def double(x: int) -> int:\n    return 2 * x\n"
TEST_OK = "from lanedpkg.w9 import double\n\n\ndef test_double():\n    assert double(3) == 6\n"
# Built by concatenation so this module's own test names never match the
# invariant-10 convention.
UNARMED = "unchanged_when_" + "unarmed"
ARMED = "armed_nonempty_" + "lane_d_effect"


def _repo(tmp_path, tests=None, impl=IMPL, extra=None, wp_dir="w9"):
    repo = init_repo(tmp_path / "repo")
    write(repo, "lanedpkg/__init__.py", "")
    write(repo, "lanedpkg/w9.py", impl)
    write(repo, f"tests/accept/{wp_dir}/PATHS", "lanedpkg/w9.py\n")
    for name, src in (tests or {"test_w9.py": TEST_OK}).items():
        write(repo, f"tests/accept/{wp_dir}/{name}", src)
    for rel, src in (extra or {}).items():
        write(repo, rel, src)
    commit(repo, "fixture", ["."])
    return repo


def _accept(repo, wp="W9", *args):
    return run_tool("accept", wp, "--root", repo, *args)


def test_accept_passes_on_green_tests_and_clean_paths(tmp_path):
    proc = _accept(_repo(tmp_path))
    assert proc.returncode == 0, output(proc)


def test_accept_maps_wp_id_to_lowercase_underscore_dir(tmp_path):
    repo = _repo(tmp_path, wp_dir="w_s1")
    proc = _accept(repo, "W-S1")
    assert proc.returncode == 0, output(proc)


def test_accept_fails_on_red_tests(tmp_path):
    tests = {"test_w9.py": TEST_OK + "\n\ndef test_red():\n    assert double(1) == 3\n"}
    proc = _accept(_repo(tmp_path, tests=tests))
    assert proc.returncode != 0, output(proc)
    assert "FAIL pytest" in output(proc), output(proc)


def test_accept_fails_on_stub_in_declared_paths(tmp_path):
    impl = "def double(x: int) -> int:\n    # TODO: overflow guard\n    return 2 * x\n"
    proc = _accept(_repo(tmp_path, impl=impl))
    assert proc.returncode != 0, output(proc)
    assert "FAIL stub_scan" in output(proc), output(proc)


def test_accept_stub_scan_limited_to_declared_paths(tmp_path):
    extra = {"tools/scratch.py": "def f():\n    # TODO: not this WP's path\n    return 1\n"}
    proc = _accept(_repo(tmp_path, extra=extra))
    assert proc.returncode == 0, output(proc)


# ---- invariant 10 ---------------------------------------------------------

def test_inv10_unpaired_unarmed_test_fails_accept(tmp_path):
    name = f"test_receipt_{UNARMED}"
    tests = {"test_w9.py": TEST_OK + f"\n\ndef {name}():\n    assert double(0) == 0\n"}
    proc = _accept(_repo(tmp_path, tests=tests))
    assert proc.returncode != 0, output(proc)
    assert "FAIL invariant10" in output(proc), output(proc)
    assert name in output(proc), output(proc)


def test_inv10_pair_in_another_file_passes(tmp_path):
    tests = {
        "test_w9.py": TEST_OK + f"\n\ndef test_receipt_{UNARMED}():\n    assert double(0) == 0\n",
        "test_w9_armed.py": ("from lanedpkg.w9 import double\n\n\n"
                             f"def test_receipt_{ARMED}():\n    assert double(2) == 4\n"),
    }
    proc = _accept(_repo(tmp_path, tests=tests))
    assert proc.returncode == 0, output(proc)


def test_inv10_pair_as_class_methods_passes(tmp_path):
    tests = {"test_w9.py": TEST_OK + (
        "\n\nclass TestWeights:\n"
        f"    def test_vector_{UNARMED}(self):\n        assert double(0) == 0\n\n"
        f"    def test_vector_{ARMED}(self):\n        assert double(1) == 2\n")}
    proc = _accept(_repo(tmp_path, tests=tests))
    assert proc.returncode == 0, output(proc)


def test_inv10_pair_with_different_subject_does_not_count(tmp_path):
    tests = {"test_w9.py": TEST_OK + (
        f"\n\ndef test_receipt_{UNARMED}():\n    assert double(0) == 0\n"
        f"\n\ndef test_manifest_{ARMED}():\n    assert double(1) == 2\n")}
    proc = _accept(_repo(tmp_path, tests=tests))
    assert proc.returncode != 0, output(proc)
    assert "FAIL invariant10" in output(proc), output(proc)


# ---- evidence ---------------------------------------------------------------

@pytest.mark.parametrize("body,passes", [
    ({"wp": "W9", "passed": True, "request_ids": ["r-1"]}, True),
    ({"wp": "W8", "passed": True}, False),
    ({"wp": "W9", "passed": False}, False),
    ({"wp": "W9"}, False),
])
def test_accept_evidence_json(tmp_path, body, passes):
    repo = _repo(tmp_path, extra={"evidence/w9.json": json.dumps(body)})
    proc = _accept(repo, "W9", "--evidence", "evidence/w9.json")
    if passes:
        assert proc.returncode == 0, output(proc)
    else:
        assert proc.returncode != 0, output(proc)
        assert "FAIL evidence" in output(proc), output(proc)


def test_accept_evidence_missing_file_fails(tmp_path):
    proc = _accept(_repo(tmp_path), "W9", "--evidence", "evidence/w9.json")
    assert proc.returncode != 0, output(proc)
    assert "FAIL evidence" in output(proc), output(proc)


def test_accept_evidence_not_json_fails(tmp_path):
    repo = _repo(tmp_path, extra={"evidence/w9.json": "{truncated"})
    proc = _accept(repo, "W9", "--evidence", "evidence/w9.json")
    assert proc.returncode != 0, output(proc)


@pytest.mark.parametrize("review,passes", [
    ("Covers: W1, W2\n\n- [x] BLOCKING W1: fixed in #500\n- [ ] note: style nit\n", True),
    ("Covers: W1, W2\n\n- [x] BLOCKING W1: fixed in #500\n- [ ] BLOCKING W2: unresolved\n", False),
])
def test_accept_review_markdown_evidence(tmp_path, review, passes):
    repo = _repo(tmp_path, wp_dir="rv1", extra={"evidence/rv1.md": review})
    proc = _accept(repo, "RV1", "--evidence", "evidence/rv1.md")
    if passes:
        assert proc.returncode == 0, output(proc)
    else:
        assert proc.returncode != 0, output(proc)


# ---- make laned-e2e ---------------------------------------------------------

@pytest.mark.parametrize("recipe,passes", [("true", True), ("false", False)])
def test_accept_runs_make_laned_e2e_when_defined(tmp_path, recipe, passes):
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    repo = _repo(tmp_path, extra={"Makefile": f".PHONY: laned-e2e\nlaned-e2e:\n\t{recipe}\n"})
    proc = _accept(repo)
    if passes:
        assert proc.returncode == 0, output(proc)
    else:
        assert proc.returncode != 0, output(proc)
        assert "FAIL e2e" in output(proc), output(proc)
