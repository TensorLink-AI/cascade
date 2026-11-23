"""Shared helpers for the W0c acceptance tests.

Every W0c tool is driven as a subprocess (``python scripts/laned/<tool> ...``)
so the tests pin the CLI contract, not internals. Fixture repos are real git
repos built in ``tmp_path``; the only mocked service is GitHub Actions, whose
results are fed to ``scripts/laned/status`` as a JSON file.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LANED = REPO / "scripts" / "laned"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

TOOLS = ("accept", "status", "stub_scan.py", "unread_keys.py", "follower_patterns.py")

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "laned-fixture",
    "GIT_AUTHOR_EMAIL": "laned-fixture@example.invalid",
    "GIT_COMMITTER_NAME": "laned-fixture",
    "GIT_COMMITTER_EMAIL": "laned-fixture@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def tool_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # fixture repos use `(#N)` subjects: never the real repo's merged-PR
    # labels or CI ledger (the laned-accept workflow exports both, rev6.17)
    env.pop("LANED_PR_LABELS", None)
    env.pop("LANED_LEDGER", None)
    env.update(_GIT_IDENTITY)
    # accept/expect-fail run this instead of `uv run --frozen pytest` (README).
    env["LANED_PYTEST_CMD"] = shlex.join([sys.executable, "-m", "pytest", "-p", "no:cacheprovider"])
    if extra:
        env.update(extra)
    return env


def tool_path(name: str) -> Path:
    path = LANED / name
    assert path.is_file(), f"W0c tool missing: {path.relative_to(REPO)}"
    return path


def run_tool(name: str, *args: object, cwd: Path | None = None,
             env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    path = tool_path(name)
    return subprocess.run(
        [sys.executable, str(path), *[str(a) for a in args]],
        cwd=str(cwd) if cwd else None,
        env=tool_env(env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=tool_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def git_bare(bare: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--git-dir", str(bare), *args],
        env=tool_env(),
        capture_output=True,
        text=True,
    )


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    return path


def write(repo: Path, rel: str, text: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def copy_fixture(src: Path, repo: Path, rel: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)


def commit(repo: Path, message: str, paths: list[str]) -> str:
    git(repo, "add", "--", *paths)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# The status fixture repo (see README "status fixture").
# ---------------------------------------------------------------------------

STATUS_FIXTURE = FIXTURES / "status"

_WP_PATHS = {
    "W0c": "scripts/laned/\n",
    "W0": "decisions/\n",
    "W1": "cascade/w1.py\n",
    "W2": "cascade/w2.py\n",
    "W3": "cascade/w3.py\n",
    "W4": "cascade/w4.py\n",
    "W7": "cascade/w7.py\n",
    "R6x": "deploy/\n",
    "R2x": "scripts/r2x.py\n",
}

_WP_TESTS = {
    "W0c": "def test_inv10_pairing():\n    assert True\n",
    "W0": "def test_inv9_share_zero():\n    assert True\n",
    "W1": "def test_inv1_unarmed():\n    assert True\n",
}

_EVIDENCE = (
    "evidence/s0a_diff.json",
    "evidence/r2x.json",
    "evidence/rv1.md",
    "evidence/l7/w0c.json",
    "evidence/blocked/w7.md",
    "evidence/gates/s0.json",
    "evidence/gates/s0b.json",
    "evidence/s3_matrix.json",
    "evidence/owner/oa8.json",
)


def wp_dir(wp: str) -> str:
    return wp.lower().replace(".", "_").replace("-", "_")


def build_status_repo(root: Path, d99_open: bool = True) -> tuple[Path, dict[str, str]]:
    """Build the fixture repo; return (repo, {label: sha})."""
    repo = init_repo(root / "fixture")
    shas: dict[str, str] = {}

    plan = (STATUS_FIXTURE / "plan.md").read_text()
    if not d99_open:
        plan = plan.replace("| D99 | open | — | S | — |",
                            "| D99 | decided | (a) | S | owner, 2026-11-21 |")
    write(repo, "docs/plans/LANE_D_PLAN.md", plan)
    write(repo, "README.md", "fixture repo for scripts/laned/status\n")
    shas["base"] = commit(repo, "plan", ["docs/plans/LANE_D_PLAN.md", "README.md"])

    # A PR head that never lands on main (squash merge): expect-fail runs.
    git(repo, "checkout", "-q", "-b", "pr-head")
    write(repo, "pr.txt", "pr head\n")
    shas["pr"] = commit(repo, "pr head", ["pr.txt"])
    git(repo, "checkout", "-q", "main")

    for wp, paths in _WP_PATHS.items():
        d = wp_dir(wp)
        test_body = _WP_TESTS.get(wp, "def test_spec():\n    assert True\n")
        write(repo, f"tests/accept/{d}/PATHS", paths)
        write(repo, f"tests/accept/{d}/test_{d}.py", test_body)
        shas[f"tests_{d}"] = commit(repo, f"tests(laned): {wp} acceptance tests",
                                    [f"tests/accept/{d}/PATHS", f"tests/accept/{d}/test_{d}.py"])

    impls = [
        ("impl_w0c", ["scripts/laned/tool.py"]),
        ("impl_w0", ["decisions/DEC-CA-0070.md"]),
        ("impl_r6x", ["deploy/reconciled.toml"]),
        ("impl_r2x", ["scripts/r2x.py"]),
    ]
    for label, paths in impls:
        for p in paths:
            write(repo, p, f"implementation for {label}\n")
        shas[label] = commit(repo, label, paths)
    git(repo, "tag", "cap-1")

    write(repo, "cascade/w1.py", "VALUE = 1\n")
    shas["impl_w1"] = commit(repo, "impl_w1", ["cascade/w1.py"])
    # W4's implementation also edits a path outside its PATHS: not "built".
    write(repo, "cascade/w4.py", "VALUE = 4\n")
    write(repo, "README.md", "fixture repo for scripts/laned/status (edited by W4)\n")
    shas["impl_w4"] = commit(repo, "impl_w4", ["cascade/w4.py", "README.md"])

    for rel in _EVIDENCE:
        copy_fixture(STATUS_FIXTURE / rel, repo, rel)
    shas["evidence"] = commit(repo, "evidence", list(_EVIDENCE))

    write(repo, "docs/notes.md", "notes\n")
    shas["docs"] = commit(repo, "docs", ["docs/notes.md"])
    return repo, shas


def materialise_ci_results(shas: dict[str, str], dest: Path) -> Path:
    data = json.loads((STATUS_FIXTURE / "ci_results.json").read_text())
    for run in data["runs"]:
        label = run["head_sha"]
        assert label.startswith("@"), label
        run["head_sha"] = shas[label[1:]]
    dest.write_text(json.dumps(data, indent=2))
    return dest
