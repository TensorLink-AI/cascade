"""R12: venv sync (plan §10.2 "the release's uv sync --frozen venv
(torch 2.4.1+cu124), re-synced with every validator deploy").

``evalhost.py sync --dry-run`` lists EXACTLY the files it would copy to the
eval host — the release's ``cascade/`` package, the bench code
(``benchmarks/``), ``pyproject.toml``, ``uv.lock`` and ``.python-version`` —
and the remote commands it would run (a pinned-key transfer, then
``uv sync --frozen``). Never a secret: no ``.env``, keys, known_hosts,
credentials or symlinks out of the tree; no secret value in the output.
Contract: README.md.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from .helpers import (
    REPO,
    Kit,
    assert_no_secrets,
    assert_remote_pinned,
    assert_safe_text,
    output,
    tree_state,
)

EXPECTED = {
    "cascade/__init__.py",
    "cascade/eval/worker.py",
    "cascade/eval/data/table.json",
    "benchmarks/pyproject.toml",
    "benchmarks/uv.lock",
    "benchmarks/cascade_benchmark/results.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
}
NOT_COPIED = (
    # caches and nested environments
    "cascade/eval/__pycache__/worker.cpython-311.pyc",
    "cascade/eval/stale.pyc",
    "benchmarks/.venv/lib/python3.11/site-packages/x.py",
    "cascade/.git/config",
    # secrets, anywhere in the copied trees
    "cascade/.env",
    "cascade/eval/.env.local",
    "benchmarks/bench.env",
    "cascade/eval/server.pem",
    "cascade/eval/tls.key",
    "cascade/id_ed25519",
    "cascade/id_ed25519.pub",
    "benchmarks/id_rsa",
    "cascade/known_hosts_valeval4",
    "benchmarks/.netrc",
    "cascade/eval/hippius_secret.toml",
    "cascade/eval/Credentials.json",
    # outside the copied set
    ".env",
    "chain.toml",
    "targon-eval.toml",
    "docs/OPERATOR.md",
    "tests/unit/test_a.py",
    "scripts/deploy_gate.py",
    "wallets/coldkey",
)
SECRET_BODY = "HIPPIUS_S3_SECRET_KEY=r12-planted-file-secret-6b19\n"


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _release(kit) -> os.PathLike:
    rel = kit.tmp / "release"
    for p in EXPECTED:
        kit.write(f"release/{p}", f"# {p}\n")
    for p in NOT_COPIED:
        kit.write(f"release/{p}", SECRET_BODY)
    outside = kit.write("outside/.env", SECRET_BODY)
    (rel / "cascade" / "eval" / "linked_env").symlink_to(outside)
    (rel / "benchmarks" / "linked_dir").symlink_to(outside.parent, target_is_directory=True)
    return rel


def _sync(kit, release, *extra):
    kh = kit.write("known_hosts_eval", "")
    proc = kit.tool("sync", "--release-dir", release, "--host", "root@203.0.113.9",
                    "--port", "41100", "--known-hosts", kh, "--dest", "/root/cascade",
                    "--dry-run", *extra)
    return proc, kh


def _copies(stdout: str) -> list[str]:
    return [line[len("COPY "):] for line in stdout.splitlines() if line.startswith("COPY ")]


def _runs(stdout: str) -> list[str]:
    return [line[len("RUN "):] for line in stdout.splitlines() if line.startswith("RUN ")]


def test_sync_dry_run_lists_exactly_the_release_files(kit):
    release = _release(kit)
    proc, _ = _sync(kit, release)
    assert proc.returncode == 0, output(proc)
    copies = _copies(proc.stdout)
    assert copies == sorted(copies), "COPY lines are sorted"
    assert len(copies) == len(set(copies)), "no file listed twice"
    assert set(copies) == EXPECTED, (
        f"missing {sorted(EXPECTED - set(copies))}, extra {sorted(set(copies) - EXPECTED)}")


def test_sync_never_lists_or_prints_a_secret(kit):
    release = _release(kit)
    proc, _ = _sync(kit, release)
    assert proc.returncode == 0, output(proc)
    text = proc.stdout + proc.stderr
    assert "r12-planted-file-secret" not in text, "a secret file's contents reached the output"
    assert_no_secrets(text)
    assert_safe_text(text, what="sync --dry-run")


def test_sync_dry_run_plans_a_pinned_transfer_then_uv_sync_frozen(kit):
    release = _release(kit)
    proc, kh = _sync(kit, release)
    assert proc.returncode == 0, output(proc)
    runs = _runs(proc.stdout)
    transfer = [i for i, r in enumerate(runs) if re.search(r"\b(rsync|scp)\b", r)]
    uv = [i for i, r in enumerate(runs) if "uv sync --frozen" in r]
    assert transfer, f"no rsync/scp transfer planned: {runs}"
    assert uv, f"no `uv sync --frozen` planned on the eval host: {runs}"
    assert min(transfer) < min(uv), "the files are copied before the venv is synced"
    assert all("-p 41100" in r or "--port=41100" in r or "-P 41100" in r
               for r in runs if re.search(r"\b(ssh|rsync|scp)\b", r)), runs
    n = assert_remote_pinned("\n".join(runs), kh, what="sync --dry-run")
    assert n >= 2, runs
    assert not any("uv sync" in r and "--frozen" not in r for r in runs), runs
    assert not any(re.search(r"\buv\s+(lock|add|pip)\b", r) for r in runs), (
        "the pinned lock is never re-resolved on the pod")


def test_sync_dry_run_changes_nothing(kit):
    release = _release(kit)
    before = tree_state(kit.tmp)
    proc1, _ = _sync(kit, release)
    proc2, _ = _sync(kit, release)
    assert proc1.returncode == 0 and proc2.returncode == 0, output(proc1)
    assert proc1.stdout == proc2.stdout, "the dry-run is deterministic"
    after = tree_state(kit.tmp)
    after.pop("known_hosts_eval", None)
    before.pop("known_hosts_eval", None)
    assert after == before
    assert kit.shim_calls() == [], f"--dry-run ran {kit.shim_calls()}"


def test_sync_refuses_a_release_without_the_lock(kit):
    release = _release(kit)
    (release / "uv.lock").unlink()
    proc, _ = _sync(kit, release)
    assert proc.returncode == 2, output(proc)
    assert _copies(proc.stdout) == []
    assert kit.shim_calls() == []


def test_sync_of_this_repo_copies_every_tracked_package_and_bench_file(kit):
    proc, _ = _sync(kit, REPO)
    assert proc.returncode == 0, output(proc)
    copies = set(_copies(proc.stdout))
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "cascade", "benchmarks"],
                             capture_output=True, text=True, check=True).stdout.split()
    want = {p for p in tracked if (REPO / p).is_file() and not (REPO / p).is_symlink()}
    assert want, "the repo has tracked cascade/ files"
    missing = sorted(want - copies)
    assert not missing, f"tracked release files not synced: {missing[:20]}"
    assert {"pyproject.toml", "uv.lock"} <= copies
    # #784: plus exactly the build inputs pyproject.toml names (README, hatch
    # force-includes), derived here independently of the tool.
    build = _repo_build_inputs()
    assert build >= {"README.md", "chain.toml", "chain.testnet.toml", "docs/AGENT_MINING.md"}
    for p in copies:
        assert (p.startswith(("cascade/", "benchmarks/"))
                or p in {"pyproject.toml", "uv.lock", ".python-version"} | build), p
        assert "__pycache__" not in p and not p.endswith((".pyc", ".pyo")), p
        assert "/.venv/" not in p, p
    assert build <= copies, sorted(build - copies)


def _repo_build_inputs() -> set[str]:
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    out = set()
    readme = data["project"].get("readme")
    if isinstance(readme, dict):
        readme = readme.get("file")
    if readme:
        out.add(readme)
    force = (data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
             .get("wheel", {}).get("force-include", {}))
    out.update(force)
    return out
