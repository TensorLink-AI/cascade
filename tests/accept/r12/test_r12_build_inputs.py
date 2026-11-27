"""R12, #784 (testnet E2E 2026-11-23, a fresh 1x RTX 4090 Lium pod): the
release sync must let `uv sync --frozen` build the project on a fresh host,
and the launch eval host must get the pinned torch.

1. The editable build reads files outside `cascade/` and `benchmarks/`:
   `[project] readme` (README.md) and every source of
   `[tool.hatch.build.targets.wheel.force-include]` (chain.toml,
   chain.testnet.toml, docs/AGENT_MINING.md). `release_files()` now ships
   exactly those, DERIVED from the release's own pyproject.toml (not a
   hard-coded list); a release missing one is refused (exit 2, no COPY), since
   its `uv sync --frozen` would fail on the host.
2. torch 2.4.1+cu124 is in the `train` extra. `sync --extra NAME` plans
   `uv sync --frozen --extra NAME` (only a group the release's pyproject
   declares); the launch profile declares `uv_extras = ["train"]` and the
   owner scripts pass it. Pinned dependencies are untouched: still `--frozen`.

Dry-runs only: no ssh, no rsync, no pod. Contract: README.md.
"""

from __future__ import annotations

import re
import tomllib

import pytest

from .helpers import GATE_PROFILE, LAUNCH_PROFILE, NEW_HOST, PORT, REPO, SETUP, Kit, output

PYPROJECT = """\
[project]
name = "cascade"
version = "0"
readme = "README.md"

[project.optional-dependencies]
train = ["torch==2.4.1"]
hub = ["x"]

[tool.hatch.build.targets.wheel.force-include]
"chain.toml" = "cascade/data/chain.toml"
"chain.testnet.toml" = "cascade/data/chain.testnet.toml"
"docs/AGENT_MINING.md" = "cascade/data/AGENT_MINING.md"
"""
BUILD = {"README.md", "chain.toml", "chain.testnet.toml", "docs/AGENT_MINING.md"}


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _release(kit, pyproject: str = PYPROJECT, *, skip: tuple[str, ...] = ()):
    rel = kit.tmp / "release"
    kit.write("release/pyproject.toml", pyproject)
    kit.write("release/uv.lock", "# lock\n")
    kit.write("release/cascade/__init__.py", "")
    for p in BUILD | {"docs/OPERATOR.md", "scripts/deploy_gate.py"}:
        if p not in skip:
            kit.write(f"release/{p}", f"# {p}\n")
    return rel


def _sync(kit, release, *extra):
    kh = kit.write("known_hosts_eval", "")
    return kit.tool("sync", "--release-dir", release, "--host", "root@203.0.113.9",
                    "--port", "41100", "--known-hosts", kh, "--dest", "/root/cascade",
                    "--dry-run", *extra)


def _copies(stdout: str) -> set[str]:
    return {line[len("COPY "):] for line in stdout.splitlines() if line.startswith("COPY ")}


def _runs(stdout: str) -> list[str]:
    return [line[len("RUN "):] for line in stdout.splitlines() if line.startswith("RUN ")]


# ── 1. the build inputs ride along, derived from pyproject.toml ──────────────


def test_sync_ships_the_readme_and_every_force_include(kit):
    proc = _sync(kit, _release(kit))
    assert proc.returncode == 0, output(proc)
    copies = _copies(proc.stdout)
    assert copies >= BUILD, f"missing build inputs {sorted(BUILD - copies)}"
    assert copies == BUILD | {"pyproject.toml", "uv.lock", "cascade/__init__.py"}, (
        sorted(copies))          # docs/OPERATOR.md, scripts/ stay out


def test_the_build_inputs_are_derived_not_hard_coded(kit):
    """Another pyproject naming other files: exactly those are shipped."""
    pyproject = PYPROJECT.replace('readme = "README.md"', 'readme = {file = "docs/OPERATOR.md"}')
    pyproject = pyproject.replace('"docs/AGENT_MINING.md" = "cascade/data/AGENT_MINING.md"\n', "")
    proc = _sync(kit, _release(kit, pyproject))
    assert proc.returncode == 0, output(proc)
    copies = _copies(proc.stdout)
    assert {"docs/OPERATOR.md", "chain.toml", "chain.testnet.toml"} <= copies, sorted(copies)
    assert "README.md" not in copies and "docs/AGENT_MINING.md" not in copies, sorted(copies)


@pytest.mark.parametrize("missing", sorted(BUILD))
def test_a_release_missing_a_build_input_is_refused(kit, missing):
    proc = _sync(kit, _release(kit, skip=(missing,)))
    assert proc.returncode == 2, output(proc)
    assert _copies(proc.stdout) == set(), proc.stdout
    assert missing in proc.stderr, proc.stderr
    assert kit.shim_calls() == []


@pytest.mark.parametrize("bad", ["../outside.md", "/etc/passwd", ".env"])
def test_a_build_input_outside_the_release_or_secret_is_refused(kit, bad):
    kit.write("outside.md", "x\n")
    proc = _sync(kit, _release(kit, PYPROJECT.replace('"README.md"', f'"{bad}"', 1)))
    assert proc.returncode == 2, output(proc)
    assert _copies(proc.stdout) == set(), proc.stdout


def test_this_repo_ships_its_own_build_inputs(kit):
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    want = {data["project"]["readme"]} | set(
        data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"])
    assert want >= BUILD, want
    proc = _sync(kit, REPO)
    assert proc.returncode == 0, output(proc)
    assert want <= _copies(proc.stdout), sorted(want - _copies(proc.stdout))


# ── 2. the role's extra: the launch eval host gets the pinned torch ──────────


def test_sync_extra_plans_a_frozen_sync_with_the_extra(kit):
    proc = _sync(kit, _release(kit), "--extra", "train")
    assert proc.returncode == 0, output(proc)
    runs = _runs(proc.stdout)
    uv = [r for r in runs if "uv sync" in r]
    assert len(uv) == 1, runs
    assert re.search(r"uv sync --frozen --extra train\b", uv[0]), uv
    assert not any(re.search(r"\buv\s+(lock|add|pip)\b", r) for r in runs), runs
    # without --extra the planned sync is today's plain frozen sync
    proc = _sync(kit, _release(kit))
    assert proc.returncode == 0, output(proc)
    uv = [r for r in _runs(proc.stdout) if "uv sync" in r]
    assert len(uv) == 1 and re.search(r"uv sync --frozen'?$", uv[0]), uv


@pytest.mark.parametrize("bad", ["gpu", "train;rm", "--all-extras"])
def test_only_a_declared_extra_is_accepted(kit, bad):
    proc = _sync(kit, _release(kit), "--extra=hub")           # declared: accepted
    assert proc.returncode == 0, output(proc)
    assert any("uv sync --frozen --extra hub" in r for r in _runs(proc.stdout)), proc.stdout
    proc = _sync(kit, _release(kit), f"--extra={bad}")
    assert proc.returncode == 2, output(proc)
    assert _copies(proc.stdout) == set() and _runs(proc.stdout) == [], proc.stdout


EVAL_HOST_EXTRAS = ["train", "hippius", "chain", "pool-forge"]


def test_profiles_name_the_role_extra(kit):
    # 2026-11-26: a train-only eval host had no pyarrow and the lane D public
    # half failed ("mirror read failed (ModuleNotFoundError: ...pyarrow)").
    launch = tomllib.loads(LAUNCH_PROFILE.read_text())
    assert launch.get("uv_extras") == EVAL_HOST_EXTRAS, launch.get("uv_extras")
    proc = kit.tool("profile-info", "--profile", LAUNCH_PROFILE, "--get", "uv_extras")
    assert proc.returncode == 0, output(proc)
    assert proc.stdout.strip() == " ".join(EVAL_HOST_EXTRAS)
    for name in ("elastic_h200.toml", "elastic_rtxpro6000.toml"):
        prof = tomllib.loads((LAUNCH_PROFILE.parent / name).read_text())
        assert prof.get("uv_extras") == EVAL_HOST_EXTRAS, (name, prof.get("uv_extras"))
    # every extra exists, and pyarrow / boto3 ride in them
    groups = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]
    assert set(EVAL_HOST_EXTRAS) <= set(groups), sorted(groups)
    deps = " ".join(d for e in EVAL_HOST_EXTRAS for d in groups[e])
    assert "pyarrow" in deps and "boto3" in deps, deps
    proc = kit.tool("profile-info", "--profile", GATE_PROFILE, "--get", "uv_extras")
    assert proc.returncode == 0, output(proc)
    assert proc.stdout.strip() == ""
    # torch is in that extra in this repo's pyproject (the pin is not touched)
    train = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]["train"]
    assert "torch==2.4.1" in train, train


def test_setup_script_passes_the_launch_extra_to_sync(kit):
    proc = kit.script(SETUP, "--profile", LAUNCH_PROFILE, "--pod", "valpriv-4090",
                      "--host", f"root@{NEW_HOST}", "--port", str(PORT),
                      "--known-hosts", kit.tmp / "known_hosts_eval",
                      "--release-dir", kit.tmp / "release", "--work-dir", kit.tmp / "work",
                      "--dry-run")
    assert proc.returncode == 0, output(proc)
    sync = [line for line in proc.stdout.splitlines() if line.startswith("PLAN venv_sync ")]
    assert sync, proc.stdout
    for extra in EVAL_HOST_EXTRAS:
        assert any(re.search(rf" sync .*--extra '?{extra}'?(\s|$)", s) for s in sync), (extra, sync)
    assert kit.shim_calls() == []
