"""R8: `deploy_gate` for the lane D services (plan §3.0 R8 row, §3 "Blast radius
by path", §10.5 S5, §10.8; CI-checkable parts of T19 and T28).

The gate is loaded from `scripts/deploy_gate.py` by file path and its CLI is
driven as a subprocess against throwaway git repos in `tmp_path`. Nothing
here reads the production box: `--root` and `--repo` always point at
`tmp_path`.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE_PY = REPO / "scripts" / "deploy_gate.py"
WORKER_PY = REPO / "cascade" / "trainer" / "worker.py"
NEW_SERVICES = ("intake-laned", "eval-host")
OLD_SERVICES = ("trainer", "validator")

# The explicit docs/tests allow-list for the every-path-classifies rule. It
# lives here, read-only to the builder, so it can't be widened to hide a code
# path. Everything else that git tracks must land in at least one AREA.
DOCS_TESTS_ALLOWED = (
    r"^docs/",
    r"^decisions/",
    r"(^|/)[^/]+\.md$",
    r"(^|/)(LICENSE|NOTICE)$",
    r"^llms\.txt$",
    r"^tests/",
    r"^[^/]+/tests/",
    r"(^|/)\.gitignore$",
)

NOW = datetime(2026, 11, 16, 3, 0, tzinfo=UTC)          # a Monday, outside the freeze


def _vline(ts: str, text: str) -> str:
    return f"2026-11-16 {ts},000 INFO cascade.validator: {text}"


# 03:00 with block 9516100 sampled at 02:59: the next boundary is ~1 h 40 min away.
CLOCK = [_vline("02:59:00", "re-asserting weights (last set 3 blocks ago, block 9516100)")]
SCORING = CLOCK + [_vline("02:59:30", "new manifest round=42 entries=2 (king:uid1)")]


def _load_gate():
    spec = importlib.util.spec_from_file_location("r8_deploy_gate", GATE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


g = _load_gate()


def _gate(tmp_path, *, service, paths=(), vlines=CLOCK, diff=""):
    return g.gate(service=service, areas=g.classify(list(paths)), diff_text=diff, now=NOW,
                  trainer_lines=[], validator_lines=list(vlines), work=tmp_path,
                  on_main=True, acks=set())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=r8", "-c", "user.email=r8@example.invalid",
         "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        capture_output=True, text=True, check=True, timeout=60).stdout.strip()


def _two_commit_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base")
    rev_from = _git(repo, "rev-parse", "HEAD")
    (repo / "cascade" / "lane_d").mkdir(parents=True)
    (repo / "cascade" / "lane_d" / "intake.py").write_text("X = 1\n")
    _git(repo, "add", "cascade/lane_d/intake.py")
    _git(repo, "commit", "-q", "-m", "intake")
    rev_to = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", rev_to)
    return repo, rev_from, rev_to


def _run_gate(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-P", str(GATE_PY), *argv], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)


# ── (R8 head) --service intake-laned|eval-host on check and verify ───────────


@pytest.mark.parametrize("service", NEW_SERVICES + OLD_SERVICES)
def test_check_accepts_service(tmp_path, service):
    repo, rev_from, rev_to = _two_commit_repo(tmp_path)
    root = tmp_path / "box"
    root.mkdir()
    r = _run_gate(tmp_path, "check", "--service", service, "--from", rev_from,
                  "--to", rev_to, "--repo", str(repo), "--root", str(root))
    assert r.returncode in (0, 1), (
        f"`check --service {service}` must be accepted (exit 0 GO / 1 REFUSE), "
        f"got {r.returncode}: {r.stderr.strip()[-400:]}")
    assert "Traceback" not in r.stderr, r.stderr[-800:]
    lines = r.stdout.strip().splitlines()
    assert lines and lines[0].startswith(f"{service} "), r.stdout
    assert lines[-1] in ("GO", "REFUSE"), r.stdout
    # the empty box root has no chain clock, so the gate must refuse on it
    assert "REFUSE [clock]" in r.stdout, r.stdout


@pytest.mark.parametrize("service", NEW_SERVICES)
def test_check_reports_lane_d_intake_area_for_new_services(tmp_path, service):
    repo, rev_from, rev_to = _two_commit_repo(tmp_path)
    root = tmp_path / "box"
    root.mkdir()
    r = _run_gate(tmp_path, "check", "--service", service, "--from", rev_from,
                  "--to", rev_to, "--repo", str(repo), "--root", str(root))
    assert r.returncode in (0, 1), r.stderr[-400:]
    assert "lane_d_intake(1)" in r.stdout.splitlines()[0], r.stdout


@pytest.mark.parametrize("service", NEW_SERVICES + OLD_SERVICES)
def test_verify_accepts_service(tmp_path, service):
    release = tmp_path / "release"
    release.mkdir()
    root = tmp_path / "box"
    root.mkdir()
    r = _run_gate(tmp_path, "verify", "--service", service, "--release", str(release),
                  "--root", str(root))
    assert r.returncode != 2, (
        f"`verify --service {service}` must be accepted, got a usage error: "
        f"{r.stderr.strip()[-400:]}")
    assert "Traceback" not in r.stderr, r.stderr[-800:]
    # no such process runs from tmp_path, so verify must report a failure, not crash
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr[-400:])
    assert "VERIFY FAIL" in r.stdout, r.stdout


# ── service-specific guards (§10.5 S5: outside scoring windows; T19/T28) ─────


@pytest.mark.parametrize("service", NEW_SERVICES)
def test_new_service_refused_while_validator_scoring(tmp_path, service):
    v = _gate(tmp_path, service=service, vlines=SCORING)
    assert "scoring" in [r for r, _ in v.refuse], v.refuse
    done = SCORING + [_vline("02:59:50", "round=42 weights set under the X policy {}: {}")]
    assert not _gate(tmp_path, service=service, vlines=done).refuse


@pytest.mark.parametrize("service", NEW_SERVICES)
def test_new_service_keeps_settlement_and_clock_guards(tmp_path, service):
    near = [_vline("02:59:00", "x block 9516510")]          # ~18 min before 9516600
    assert [r for r, _ in _gate(tmp_path, service=service, vlines=near).refuse] == [
        "settlement"]
    assert [r for r, _ in _gate(tmp_path, service=service, vlines=()).refuse] == ["clock"]


@pytest.mark.parametrize("service", OLD_SERVICES)
def test_lane_d_intake_diff_names_the_intake_service(tmp_path, service):
    v = _gate(tmp_path, service=service, paths=["cascade/lane_d/intake.py"])
    assert any("intake-laned" in n for n in v.notes), v.notes


def test_eval_worker_diff_names_the_eval_host(tmp_path):
    # T19/T28: the eval host runs cascade/validator/private_worker.py and must be
    # updated with every validator deploy that changes it (eval host first).
    v = _gate(tmp_path, service="validator", paths=["cascade/validator/private_worker.py"])
    assert any("eval-host" in n for n in v.notes), v.notes


# ── (i) the lane_d_intake AREA ───────────────────────────────────────────────


@pytest.mark.parametrize("path", ["cascade/lane_d/__init__.py", "cascade/lane_d/intake.py",
                                  "cascade/lane_d/api/routes.py"])
def test_lane_d_intake_area_matches_cascade_lane_d(path):
    assert "lane_d_intake" in g.AREAS
    assert path in g.classify([path]).get("lane_d_intake", []), g.classify([path])


@pytest.mark.parametrize("path", ["cascade/validator/lane_d.py",
                                  "cascade/eval/lane_d_scorer.py",
                                  "cascade/shared/config/lane_d.py",
                                  "cascade/lane_d_other.py"])
def test_lane_d_intake_area_is_only_the_intake_package(path):
    assert "lane_d_intake" not in g.classify([path])


def test_rollout_tooling_package_classifies_as_ops_tooling():
    # Decision (R8): cascade/laned/ (R1/R6/R2 rollout tools) is AREA ops_tooling;
    # it isn't consensus and isn't the intake service.
    a = g.classify(["cascade/laned/release.py"])
    assert "cascade/laned/release.py" in a.get("ops_tooling", []), a
    assert "consensus" not in a and "lane_d_intake" not in a


# ── (ii) every tracked path classifies, or is on the docs/tests allow-list ───


def test_every_tracked_path_classifies_or_is_allowlisted():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True,
                         check=True, timeout=60).stdout.decode()
    paths = [p for p in out.split("\0") if p]
    assert len(paths) > 100, "git ls-files returned too little; not the repo root"
    unclassified = [p for p in paths
                    if not g.classify([p])
                    and not any(re.search(x, p) for x in DOCS_TESTS_ALLOWED)]
    assert not unclassified, (
        f"{len(unclassified)} tracked path(s) match no deploy_gate AREA and are not "
        f"docs/tests: {unclassified[:25]}")


def test_allowlist_does_not_hide_code_paths():
    # sanity on this test's own allow-list: code directories stay out of it
    for p in ("cascade/miner/cli.py", "scripts/deploy_gate.py", "deploy/provision.py",
              "benchmarks/Dockerfile", "chain.toml", "uv.lock"):
        assert not any(re.search(x, p) for x in DOCS_TESTS_ALLOWED), p


# ── (iii) WORKER_IMAGE covers the on-pod import closure of worker.py ─────────


def _module_file(mod: str) -> Path | None:
    p = REPO.joinpath(*mod.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    return None


def _module_name(f: Path) -> str:
    parts = list(f.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _imports(tree: ast.AST):
    """Every Import/ImportFrom anywhere (lazy function-level ones included),
    skipping `if TYPE_CHECKING:` bodies."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def worker_import_closure() -> set[str]:
    """Repo-relative files reachable from cascade/trainer/worker.py by static
    import (absolute and relative, module-level and lazy). This over-approximates
    what `worker.py --role expert` runs on the pod, which is the safe direction."""
    seen: set[Path] = set()
    todo = [WORKER_PY]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        name = _module_name(f)
        pkg = name if f.name == "__init__.py" else name.rpartition(".")[0]
        for node in _imports(ast.parse(f.read_text(), filename=str(f))):
            if isinstance(node, ast.Import):
                cands = [a.name for a in node.names]
            else:
                if node.level:
                    base = pkg.split(".")
                    base = base[:len(base) - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                cands = [mod] + [f"{mod}.{a.name}" for a in node.names]
            for c in cands:
                if c != "cascade" and not c.startswith("cascade."):
                    continue
                parts = c.split(".")
                for i in range(1, len(parts) + 1):
                    mf = _module_file(".".join(parts[:i]))
                    if mf is not None:
                        todo.append(mf)
    return {str(p.relative_to(REPO)) for p in seen}


def test_worker_closure_reaches_the_expert_training_module():
    closure = worker_import_closure()
    assert "cascade/trainer/worker.py" in closure
    assert "cascade/trainer/expert_training.py" in closure, sorted(closure)[:40]


def test_worker_image_covers_the_worker_import_closure():
    closure = worker_import_closure()
    missing = sorted(p for p in closure if not any(re.search(x, p) for x in g.WORKER_IMAGE))
    assert not missing, (
        f"{len(missing)} of {len(closure)} on-pod modules reachable from worker.py are not "
        f"in WORKER_IMAGE: {missing[:25]}")
    assert "worker_image" in g.classify(["cascade/trainer/expert_training.py"])


@pytest.mark.parametrize("path", ["deploy/Dockerfile", "cascade/trainer/worker.py",
                                  "cascade/interface/generator.py", "uv.lock",
                                  "pyproject.toml"])
def test_worker_image_keeps_its_existing_paths(path):
    assert "worker_image" in g.classify([path])
