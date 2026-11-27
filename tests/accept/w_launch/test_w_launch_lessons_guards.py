"""W-LAUNCH lessons (owner 2026-11-27): the launch's failure classes as CI guards.

docs/launch-lessons-2026-11.md lists every failure class of the lane B/C/D
launch (2026-11-24 → 27) with its fix PR and the guard that now catches it.
These tests pin the guards that are new in that PR; each fails on a base
without it by assertion (rule absent, attribute unset, script or doc missing).
All torch-free: AST/regex checks, git in a temporary repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "launch-lessons-2026-11.md"


def _load(rel: str, name: str):
    path = ROOT / rel
    assert path.is_file(), f"missing {rel}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 1. IR008: a remote python request without an on-host bound (#969) ─────────

PRE_969 = '''\
class SSHEvalHost:
    def _ssh(self, host, args, payload):
        cmd = (f"cd {shlex.quote(host.workdir)} && "
               + " ".join(shlex.quote(a) for a in
                          [host.remote_python, "-m", "cascade.eval.lane_d_forecasts", *args]))
        proc = self._run(build_ssh_argv(host, cmd), input=payload, timeout=self.timeout_s)
'''


def _ir008():
    cir = _load("scripts/check_incident_rules.py", "check_incident_rules_wl")
    rules = {r.id: r for r in cir.RULES}
    assert "IR008" in rules, "no incident rule for remote commands without an on-host timeout"
    return cir, rules["IR008"]


def test_ir008_flags_the_lane_d_orphan_shape(tmp_path):
    cir, rule = _ir008()
    p = tmp_path / "cascade" / "validator" / "lane_d_inputs.py"
    p.parent.mkdir(parents=True)
    p.write_text(PRE_969)
    got = [(h.rule, h.line) for h in cir.check(tmp_path, ["cascade/validator/lane_d_inputs.py"],
                                               (rule,))]
    assert got == [("IR008", 2)]


def test_ir008_is_clean_on_the_repository():
    cir, rule = _ir008()
    found = cir.check(ROOT, None, (rule,))
    assert not found, "\n".join(map(str, found))
    # the #969 fix itself is recognised as bounded, not merely reviewed
    assert "cascade/validator/lane_d_inputs.py::remote_command" not in cir.IR008_REVIEWED


# ── 2. PATHS merges are union merges (#964/#966 PATHS conflicts) ───────────────

def test_paths_files_merge_by_union():
    got = subprocess.run(["git", "-C", str(ROOT), "check-attr", "merge", "--",
                          "tests/accept/wb27/PATHS"], capture_output=True, text=True).stdout
    assert got.strip().endswith("merge: union"), f"PATHS has no union merge driver: {got!r}"


def test_two_prs_appending_to_one_paths_merge_cleanly(tmp_path):
    ga = ROOT / ".gitattributes"
    assert ga.is_file(), "no .gitattributes"

    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True, text=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "ci")
    paths = tmp_path / "tests" / "accept" / "wx" / "PATHS"
    paths.parent.mkdir(parents=True)
    (tmp_path / ".gitattributes").write_text(ga.read_text())
    paths.write_text("# WX\ncascade/a.py\n")
    git("add", ".gitattributes", "tests/accept/wx/PATHS")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "pr1")
    paths.write_text("# WX\ncascade/a.py\n# pr1\ncascade/b.py\n")
    git("commit", "-q", "-am", "pr1")
    git("checkout", "-q", "main")
    paths.write_text("# WX\ncascade/a.py\n# pr2\ncascade/c.py\n")
    git("commit", "-q", "-am", "pr2")
    git("checkout", "-q", "pr1")
    merged = git("merge", "-q", "--no-edit", "main")
    assert merged.returncode == 0, f"PATHS conflict: {merged.stdout}{merged.stderr}"
    text = paths.read_text()
    assert "cascade/b.py" in text and "cascade/c.py" in text


# ── 3. image freshness: worker files changed since the pinned build (#960) ─────

def test_image_freshness_lists_worker_files_changed_since_the_build(tmp_path, capsys):
    fresh = _load("scripts/check_image_freshness.py", "check_image_freshness_wl")

    def git(*a):
        out = subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True, text=True)
        return out.stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "ci")
    worker = tmp_path / "cascade" / "trainer" / "pyramid_trainer.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("world = 1\n")
    (tmp_path / "cascade" / "trainer" / "main.py").write_text("x = 1\n")
    git("add", "cascade")
    git("commit", "-q", "-m", "image build")
    build = git("rev-parse", "--short=7", "HEAD")
    digest = "sha256:" + "5d" * 32
    (tmp_path / "chain.toml").write_text(
        f'[round]\nfunded_pod_image = "ghcr.io/o/cascade-worker:worker-rc15@{digest}"'
        f"   # rc15 = main {build}\n"
        f'[v2.training]\ntrain_image_digest = "{digest}"\n')
    assert fresh.main(["--root", str(tmp_path)]) == 0
    assert "no worker-side change" in capsys.readouterr().out
    worker.write_text("world = gang_size()\n")            # the #748 fix lands after the build
    (tmp_path / "cascade" / "trainer" / "main.py").write_text("x = 2\n")   # box-side only
    git("commit", "-q", "-am", "fix(lane-b): expert legs train on every GPU")
    assert fresh.main(["--root", str(tmp_path)]) == 0                       # warn-only
    out = capsys.readouterr().out
    assert "cascade/trainer/pyramid_trainer.py" in out
    assert "cascade/trainer/main.py" not in out
    assert "[round] funded_pod_image, [v2.training] train_image_digest" in out
    assert fresh.main(["--root", str(tmp_path), "--strict"]) == 1


def test_image_freshness_runs_in_ci_with_full_history():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "scripts/check_image_freshness.py" in ci
    lint = ci.split("\n  lint:", 1)[1].split("\n  unit:", 1)[0]
    assert "fetch-depth: 0" in lint, "the lint job checks out shallow: freshness cannot diff"


# ── 4. the lessons doc: one row per failure class, with its fix and its guard ──

FIX_PRS = ("#951", "#952", "#955", "#957", "#958", "#959", "#960", "#963", "#964",
           "#965", "#966", "#967", "#968", "#969")


def test_lessons_doc_covers_every_failure_class():
    assert DOC.is_file(), "missing docs/launch-lessons-2026-11.md"
    text = DOC.read_text()
    for pr in FIX_PRS:
        assert pr in text, f"lessons doc has no row for {pr}"
    for guard in ("IR008", "test_box_side_torch_free", "test_runner_cfg_consumers_follow_overlay",
                  "test_stage_pods_gate_on_their_own_pins", "check_image_freshness",
                  "merge=union", "IR007", "IR003"):
        assert guard in text, f"lessons doc does not name the guard {guard}"
    assert "## Pre-upgrade checklist" in text
    assert "launch-lessons-2026-11.md" in (ROOT / "docs" / "OPERATOR.md").read_text(), \
        "docs/OPERATOR.md does not link the lessons doc"
