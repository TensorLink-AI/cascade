"""R12: the owner-paste setup and recovery scripts (plan §10.4 "Pre-staged,
reviewed owner-paste scripts cover launch-eval-host (and, from S8,
valeval-laned) setup and recovery, because remote writes are often
classifier-blocked"; §3.0 R12 Accept "shellcheck + dry-run").

Both scripts are printable (``--print`` emits a self-contained bash block
for the owner to paste), idempotent (they pin keys through ``pin-host-key``,
never by appending; the same inputs print the same text), print no secrets,
and ``--dry-run`` runs nothing. The tests never run them without
``--dry-run``/``--print``. Contract: README.md.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from .helpers import (
    KIT,
    NEW_HOST,
    OLD_HOST,
    PORT,
    RECOVER,
    SETUP,
    Kit,
    assert_remote_pinned,
    assert_safe_text,
    output,
    require,
    tree_state,
)

SETUP_STEPS = ("known_hosts", "on_rent", "venv_sync", "handshake", "evidence")
RECOVER_STEPS = ("resolve_endpoint", "known_hosts", "on_rent", "handshake")
DESTRUCTIVE = (
    re.compile(r"\blium\s+(rm|down|stop|terminate|delete|kill)\b"),
    re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r"),
    re.compile(r"\bshred\b"),
    re.compile(r"\bmkfs\b"),
)


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _setup_args(kit):
    return ["--profile", KIT / "profiles" / "bench_gate.toml", "--pod", "valeval-blackwell4",
            "--host", f"root@{NEW_HOST}", "--port", str(PORT),
            "--known-hosts", kit.tmp / "known_hosts_eval",
            "--release-dir", kit.tmp / "release", "--work-dir", kit.tmp / "work"]


def _recover_args(kit):
    return ["--profile", KIT / "profiles" / "bench_gate.toml", "--pod", "valeval-blackwell4",
            "--configured", f"{OLD_HOST}:{PORT}", "--user", "root",
            "--known-hosts", kit.tmp / "known_hosts_eval",
            "--release-dir", kit.tmp / "release", "--work-dir", kit.tmp / "work"]


CASES = {"setup": (SETUP, _setup_args, SETUP_STEPS),
         "recover": (RECOVER, _recover_args, RECOVER_STEPS)}


def _plan_steps(stdout: str) -> list[str]:
    steps = []
    for line in stdout.splitlines():
        m = re.match(r"PLAN (\w+) ", line)
        if m and (not steps or steps[-1] != m.group(1)):
            steps.append(m.group(1))
    return steps


def _print_steps(text: str) -> list[str]:
    return re.findall(r"^# STEP (\w+)\s*$", text, flags=re.M)


def _first_order(seq: list[str], want: tuple[str, ...]) -> list[str]:
    seen = []
    for s in seq:
        if s in want and s not in seen:
            seen.append(s)
    return seen


# ── static checks ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", sorted(CASES))
def test_script_header_and_bash_n(which):
    path = CASES[which][0]
    require(path)
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash\n"), f"{path.name}: want a bash shebang"
    assert re.search(r"^set -euo pipefail$", text, flags=re.M), f"{path.name}: set -euo pipefail"
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert_safe_text(code, what=path.name)


def test_all_kit_shell_scripts_are_shellcheck_clean():
    require(SETUP)
    require(RECOVER)
    scripts = sorted(KIT.rglob("*.sh"))
    exe = shutil.which("shellcheck")
    if exe is None:
        pytest.skip("shellcheck is not installed on this host; the CI runner (ubuntu-latest, "
                    "shellcheck preinstalled) runs it")
    proc = subprocess.run([exe, "-x", *map(str, scripts)], capture_output=True, text=True,
                          cwd=str(KIT))
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ── CLI ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", sorted(CASES))
def test_help_exits_zero_and_names_dry_run(kit, which):
    proc = kit.script(CASES[which][0], "--help")
    assert proc.returncode == 0, output(proc)
    assert "--dry-run" in proc.stdout and "--print" in proc.stdout, proc.stdout
    assert kit.shim_calls() == []


@pytest.mark.parametrize("which", sorted(CASES))
def test_missing_required_flag_is_refused(kit, which):
    path, args, _ = CASES[which]
    full = list(map(str, args(kit)))
    i = full.index("--known-hosts")
    proc = kit.script(path, "--dry-run", *(full[:i] + full[i + 2:]))
    assert proc.returncode != 0, output(proc)
    assert kit.shim_calls() == []


@pytest.mark.parametrize("which", sorted(CASES))
def test_dry_run_plans_the_steps_in_order_and_changes_nothing(kit, which):
    path, args, steps = CASES[which]
    before = tree_state(kit.tmp)
    proc = kit.script(path, "--dry-run", *args(kit))
    assert proc.returncode == 0, output(proc)
    got = _first_order(_plan_steps(proc.stdout), steps)
    assert got == list(steps), f"want PLAN steps {steps} in order, got {got}\n{proc.stdout}"
    assert kit.shim_calls() == [], f"--dry-run ran {kit.shim_calls()}"
    assert tree_state(kit.tmp) == before, "--dry-run created or changed files"
    again = kit.script(path, "--dry-run", *args(kit))
    assert again.stdout == proc.stdout, "the dry-run plan is deterministic"


@pytest.mark.parametrize("which", sorted(CASES))
def test_print_emits_a_pasteable_idempotent_block(kit, which):
    path, args, steps = CASES[which]
    before = tree_state(kit.tmp)
    proc = kit.script(path, "--print", *args(kit))
    assert proc.returncode == 0, output(proc)
    text = proc.stdout
    assert kit.shim_calls() == [], f"--print ran {kit.shim_calls()}"
    assert tree_state(kit.tmp) == before, "--print created or changed files"
    assert _first_order(_print_steps(text), steps) == list(steps), text
    pasted = kit.tmp / f"{which}_pasted.sh"
    pasted.write_text(text)
    n = subprocess.run(["bash", "-n", str(pasted)], capture_output=True, text=True)
    assert n.returncode == 0, f"the printed block is not valid bash: {n.stderr}"
    assert re.search(r"^set -euo pipefail$", text, flags=re.M), "the pasted block stops on error"
    again = kit.script(path, "--print", *args(kit))
    assert again.stdout == text, "printing twice gives the same block"
    kh = str(kit.tmp / "known_hosts_eval")
    assert not re.search(r">>\s*['\"]?" + re.escape(kh), text), (
        "known_hosts is pinned through `evalhost.py pin-host-key`, never appended to directly")
    for line in text.splitlines():
        if "evalhost.py" in line and not line.lstrip().startswith("#"):
            assert " -P " in line, f"the kit runs with python -P (cwd shadowing): {line}"


@pytest.mark.parametrize("which", sorted(CASES))
@pytest.mark.parametrize("mode", ["--dry-run", "--print"])
def test_output_pins_host_keys_and_holds_no_secret(kit, which, mode):
    path, args, _ = CASES[which]
    proc = kit.script(path, mode, *args(kit))
    assert proc.returncode == 0, output(proc)
    text = proc.stdout + proc.stderr
    assert_safe_text(text, what=f"{path.name} {mode}")
    n = assert_remote_pinned(proc.stdout, kit.tmp / "known_hosts_eval", what=f"{path.name} {mode}")
    assert n >= 1, f"{path.name} {mode} plans no pinned remote command"
    for rx in DESTRUCTIVE:
        m = rx.search(proc.stdout)
        assert m is None, f"{path.name} {mode}: destructive command {m.group(0)!r}"


def test_setup_plan_uses_the_kit_subcommands(kit):
    proc = kit.script(SETUP, "--dry-run", *_setup_args(kit))
    assert proc.returncode == 0, output(proc)
    out = proc.stdout
    for sub in ("pin-host-key", "on-rent-check", "sync", "handshake", "evidence"):
        assert f"evalhost.py {sub}" in out, f"setup plan never runs `evalhost.py {sub}`"
    assert f"ssh-keyscan -p {PORT} {NEW_HOST}" in out, out
    assert "nvidia-smi -L" in out, out
    assert "evidence/r12_pod.json" in out, out


def test_recover_resolves_through_lium_ps_before_anything_else(kit):
    proc = kit.script(RECOVER, "--dry-run", *_recover_args(kit))
    assert proc.returncode == 0, output(proc)
    plan = [line for line in proc.stdout.splitlines() if line.startswith("PLAN ")]
    assert plan and plan[0].startswith("PLAN resolve_endpoint "), plan[:3]
    first = "\n".join(line for line in plan if line.startswith("PLAN resolve_endpoint "))
    assert "lium ps" in first and "evalhost.py resolve-endpoint" in first, first
    assert "df " in proc.stdout, "recovery re-checks the pod's free disk (two-checkpoint headroom)"
