"""R5 (plan §3.0 R5 row, §10.5 S5, §10.8): `--dry-run` prints the exact ordered
plan (worktree, deploy_gate check, §10.8 snapshots, the restart command,
deploy_gate verify) and changes nothing."""

from __future__ import annotations

import pytest

from .helpers import (
    ABSENT_STATE,
    BOX_SERVICES,
    PRESENT_STATE,
    QUIET_TS,
    SERVICES,
    build_box,
    output,
    plan_lines,
    script_path,
)

ALL = tuple(SERVICES)
STEP_ORDER = ("worktree", "check", "snapshot", "restart", "verify")


@pytest.fixture
def box(tmp_path):
    return build_box(tmp_path)


@pytest.mark.parametrize("svc", ALL)
def test_dry_run_is_go_in_a_quiet_window(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    assert "REFUSE" not in output(proc), output(proc)


@pytest.mark.parametrize("svc", ALL)
def test_plan_steps_in_order(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    lines = plan_lines(proc.stdout)
    steps = [ln.split()[1] for ln in lines]
    for step in STEP_ORDER:
        assert step in steps, f"no `PLAN {step}` line:\n{proc.stdout}"
    firsts = [steps.index(s) for s in STEP_ORDER]
    lasts = [len(steps) - 1 - steps[::-1].index(s) for s in STEP_ORDER]
    for i in range(len(STEP_ORDER) - 1):
        assert lasts[i] < firsts[i + 1], (
            f"every `PLAN {STEP_ORDER[i]}` must precede every `PLAN {STEP_ORDER[i + 1]}`:\n"
            + proc.stdout)


@pytest.mark.parametrize("svc", ALL)
def test_dry_run_changes_nothing(box, svc):
    before_files = box.fingerprint()
    before_wt = box.worktrees()
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    assert box.fingerprint() == before_files, "a dry-run modified the fixture tree"
    assert box.worktrees() == before_wt, "a dry-run added or removed a git worktree"
    assert not box.release_dir.exists(), "a dry-run created the release worktree"
    assert not (box.root / "_state_backups").exists(), "a dry-run wrote snapshots"
    assert box.shim_calls() == [], f"a dry-run called box-changing commands: {box.shim_calls()}"


@pytest.mark.parametrize("svc", ALL)
def test_dry_run_is_deterministic(box, svc):
    first = box.run(svc)
    second = box.run(svc)
    assert first.returncode == 0, output(first)
    assert plan_lines(first.stdout), first.stdout
    assert plan_lines(first.stdout) == plan_lines(second.stdout)


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_lists_every_state_file(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    snaps = plan_lines(proc.stdout, "snapshot")
    backup = box.root / "_state_backups" / QUIET_TS
    for name in PRESENT_STATE:
        src, dst = str(box.root / name), str(backup / name)
        assert any(src in ln and dst in ln for ln in snaps), (
            f"no snapshot line copying {src} -> {dst}:\n" + "\n".join(snaps))
    for name in ABSENT_STATE:
        assert any(name in ln and "absent" in ln for ln in snaps), (
            f"no snapshot line reporting {name} absent:\n" + "\n".join(snaps))


@pytest.mark.parametrize("svc", ALL)
def test_check_and_verify_lines_use_release_deploy_gate(box, svc):
    gate_svc = SERVICES[svc][0]
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    gate = f"{box.release_dir}/scripts/deploy_gate.py"
    checks = plan_lines(proc.stdout, "check")
    assert any("python -P" in ln and f"{gate} check --service {gate_svc}" in ln
               and f"--from {box.previous_ref}" in ln and f"--to {box.release_ref}" in ln
               for ln in checks), "\n".join(checks) or proc.stdout
    verifies = plan_lines(proc.stdout, "verify")
    assert any("python -P" in ln and f"{gate} verify --service {gate_svc}" in ln
               and f"--release {box.release_dir}" in ln
               for ln in verifies), "\n".join(verifies) or proc.stdout


@pytest.mark.parametrize("svc", BOX_SERVICES)
def test_box_restart_command_is_pinned_release_with_python_P(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    restart = "\n".join(plan_lines(proc.stdout, "restart"))
    assert restart, proc.stdout
    assert f"PYTHONPATH={box.release_dir}" in restart, restart
    assert f"{box.venv}/bin/python -P" in restart, restart
    assert any(n in restart for n in SERVICES[svc][1]), restart
    assert "choom -n -500" in restart or "OOMScoreAdjust=-500" in restart, restart
    if svc == "intake_laned":
        assert "systemd" in restart, restart
    else:
        assert "systemd-run --scope" in restart, restart
    for ln in restart.splitlines():
        if "python" in ln and " -m " in ln:
            assert " -P" in ln, f"`python -m` without -P (cwd shadowing): {ln}"


def test_eval_host_restart_pins_known_hosts_and_frozen_sync(box):
    proc = box.run("eval_host")
    assert proc.returncode == 0, output(proc)
    text = "\n".join(plan_lines(proc.stdout))
    ssh_lines = [ln for ln in text.splitlines() if "ssh " in ln]
    assert ssh_lines, f"no ssh in the eval host plan:\n{text}"
    for ln in ssh_lines:
        assert f"UserKnownHostsFile={box.known_hosts}" in ln, ln
        assert "StrictHostKeyChecking=yes" in ln, ln
        assert f"-p {box.port}" in ln, ln
        assert box.host in ln, ln
    assert "uv sync --frozen" in text, text
    assert "StrictHostKeyChecking=no" not in proc.stdout
    assert "UserKnownHostsFile=/dev/null" not in proc.stdout


def test_eval_host_restart_sync_keeps_the_profile_extras(box):
    """#892 class: a bare ``uv sync --frozen`` makes the venv exact to the
    lock's base set and UNINSTALLS the host profile's extras (launch_eval
    installs ``--extra train``, so torch would vanish). The restart sync must
    keep what the profile installed."""
    proc = box.run("eval_host")
    assert proc.returncode == 0, output(proc)
    syncs = [ln for ln in plan_lines(proc.stdout) if "uv sync --frozen" in ln]
    assert syncs, f"no uv sync in the eval host plan:\n{proc.stdout}"
    for ln in syncs:
        assert "--inexact" in ln or "--extra" in ln or "--all-extras" in ln, (
            f"bare `uv sync --frozen` on the eval host uninstalls its extras: {ln}")


@pytest.mark.parametrize("svc", ALL)
def test_scripts_never_touch_keys(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    sources = [p.read_text() for p in script_path(f"restart_{svc}").parent.glob("*.sh")]
    for banned in ("coldkey", "BT_WALLET_PASSWORD", ".bittensor/wallets", "cat .env"):
        assert banned not in proc.stdout, f"plan names {banned!r}"
        assert all(banned not in s for s in sources), f"a deploy/laned script names {banned!r}"
