"""R5 (§5 row 10 S5 order: eval host -> validator -> lane D intake -> trainer;
§3.0 R5 Accept: shellcheck)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from .helpers import LANED, S5_ORDER, SERVICES, build_box, output, plan_lines, script_path


@pytest.fixture
def box(tmp_path):
    return build_box(tmp_path)


def _all_scripts():
    paths = [script_path(f"restart_{svc}") for svc in SERVICES] + [script_path("upgrade_s5")]
    return sorted(set(paths) | set(LANED.glob("*.sh")))


def test_upgrade_s5_dry_run_restarts_in_s5_order(box):
    proc = box.run("upgrade_s5")
    assert proc.returncode == 0, output(proc)
    restarts = plan_lines(proc.stdout, "restart")
    text = proc.stdout
    pos = []
    for svc in S5_ORDER:
        gate_svc = SERVICES[svc][0]
        marker = f"verify --service {gate_svc}"
        assert marker in text, text
        pos.append(text.index(marker))
    assert pos == sorted(pos), "services not planned in the S5 order"
    assert len(restarts) >= 4, "\n".join(restarts)
    assert box.shim_calls() == []
    assert not box.release_dir.exists()


def test_upgrade_s5_stops_at_first_refusal(box):
    box.write_queue({"x": {"status": "in_flight", "phase": "training", "hotkey": "5Busy"}})
    proc = box.run("upgrade_s5")
    text = output(proc)
    assert proc.returncode != 0, text
    assert "REFUSE [in_flight]" in text, text
    assert plan_lines(proc.stdout, "restart") == [], proc.stdout


def test_scripts_parse_with_bash_n():
    for path in _all_scripts():
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{path.name}: {proc.stderr}"


def test_scripts_are_shellcheck_clean():
    scripts = _all_scripts()
    exe = shutil.which("shellcheck")
    if exe is None:
        pytest.skip("shellcheck is not installed on this host; the CI runner runs it")
    proc = subprocess.run([exe, "-x", *map(str, scripts)], capture_output=True, text=True,
                          cwd=str(LANED))
    assert proc.returncode == 0, proc.stdout + proc.stderr
