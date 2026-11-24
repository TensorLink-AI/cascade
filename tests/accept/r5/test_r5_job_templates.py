"""R5 (lane D ops tooling): the systemd TEMPLATES for the lane D recurring jobs
in ``deploy/laned/systemd/`` and their runbook section in docs/OPERATOR.md.

Templates only: nothing in the repo installs or starts them (no script names
them), every mutating job ships in ``--dry-run``, every unit runs the release
with ``python -P`` under the code-path guard's PYTHONPATH, every ``${VAR}`` is
named in the env example (names and paths, no secrets), and every ExecStart
parses with the real command's own argument parser.
"""

from __future__ import annotations

import configparser
import importlib
import re
import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SYSTEMD = REPO / "deploy" / "laned" / "systemd"
ENV_EXAMPLE = SYSTEMD / "cascade-laned-jobs.env.example"
UNITS = ("cascade-laned-house-pointer", "cascade-laned-published-weights",
         "cascade-laned-readiness-measure", "cascade-laned-evidence",
         "cascade-laned-vault-sweep", "cascade-laned-king-check")
# job unit -> it may write (signs, installs, deletes): ships --dry-run
MUTATING = ("cascade-laned-house-pointer", "cascade-laned-published-weights",
            "cascade-laned-evidence", "cascade-laned-vault-sweep")
PYTHON = "/root/cascade/.venv/bin/python"


def _unit(name: str, suffix: str) -> configparser.ConfigParser:
    path = SYSTEMD / f"{name}{suffix}"
    assert path.is_file(), f"missing template {path.relative_to(REPO)}"
    cp = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    cp.optionxform = str
    cp.read_string(path.read_text(encoding="utf-8"))
    return cp


def _env() -> dict[str, str]:
    assert ENV_EXAMPLE.is_file(), "missing deploy/laned/systemd/cascade-laned-jobs.env.example"
    out = {}
    for ln in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.lstrip().startswith("#"):
            k, _, v = ln.partition("=")
            out[k.strip()] = v.strip() or f"/fixture/{k.strip().lower()}"
    return out


def _commands(cp) -> list[list[str]]:
    env = _env()
    cmds = []
    for key in ("ExecStartPre", "ExecStart"):
        if cp.has_option("Service", key):
            raw = cp.get("Service", key)
            used = set(re.findall(r"\$\{([A-Z0-9_]+)\}", raw))
            assert used <= set(env), f"{key} uses {sorted(used - set(env))} not in the env example"
            cmds.append(shlex.split(re.sub(r"\$\{([A-Z0-9_]+)\}", lambda m: env[m.group(1)], raw)))
    assert cmds, "no ExecStart"
    return cmds


@pytest.mark.parametrize("name", UNITS)
def test_unit_runs_the_release_under_the_code_path_guard(name):
    cp = _unit(name, ".service")
    assert cp.get("Service", "Type") == "oneshot"
    text = (SYSTEMD / f"{name}.service").read_text(encoding="utf-8")
    assert "Environment=PYTHONPATH=/root/cascade" in text
    assert "EnvironmentFile=/etc/cascade-laned-jobs.env" in text
    assert not cp.has_section("Install"), "a job service is started by its timer only"
    for argv in _commands(cp):
        assert argv[:2] == [PYTHON, "-P"], argv[:2]


@pytest.mark.parametrize("name", UNITS)
def test_every_service_has_a_timer(name):
    cp = _unit(name, ".timer")
    assert cp.has_section("Timer") and cp.get("Install", "WantedBy") == "timers.target"
    assert cp.has_option("Timer", "OnUnitActiveSec") or cp.has_option("Timer", "OnCalendar")


@pytest.mark.parametrize("name", MUTATING)
def test_mutating_jobs_ship_in_dry_run(name):
    cmds = _commands(_unit(name, ".service"))
    last = cmds[-1]
    assert "--dry-run" in last, f"{name} must ship in --dry-run"
    for argv in cmds:
        for banned in ("--live", "--replace", "--allow-disarm", "--execute"):
            assert banned not in argv, f"{name} ships {banned}"


def _parse(argv: list[str]) -> None:
    if argv[2] == "-m":
        module = argv[3]
        rest = argv[4:]
        if module == "cascade.lane_d.ops_jobs":
            import argparse

            jobs = importlib.import_module(module)
            p = argparse.ArgumentParser()
            jobs.add_parser(p.add_subparsers(dest="command", required=True))
            p.parse_args(["jobs", *rest])
        elif module == "cascade.lane_d.published_weights":
            importlib.import_module(module).build_arg_parser().parse_args(rest)
        elif module == "cascade.laned.vault_sweeper":
            importlib.import_module(module)._parser().parse_args(rest)
        else:
            raise AssertionError(f"unexpected module {module}")
    else:
        script = Path(argv[2])
        assert script.as_posix() == "/root/cascade/scripts/laned_w1_measure.py", script
        src = (REPO / "scripts" / "laned_w1_measure.py").read_text(encoding="utf-8")
        for flag in (a for a in argv[3:] if a.startswith("--")):
            assert f'"{flag}"' in src, f"laned_w1_measure.py has no {flag}"


@pytest.mark.parametrize("name", UNITS)
def test_exec_lines_parse_with_the_real_commands(name):
    for argv in _commands(_unit(name, ".service")):
        _parse(argv)


def test_env_example_holds_no_secret():
    env = _env()
    assert env, "the env example names the job inputs"
    for k in env:
        assert not re.search(r"PASSWORD|SECRET|TOKEN|ACCESS_KEY|MNEMONIC|SEED", k), k
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "BT_WALLET_PASSWORD" not in text and "WEBHOOK" not in text


def test_nothing_installs_the_templates():
    assert SYSTEMD.is_dir()
    assert not list(SYSTEMD.glob("*.sh")), "templates only: no installer next to them"
    for sh in (REPO / "deploy").rglob("*.sh"):
        text = sh.read_text(encoding="utf-8", errors="replace")
        assert "laned/systemd" not in text, f"{sh.relative_to(REPO)} installs a job template"


def test_operator_doc_lists_every_job_and_tag():
    text = (REPO / "docs" / "OPERATOR.md").read_text(encoding="utf-8")
    assert "### Lane D recurring jobs" in text
    section = text.split("### Lane D recurring jobs", 1)[1].split("\n## ", 1)[0]
    for name in UNITS:
        assert name in section, name
    jobs = importlib.import_module("cascade.lane_d.ops_jobs")
    for tag in jobs.LOG_TAGS:
        assert tag in section, tag
    for word in ("--preflight-v2-baseline-check", "owner step", "D16"):
        assert word in section, word
