"""Shared helpers for the R5 acceptance tests (see README.md).

The restart scripts are only ever run with ``--dry-run`` against a fixture box
root, a fixture git repo and a fixture venv in ``tmp_path``. Every command a
real restart would use to change the box is shimmed first on ``PATH``: a shim
records its argv and exits 97, so a dry-run that calls one is caught.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LANED = REPO / "deploy" / "laned"

SERVICES = {
    # script stem -> (deploy_gate service name, entry-point needles)
    "eval_host": ("eval-host", ()),
    "validator": ("validator", ("cascade-validator",)),
    "intake_laned": ("intake-laned", ("cascade-laned-intake", "cascade.lane_d")),
    "trainer": ("trainer", ("cascade-trainer",)),
}
S5_ORDER = ("eval_host", "validator", "intake_laned", "trainer")
BOX_SERVICES = ("validator", "intake_laned", "trainer")

SHIMMED = ("systemd-run", "systemctl", "choom", "ssh", "scp", "rsync", "pkill",
           "sqlite3", "uv", "busctl", "nohup", "sudo", "docker", "lium", "btcli")

PRESENT_STATE = (
    "validator_state.db", "cascade_state.json", "warm_start_init.json",
    "netconfig_state.json", "activation_forfeit_state.json",
    "activation_margin_v2_state.json", "satellite_state.json", "decided_votes.toml",
)
ABSENT_STATE = ("lane_d_state.json",)

# Monday 2026-11-16 12:00 UTC: outside the Wed-Thu freeze.
QUIET_NOW = "2026-11-16T12:00:00Z"
QUIET_TS = "20261116T120000Z"
BOUNDARY = 9518400  # a multiple of 900

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "laned-fixture",
    "GIT_AUTHOR_EMAIL": "laned-fixture@example.invalid",
    "GIT_COMMITTER_NAME": "laned-fixture",
    "GIT_COMMITTER_EMAIL": "laned-fixture@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], env={**os.environ, **_GIT_IDENTITY},
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def log_line(stamp: str, text: str) -> str:
    """``2026-11-16T12:00:00Z`` -> a box log line ``2026-11-16 12:00:00,000 INFO text``."""
    day, clock = stamp.rstrip("Z").split("T")
    return f"{day} {clock},000 INFO {text}\n"


@dataclass
class Box:
    tmp: Path
    root: Path
    repo: Path
    venv: Path
    shim_dir: Path
    shim_log: Path
    release_dir: Path
    previous_dir: Path
    known_hosts: Path
    release_ref: str = "rc42"
    previous_ref: str = "rc41"
    host: str = "root@203.0.113.7"
    port: str = "2222"

    # -- fixture state writers ------------------------------------------------

    def write_logs(self, *, now: str = QUIET_NOW, block: int | None = BOUNDARY + 450,
                   validator_extra: str = "") -> None:
        """Both logs get a header line, then (optionally) a ``block N`` sample at ``now``."""
        for name in ("validator.log", "trainer.log"):
            text = log_line(now, "log opened")
            if block is not None:
                text += log_line(now, f"poll ok block {block}")
            if name == "validator.log":
                text += validator_extra
            (self.root / name).write_text(text)

    def write_queue(self, entries: dict) -> None:
        (self.root / "_train_work" / "funded_queue.json").write_text(
            json.dumps({"entries": entries}))

    def write_era_state(self, state: dict) -> None:
        (self.root / "_train_work" / "era_state.json").write_text(json.dumps(state))

    # -- running ----------------------------------------------------------------

    def args(self, svc: str, *extra: str) -> list[str]:
        out = ["--dry-run", "--root", str(self.root), "--repo", str(self.repo),
               "--venv", str(self.venv),
               "--release-ref", self.release_ref, "--release-dir", str(self.release_dir),
               "--previous-ref", self.previous_ref, "--previous-dir", str(self.previous_dir)]
        if svc in ("eval_host", "upgrade_s5"):
            out += ["--host", self.host, "--port", self.port,
                    "--known-hosts", str(self.known_hosts)]
        return out + list(extra)

    def env(self, now: str) -> dict[str, str]:
        env = {**os.environ, **_GIT_IDENTITY}
        env["PATH"] = f"{self.shim_dir}{os.pathsep}{env.get('PATH', '')}"
        env["LANED_NOW"] = now
        env["CASCADE_ROOT"] = str(self.root)
        env["HOME"] = str(self.tmp / "home")
        return env

    def run(self, svc: str, *extra: str, now: str = QUIET_NOW,
            args: list[str] | None = None) -> subprocess.CompletedProcess:
        script = script_path("upgrade_s5" if svc == "upgrade_s5" else f"restart_{svc}")
        argv = self.args(svc, *extra) if args is None else args
        return subprocess.run(["bash", str(script), *argv], env=self.env(now),
                              capture_output=True, text=True, timeout=120,
                              cwd=str(self.tmp))

    def shim_calls(self) -> list[str]:
        return self.shim_log.read_text().splitlines() if self.shim_log.exists() else []

    def fingerprint(self) -> dict[str, tuple]:
        """Every file under the fixture tree with its size, mtime and digest."""
        out = {}
        for base in (self.root, self.repo, self.previous_dir, self.venv):
            for path in sorted(base.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                st = path.stat()
                out[str(path)] = (st.st_size, st.st_mtime_ns,
                                  hashlib.sha256(path.read_bytes()).hexdigest())
        return out

    def worktrees(self) -> str:
        return git(self.repo, "worktree", "list", "--porcelain")


def script_path(stem: str) -> Path:
    path = LANED / f"{stem}.sh"
    assert path.is_file(), f"R5 script missing: {path.relative_to(REPO)}"
    return path


def output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


def plan_lines(text: str, step: str | None = None) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("PLAN ")]
    if step is None:
        return lines
    return [ln for ln in lines if ln.split()[1] == step]


def build_box(tmp: Path) -> Box:
    tmp = tmp.resolve()
    # The source repo with two release tags.
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "VERSION").write_text("rc41\n")
    git(repo, "add", "VERSION")
    git(repo, "commit", "-q", "-m", "rc41")
    git(repo, "tag", "rc41")
    (repo / "VERSION").write_text("rc42\n")
    git(repo, "add", "VERSION")
    git(repo, "commit", "-q", "-m", "rc42")
    git(repo, "tag", "rc42")

    # The previous release already has its worktree (kept for rollback).
    previous_dir = tmp / "cascade-rc41"
    git(repo, "worktree", "add", "-q", "--detach", str(previous_dir), "rc41")
    release_dir = tmp / "cascade-rc42"

    # The box root.
    root = tmp / "box"
    (root / "_train_work").mkdir(parents=True)
    for name in PRESENT_STATE:
        (root / name).write_text(f"{name} state\n")

    # The pinned venv.
    venv = tmp / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    for entry in ("cascade-validator", "cascade-trainer", "cascade-laned-intake"):
        p = venv / "bin" / entry
        p.write_text("")
        p.chmod(0o755)

    # Shims for every command that changes the box.
    shim_dir = tmp / "shims"
    shim_dir.mkdir()
    shim_log = tmp / "shim_calls.log"
    for name in SHIMMED:
        p = shim_dir / name
        p.write_text(f'#!/bin/sh\necho "{name} $*" >> "{shim_log}"\nexit 97\n')
        p.chmod(0o755)

    known_hosts = tmp / "known_hosts.evalhost"
    known_hosts.write_text("[203.0.113.7]:2222 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixture\n")
    (tmp / "home").mkdir()

    box = Box(tmp=tmp, root=root, repo=repo, venv=venv, shim_dir=shim_dir, shim_log=shim_log,
              release_dir=release_dir, previous_dir=previous_dir, known_hosts=known_hosts)
    box.write_logs(validator_extra=(log_line(QUIET_NOW, "new manifest round=776")
                                    + log_line(QUIET_NOW, "round=776 weights set")))
    box.write_queue({"a": {"status": "done", "phase": "", "hotkey": "5DoneHotkey"}})
    box.write_era_state({})
    return box
