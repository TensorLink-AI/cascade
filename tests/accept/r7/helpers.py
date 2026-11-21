"""Shared spec and helpers for the R7 acceptance tests (see README.md).

R7 ships the S0c overlay bring-up artefacts (plan §3.0 row R7, §5 row 4,
§14.4): a benign seq-1 body generator, a deliberately refused body, a
kill-switch body (T21 / T27, CI half) and a dry-run-first runbook wrapper
around ``cascade-validator --netconfig-overlay ... --netconfig-publish``.

Everything is driven as a subprocess, ``python -P -m cascade.laned.cli overlay
<command> ...``, with cwd and PYTHONPATH set to this repo. Nothing here touches
a chain, a store or a wallet:

- the "live grid" is a JSON fixture in R1's ``laned_grid/1`` schema;
- every program a real publish could reach (``cascade-validator``, ``btcli``,
  ``ssh``, ``aws``, ``s3cmd``, ``curl``) is shimmed first on ``PATH``: a shim
  records its argv and exits 97, so an invocation is caught;
- storage / chain credentials are stripped from the environment.

The netconfig checks (``validate_overlay``, ``check_overrides``,
``body_digest``, ``diff_tables``, ``plan_overlay``) are imported lazily inside
the helpers, never at module level.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MAINNET = "chain.toml"
TESTNET = "chain.testnet.toml"

# R1's live grid for mainnet (tests/accept/r1/README.md "The live grid";
# chain.toml:287,292,293,316,518): 3600-block settlements before the
# DEC-CA-0043 rollover at 9136800, 900 after; eras of 4 settlements.
MAINNET_GRID = {
    "schema": "laned_grid/1",
    "round": {"epoch_blocks": 900, "epoch_blocks_prev": 3600,
              "epoch_activation_block": 9136800, "era_settlements": 4,
              "reveal_margin_blocks": 25},
    "segments": [{"from_block": 0, "epoch_blocks": 3600, "era_blocks": 14400},
                 {"from_block": 9136800, "epoch_blocks": 900, "era_blocks": 3600}],
}
# The testnet grid (chain.testnet.toml:182,186,187,195,249): 300-block
# settlements before 600, 150 after; eras of 4 settlements.
TESTNET_GRID = {
    "schema": "laned_grid/1",
    "round": {"epoch_blocks": 150, "epoch_blocks_prev": 300,
              "epoch_activation_block": 600, "era_settlements": 4,
              "reveal_margin_blocks": 25},
    "segments": [{"from_block": 0, "epoch_blocks": 300, "era_blocks": 1200},
                 {"from_block": 600, "epoch_blocks": 150, "era_blocks": 600}],
}
GRIDS = {MAINNET: MAINNET_GRID, TESTNET: TESTNET_GRID}

# "--after" blocks (a head plus margin, as the owner would pass) per base.
# Their next era starts: 9_802_800 on mainnet (3600-block eras), 6_000_600 on
# testnet (600-block eras); cross-checked with R1's tool in the tests.
AFTER = {MAINNET: 9_800_001, TESTNET: 6_000_001}

# The lane D activation of D10 / D1 (seq N in §5 S6), at an era start on the
# mainnet grid: from_block 9_720_000 = 2700 x 3600, effective 9_712_800.
ARM = {"from_block": 9_720_000, "share": 0.25, "takes_from": "lane_a"}
ARM_EFFECTIVE = 9_712_800
SETTLEMENT_MAIN = 900

SHIMMED = ("cascade-validator", "btcli", "ssh", "scp", "aws", "s3cmd", "curl", "wget")
_SECRET_ENV = re.compile(r"^(HIPPIUS_|HF_|WANDB_|BT_|AWS_|R2_|CLOUDFLARE_)")

# Key material and signing/publishing paths the wrapper must never touch: it
# only prints (and, after explicit approval, runs) cascade-validator.
FORBIDDEN_IN_SOURCE = ("coldkey", "BT_WALLET_PASSWORD", ".bittensor/wallets", "keyfile",
                       "import bittensor", "bt.Wallet", "sign_overlay", "publish_overlay",
                       "open_manifest_store", "put_text")

# Flags the printed command may carry besides the four required ones, each
# with one value (README "The printed command").
OPTIONAL_COMMAND_FLAGS = ("--network", "--netconfig-comment", "--wallet-name",
                          "--wallet-hotkey")


# ── fixtures on disk (plain functions, never pytest fixtures) ────────────────


def write_grid(tmp_path: Path, grid: dict, name: str = "grid.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(grid), encoding="utf-8")
    return p


def copy_base(tmp_path: Path, name: str, text: str | None = None) -> Path:
    """A copy of a shipped config (or ``text``) under ``tmp_path``."""
    d = tmp_path / "base"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text((REPO / name).read_text(encoding="utf-8") if text is None else text,
                 encoding="utf-8")
    return p


def armed_mainnet_text() -> str:
    """chain.toml with lane D armed in the BASE itself (a deployed file that
    already carries the activation): from_block, share and takes_from of
    :data:`ARM` typed into its ``[lane_d]`` table."""
    text = (REPO / MAINNET).read_text(encoding="utf-8")
    i = text.index("\n[lane_d]\n")
    j = text.index("\n[", i + len("\n[lane_d]\n"))
    blk = text[i:j]
    subs = ((r"\nfrom_block\s*=\s*0\b", f"\nfrom_block = {ARM['from_block']}"),
            (r"\nshare\s*=\s*0\.0\b", f"\nshare = {ARM['share']}"),
            (r'\ntakes_from\s*=\s*"lane_b"', f'\ntakes_from = "{ARM["takes_from"]}"'))
    for pat, rep in subs:
        blk, n = re.subn(pat, rep, blk, count=1)
        assert n == 1, f"chain.toml [lane_d] no longer matches {pat!r}"
    return text[:i] + blk + text[j:]


def make_shims(tmp_path: Path) -> tuple[Path, Path]:
    """A bin dir of recording shims and the log they append to."""
    bindir = tmp_path / "shims"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "shim_calls.log"
    for name in SHIMMED:
        p = bindir / name
        p.write_text("#!/bin/sh\n"
                     f'printf "%s\\n" "{name} $*" >> "{log}"\n'
                     "exit 97\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bindir, log


def shim_calls(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


# ── running the tool ─────────────────────────────────────────────────────────


def run_overlay(tmp_path: Path, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """``python -P -m cascade.laned.cli overlay <args>``, shims first on PATH,
    no credentials in the environment, HOME inside ``tmp_path``."""
    bindir, _ = make_shims(tmp_path)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.match(k)}
    env["PYTHONPATH"] = str(REPO)
    env["HOME"] = str(home)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    return subprocess.run([sys.executable, "-P", "-m", "cascade.laned.cli", "overlay", *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


def run_blocks(*args: str) -> dict:
    """R1's ``cascade-laned blocks`` (on main), one cross-checked answer."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.match(k)}
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run([sys.executable, "-P", "-m", "cascade.laned.cli", "blocks", *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (f"cascade-laned blocks {' '.join(args)} exited "
                                  f"{proc.returncode}: {proc.stderr}")
    out = json.loads(proc.stdout)
    assert out.get("cross_checked") is True, out
    return out


def next_era_start(grid: Path, base: Path, after: int) -> int:
    return int(run_blocks("next-era-start", str(after), "--grid", str(grid),
                          "--config", str(base))["next_era_start"])


def describe(proc: subprocess.CompletedProcess) -> str:
    return (f"exit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}")


def ok(proc: subprocess.CompletedProcess, what: str) -> None:
    assert proc.returncode == 0, f"{what}: {describe(proc)}"


def generate(tmp_path: Path, kind: str, base: Path, grid: Path, after: int,
             out: Path, *extra: str) -> tuple[dict, dict, dict]:
    """Run a generator; returns (stdout JSON, body.json, overrides.toml)."""
    proc = run_overlay(tmp_path, kind, "--config", str(base), "--grid", str(grid),
                       "--after", str(after), "--out", str(out), *extra)
    ok(proc, f"cascade-laned overlay {kind}")
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"overlay {kind} stdout is not one JSON object: {e}\n"
                             f"{describe(proc)}") from None
    assert isinstance(summary, dict), f"overlay {kind} stdout is not a JSON object: {summary!r}"
    body_p, ov_p = out / "body.json", out / "overrides.toml"
    assert body_p.is_file(), f"overlay {kind} wrote no {body_p}"
    assert ov_p.is_file(), f"overlay {kind} wrote no {ov_p}"
    body = json.loads(body_p.read_text(encoding="utf-8"))
    assert isinstance(body, dict), f"{body_p} is not a JSON object"
    overrides = tomllib.loads(ov_p.read_text(encoding="utf-8"))
    return summary, body, overrides


# ── netconfig (lazy imports) ─────────────────────────────────────────────────


def load(base: Path):
    from cascade.shared.config import load_chain_config

    return load_chain_config(base)


def nc():
    from cascade.shared import netconfig

    return netconfig


def identity(cfg) -> dict:
    return nc().network_identity(cfg)


def build_body(cfg, overrides: dict, effective_block: int, previous: dict | None = None) -> dict:
    return nc().next_overlay(overrides, effective_block, previous, network=identity(cfg))


def validate(cfg, body: dict, why: str):
    """validate_overlay; a refusal is an assertion failure."""
    try:
        return nc().validate_overlay(cfg, body)
    except Exception as e:  # noqa: BLE001 - every refusal is the failure under test
        raise AssertionError(f"{why}: validate_overlay refused it: "
                             f"{type(e).__name__}: {e}") from None


def refusal(fn, *args) -> str | None:
    """The NetConfigError text ``fn(*args)`` raises, else None."""
    try:
        fn(*args)
    except nc().NetConfigError as e:
        return str(e)
    return None


def parsed(cfg):
    """The parsed config without its raw table and its overlay stamp."""
    return replace(cfg, raw={}, netconfig=replace(cfg.netconfig, seq=0, digest="",
                                                  applied_block=0))


def changed_fields(a, b) -> list[str]:
    """Top-level ChainConfig fields whose parsed value differs."""
    from dataclasses import fields

    pa, pb = parsed(a), parsed(b)
    return [f.name for f in fields(pa) if getattr(pa, f.name) != getattr(pb, f.name)]


def offending(overrides: dict) -> list[tuple[str, str | None]]:
    """Each (table, key) - key None for a whole table - check_overrides refuses."""
    out: list[tuple[str, str | None]] = []
    for table, kv in overrides.items():
        if refusal(nc().check_overrides, {table: {}}) is not None or not isinstance(kv, dict):
            out.append((table, None))
            continue
        for key, value in kv.items():
            if refusal(nc().check_overrides, {table: {key: value}}) is not None:
                out.append((table, key))
    return out


def strip(overrides: dict, bad: list[tuple[str, str | None]]) -> dict:
    out = copy.deepcopy(overrides)
    for table, key in bad:
        if key is None:
            out.pop(table, None)
        else:
            out[table].pop(key, None)
    return {t: kv for t, kv in out.items() if kv}


# ── the wrapper's dry-run output ─────────────────────────────────────────────


def lines(proc: subprocess.CompletedProcess, tag: str) -> list[str]:
    """Rest of every stdout line starting ``<tag> `` (or equal to ``tag``)."""
    out = []
    for line in proc.stdout.splitlines():
        if line == tag:
            out.append("")
        elif line.startswith(tag + " "):
            out.append(line[len(tag) + 1:])
    return out


def one(proc: subprocess.CompletedProcess, tag: str) -> str:
    got = lines(proc, tag)
    assert len(got) == 1, f"expected exactly one {tag!r} line, got {got!r}\n{describe(proc)}"
    return got[0]


def command_argv(proc: subprocess.CompletedProcess) -> list[str]:
    return shlex.split(one(proc, "COMMAND"))


def toml_text(overrides: dict) -> str:
    """A TOML file for a flat ``{table: {key: scalar}}`` overrides table."""
    out = []
    for table, kv in overrides.items():
        out.append(f"[{table}]")
        for key, v in kv.items():
            if isinstance(v, bool):
                out.append(f"{key} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                out.append(f"{key} = {v!r}")
            else:
                out.append(f"{key} = {json.dumps(str(v))}")
        out.append("")
    return "\n".join(out)
