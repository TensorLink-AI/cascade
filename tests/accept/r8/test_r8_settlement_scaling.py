"""R8, #787 (testnet E2E 2026-11-23): the settlement blackout scales with the
settlement length, and a stale block sample is refused as `clock`.

1. The blackout was fixed at 20 min before + 15 min after a boundary. On
   testnet's 30-min settlement (150 blocks) that is 35 min > 30 min, so
   `check` never said GO. It now scales with the settlement length in force:
   mainnet's 3 h (900-block) settlement keeps exactly 20 + 15 min (pinned
   here), a 12 h (3600-block) one keeps them too (capped, never more), and
   testnet's 30 min has a window.
2. The block clock fell back to `<root>/trainer.log` whenever validator.log
   was absent and extrapolated from a days-old sample. The gate now takes the
   NEWEST sample of the two logs, refuses a sample older than 2 h as `clock`,
   and `check --validator-log FILE` names the validator's log (testnet's main
   validator writes val0.log).

The gate is loaded by file path; the CLI runs against throwaway dirs in
`tmp_path`. Nothing reads the production box.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE_PY = REPO / "scripts" / "deploy_gate.py"
NOW = datetime(2026, 11, 16, 3, 0, tzinfo=UTC)          # a Monday, outside the freeze


def _load_gate():
    spec = importlib.util.spec_from_file_location("r8_deploy_gate_787", GATE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


g = _load_gate()


def _line(ts: datetime, block: int, who: str = "cascade.validator") -> str:
    return (f"{ts:%Y-%m-%d %H:%M:%S},000 INFO {who}: re-asserting weights "
            f"(last set 3 blocks ago, block {block})")


def _rules(*, block: int, grid=None, at: datetime = NOW, vlines=None, tlines=()):
    """Refused rules for a validator restart at `at`, the chain at `block` then
    (sampled exactly at `at` unless explicit lines are given)."""
    v = g.gate(service="validator", areas={}, diff_text="", now=at,
               trainer_lines=list(tlines),
               validator_lines=[_line(at, block)] if vlines is None else list(vlines),
               work=Path("/nonexistent-r8-work"), on_main=True, acks=set(),
               **({} if grid is None else {"grid": grid}))
    return [r for r, _ in v.refuse]


def _grid(path: str):
    grid = g.load_grid(REPO / path)
    assert grid is not None, path
    return grid


# ── 1. the blackout scales; mainnet's minutes are pinned ─────────────────────


# Mainnet grid (900 blocks, 12 s): boundary B = 9516600. 1 block = 0.2 min.
B = 9516600


def _go_blocks(grid, start: int, length: int) -> list[int]:
    return [b for b in range(start, start + length) if not _rules(block=b, grid=grid)]


def _testnet_window() -> list[int]:
    grid = _grid("chain.testnet.toml")
    assert grid.epoch_blocks == 150
    start = 9516150 // 150 * 150
    go = _go_blocks(grid, start, 150)
    assert go, "no GO instant in a whole 30-min testnet settlement (fixed 20 + 15 min)"
    return go


def test_testnet_settlement_has_a_restart_window():
    """chain.testnet.toml: 150 blocks = 30 min. The old fixed 20 + 15 min
    refused every instant; now a window exists in every settlement."""
    grid = _grid("chain.testnet.toml")
    start = 9516150 // 150 * 150
    go = _testnet_window()
    # the same fractions as mainnet (1/9 before, 1/12 after): most of it is open
    assert len(go) >= 150 // 2, len(go)
    # and the boundary's neighbourhood is still refused
    assert start not in go and start + 149 not in go
    assert "settlement" in _rules(block=start + 1, grid=grid)
    assert "settlement" in _rules(block=start + 149, grid=grid)


def test_mainnet_minutes_are_pinned_while_testnet_scales():
    """The scaling never moves mainnet's 3 h settlement: exactly 20 min before
    and 15 min after a boundary (pinned to the block), 726 of 900 blocks open."""
    _testnet_window()
    grid = _grid("chain.toml")
    assert grid.epoch_blocks == 900
    for blocks_before, refused in ((101, False), (100, False), (99, True), (1, True)):
        rules = _rules(block=B - blocks_before, grid=grid)          # 20.2 / 20.0 / 19.8 min
        assert ("settlement" in rules) is refused, (blocks_before, rules)
    for blocks_after, refused in ((0, True), (74, True), (75, False), (76, False)):
        rules = _rules(block=B + blocks_after, grid=grid)           # 14.8 / 15.0 / 15.2 min
        assert ("settlement" in rules) is refused, (blocks_after, rules)
    assert len(_go_blocks(grid, B, 900)) == 900 - 75 - 99


def test_a_longer_settlement_keeps_the_mainnet_minutes():
    """12 h (3600 blocks): the minutes are capped at mainnet's, never longer."""
    _testnet_window()
    grid = g.Grid(3600)
    assert "settlement" in _rules(block=3600 * 2700 - 99, grid=grid)      # 19.8 min before
    assert "settlement" not in _rules(block=3600 * 2700 - 101, grid=grid)
    assert "settlement" in _rules(block=3600 * 2700 + 74, grid=grid)
    assert "settlement" not in _rules(block=3600 * 2700 + 76, grid=grid)
    assert "settlement" not in _rules(block=3600 * 2700 + 1800, grid=grid)  # mid-settlement


def test_the_blackout_follows_the_grid_in_force_across_a_switch():
    """Before the activation block the previous (3600) length applies: its
    full 20 min; after it the 150-block length's scaled minutes."""
    grid = g.Grid(150, epoch_blocks_prev=3600, activation_block=3600 * 3000)
    act = 3600 * 3000
    assert "settlement" in _rules(block=act - 99, grid=grid)        # 19.8 min before
    assert "settlement" not in _rules(block=act + 150 - 30, grid=grid)  # 6 min before
    assert "settlement" in _rules(block=act + 150 - 5, grid=grid)


# ── 2. the block clock: freshest sample, stale = clock ───────────────────────


def test_a_days_old_trainer_log_is_refused_as_clock():
    """The testnet case: no validator.log, a trainer.log last written 3 days
    ago. The extrapolation is not trusted: `clock`, not a settlement verdict."""
    old = NOW - timedelta(days=3)
    rules = _rules(block=0, vlines=[], tlines=[_line(old, 9516100, "cascade.trainer")])
    assert rules == ["clock"], rules


def test_a_stale_validator_sample_is_refused_as_clock():
    old = NOW - timedelta(hours=3)
    assert _rules(block=0, vlines=[_line(old, 9516100)]) == ["clock"]
    recent = NOW - timedelta(minutes=70)                # a skipped re-assert: still fine
    assert "clock" not in _rules(block=0, vlines=[_line(recent, 9516100)])


def test_the_newest_sample_of_either_log_places_the_window():
    """A stale validator sample beside a fresh trainer one (or the reverse):
    the fresh one is used."""
    fresh = [_line(NOW - timedelta(minutes=1), 9516100 - 5, "cascade.trainer")]
    stale_v = [_line(NOW - timedelta(days=3), 9516590)]
    # from the fresh sample: ~1 h 40 min before 9516600 -> GO
    assert _rules(block=0, vlines=stale_v, tlines=fresh) == []
    near_t = [_line(NOW - timedelta(days=3), 9516100, "cascade.trainer")]
    near_v = [_line(NOW - timedelta(minutes=1), 9516590)]    # 2 min before 9516600
    assert _rules(block=0, vlines=near_v, tlines=near_t) == ["settlement"]


# ── CLI: --validator-log ─────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=r8", "-c", "user.email=r8@example.invalid",
         "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        capture_output=True, text=True, check=True, timeout=60).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base")
    rev_from = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("next\n")
    _git(repo, "commit", "-q", "-am", "next")
    rev_to = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", rev_to)
    return repo, rev_from, rev_to


def _check(tmp_path: Path, root: Path, *extra: str) -> subprocess.CompletedProcess:
    repo, rev_from, rev_to = _repo(tmp_path)
    return subprocess.run(
        [sys.executable, "-P", str(GATE_PY), "check", "--service", "validator",
         "--from", rev_from, "--to", rev_to, "--repo", str(repo), "--root", str(root),
         "--chain-toml", str(REPO / "chain.testnet.toml"), *extra],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)


def test_cli_refuses_a_stale_trainer_log_as_clock(tmp_path):
    root = tmp_path / "box"
    root.mkdir()
    old = datetime.now(UTC) - timedelta(days=3)
    (root / "trainer.log").write_text("\n" + _line(old, 9516100, "cascade.trainer") + "\n")
    r = _check(tmp_path, root)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr[-400:])
    assert "REFUSE [clock]" in r.stdout, r.stdout
    assert "REFUSE [settlement]" not in r.stdout, r.stdout


def test_cli_reads_the_named_validator_log(tmp_path):
    """--validator-log val0.log: the fresh sample places the window (no
    `clock` refusal), however old trainer.log is."""
    root = tmp_path / "box"
    root.mkdir()
    now = datetime.now(UTC)
    (root / "trainer.log").write_text(
        "\n" + _line(now - timedelta(days=3), 9516100, "cascade.trainer") + "\n")
    (root / "val0.log").write_text("\n" + _line(now - timedelta(seconds=30), 9516100) + "\n")
    r = _check(tmp_path, root, "--validator-log", str(root / "val0.log"))
    assert r.returncode in (0, 1), (r.returncode, r.stdout, r.stderr[-400:])
    assert "Traceback" not in r.stderr, r.stderr[-800:]
    assert "REFUSE [clock]" not in r.stdout, r.stdout
