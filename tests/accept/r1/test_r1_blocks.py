"""R1 acceptance tests: the block-arithmetic tool ``cascade-laned blocks``.

Plan: docs/plans/LANE_D_PLAN.md §3.0 row R1, §5 "Block arithmetic", §13.
Contract: tests/accept/r1/README.md.

The tool is driven as a subprocess, ``python -m cascade.laned.cli blocks
<query> ... --grid FILE [--config FILE]``, with cwd and PYTHONPATH set to this
repo. Nothing here touches a chain: the "live grid" is a JSON fixture.

Every golden vector was computed from the real code on main
(cascade/shared/era.py, cascade/shared/activation.py,
cascade/shared/netconfig.py, cascade/shared/chain.py) and the real chain.toml
values, and most are real blocks from the OPSLOG. Each cites its source. The
test ``test_golden_vectors_agree_with_shared_code`` re-derives them from that
code so a vector can never drift from ground truth.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]

# The live mainnet grid. `round` is what chain.toml [round] says
# (chain.toml:287 epoch_blocks = 900, :292 epoch_blocks_prev = 3600,
# :293 epoch_activation_block = 9136800, :316 era_settlements = 4,
# :518 reveal_margin_blocks = 25). `segments` is the grid as observed on the
# chain: 3600-block settlements before the DEC-CA-0043 rollover at 9136800,
# 900 after; eras of 4 settlements (era.py:131-137, era_length_blocks).
LIVE_GRID = {
    "schema": "laned_grid/1",
    "round": {
        "epoch_blocks": 900,
        "epoch_blocks_prev": 3600,
        "epoch_activation_block": 9136800,
        "era_settlements": 4,
        "reveal_margin_blocks": 25,
    },
    "segments": [
        {"from_block": 0, "epoch_blocks": 3600, "era_blocks": 14400},
        {"from_block": 9136800, "epoch_blocks": 900, "era_blocks": 3600},
    ],
}


def _grid_file(tmp_path: Path, grid: dict, name: str = "grid.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(grid))
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, "-m", "cascade.laned.cli", "blocks", *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )


def _ok(tmp_path: Path, *args: str, grid: dict | None = None) -> dict:
    g = _grid_file(tmp_path, LIVE_GRID if grid is None else grid)
    proc = _run(*args, "--grid", str(g))
    assert proc.returncode == 0, (
        f"cascade-laned blocks {' '.join(args)} exited {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    out = json.loads(proc.stdout)
    assert isinstance(out, dict), out
    assert out.get("cross_checked") is True, f"answer not cross-checked from two computations: {out}"
    return out


def _subset(out: dict, expected: dict) -> None:
    got = {k: out.get(k) for k in expected}
    assert got == expected, f"got {got}, expected {expected} (full output {out})"


# ── era-start <block> ────────────────────────────────────────────────────────
# (block, era_index, era_start, is_era_start, epoch_blocks, era_blocks)
ERA_START_VECTORS = [
    # OPSLOG 2026-11-20 14:2x: "boundary 9547200, era 2652 start".
    (9547200, 2652, 9547200, True, 900, 3600),
    # One block earlier is still era 2651 (era.py:142-148).
    (9547199, 2651, 9543600, False, 900, 3600),
    # OPSLOG 2026-11-20 (21:2x entry): next king check due at boundary 9561600.
    (9561600, 2656, 9561600, True, 900, 3600),
    # chain.toml:1241: image_decoupled_from_block "applied at the era start 9493200".
    (9493200, 2637, 9493200, True, 900, 3600),
    # The rollover block is an era start of the NEW grid (chain.toml:293,312).
    (9136800, 2538, 9136800, True, 900, 3600),
    # Off-by-one trap (rounds.py:587-590, "never the current head"): the
    # rolling-era-king lock-in block 9133200 lies on the OLD grid
    # (3600-block settlements, 14400-block eras), so it is NOT an era start
    # and its era began at 9129600. Using the head grid (900/3600) would say
    # era start 9133200.
    (9133200, 634, 9129600, False, 3600, 14400),
    (9136799, 634, 9129600, False, 3600, 14400),
    # netconfig.py:446-448: is_era_start requires block > 0.
    (0, 0, 0, False, 3600, 14400),
]


@pytest.mark.parametrize("block,index,start,is_start,eb,era", ERA_START_VECTORS)
def test_era_start_golden(tmp_path, block, index, start, is_start, eb, era):
    out = _ok(tmp_path, "era-start", str(block))
    _subset(out, {"query": "era-start", "block": block, "era_index": index, "era_start": start,
                  "is_era_start": is_start, "epoch_blocks": eb, "era_blocks": era})


# ── next-era-start <block> ───────────────────────────────────────────────────
# era.py:167-170: the first era boundary STRICTLY after the block.
NEXT_ERA_VECTORS = [
    (9547200, 9550800, 2653),   # on an era start ⇒ the following one, never itself
    (9547199, 9547200, 2652),   # OPSLOG 2026-11-20: era 2652 starts at 9547200
    (9558000, 9561600, 2656),   # OPSLOG 2026-11-20: 9561600
    (9486000, 9489600, 2636),   # chain.toml:1241 lock-in block 9486000 (an era start)
]


@pytest.mark.parametrize("block,nxt,idx", NEXT_ERA_VECTORS)
def test_next_era_start_golden(tmp_path, block, nxt, idx):
    out = _ok(tmp_path, "next-era-start", str(block))
    _subset(out, {"query": "next-era-start", "block": block, "next_era_start": nxt,
                  "next_era_index": idx})


# ── settlement <boundary> ────────────────────────────────────────────────────
# Two era conventions coexist in the code and both are needed:
#   era_index      = era containing the boundary (era.py:139, used by
#                    v2_check_due / min_effective_era, era.py:500);
#   settlement_era = era the settlement is judged in = era containing
#                    boundary − 1 (era.py:151-164): an era's LAST settlement
#                    is the next era's start block.
SETTLEMENT_VECTORS = [
    # OPSLOG 2026-11-08 21:00 entry: "22:23 settlement (boundary 9464400): era
    # 2628 closed"; OPSLOG 2026-11-06 entry: v2_check_due's era(B) for 9464400
    # is 2629 ("first settlement of era 2629 = 9464400").
    (9464400, 2629, 2628, 9460800, True),
    # OPSLOG 2026-11-20: boundary 9547200 is the era 2652 start, which is the
    # last settlement of era 2651.
    (9547200, 2652, 2651, 9543600, True),
    # OPSLOG 2026-11-09 13:28 entry: "boundary 9468900 (era 2630 …)".
    (9468900, 2630, 2630, 9468000, False),
    # The rollover seam: the settlement at 9136800 closes the last OLD-grid
    # era (634, start 9129600), while 9136800 itself opens new-grid era 2538.
    (9136800, 2538, 634, 9129600, True),
]


@pytest.mark.parametrize("boundary,idx,s_era,s_start,is_start", SETTLEMENT_VECTORS)
def test_settlement_golden(tmp_path, boundary, idx, s_era, s_start, is_start):
    out = _ok(tmp_path, "settlement", str(boundary))
    _subset(out, {"query": "settlement", "boundary": boundary, "era_index": idx,
                  "settlement_era": s_era, "settlement_era_start": s_start,
                  "is_era_start": is_start})


def test_settlement_refuses_a_block_that_is_not_a_boundary(tmp_path):
    g = _grid_file(tmp_path, LIVE_GRID)
    proc = _run("settlement", "9464450", "--grid", str(g))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == ""


# ── activation --lock-block L --align boundary|era|epoch ─────────────────────
# activation.py:373-394 activation_block_for.
ACTIVATION_VECTORS = [
    # chain.toml:287 "lock 9133200 → 9136800" (rolling-era-king, boundary
    # align). Off-by-one trap: the NEXT boundary on the grid in force AT the
    # lock block (3600), not the head grid (900 would give 9134100).
    (9133200, "boundary", 9136800),
    # chain.toml:772 forfeit: "locked in at 9165600, applied at 9166500".
    (9165600, "boundary", 9166500),
    # chain.toml:740 margin-v2 (era align, OPSLOG 2026-10-01 22:03 "LOCKED IN
    # at boundary 9190800 … rollover at block 9194400"). 9190800 is itself an
    # era start: "a lock-in exactly on an era start takes the next one"
    # (activation.py:379), never 9190800.
    (9190800, "era", 9194400),
    # Same lock under the other aligns, so an align mix-up cannot pass.
    (9190800, "boundary", 9191700),
    (9190800, "epoch", 9198000),
    # chain.toml:1241 image_decoupled (v2b, epoch align): "locked in at
    # 9486000 … applied at the era start 9493200" = TWO eras after the
    # lock-in era (era.py:173-184 min_effective_era).
    (9486000, "epoch", 9493200),
    (9486000, "era", 9489600),
    # OPSLOG 2026-11-06 19:40 entry: "lock in era 2625 → 9457200; era 2626 →
    # 9460800" (epoch align from a 900-block lock-in boundary).
    (9452700, "epoch", 9457200),
    (9454500, "epoch", 9460800),
]


@pytest.mark.parametrize("lock,align,act", ACTIVATION_VECTORS)
def test_activation_golden(tmp_path, lock, align, act):
    out = _ok(tmp_path, "activation", "--lock-block", str(lock), "--align", align)
    _subset(out, {"query": "activation", "lock_block": lock, "align": align,
                  "activation_block": act})


# ── effective-block --effective-block E --lock-block L (overlay apply block) ─
# netconfig.py:508-516 apply_block_for: E when the lock-in lands strictly
# before E, else the first era start strictly after the lock-in.
EFFECTIVE_VECTORS = [
    (9561600, 9558000, 9561600),   # lock one era early ⇒ E
    (9561600, 9560700, 9561600),   # lock one settlement early ⇒ E
    # Off-by-one trap: a lock-in exactly AT E is not "before" it, so the
    # overlay slips a whole era to 9565200.
    (9561600, 9561600, 9565200),
    (9561600, 9562500, 9565200),   # late lock-in ⇒ first era start after it
]


@pytest.mark.parametrize("eff,lock,apply", EFFECTIVE_VECTORS)
def test_effective_block_golden(tmp_path, eff, lock, apply):
    out = _ok(tmp_path, "effective-block", "--effective-block", str(eff), "--lock-block", str(lock))
    _subset(out, {"query": "effective-block", "effective_block": eff, "lock_block": lock,
                  "apply_block": apply})


def test_effective_block_refuses_a_non_era_start(tmp_path):
    # netconfig.py:500-505 validate_overlay refuses an effective_block that is
    # not an era start; 9562500 is a settlement boundary, not an era start.
    g = _grid_file(tmp_path, LIVE_GRID)
    proc = _run("effective-block", "--effective-block", "9562500", "--lock-block", "9558000",
                "--grid", str(g))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == ""


# ── min-effective-era <block> ────────────────────────────────────────────────
# era.py:173-184: created in era e ⇒ effective no earlier than era e + 2.
MIN_EFFECTIVE_VECTORS = [
    # OPSLOG 2026-11-20 14:2x: "eff block 2654·3600 = 9554400" for 9547200.
    (9547200, 2654, 9554400),
    # OPSLOG 2026-11-09 13:28: "boundary 9468900 (era 2630, effective 2632)".
    (9468900, 2632, 9475200),
    # … and "a fire lands in era 2631 where min_effective = 2633".
    (9471600, 2633, 9478800),
    # OPSLOG 2026-11-14 12:15: reign clock → 9504000 (era 2640) ⇒ effective era 2642.
    (9504000, 2642, 9511200),
]


@pytest.mark.parametrize("block,era,eff_block", MIN_EFFECTIVE_VECTORS)
def test_min_effective_era_golden(tmp_path, block, era, eff_block):
    out = _ok(tmp_path, "min-effective-era", str(block))
    _subset(out, {"query": "min-effective-era", "block": block, "min_effective_era": era,
                  "effective_block": eff_block})


# ── promotion-fire --check-boundary B --fire-block F ("arrived too late") ────
# runner_promotion.py:500-509: fires only if min_effective_era(B) >=
# min_effective_era(F). OPSLOG 2026-11-14 02:20: "only an era's 1st/2nd
# settlement check can fire" (the fire comes 2 settlements after the check).
FIRE_VECTORS = [
    # OPSLOG 2026-11-09 13:28: check at 9468900 (era 2630); fire 2
    # settlements later, still in era 2630 ⇒ fires.
    (9468900, 9470700, 2632, 2632, True),
    # … the same check fired in era 2631 ⇒ "arrived too late".
    (9468900, 9471600, 2632, 2633, False),
    # Era 2639 = [9500400, 9504000): a check at its 2nd settlement fires at
    # the 4th; a check at its 3rd would fire at 9504000 = era 2640 ⇒ too late.
    (9501300, 9503100, 2641, 2641, True),
    (9502200, 9504000, 2641, 2642, False),
]


@pytest.mark.parametrize("check,fire,eff,at_fire,fires", FIRE_VECTORS)
def test_promotion_fire_golden(tmp_path, check, fire, eff, at_fire, fires):
    out = _ok(tmp_path, "promotion-fire", "--check-boundary", str(check), "--fire-block", str(fire))
    _subset(out, {"query": "promotion-fire", "check_boundary": check, "fire_block": fire,
                  "effective_era": eff, "min_effective_era_at_fire": at_fire, "fires": fires})


# ── reveal <block> [--margin M] [--next-epoch] ───────────────────────────────
# chain.py:81-118 blocks_until_boundary_reveal, with the grid in force AT the
# block (miner/cli.py:913) and the targeted boundary as in miner/cli.py:920-921.
# Margin defaults to the grid's reveal_margin_blocks (chain.toml:518 = 25).
REVEAL_VECTORS = [
    # (block, extra args, margin, boundary, blocks_until_reveal, reveal_block)
    (9547000, [], 25, 9547200, 175, 9547175),
    (9547174, [], 25, 9547200, 1, 9547175),
    # Inside the margin: floors to 1 (reveal now), still the imminent boundary.
    (9547190, [], 25, 9547200, 1, 9547191),
    # Exactly ON a boundary: targets the NEXT one, never the current block.
    (9547200, [], 25, 9548100, 875, 9548075),
    (9547000, ["--next-epoch"], 25, 9548100, 1075, 9548075),
    (9547000, ["--margin", "50"], 50, 9547200, 150, 9547150),
    # Before the rollover the 3600-block grid applies (rounds.py:571-599).
    (9136700, [], 25, 9136800, 75, 9136775),
]


@pytest.mark.parametrize("block,extra,margin,boundary,delay,reveal", REVEAL_VECTORS)
def test_reveal_golden(tmp_path, block, extra, margin, boundary, delay, reveal):
    out = _ok(tmp_path, "reveal", str(block), *extra)
    _subset(out, {"query": "reveal", "block": block, "margin_blocks": margin, "boundary": boundary,
                  "blocks_until_reveal": delay, "reveal_block": reveal})


# ── the two-computation cross-check ──────────────────────────────────────────
# Computation 1 runs the shared code on the `round` keys; computation 2 walks
# the observed `segments`. A grid whose two descriptions disagree for the
# query must be refused: exit 3, "mismatch" on stderr, nothing on stdout.

def _mismatched_activation_grid() -> dict:
    # `round` names the lock-in block 9133200 as the grid switch, the chain
    # switched at 9136800 (chain.toml:287: lock 9133200 → rollover 9136800).
    g = json.loads(json.dumps(LIVE_GRID))
    g["round"]["epoch_activation_block"] = 9133200
    return g


def _pre_8726400_history_grid() -> dict:
    # chain.toml:284-285: "Last used: 7200 → 3600 at 8726400". chain.toml only
    # remembers the LAST switch, so the code computes 3600 before 8726400
    # while the observed chain had 7200.
    g = json.loads(json.dumps(LIVE_GRID))
    g["segments"] = [
        {"from_block": 0, "epoch_blocks": 7200, "era_blocks": 28800},
        {"from_block": 8726400, "epoch_blocks": 3600, "era_blocks": 14400},
        {"from_block": 9136800, "epoch_blocks": 900, "era_blocks": 3600},
    ]
    return g


def _era_settlements_grid() -> dict:
    g = json.loads(json.dumps(LIVE_GRID))
    g["round"]["era_settlements"] = 3
    return g


MISMATCH_CASES = [
    ("activation_seam_era_start", _mismatched_activation_grid, ["era-start", "9134100"]),
    ("activation_seam_lock", _mismatched_activation_grid,
     ["activation", "--lock-block", "9133200", "--align", "boundary"]),
    ("old_7200_grid", _pre_8726400_history_grid, ["era-start", "8712000"]),
    ("era_settlements", _era_settlements_grid, ["era-start", "9547200"]),
    ("era_settlements_min_effective", _era_settlements_grid, ["min-effective-era", "9547200"]),
]


@pytest.mark.parametrize("name,grid_fn,args", MISMATCH_CASES, ids=[c[0] for c in MISMATCH_CASES])
def test_two_computations_disagree_is_refused(tmp_path, name, grid_fn, args):
    g = _grid_file(tmp_path, grid_fn())
    proc = _run(*args, "--grid", str(g))
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "mismatch" in proc.stderr.lower(), proc.stderr
    assert proc.stdout.strip() == "", "a refused answer must not be printed"


def test_consistent_grid_same_queries_pass(tmp_path):
    # The same queries on the consistent live grid succeed, so the refusals
    # above come from the disagreement, not from the queries.
    for args in (["era-start", "9134100"], ["era-start", "9547200"],
                 ["min-effective-era", "9547200"]):
        _ok(tmp_path, *args)
    out = _ok(tmp_path, "activation", "--lock-block", "9133200", "--align", "boundary")
    assert out["activation_block"] == 9136800


# ── --config: chain.toml vs the live grid ────────────────────────────────────

def test_config_matching_repo_chain_toml_passes(tmp_path):
    g = _grid_file(tmp_path, LIVE_GRID)
    proc = _run("era-start", "9547200", "--grid", str(g), "--config", str(REPO / "chain.toml"))
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    out = json.loads(proc.stdout)
    assert out["era_index"] == 2652 and out["is_era_start"] is True and out["cross_checked"] is True


def test_config_drift_from_live_grid_is_refused(tmp_path):
    # A deployed chain.toml whose [round] grid differs from the live grid
    # (here a stale epoch_activation_block) must be refused, not silently used.
    toml = tmp_path / "chain.toml"
    toml.write_text("[round]\nepoch_blocks = 900\nepoch_blocks_prev = 3600\n"
                    "epoch_activation_block = 9133200\nera_settlements = 4\n"
                    "reveal_margin_blocks = 25\n")
    g = _grid_file(tmp_path, LIVE_GRID)
    proc = _run("era-start", "9547200", "--grid", str(g), "--config", str(toml))
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "drift" in proc.stderr.lower(), proc.stderr
    assert proc.stdout.strip() == ""


# ── invalid input ────────────────────────────────────────────────────────────

def _no_segments() -> dict:
    g = json.loads(json.dumps(LIVE_GRID))
    del g["segments"]
    return g


def _wrong_schema() -> dict:
    g = json.loads(json.dumps(LIVE_GRID))
    g["schema"] = "laned_grid/0"
    return g


def _segments_not_from_zero() -> dict:
    g = json.loads(json.dumps(LIVE_GRID))
    g["segments"] = g["segments"][1:]
    return g


def _string_epoch() -> dict:
    g = json.loads(json.dumps(LIVE_GRID))
    g["round"]["epoch_blocks"] = "900"
    return g


@pytest.mark.parametrize("grid_fn", [_no_segments, _wrong_schema, _segments_not_from_zero, _string_epoch])
def test_invalid_grid_fixture_exits_2(tmp_path, grid_fn):
    g = _grid_file(tmp_path, grid_fn())
    proc = _run("era-start", "9547200", "--grid", str(g))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == ""


def test_missing_grid_exits_2():
    proc = _run("era-start", "9547200")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)


def test_negative_block_exits_2(tmp_path):
    g = _grid_file(tmp_path, LIVE_GRID)
    proc = _run("era-start", "-5", "--grid", str(g))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)


# ── entry point ──────────────────────────────────────────────────────────────

def test_pyproject_declares_cascade_laned_entry_point():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("cascade-laned") == "cascade.laned.cli:main", scripts


# ── the golden vectors are ground truth (passes on main) ─────────────────────

def test_golden_vectors_agree_with_shared_code():
    """Re-derive every vector above from the shared code on main, so a vector
    in this file can never drift from what the validators compute."""
    from cascade.shared import activation as act
    from cascade.shared import era as era_mod
    from cascade.shared import netconfig as nc
    from cascade.shared.chain import blocks_until_boundary_reveal
    from cascade.shared.config import RoundConfig, effective_epoch_blocks

    r = LIVE_GRID["round"]
    rc = RoundConfig(epoch_blocks=r["epoch_blocks"], epoch_blocks_prev=r["epoch_blocks_prev"],
                     epoch_activation_block=r["epoch_activation_block"],
                     era_settlements=r["era_settlements"])
    cfg = SimpleNamespace(round=rc)
    for block, index, start, is_start, eb, era in ERA_START_VECTORS:
        assert era_mod.era_index_of(rc, block) == index
        assert era_mod.era_start_block(rc, block) == start
        assert nc.is_era_start(cfg, block) is is_start
        assert effective_epoch_blocks(rc, block) == eb
        assert era_mod.era_length_blocks(rc, block) == era
    for block, nxt, idx in NEXT_ERA_VECTORS:
        assert era_mod.next_era_start(rc, block) == nxt
        assert era_mod.era_index_of(rc, nxt) == idx
    for boundary, idx, s_era, s_start, is_start in SETTLEMENT_VECTORS:
        spec = era_mod.settlement_era(rc, boundary)
        assert (era_mod.era_index_of(rc, boundary), spec.index, spec.start_block) == (idx, s_era, s_start)
        assert nc.is_era_start(cfg, boundary) is is_start
    for lock, align, a in ACTIVATION_VECTORS:
        assert act.activation_block_for(rc, lock, align) == a
    for eff, lock, apply in EFFECTIVE_VECTORS:
        assert nc.is_era_start(cfg, eff)
        rec = act.ActivationRecord(lock_block=lock)
        assert nc.apply_block_for({"effective_block": eff}, rec, rc) == apply
    assert not nc.is_era_start(cfg, 9562500)
    for block, e, eff_block in MIN_EFFECTIVE_VECTORS:
        assert era_mod.min_effective_era(rc, block) == e
        assert e * era_mod.era_length_blocks(rc, block) == eff_block
    for check, fire, eff, at_fire, fires in FIRE_VECTORS:
        assert era_mod.min_effective_era(rc, check) == eff
        assert era_mod.min_effective_era(rc, fire) == at_fire
        assert (eff >= at_fire) is fires
    for block, extra, margin, boundary, delay, reveal in REVEAL_VECTORS:
        eb = effective_epoch_blocks(rc, block)
        d = blocks_until_boundary_reveal(block, eb, margin, next_epoch="--next-epoch" in extra)
        assert d == delay and block + d == reveal
        assert ((block + d) // eb + 1) * eb == boundary
