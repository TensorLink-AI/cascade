"""R7: the S0c overlay bodies — benign seq-1, deliberately refused, kill switch.

Plan: LANE_D_PLAN.md §3.0 row R7 ("generated bodies pass validate_overlay on a
fixture base; the refused body is refused whole"), §5 row 4 (S0c), §5 "Kill
switch and latency", §14.1, §14.4 item 2, T21 / T27 (CI half).
Contract: tests/accept/r7/README.md.
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    AFTER,
    ARM,
    ARM_EFFECTIVE,
    GRIDS,
    MAINNET,
    MAINNET_GRID,
    SETTLEMENT_MAIN,
    TESTNET,
    armed_mainnet_text,
    build_body,
    changed_fields,
    copy_base,
    generate,
    identity,
    load,
    nc,
    next_era_start,
    offending,
    refusal,
    run_overlay,
    strip,
    validate,
    write_grid,
)

BASES = {
    "mainnet": (MAINNET, None),
    "testnet": (TESTNET, None),
    "mainnet-armed": (MAINNET, "armed"),
}


def _base(tmp_path, which: str):
    name, variant = BASES[which]
    text = armed_mainnet_text() if variant == "armed" else None
    base = copy_base(tmp_path, name, text)
    grid = write_grid(tmp_path, GRIDS[name])
    return name, base, grid


def _shape(body: dict, cfg, eff: int, seq: int, prev: str | None) -> None:
    m = nc()
    assert set(body) >= m.BODY_FIELDS, f"body lacks {sorted(m.BODY_FIELDS - set(body))}"
    assert not set(body) - m.BODY_FIELDS - m.OPTIONAL_FIELDS, (
        f"body has unknown fields {sorted(set(body) - m.BODY_FIELDS - m.OPTIONAL_FIELDS)}")
    assert body["protocol"] == m.PROTOCOL
    assert body["network"] == identity(cfg), (body["network"], identity(cfg))
    assert body["seq"] == seq, f"seq {body['seq']!r} != {seq}"
    assert body["prev"] == prev, f"prev {body['prev']!r} != {prev!r}"
    assert body["effective_block"] == eff, (
        f"effective_block {body['effective_block']!r} != {eff} (R1's next-era-start)")


def _summary(summary: dict, body: dict, eff: int) -> None:
    assert summary.get("effective_block") == eff, summary
    assert summary.get("seq") == body["seq"], summary
    assert summary.get("digest") == nc().body_digest(body), (
        f"stdout digest {summary.get('digest')!r} != body_digest(body.json)")


def _same(a, b) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


# ── benign seq-1 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", sorted(BASES))
def test_r7_benign_body_restates_base_and_validates(tmp_path, which):
    name, base, grid = _base(tmp_path, which)
    eff = next_era_start(grid, base, AFTER[name])
    summary, body, overrides = generate(tmp_path, "benign", base, grid, AFTER[name],
                                        tmp_path / "out")
    cfg = load(base)
    _shape(body, cfg, eff, 1, None)
    _summary(summary, body, eff)
    assert body["overrides"] == overrides, "overrides.toml != body.json overrides"

    ov = body["overrides"]
    assert ov and all(isinstance(kv, dict) and kv for kv in ov.values()), (
        f"the benign body restates nothing: {ov!r}")
    assert offending(ov) == [], f"benign body carries refused keys {offending(ov)}"
    raw = cfg.raw or {}
    for table, kv in ov.items():
        assert table in nc().ALLOWED, f"[{table}] is not overlay-settable"
        for key, value in kv.items():
            assert key in (raw.get(table) or {}), (
                f"[{table}] {key} is not typed in the base: a benign body restates only")
            assert _same(value, raw[table][key]), (
                f"[{table}] {key} = {value!r} but the base has {raw[table][key]!r}")

    built = validate(cfg, body, f"benign body on {which}")
    assert changed_fields(cfg, built) == [], (
        f"the benign body changes parsed values: {changed_fields(cfg, built)}")
    plan = nc().plan_overlay(cfg, ov, eff, None)
    assert plan.ok, plan.render()
    assert plan.diff == [], f"benign body has a diff vs the base: {plan.diff}"


def test_r7_benign_refuses_base_drifted_from_grid(tmp_path):
    """The effective block comes from R1's tool with --config: a base whose
    [round] differs from the live grid is refused, nothing written."""
    base = copy_base(tmp_path, TESTNET)
    grid = write_grid(tmp_path, MAINNET_GRID)
    out = tmp_path / "out"
    proc = run_overlay(tmp_path, "benign", "--config", str(base), "--grid", str(grid),
                       "--after", str(AFTER[MAINNET]), "--out", str(out))
    assert proc.returncode != 0, f"drifted base accepted: exit 0\n{proc.stdout}"
    assert not (out / "body.json").exists(), "a refused run wrote body.json"
    assert not (out / "overrides.toml").exists(), "a refused run wrote overrides.toml"


# ── the deliberately refused body ────────────────────────────────────────────


@pytest.mark.parametrize("which", ["mainnet", "testnet"])
def test_r7_refused_body_is_refused_whole(tmp_path, which):
    name, base, grid = _base(tmp_path, which)
    eff = next_era_start(grid, base, AFTER[name])
    summary, body, overrides = generate(tmp_path, "refused", base, grid, AFTER[name],
                                        tmp_path / "out")
    cfg = load(base)
    _shape(body, cfg, eff, 1, None)
    _summary(summary, body, eff)
    assert body["overrides"] == overrides, "overrides.toml != body.json overrides"
    assert "release-only" in str(summary.get("refusal", "")), (
        f"stdout names no release-only refusal: {summary!r}")

    m = nc()
    why = refusal(m.validate_overlay, cfg, body)
    assert why is not None, "validate_overlay ACCEPTED the refused body"
    assert "release-only" in why, f"refused for the wrong reason: {why}"
    assert refusal(m.check_overrides, body["overrides"]) is not None
    assert refusal(m.build_config, cfg, body) is not None, "build_config applied it"
    plan = m.plan_overlay(cfg, body["overrides"], eff, None)
    assert not plan.ok, "plan_overlay calls the refused body publishable"

    bad = offending(body["overrides"])
    assert bad, "no single entry is refused: the refusal is not a release-only key or table"
    rest = strip(body["overrides"], bad)
    assert rest, ("the refused body carries nothing valid, so 'refused whole' is not shown: "
                  "it must also restate valid keys")
    good = validate(cfg, build_body(cfg, rest, eff), "the valid part of the refused body")
    assert changed_fields(cfg, good) == [], (
        f"the valid part of the refused body is not a restatement: {changed_fields(cfg, good)}")


# ── the kill switch (T21 / T27, CI half) ─────────────────────────────────────


def _write(path, body: dict):
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("which", ["mainnet", "mainnet-armed"])
def test_r7_kill_body_ends_lane_d_only(tmp_path, which):
    name, base, grid = _base(tmp_path, which)
    cfg = load(base)
    arm = {"lane_d": dict(ARM)} if which == "mainnet" else {
        "lane_d": {"from_block": ARM["from_block"]}}
    previous = build_body(cfg, arm, ARM_EFFECTIVE)
    prev_cfg = validate(cfg, previous, "fixture: the seq-1 activation overlay")
    prev_path = _write(tmp_path / "previous.json", previous)

    eff = next_era_start(grid, base, AFTER[name])
    summary, body, overrides = generate(tmp_path, "kill", base, grid, AFTER[name],
                                        tmp_path / "out", "--previous", str(prev_path))
    _shape(body, cfg, eff, 2, nc().body_digest(previous))
    _summary(summary, body, eff)
    assert body["overrides"] == overrides, "overrides.toml != body.json overrides"

    ov, pov = body["overrides"], previous["overrides"]
    assert set(ov) == set(pov), f"tables {sorted(ov)} != the previous overlay's {sorted(pov)}"
    for table in pov:
        if table != "lane_d":
            assert _same(ov[table], pov[table]), f"[{table}] not restated byte-identically"
    for key, value in pov["lane_d"].items():
        if key not in ("until_block", "share"):
            assert key in ov["lane_d"] and _same(ov["lane_d"][key], value), (
                f"[lane_d] {key} not restated from the previous overlay")
    extra = set(ov["lane_d"]) - set(pov["lane_d"])
    assert extra <= {"until_block", "share"}, f"kill body adds [lane_d] {sorted(extra)}"

    killed = validate(cfg, body, "the kill-switch body")
    ld = killed.lane_d
    assert ld.until_block == eff or ld.share == 0.0, (
        f"kill body neither ends lane D at {eff} (until_block={ld.until_block}) nor sets "
        f"share = 0 (share={ld.share})")
    assert changed_fields(prev_cfg, killed) == ["lane_d"], (
        f"the kill body changes more than [lane_d]: {changed_fields(prev_cfg, killed)}")

    first = eff + SETTLEMENT_MAIN   # first settlement of the stated era
    assert prev_cfg.lane_d.active(first) and prev_cfg.lane_d.share > 0, "fixture not armed"
    assert not (ld.active(first) and ld.share > 0), (
        f"lane D still paid at settlement {first} after the kill")

    plan = nc().plan_overlay(cfg, ov, eff, previous)
    assert plan.ok, plan.render()


def test_r7_kill_refuses_when_lane_d_is_not_armed(tmp_path):
    base = copy_base(tmp_path, MAINNET)
    grid = write_grid(tmp_path, MAINNET_GRID)
    cfg = load(base)
    previous = build_body(cfg, {"lane_d": {"share": 0.0}}, ARM_EFFECTIVE)
    validate(cfg, previous, "fixture: an unarmed seq-1 restatement")
    prev_path = _write(tmp_path / "previous.json", previous)
    out = tmp_path / "out"
    proc = run_overlay(tmp_path, "kill", "--config", str(base), "--grid", str(grid),
                       "--after", str(AFTER[MAINNET]), "--previous", str(prev_path),
                       "--out", str(out))
    assert proc.returncode != 0, f"kill of an unarmed lane D accepted\n{proc.stdout}"
    assert not (out / "body.json").exists(), "a refused kill wrote body.json"


# ── fixture sanity (passes on main: proves the fixtures, not R7) ─────────────


@pytest.mark.parametrize("which", sorted(BASES))
def test_r7_fixtures_are_valid_on_main(tmp_path, which):
    name, base, grid = _base(tmp_path, which)
    cfg = load(base)
    eff = next_era_start(grid, base, AFTER[name])
    assert nc().is_era_start(cfg, eff)
    restate = {"lane_d": {"from_block": cfg.raw["lane_d"]["from_block"],
                          "share": cfg.raw["lane_d"]["share"]}}
    built = validate(cfg, build_body(cfg, restate, eff), "fixture restatement")
    assert changed_fields(cfg, built) == []
    if name == MAINNET:
        prev = build_body(cfg, {"lane_d": dict(ARM)}, ARM_EFFECTIVE)
        prev_cfg = validate(cfg, prev, "fixture arm")
        kill = build_body(cfg, {"lane_d": {**ARM, "until_block": eff}}, eff, prev)
        killed = validate(cfg, kill, "fixture kill")
        assert changed_fields(prev_cfg, killed) == ["lane_d"]
        assert not killed.lane_d.active(eff + SETTLEMENT_MAIN)
