"""Containment (owner 2026-11-24, launch release): after this one validator
release, anything testnet finds in lanes B / C / D must be fixable or
containable by a signed overlay — no second validator update.

1. Lane C / MarketCast's typed launch is movable LATER by overlay (never
   earlier than typed, never into the past, never once started); lane C has
   an overlay-settable ``until_block`` kill switch that also stops MarketCast.
2. A satellite's lane B / lane C check failure burns that lane's share only;
   lane A is byte-identical when nothing fails.
3. The main validator's lane D step is always built and judges each boundary
   under the config in force there (an overlay-armed lane D settles).
4. ``[lane_b] until_block`` kill switch.
6. Overlay refresh signals the newest overlay it can validate.

Every new key: a W-OV bound + a dangerous-value test + a loader round trip.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .helpers import ROOT, SHIPPED, refusal, shipped_cfg, shipped_raw

LAUNCH = 9586800        # the shipped [lane_c] / [marketcast] from_block
ERA = 3600              # 900-block settlements x 4
EFF = LAUNCH - 2 * ERA  # an era start ahead of the launch


def _overlay(cfg, overrides, eff=EFF):
    from cascade.shared import netconfig as nc

    try:
        return nc.validate_overlay(cfg, nc.next_overlay(overrides, eff, None,
                                                        network=nc.network_identity(cfg)))
    except nc.NetConfigError as e:
        raise AssertionError(f"overlay refused: {e}") from None


def _built(fn, *a, **kw):
    """``fn(*a, **kw)``; a TypeError / AttributeError (the containment API
    is absent) is an assertion failure, not a crash."""
    try:
        return fn(*a, **kw)
    except (TypeError, AttributeError) as e:
        raise AssertionError(f"containment API missing: {e}") from None


def _error(fn, *a, **kw) -> str:
    """The ValueError text ``fn`` raises ("" = it did not raise)."""
    try:
        fn(*a, **kw)
    except ValueError as e:
        return str(e)
    return ""


# ── 1. lane C / MarketCast launch: overlay-gated, later only ────────────────


def test_shipped_launch_is_typed_and_the_kill_switches_default_off():
    cfg = shipped_cfg()
    assert cfg.lane_c.from_block == LAUNCH and cfg.marketcast.from_block == LAUNCH
    for sec in (cfg.lane_c, cfg.marketcast, cfg.lane_b, cfg.lane_d):
        assert getattr(sec, "until_block", None) == 0, sec


@pytest.mark.parametrize("overrides", [
    {"marketcast": {"from_block": LAUNCH + ERA}},                  # was "not an overlay key"
    {"lane_c": {"from_block": LAUNCH + ERA}},                      # was "marketcast >= lane_c"
    {"lane_c": {"from_block": LAUNCH + ERA}, "marketcast": {"from_block": LAUNCH + ERA}},
])
def test_each_postpone_path_the_audit_found_refused_now_validates(overrides):
    out = _overlay(shipped_cfg(), overrides)
    assert out.marketcast.from_block == LAUNCH + ERA
    # MarketCast never judges before its (postponed) start
    assert not out.marketcast.active(LAUNCH)
    if "lane_c" in overrides:
        # postponing lane C alone carries MarketCast's start along, and the
        # emission split at the launch settlement stays lane A's
        assert out.lane_c.from_block == LAUNCH + ERA
        assert not out.lane_c.active(LAUNCH + 900)


def test_postponed_lane_c_returns_its_share_to_lane_a():
    from cascade.shared.lane_c import lane_c_shares

    base = shipped_cfg()
    out = _overlay(base, {"lane_c": {"from_block": LAUNCH + ERA}})
    at = LAUNCH + 900
    assert lane_c_shares(base.lane_b, base.lane_c, at)[1] > 0.0
    assert lane_c_shares(out.lane_b, out.lane_c, at) == (0.0, 0.0)


@pytest.mark.parametrize("key", ["lane_c", "marketcast"])
def test_start_is_never_moved_earlier_than_typed(key):
    why = refusal(shipped_cfg(), {key: {"from_block": LAUNCH - ERA}}, EFF)
    assert why and "earlier than the typed launch" in why


def test_start_is_never_moved_into_the_past_or_once_started():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    later = _overlay(base, {"lane_c": {"from_block": LAUNCH + 4 * ERA}})
    # into the past: a start before the block the overlay applies at
    assert any("lies before" in w for w in nc.retroactive_changes(base, later, LAUNCH + 5 * ERA))
    # once started (applies after the typed launch): refused — end it with
    # until_block instead
    why = nc.retroactive_changes(base, later, LAUNCH + ERA)
    assert any("started before" in w for w in why)
    # applying before the launch it postpones: accepted
    assert nc.retroactive_changes(base, later, EFF) == []


def test_publish_tool_refuses_a_start_less_than_one_era_past_the_head():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    plan = nc.plan_overlay(base, {"lane_c": {"from_block": LAUNCH + ERA}}, EFF, None,
                           head_block=LAUNCH + 1)
    assert not plan.ok and any("less than one era" in e for e in plan.errors)
    plan = nc.plan_overlay(base, {"lane_c": {"from_block": LAUNCH + ERA}}, EFF, None,
                           head_block=EFF - ERA)
    assert plan.ok, plan.render()


def test_lane_c_until_block_kills_lane_c_and_marketcast():
    from cascade.shared.lane_c import lane_c_shares

    until = LAUNCH + 4 * ERA
    out = _overlay(shipped_cfg(), {"lane_c": {"until_block": until}})
    assert getattr(out.lane_c, "until_block", None) == until
    assert getattr(out.marketcast, "until_block", None) == until
    assert out.lane_c.active(until) and not out.lane_c.active(until + 900)
    assert out.marketcast.active(until) and not out.marketcast.active(until + 900)
    assert out.lane_c.era_active(until - ERA) and not out.lane_c.era_active(until)
    assert lane_c_shares(out.lane_b, out.lane_c, until + 900) == (0.0, 0.0)


# ── 4. lane B kill switch ───────────────────────────────────────────────────


def test_lane_b_until_block_kills_lane_b():
    from cascade.shared.config import parse_chain_config

    raw = shipped_raw("chain.toml")
    raw["lane_b"]["from_block"] = LAUNCH
    raw["lane_b"]["until_block"] = LAUNCH + 4 * ERA
    cfg = parse_chain_config(raw)
    lb = cfg.lane_b
    assert lb.active(LAUNCH + 900) and lb.active(LAUNCH + 4 * ERA)
    assert not lb.active(LAUNCH + 4 * ERA + 900)
    # by overlay, on the overlay-armed lane B of the launch step
    base = shipped_cfg()
    out = _overlay(base, {"lane_b": {"from_block": LAUNCH, "until_block": LAUNCH + 4 * ERA}})
    assert getattr(out.lane_b, "until_block", None) == LAUNCH + 4 * ERA


# ── every new key: W-OV bound, dangerous values, loader round trip ──────────

NEW_KEYS = [("lane_b", "until_block"), ("lane_c", "until_block"),
            ("lane_c", "from_block"), ("marketcast", "from_block"), ("lane_d", "until_block")]


@pytest.mark.parametrize("table, key", NEW_KEYS)
def test_new_block_keys_carry_a_w_ov_bound(table, key):
    from cascade.shared import netconfig_bounds as nb

    b = nb.BOUNDS.get((table, key))
    assert b is not None and b.integer and b.lo == 0 and b.hi == 2**32 - 1
    assert f"{table}.{key}" in nb.policy()["bounds"]


@pytest.mark.parametrize("table, key", NEW_KEYS)
@pytest.mark.parametrize("value", [-3600, 2**32, 9586800.0, True, "9586800"])
def test_dangerous_block_values_are_refused(table, key, value):
    from cascade.shared import netconfig_bounds as nb

    assert nb.value_refusals({table: {key: value}})
    why = refusal(shipped_cfg(), {table: {key: value}}, EFF)
    assert why, (table, key, value)


@pytest.mark.parametrize("table", ["lane_b", "lane_c"])
@pytest.mark.parametrize("value", [LAUNCH + 7, LAUNCH])
def test_until_block_off_the_era_grid_or_not_after_the_start_is_refused(table, value):
    from cascade.shared.config import parse_chain_config

    raw = shipped_raw("chain.toml")
    raw["lane_b"]["from_block"] = LAUNCH
    raw[table]["until_block"] = value
    assert "until_block" in _error(parse_chain_config, raw)


@pytest.mark.parametrize("table", ["lane_b", "lane_c", "lane_d"])
def test_until_block_never_crosses_the_apply_block(table):
    from cascade.shared import netconfig as nc
    from cascade.shared.config import parse_chain_config

    raw = shipped_raw("chain.toml")
    raw["lane_b"]["from_block"] = LAUNCH
    raw["lane_d"]["from_block"] = LAUNCH
    prev = parse_chain_config(raw)
    raw2 = copy.deepcopy(raw)
    raw2[table]["until_block"] = LAUNCH + ERA
    new = parse_chain_config(raw2)
    assert any("crosses a block" in w
               for w in nc.retroactive_changes(prev, new, LAUNCH + 2 * ERA))
    assert not any("crosses a block" in w for w in nc.retroactive_changes(prev, new, EFF))


@pytest.mark.parametrize("name", SHIPPED)
def test_loader_round_trip_of_the_new_keys(name, tmp_path):
    import tomllib

    from cascade.shared.config import load_chain_config

    raw = shipped_raw(name)
    v2 = int(load_chain_config(ROOT / name).v2.from_block or 0)
    start = ((max(v2, LAUNCH) + ERA - 1) // ERA) * ERA
    raw.setdefault("lane_b", {})["from_block"] = start
    raw["lane_b"]["until_block"] = start + 8 * ERA
    raw.setdefault("lane_c", {})["until_block"] = (
        max(int(raw["lane_c"].get("from_block", 0) or 0), start) + 8 * ERA)
    text = _dump_toml(raw)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    assert tomllib.loads(text)["lane_b"]["until_block"] == start + 8 * ERA
    cfg = load_chain_config(p)
    assert getattr(cfg.lane_b, "until_block", None) == start + 8 * ERA
    assert getattr(cfg.lane_c, "until_block", None) == raw["lane_c"]["until_block"]
    assert getattr(cfg.marketcast, "until_block", None) == raw["lane_c"]["until_block"]


def _dump_toml(raw: dict) -> str:
    """A minimal TOML writer for the shipped tables (scalars, lists, nested)."""
    import json

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{json.dumps(k)} = {val(x)}" for k, x in v.items()) + "}"
        raise TypeError(v)

    out: list[str] = []

    def table(path: list[str], t: dict) -> None:
        scalars = {k: v for k, v in t.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in t.items() if isinstance(v, dict)}
        if path:
            out.append("[" + ".".join(json.dumps(p) for p in path) + "]")
        for k, v in scalars.items():
            out.append(f"{json.dumps(k)} = {val(v)}")
        for k, v in subs.items():
            table(path + [k], v)

    table([], raw)
    return "\n".join(out) + "\n"


# ── 2. satellite: lane B / C failures burn that lane's share only ───────────


def test_derive_weights_burn_lanes_default_is_byte_identical(tmp_path):
    from cascade.audit import checks

    sig = __import__("inspect").signature(checks.derive_weights)
    p = sig.parameters.get("burn_lanes")
    assert p is not None and p.default == frozenset()
    # a lane not active at the settlement is ignored (pre-launch receipts)
    cfg = shipped_cfg()
    rc = SimpleNamespace(epoch_start_block=LAUNCH - ERA, weights=(0.5, 0.5),
                         emission=None, lane_c_pay=None, lane_b_pay=None,
                         reward_uids=(1,))
    assert (checks.derive_weights(rc, cfg, burn_lanes=frozenset({"b", "c"}))
            == checks.derive_weights(rc, cfg))


def test_lane_rows_are_scoped_and_lane_a_rows_are_not():
    from cascade.validator import satellite as s

    for name in ("lane_b",):
        assert s._lane_of_row(name) == "b"
    for name in ("lane_c", "lane_c_chain", "marketcast", "marketcast:fx"):
        assert s._lane_of_row(name) == "c"
    for name in ("weights", "sentinel", "netconfig", "verdict", "king_check",
                 "continuity", "lane_d"):
        assert s._lane_of_row(name) is None


def test_weights_envelope_attributes_only_a_lane_sized_move_off_burn():
    from cascade.validator.satellite import within_lane_envelope

    burned = [0.4, 0.2, 0.0, 0.4]          # lane B's 0.35 + lane A's burn on uid 3
    assert within_lane_envelope([0.4, 0.2, 0.35, 0.05], burned, burn_uid=3, share=0.35)
    assert not within_lane_envelope([0.4, 0.2, 0.36, 0.04], burned, burn_uid=3, share=0.35)
    # a lane A uid paid LESS than derived is never a lane B payout fault
    assert not within_lane_envelope([0.3, 0.3, 0.0, 0.4], burned, burn_uid=3, share=0.35)


def _with_lane_b(lb_cfg, t, mult_bump: float, extra=None):
    """The unit suite's lane B payout tamper (tests/unit/test_satellite_replay):
    the recorded vector pays the slot 1.0, the record claims ``mult_bump``."""
    from cascade.eval.expert_lane import LaneBWin
    from cascade.shared.emission import lane_parts
    from cascade.shared.era import era_index_of, era_length_blocks

    domain = lb_cfg.lane_b.domains[0]

    def tamper(rc):
        b = int(rc.epoch_start_block)
        eb = int(era_length_blocks(lb_cfg.round, b))
        merge = t.BOUNDARIES[0] - t.GRID
        age = max(0, (b - merge) // eb)
        a, bb = lane_parts(list(rc.reward_uids), len(rc.weights),
                           king_decay=lb_cfg.scoring.king_decay,
                           burn_uid=lb_cfg.scoring.burn_uid, lane_b_cfg=lb_cfg.lane_b,
                           wins=[LaneBWin(uid=2, domain=domain, multiplier=1.0)])
        pay = {"block": b, "era_blocks": eb,
               "wins": [{"domain": domain, "hotkey": "x2_hk", "uid": 2, "merge_block": merge,
                         "age": age, "since_era": era_index_of(lb_cfg.round, merge),
                         "multiplier": mult_bump, "gain_lcb": 0.03}]}
        post = dict(rc.state_post, lane_b={domain: ["x2_hk", 2, merge, "r0", 0.03, None,
                                                    None]})
        out = replace(rc, lane_b=(), lane_b_pay=pay, state_post=post,
                      weights=tuple(x + y for x, y in zip(a, bb, strict=True)))
        return extra(out) if extra is not None else out
    return tamper


def test_satellite_lane_b_fault_burns_lane_b_only_and_a_lane_a_fault_burns_all(tmp_path,
                                                                             monkeypatch):
    pytest.importorskip("bittensor.sp_core")
    import cascade.audit.main as audit_main

    def _offline(cfg, key):
        raise RuntimeError("offline")

    monkeypatch.setattr(audit_main, "_fetch_text", _offline)     # no funded-roster read
    import json

    from cascade.audit.checks import derive_weights
    from cascade.shared.hippius import receipt_round_key
    from tests.unit import test_satellite_replay as t

    from .helpers import fixture_raw

    scfg = t._keyed(SimpleNamespace(raw=fixture_raw(tmp_path)))
    lb_cfg = replace(scfg, lane_b=replace(scfg.lane_b, from_block=t.BOUNDARIES[0] - t.GRID))
    store = t.DictStore()
    _, l0 = t.make_settlement(lb_cfg, store, 0, tamper=_with_lane_b(lb_cfg, t, 1.0))
    l0["state_post"] = json.loads(store.get_text(
        receipt_round_key(l0["round_id"], t.MAIN_KP.ss58_address)))["state_post"]
    runner = t._runner(lb_cfg, store, tmp_path)
    client = t.FakeClient(t.BOUNDARIES[0] + 300)
    assert runner.poll(client)["last_failing_check"] == ""
    # a lane B payout fault: lane B's share burns, lane A stays paid
    r1, l1 = t.make_settlement(lb_cfg, store, 1, prev=l0, tamper=_with_lane_b(lb_cfg, t, 0.5))
    l1["state_post"] = json.loads(store.get_text(
        receipt_round_key(l1["round_id"], t.MAIN_KP.ss58_address)))["state_post"]
    client.block = t.BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "" and client.pushed[-1] != t._burn(lb_cfg), doc
    got = client.pushed[-1]
    assert got[2] == 0.0 and sum(got) == pytest.approx(1.0)
    assert list(got) == list(_built(derive_weights, r1, lb_cfg,
                                    burn_lanes=frozenset({"b"}))[0])
    # a lane A fault (weight moved off a lane A uid) with lane B live: the
    # whole vector burns, exactly as before
    t.make_settlement(lb_cfg, store, 2, prev=l1,
                      tamper=_with_lane_b(lb_cfg, t, 1.0, extra=t._shift_weights))
    client.block = t.BOUNDARIES[2] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "weights" and client.pushed[-1] == t._burn(lb_cfg)


# ── 3. lane D: the step is always built and follows the config in force ────


def _armed(cfg, block=LAUNCH):
    return replace(cfg, lane_d=replace(cfg.lane_d, from_block=block, share=0.5,
                                       takes_from="lane_a"))


def test_unarmed_startup_builds_an_inert_step_that_an_overlay_arms(tmp_path):
    from cascade.validator.lane_d_wiring import LaneDWiringError, build_lane_d_step

    base = shipped_cfg()
    assert base.lane_d.from_block == 0
    kw = dict(state_dir=tmp_path, signer=None, chain_factory=lambda: None, store=None)
    assert build_lane_d_step(base, **kw) is None                 # unchanged default
    in_force = {"cfg": base}
    step = _built(build_lane_d_step, base, **kw, always=True,
                  cfg_at=lambda b: in_force["cfg"])
    assert step is not None
    assert step(LAUNCH + 900) is None                            # unarmed: inert
    in_force["cfg"] = _armed(base)
    # armed by the config in force: it proceeds and fails CLOSED (no signer),
    # never "not armed" and never a crash of the caller's thread contract
    with pytest.raises(LaneDWiringError) as e:
        step(LAUNCH + 900)
    assert e.value.code == "signer_unavailable"


def test_unarmed_startup_builds_inputs_that_refuse_until_armed(tmp_path):
    from cascade.validator.lane_d_inputs import LaneDInputsError, build_lane_d_inputs

    base = shipped_cfg()
    kw = dict(state_dir=tmp_path, intake_state_dir=tmp_path, chain_factory=lambda: None,
              pool=lambda b: [], eval_host=None, house_pointer=tmp_path / "house",
              house_signer_ss58="", readiness=lambda: b"", t7=lambda: b"")
    assert build_lane_d_inputs(base, **kw) is None
    src = _built(build_lane_d_inputs, base, **kw, always=True, cfg_at=lambda b: base)
    with pytest.raises(LaneDInputsError) as e:
        src(LAUNCH + 900)
    assert e.value.code == "not_active"


def test_loop_tick_judges_lane_d_under_the_adopted_overlay():
    from cascade.validator.loop import ValidatorRunner

    base = shipped_cfg()
    armed = _armed(base)
    calls: list[int] = []

    class Step:
        def __call__(self, b):
            calls.append(int(b))

    def fake(history, cfg_at):
        return SimpleNamespace(
            lane_d_step=Step(), cfg=base, _lane_d_worker=None,
            netconfig=SimpleNamespace(enabled=True, state=SimpleNamespace(history=history)),
            _reassert_boundary=lambda cur: cur // 900 * 900, cfg_at=cfg_at)

    client = SimpleNamespace(current_block=lambda: LAUNCH + 950)
    reads = []
    me = fake([], lambda b: reads.append(b) or armed)
    ValidatorRunner._lane_d_tick(me, client)                 # nothing adopted: inert
    assert me._lane_d_worker is None and reads == []
    me = fake(["adopted"], lambda b: armed)
    ValidatorRunner._lane_d_tick(me, client)
    assert me._lane_d_worker is not None
    assert me._lane_d_worker.join(5) is not False
    assert calls == [LAUNCH + 900]


def test_validator_main_always_wires_the_step_with_cfg_at():
    tree = ast.parse((ROOT / "cascade/validator/main.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") in ("build_lane_d_step", "build_lane_d_inputs")]
    assert {c.func.id for c in calls} == {"build_lane_d_step", "build_lane_d_inputs"}
    for c in calls:
        kws = {k.arg: k.value for k in c.keywords}
        assert isinstance(kws.get("always"), ast.Constant) and kws["always"].value is True
        assert ast.unparse(kws["cfg_at"]) == "runner.cfg_at"


# ── 6. refresh: the newest overlay this node can validate ───────────────────


def test_a_refused_newer_overlay_never_drops_the_signalled_one():
    pytest.importorskip("bittensor.sp_core")
    from cascade.shared import netconfig as nc
    from tests.unit.test_netconfig_core import E1, E2, E3, MemStore, base_cfg, publish

    cfg = base_cfg()
    store = MemStore()
    one = publish(store, {"scoring": {"min_windows": 7}}, E1)
    r = nc.NetConfigResolver(cfg, store=store, signal=True)
    assert r.refresh(cfg, force=True) and r.candidate_body() == one
    # step 2 refused by the envelope (lane B share above its bound)
    two = publish(store, {"lane_b": {"share": 0.9}}, E2, one)
    assert r.refresh(cfg, force=True)
    assert nc.body_digest(two) in r.state.refused
    assert r.candidate_body() == one                        # still signalling step 1
    assert r.segment().feature == nc.cfg_feature_name(one)
    # two new bodies at once, the newest refused: the older valid one wins
    three = publish(store, {"scoring": {"min_windows": 8}}, E3, two)
    four = publish(store, {"lane_b": {"share": 0.95}}, E3, three)
    assert r.refresh(cfg, force=True)
    assert r.candidate_body() == three and nc.body_digest(four) in r.state.refused
