"""W-OV #891 (b274626d testnet rollout, 2026-11-24): an overlay already IN
FORCE (adopted / locked in under an older release) whose values fall outside
THIS release's envelope must never abort round handling. Every node on the
release resolves it the same way: each out-of-bounds key takes the RELEASE
(chain.toml) value, the rest of the overlay stands, a loud line is logged
once per key and the status surfaces list it. A NEW overlay with the same
value is still refused for adoption. With no key out of bounds the resolved
config is identical to the strict build (lane A byte-identical)."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .helpers import shipped_cfg

LAUNCH = 9586800
ERA = 3600
APPLY = LAUNCH - ERA                 # an era start: the overlay is in force here
# the testnet seq 4 shape: lane D armed, [lane_d] min_windows 40 (envelope [100, 20000])
ARMED = {"lane_d": {"from_block": LAUNCH, "share": 0.5, "takes_from": "lane_a"},
         "lane_b": {"share": 0.35}, "lane_c": {"share": 0.15, "takes_from": "lane_a"}}


def _oob_overrides() -> dict:
    ov = json.loads(json.dumps(ARMED))
    ov["lane_d"]["min_windows"] = 40
    ov["scoring"] = {"king_decay": 0.7}          # an in-bounds key that must survive
    return ov


def _body(base, overrides, comment=""):
    """An overlay body as an OLDER release would have accepted it (no W-OV
    envelope): the shape check only. ``comment`` keeps each test's digest
    distinct (the loud line is logged once per overlay and key, per process)."""
    from cascade.shared import netconfig as nc

    body = {"protocol": nc.PROTOCOL, "network": nc.network_identity(base), "seq": 1,
            "prev": None, "effective_block": APPLY,
            "overrides": json.loads(json.dumps(overrides))}
    if comment:
        body["comment"] = comment
    return body


def _history(body):
    from cascade.shared import netconfig as nc

    return [nc.Adopted(apply_block=APPLY, seq=int(body["seq"]), digest=nc.body_digest(body),
                       envelope={"body": body})]


def _resolver(base, body, tmp_path):
    from cascade.shared import netconfig as nc

    r = nc.NetConfigResolver(base, state_path=tmp_path / "netconfig_state.json")
    r.state.history = _history(body)
    return r


def test_in_force_oob_overlay_resolves_to_release_value_and_keeps_the_rest(caplog):
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    base = shipped_cfg()
    body = _body(base, _oob_overrides(), "log-once")
    assert any("min_windows" in x for x in nb.value_refusals(body["overrides"]))
    # the strict build (a new overlay being validated) still refuses it
    with pytest.raises(nc.NetConfigError, match="out of bounds"):
        nc.build_config(base, body)
    with pytest.raises(nc.NetConfigError, match="out of bounds"):
        nc.validate_overlay(base, body)
    caplog.set_level(logging.ERROR, logger="cascade.netconfig")
    got = nc.config_from_history(base, _history(body), APPLY + 900)
    assert got is not base
    assert got.lane_d.min_windows == base.lane_d.min_windows == 1000    # release value
    assert int(got.lane_d.from_block) == LAUNCH and got.lane_d.share == 0.5
    assert got.scoring.king_decay == 0.7 != base.scoring.king_decay      # the rest stands
    assert got.netconfig.seq == 1 and got.netconfig.digest == nc.body_digest(body)
    lines = [r.getMessage() for r in caplog.records
             if "overlay key out of current bounds" in r.getMessage()]
    assert len(lines) == 1 and "[lane_d] min_windows = 40" in lines[0]
    assert "using release value 1000" in lines[0]
    # logged ONCE per key: a fresh (uncached) resolution does not repeat it
    caplog.clear()
    again = nc.config_from_history(base, _history(body), APPLY + 1800)
    assert again == got
    assert not [r for r in caplog.records if "out of current bounds" in r.getMessage()]
    rep = nc.out_of_bounds_report()[nc.body_digest(body)]
    assert any("[lane_d] min_windows = 40" in x and "release value 1000" in x for x in rep)


def test_resolver_config_at_and_status_surfaces(tmp_path):
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    body = _body(base, _oob_overrides())
    r = _resolver(base, body, tmp_path)
    assert r.config_at(base, APPLY - 1) is base
    at = r.config_at(base, APPLY + 900)                  # never raises
    assert at.lane_d.min_windows == 1000
    s = r.summary()["out_of_bounds"]
    assert list(s) == [nc.body_digest(body)]
    assert "min_windows" in s[nc.body_digest(body)][0]


def test_satellite_status_lists_the_resolved_keys(tmp_path):
    from cascade.shared import netconfig as nc
    from cascade.validator.satellite import SatelliteRunner

    base = shipped_cfg()
    body = _body(base, _oob_overrides())
    nc.config_from_history(base, _history(body), APPLY)
    doc = SatelliteRunner.status(SimpleNamespace(
        state=SimpleNamespace(verified_round="", verified_boundary=0, failing="",
                              failing_detail="", failing_round=""),
        _push_failing="", hotkey="h", owner="o", _mode="following",
        sat=SimpleNamespace(stale_after_blocks=1)), APPLY)
    assert nc.body_digest(body) in doc["netconfig_out_of_bounds"]


def test_settlement_view_weight_reassert_and_lane_d_tick_proceed(tmp_path):
    """The three loop paths of the rollout traceback (round handling and the
    weight re-assert run under ``_settlement_cfg``; the lane D tick reads
    ``cfg_at``) proceed under an in-force out-of-bounds overlay."""
    from cascade.validator.activation_mixin import ActivationMixin
    from cascade.validator.loop import ValidatorRunner

    base = shipped_cfg()
    body = _body(base, _oob_overrides())
    resolver = _resolver(base, body, tmp_path)

    class R(ActivationMixin):
        pass

    me = R()
    me.cfg, me._cfg_base, me._settlement_block = base, None, None
    me.cascade, me.netconfig = None, resolver
    seen = []
    with me._settlement_cfg(APPLY + 900):                 # _handle_manifest / re-assert
        seen.append(me.cfg)
    assert seen[0].lane_d.min_windows == 1000 and int(seen[0].lane_d.from_block) == LAUNCH
    assert me.cfg is base                                  # restored

    calls: list[int] = []

    class Step:
        def __call__(self, b):
            calls.append(int(b))
            assert me.cfg_at(b).lane_d.active(b)

    tick = SimpleNamespace(lane_d_step=Step(), cfg=base, _lane_d_worker=None,
                           netconfig=resolver, _reassert_boundary=lambda cur: cur // 900 * 900,
                           cfg_at=me.cfg_at)
    ValidatorRunner._lane_d_tick(tick, SimpleNamespace(current_block=lambda: LAUNCH + 950))
    assert tick._lane_d_worker is not None and tick._lane_d_worker.join(5) is not False
    assert calls == [LAUNCH + 900]


def test_audit_replay_resolves_the_identical_config():
    """The audit / satellite replay of a receipt stamping the in-force
    overlay re-derives exactly the validator's resolved config."""
    from cascade.audit.main import apply_receipt_overlay
    from cascade.shared import netconfig as nc

    base = replace(shipped_cfg(), netconfig=replace(shipped_cfg().netconfig, owner_hotkey=""))
    body = _body(base, _oob_overrides())
    validator = nc.config_from_history(base, _history(body), APPLY)

    class Store:
        def get_text(self, key):
            assert key == nc.seq_key(1)
            return json.dumps({"body": body})

    receipt = SimpleNamespace(netconfig_seq=1, netconfig_digest=nc.body_digest(body))
    audit, row = apply_receipt_overlay(receipt, base, store=Store())
    assert row is not None and row.status != "fail", row
    assert audit.lane_d == validator.lane_d and audit.scoring == validator.scoring
    assert audit.netconfig.digest == validator.netconfig.digest
    # the strict path on the same body refuses: only the in-force rule resolves it
    with pytest.raises(nc.NetConfigError):
        nc.build_config(base, body)


def test_in_force_release_only_key_and_clamped_share_resolve_to_release():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    ov = json.loads(json.dumps(ARMED))
    ov["lane_d"]["teacher_ref"] = "someone/else@abc"      # release-only since W-OV
    ov["lane_d"]["share"] = 0.9                           # above its envelope
    body = _body(base, ov)
    with pytest.raises(nc.NetConfigError, match="release-only"):
        nc.check_body(body)
    assert nc.check_body(body, in_force=True) is body
    got = nc.build_config(base, body, applied_block=APPLY, in_force=True)
    assert (got.raw.get("lane_d") or {}).get("teacher_ref") == \
        (base.raw.get("lane_d") or {}).get("teacher_ref")
    assert got.lane_d.share == base.lane_d.share
    assert int(got.lane_d.from_block) == LAUNCH and got.lane_b.share == 0.35


def test_compliant_overlay_in_force_is_identical_to_the_strict_build():
    """Lane A byte-identical: no key out of bounds ⇒ the in-force rule is the
    strict build, value for value."""
    from cascade.shared import netconfig as nc

    for name in ("chain.toml", "chain.testnet.toml"):
        base = shipped_cfg(name)
        body = _body(base, {"scoring": {"king_decay": 0.7}, **ARMED})
        strict = nc.build_config(base, body, applied_block=APPLY)
        assert nc.build_config(base, body, applied_block=APPLY, in_force=True) == strict
        assert nc.config_from_history(base, _history(body), APPLY) == strict
        assert nc.in_force_out_of_bounds(base, body["overrides"]) == []


def test_publish_tool_plans_against_the_resolved_previous():
    """The owner's next overlay on top of an out-of-bounds in-force one: the
    plan warns (``no longer builds``) and judges against the resolved config
    every node runs, so the compliant replacement is publishable."""
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    prev = _body(base, _oob_overrides())
    fix = json.loads(json.dumps(_oob_overrides()))
    fix["lane_d"]["min_windows"] = 100
    plan = nc.plan_overlay(base, fix, APPLY + ERA, prev)
    assert plan.ok, plan.render()
    assert any("no longer builds" in w for w in plan.warnings)
