"""W-CMP2 (DEC-CA-0075): a new competition's spec compiles into an instance
(its kind, targets, competition, table); the trainer dispatches, stamps and
settles its legs under its own table and records its reports under
``reports[<id>]``; its training data is its standard pool clipped to the
recipe's range before the dispatch; and the overlay adds / removes it without
a release, never retroactively."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import EB, KIND, instance_mc, lane_c, mkt_mc, register_kind, unregister_kind

SPEC = {"runner": "marketcast",
        "pool": {"prefix": "pool_energy", "signer": "owner", "schema": "cascade-std-pool/1"},
        "select": {"domain": "energy"}, "metrics": ["returns", "vol"],
        "universe": {"power": ["DE", "FR"]}, "timeframes": {"returns": ["1h"], "vol": ["1h"]},
        "horizons": {"returns": ["1h", "4h"], "vol": ["1h"]}, "from_block": 9590400,
        "weight": 1.0, "channels": {"close": "load"}}


@pytest.fixture(autouse=True)
def _kind():
    yield
    unregister_kind()


def _raw():
    import tomllib

    from .helpers import ROOT

    return tomllib.loads((ROOT / "chain.toml").read_text())


def _with(raw, specs):
    r = copy.deepcopy(raw)
    r.setdefault("lane_c", {})["specs"] = copy.deepcopy(specs)
    return r


def test_w_cmp2_a_spec_compiles_into_an_instance():
    from cascade.shared.config import parse_chain_config
    from cascade.shared.config.lane_c import SPEC_KINDS
    from cascade.shared.lane_c import lane_c_split

    base = parse_chain_config(_raw())
    cfg = parse_chain_config(_with(_raw(), {KIND: SPEC}))
    assert KIND in SPEC_KINDS and not cfg.lane_c.spec_refusals
    assert cfg.lane_c.targets == base.lane_c.targets + (f"{KIND}:market",)
    mc = cfg.lane_c.instance_mc(KIND)
    assert (mc.from_block, mc.pool_prefix, mc.suites, mc.open_classes) == (
        9590400, "pool_energy", ("returns", "vol"), True)
    assert mc.suite_assets("returns") == [("power", "DE"), ("power", "FR")]
    assert (mc.base_ref, mc.base_sha256) == (base.marketcast.base_ref,
                                              base.marketcast.base_sha256)
    assert cfg.marketcast == base.marketcast                 # mkt untouched
    _c, slot = lane_c_split(cfg.lane_c, 9600000, 3600, cfg.marketcast)
    assert slot == {"mkt:market": 0.5, f"{KIND}:market": 0.5}
    # before its start it is pending: its half burns, mkt's half is unchanged
    _c, slot = lane_c_split(cfg.lane_c, 9590400 - 3600, 3600, cfg.marketcast)
    assert slot == {"mkt:market": 0.5}


@pytest.mark.parametrize("bad,code", [
    ({k: v for k, v in SPEC.items() if k != "pool"}, "bad_value"),
    ({**SPEC, "from_block": 9590401}, "bad_value"),                 # not an era boundary
    ({**SPEC, "metrics": ["returns", "sharpe9"]}, "unknown_block"),
    ({**SPEC, "channels": {"wind": "load"}}, "unknown_block"),
])
def test_w_cmp2_a_bad_instance_is_refused_and_burns(bad, code):
    from cascade.shared.config import parse_chain_config

    cfg = parse_chain_config(_with(_raw(), {KIND: bad}))
    r = cfg.lane_c.spec_refusal(KIND)
    assert r is not None and r.code == code, (r, code)
    assert cfg.lane_c.refused_weight == 1.0 and f"{KIND}:market" not in cfg.lane_c.targets


def test_w_cmp2_reserved_ids_are_refused():
    from cascade.shared.competition_spec import SpecRefusal, parse_spec

    for sid in ("domain", "market", "x", "Bad"):
        assert isinstance(parse_spec(sid, SPEC), SpecRefusal)


def test_w_cmp2_overlay_adds_and_removes_a_competition_never_retroactively():
    from cascade.shared import netconfig as nc
    from cascade.shared.config import load_chain_config

    from .helpers import ROOT

    base = load_chain_config(ROOT / "chain.toml")

    def build(specs, eff):
        body = nc.next_overlay({"lane_c": {"specs": specs}}, eff, None,
                               network=nc.network_identity(base))
        return nc.build_config(base, body, applied_block=eff)

    out = build({KIND: SPEC}, 9590400)
    assert out.lane_c.instance_mc(KIND) is not None
    assert nc.retroactive_changes(base, out, 9590400) == []
    late = nc.retroactive_changes(base, out, 9594000)
    assert late and "from_block" in late[0]
    # removing it again (a later overlay restating only mkt) is no refusal
    gone = build({"mkt": {"runner": "marketcast"}}, 9590400)
    assert nc.lane_c_competition_changes(out, gone) == []


def _drv_cfg():
    register_kind()
    lc = replace(lane_c(True), legs_per_target=2)
    return SimpleNamespace(lane_c=lc, marketcast=replace(mkt_mc(), from_block=0))


def test_w_cmp2_trainer_dispatches_stamps_and_settles_an_instance_leg(tmp_path):
    from cascade.shared.chain import Commitment
    from cascade.shared.lane_c_manifest import format_finetune_commit, load_lane_c_manifest
    from cascade.shared.marketcast_report import instance_report_key
    from cascade.trainer.lane_c_driver import LaneCDriver
    from tests.unit.test_lane_c_trainer_driver import REF, UID, Clock, FakeOps, _ptr

    cfg = _drv_cfg()
    ops = FakeOps(tmp_path, Clock())
    mc = instance_mc()
    ops.train = lambda leg, recipe: {"trained_pointer": _ptr("ft-" + leg["hotkey"]),
                                     "train_block": leg["started_block"],
                                     "marketcast": {"base_ref": mc.base_ref,
                                                    "base_sha256": mc.base_sha256,
                                                    "checkpoint_sha256": "c" * 64}}
    reports: dict = {}
    ops.read_report = lambda key: reports.get(key)
    t = f"{KIND}:returns"
    ops.q.add_lane_c("ALFA", t, REF["ALFA"], 100)
    ops.commits.append(Commitment(uid=UID["ALFA"], hotkey="ALFA", coldkey=None,
                                  payload=format_finetune_commit(t, REF["ALFA"]),
                                  commit_block=100))
    drv = LaneCDriver(cfg, ops, tmp_path / "d.json")
    assert drv.tick(EB - 5, 1, _ptr("king"), EB - 3600) == []          # not live yet
    started = drv.tick(EB + 10, 2, _ptr("king"), EB)
    assert [x["target"] for x in started] == [t]
    assert (started[0]["embargo_days"], started[0]["embargo_eval_days"]) == (
        mc.embargo_days, mc.embargo_eval_days)
    drv.settle(round_id="1", block=EB + 900, era_index=2, base_pointer=_ptr("king"))
    m1 = load_lane_c_manifest(ops.published[-1][1])
    (e,) = m1.entries
    assert e.target == t and e.marketcast["embargo_days"] == mc.embargo_days
    assert e.competition == {} and m1.mkt_reports == {}
    assert drv.st.awaits == {KIND: [{"rid": "1", "age": 0}]}
    reports[instance_report_key(KIND, "1")] = '{"rows": []}'
    drv.settle(round_id="2", block=EB + 1800, era_index=2, base_pointer=_ptr("king"))
    m2 = load_lane_c_manifest(ops.published[-1][1])
    assert m2.reports == {KIND: {"1": hashlib.sha256(b'{"rows": []}').hexdigest()}}
    assert m2.mkt_reports == {}


def test_w_cmp2_training_data_is_the_pool_clipped_before_the_dispatch(tmp_path):
    from cascade.pool import marketcast as P
    from cascade.trainer.marketcast_leg import clip_training_snapshot, training_span

    from .helpers import T0, energy_pool

    snap = energy_pool(tmp_path / "pool", 4 * 1440)       # four days of minute bars
    day = lambda i: __import__("datetime").datetime.fromtimestamp(  # noqa: E731
        T0 + i * 86400, tz=__import__("datetime").UTC).date().isoformat()
    recipe = SimpleNamespace(start=day(0), end=day(1), assets=(("power", ("DE",)),))
    not_after = float(T0 + 86400 + 6 * 3600)              # dispatched day 1, 06:00
    meta = clip_training_snapshot(snap, tmp_path / "train", instance_mc(), recipe,
                                  not_after=not_after)
    t0, t1 = training_span(recipe, not_after)
    train = P.load_snapshot(tmp_path / "train", verify_signature=False)
    assert train.assets("power") == ["DE"]
    assert meta["t0"] == t0 and meta["end"] <= t1 <= not_after
    np.testing.assert_array_equal(train.bars("power", "DE")["close"],
                                  snap.bars("power", "DE")["close"][: (t1 - t0) // 60])


def test_w_cmp2_the_audit_reads_instance_rows_as_marketcast_rows():
    from cascade.audit.checks import _family_mc, _is_family, _lane_c_row_terms

    register_kind()
    cfg = SimpleNamespace(marketcast=mkt_mc(), lane_c=lane_c(True))
    t = f"{KIND}:returns"
    assert _is_family(t) and _family_mc(cfg, t) is cfg.lane_c.instance_mc(KIND)
    assert _lane_c_row_terms(t, cfg.lane_c, _family_mc(cfg, t))["mkt"] is True
