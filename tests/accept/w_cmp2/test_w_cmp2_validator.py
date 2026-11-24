"""W-CMP2 (DEC-CA-0075): a NEW lane-C competition — the ``energy`` spec
instance of the multi-instance MarketCast runner, on a GENERIC standard pool
from an energy builder — is registered, rowed, recorded, judged, confirmed,
slotted, paid and audited exactly like ``mkt``, keyed by its own id; and
``mkt``'s results are byte-identical with the instance configured beside it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from .helpers import (
    BLOCK,
    EB,
    FACTORS,
    KIND,
    bhash,
    energy_pool,
    entry,
    fixture_cfg,
    history,
    instance_mc,
    lane_c,
    mkt_mc,
    ptr,
    register_kind,
    unregister_kind,
)

bt = pytest.importorskip("bittensor.sp_core")
MAIN_KP = bt.Keypair.create_from_uri("//WCmp2Main")
TRAINER_KP = bt.Keypair.create_from_uri("//WCmp2Trainer")
KING = "5King"


@pytest.fixture(autouse=True)
def _kind():
    register_kind()
    yield
    unregister_kind()


class _Fc:
    def __init__(self, factor):
        from cascade.shared.config.marketcast import TIMESTRAL_KNOTS

        self.factor = factor
        self.quantile_levels = TIMESTRAL_KNOTS
        self.z = norm.ppf(np.asarray(TIMESTRAL_KNOTS))
        self.n = 0

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        self.n += 1
        c = np.asarray(context)
        sd = float(np.nanstd(c[np.isfinite(c)])) or 1.0
        return np.tile(self.z * sd * self.factor, (int(horizon), 1))


def _runner(tmp_path, store, snaps, *, hist, with_instance=True):
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    cfg = fixture_cfg(tmp_path / "cfg")
    v = replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    c = replace(cfg, lane_c=lane_c(with_instance), marketcast=mkt_mc(), validator=v,
                netconfig=replace(cfg.netconfig, owner_hotkey=MAIN_KP.ss58_address),
                manifest=replace(cfg.manifest, trainer_hotkey=TRAINER_KP.ss58_address))
    r = ValidatorRunner(cfg=c, state=genesis(KING, 3), verify_signatures=False)
    r._lane_c_store = store
    r._mkt_store_obj = store
    r.commitment_history_fn = lambda: hist
    r.bench_signer = MAIN_KP
    calls = {"base": _Fc(3.0)}
    for hk, f in FACTORS.items():
        for t in ("mkt:returns", f"{KIND}:returns"):
            calls[ptr(hk + "-" + t.partition(":")[0])] = _Fc(f)
    r.mkt_forecaster_fn = lambda key: (calls[key], ())
    r.mkt_snapshot_fn = lambda block: (snaps(block), f"pool/snapshots/block-{block}.tar",
                                       "e" * 64)
    r.mkt_block_hash_fn = bhash
    return r


def _settle(store, rid, entries, *, prev="", block=BLOCK, mkt_reports=None, reports=None):
    from cascade.shared.hippius import manifest_round_key
    from cascade.shared.lane_c_manifest import (
        LaneCManifest,
        dump_lane_c_manifest,
        lane_c_manifest_key,
        lane_c_text_digest,
        sign_lane_c_manifest,
    )
    from cascade.shared.manifest import TrainedEntry, TrainingManifest, dump_manifest

    king = ptr("king")
    cm = LaneCManifest(round_id=rid, era_index=1, created_block=block + 10, base_pointer=king,
                       base_round_id=rid, pool_block=block, entries=tuple(entries),
                       mkt_reports=dict(mkt_reports or {}), reports=dict(reports or {}))
    text = dump_lane_c_manifest(sign_lane_c_manifest(cm, TRAINER_KP))
    store.objs[lane_c_manifest_key(rid)] = text
    ents = [TrainedEntry(KING, 3, "king", "k/g@sha256:" + "3" * 64, king, "4" * 64, block - 1)]
    m = TrainingManifest(round_id=rid, created_block=block + 10, contract_digest="1" * 64,
                         base_arch_digest="2" * 64, eval_dataset="d", entries=ents,
                         prev_round_id=prev, lane_c_digest=lane_c_text_digest(text))
    store.objs[manifest_round_key(rid)] = dump_manifest(m)
    return m


def _rows(r, m):
    r._lane_c_block = r._epoch_start_block(m)
    return r._lane_c_commit(r._lane_c_prepare(m))


def _digest(store, key):
    from cascade.shared.marketcast_report import mkt_text_digest

    return mkt_text_digest(store.objs[key])


def _energy_rounds(tmp_path):
    from .helpers import Store

    s1 = energy_pool(tmp_path / "p1", 5000)
    s2 = energy_pool(tmp_path / "p2", 9000)
    store = Store()
    mc_i = instance_mc()
    es = [entry("alice", mc_i, f"{KIND}:returns"), entry("carol", mc_i, f"{KIND}:returns")]
    r = _runner(tmp_path / "v", store, lambda b: s1 if b < BLOCK + EB else s2,
                hist=history(es))
    key = f"marketcast/{KIND}/round-%s.json"
    rows1 = _rows(r, _settle(store, "1001", es))
    rows2 = _rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                             reports={KIND: {"1001": _digest(store, key % "1001")}}))
    rows3 = _rows(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                             reports={KIND: {"1002": _digest(store, key % "1002")}}))
    return r, store, (s1, s2), (rows1, rows2, rows3)


def test_w_cmp2_a_new_competition_is_judged_end_to_end_by_its_own_id(tmp_path):
    from cascade.shared.lane_c import lane_c_split

    r, store, (s1, s2), (rows1, rows2, rows3) = _energy_rounds(tmp_path)
    # its own report key, never MarketCast's
    assert f"marketcast/{KIND}/round-1001.json" in store.puts
    assert not any(k.startswith("marketcast/round-") and k.endswith(".json")
                   for k in store.puts)
    by2 = {(x["hotkey"], x["phase"]): x for x in rows2 if "error" not in x}
    a, c = by2[("alice", "judge")], by2[("carol", "judge")]
    assert a["target"] == f"{KIND}:returns" and a["passed"] and a["outcome"] == "pending"
    assert a["mkt"]["report_key"] == f"marketcast/{KIND}/round-1001.json"
    assert not c["passed"]
    a3 = {(x["hotkey"], x["phase"]): x for x in rows3 if "error" not in x}[("alice", "confirm")]
    assert a3["outcome"] == "confirmed"
    st = r._lane_c()
    assert st.slots[f"{KIND}:returns"].hotkey == "alice"
    assert "mkt:returns" not in st.slots and st.mkt_backlog == [] and st.mkt_seen == {}
    assert set(st.comp.get(KIND, {})) <= {"backlog", "seen"} and st.comp[KIND]["seen"]
    # paid by its own competition weight
    _c, slot = lane_c_split(r.cfg.lane_c, BLOCK + 3 * EB, EB, r.cfg.marketcast)
    assert slot[f"{KIND}:returns"] == pytest.approx(0.5)
    # audited: the generic lane-C rows and the instance's full replay
    from types import SimpleNamespace

    from cascade.audit.checks import check_lane_c, check_marketcast

    rc = SimpleNamespace(status="scored", epoch_start_block=BLOCK + EB, round_id="1002",
                         lane_c=list(rows2), lane_c_pay=None)
    assert check_lane_c(rc, r.cfg).status == "PASS", check_lane_c(rc, r.cfg).detail
    reps = {k: v for k, v in store.objs.items()
            if k.startswith("marketcast/") and k.endswith(".json")}
    inputs = {"lane_c_manifests": {k.split("round-")[1][:-5]: v for k, v in store.objs.items()
                                   if k.startswith("lane_c/")},
              "quantiles": {k: v for k, v in store.objs.items() if k.endswith(".npy")},
              "snapshots": {s1.digest: s1, s2.digest: s2},
              "block_hashes": {b: bhash(b) for b in (BLOCK, BLOCK + EB, BLOCK + 2 * EB)}}
    res = check_marketcast(rc, r.cfg, reps, **inputs, kind=KIND)
    assert res.status == "PASS", res.detail
    forged = [{**x, "passed": True, "gain_lcb": 0.5} if x.get("hotkey") == "carol" else x
              for x in rc.lane_c]
    bad = SimpleNamespace(**{**rc.__dict__, "lane_c": forged})
    assert check_marketcast(bad, r.cfg, reps, **inputs, kind=KIND).status == "FAIL"
    # the legacy check sees no MarketCast rows here
    assert check_marketcast(rc, r.cfg, reps, **inputs).status in ("PASS", "SKIP")


def _mkt_rounds(tmp_path, *, with_instance):
    from cascade.pool import marketcast as P

    from .helpers import ANCHOR_T, T0, Store

    rng = np.random.default_rng(11)
    bars = {}
    for a in ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB"):
        rr = rng.normal(0, 10.0 / 1e4, 12_000)[:9000]
        c = 100.0 * np.exp(np.cumsum(rr))
        bars[("crypto", a)] = {"open": c, "high": c, "low": c, "close": c,
                               "volume": np.ones(9000)}
    snaps = {}
    for tag, n in (("s1", 5000), ("s2", 9000)):
        P.write_snapshot(tmp_path / tag, t0=T0, minutes=n,
                         bars={k: {f: v[:n] for f, v in d.items()} for k, d in bars.items()},
                         sources={"crypto": "hyperliquid"}, anchor_block=BLOCK,
                         anchor_time=float(ANCHOR_T))
        snaps[tag] = P.load_snapshot(tmp_path / tag, verify_signature=False)
    store = Store()
    mc = mkt_mc()
    es = [entry("alice", mc, "mkt:returns"), entry("carol", mc, "mkt:returns")]
    r = _runner(tmp_path / "v", store,
                lambda b: snaps["s1"] if b < BLOCK + EB else snaps["s2"], hist=history(es),
                with_instance=with_instance)
    rows = [_rows(r, _settle(store, "1001", es))]
    rows.append(_rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                                 mkt_reports={"1001": _digest(store,
                                                              "marketcast/round-1001.json")})))
    rows.append(_rows(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                                 mkt_reports={"1002": _digest(store,
                                                              "marketcast/round-1002.json")})))
    return r, store, rows


def test_w_cmp2_mkt_is_byte_identical_with_an_instance_beside_it(tmp_path):
    import json

    from cascade.shared.lane_c import LaneCState

    r0, s0, rows0 = _mkt_rounds(tmp_path / "plain", with_instance=False)
    r1, s1, rows1 = _mkt_rounds(tmp_path / "inst", with_instance=True)

    def unsigned(rows):
        # sr25519 signatures are randomised: the report digest a row carries
        # differs run to run; everything else must be byte-identical
        return json.dumps([[{**x, "mkt": {k: v for k, v in (x.get("mkt") or {}).items()
                                          if k != "report_sha256"}} for x in rr]
                           for rr in rows], sort_keys=True)

    def body(objs):
        out = {}
        for k, v in objs.items():
            if k.startswith("marketcast/"):
                if k.endswith(".json"):
                    d = json.loads(v)
                    d.pop("signature", None)
                    v = json.dumps(d, sort_keys=True)
                out[k] = v
        return out

    assert unsigned(rows0) == unsigned(rows1)
    assert body(s0.objs) == body(s1.objs)
    st0, st1 = r0._lane_c(), r1._lane_c()
    assert st0.slots == st1.slots and st0.slots["mkt:returns"].hotkey == "alice"
    assert LaneCState.from_json(st0.to_json()).mkt_seen == st1.mkt_seen
    assert st1.comp.get(KIND) is None


def test_w_cmp2_registry_and_lookup():
    from cascade.shared.config.lane_c import is_mkt_family, mc_for
    from cascade.trainer.lane_c_competitions import trainer_competition
    from cascade.validator.lane_c_competitions import judging_order, validator_competition

    assert is_mkt_family(f"{KIND}:returns") and is_mkt_family("mkt:market")
    assert not is_mkt_family("domain:energy")
    assert validator_competition(f"{KIND}:returns").kind == KIND
    assert trainer_competition(f"{KIND}:returns").kind == KIND
    order = [c.kind for c in judging_order()]
    assert order[0] == "mkt" and KIND in order
    from types import SimpleNamespace

    cfg = SimpleNamespace(marketcast=mkt_mc(), lane_c=lane_c(True))
    assert mc_for(cfg, f"{KIND}:returns").pool_channels == (("close", "load"),)
    assert mc_for(cfg, "mkt:returns") is cfg.marketcast
