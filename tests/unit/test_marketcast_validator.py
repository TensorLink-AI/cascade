"""MarketCast (DEC-CA-0064) validator half + audit, after the review fixes of
2026-10-19: the main validator computes and publishes the signed per-window
rows (+ the float32 quantile sidecars); every validator applies a MarketCast
round only when a later TRAINER-signed lane-C manifest records its report
(sha256 or "absent") — a missing report never holds the settlement; infra
failures defer (bounded) instead of costing the miner; slots compare
against the pinned base (a king change never displaces a holder, no king is
needed); exact-duplicate weights lose to the earliest commit; and
``cascade-audit`` recomputes every loss from the published snapshot +
sidecar (tamper / omission / addition / a moved ``after`` ⇒ FAIL). Inert ⇒
audit output unchanged."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from cascade.pool import marketcast as P
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS, MarketCastConfig, parse_marketcast
from cascade.shared.hippius import ObjectNotFound, manifest_round_key
from cascade.shared.lane_c import TargetWin, win_beats
from cascade.shared.lane_c_manifest import (
    LaneCEntry,
    LaneCManifest,
    dump_lane_c_manifest,
    format_finetune_commit,
    lane_c_manifest_key,
    lane_c_text_digest,
    sign_lane_c_manifest,
)
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    format_trained_pointer,
)
from cascade.shared.marketcast_report import (
    INFRA_DEFERRED,
    load_mkt_report,
    mkt_text_digest,
    sign_mkt_report,
    verify_mkt_report,
)
from cascade.validator.loop import ValidatorRunner
from cascade.validator.marketcast import MktFault, mkt_prechecks
from cascade.validator.state import genesis

bt = pytest.importorskip("bittensor.sp_core")
MAIN_KP = bt.Keypair.create_from_uri("//MarketCastMain")
TRAINER_KP = bt.Keypair.create_from_uri("//MarketCastTrainer")

KING = "5King"
EB = 7200
BLOCK = 20 * EB
T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
ANCHOR_T = T0 + 60_000                            # BLOCK ↔ T0 + 1000 minutes
ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB")
UIDS = {"alice": 7, "carol": 9, "dave": 10, "erin": 11}
REF = {hk: f"{hk}/recipe@sha256:" + hashlib.sha256(hk.encode()).hexdigest() for hk in UIDS}
SIGMA = 10.0                                      # bp per 1-minute return
Z = norm.ppf(np.asarray(TIMESTRAL_KNOTS))


def _ptr(tag):
    return format_trained_pointer(f"cascade/ckpt-{tag}@sha256:"
                                  + hashlib.sha256(tag.encode()).hexdigest())


KING_PTR = _ptr("king")
KING2_PTR = _ptr("king2")
MC = parse_marketcast({
    "from_block": EB, "suites": ["returns"], "assets": {"crypto": list(ASSETS)},
    "frequencies": {"returns": ["1m", "5m"]}, "horizons": {"returns": ["1m", "5m"]},
    "eval_context_length": 64, "windows_per_cell": 30, "bootstrap_B": 1000,
    "min_windows": 100, "min_clusters": 5, "start_margin_seconds": 60,
    "verdict_wait_blocks": 2 * EB, "max_infra_deferrals": 2,
    "min_time_blocks": 4, "min_dollar_volume_24h": 0.0,
    "coverage_tolerance": 0.1, "coverage_tolerance_over": 0.15,     # 180 windows a horizon
})


class _Fc:
    """Gaussian quantiles from the context's own scale × ``factor``."""
    quantile_levels = TIMESTRAL_KNOTS

    def __init__(self, factor, fail=0):
        self.factor = factor
        self.n = 0
        self.fail = fail                          # raise RuntimeError on the first N calls

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        self.n += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("CUDA out of memory")
        c = np.asarray(context)
        sd = float(np.nanstd(c[np.isfinite(c)])) or 1.0
        return np.tile(Z * sd * self.factor, (int(horizon), 1))


def _snapshot(tmp_path, minutes, tag):
    rng = np.random.default_rng(11)
    bars = {}
    for a in ASSETS:
        r = rng.normal(0, SIGMA / 1e4, 12_000)[:minutes]
        c = 100.0 * np.exp(np.cumsum(r))
        bars[("crypto", a)] = {"open": c, "high": c, "low": c, "close": c,
                               "volume": np.ones(minutes)}
    P.write_snapshot(tmp_path / tag, t0=T0, minutes=minutes, bars=bars,
                     sources={"crypto": "hyperliquid"}, anchor_block=BLOCK,
                     anchor_time=float(ANCHOR_T))
    return P.load_snapshot(tmp_path / tag, verify_signature=False)


class _Store:
    def __init__(self):
        self.objs: dict[str, object] = {}
        self.puts: list[str] = []
        self.fail_reads: set[str] = set()

    def get_text(self, key):
        if key in self.fail_reads:
            raise RuntimeError("s3 503")
        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, str) else v.decode()

    def get_bytes(self, key):
        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, bytes) else v.encode()

    def put_text(self, key, text, content_type="", acl=None):
        self.objs[key] = text
        self.puts.append(key)

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objs[key] = bytes(data)
        self.puts.append(key)


def _tt(train_block):
    return float(ANCHOR_T + (int(train_block) - BLOCK) * 12)


def _entry(hk, *, train_block=BLOCK - 100, base_sha=None, ckpt=None, target="mkt:returns"):
    return LaneCEntry(miner_hotkey=hk, miner_uid=UIDS[hk], target=target,
                      recipe_ref=REF[hk], trained_pointer=_ptr(hk), train_block=train_block,
                      funded_by="miner",
                      marketcast={"base_ref": MC.base_ref,
                                  "base_sha256": base_sha or MC.base_sha256,
                                  "checkpoint_sha256": ckpt or hashlib.sha256(
                                      hk.encode()).hexdigest(),
                                  "train_time": _tt(train_block)})


def _history(entries, blocks=None):
    return [SimpleNamespace(hotkey=e.miner_hotkey,
                            commit_block=(blocks or {}).get(e.miner_hotkey, BLOCK - 500),
                            payload=format_finetune_commit(e.target, e.recipe_ref,
                                                           mkt_suites=("returns",)))
            for e in entries]


def _settle(store, rid, entries, *, prev="", block=BLOCK, bind=True, reports=None,
            king=KING_PTR, with_king=True):
    cm = LaneCManifest(round_id=rid, era_index=1, created_block=block + 10, base_pointer=king,
                       base_round_id=rid, pool_block=block, entries=tuple(entries),
                       mkt_reports=dict(reports or {}))
    text = dump_lane_c_manifest(sign_lane_c_manifest(cm, TRAINER_KP))
    store.objs[lane_c_manifest_key(rid)] = text
    ents = [TrainedEntry(KING, 3, "king", "k/g@sha256:" + "3" * 64, king, "4" * 64,
                         block - 1)] if with_king else []
    m = TrainingManifest(
        round_id=rid, created_block=block + 10, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d", entries=ents,
        prev_round_id=prev, lane_c_digest=lane_c_text_digest(text) if bind else "")
    store.objs[manifest_round_key(rid)] = dump_manifest(m)
    return m


def _recorded(store, rid):
    """What the trainer's next lane-C manifest records for round ``rid``."""
    return mkt_text_digest(store.objs[f"marketcast/round-{rid}.json"])


FACTORS = {"alice": 1.0, "carol": 3.0, "dave": 1.0, "erin": 1.0}


def _runner(cfg, tmp_path, store, snaps, *, history, main=True, mc=MC, verify=False):
    lc = replace(parse_lane_c({"targets": ["mkt:returns"], "bootstrap_B": 1000},
                              mkt_suites=("returns",)), from_block=EB)
    tmp_path.mkdir(parents=True, exist_ok=True)
    v = replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    c = replace(cfg, lane_c=lc, marketcast=mc, validator=v,
                netconfig=replace(cfg.netconfig, owner_hotkey=MAIN_KP.ss58_address),
                manifest=replace(cfg.manifest, trainer_hotkey=TRAINER_KP.ss58_address))
    r = ValidatorRunner(cfg=c, state=genesis(KING, 3), verify_signatures=verify)
    r._lane_c_store = store
    r._mkt_store_obj = store
    r.commitment_history_fn = lambda: history
    if main:
        r.bench_signer = MAIN_KP
    calls = {"base": _Fc(3.0)}
    for hk, f in FACTORS.items():
        calls[_ptr(hk)] = _Fc(f)
    r._covs = {}
    r.mkt_forecaster_fn = lambda key: (calls[key], r._covs.get(key, ()))
    r.mkt_snapshot_fn = lambda block: (snaps(block), f"pool_mkt/snapshots/block-{block}.tar",
                                       "e" * 64)
    r._calls = calls
    return r


def _judge(r, m):
    r._lane_c_block = r._epoch_start_block(m)
    return {(row.get("hotkey"), row.get("phase"), (row.get("mkt") or {}).get("at", "")): row
            for row in r._lane_c_commit(r._lane_c_prepare(m))}


def _rows(r, m):
    r._lane_c_block = r._epoch_start_block(m)
    return r._lane_c_commit(r._lane_c_prepare(m))


def _rc(rid, block, rows):
    return SimpleNamespace(status="scored", epoch_start_block=block, round_id=rid,
                           lane_c=list(rows), lane_c_pay=None)


# ── the full flow: register → rows → recorded → judged → confirm → slot ─────


def _three_rounds(cfg, tmp_path, *, king2=False, before_confirm=None):
    s1, s2 = _snapshot(tmp_path, 5000, "s1"), _snapshot(tmp_path, 9000, "s2")
    store = _Store()
    es = [_entry("alice"), _entry("carol")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1 if b < BLOCK + EB else s2,
                history=_history(es))
    rows1 = _rows(r, _settle(store, "1001", es))
    rows2 = _rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB, bind=True,
                             reports={"1001": _recorded(store, "1001")}))
    if before_confirm is not None:
        before_confirm(r)
    rows3 = _rows(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                             reports={"1002": _recorded(store, "1002")},
                             king=KING2_PTR if king2 else KING_PTR))
    return SimpleNamespace(s1=s1, s2=s2, store=store, r=r, rows=(rows1, rows2, rows3))


@pytest.mark.parametrize("reason", ["expired", "deregistered"])
def test_ineligible_holder_cannot_block_a_lower_lcb_confirmed_win(cfg, tmp_path, reason):
    from cascade.shared.lane_c import TargetWin

    def holder(r):
        if reason == "expired":
            r.cfg = replace(r.cfg, lane_c=replace(r.cfg.lane_c, win_ttl_eras=10))
        else:
            r.registration_at_block_fn = lambda hotkey, block: None if hotkey == "former" else 1
        old = TargetWin("mkt:returns", "former", 5, "old", 0 if reason == "expired" else BLOCK,
                        base_pointer=MC.base_pointer, gain_lcb=0.99, tier=0.05, full_eras=10,
                        trained_pointer=_ptr("former"))
        r._lane_c().slots[old.target] = old

    f = _three_rounds(cfg, tmp_path, before_confirm=holder)
    assert f.r._lane_c().slots["mkt:returns"].hotkey == "alice"
    confirmed = [r for r in f.rows[2] if r.get("outcome") == "confirmed"]
    assert [r["hotkey"] for r in confirmed] == ["alice"]


def test_register_record_judge_confirm_and_slot(cfg, tmp_path):
    f = _three_rounds(cfg, tmp_path)
    rows1, rows2, rows3 = f.rows
    # round 1001: registered + its report published — nothing judged yet
    assert [x for x in rows1 if str(x.get("target", "")).startswith("mkt:")] == []
    assert "marketcast/round-1001.json" in f.store.puts
    rep1 = load_mkt_report(f.store.objs["marketcast/round-1001.json"])
    assert verify_mkt_report(rep1, MAIN_KP.ss58_address)
    assert rep1.snapshot["digest"] == f.s1.digest and len(rep1.entries) == 2
    for e in rep1.entries:                        # every window a row, sidecar bound
        assert e.quantiles_key in f.store.objs
        assert hashlib.sha256(f.store.objs[e.quantiles_key]).hexdigest() == e.quantiles_sha256
    # base forecasts are cached per window: carol reused alice's (and round
    # 1002 forecast only its own, fresh windows)
    rep2 = load_mkt_report(f.store.objs["marketcast/round-1002.json"])
    assert f.r._calls["base"].n == len(rep1.entries[0].windows) + len(rep2.entries[0].windows)
    by = {(x["hotkey"], x["phase"]): x for x in rows2 if "error" not in x}
    a, c = by[("alice", "judge")], by[("carol", "judge")]
    assert a["passed"] and a["outcome"] == "pending" and a["k"] == 2
    assert a["alpha"] == pytest.approx(MC.alpha / 2)
    assert a["mkt"]["round"] == "1001" and a["mkt"]["entry_round"] == "1001"
    assert a["mkt"]["train_time"] == _tt(BLOCK - 100)
    assert a["base_pointer"] == MC.base_pointer               # the pinned base, never the king
    assert not c["passed"] and c["outcome"] == "dropped"
    a3 = {(x["hotkey"], x["phase"]): x for x in rows3 if "error" not in x}[("alice", "confirm")]
    assert a3["outcome"] == "confirmed" and a3["mkt"]["fresh_after"] == float(f.s1.end)
    assert all(w.start > f.s1.end for w in rep2.entries[0].windows)
    slot = f.r._lane_c().slots["mkt:returns"]
    assert (slot.hotkey, slot.win_block, slot.base_pointer) == ("alice", BLOCK + 2 * EB,
                                                               MC.base_pointer)
    assert f.r._lane_c().mkt_backlog == []
    # the audit: tier-0 lane-C checks + the full MarketCast replay (every loss
    # recomputed from the snapshot + sidecars)
    from cascade.audit.checks import check_lane_c, check_lane_c_chain, check_marketcast

    rcs = [_rc("1001", BLOCK, rows1), _rc("1002", BLOCK + EB, rows2),
           _rc("1003", BLOCK + 2 * EB, rows3)]
    reps = {k: v for k, v in f.store.objs.items() if k.endswith(".json")
            and k.startswith("marketcast/")}
    inputs = {"lane_c_manifests": {k.split("round-")[1][:-5]: v for k, v in f.store.objs.items()
                                   if k.startswith("lane_c/")},
              "quantiles": {k: v for k, v in f.store.objs.items() if k.endswith(".npy")},
              "snapshots": {f.s1.digest: f.s1, f.s2.digest: f.s2}}
    for rc in rcs:
        res = check_marketcast(rc, f.r.cfg, reps, **inputs)
        assert res.status == "PASS", res.detail
        assert check_lane_c(rc, f.r.cfg).status == "PASS", check_lane_c(rc, f.r.cfg).detail
    assert "recomputed" in check_marketcast(rcs[1], f.r.cfg, reps, **inputs).detail
    assert check_lane_c_chain(rcs, f.r.cfg).status == "PASS"
    assert check_marketcast(rcs[1], f.r.cfg, reps).status == "WARN"        # inputs not fetched
    assert check_marketcast(rcs[1], f.r.cfg, {}).status == "WARN"
    # tamper in the receipt rows: a forged pass, a forged LCB
    rc2 = rcs[1]

    def _with(rc, **kw):
        return SimpleNamespace(**{**rc.__dict__, **kw})

    forged = [{**x, "passed": True, "gain_lcb": 0.5} if x.get("hotkey") == "carol" else x
              for x in rc2.lane_c]
    assert check_marketcast(_with(rc2, lane_c=forged), f.r.cfg, reps, **inputs).status == "FAIL"
    lcb = [{**x, "gain_lcb": x["gain_lcb"] + 1e-3} if x.get("hotkey") == "alice" else x
           for x in rc2.lane_c]
    assert check_marketcast(_with(rc2, lane_c=lcb), f.r.cfg, reps, **inputs).status == "FAIL"
    edited = dict(reps)
    edited["marketcast/round-1001.json"] = edited["marketcast/round-1001.json"].replace(
        '"base":', '"base": ', 1)
    assert check_marketcast(rc2, f.r.cfg, edited, **inputs).status == "FAIL"

    # tamper by the MAIN VALIDATOR itself (validly re-signed, re-bound): the
    # recomputation catches it
    def _resigned(mutate, sidecar=None):
        rep = load_mkt_report(f.store.objs["marketcast/round-1001.json"])
        ents = list(rep.entries)
        i = next(j for j, e in enumerate(ents) if e.hotkey == "alice")
        ents[i] = mutate(ents[i])
        new = sign_mkt_report(replace(rep, entries=tuple(ents), signature=""), MAIN_KP)
        text = new.to_json()
        rows = [{**x, "mkt": {**x["mkt"], "report_sha256": mkt_text_digest(text)}}
                if str(x.get("target", "")).startswith("mkt:") else x for x in rc2.lane_c]
        q = dict(inputs["quantiles"])
        if sidecar is not None:
            q[sidecar[0]] = sidecar[1]
        return (check_marketcast(_with(rc2, lane_c=rows), f.r.cfg,
                                 {**reps, "marketcast/round-1001.json": text},
                                 **{**inputs, "quantiles": q}))

    w0 = lambda e, **kw: replace(e, windows=(replace(e.windows[0], **kw),) + e.windows[1:])  # noqa: E731
    res = _resigned(lambda e: w0(e, base=e.windows[0].base * 1.01))
    assert res.status == "FAIL" and "recomputes" in res.detail
    res = _resigned(lambda e: w0(e, ft=float("nan")))                  # review M11
    assert res.status == "FAIL" and "non-finite" in res.detail
    res = _resigned(lambda e: replace(e, windows=e.windows[1:]))
    assert res.status == "FAIL" and "omitted" in res.detail
    extra = replace(rep1.entries[0].windows[0], wid="crypto:BTC|returns|1m|" + str(
        int(f.s1.end) - 60))
    res = _resigned(lambda e: replace(e, windows=e.windows + (extra,)))
    assert res.status == "FAIL"
    res = _resigned(lambda e: replace(e, after=e.after + 600.0))
    assert res.status == "FAIL" and "dispatch time" in res.detail
    # a tampered quantile (sidecar re-bound by sha256 in the re-signed report)
    a_ent = next(e for e in rep1.entries if e.hotkey == "alice")
    arr = np.load(io.BytesIO(f.store.objs[a_ent.quantiles_key]))
    arr[0, 1, 10] += np.float32(0.5)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    res = _resigned(lambda e: replace(e, quantiles_sha256=hashlib.sha256(
        buf.getvalue()).hexdigest()), sidecar=(a_ent.quantiles_key, buf.getvalue()))
    assert res.status == "FAIL" and "recomputes" in res.detail


def test_king_change_never_displaces_an_mkt_holder(cfg, tmp_path):
    f = _three_rounds(cfg, tmp_path, king2=True)          # the king moved at 1003
    slot = f.r._lane_c().slots["mkt:returns"]
    assert slot.hotkey == "alice"                         # confirmed on a NEW king: not stale
    # a later, LOWER win does not take the slot just because a king moved
    holder = TargetWin("mkt:returns", "alice", 7, "1003", BLOCK + 2 * EB,
                       base_pointer=MC.base_pointer, gain_lcb=0.3)
    later = TargetWin("mkt:returns", "copy", 8, "1009", BLOCK + 9 * EB,
                      base_pointer=MC.base_pointer, gain_lcb=0.1)
    assert not win_beats(later, holder)
    assert win_beats(replace(later, gain_lcb=0.31), holder)


# ── liveness: a missing report never holds; the manifest arbitrates ─────────


def test_missing_report_never_holds_and_the_recorded_sha_judges(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    other = _runner(cfg, tmp_path / "o", store, lambda b: s1, history=_history(es), main=False)
    # no report anywhere: the settlement is NOT held — the MarketCast entry
    # waits in the backlog
    rows = _rows(other, _settle(store, "1001", es))
    assert not any(str(x.get("target", "")).startswith("mkt:") for x in rows)
    assert [b["round_id"] for b in other._lane_c().mkt_backlog] == ["1001"]
    # the next settlement records nothing either (report still missing): proceeds
    _rows(other, _settle(store, "1002", [], prev="1001", block=BLOCK + EB))
    assert [b["round_id"] for b in other._lane_c().mkt_backlog] == ["1001"]
    # the main validator publishes; a later manifest records the sha: judged
    main = _runner(cfg, tmp_path / "m", store, lambda b: s1, history=_history(es))
    _rows(main, _settle(store, "1001", es))
    m3 = _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                 reports={"1001": _recorded(store, "1001")})
    other.verify_signatures = True
    row = _judge(other, m3)[("alice", "judge", "")]
    assert row["stage"] == "verdict" and row["passed"], (row["reason"], row["mkt"])
    assert row["mkt"]["round"] == "1001"
    assert all(fc.n == 0 for fc in other._calls.values())            # never forecasts
    # the wait is measured on the ROUND's block: a late report never eats it
    assert row["mkt"]["origin_block"] == BLOCK and row["mkt"]["within_wait"]
    # a recorded sha whose bytes cannot be READ holds (the bytes provably exist)
    from cascade.validator.lane_c import LaneCTransient

    third = _runner(cfg, tmp_path / "t", store, lambda b: s1, history=_history(es), main=False)
    _rows(third, _settle(store, "2001", es))
    store.objs["marketcast/round-2001.json"] = store.objs["marketcast/round-1001.json"]
    store.fail_reads.add("marketcast/round-2001.json")
    m4 = _settle(store, "2002", [], prev="2001", block=BLOCK + EB,
                 reports={"2001": mkt_text_digest(store.objs["marketcast/round-1001.json"])})
    with pytest.raises(LaneCTransient, match="unreadable"):
        third._lane_c_prepare(m4)


def test_unsigned_or_mismatched_report_is_infra_never_a_loss(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    main = _runner(cfg, tmp_path / "m", store, lambda b: s1, history=_history(es))
    _rows(main, _settle(store, "1001", es))
    rep = load_mkt_report(store.objs["marketcast/round-1001.json"])
    store.objs["marketcast/round-1001.json"] = replace(rep, signature="").to_json()
    other = _runner(cfg, tmp_path / "o", store, lambda b: s1, history=_history(es), main=False,
                    verify=True)
    _rows(other, _settle(store, "1001", es))
    row = _judge(other, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                                reports={"1001": _recorded(store, "1001")}))[
        ("alice", "judge", "")]
    assert row["stage"] == "infra" and not row["passed"] and row["outcome"] == "deferred"
    assert "signature" in row["reason"] and row["mkt"]["infra_deferrals"] == 1
    # bytes that are not the recorded ones: infra too (never a hold, never a loss)
    third = _runner(cfg, tmp_path / "t", store, lambda b: s1, history=_history(es), main=False)
    _rows(third, _settle(store, "1001", es))
    row = _judge(third, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                                reports={"1001": "0" * 64}))[("alice", "judge", "")]
    assert row["stage"] == "infra" and "not the ones the trainer recorded" in row["reason"]


def test_absent_report_defers_as_infra_then_drops_past_the_cap(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es), main=False)
    _rows(r, _settle(store, "1001", es))
    rows = []
    for i, rid in enumerate(("1002", "1003", "1004")):
        prev = str(int(rid) - 1)
        got = _judge(r, _settle(store, rid, [], prev=prev, block=BLOCK + (i + 1) * EB,
                                reports={prev: "absent"}))
        rows.append(got[("alice", "judge", "")])
    assert [x["stage"] for x in rows] == ["infra"] * 3
    assert [x["mkt"]["infra_deferrals"] for x in rows] == [1, 2, 3]
    assert [x["outcome"] for x in rows] == ["deferred", "deferred", "dropped"]
    assert rows[-1]["mkt"]["error_class"] == "operator"           # requeue-eligible, uncharged
    assert "error_class" not in rows[0]["mkt"]
    # infra time never eats the wait: the origin shifts with each deferral
    assert rows[0]["mkt"]["origin_block"] == BLOCK
    assert rows[1]["mkt"]["origin_block"] == BLOCK + EB
    assert r._lane_c().mkt_backlog == []
    from cascade.audit.checks import check_lane_c, check_marketcast

    for i, x in enumerate(rows):
        rc = _rc(str(1002 + i), BLOCK + (i + 1) * EB, [x])
        assert check_lane_c(rc, r.cfg).status == "PASS", check_lane_c(rc, r.cfg).detail
        assert check_marketcast(rc, r.cfg, {}).status in ("PASS", "WARN")
    bad = _rc("1004", BLOCK + 3 * EB, [{**rows[-1], "outcome": "deferred"}])
    assert check_lane_c(bad, r.cfg).status == "FAIL"


def test_a_round_never_recorded_expires_to_absent(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    mc = replace(MC, report_timeout_settlements=1)
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es), main=False,
                mc=mc)
    _rows(r, _settle(store, "1001", es))
    _rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB))
    _rows(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB))
    assert r._lane_c().mkt_backlog[0]["round_id"] == "1001"      # 2 × timeout not yet passed
    got = _judge(r, _settle(store, "1004", [], prev="1003", block=BLOCK + 3 * EB))
    row = got[("alice", "judge", "")]
    assert row["stage"] == "infra" and row["mkt"]["report_expired"]


def test_compute_failure_is_retried_then_infra_deferred_and_rejudged(cfg, tmp_path):
    s1, s2 = _snapshot(tmp_path, 5000, "s1"), _snapshot(tmp_path, 9000, "s2")
    store = _Store()
    es = [_entry("alice")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1 if b < BLOCK + EB else s2,
                history=_history(es))
    r._calls[_ptr("alice")].fail = 2                   # OOM on both attempts of round 1001
    _rows(r, _settle(store, "1001", es))
    rep = load_mkt_report(store.objs["marketcast/round-1001.json"])
    assert rep.entries[0].status == INFRA_DEFERRED and "2x" in rep.entries[0].infra_reason
    row = _judge(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                            reports={"1001": _recorded(store, "1001")}))[("alice", "judge", "")]
    assert row["stage"] == "infra" and row["outcome"] == "deferred"
    # retried at the next round's report: now computed and judged
    rep2 = load_mkt_report(store.objs["marketcast/round-1002.json"])
    assert rep2.entries[0].status == "" and rep2.entries[0].windows
    row2 = _judge(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                             reports={"1002": _recorded(store, "1002")}))[
        ("alice", "judge", "")]
    assert row2["stage"] == "verdict" and row2["origin_round"] == "1001"
    assert row2["mkt"]["infra_deferrals"] == 1
    # the infra round did not consume the wait: origin moved by one round
    assert row2["mkt"]["origin_block"] == BLOCK + EB


def test_deterministic_meta_error_is_a_fault_never_a_hold(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice"), _entry("dave")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es))
    r._covs[_ptr("alice")] = ("open_interest",)      # not in the allow-list / no builder
    orig = r.mkt_forecaster_fn

    def _fc(key):
        if key == _ptr("dave"):
            raise MktFault("fine-tune is not the base architecture")
        return orig(key)

    r.mkt_forecaster_fn = _fc
    _rows(r, _settle(store, "1001", es))               # no LaneCTransient
    got = _judge(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                            reports={"1001": _recorded(store, "1001")}))
    a, d = got[("alice", "judge", "")], got[("dave", "judge", "")]
    assert a["stage"] == "fault" and "open_interest" in a["reason"]
    assert d["stage"] == "fault" and not d["passed"]


def test_prechecks_rejected_at_registration(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    stale = _entry("alice", base_sha="0" * 64)
    unbound = _entry("carol")
    notime = replace(_entry("dave"), marketcast={k: v for k, v in _entry("dave").marketcast.items()
                                                 if k != "train_time"})
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1,
                history=_history([stale, notime]))
    got = _judge(r, _settle(store, "1001", [stale, unbound, notime]))
    assert got[("alice", "judge", "register")]["stage"] == "stale"
    assert got[("carol", "judge", "register")]["stage"] == "binding"
    assert "train_time" in got[("dave", "judge", "register")]["reason"]
    assert r._lane_c().mkt_backlog == []
    lc = r.cfg.lane_c
    assert mkt_prechecks(replace(_entry("dave"), target="mkt:vol"), lc=lc, mc=MC,
                         commitments=[]) == ("target", "'mkt:vol' is not an enabled "
                                                       "MarketCast target")


def test_inconclusive_entries_wait_then_drop(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    # trained right at the snapshot's end: no window has closed yet
    late = _entry("dave", train_block=BLOCK + (5000 - 1000) * 5)
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history([late]))
    _rows(r, _settle(store, "1001", [late]))
    out = []
    for rid, k in (("1002", 1), ("1003", 3), ("1004", 4)):
        prev = str(int(rid) - 1)
        got = _judge(r, _settle(store, rid, [], prev=prev, block=BLOCK + k * EB,
                                reports={prev: _recorded(store, prev)}))
        out.append(got[("dave", "judge", "")])
    assert out[0]["outcome"] == "deferred" and out[0]["mkt"]["inconclusive"] == "windows"
    assert out[0]["mkt"]["within_wait"] and out[0]["n"] == 0
    assert out[1]["outcome"] == "deferred" and out[1]["origin_round"] == "1001"
    assert not out[2]["mkt"]["within_wait"] and out[2]["outcome"] == "dropped"
    assert r._lane_c().mkt_backlog == []


def test_mkt_is_judged_with_no_king(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es))
    _rows(r, _settle(store, "1001", es, with_king=False))
    rows = _rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB, with_king=False,
                            reports={"1001": _recorded(store, "1001")}))
    by = {(x.get("hotkey"), x.get("phase")): x for x in rows if "error" not in x}
    assert by[("alice", "judge")]["passed"] and by[("alice", "judge")]["outcome"] == "pending"
    rows = _rows(r, _settle(store, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                            with_king=False, reports={"1002": _recorded(store, "1002")}))
    assert {"error": "no_king"} not in rows                         # never needs one
    assert any(x.get("hotkey") == "alice" and x.get("phase") == "confirm" for x in rows)
    from cascade.audit.checks import check_lane_c_chain

    assert check_lane_c_chain([_rc("1002", BLOCK + EB, rows)], r.cfg).status == "PASS"


def test_duplicate_weights_lose_to_the_earliest_commit(cfg, tmp_path):
    s1, s2 = _snapshot(tmp_path, 5000, "s1"), _snapshot(tmp_path, 9000, "s2")
    store = _Store()
    same = "c" * 64
    orig = _entry("alice", ckpt=same)
    copy = _entry("erin", ckpt=same)
    hist = _history([orig, copy], blocks={"alice": BLOCK - 900, "erin": BLOCK - 400})
    # same settlement: the later commit is refused at registration
    r = _runner(cfg, tmp_path / "a", store, lambda b: s1, history=hist)
    got = _judge(r, _settle(store, "1001", [copy, orig]))
    assert got[("erin", "judge", "register")]["stage"] == "duplicate"
    assert [c["entry"]["miner_hotkey"] for c in r._lane_c().mkt_backlog[0]["candidates"]] == [
        "alice"]
    # the copy settles FIRST (it may even take the slot); the original lands
    # later with the earlier commit: the copy loses its slot, never by UID
    store2 = _Store()
    r2 = _runner(cfg, tmp_path / "b", store2, lambda b: s1 if b < BLOCK + EB else s2,
                 history=hist)
    _rows(r2, _settle(store2, "1001", [copy]))
    _rows(r2, _settle(store2, "1002", [], prev="1001", block=BLOCK + EB,
                      reports={"1001": _recorded(store2, "1001")}))
    _rows(r2, _settle(store2, "1003", [], prev="1002", block=BLOCK + 2 * EB,
                      reports={"1002": _recorded(store2, "1002")}))
    assert r2._lane_c().slots["mkt:returns"].hotkey == "erin"
    got = _judge(r2, _settle(store2, "1004", [orig], prev="1003", block=BLOCK + 3 * EB))
    ev = got[("erin", "judge", "register")]
    assert ev["stage"] == "duplicate" and ev["mkt"]["evicted"] == "slot"
    assert "mkt:returns" not in r2._lane_c().slots
    assert r2._lane_c().mkt_seen[same]["hotkey"] == "alice"


# ── inert / unchanged outputs ────────────────────────────────────────────────


def test_marketcast_inactive_refuses_mkt_entries(cfg, tmp_path):
    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice")]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es),
                mc=replace(MC, from_block=BLOCK + EB))
    row = _judge(r, _settle(store, "1001", es))[("alice", "judge", "")]
    assert row["stage"] == "target" and "not active" in row["reason"]
    assert store.puts == [] and r._lane_c().mkt_backlog == []


def test_inert_audit_output_is_byte_identical(cfg):
    from cascade.audit import checks as C

    from .receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt(cfg, validator_hotkey="5V")
    base = C.run_tier0(receipt, cfg)
    assert "marketcast" not in {r.name for r in base}
    later = replace(cfg, marketcast=replace(MarketCastConfig(),
                                            from_block=int(receipt.epoch_start_block) + EB))
    assert [repr(r) for r in C.run_tier0(receipt, later)] == [repr(r) for r in base]
    live = replace(cfg, marketcast=replace(MarketCastConfig(),
                                           from_block=int(receipt.epoch_start_block)))
    names = [r.name for r in C.run_tier0(receipt, live)]
    assert "marketcast" in names
    assert names.index("marketcast") == names.index("lane_c") + 1


def test_receipt_rows_state_and_manifests_without_marketcast_are_unchanged(cfg, tmp_path):
    """A lane-C judgement with no MarketCast record carries no ``mkt`` key, a
    pending entry without one and a state without MarketCast rounds persist
    without the new keys, and a lane-C manifest without ``mkt_reports`` signs
    the same body as before."""
    from cascade.shared.lane_c import LaneCState, PendingEntry
    from cascade.validator.lane_c import LaneCJudgement, judgements_record

    e = LaneCEntry("hk", 1, "mkt:returns", REF["alice"], _ptr("x"), 5)
    row = judgements_record([LaneCJudgement(e, False, "verdict", "r")])[0]
    assert "mkt" not in row
    st = LaneCState()
    st.set_pending([PendingEntry("mkt:returns", "confirm", {"a": 1}, KING_PTR, "r", 1)])
    js = json.loads(st.to_json())
    assert '"mkt"' not in st.to_json() and "mkt_backlog" not in js and "mkt_seen" not in js
    assert LaneCState.from_json(st.to_json()).to_json() == st.to_json()
    cm = LaneCManifest("r", 1, 2, KING_PTR, "r", 0, (e,))
    assert "mkt_reports" not in json.loads(cm.canonical_body())
    assert cm.canonical_body() == replace(cm, mkt_reports={}).canonical_body()
    with pytest.raises(ValueError):
        replace(cm, mkt_reports={"r0": "nope"})
