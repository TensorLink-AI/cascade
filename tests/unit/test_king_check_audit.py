"""DEC-CA-0058: the audit re-derives the DEC-CA-0055 king check instead of
trusting the main validator's pass/fail — the pool half replayed on the
recorded rows, the bench half against the trainer-signed report, due-ness
from the recorded reign clock — and the receipt bytes stay unchanged when no
check runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.audit.checks import FAIL, PASS, SKIP, WARN, check_king_check, run_tier0
from cascade.eval.scoring import WindowScore
from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.config import load_chain_config
from cascade.shared.era import era_length_blocks, v2_check_due
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    TrainingManifest,
    format_trained_pointer,
)
from cascade.validator.cascade import CascadeController, CascadeState
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import ChampionState

from .receipt_fixture import make_scores

ROOT = Path(__file__).resolve().parents[2]
KING = "5King"
CID = "alice/gen@sha256:" + "a" * 64
KP = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
IP = format_trained_pointer("cascade/init@sha256:" + "c" * 64)
SEED = 424242


class _Client:
    def __init__(self, block, uids):
        self.block, self.uids = block, uids

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return self.uids.get(hk)

    def set_weight_vector(self, vec):
        return True


@pytest.fixture
def env(tmp_path):
    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  netconfig=replace(cfg.netconfig, owner_hotkey=""),
                  training=replace(cfg.training, v2_from_block=gate),
                  lane_b=replace(cfg.lane_b, from_block=gate + 1),
                  validator=replace(cfg.validator, state_db_path=str(tmp_path / "s.json")))
    floor = gate + cfg.v2.random_init_eras * era
    due = next(x for x in range(floor, floor + 10 * era, cfg.round.epoch_blocks)
               if v2_check_due(cfg, boundary=x, reign_start_block=gate))
    return cfg, gate, era, due


def _manifest(boundary, *, warm=""):
    return TrainingManifest(
        round_id=str(boundary + 5), created_block=boundary + 5, contract_digest="c",
        base_arch_digest="d", eval_dataset="e", prev_round_id="77", warm_start_ckpt=warm,
        entries=[TrainedEntry(KING, 3, "king", CID, KP, "d", boundary - 10)])


def _report(king_better=True, with_init=True):
    good, bad = BenchScores(*[1.0] * 6), BenchScores(*[2.0] * 6)
    entries = [BenchEntry(role="king", size="m", miner_hotkey=KING, miner_uid=3,
                          trained_pointer=KP, scores=good if king_better else bad)]
    if with_init:
        entries.append(BenchEntry(role="init", size="m", miner_hotkey="", miner_uid=-1,
                                  trained_pointer=IP, scores=BenchScores(*[1.5] * 6)))
    return BenchReport(round_id="77", created_block=1, entries=tuple(entries))


def _scaled(scores, k):
    return [WindowScore(s.series_id, s.mase * k, s.qloss_per_q * k, s.abs_target)
            for s in scores]


def _settle(cfg, gate, era, b, *, report, king_scale=0.7, n=256, reign_start=None,
            stash=False):
    """Run the validator's settlement economics (the real king check, the real
    ``evaluate_round``) and build the no-contest receipt it would publish.
    ``stash`` reuses the duel's king + baseline rows (scored-settlement path):
    they ride ``entry_scores`` and the check records no rows of its own."""
    from cascade.shared.receipt import EntryScores, WindowScoreRecord

    r = ValidatorRunner(cfg=cfg, state=ChampionState(
        king_hotkey=KING, king_uid=3, king_since_block=gate + era, king_pointer=KP))
    r.cascade = CascadeController(reign_days=1.0, state=CascadeState(
        king_hotkey=KING, reign_start_block=gate + era if reign_start is None else reign_start,
        clock_observed=True))
    r._fetch_bench_report = lambda rid, **k: report
    init = make_scores(1.0, 7, n=n)
    king = _scaled(init, king_scale)
    r._evaluate = lambda entry, windows, training: list(
        init if entry.trained_pointer == IP else king)
    m = _manifest(b, warm=(IP if stash else ""))
    entry_scores = ()
    if stash:
        r._king_eval = (m.round_id, king, init, 0)
        size = cfg.training.arch_preset
        entry_scores = (
            EntryScores("king", size, KING, 3,
                        tuple(WindowScoreRecord.from_score(s) for s in king)),
            EntryScores("baseline", size, r.BASELINE_HOTKEY, -1,
                        tuple(WindowScoreRecord.from_score(s) for s in init)))
    r._v2_settlement_economics(m, [], SEED, None, KING)
    c = _Client(b, {KING: 3})
    uids = r._reward_uids(m, None, c)
    r._lane_b_block = b
    vec = r._apply_weights(c, m.round_id, uids, boundary=b)
    rc = r.build_round_receipt(
        m, base_seed=SEED, epoch_start_block=b, epoch_block_hash="0x",
        reject_reason="no_king_challenger_pair", reward_uids=tuple(uids), weights=vec,
        emission=r._last_emission, king_check=r._king_check_for(m.round_id),
        king_check_scores=r._king_check_scores_for(m.round_id))
    return replace(rc, entry_scores=entry_scores), r


def _kc(rc, **kw):
    return replace(rc, king_check=dict(json.loads(json.dumps(rc.king_check)), **kw))


def _ladder(rc, **kw):
    em = json.loads(json.dumps(rc.emission))
    em["ladder"].update(kw)
    return replace(rc, emission=em)


# ── honest checks replay ─────────────────────────────────────────────────────


@pytest.mark.parametrize("report, scale, result", [
    (_report(), 0.7, "pass"),
    (_report(king_better=False), 0.7, "fail"),      # the bench half fails it
    (_report(), 1.05, "fail"),                      # the pool half fails it
])
def test_honest_check_replays(env, report, scale, result):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=report, king_scale=scale)
    assert rc.king_check["result"] == result
    assert {e.role for e in rc.king_check_scores} == {"king", "init"}
    assert rc.emission["ladder"]["reign_start_block"] == gate + era
    row = check_king_check(rc, cfg, {"report": report})
    assert row.status == PASS, row.detail
    # it is wired into tier 0, ahead of the weight rebuild
    rows = {x.name: x for x in run_tier0(rc, cfg, king_bench={"report": report})}
    assert rows["king-check"].status == PASS


def test_reused_duel_rows_replay(env):
    """A scored settlement: the king's and the init's rows are the verdict's
    (``entry_scores`` king + baseline); the check records none of its own."""
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report(), stash=True)
    assert rc.king_check["result"] == "pass" and rc.king_check_scores == ()
    assert check_king_check(rc, cfg, {"report": _report()}).status == PASS
    # the baseline is the era init: a bench init of another checkpoint fails
    bad = _kc(rc, bench=dict(rc.king_check["bench"], init_pointer=KP))
    assert check_king_check(bad, cfg, {"report": _report()}).status == FAIL


# ── tampers fail ─────────────────────────────────────────────────────────────


def test_recorded_pass_whose_rows_give_a_fail(env):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report(), king_scale=1.05)
    assert rc.king_check["result"] == "fail"
    forged = _kc(rc, result="pass", reasons=[])
    row = check_king_check(forged, cfg, {"report": _report()})
    assert row.status == FAIL and "give fail" in row.detail
    # same for honest pass bytes over rows edited to a worse king
    rc, _ = _settle(cfg, gate, era, b, report=_report())
    rows = list(rc.king_check_scores)
    k = next(i for i, e in enumerate(rows) if e.role == "king")
    rows[k] = replace(rows[k], scores=tuple(replace(s, mase=s.mase * 2,
                                                    qloss_per_q=tuple(q * 2 for q in
                                                                      s.qloss_per_q))
                                            for s in rows[k].scores))
    row = check_king_check(replace(rc, king_check_scores=tuple(rows)), cfg,
                           {"report": _report()})
    assert row.status == FAIL


def test_tampered_lcb_fails(env):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report())
    row = check_king_check(_kc(rc, lcb=rc.king_check["lcb"] + 0.01), cfg, {"report": _report()})
    assert row.status == FAIL and "lcb" in row.detail
    row = check_king_check(_kc(rc, margin=0.0), cfg, {"report": _report()})
    assert row.status == FAIL and "margin" in row.detail


def test_recorded_pass_with_a_failing_signed_bench(env):
    """The main validator read a report where the king is worse, then claims a
    pass with flattering suite means: the signed report contradicts it."""
    cfg, gate, era, b = env
    bad = _report(king_better=False)
    rc, _ = _settle(cfg, gate, era, b, report=bad)
    assert rc.king_check["result"] == "fail"
    forged = _kc(rc, result="pass", reasons=[],
                 bench=dict(rc.king_check["bench"], king={k: 1.0 for k in
                                                          rc.king_check["bench"]["king"]}))
    row = check_king_check(forged, cfg, {"report": bad})
    assert row.status == FAIL and "signed report" in row.detail
    # a pass whose own recorded means fail the bench half fails internally
    forged = _kc(rc, result="pass", reasons=[])
    assert check_king_check(forged, cfg, None).status == FAIL


def test_missing_report_warns_after_internal_consistency(env):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report())
    for bench in (None, {"report": None}):
        row = check_king_check(rc, cfg, bench)
        assert row.status == WARN and row.name == "king-check-bench", row
    # the pool half is still replayed without the report
    assert check_king_check(_kc(rc, lcb=0.0), cfg, None).status == FAIL


def test_check_recorded_when_not_due_fails(env):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report())
    # judged under a config (e.g. its DEC-CA-0057 overlay) where the reign is
    # not yet long enough: the recorded check was not due
    longer = replace(cfg, v2=replace(cfg.v2, reign_warm_start_blocks=50 * era))
    assert not v2_check_due(longer, boundary=b, reign_start_block=gate + era)
    row = check_king_check(rc, longer, {"report": _report()})
    assert row.status == FAIL and "not due" in row.detail
    # a reign clock before the crowning is not the clock the validator read
    row = check_king_check(_ladder(rc, reign_start_block=gate), cfg, {"report": _report()})
    assert row.status == FAIL and "precedes the crowning" in row.detail
    row = check_king_check(_ladder(rc, reign_start_block=b + 1), cfg, {"report": _report()})
    assert row.status == FAIL and "after the settlement" in row.detail


def test_missing_check_when_due(env):
    """FAIL when the recorded clock says due; WARN when the receipt records no
    clock (the validator skips while its cascade view lags the champion)."""
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=_report())
    gone = replace(rc, king_check=None, king_check_scores=())
    row = check_king_check(gone, cfg)
    assert row.status == FAIL and "due" in row.detail
    em = json.loads(json.dumps(rc.emission))
    em["ladder"].pop("reign_start_block")
    assert check_king_check(replace(gone, emission=em), cfg).status == WARN
    # not due: no check is right
    longer = replace(cfg, v2=replace(cfg.v2, reign_warm_start_blocks=50 * era))
    assert check_king_check(gone, longer).status == PASS
    # the validator really does skip when its cascade view lags
    _, r = _settle(cfg, gate, era, b, report=_report())
    r.cascade.state = replace(r.cascade.state, king_hotkey="5Other")
    r._v2_settlement_economics(_manifest(b), [], SEED, None, KING)
    assert r._king_check_for(str(b + 5)) is None
    assert "reign_start_block" not in r._settlement_ladder[1]


# ── deferrals ────────────────────────────────────────────────────────────────


def test_deferrals_are_checked_against_what_is_derivable(env):
    cfg, gate, era, b = env
    rc, _ = _settle(cfg, gate, era, b, report=None)
    assert rc.king_check["result"] == "deferred" and rc.king_check_scores == ()
    assert check_king_check(rc, cfg, {"report": None}).status == PASS
    assert check_king_check(rc, cfg, {"report": _report()}).status == WARN   # published later
    rc, _ = _settle(cfg, gate, era, b, report=_report(with_init=False))
    assert check_king_check(rc, cfg, {"report": _report(with_init=False)}).status == PASS
    # too few windows: the replay must agree it is inconclusive
    rc, _ = _settle(cfg, gate, era, b, report=_report(), n=50)
    assert rc.king_check["result"] == "deferred" and rc.king_check["lcb"] is None
    assert check_king_check(rc, cfg, {"report": _report()}).status == PASS
    full, _ = _settle(cfg, gate, era, b, report=_report())
    forged = _kc(full, result="deferred", lcb=None, reasons=list(rc.king_check["reasons"]))
    row = check_king_check(forged, cfg, {"report": _report()})
    assert row.status == FAIL and "conclusive" in row.detail
    # a transient reason is not derivable: WARN
    assert check_king_check(_kc(rc, reasons=["pool check unscorable: OSError: x"]),
                            cfg).status == WARN


# ── bytes unchanged without a check ──────────────────────────────────────────


@pytest.mark.parametrize("variant", ["non-v2", "ladder-off"])
def test_receipt_bytes_unchanged_without_a_check(env, variant):
    from cascade.shared.receipt import build_receipt
    from cascade.trainer.contract import RoundSeeds

    cfg, gate, era, b = env
    cfg = (replace(cfg, v2=replace(cfg.v2, from_block=0),
                   training=replace(cfg.training, v2_from_block=0),
                   lane_b=replace(cfg.lane_b, from_block=0))
           if variant == "non-v2" else replace(cfg, v2=replace(cfg.v2, king_ladder=False)))
    r = ValidatorRunner(cfg=cfg, state=ChampionState(
        king_hotkey=KING, king_uid=3, king_since_block=gate + era, king_pointer=KP))
    r.cascade = CascadeController(reign_days=1.0, state=CascadeState(
        king_hotkey=KING, reign_start_block=gate + era, clock_observed=True))
    r._fetch_bench_report = lambda rid, **k: _report()
    r._evaluate = lambda *a: make_scores(1.0, 7)
    m = _manifest(b)
    r._v2_settlement_economics(m, [], SEED, None, KING)
    assert r._king_check_for(m.round_id) is None
    assert r._king_check_scores_for(m.round_id) == ()
    rc = r.build_round_receipt(m, base_seed=SEED, epoch_start_block=b, epoch_block_hash="0x",
                               reject_reason="no_king_challenger_pair",
                               king_check=r._king_check_for(m.round_id),
                               king_check_scores=r._king_check_scores_for(m.round_id))
    body = json.loads(rc.canonical_body())
    assert "king_check" not in body and "king_check_scores" not in body
    legacy = build_receipt(
        round_id=m.round_id, status="rejected", epoch_start_block=b, epoch_block_hash="0x",
        base_seed=SEED, seeds=RoundSeeds.derive(SEED, cfg.training), manifest=m,
        reject_reason="no_king_challenger_pair", v2_block=rc.v2_block)
    assert rc.canonical_body() == legacy.canonical_body()
    assert check_king_check(rc, cfg).status == SKIP


def test_fetch_king_check_report_absent_vs_unreadable(env):
    from cascade.audit.main import fetch_king_check_report
    from cascade.shared.hippius import ObjectNotFound

    cfg = env[0]

    class Absent:
        def get_text(self, key):
            raise ObjectNotFound(key)

    class Down:
        def get_text(self, key):
            raise OSError("store down")

    assert fetch_king_check_report(cfg, "77", Absent()) == {"report": None}
    assert fetch_king_check_report(cfg, "77", Down()) is None


# ── DEC-CA-0061 rule v2: the bench half is the gate's rule on task rows ──────


def _task_rows(scale: float, n: int = 20) -> dict:
    return {s: {"status": "ok", "rows": [
        {"item": f"{s}/{i}", "crps": (1.0 + 0.01 * i) * scale,
         "mase": (2.0 + 0.02 * i) * scale} for i in range(n)]}
        for s in ("gift", "boom", "time", "fev")}


def _rule_v2_settle(env, king_scale: float):
    from cascade.shared.bench_report import dump_bench_rows

    cfg, gate, era, b = env
    cfg = replace(cfg, v2=replace(cfg.v2, bench_gate_v2_from_block=gate))
    rows = {KP: _task_rows(king_scale), IP: _task_rows(1.0)}
    texts = {f"rows/{p[-8:]}": dump_bench_rows(r) for p, r in rows.items()}
    import hashlib

    entries = []
    for role, ptr, hk, uid in (("king", KP, KING, 3), ("init", IP, "", -1)):
        key = f"rows/{ptr[-8:]}"
        entries.append(BenchEntry(
            role=role, size="m", miner_hotkey=hk, miner_uid=uid, trained_pointer=ptr,
            scores=BenchScores(*[1.0] * 6, fev_crps=1.0, fev_mase=1.0), rows_key=key,
            rows_sha256=hashlib.sha256(texts[key].encode()).hexdigest()))
    report = BenchReport(round_id="77", created_block=1, entries=tuple(entries))

    class Store:
        def get_text(self, key):
            return texts[key]

    orig = ValidatorRunner._bench_store
    ValidatorRunner._bench_store = lambda self: Store()
    try:
        rc, _ = _settle(cfg, gate, era, b, report=report)
    finally:
        ValidatorRunner._bench_store = orig
    signed = {"report": report, "rows": {KP: (rows[KP], {}), IP: (rows[IP], {})}}
    return cfg, rc, signed


@pytest.mark.parametrize("scale, result", [(0.9, "pass"), (1.1, "fail")])
def test_rule_v2_king_check_replays_from_the_task_rows(env, scale, result):
    from cascade.validator.king_check import RULE_V2

    cfg, rc, signed = _rule_v2_settle(env, scale)
    kc = rc.king_check
    assert kc["result"] == result and kc["bench"]["rule"] == RULE_V2
    gd = kc["bench"]["gate"]
    assert {"avg_ratio", "worst_lcb", "win_rate"} <= set(gd)
    assert gd["avg_ratio"] == pytest.approx(scale)
    row = check_king_check(rc, cfg, signed)
    assert row.status == PASS, row.detail
    # without the rows: internally consistent, but not replayed
    row = check_king_check(rc, cfg, {"report": signed["report"]})
    assert row.status == WARN and row.name == "king-check-bench"
    # the recorded detail edited (consistently) to a pass the rows refute
    if result == "fail":
        forged = _kc(rc, result="pass", reasons=[], bench=dict(
            kc["bench"], gate=dict(gd, passed=True, avg_ratio=0.9, win_rate=1.0,
                                   suites={s: dict(v, ratio=0.9, win_rate=1.0, ratio_lcb=0.95)
                                           for s, v in gd["suites"].items()},
                                   worst_lcb=0.95, reason="")))
        row = check_king_check(forged, cfg, signed)
        assert row.status == FAIL and "replayed over the signed rows" in row.detail
        # a recorded pass whose own detail implies a fail fails without rows
        assert check_king_check(_kc(rc, result="pass", reasons=[]), cfg, None).status == FAIL
