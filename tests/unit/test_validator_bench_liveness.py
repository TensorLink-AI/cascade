"""DEC-CA-0063 design-review fixes: the trainer-recorded arbiter, infra (never a
loss), sweep retry / budget, data revisions, write-once reports, the king
check's init pointer and rule, the promotion member hold, the audit's
coverage and reproduce rows, and the legacy paths staying byte-identical."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.eval.bench_gate import RULE_V2, evaluate_bench_gate
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
    load_bench_report,
    validator_bench_report_key,
    validator_bench_rows_key,
)
from cascade.shared.config.v2 import V2Config, check_bench_source, parse_bench_source_bounds
from cascade.shared.hippius import StorageError
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    dump_manifest,
    load_manifest,
)
from cascade.validator.king_check import BENCH_FIELDS, BENCH_FIELDS_V2, bench_failures
from cascade.validator.validator_bench import (
    ValidatorBenchHold,
    check_main_bench_signer,
)

from .test_validator_bench import (
    CHAL2_PTR,
    CHAL_PTR,
    CID,
    INIT_PTR,
    KING_PTR,
    MAIN,
    _era_start,
    _manifest,
    _record,
    _runner,
    _settle,
    _sidecar_report,
    _Store,
    _vcfg,
    _wire_loop,
)


def _two_chal_manifest(cfg, rid="101"):
    m = _manifest(cfg, rid=rid)
    return replace(m, entries=[*m.entries,
                               TrainedEntry("chal2", 2, "challenger", CID, CHAL2_PTR, "d", 10)])


# ── the arbiter ──────────────────────────────────────────────────────────────


def test_non_main_judges_without_the_report_when_no_clearer(cfg):
    v = _vcfg(cfg)
    store = _Store()                           # the main validator published nothing
    other, calls = _runner(v, chal_scale=1.3, store=store, signer="5Other")
    _wire_loop(other)
    m = _manifest(v)
    assert other._handle_manifest(object(), object(), m, "d" * 64) is True
    assert calls == [] and other.state.king_hotkey == "king_hk"
    assert other._vbench_receipt_kw(m) == {}   # judged without needing it
    assert "101|validator_bench" not in other._read_holds


def test_recorded_sha_is_read_and_mismatching_bytes_are_uncomputable(cfg):
    v = _vcfg(cfg)
    store = _Store()
    main, _ = _runner(v, store=store)
    m = _manifest(v)
    _settle(main, m)
    other, _ = _runner(v, store=store, signer="5Other")
    sha = _record(other, store)
    out = _settle(other, m)
    assert out.result.challenger_wins_round and other._vbench["sha256"] == sha
    # a record that does not match the published bytes ⇒ uncomputable (king holds)
    third, _ = _runner(v, store=store, signer="5Third")
    later = replace(_manifest(v, rid="102", prev="101", offset=3),
                    validator_bench_reports={"101": "0" * 64})
    third._vbench_later_raw = [dump_manifest(later)]          # raw text works too
    out3 = _settle(third, m)
    assert out3.result.inconclusive and out3.result.bench_gate_passed is None


def test_recorded_absent_is_infra_never_a_crown_loss(cfg):
    v = _vcfg(cfg)
    store = _Store()
    other, calls = _runner(v, store=store, signer="5Other")
    other.state = replace(other.state, streaks={"chal_hk": 1})
    _record(other, store, absent=True)
    m = _manifest(v)
    out = _settle(other, m)
    assert calls == []
    assert out.result.inconclusive and not out.result.challenger_wins_round
    assert out.transition.state.streaks == {"chal_hk": 1}      # untouched: not a loss
    assert "bench_infra chal_hk: validator bench report recorded absent" in out.transition.note
    kw = other._vbench_receipt_kw(m)["bench_report"]
    assert kw["record"] == "absent" and "sha256" not in kw


def test_infra_clearer_falls_through_to_the_next(cfg):
    v = _vcfg(cfg)
    store = _Store()
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: None,
               CHAL2_PTR: _sidecar_report(0.9)}
    main, calls = _runner(v, store=store, reports=reports)
    m = _two_chal_manifest(v)
    out = _settle(main, m)
    assert out.result.challenger_wins_round and out.transition.new_king_hotkey == "chal2"
    assert "bench_infra chal_hk" in out.transition.note
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_reason(CHAL_PTR) and rep.entry_for_pointer(CHAL2_PTR) is not None
    # a reader reaches the same crown from the recorded bytes
    other, _ = _runner(v, store=store, signer="5Other")
    _record(other, store)
    o2 = _settle(other, m)
    assert o2.transition.new_king_hotkey == "chal2" and o2.transition.note == out.transition.note


def test_main_validator_holds_after_publishing_until_recorded(cfg):
    v = _vcfg(cfg)
    store = _Store()
    main, calls = _runner(v, store=store)
    _wire_loop(main)
    m = _manifest(v)
    before = main.state
    assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    assert main.state == before                                # rolled back
    assert validator_bench_report_key("101") in store.objects   # but published
    _record(main, store)
    assert main._handle_manifest(object(), object(), m, "d" * 64) is True
    assert main.state.king_hotkey == "chal_hk"
    assert sorted(calls) == sorted([KING_PTR, CHAL_PTR])       # swept once, re-read after


def test_record_lookup_takes_the_earliest_later_manifest(cfg):
    v = _vcfg(cfg)
    runner, _ = _runner(v, signer="5Other")
    a = replace(_manifest(v, rid="102", prev="101", offset=3),
                validator_bench_reports={"101": "a" * 64})
    b = replace(_manifest(v, rid="103", prev="102", offset=4),
                validator_bench_reports={"101": "absent"})
    runner._vbench_later_raw = [a, b]
    assert runner._vbench_record("101") == "a" * 64
    runner._vbench_later_raw = [b]
    assert runner._vbench_record("101") == "absent"
    assert runner._vbench_record("999") is None


# ── sweep retry, budget, data revisions ──────────────────────────────────────


def test_sweep_retry_recovers(cfg):
    v = _vcfg(cfg)
    store = _Store()
    seq = {CHAL_PTR: [None, _sidecar_report(0.9)]}
    main, calls = _runner(v, store=store)
    main.bench_sweep_fn = lambda e: (calls.append(e.trained_pointer),
                                     seq[e.trained_pointer].pop(0)
                                     if e.trained_pointer in seq else _sidecar_report(1.0))[1]
    out = _settle(main, _manifest(v))
    assert calls.count(CHAL_PTR) == 2 and out.result.challenger_wins_round
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_failed == ()


def test_bench_cap_defers_a_clearer_never_infra(cfg):
    # DEC-CA-0063 item 9: a clearer the settlement cap cannot hold a whole
    # allowance for is BUDGET-DEFERRED (not infra-failed), streak untouched
    v = _vcfg(cfg)
    v = replace(v, v2=replace(v.v2, bench_budget_hours=4.0, bench_sweep_hours=2.0))
    store = _Store()
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: None,
               CHAL2_PTR: _sidecar_report(0.9)}
    main, calls = _runner(v, store=store, reports=reports)
    main.state = replace(main.state, streaks={"chal2": 1})
    clock = {"t": 0.0}
    main.vbench_clock = lambda: clock["t"]
    inner = main.bench_sweep_fn

    def slow(entry):
        clock["t"] += 1.0 * 3600.0                   # every attempt takes 1 h
        return inner(entry)

    main.bench_sweep_fn = slow
    out = _settle(main, _two_chal_manifest(v))
    assert calls == [CHAL_PTR, CHAL_PTR]             # chal2 (needs 2 h + the king's) never ran
    assert out.result.inconclusive
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_reason(CHAL_PTR).startswith("sweep failed (2 attempts)")
    assert rep.deferred_reason(CHAL2_PTR) and rep.infra_reason(CHAL2_PTR) is None
    assert "bench_deferred chal2: bench budget exhausted" in out.transition.note
    assert out.transition.state.streaks == {"chal2": 1}


def test_cached_king_is_reswept_when_the_data_revision_moves(cfg):
    v = _vcfg(cfg)
    store = _Store()

    def rep(scale, rev):
        r = _sidecar_report(scale)
        r["data_revisions"] = {"gift-eval": rev, "boom": "b1", "time": "t1"}
        return r

    reports = {KING_PTR: rep(1.0, "g1"), CHAL_PTR: rep(0.9, "g1"), CHAL2_PTR: rep(0.9, "g2")}
    main, calls = _runner(v, store=store, reports=reports)
    _settle(main, _manifest(v))
    assert calls.count(KING_PTR) == 1
    reports[KING_PTR] = rep(1.0, "g2")
    from cascade.validator.state import genesis

    main.state = genesis("king_hk", 0)
    m2 = _manifest(v, rid="102", chal_ptr=CHAL2_PTR, chal_hk="chal2", prev="101", offset=3)
    out = _settle(main, m2)
    assert calls.count(KING_PTR) == 2               # cache invalidated by revision
    assert out.result.bench_gate_passed is True
    rep2 = load_bench_report(store.objects[validator_bench_report_key("102")])
    assert rep2.entry_for_pointer(KING_PTR).data_revisions["gift-eval"] == "g2"


def test_gate_refuses_entries_on_different_revisions():
    from .test_bench_gate_v2 import _eight, _rows

    g = evaluate_bench_gate(_rows(), _rows({"gift": 0.9}), _eight(1.0), _eight(0.9),
                            alpha=0.05, B=200, seed=1, rule=RULE_V2,
                            king_revisions={"gift-eval": "a"}, chal_revisions={"gift-eval": "b"})
    assert not g.computed and "data-revision mismatch" in g.reason
    same = evaluate_bench_gate(_rows(), _rows(), _eight(1.0), _eight(0.9), alpha=0.05, B=200,
                               seed=1, rule=RULE_V2, king_revisions={"gift-eval": "a"},
                               chal_revisions={"gift-eval": "a"})
    assert same.computed


def test_rule_v2_refuses_challenger_only_missing_tasks():
    from .test_bench_gate_v2 import SUITES_V2, _eight, _rows

    king = _rows()
    chal = _rows(dict.fromkeys(SUITES_V2, 0.9))
    # Omitting a challenger's worst tasks must not turn a failed suite into a pass.
    chal["gift"]["rows"] = chal["gift"]["rows"][5:]
    g = evaluate_bench_gate(king, chal, _eight(1.0), _eight(0.9, gifteval_crps=1.2),
                            alpha=0.05, B=200, seed=1, rule=RULE_V2)
    assert g.computed and not g.passed
    assert "challenger failed required task gift/0" in g.reason


# ── write-once reports, content-addressed rows, signing ──────────────────────


def test_rows_keys_are_content_addressed():
    a = validator_bench_rows_key("101", KING_PTR, "1" * 64)
    b = validator_bench_rows_key("101", KING_PTR, "2" * 64)
    assert a != b and a.endswith("/" + "1" * 64 + ".json")
    assert validator_bench_rows_key("101", KING_PTR) == (
        f"bench_v/rows/101/{hashlib.sha256(KING_PTR.encode()).hexdigest()}.json")


def test_a_published_report_is_never_overwritten(cfg):
    v = _vcfg(cfg)
    store = _Store()
    main, _ = _runner(v, store=store)
    m = _manifest(v)
    main._vbench_begin(m)
    main.process_round(m, windows=[], base_seed=7)
    # someone (a second instance) published first
    store.objects[validator_bench_report_key("101")] = "first"
    main._vbench_publish(m)
    assert store.objects[validator_bench_report_key("101")] == "first"
    assert main._vbench["mode"] == "read"


def test_main_validator_without_a_signing_wallet_refuses(cfg):
    v = _vcfg(cfg)

    def boom(_body):
        raise RuntimeError("public key only")

    pub_only = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=MAIN, sign=boom))
    with pytest.raises(RuntimeError, match="cannot sign"):
        check_main_bench_signer(v, pub_only)
    check_main_bench_signer(v, SimpleNamespace(hotkey=SimpleNamespace(
        ss58_address="5Other", sign=boom)))                    # not the main validator
    check_main_bench_signer(cfg, pub_only)                     # trainer source: inert
    # and the publish path never publishes unsigned
    store = _Store()
    main, _ = _runner(v, store=store, signer=pub_only)
    m = _manifest(v)
    main._vbench_begin(m)
    main.process_round(m, windows=[], base_seed=7)
    with pytest.raises(StorageError, match="unsigned"):
        main._vbench_publish(m)
    assert validator_bench_report_key("101") not in store.objects


# ── config ───────────────────────────────────────────────────────────────────


def test_new_v2_keys_parse_and_refuse():
    assert parse_bench_source_bounds({}) == {"bench_report_timeout_settlements": 8,
                                             "bench_budget_hours": 4.0,
                                             "bench_sweep_hours": 2.0}
    for bad in ({"bench_report_timeout_settlements": 0},
                {"bench_report_timeout_settlements": True},
                {"bench_budget_hours": 0}, {"bench_budget_hours": "4"},
                {"bench_sweep_hours": 0}, {"bench_sweep_hours": True},
                {"bench_sweep_hours": 5.0}):              # > the 4 h cap
        with pytest.raises(ValueError):
            parse_bench_source_bounds(bad)


def test_overlay_can_tune_bench_liveness():
    from cascade.shared.netconfig import check_overrides

    for key, val in (("bench_report_timeout_settlements", 2), ("bench_budget_hours", 1.0),
                     ("bench_sweep_hours", 1.0)):
        assert check_overrides({"v2": {key: val}}) == {"v2": {key: val}}


def test_manifest_validator_hotkey_must_match_the_owner():
    ok = V2Config(from_block=7200, king_ladder=True, bench_source="validator",
                  bench_source_from_block=7200)
    check_bench_source(v2=ok, owner_hotkey=MAIN, era_blocks=3600,
                       manifest_validator_hotkey=MAIN)
    check_bench_source(v2=ok, owner_hotkey=MAIN, era_blocks=3600)    # unset: fine
    with pytest.raises(ValueError, match="same main validator"):
        check_bench_source(v2=ok, owner_hotkey=MAIN, era_blocks=3600,
                           manifest_validator_hotkey="5Else")


# ── the king check ───────────────────────────────────────────────────────────


def test_king_check_init_pointer_comes_from_the_report_or_holds(cfg):
    v = _vcfg(cfg)
    store = _Store()

    class _Flaky(_Store):
        def get_text(self, key):
            if key == bench_report_key("100"):
                raise ConnectionError("503")
            return super().get_text(key)

    flaky = _Flaky()
    main, _ = _runner(v, store=flaky)
    m = replace(_manifest(v, prev="100"), warm_start_ckpt="")     # random-init epoch
    main._vbench_begin(m)
    king = m.entry_for_role("king")
    with pytest.raises(ValidatorBenchHold, match="init pointer unreadable"):
        main._v2_king_check_validator_bench(m, king, KING_PTR, {"reasons": []})
    # a reader takes the init pointer from the main validator's report — it
    # never reads the trainer's report (which could fail only for it)
    e = BenchEntry(role="init", size="", miner_hotkey="", miner_uid=-1,
                   trained_pointer=INIT_PTR, scores=BenchScores(*([2.0] * 6)))
    k = BenchEntry(role="king", size="", miner_hotkey="king_hk", miner_uid=0,
                   trained_pointer=KING_PTR, scores=BenchScores(*([1.0] * 6)))
    store.objects[validator_bench_report_key("101")] = dump_bench_report(
        BenchReport(round_id="101", created_block=m.created_block, entries=(k, e)))
    other, _ = _runner(v, store=store, signer="5Other")
    other._fetch_bench_report = lambda *a, **kw: (_ for _ in ()).throw(AssertionError)
    _record(other, store)
    other._vbench_begin(m)
    got = other._v2_king_check_validator_bench(m, king, KING_PTR, {"reasons": []})
    assert got is not None and got[2] == INIT_PTR
    # unrecorded ⇒ HOLD, never a local deferral
    third, _ = _runner(v, store=store, signer="5Third")
    third._vbench_begin(m)
    with pytest.raises(ValidatorBenchHold):
        third._v2_king_check_validator_bench(m, king, KING_PTR, {"reasons": []})


def test_king_check_uses_the_0061_rule_when_in_force(cfg):
    from cascade.validator.king_check import KC_GATE_UNCOMPUTABLE, RULE_V2

    from .test_bench_gate_v2 import SUITES_V2, _rows

    king = BenchScores(*([1.0] * 6), fev_crps=1.0, fev_mase=2.0)
    init = BenchScores(*([2.0] * 6), fev_crps=2.0, fev_mase=1.0)
    assert bench_failures(king, init) == []                     # DEC-CA-0054: six
    assert len(BENCH_FIELDS_V2) == len(BENCH_FIELDS) + 2

    v = _vcfg(cfg)
    v = replace(v, v2=replace(v.v2, bench_gate_v2_from_block=_era_start(cfg)))
    runner, _ = _runner(v)
    m = _manifest(v)
    runner._vbench_begin(m)

    def side(rows_k, rows_i):
        def fn(*a):
            runner._king_check_bench_rows = (rows_k, rows_i, None, None)
            return king, init, INIT_PTR
        return fn

    # no task rows for the init ⇒ deferred (never a pass on suite means)
    runner._v2_king_check_validator_bench = side(_rows(), None)
    rec = runner._v2_king_check(m, [], 7)
    assert rec["result"] == "deferred"
    assert rec["reasons"] == [KC_GATE_UNCOMPUTABLE + "signed bench rows missing for init"]
    seen = {}

    def pool(manifest, windows, seed, king_e, init_ptr, bench_bad, rec):
        seen["bad"] = bench_bad
        return rec

    runner._v2_king_check_pool = pool
    # the king (challenger) beats the init (reference) on every task: pass,
    # even though one FEV suite mean is worse (the strict rule would fail it)
    runner._v2_king_check_validator_bench = side(
        _rows(dict.fromkeys(SUITES_V2, 0.9)), _rows())
    rec = runner._v2_king_check(m, [], 7)
    assert rec["bench"]["rule"] == RULE_V2 and seen["bad"] == []
    gd = rec["bench"]["gate"]
    assert gd["passed"] is True and gd["avg_ratio"] == pytest.approx(0.9)
    assert gd["win_rate"] == 1.0 and gd["worst_lcb"] < 1.0
    # the king worse than the init on every task ⇒ the bench half fails
    runner._v2_king_check_validator_bench = side(
        _rows(dict.fromkeys(SUITES_V2, 1.1)), _rows())
    rec = runner._v2_king_check(m, [], 7)
    assert len(seen["bad"]) == 1 and seen["bad"][0].startswith(f"bench rule {RULE_V2}: ")
    assert rec["bench"]["gate"]["passed"] is False
    # DEC-CA-0054 in force: no rule recorded (record bytes unchanged)
    runner2, _ = _runner(_vcfg(cfg))
    runner2._vbench_begin(m)
    runner2._v2_king_check_pool = pool
    runner2._v2_king_check_validator_bench = lambda *a: (king, init, INIT_PTR)
    rec2 = runner2._v2_king_check(m, [], 7)
    assert "rule" not in rec2["bench"] and seen["bad"] == []


# ── promotion members ────────────────────────────────────────────────────────


def test_promotion_member_read_holds_on_absent_or_unreadable(cfg):
    v = _vcfg(cfg)
    runner, _ = _runner(v, store=_Store(), signer="5Other")
    assert runner._fetch_validator_bench_report("77") is None          # reign log: None
    with pytest.raises(StorageError, match="unpublished"):
        runner._fetch_validator_bench_report("77", raise_unavailable=True)

    class _Down(_Store):
        def get_text(self, key):
            raise ConnectionError("503")

    down, _ = _runner(v, store=_Down(), signer="5Other")
    with pytest.raises(StorageError, match="unavailable"):
        down._fetch_validator_bench_report("77", raise_unavailable=True)


# ── the audit ────────────────────────────────────────────────────────────────


def test_audit_flags_an_omitted_clearer(cfg):
    from bittensor.sp_core import Keypair

    from cascade.audit.checks import check_bench_coverage
    from cascade.audit.main import fetch_validator_bench

    alice = Keypair.create_from_uri("//Alice")
    v = _vcfg(cfg, owner=alice.ss58_address)
    store = _Store()
    main, _ = _runner(v, store=store, signer=alice)
    m = _manifest(v)
    _settle(main, m)
    _record(main, store)
    out = _settle(main, m)
    receipt = main.build_round_receipt(m, base_seed=7, epoch_start_block=main._epoch_start_block(m),
                                       epoch_block_hash="0x00", outcome=out, windows=[])
    vb = fetch_validator_bench(v, receipt, store)
    assert check_bench_coverage(receipt, v, vb).status == "PASS"
    # the same verdict against a report that omits the clearer
    empty = BenchReport(round_id="101", created_block=m.created_block)
    assert check_bench_coverage(receipt, v, {"report": empty}).status == "FAIL"
    assert "OMITTED" in check_bench_coverage(receipt, v, {"report": empty}).detail
    # infra-failed counts as covered
    infra = replace(empty, infra_failed=({"trained_pointer": CHAL_PTR, "role": "challenger",
                                          "reason": "sweep failed"},))
    assert check_bench_coverage(receipt, v, {"report": infra}).status == "PASS"
    assert check_bench_coverage(receipt, cfg, vb).status == "SKIP"   # trainer source


def test_audit_reproduce_compares_sampled_items(cfg):
    from bittensor.sp_core import Keypair

    from cascade.audit.main import audit_validator_bench

    alice = Keypair.create_from_uri("//Alice")
    v = _vcfg(cfg, owner=alice.ss58_address)
    store = _Store()
    main, _ = _runner(v, store=store, signer=alice)
    _settle(main, _manifest(v))
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    rows = {e.trained_pointer: json.loads(store.objects[e.rows_key]) for e in rep.entries}

    def faithful(e):
        r = json.loads(json.dumps(rows[e.trained_pointer]))
        for suite in r.values():                       # GPU noise well inside 5 %
            for x in suite["rows"]:
                x["crps"] *= 1.01
        return r, {}

    res = audit_validator_bench(v, "101", store, reproduce=True, sweep_fn=faithful)
    assert [r.status for r in res] == ["PASS"] * len(res), [r.detail for r in res]

    def forged(e):
        r = json.loads(json.dumps(rows[e.trained_pointer]))
        for suite in r.values():
            for x in suite["rows"]:
                x["mase"] *= 1.3
        return r, {}

    res = audit_validator_bench(v, "101", store, reproduce=True, sweep_fn=forged)
    assert any(r.status == "FAIL" and "do not reproduce" in r.detail for r in res)
    res = audit_validator_bench(v, "101", store, reproduce=True, sweep_fn=None)
    assert res[-1].status == "WARN"                     # torch unavailable here


# ── the trainer's record ─────────────────────────────────────────────────────


def test_trainer_records_sha_then_absent_at_the_timeout(cfg):
    from cascade.trainer.rolling import RollingScheduler, RollingState

    v = _vcfg(cfg)
    v = replace(v, v2=replace(v.v2, bench_report_timeout_settlements=2))
    store = _Store()
    sched = SimpleNamespace(cfg=v, state=RollingState(vbench_pending={"100": 0, "101": 1}),
                            _lock=__import__("threading").Lock(),
                            ops=SimpleNamespace(read_text=store.get_text))
    store.objects[validator_bench_report_key("100")] = "report-100"
    boundary = _era_start(cfg) + 3 * cfg.round.epoch_blocks
    recs, after = RollingScheduler._vbench_records(sched, "102", boundary)
    assert recs == {"100": hashlib.sha256(b"report-100").hexdigest(), "101": "absent"}
    assert after == {"102": 0}                    # this settlement is tracked next
    # a trainer-sourced settlement with nothing pending: untouched (byte-identical)
    t = SimpleNamespace(cfg=cfg, state=RollingState(), _lock=sched._lock, ops=sched.ops)
    assert RollingScheduler._vbench_records(t, "102", boundary) == ({}, None)
    # a read failure neither counts nor records
    class _Down:
        def get_text(self, key):
            raise ConnectionError("503")

    sched.ops = SimpleNamespace(read_text=_Down().get_text)
    sched.state = RollingState(vbench_pending={"100": 1})
    recs, after = RollingScheduler._vbench_records(sched, "102", boundary)
    assert recs == {} and after == {"100": 1, "102": 0}


# ── legacy byte-identity ─────────────────────────────────────────────────────


def test_manifest_and_report_bytes_unchanged_without_the_new_fields(cfg):
    m = _manifest(cfg)
    text = dump_manifest(m)
    assert "validator_bench_reports" not in text
    assert load_manifest(text).validator_bench_reports is None
    with_rec = replace(m, validator_bench_reports={"100": "absent"})
    assert load_manifest(dump_manifest(with_rec)).validator_bench_reports == {"100": "absent"}
    assert with_rec.canonical_body() != m.canonical_body()
    e = BenchEntry(role="king", size="", miner_hotkey="k", miner_uid=0,
                   trained_pointer=KING_PTR, scores=BenchScores(*([1.0] * 6)))
    r = BenchReport(round_id="1", created_block=5, entries=(e,))
    body = json.loads(r.canonical_body())
    assert "infra_failed" not in body and "data_revisions" not in body["entries"][0]
    assert load_bench_report(dump_bench_report(r)) == r


def test_testnet_keeps_the_trainer_source_and_defaults(cfg):
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    root = Path(__file__).resolve().parents[2]
    test = load_chain_config(root / "chain.testnet.toml")
    assert test.v2.bench_source == "trainer"
    assert test.v2.bench_report_timeout_settlements == 8 and test.v2.bench_budget_hours == 4.0
    runner, calls = _runner(replace(cfg, scoring=replace(cfg.scoring, bench_gate_mode="enforce")))
    m = _manifest(cfg)
    runner._vbench_begin(m)
    assert runner._vbench is None and runner._vbench_receipt_kw(m) == {}
