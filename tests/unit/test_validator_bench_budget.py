"""DEC-CA-0063 items 9-10: per-clearer sweep allowances under a settlement
cap, budget-deferred clearers carried into a later settlement, and sweep
failures classified (checkpoint fault = a loss on every validator; infra
never)."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace

from cascade.eval.benchmarks import (
    SWEEP_FAULT_CHECKPOINT,
    SWEEP_FAULT_INFRA,
    classify_sweep_failure,
)
from cascade.shared.bench_report import (
    VBENCH_MAX_CARRIES,
    BenchReport,
    dump_bench_report,
    load_bench_report,
    validator_bench_report_key,
)
from cascade.shared.manifest import TrainedEntry

from .test_validator_bench import (
    CHAL2_PTR,
    CHAL_PTR,
    CID,
    KING_PTR,
    _manifest,
    _record,
    _runner,
    _settle,
    _sidecar_report,
    _Store,
    _vcfg,
)
from .test_validator_bench_liveness import _two_chal_manifest


def _cfg(cfg, *, cap=4.0, allowance=2.0):
    v = _vcfg(cfg)
    return replace(v, v2=replace(v.v2, bench_budget_hours=cap, bench_sweep_hours=allowance))


def _timed(runner, hours: dict):
    """Every sweep of pointer p advances the runner's clock by hours[p]."""
    clock = {"t": 0.0}
    runner.vbench_clock = lambda: clock["t"]
    inner = runner.bench_sweep_fn

    def sweep(entry):
        clock["t"] += hours.get(entry.trained_pointer, 0.0) * 3600.0
        return inner(entry)

    runner.bench_sweep_fn = sweep
    return clock


def _nonfinite(scale=0.9):
    r = _sidecar_report(scale)
    r["suites"][0]["metrics"]["crps"] = float("nan")
    return r


def _oom():
    r = _sidecar_report(0.9)
    r["suites"][1] = {"suite": "boom", "status": "error", "detail": "no BOOM configs scored",
                      "skipped": [{"full": "boom/0", "stage": "score",
                                   "reason": "RuntimeError: CUDA out of memory"}]}
    return r


def _score_errors():
    r = _sidecar_report(0.9)
    r["suites"][2] = {"suite": "time", "status": "error", "detail": "no TIME tasks scored",
                      "skipped": [{"full": f"time/{i}", "stage": "score",
                                   "reason": "RuntimeError: size mismatch for head.weight"}
                                  for i in range(3)]}
    return r


# ── item 9a: per-clearer allowance, king outside it ──────────────────────────


def test_each_clearer_has_its_own_allowance(cfg):
    # chal_hk burns its whole 2 h allowance failing; chal2 still gets a full
    # one (under the old shared budget it would have inherited what was left)
    v = _cfg(cfg, cap=6.0, allowance=2.0)
    store = _Store()
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: None,
               CHAL2_PTR: _sidecar_report(0.9)}
    main, calls = _runner(v, store=store, reports=reports)
    _timed(main, {CHAL_PTR: 1.0, CHAL2_PTR: 1.9, KING_PTR: 1.9})
    out = _settle(main, _two_chal_manifest(v))
    assert calls == [CHAL_PTR, CHAL_PTR, CHAL2_PTR, KING_PTR]
    assert out.transition.new_king_hotkey == "chal2"
    assert "bench_infra chal_hk" in out.transition.note


def test_king_sweep_never_spends_the_clearers_allowance(cfg):
    # each side takes 1.9 h of a 2 h allowance: 3.8 h in all, inside the cap
    v = _cfg(cfg, cap=4.0, allowance=2.0)
    main, calls = _runner(v, store=_Store())
    _timed(main, {CHAL_PTR: 1.9, KING_PTR: 1.9})
    out = _settle(main, _manifest(v))
    assert sorted(calls) == sorted([CHAL_PTR, KING_PTR])
    assert out.result.challenger_wins_round


def test_cap_reserves_the_kings_allowance_only_when_it_needs_a_sweep(cfg):
    v = _cfg(cfg, cap=3.0, allowance=2.0)
    store = _Store()
    main, calls = _runner(v, store=store)
    out = _settle(main, _manifest(v))            # 2 h + the king's 2 h > the 3 h cap
    assert calls == [] and out.result.inconclusive
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.deferred_reason(CHAL_PTR) and rep.entries == ()
    # the king cached from an earlier settlement: the clearer alone fits
    main2, calls2 = _runner(v, store=_Store())
    main2._vbench_cache_put(KING_PTR, {k: 1.0 for k in ("gifteval_crps", "gifteval_mase",
                                                         "boom_crps", "boom_mase",
                                                         "time_crps", "time_mase")},
                            json.loads(json.dumps({"gift": {"status": "ok", "rows": []}})))
    main2._vbench_begin(_manifest(v))
    main2._vbench_side(TrainedEntry("chal_hk", 1, "challenger", CID, CHAL_PTR, "d", 10),
                       role="challenger",
                       reserve_for=TrainedEntry("king_hk", 0, "king", CID, KING_PTR, "d", 10))
    assert calls2 == [CHAL_PTR]


# ── item 9b: budget_deferred is re-judged at a later settlement ──────────────


def test_budget_deferred_clearer_is_skipped_then_rejudged_when_carried(cfg):
    v = _cfg(cfg, cap=4.0, allowance=2.0)
    store = _Store()
    # chal_hk clears the pool first but fails the bench gate (worse benches);
    # its sweep + the king's spend the cap, so chal2 is deferred
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: _sidecar_report(1.6),
               CHAL2_PTR: _sidecar_report(0.9)}
    main, calls = _runner(v, store=store, reports=reports)
    main.state = replace(main.state, streaks={"chal2": 1})
    _timed(main, {CHAL_PTR: 2.0, KING_PTR: 2.0, CHAL2_PTR: 1.0})
    m = _two_chal_manifest(v)
    out = _settle(main, m)
    assert CHAL2_PTR not in calls and not out.result.challenger_wins_round
    assert out.transition.state.streaks.get("chal2") == 1      # deferred: never a loss
    assert "bench_deferred chal2" in out.transition.note
    text = store.objects[validator_bench_report_key("101")]
    rep = load_bench_report(text)
    assert [d["trained_pointer"] for d in rep.budget_deferred] == [CHAL2_PTR]
    # every reader applies the same deferral from the recorded bytes
    other, ocalls = _runner(v, store=store, signer="5Other")
    other.state = replace(other.state, streaks={"chal2": 1})
    _record(other, store)
    o = _settle(other, m)
    assert ocalls == [] and o.transition.note == out.transition.note
    assert o.transition.state == out.transition.state

    # the trainer carries chal2's leg into the settlement that records report 101
    from cascade.trainer.rolling import FinishedLeg, RollingScheduler, RollingState

    chal2 = next(e for e in m.entries if e.miner_hotkey == "chal2")
    leg = FinishedLeg(hotkey="chal2", uid=2, ref=CID, era_index=7, started_block=10,
                      reveal_block=5, entry=asdict(chal2))
    sched = SimpleNamespace(state=RollingState(published=[leg]), _lock=threading.Lock())
    cur = SimpleNamespace(index=7, king_hotkey="king_hk")
    assert RollingScheduler._vbench_carry(sched, cur, [], {"101": text}) == [leg]
    # not across eras, not over a fresh leg of the same hotkey, bounded
    assert RollingScheduler._vbench_carry(sched, SimpleNamespace(index=8, king_hotkey="k"),
                                          [], {"101": text}) == []
    assert RollingScheduler._vbench_carry(sched, cur, [leg], {"101": text}) == []
    sched.state.vbench_carries = {CHAL2_PTR: VBENCH_MAX_CARRIES}
    assert RollingScheduler._vbench_carry(sched, cur, [], {"101": text}) == []

    # settlement 102 carries chal2 alone; it is swept first and judged on its merits
    m2 = replace(_manifest(v, rid="102", prev="101", offset=3),
                 entries=[m.entries[0], chal2])
    main2, calls2 = _runner(v, store=store, reports=reports)
    out2 = _settle(main2, m2)
    assert CHAL2_PTR in calls2 and out2.transition.new_king_hotkey == "chal2"


def test_trainer_carry_state_round_trips_and_stays_off_the_file_when_empty(tmp_path):
    from cascade.trainer.rolling import RollingState, load_state, save_state

    p = tmp_path / "s.json"
    save_state(p, RollingState())
    assert "vbench_carries" not in p.read_text()
    save_state(p, RollingState(vbench_carries={CHAL_PTR: 1}))
    assert load_state(p).vbench_carries == {CHAL_PTR: 1}


# ── item 10: classification ──────────────────────────────────────────────────


def test_classifier_is_conservative():
    assert classify_sweep_failure(None)[0] == SWEEP_FAULT_INFRA
    assert classify_sweep_failure(_nonfinite())[0] == SWEEP_FAULT_CHECKPOINT
    assert classify_sweep_failure(_score_errors())[0] == SWEEP_FAULT_CHECKPOINT
    assert classify_sweep_failure(_oom())[0] == SWEEP_FAULT_INFRA
    # a checkpoint-looking suite next to an environmental one ⇒ infra
    both = _score_errors()
    both["suites"][1] = _oom()["suites"][1]
    assert classify_sweep_failure(both)[0] == SWEEP_FAULT_INFRA
    # data-side (load stage) failures are never the checkpoint's
    load = _score_errors()
    for x in load["suites"][2]["skipped"]:
        x["stage"] = "load"
    assert classify_sweep_failure(load)[0] == SWEEP_FAULT_INFRA
    skipped = _sidecar_report(0.9)
    skipped["suites"][0] = {"suite": "gift-eval", "status": "skipped", "detail": "no data"}
    assert classify_sweep_failure(skipped)[0] == SWEEP_FAULT_INFRA


def test_checkpoint_fault_is_a_loss_on_every_validator(cfg):
    v = _cfg(cfg)
    store = _Store()
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: _nonfinite()}
    main, calls = _runner(v, store=store, reports=reports)
    main.state = replace(main.state, streaks={"chal_hk": 1})
    m = _manifest(v)
    out = _settle(main, m)
    assert calls == [CHAL_PTR, CHAL_PTR]                 # reproduced on the retry
    assert not out.result.challenger_wins_round and not out.result.inconclusive
    assert out.result.bench_gate_passed is False
    assert out.transition.state.streaks.get("chal_hk", 0) == 0   # a loss
    assert "bench_fault chal_hk: gift-eval: non-finite" in out.transition.note
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.fault_reason(CHAL_PTR) and rep.infra_failed == ()
    other, ocalls = _runner(v, store=store, signer="5Other")
    other.state = replace(other.state, streaks={"chal_hk": 1})
    _record(other, store)
    o = _settle(other, m)
    assert ocalls == [] and o.result == out.result and o.transition.state == out.transition.state


def test_one_infra_attempt_keeps_the_failure_infra(cfg):
    v = _cfg(cfg)
    store = _Store()
    seq = [_nonfinite(), _oom()]
    main, calls = _runner(v, store=store)
    inner = main.bench_sweep_fn
    main.bench_sweep_fn = lambda e: seq.pop(0) if e.trained_pointer == CHAL_PTR else inner(e)
    main.state = replace(main.state, streaks={"chal_hk": 1})
    out = _settle(main, _manifest(v))
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.checkpoint_fault == () and rep.infra_reason(CHAL_PTR)
    assert out.result.inconclusive and out.transition.state.streaks == {"chal_hk": 1}


def test_oom_stays_infra(cfg):
    v = _cfg(cfg)
    store = _Store()
    main, _ = _runner(v, store=store, reports={KING_PTR: _sidecar_report(1.0),
                                               CHAL_PTR: _oom()})
    main.state = replace(main.state, streaks={"chal_hk": 1})
    out = _settle(main, _manifest(v))
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert "environmental (cuda)" in rep.infra_reason(CHAL_PTR)
    assert rep.checkpoint_fault == () and out.transition.state.streaks == {"chal_hk": 1}
    assert "bench_infra chal_hk" in out.transition.note


def test_a_kings_checkpoint_fault_is_infra_for_the_clearer(cfg):
    v = _cfg(cfg)
    store = _Store()
    main, _ = _runner(v, store=store, reports={KING_PTR: _nonfinite(1.0),
                                               CHAL_PTR: _sidecar_report(0.9)})
    main.state = replace(main.state, streaks={"chal_hk": 1})
    out = _settle(main, _manifest(v))
    assert out.result.inconclusive and out.transition.state.streaks == {"chal_hk": 1}
    assert "bench_infra chal_hk: checkpoint fault" in out.transition.note


# ── the audit ────────────────────────────────────────────────────────────────


def test_audit_replays_fault_and_deferral(cfg):
    from bittensor.sp_core import Keypair

    from cascade.audit.checks import check_bench_coverage, check_bench_gate
    from cascade.audit.main import fetch_validator_bench, validator_bench_rows

    alice = Keypair.create_from_uri("//Alice")
    v = replace(_cfg(cfg), netconfig=replace(_cfg(cfg).netconfig,
                                             owner_hotkey=alice.ss58_address))
    store = _Store()
    main, _ = _runner(v, store=store, signer=alice,
                      reports={KING_PTR: _sidecar_report(1.0), CHAL_PTR: _nonfinite()})
    m = _manifest(v)
    _settle(main, m)
    _record(main, store)
    out = _settle(main, m)
    receipt = main.build_round_receipt(m, base_seed=7, epoch_start_block=main._epoch_start_block(m),
                                       epoch_block_hash="0x00", outcome=out, windows=[])
    vb = fetch_validator_bench(v, receipt, store)
    assert check_bench_coverage(receipt, v, vb).status == "PASS"
    bench = validator_bench_rows(vb["report"], store)
    assert check_bench_gate(receipt, v, bench).status == "PASS"
    # a verdict that ignored the signed fault is flagged
    nofault = replace(vb["report"], checkpoint_fault=())
    assert check_bench_coverage(receipt, v, {"report": nofault}).status == "FAIL"
    assert check_bench_gate(receipt, v, validator_bench_rows(nofault, store)).status == "FAIL"
    # a report listing a clearer deferred that the note does not skip is flagged
    deferred = replace(nofault, budget_deferred=({"trained_pointer": CHAL_PTR,
                                                  "role": "challenger", "reason": "cap"},))
    res = check_bench_coverage(receipt, v, {"report": deferred})
    assert res.status == "FAIL" and "does not apply" in res.detail


# ── byte-identity ────────────────────────────────────────────────────────────


def test_report_bytes_unchanged_without_the_new_lists():
    r = BenchReport(round_id="1", created_block=5)
    body = json.loads(r.canonical_body())
    assert "budget_deferred" not in body and "checkpoint_fault" not in body
    full = replace(r, budget_deferred=({"trained_pointer": "p", "role": "challenger",
                                        "reason": "cap"},),
                   checkpoint_fault=({"trained_pointer": "q", "role": "challenger",
                                      "reason": "nan"},))
    assert full.canonical_body() != r.canonical_body()
    assert load_bench_report(dump_bench_report(full)) == full


def test_testnet_and_trainer_source_untouched(cfg):
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    root = Path(__file__).resolve().parents[2]
    test = load_chain_config(root / "chain.testnet.toml")
    assert test.v2.bench_source == "trainer" and test.v2.bench_sweep_hours == 2.0
    main = load_chain_config(root / "chain.toml")
    assert main.v2.bench_sweep_hours == 2.0 and main.v2.bench_budget_hours == 6.0
    runner, calls = _runner(replace(cfg, scoring=replace(cfg.scoring,
                                                         bench_gate_mode="enforce")))
    m = _manifest(cfg)
    runner._vbench_begin(m)
    assert runner._vbench is None and calls == []
