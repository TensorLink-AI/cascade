"""Elastic bench host on the main validator: nothing about renting a pod may
ever cost a miner anything.

These run the real settlement path (``_handle_manifest`` → private pool →
bench gate → publish) with a scripted lease. The invariants, for every way a
rent can be slow or fail:

* while the pod is not ready the settlement HOLDS: no verdict, no report, no
  streak / king / weight change — the same hold every validator already
  takes for a pending report;
* the hold never re-scores the private pool on every poll;
* once the pod is ready the verdict is exactly the static-host verdict;
* past the bounded wait a clearer is published ``budget_deferred`` — never
  swept, never a loss, carried by the trainer — never ``infra_failed``;
* a pod that dies under a sweep re-rents and holds again; it is never
  published as the clearer's infra failure;
* rows from different hardware never pair: a cached king swept on another
  fingerprint is re-swept;
* the sweep never falls back to the validator box's own device.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.bench_report import load_bench_report, validator_bench_report_key
from cascade.validator.bench_lease import HW_KEY
from cascade.validator.state import genesis
from cascade.validator.validator_bench import HOST_DEFERRED_REASON, ValidatorBenchHold

from .test_validator_bench import (
    CHAL2_PTR,
    CHAL_PTR,
    KING_PTR,
    _manifest,
    _record,
    _runner,
    _settle,
    _sidecar_report,
    _Store,
    _vcfg,
    _wire_loop,
)

HW_A = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition|driver 580|torch 2.12.1+cu130 cuda 13.0"
HW_B = "NVIDIA GeForce RTX 5090|driver 580|torch 2.12.1+cu130 cuda 13.0"


class ScriptedLease:
    """The validator-facing surface of BenchLease, scripted per test."""

    def __init__(self, *, ready=False, expired=False, hw=HW_A, reachable=True):
        self._ready, self._expired, self._hw, self._reachable = ready, expired, hw, reachable
        self.needs = 0
        self.lost: list[str] = []
        self.ended = 0
        self.ticks = 0
        self.oks = 0

    def need(self):
        self.needs += 1
        return self._ready

    def ready(self):
        return self._ready

    def hw(self):
        return self._hw if self._ready else ""

    def expired(self):
        return self._expired and not self._ready

    def status(self):
        return "ready" if self._ready else ("expired" if self._expired else "renting")

    def reachable(self):
        return self._ready and self._reachable

    def host_lost(self, why):
        self.lost.append(why)
        self._ready = False

    def sweep_ok(self):
        self.oks += 1

    def end_need(self):
        self.ended += 1

    def tick(self):
        self.ticks += 1


def _main(cfg, lease, **kw):
    v = _vcfg(cfg)
    store = kw.pop("store", None) or _Store()
    runner, calls = _runner(v, store=store, **kw)
    runner.bench_lease = lease
    evals = []
    inner = runner.evaluate_fn

    def counted(entry, windows):
        evals.append(entry.role)
        return inner(entry, windows)

    runner.evaluate_fn = counted
    return v, runner, calls, store, evals


# ── holding while the pod is rented ──────────────────────────────────────────


def test_pod_not_ready_holds_the_settlement_with_nothing_judged(cfg):
    lease = ScriptedLease(ready=False)
    v, main, calls, store, _ = _main(cfg, lease)
    main.state = replace(main.state, streaks={"chal_hk": 2})
    _wire_loop(main)
    before = main.state
    m = _manifest(v)
    assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    assert main.state == before                          # no king / streak / tenure move
    assert "101|validator_bench" in main._read_holds     # an ordinary hold
    assert validator_bench_report_key("101") not in store.objects   # nothing published
    assert calls == [] and lease.needs >= 1


def test_held_polls_do_not_rescore_the_private_pool(cfg):
    lease = ScriptedLease(ready=False)
    v, main, _, _, evals = _main(cfg, lease)
    _wire_loop(main)
    m = _manifest(v)
    main._handle_manifest(object(), object(), m, "d" * 64)
    scored = len(evals)
    assert scored > 0
    for _ in range(20):                                   # 20 polls while it rents
        assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    assert len(evals) == scored and lease.ticks == 20


def test_ready_pod_gives_exactly_the_static_host_verdict(cfg):
    lease = ScriptedLease(ready=False)
    v, main, calls, store, _ = _main(cfg, lease)
    _wire_loop(main)
    m = _manifest(v)
    for _ in range(3):
        assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    lease._ready = True
    # the main validator publishes, then (like everyone) holds for the record
    assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    assert sorted(calls) == sorted([KING_PTR, CHAL_PTR])
    _record(main, store)
    assert main._handle_manifest(object(), object(), m, "d" * 64) is True
    assert main.state.king_hotkey == "chal_hk"            # the dethrone stands

    # the same settlement on a static host decides identically
    static_store = _Store()
    sv, static, _, _, _ = _main(cfg, None, store=static_store)
    _wire_loop(static)
    static._handle_manifest(object(), object(), m, "d" * 64)
    _record(static, static_store)
    assert static._handle_manifest(object(), object(), m, "d" * 64) is True
    assert static.state.king_hotkey == main.state.king_hotkey
    assert static.state.streaks == main.state.streaks


def test_swept_rows_carry_the_hardware_fingerprint(cfg):
    lease = ScriptedLease(ready=True)
    v, main, _, store, _ = _main(cfg, lease)
    _settle(main, _manifest(v))
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert {e.data_revisions[HW_KEY] for e in rep.entries} == {HW_A}
    assert lease.oks == 2


# ── the bounded wait spent ───────────────────────────────────────────────────


def test_no_pod_within_the_bound_defers_the_clearer_never_a_loss(cfg):
    lease = ScriptedLease(ready=False, expired=True)
    v, main, calls, store, _ = _main(cfg, lease)
    main.state = replace(main.state, streaks={"chal_hk": 2})
    out = _settle(main, _manifest(v))
    assert calls == []
    assert out.result.inconclusive and not out.result.challenger_wins_round
    assert out.transition.state.streaks == {"chal_hk": 2}     # untouched
    assert out.transition.state.king_hotkey == "king_hk"
    assert "bench_deferred chal_hk" in out.transition.note
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_failed == ()                             # never infra
    assert rep.deferred_reason(CHAL_PTR) == HOST_DEFERRED_REASON
    assert lease.ended == 1                                   # the next need waits afresh


def test_deferred_clearer_is_judged_normally_once_a_pod_is_back(cfg):
    lease = ScriptedLease(ready=False, expired=True)
    v, main, calls, store, _ = _main(cfg, lease)
    _settle(main, _manifest(v))
    lease._ready, lease._expired = True, False
    main.state = genesis("king_hk", 0)
    out = _settle(main, _manifest(v, rid="102", prev="101", offset=3))
    assert out.result.challenger_wins_round and out.result.bench_gate_passed is True
    assert sorted(calls) == sorted([KING_PTR, CHAL_PTR])


def test_king_check_without_a_pod_holds_then_defers_as_infra(cfg):
    lease = ScriptedLease(ready=False)
    v, main, calls, _, _ = _main(cfg, lease)
    m = _manifest(v)
    main._vbench_begin(m)
    king = m.entry_for_role("king")
    with pytest.raises(ValidatorBenchHold, match="bench_host_pending"):
        main._v2_king_check_validator_bench(m, king, KING_PTR,
                                            {"reasons": [], "bench_round": ""})
    lease._expired = True
    main._vbench_begin(m)
    rec = {"reasons": [], "bench_round": ""}
    assert main._v2_king_check_validator_bench(m, king, KING_PTR, rec) is None
    assert rec["reasons"] and "bench host unavailable" in rec["reasons"][0]
    assert calls == []


# ── a pod lost under a sweep ─────────────────────────────────────────────────


def test_pod_dying_mid_sweep_re_rents_and_holds_instead_of_infra(cfg):
    lease = ScriptedLease(ready=True, reachable=False)
    reports = {KING_PTR: None, CHAL_PTR: None}             # the sidecar never answers
    v, main, calls, store, _ = _main(cfg, lease, reports=reports)
    main.state = replace(main.state, streaks={"chal_hk": 1})
    _wire_loop(main)
    before = main.state
    m = _manifest(v)
    assert main._handle_manifest(object(), object(), m, "d" * 64) is False
    assert main.state == before and lease.lost
    assert validator_bench_report_key("101") not in store.objects     # never published
    # a replacement pod: the same settlement is judged normally
    lease._ready, lease._reachable = True, True
    main.bench_sweep_fn = lambda e: _sidecar_report(1.0 if e.trained_pointer == KING_PTR
                                                    else 0.9)
    assert main._handle_manifest(object(), object(), m, "d" * 64) is False   # published
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_failed == () and len(rep.entries) == 2


def test_failure_on_a_healthy_pod_is_an_ordinary_infra_failure(cfg):
    lease = ScriptedLease(ready=True, reachable=True)
    reports = {KING_PTR: None, CHAL_PTR: None}
    v, main, _, store, _ = _main(cfg, lease, reports=reports)
    out = _settle(main, _manifest(v))
    assert not lease.lost
    assert out.result.inconclusive
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.infra_reason(CHAL_PTR) is not None              # never a loss either


# ── hardware provenance ──────────────────────────────────────────────────────


def test_cached_king_from_other_hardware_is_re_swept(cfg):
    lease = ScriptedLease(ready=True, hw=HW_A)
    v, main, calls, _, _ = _main(cfg, lease)
    _settle(main, _manifest(v))
    assert calls.count(KING_PTR) == 1
    lease._hw = HW_B                                          # a different card next time
    main.state = genesis("king_hk", 0)
    _settle(main, _manifest(v, rid="102", chal_ptr=CHAL2_PTR, chal_hk="chal2", prev="101",
                            offset=3))
    assert calls.count(KING_PTR) == 2                         # re-swept on HW_B


def test_cached_king_on_the_same_hardware_is_reused(cfg):
    lease = ScriptedLease(ready=True, hw=HW_A)
    v, main, calls, _, _ = _main(cfg, lease)
    _settle(main, _manifest(v))
    main.state = genesis("king_hk", 0)
    _settle(main, _manifest(v, rid="102", chal_ptr=CHAL2_PTR, chal_hk="chal2", prev="101",
                            offset=3))
    assert calls.count(KING_PTR) == 1


def test_cached_king_from_a_static_host_is_never_reused_by_the_lease(cfg):
    v, main, calls, _, _ = _main(cfg, None)
    _settle(main, _manifest(v))                               # standing pod, no fingerprint
    main.bench_lease = ScriptedLease(ready=True, hw=HW_A)
    main.state = genesis("king_hk", 0)
    _settle(main, _manifest(v, rid="102", chal_ptr=CHAL2_PTR, chal_hk="chal2", prev="101",
                            offset=3))
    assert calls.count(KING_PTR) == 2


# ── never the validator box's own device ─────────────────────────────────────


def test_no_local_fallback_with_a_lease(cfg):
    lease = ScriptedLease(ready=True)
    v, main, _, _, _ = _main(cfg, lease)
    main.eval_host_fn = lambda: None
    m = _manifest(v)
    with pytest.raises(RuntimeError, match="no bench host published"):
        main._vbench_run_sidecar_many([m.entry_for_role("challenger")], require_fev=False)


# ── everyone else is untouched ───────────────────────────────────────────────


def test_non_main_validator_never_consults_a_lease(cfg):
    v = _vcfg(cfg)
    store = _Store()
    other, calls = _runner(v, store=store, signer="5Other")
    other.bench_lease = lease = ScriptedLease(ready=False)
    _wire_loop(other)
    m = _manifest(v)
    assert other._handle_manifest(object(), object(), m, "d" * 64) is False
    assert lease.needs == 0 and calls == []


def test_no_clearer_never_rents(cfg):
    lease = ScriptedLease(ready=False)
    v, main, calls, store, _ = _main(cfg, lease, chal_scale=1.3)    # challenger loses
    out = _settle(main, _manifest(v))
    assert out is not None and lease.needs == 0 and calls == []
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.entries == ()

