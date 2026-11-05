"""An infra-failed challenger sweep (our timeout / OOM / store / sidecar
crash — never the miner's loss) is carried into the settlement that records
the report, exactly like a budget-deferred one, so the clearer is judged
again instead of silently dropped (2026-11-03: a +0.0072 clearer's sweep
timed out on our upload). King / init rows never ride; the carry cap holds."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from cascade.shared.bench_report import VBENCH_MAX_CARRIES, BenchReport, dump_bench_report
from cascade.trainer.rolling import FinishedLeg, RollingScheduler, RollingState

from .test_validator_bench import CHAL_PTR as CHAL
from .test_validator_bench import KING_PTR as KING


def _leg(hotkey, ptr, era=7):
    return FinishedLeg(hotkey=hotkey, uid=2, ref="cid", era_index=era, started_block=10,
                       reveal_block=5, entry={"miner_hotkey": hotkey, "miner_uid": 2,
                                              "role": "challenger", "gen_ref": "g",
                                              "trained_pointer": ptr, "corpus_digest": "c",
                                              "train_block": 9})


def _report(**lists):
    return dump_bench_report(BenchReport(round_id="101", created_block=5, **lists))


def _sched(*legs):
    return SimpleNamespace(state=RollingState(published=list(legs)), _lock=threading.Lock())


def test_an_infra_failed_challenger_is_carried_like_a_deferred_one():
    leg = _leg("chal", CHAL)
    text = _report(infra_failed=({"trained_pointer": CHAL, "role": "challenger",
                                  "reason": "sweep failed (2 attempts): sidecar produced no report"},))
    cur = SimpleNamespace(index=7, king_hotkey="king_hk")
    assert RollingScheduler._vbench_carry(_sched(leg), cur, [], {"101": text}) == [leg]


def test_infra_failed_king_or_init_rows_never_ride():
    king_leg = _leg("king_hk", KING)
    text = _report(infra_failed=({"trained_pointer": KING, "role": "king", "reason": "x"},
                                 {"trained_pointer": KING, "role": "init", "reason": "x"}))
    cur = SimpleNamespace(index=7, king_hotkey="someone_else")
    assert RollingScheduler._vbench_carry(_sched(king_leg), cur, [], {"101": text}) == []


def test_infra_carry_is_bounded_and_one_per_hotkey():
    leg = _leg("chal", CHAL)
    both = _report(infra_failed=({"trained_pointer": CHAL, "role": "challenger", "reason": "x"},),
                   budget_deferred=({"trained_pointer": CHAL, "role": "challenger",
                                     "reason": "cap"},))
    cur = SimpleNamespace(index=7, king_hotkey="king_hk")
    assert RollingScheduler._vbench_carry(_sched(leg), cur, [], {"101": both}) == [leg]
    s = _sched(leg)
    s.state.vbench_carries = {CHAL: VBENCH_MAX_CARRIES}
    assert RollingScheduler._vbench_carry(s, cur, [], {"101": both}) == []
    # a fresh leg of the same hotkey in this settlement wins over the carry
    assert RollingScheduler._vbench_carry(_sched(leg), cur, [leg], {"101": both}) == []
    # never across eras
    assert RollingScheduler._vbench_carry(_sched(leg), SimpleNamespace(index=8, king_hotkey="k"),
                                          [], {"101": both}) == []
