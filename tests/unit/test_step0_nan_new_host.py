"""A step-0, zero-token non-finite loss is a suspected GPU fault the first time.

2026-10-08 19:01: the era-2567 king leg NaN'd at step 0 on RTX4090 host
146.120.227.147; the same-pod retry re-attached to the dead run and failed
the same way; only after rotation did the leg train clean elsewhere. For a
payer challenger that path would have spent the miner's shot. The first
step-0 NaN is now infra-side (unburned, host quarantined, retried on a
different machine); a repeat, or any NaN after real training, is the miner's.
"""

from __future__ import annotations

import math

import pytest

from cascade.trainer.corpus import CorpusError
from cascade.trainer.loop import (
    NAN0_CLASS,
    STALL_CLASS,
    classify_funded_worker_failure,
    is_step0_divergence,
)
from cascade.trainer.rolling import king_pod_should_rotate
from cascade.trainer.toto2_trainer import check_loss_finite


def _diverged(step: int, tokens: int) -> str:
    with pytest.raises(CorpusError) as ei:
        check_loss_finite(math.nan, step=step, tokens=tokens)
    return f"remote challenger on funded-x: miner submission rejected: {ei.value}"


STEP0 = _diverged(0, 0)
LATE = _diverged(150, 39321600)


def test_signature_is_narrow():
    assert is_step0_divergence(STEP0)
    assert not is_step0_divergence(LATE)                 # real training happened
    assert not is_step0_divergence(_diverged(0, 4096))   # tokens were consumed
    assert not is_step0_divergence("generator_stalled: no series for 1800s")
    assert not is_step0_divergence(None)


def test_first_step0_nan_is_infra_and_unburned():
    assert classify_funded_worker_failure(3, STEP0, stalled_before=False) \
        == (False, NAN0_CLASS, False)


def test_second_step0_nan_is_the_miners_shot():
    assert classify_funded_worker_failure(3, STEP0, stalled_before=False,
                                          nan0_before=True) == (True, "generator", False)


def test_nan_after_real_steps_stays_the_miners_fault():
    assert classify_funded_worker_failure(3, LATE, stalled_before=False) \
        == (True, "generator", False)
    # ...even right after a step-0 NaN, and a stall exemption is unaffected
    assert classify_funded_worker_failure(3, LATE, stalled_before=False,
                                          nan0_before=True) == (True, "generator", False)
    stall = "miner submission rejected: generator_stalled: no series for 1800s"
    assert classify_funded_worker_failure(3, stall, stalled_before=False,
                                          nan0_before=True) == (False, STALL_CLASS, False)


# ── funded challenger leg: quarantine the host, requeue unburned ─────────────

class _NanDisp:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def dispatch(self, host, **kw):
        from cascade.trainer.remote import RemoteDispatchError

        self.calls.append(host)
        raise RemoteDispatchError(self.text, returncode=3)


def _run_leg(tmp_path, monkeypatch, text, *, prior_class=""):
    from tests.unit.test_funded_pod_wiring import REF, _challenger, _leg_runner

    disp = _NanDisp(text)
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    quarantined = []
    monkeypatch.setattr("cascade.trainer.loop._quarantine_lane_host",
                        lambda host, reason: quarantined.append((host.host, reason)))
    if prior_class:
        q = runner._funded_queue()
        q.add("hkA", REF, reveal_block=1)
        q.requeue("hkA", error="earlier leg", error_class=prior_class, burn_attempt=False)
    from cascade.trainer.remote import RemoteDispatchError

    with pytest.raises(RemoteDispatchError):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100,
                               contract, "", warm_start_ref=None)
    return runner, torn, quarantined


def test_first_step0_nan_quarantines_the_host_and_settles_unburned(tmp_path, monkeypatch):
    runner, torn, quarantined = _run_leg(tmp_path, monkeypatch, STEP0)
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, NAN0_CLASS, False)
    assert [h for h, _ in quarantined] == ["10.9.9.9"]   # next rent lands elsewhere
    assert torn == ["cascade-n91-777-funded-hka-0"]       # the pod is never re-attached


def test_repeat_step0_nan_on_the_next_host_is_the_miners_fault(tmp_path, monkeypatch):
    runner, torn, quarantined = _run_leg(tmp_path, monkeypatch, STEP0, prior_class=NAN0_CLASS)
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls) == (True, "generator")
    assert quarantined == []                              # no healthy host blamed


def test_late_nan_on_a_leg_is_unchanged(tmp_path, monkeypatch):
    runner, torn, quarantined = _run_leg(tmp_path, monkeypatch, LATE)
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls) == (True, "generator")
    assert quarantined == []


# ── king leg: rotate at once on the era's first step-0 NaN ───────────────────

def test_king_rotates_at_once_on_a_step0_nan_not_after_a_same_pod_retry():
    exc = RuntimeError(STEP0)
    assert king_pod_should_rotate(exc, 1)                 # first failure: rotate, no re-attach
    # once per era: a generator that NaNs everywhere falls back to the budget
    assert not king_pod_should_rotate(exc, 1, nan0_rotated=True)
    assert king_pod_should_rotate(exc, 2, nan0_rotated=True)
    # a late NaN keeps the ordinary same-pod retry
    assert not king_pod_should_rotate(RuntimeError(LATE), 1)


def test_king_leg_step0_nan_rotates_once_per_era(cfg, tmp_path):
    from tests.unit.test_rolling import Clock, FakeClient, _advance, _armed, _join, _sched

    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.fail_king = True
    ops.king_exc = RuntimeError(STEP0)
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    sched.tick(client, b0)
    _join(sched)
    cur = sched.state.current
    assert [i for i, _ in ops.rotated] == [cur.index]     # rotated on the FIRST failure
    assert cur.king_leg_failures == 0
    # the same NaN on the fresh pod: ordinary same-pod budget, no second rotation yet
    _advance(clock, ops, sched, client, from_block=b0, to_block=b0 + 1)
    assert len(ops.rotated) == 1 and cur.king_leg_failures == 1
    ops.fail_king = False
    _advance(clock, ops, sched, client, from_block=b0 + 1, to_block=b0 + 2)
    assert cur.king_entry is not None
