"""Funded/king rents WAIT for marketplace capacity instead of failing the leg (2026-09-12).

Owner: "keep trying over the next 3 hours to bring up more pods" — legs must start
independently as GPUs appear (batches of one or more), not all-or-nothing. A leg whose
rent finds no capacity polls the marketplace until the round's latest safe start
(epoch end − final leg length − publish margin); only then does it requeue as before.
"""
from __future__ import annotations

import pytest

from cascade.provision import funded as funded_mod
from cascade.trainer.loop import TrainerRunner, _FundedLegSkip
from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault


def _arm_wait(runner, *, deadline_offsets, capacity_seq):
    """Fake clock + capacity: ``deadline_offsets`` = seconds the deadline sits
    ahead of a monotonically advancing fake clock; ``capacity_seq`` = successive
    _probe_funded_capacity answers."""
    clock = {"t": 1000.0}
    runner._rent_wait_now = lambda: clock["t"]
    runner._rent_wait_sleep = lambda s: clock.__setitem__("t", clock["t"] + max(s, 1.0))
    runner._funded_rent_wait_deadline = lambda: 1000.0 + deadline_offsets
    seq = list(capacity_seq)
    runner._probe_funded_capacity = lambda sku: seq.pop(0) if seq else 0
    runner._wait_for_funded_capacity = TrainerRunner._wait_for_funded_capacity.__get__(runner)
    runner.FUNDED_RENT_RETRY_SECONDS = TrainerRunner.FUNDED_RENT_RETRY_SECONDS
    return clock


def test_wait_returns_true_when_capacity_appears_before_the_deadline(tmp_path):
    runner = _runner(tmp_path)
    clock = _arm_wait(runner, deadline_offsets=3600, capacity_seq=[0, 0, 2])
    assert runner._wait_for_funded_capacity("RTX4090", describe="x") is True
    assert clock["t"] >= 1000.0 + 2 * TrainerRunner.FUNDED_RENT_RETRY_SECONDS   # two polls slept


def test_wait_gives_up_at_the_deadline(tmp_path):
    runner = _runner(tmp_path)
    clock = _arm_wait(runner, deadline_offsets=200, capacity_seq=[0, 0, 0, 0, 0])
    assert runner._wait_for_funded_capacity("RTX4090", describe="x") is False
    assert clock["t"] >= 1200.0                                   # ran out the clock, no further


def test_wait_is_instant_when_the_deadline_has_passed(tmp_path):
    runner = _runner(tmp_path)
    _arm_wait(runner, deadline_offsets=-1, capacity_seq=[5])
    assert runner._wait_for_funded_capacity("RTX4090", describe="x") is False   # never polled


def test_funded_rent_retries_after_no_capacity_and_lands(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    attempts = []

    def fake_rent(**kw):
        attempts.append(kw["hotkey"])
        if len(attempts) < 3:
            return funded_mod.FundedRentResult(hotkey="hkA", ok=False, error="sold out",
                                               error_class="no_capacity", burn_attempt=False)
        return _rent_ok("hkA")

    monkeypatch.setattr(funded_mod, "rent_funded_pod", fake_rent)
    _arm_wait(runner, deadline_offsets=3600, capacity_seq=[0, 1, 1])
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert len(attempts) == 3                                    # two sold-out rents, then a pod
    assert host.host and pod.payer_hotkey == "hkA"
    assert "hkA" not in runner._funded_leg_failures                # nothing recorded as a fault


def test_funded_rent_requeues_only_after_the_deadline(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    attempts = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: (
        attempts.append(1) or funded_mod.FundedRentResult(
            hotkey="hkA", ok=False, error="sold out", error_class="no_capacity",
            burn_attempt=False)))
    _arm_wait(runner, deadline_offsets=200, capacity_seq=[0, 0, 0, 0])
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)   # requeue, never a burn
    assert len(attempts) == 1                                   # waited, no capacity, gave up


def test_other_rent_failures_do_not_wait(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: funded_mod.FundedRentResult(
        hotkey="hkA", ok=False, error="bad key", error_class="auth", burn_attempt=False))
    polled = []
    runner._wait_for_funded_capacity = lambda *a, **k: polled.append(1) or True
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert polled == []                                          # auth is a verdict, not a wait


def test_deadline_math_from_epoch_geometry(cfg, tmp_path):
    runner = TrainerRunner(cfg=cfg, base_trainer=None, work_root=tmp_path)
    leg = max(c.max_train_seconds for c in cfg.throne_contracts())
    epoch_blocks = int(__import__("cascade.shared.config", fromlist=["effective_epoch_blocks"])
                       .effective_epoch_blocks(cfg.round, 9050400))
    runner._stage_ctx = {"epoch_start_block": 9050400}
    runner._funded_gate_block = 9050400 + epoch_blocks // 2      # halfway through the epoch
    import time
    before = time.time()
    dl = runner._funded_rent_wait_deadline()
    remaining = (epoch_blocks - epoch_blocks // 2) * 12.0
    expected = before + remaining - leg - TrainerRunner.FUNDED_PUBLISH_MARGIN_SECONDS
    assert abs(dl - expected) < 5.0
    # Unknown context ⇒ no waiting (offline tools, tests).
    runner._stage_ctx = {}
    assert runner._funded_rent_wait_deadline() <= time.time() + 1.0
