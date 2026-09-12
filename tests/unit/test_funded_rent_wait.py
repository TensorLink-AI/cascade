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


def _king_runner(tmp_path, monkeypatch, *, ready_seq, deadline_offsets):
    """A runner whose king rent hits a fake Lium: ``ready_seq`` = successive
    wait_ready answers (a False = lemon pod → torn down, retried elsewhere)."""
    import threading

    import cascade.provision.core as core_mod
    import cascade.provision.funded as pf
    from cascade.provision.core import PodAddress

    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    torn = []
    monkeypatch.setattr(pf, "terminate_verified", lambda prov, pod_id: torn.append(pod_id) or True)
    r = _runner(tmp_path, funded_king_rent=True)
    r._funded_round_sku = "RTX4090"
    r._funded_wait_abort = threading.Event()
    _arm_wait(r, deadline_offsets=deadline_offsets, capacity_seq=[1] * 10)
    launched = []
    seq = list(ready_seq)

    class _Prov:
        name = "lium"
        def capacity(self, sku, *, gpus=1):
            return 1
        def launch(self, spec):
            launched.append(spec.exclude_ids)
            return [f"{spec.name_prefix}-0"]
        def wait_ready(self, pod_id, *, timeout):
            return seq.pop(0) if seq else True
        def get_ip(self, pod_id):
            return PodAddress(ip="9.9.9.9", ssh_port=41000)
        def machine_of(self, pod_id):
            return f"exec-{len(launched)}"
        def terminate(self, pod_id):
            torn.append(pod_id)

    monkeypatch.setattr(core_mod, "LiumProvider", _Prov)
    return r, launched, torn


def test_king_rent_retries_on_a_lemon_pod_and_excludes_its_executor(tmp_path, monkeypatch):
    r, launched, torn = _king_runner(tmp_path, monkeypatch, ready_seq=[False, True],
                                     deadline_offsets=3600)
    host = r._rent_king_host("42")
    assert (host.host, host.port) == ("9.9.9.9", 41000)
    assert len(launched) == 2                                   # lemon, then a second rent
    assert torn == ["cascade-n91-42-funded-king-0"]             # the lemon was torn down
    assert "exec-1" in launched[1]                              # …and its executor excluded
    assert not r._funded_wait_abort.is_set()                    # challengers keep waiting


def test_king_rent_gives_up_at_the_deadline_and_releases_the_waiters(tmp_path, monkeypatch):
    from cascade.provision.core import ProvisionError

    r, launched, torn = _king_runner(tmp_path, monkeypatch, ready_seq=[False, False],
                                     deadline_offsets=-1)            # already past
    with pytest.raises(ProvisionError, match="gave up at the latest safe start"):
        r._rent_king_host("42")
    assert len(launched) == 1 and len(torn) == 1
    assert r._funded_wait_abort.is_set()
    # A challenger still polling sees the abort and stops immediately.
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0, 0])
    assert r._wait_for_funded_capacity("RTX4090", describe="funded leg x") is False


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
    # The epoch-end estimate is fixed for the attempt: a later block stamp
    # must not move the deadline.
    runner._funded_gate_block = 9050400 + epoch_blocks - 10
    assert abs(runner._funded_rent_wait_deadline() - dl) < 1.0
    # Unknown context (fresh attempt, no stage ctx) ⇒ no waiting.
    runner._funded_epoch_end_wall = None
    runner._stage_ctx = {}
    assert runner._funded_rent_wait_deadline() <= time.time() + 1.0
