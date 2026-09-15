"""King-first rent (2026-09-15, round 9072000): one RTX4090 appeared after 13
polls, the king and a waiting challenger both went for it, the challenger's
`lium up` landed first and the REQUIRED king polled on. While the round's JIT
king still needs a pod, challenger rents yield the marketplace: they neither
take the rent lock nor treat capacity as theirs. The yield ends the moment the
king's rent reaches any outcome (pod, operator fallback, give-up, cached
retry), at the round's latest safe start, or on the king's abort."""
from __future__ import annotations

import threading

from cascade.provision import funded as funded_mod
from cascade.trainer.loop import TrainerRunner
from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault
from tests.unit.test_funded_rent_wait import _arm_wait


def test_king_pending_only_while_the_jit_king_is_armed_and_unrented(tmp_path):
    r = _runner(tmp_path, funded_king_rent=True)
    assert r._king_pending() is True
    r._funded_king_host = object()                  # the king has its pod
    assert r._king_pending() is False
    r._funded_king_host = None
    r._king_rent_done = True                        # any outcome of the king's rent
    assert r._king_pending() is False
    assert _runner(tmp_path).__class__ and _runner(tmp_path)._king_pending() is False   # JIT king off
    gated = _runner(tmp_path, funded_king_rent=True, funded_activation_block=10**9)
    assert gated._king_pending() is False           # funded gate closed ⇒ the king never rents


def test_challenger_capacity_wait_yields_to_the_king_then_proceeds(tmp_path):
    r = _runner(tmp_path, funded_king_rent=True)
    clock = _arm_wait(r, deadline_offsets=3600, capacity_seq=[3, 3, 3, 3])
    sleeps = []
    base_sleep = r._rent_wait_sleep

    def sleep(s):
        sleeps.append(s)
        base_sleep(s)
        if len(sleeps) == 2:
            r._king_rent_done = True                # the king's rent lands mid-wait

    r._rent_wait_sleep = sleep
    assert r._wait_for_funded_capacity("RTX4090", describe="funded leg x") is True
    assert len(sleeps) == 2 and all(s == TrainerRunner.KING_YIELD_POLL_SECONDS for s in sleeps)
    assert clock["t"] > 1000.0


def test_the_kings_own_wait_is_never_yielded(tmp_path):
    r = _runner(tmp_path, funded_king_rent=True)
    clock = _arm_wait(r, deadline_offsets=3600, capacity_seq=[1])
    assert r._wait_for_funded_capacity("RTX4090", describe="king rent", for_king=True) is True
    assert clock["t"] == 1000.0                     # no poll slept


def test_challenger_rent_holds_until_the_kings_outcome(tmp_path, monkeypatch):
    r = _runner(tmp_path, funded_king_rent=True)
    _vault(tmp_path, "hkA")
    order = []
    clock = {"t": 1000.0}
    r._rent_wait_now = lambda: clock["t"]
    r._funded_rent_wait_deadline = lambda: 1000.0 + 3600

    def sleep(s):
        clock["t"] += s
        order.append("yield")
        r._king_rent_done = True                    # the king got its pod

    r._rent_wait_sleep = sleep
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: order.append("rent") or _rent_ok())
    host, pod = r._rent_funded_host("777", _challenger("hkA"))
    assert order == ["yield", "rent"]               # no rent before the king was served
    assert host.host == "10.9.9.9"


def test_yield_ends_at_the_deadline_and_on_the_kings_abort(tmp_path):
    r = _runner(tmp_path, funded_king_rent=True)
    clock = {"t": 1000.0}
    r._rent_wait_now = lambda: clock["t"]
    slept = []
    r._rent_wait_sleep = lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s))
    # Deadline passed ⇒ no hold at all (the leg's own wait requeues it).
    r._funded_rent_wait_deadline = lambda: 999.0
    r._yield_to_king("funded leg x")
    assert slept == []
    # King abort ⇒ released at once.
    r._funded_rent_wait_deadline = lambda: 1000.0 + 3600
    r._funded_wait_abort = threading.Event()
    r._funded_wait_abort.set()
    r._yield_to_king("funded leg x")
    assert slept == []
    # Otherwise the hold runs to the deadline (bounded: never a hang).
    r._funded_wait_abort.clear()
    r._funded_rent_wait_deadline = lambda: 1000.0 + 12
    r._yield_to_king("funded leg x")
    assert slept and clock["t"] >= 1012.0 and r._king_pending()


def test_unarmed_king_never_holds_a_challenger(tmp_path, monkeypatch):
    r = _runner(tmp_path)                           # funded_king_rent off
    _vault(tmp_path, "hkA")
    r._rent_wait_sleep = lambda s: (_ for _ in ()).throw(AssertionError("held"))
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    host, _ = r._rent_funded_host("777", _challenger("hkA"))
    assert host.host == "10.9.9.9"
