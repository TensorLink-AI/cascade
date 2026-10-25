"""T1: executor claims are released when their pod is torn down and age out as
a backstop, instead of living for the whole process."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from cascade.provision.state import PodInstance
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.runner_common import (
    EXEC_CLAIM_MAX_AGE_SECONDS,
    claim_executor,
    live_claimed_executors,
    release_executor_of,
)


def _fake(**kw):
    f = SimpleNamespace(_funded_claimed_execs=set(), _funded_claim_meta={},
                        _funded_exec_lock=threading.Lock(), **kw)
    return f


def _pod(iid="cascade-n91-7-funded-hka-0", payer="hkA"):
    return PodInstance(provider="lium", instance_id=iid, stage="funded",
                       rented_at_iso="", payer_hotkey=payer)


def test_claim_release_and_age_out():
    r = _fake()
    claim_executor(r, "m1", "pod-1", now=1000.0)
    claim_executor(r, "m2", "pod-2", now=1000.0)
    claim_executor(r, "m3", now=1000.0)               # excluded executor, no pod
    assert live_claimed_executors(r, now=1001.0) == ("m1", "m2", "m3")
    assert release_executor_of(r, "pod-1") == ["m1"]
    assert release_executor_of(r, "") == []             # never matches instance-less
    assert live_claimed_executors(r, now=1001.0) == ("m2", "m3")
    # backstop: everything older than the max age is dropped
    assert live_claimed_executors(r, now=1000.0 + EXEC_CLAIM_MAX_AGE_SECONDS + 1) == ()
    assert r._funded_claimed_execs == set() and r._funded_claim_meta == {}


def test_claims_without_meta_are_kept_and_still_listed():
    """A set assigned directly (tests, older state) is never aged out blindly."""
    r = SimpleNamespace(_funded_claimed_execs={"b", "a"}, _funded_exec_lock=threading.Lock())
    assert live_claimed_executors(r, now=10**12) == ("a", "b")


def test_funded_teardown_releases_the_pods_executor(monkeypatch):
    r = _fake(_payer_vault=lambda: object())
    r._revoke_robot = lambda pod: None
    removed = []
    r._ledger_remove = removed.append
    monkeypatch.setattr("cascade.provision.funded.teardown_funded", lambda pods, vault: [])
    claim_executor(r, "m1", "cascade-n91-7-funded-hka-0")
    claim_executor(r, "m9", "other-pod")
    TrainerRunner._teardown_funded_pod(r, _pod())
    assert removed == ["cascade-n91-7-funded-hka-0"]
    assert r._funded_claimed_execs == {"m9"}


def test_funded_teardown_not_confirmed_keeps_the_claim(monkeypatch):
    r = _fake(_payer_vault=lambda: object())
    r._revoke_robot = lambda pod: None
    r._ledger_remove = lambda iid: None
    pod = _pod()
    monkeypatch.setattr("cascade.provision.funded.teardown_funded", lambda pods, vault: [pod])
    claim_executor(r, "m1", pod.instance_id)
    TrainerRunner._teardown_funded_pod(r, pod)
    assert r._funded_claimed_execs == {"m1"}           # still live ⇒ still claimed


def test_operator_teardown_releases_the_pods_executor(monkeypatch):
    r = _fake()
    r._ledger_remove = lambda iid: None
    monkeypatch.setattr("cascade.provision.core.LiumProvider", lambda: object())
    monkeypatch.setattr("cascade.provision.funded.terminate_verified", lambda prov, iid: True)
    claim_executor(r, "mk", "cascade-n91-king-0")
    TrainerRunner._teardown_operator_pod(r, _pod("cascade-n91-king-0", payer=""))
    assert r._funded_claimed_execs == set()
