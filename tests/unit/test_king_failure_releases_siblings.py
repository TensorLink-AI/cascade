"""A failed king leg releases its sibling legs at once (2026-09-15).

The final pool re-raised a king failure only after joining EVERY challenger
thread. With funded legs polling the marketplace until the round's latest
safe start, a king that died at 21:09 would have surfaced at 05:49 — the
round lost, the king pod idle all night (third occurrence; 2026-09-13 was a
2 h silent wait). Now the king's failure sets the funded wait abort (capacity
polls and king-yields return immediately, those legs requeue unburned),
cancels unstarted legs, logs the in-flight ones it still waits for, and the
RuntimeError surfaces as soon as they finish. A training leg is never killed.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.manifest import TrainedEntry
from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from cascade.trainer.remote import RemoteHost
from tests.unit.test_trainer_round import _FakeBaseTrainer

REF_K = "king/gen-k@sha256:" + "a" * 64
REF_A = "alice/gen-a@sha256:" + "b" * 64
REF_B = "bob/gen-b@sha256:" + "c" * 64
PTR = "metro-v1:trained:hippius:cascade/ckpt-x@sha256:" + "e" * 64


def _entry(role, hotkey, ref, size):
    return TrainedEntry(miner_hotkey=hotkey, miner_uid=1, role=role, gen_ref=ref,
                        trained_pointer=PTR, corpus_digest="cd", train_block=10,
                        gpu_name="NVIDIA L40S", size=size)


def _runner(cfg, tmp_path):
    rnd = replace(cfg.round, funded_mode="off")
    r = TrainerRunner(cfg=replace(cfg, round=rnd), base_trainer=_FakeBaseTrainer(),
                      work_root=tmp_path, trainer_spec="m:C")
    r.remote_hosts = [RemoteHost(name=f"lane-{i}", host=f"10.0.0.{i}", stage="final")
                      for i in range(1, 4)]
    r._funded_field = {}
    r._funded_leg_failures = {}
    r._funded_roster = {"seated": [], "waiting": [], "terminal": [], "outcomes": []}
    r._funded_king_lock = threading.Lock()
    r._funded_king_host = None
    r._final_role_hosts = {}
    r._funded_wait_abort = threading.Event()        # what run_round arms per attempt
    return r


def test_king_failure_sets_the_abort_and_surfaces_without_waiting_for_pollers(
        cfg, tmp_path, monkeypatch):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="kkkkkkkkkkkkkkkk", uid=0, ref=REF_K)
    a = ResolvedGenerator(hotkey="aaaaaaaaaaaaaaaa", uid=1, ref=REF_A)
    b = ResolvedGenerator(hotkey="bbbbbbbbbbbbbbbb", uid=2, ref=REF_B)
    seen_abort = {}

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            if kw["role"] == "king":
                time.sleep(0.05)                    # the siblings are already waiting
                raise RuntimeError("Host key verification failed")
            # A challenger "polling for capacity": it can only leave on the abort.
            seen_abort[kw["hotkey"]] = r._funded_wait_abort.wait(timeout=10.0)
            raise RuntimeError("capacity wait cancelled")

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    errors = []
    monkeypatch.setattr(loop_mod.log, "error",
                        lambda msg, *args, **kw: errors.append(msg % args if args else msg))

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="king training failed on remote"):
        r._train_remote([(king, "king"), (a, "challenger"), (b, "challenger")],
                        SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)
    elapsed = time.monotonic() - t0

    assert r._funded_wait_abort.is_set()
    assert seen_abort == {a.hotkey: True, b.hotkey: True}   # released by the abort, not the timeout
    assert elapsed < 5.0
    assert any("king leg FAILED" in e and "Host key verification failed" in e for e in errors)


def test_in_flight_legs_are_named_and_finish_before_the_round_aborts(cfg, tmp_path, monkeypatch):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="kkkkkkkkkkkkkkkk", uid=0, ref=REF_K)
    a = ResolvedGenerator(hotkey="aaaaaaaaaaaaaaaa", uid=1, ref=REF_A)
    done = []

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            if kw["role"] == "king":
                time.sleep(0.05)
                raise RuntimeError("boom")
            time.sleep(0.4)                         # a leg mid-training: never killed
            done.append(kw["hotkey"])
            return _entry("challenger", kw["hotkey"], kw["gen_ref"], contract.arch_preset)

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    errors = []
    monkeypatch.setattr(loop_mod.log, "error",
                        lambda msg, *args, **kw: errors.append(msg % args if args else msg))

    with pytest.raises(RuntimeError, match="king training failed on remote: boom"):
        r._train_remote([(king, "king"), (a, "challenger")],
                        SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)

    assert done == [a.hotkey]                       # the pool still joined the training leg
    assert any("1 in-flight leg(s) [aaaaaaaaaaaa]" in e for e in errors)
    # …and its result is persisted for the round's retry to reuse.
    assert r._load_completed_leg(round_id=1, contract=contract, role="challenger",
                                 hotkey=a.hotkey, gen_ref=REF_A) is not None


def test_challenger_failure_never_touches_the_abort(cfg, tmp_path, monkeypatch):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="kkkkkkkkkkkkkkkk", uid=0, ref=REF_K)
    a = ResolvedGenerator(hotkey="aaaaaaaaaaaaaaaa", uid=1, ref=REF_A)

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            if kw["role"] == "challenger":
                raise RuntimeError("challenger boom")
            return _entry("king", kw["hotkey"], kw["gen_ref"], contract.arch_preset)

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    out = r._train_remote([(king, "king"), (a, "challenger")],
                          SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)
    assert [e.role for e in out] == ["king"]
    assert not r._funded_wait_abort.is_set()
