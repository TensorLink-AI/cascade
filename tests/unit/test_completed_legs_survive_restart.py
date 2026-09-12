"""Finished remote legs survive a trainer restart (2026-09-12).

A remote leg's TrainedEntry lived only in memory until the manifest was built, so a
mid-round restart (deploy, crash) re-trained every finished leg — and re-billed its
payer. Each finished leg is now persisted the moment its worker returns and a retry
of the same job reuses it instead of dispatching.
"""
from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

from cascade.shared.manifest import TrainedEntry
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from cascade.trainer.remote import RemoteHost
from tests.unit.test_trainer_round import _FakeBaseTrainer

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_C = "carol/gen-c@sha256:" + "c" * 64
PTR = "metro-v1:trained:hippius:cascade/ckpt-x@sha256:" + "e" * 64


def _entry(role, hotkey, ref, size):
    return TrainedEntry(miner_hotkey=hotkey, miner_uid=1, role=role, gen_ref=ref,
                        trained_pointer=PTR, corpus_digest="cd", train_block=10,
                        gpu_name="NVIDIA L40S", size=size)


def _runner(cfg, tmp_path):
    rnd = replace(cfg.round, funded_mode="off")
    r = TrainerRunner(cfg=replace(cfg, round=rnd), base_trainer=_FakeBaseTrainer(),
                      work_root=tmp_path, trainer_spec="m:C")
    r.remote_hosts = [RemoteHost(name="lane-a", host="10.0.0.1", stage="final"),
                      RemoteHost(name="lane-b", host="10.0.0.2", stage="final")]
    r._funded_field = {}
    r._funded_leg_failures = {}
    r._funded_roster = {"seated": [], "waiting": [], "terminal": [], "outcomes": []}
    r._funded_king_lock = threading.Lock()
    r._funded_king_host = None
    r._final_role_hosts = {}
    return r


def test_persist_then_load_round_trips_and_checks_identity(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    e = _entry("challenger", "c", REF_C, contract.arch_preset)
    r._persist_completed_leg(e, round_id=7, contract=contract, role="challenger", hotkey="c")
    got = r._load_completed_leg(round_id=7, contract=contract, role="challenger",
                                hotkey="c", gen_ref=REF_C)
    assert got == e
    # A different generator ref, role or contract is a different job.
    assert r._load_completed_leg(round_id=7, contract=contract, role="challenger",
                                 hotkey="c", gen_ref=REF_A) is None
    assert r._load_completed_leg(round_id=7, contract=contract, role="king",
                                 hotkey="c", gen_ref=REF_C) is None
    other = replace(contract, max_train_seconds=contract.max_train_seconds + 1)
    assert r._load_completed_leg(round_id=7, contract=other, role="challenger",
                                 hotkey="c", gen_ref=REF_C) is None
    assert r._load_completed_leg(round_id=8, contract=contract, role="challenger",
                                 hotkey="c", gen_ref=REF_C) is None


def test_train_remote_reuses_persisted_legs_and_only_dispatches_the_rest(cfg, tmp_path,
                                                                          monkeypatch):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="k", uid=0, ref=REF_A)
    chal = ResolvedGenerator(hotkey="c", uid=1, ref=REF_C)
    # The king finished in a prior run of this round; the challenger did not.
    prior_king = _entry("king", "k", REF_A, contract.arch_preset)
    r._persist_completed_leg(prior_king, round_id=1, contract=contract, role="king", hotkey="k")
    seen = []

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            seen.append(kw["role"])
            return _entry(kw["role"], kw["hotkey"], kw["gen_ref"], contract.arch_preset)

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    out = r._train_remote([(king, "king"), (chal, "challenger")], SimpleNamespace(base_seed=1),
                          10, contract, contract.train_tokens)
    assert seen == ["challenger"]                                 # king never re-dispatched
    assert {e.role for e in out} == {"king", "challenger"}
    assert next(e for e in out if e.role == "king") == prior_king
    # …and the challenger's fresh result is now persisted for the next restart.
    assert r._load_completed_leg(round_id=1, contract=contract, role="challenger",
                                 hotkey="c", gen_ref=REF_C) is not None


def test_unreadable_record_falls_back_to_training(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    p = r._completed_leg_path(1, contract.arch_preset, "king", "k")
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert r._load_completed_leg(round_id=1, contract=contract, role="king", hotkey="k",
                                 gen_ref=REF_A) is None
