"""Finished legs survive a trainer restart: each is persisted the moment its
worker returns, and a retry of the same job (same contract, generator and init)
reuses it instead of dispatching — a restart never re-bills a payer."""
from __future__ import annotations

from dataclasses import replace

from cascade.shared.manifest import TrainedEntry
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost
from tests.unit.test_trainer_runner import _FakeBaseTrainer

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


def test_unreadable_record_falls_back_to_training(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    p = r._completed_leg_path(1, contract.arch_preset, "king", "k")
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert r._load_completed_leg(round_id=1, contract=contract, role="king", hotkey="k",
                                 gen_ref=REF_A) is None


def test_record_is_reused_only_for_the_same_init(cfg, tmp_path):
    # 2026-09-24: a legacy round's king leg (gen-11 member 2) was reused as the
    # era king whose challengers trained from member 0 — same seed, other init.
    r = _runner(cfg, tmp_path)
    contract = cfg.throne_contracts()[0]
    e = _entry("king", "k", REF_C, contract.arch_preset)
    r._persist_completed_leg(e, round_id=9, contract=contract, role="king", hotkey="k",
                             warm_start_ckpt="ckpt-member-2")

    def load(init):
        return r._load_completed_leg(round_id=9, contract=contract, role="king",
                                     hotkey="k", gen_ref=REF_C, warm_start_ckpt=init)

    assert load("ckpt-member-2") == e
    assert load("ckpt-member-0") is None
    assert load("") is None                                  # random init ≠ warm start
    assert r._load_completed_leg(round_id=9, contract=contract, role="king", hotkey="k",
                                 gen_ref=REF_C) == e         # no init given ⇒ no check
    # a record persisted before the field existed counts as random init
    r._persist_completed_leg(e, round_id=10, contract=contract, role="king", hotkey="k")
    path = r._completed_leg_path(10, contract.arch_preset, "king", "k")
    import json
    raw = json.loads(path.read_text())
    raw.pop("warm_start_ckpt")
    path.write_text(json.dumps(raw))
    assert r._load_completed_leg(round_id=10, contract=contract, role="king", hotkey="k",
                                 gen_ref=REF_C, warm_start_ckpt="ckpt-member-0") is None
    assert r._load_completed_leg(round_id=10, contract=contract, role="king", hotkey="k",
                                 gen_ref=REF_C, warm_start_ckpt="") == e
