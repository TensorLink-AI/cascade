"""DEC-CA-0050: expert-leg dispatch argv and worker flags."""

from __future__ import annotations

import json

import pytest

from cascade.trainer.remote import RemoteHost, worker_argv
from cascade.trainer.worker import _build_parser, main

HOST = RemoteHost(name="pod", host="1.2.3.4", workdir="/root/cascade", cuda_device="0")
KW = dict(gen_ref="a/b@sha256:" + "a" * 64, uid=3, hotkey="hk", base_seed=1, block=2,
          trainer_spec="m:C")


def test_expert_argv_carries_domain_and_sorted_selection():
    argv = worker_argv(HOST, role="expert", warm_start_ref="king", domain="energy",
                       expert_selection={5: [9, 2], 1: [4]}, **KW)
    i = argv.index("--selection-json")
    assert json.loads(argv[i + 1]) == {"1": [4], "5": [2, 9]}
    assert argv[argv.index("--domain") + 1] == "energy"
    args = _build_parser().parse_args(argv[3:])
    assert args.role == "expert" and args.domain == "energy"


def test_expert_argv_requires_selection():
    with pytest.raises(ValueError):
        worker_argv(HOST, role="expert", **KW)


def test_challenger_argv_unchanged():
    argv = worker_argv(HOST, role="challenger", **KW)
    assert "--domain" not in argv and "--selection-json" not in argv


def test_worker_refuses_incomplete_expert_leg(monkeypatch, tmp_path):
    import cascade.trainer.worker as w

    monkeypatch.setattr(w, "load_chain_config", lambda p: __import__(
        "cascade.shared.config", fromlist=["x"]).load_chain_config(None))
    monkeypatch.setattr(w, "_load_trainer", lambda spec: object())
    rc = main(["--gen-ref", KW["gen_ref"], "--uid", "3", "--hotkey", "hk", "--role", "expert",
               "--base-seed", "1", "--block", "2", "--trainer", "m:C", "--domain", "energy",
               "--work-root", str(tmp_path)])
    assert rc == 2


# ── audit finding 3: RunnerLaneBOps against the REAL TrainerRunner ──────────


def _runner_attrs_used() -> set[str]:
    """Every ``self.r.<attr>`` RunnerLaneBOps touches (AST, not a fake)."""
    import ast
    import inspect

    from cascade.trainer import lane_b_driver

    tree = ast.parse(inspect.getsource(lane_b_driver.RunnerLaneBOps))
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "r" and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"):
            used.add(node.attr)
    return used


def test_runner_lane_b_ops_only_calls_real_runner_attributes():
    from cascade.trainer.loop import TrainerRunner

    used = _runner_attrs_used()
    assert "_dispatch_on_free_lane" in used and "_rolling_lane_pool" in used
    fields = set(TrainerRunner.__dataclass_fields__)
    missing = sorted(a for a in used
                     if a not in fields and a != "__dict__" and not hasattr(TrainerRunner, a))
    assert missing == [], f"RunnerLaneBOps calls TrainerRunner attributes that do not exist: {missing}"


def test_expert_leg_takes_a_free_lane_never_an_occupied_one(cfg, tmp_path, monkeypatch):
    """Two operator lanes; a challenger leg holds the first (checked out of
    the shared rolling lane pool). The expert leg must land on the second —
    through the same pool — as role ``expert`` with no ``train_hours``."""
    import threading
    from types import SimpleNamespace

    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import RunnerLaneBOps
    from cascade.trainer.loop import TrainerRunner

    lanes = [RemoteHost(name=f"lane{i}", host=f"10.0.0.{i}", workdir="/w", cuda_device=str(i))
             for i in range(2)]
    import dataclasses

    # the retired operator lane pool runs only under leg_funding = "operator"
    cfg = dataclasses.replace(cfg, lane_b=dataclasses.replace(cfg.lane_b,
                                                              leg_funding="operator"))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                      remote_hosts=lanes, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=7200, index=3)
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    calls = []

    class _Disp:
        def dispatch(self, host, **kw):
            calls.append((host.name, kw))
            return SimpleNamespace(trained_pointer="ptr", corpus_digest="cd",
                                   train_block=kw["block"], gpu_name="L40S")

    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: _Disp())
    pool = r._rolling_lane_pool()
    held = pool.get()                      # a challenger leg occupies a lane
    finished = threading.Event()
    got = []

    def done(leg, ptr, *a):
        got.append(ptr)
        finished.set()

    gen = KW["gen_ref"]
    leg = OpenLeg(sub=LaneBSubmission("alice", 7, gen, "energy", 5),
                  selection={0: [1, 2]}, base_king_ref="kingref", base_era=3)
    RunnerLaneBOps(r, client=None, window_source=None).dispatch_leg(leg, 3, 7300, done)
    assert finished.wait(5) and got == ["ptr"]
    (host, kw), = calls
    assert host != held.name               # never the occupied lane
    assert kw["role"] == "expert" and "train_hours" not in kw
    assert kw["domain"] == "energy" and kw["expert_selection"] == {0: [1, 2]}
    assert kw["warm_start_ref"] == "kingref" and kw["gen_ref"] == gen
    # the lane went back to the pool after the leg (the held one still out)
    assert pool.qsize() == 1


def test_expert_worker_leg_is_not_tagged_heat(monkeypatch, tmp_path):
    """``--role expert`` resolves its budget from [lane_b] leg_hours and
    trains with heat=False (telemetry/checkpoint keys of a full leg)."""
    import cascade.trainer.worker as w
    from cascade.shared.config import load_chain_config

    seen = {}

    class _Runner:
        def __init__(self, **kw):
            pass

        def train_one(self, gen, role, seeds, block, **kw):
            seen.update(role=role, **kw)
            from types import SimpleNamespace

            return SimpleNamespace(trained_pointer="p")

    monkeypatch.setattr(w, "load_chain_config", lambda p: load_chain_config(None))
    monkeypatch.setattr(w, "_load_trainer", lambda spec: object())
    monkeypatch.setattr(w, "TrainerRunner", _Runner)
    monkeypatch.setattr("cascade.trainer.contract.assert_train_image", lambda t: None)
    monkeypatch.setattr(w, "dump_entry", lambda e: "{}", raising=False)
    argv = ["--gen-ref", KW["gen_ref"], "--uid", "3", "--hotkey", "hk", "--role", "expert",
            "--base-seed", "1", "--block", "2", "--trainer", "m:C", "--domain", "energy",
            "--selection-json", '{"0":[1]}', "--warm-start-ref", "kingref",
            "--work-root", str(tmp_path)]
    import contextlib

    with contextlib.suppress(Exception):   # printing the stub entry's receipt is not under test
        main(argv)
    cfg = load_chain_config(None)
    assert seen["heat"] is False and seen["role"] == "challenger"
    assert seen["expert_selection"] == {0: [1]}
    want = cfg.training.primary_size.for_hours(
        cfg.lane_b.leg_hours, guard_factor=cfg.round.heat_guard_factor,
        guard_floor_seconds=cfg.round.heat_guard_floor_seconds).train_tokens
    assert seen["token_budget"] == want
