"""Bench-anneal (DEC-CA-0030): the post-round bench sweep scores an annealed
copy of each duel checkpoint when ``[telemetry] bench_anneal_fraction`` is
armed — finished-form BenchScores with no contract, manifest, or wire change.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.manifest import TrainedEntry, format_trained_pointer
from cascade.trainer import worker as worker_mod
from cascade.trainer.loop import BENCH_ANNEAL_SALT, TrainerRunner
from cascade.trainer.remote import RemoteHost, worker_argv

REF_G = "alice/gen-a@sha256:" + "a" * 64
PTR = format_trained_pointer("cascade/ckpt-r1-king-toto2-4m@sha256:" + "c" * 64)


def _entry(role="king", uid=7):
    return TrainedEntry(
        miner_hotkey="hkA", miner_uid=uid, role=role, gen_ref=REF_G,
        trained_pointer=PTR, corpus_digest="d", train_block=100,
    )


# ── the recipe override ───────────────────────────────────────────────────────


def test_anneal_recipe_swaps_only_the_decay_fields(cfg):
    c = cfg.training.primary_size
    a = worker_mod.anneal_recipe(c)
    assert a.lr_schedule == "warmup_cosine"
    assert a.warmup_fraction == 0.0
    # everything else — budget, arch, objective — is untouched
    assert replace(a, lr_schedule=c.lr_schedule,
                   warmup_fraction=c.warmup_fraction) == c


def test_worker_argv_carries_anneal_flag():
    host = RemoteHost(name="h", host="1.2.3.4", remote_python="/venv/python")
    base = dict(gen_ref=REF_G, uid=7, hotkey="hkA", role="king",
                base_seed=99, block=12, trainer_spec="m:C")
    assert "--anneal" not in worker_argv(host, **base)
    argv = worker_argv(host, **base, train_hours=0.45,
                       warm_start_ref=PTR, anneal=True)
    assert "--anneal" in argv


def test_worker_rejects_anneal_without_resume_or_budget(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "load_chain_config", lambda p: cfg)
    monkeypatch.setattr(worker_mod, "_load_trainer", lambda spec: object())
    base = ["--gen-ref", REF_G, "--uid", "7", "--hotkey", "hkA", "--role", "king",
            "--base-seed", "1", "--block", "10", "--trainer", "m:C",
            "--work-root", str(tmp_path), "--anneal"]
    # no --warm-start-ref, no --train-hours ⇒ refuse before any work
    assert worker_mod.main(base) == 2
    assert worker_mod.main(base + ["--train-hours", "0.45"]) == 2
    assert worker_mod.main(base + ["--warm-start-ref", PTR]) == 2


def test_worker_anneal_applies_pure_decay_contract(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "load_chain_config", lambda p: cfg)
    monkeypatch.setattr(worker_mod, "_load_trainer", lambda spec: object())
    captured: dict = {}

    def _fake_train_one(self, gen, role, seeds, block, *, contract=None,
                        token_budget=None, repo_suffix="", heat=False,
                        warm_start_ref=None):
        captured["contract"] = contract
        captured["warm_start_ref"] = warm_start_ref
        return _entry(role=role)

    monkeypatch.setattr(TrainerRunner, "train_one", _fake_train_one)
    argv = ["--gen-ref", REF_G, "--uid", "7", "--hotkey", "hkA", "--role", "king",
            "--base-seed", "1", "--block", "10", "--trainer", "m:C",
            "--work-root", str(tmp_path), "--anneal",
            "--train-hours", "0.45", "--warm-start-ref", PTR]
    assert worker_mod.main(argv) == 0
    c = captured["contract"]
    assert c.lr_schedule == "warmup_cosine" and c.warmup_fraction == 0.0
    assert c.target_train_hours == 0.45          # for_hours applied first
    assert captured["warm_start_ref"] == PTR


# ── the loop's bench redirection ──────────────────────────────────────────────


def _runner(cfg, monkeypatch, frac):
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, bench_anneal_fraction=frac))
    r = TrainerRunner.__new__(TrainerRunner)
    r.cfg = cfg
    r.trainer_spec = "m:C"
    r.remote_timeout_seconds = 60
    r.work_root = None
    r.cascade_bench_plan = SimpleNamespace()
    r._pod_extra_forward_env = lambda: ()
    return r


def test_bench_scores_bench_annealed_dir_when_armed(cfg, monkeypatch):
    r = _runner(cfg, monkeypatch, 0.15)
    host = RemoteHost(name="h", host="1.2.3.4")
    entry = _entry(uid=7)
    round_id = "10328013751254538515"
    dispatched: dict = {}

    def _fake_dispatch(self, h, **kw):
        dispatched.update(kw)
        return _entry()

    benched: dict = {}

    def _fake_bench(h, rid, size, plan, *, work_root=None, role="king", runner=None):
        benched.update(rid=rid, role=role)
        return None  # report contents are not under test

    import cascade.trainer.loop as loop_mod
    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher.dispatch",
                        _fake_dispatch)
    monkeypatch.setattr("cascade.trainer.bench_hook.run_post_round_benchmark",
                        _fake_bench)
    r._remote_bench_scores(host, entry, round_id, "toto2-4m")

    salted = (int(round_id) ^ BENCH_ANNEAL_SALT) & ((1 << 63) - 1)
    # the anneal leg: salted seed, fresh corpus, resume the canonical pointer
    assert dispatched["base_seed"] == salted
    assert dispatched["warm_start_ref"] == PTR
    assert dispatched["anneal"] is True
    assert dispatched["repo_suffix"] == "-anneal-u7"
    frac_hours = r.cfg.training.primary_size.target_train_hours * 0.15
    assert dispatched["train_hours"] == pytest.approx(frac_hours)
    # the sweep then benches the annealed copy's dir, not the raw checkpoint's
    assert benched["rid"] == str(salted)
    assert benched["role"] == "king-anneal-u7"


def test_bench_falls_back_to_raw_checkpoint_on_anneal_failure(cfg, monkeypatch):
    r = _runner(cfg, monkeypatch, 0.15)
    host = RemoteHost(name="h", host="1.2.3.4")
    entry = _entry(uid=7)

    def _boom(self, h, **kw):
        raise RuntimeError("pod fell over")

    benched: dict = {}

    def _fake_bench(h, rid, size, plan, *, work_root=None, role="king", runner=None):
        benched.update(rid=rid, role=role)
        return None

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher.dispatch", _boom)
    monkeypatch.setattr("cascade.trainer.bench_hook.run_post_round_benchmark",
                        _fake_bench)
    r._remote_bench_scores(host, entry, "42", "toto2-4m")
    assert benched["rid"] == "42" and benched["role"] == "king"  # raw fallback


def test_bench_unarmed_is_byte_identical_to_before(cfg, monkeypatch):
    r = _runner(cfg, monkeypatch, 0.0)
    host = RemoteHost(name="h", host="1.2.3.4")
    entry = _entry()

    def _never(self, h, **kw):
        raise AssertionError("no anneal leg may dispatch when unarmed")

    benched: dict = {}

    def _fake_bench(h, rid, size, plan, *, work_root=None, role="king", runner=None):
        benched.update(rid=rid, role=role)
        return None

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher.dispatch", _never)
    monkeypatch.setattr("cascade.trainer.bench_hook.run_post_round_benchmark",
                        _fake_bench)
    r._remote_bench_scores(host, entry, "42", "toto2-4m")
    assert benched["rid"] == "42" and benched["role"] == "king"
