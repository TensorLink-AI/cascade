"""Corpus pre-generation ([generation] pregen_*): every funded, revealed
entry's corpus is built on the CPU tier as soon as it is queued, decoupled
from GPU admission — opportunistic, capped, budgeted, and never in a real
leg's way."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.generation.stage import (
    PREGEN_OWNER,
    GenerationFault,
    GenerationInfraError,
    GenerationYield,
)
from cascade.shared.config import load_chain_config
from cascade.shared.config.generation import parse_generation
from cascade.shared.config.loader import DEFAULT_CHAIN_TOML
from cascade.trainer import rolling as R
from cascade.trainer.admission import FairSlots

from .test_generation_stage import HK, FakeProvider, FakeStore, _go, _stage
from .test_rolling import EB, REF, Clock, FakeClient, FakeOps, _armed, _commit, _sched

# ── config ──────────────────────────────────────────────────────────────────


def test_pregen_ships_dark():
    g = load_chain_config().generation
    assert (g.pregen_enabled, g.pregen_max_parallel, g.pregen_max_usd_per_epoch) == (False, 4, 50.0)
    assert parse_generation(None).pregen_enabled is False


def test_pregen_knobs_round_trip_through_the_toml_loader(tmp_path):
    src = DEFAULT_CHAIN_TOML.read_text()
    assert src.count("\n[generation]\n") == 1
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("\n[generation]\n", "\n[generation]\npregen_enabled = true\n"
                             "pregen_max_parallel = 3\npregen_max_usd_per_epoch = 12.5\n"))
    g = load_chain_config(p).generation
    assert (g.pregen_enabled, g.pregen_max_parallel, g.pregen_max_usd_per_epoch) == (True, 3, 12.5)
    p.write_text(src.replace("\n[generation]\n", "\n[generation]\npregen_max_usd_per_epoch = 0\n"))
    assert load_chain_config(p).generation.pregen_max_usd_per_epoch == 0.0


@pytest.mark.parametrize("raw, match", [
    ({"pregen_enabled": "yes"}, "pregen_enabled"),
    ({"pregen_max_parallel": 0}, "pregen_max_parallel"),
    ({"pregen_max_parallel": 2.5}, "pregen_max_parallel"),
    ({"pregen_max_parallel": True}, "pregen_max_parallel"),
    ({"pregen_max_usd_per_epoch": -1}, "pregen_max_usd_per_epoch"),
    ({"pregen_max_usd_per_epoch": float("nan")}, "pregen_max_usd_per_epoch"),
    ({"pregen_max_usd_per_epoch": "50"}, "pregen_max_usd_per_epoch"),
])
def test_pregen_knobs_are_validated(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_generation(raw)


# ── the CPU gate: real legs never wait behind pregen ───────────────────────


def _bg(gate, **kw):
    return gate.slot(1, background=True, owner=PREGEN_OWNER, owner_limit=4, **kw)


def test_background_slots_never_count_against_a_real_leg():
    gate = FairSlots()
    with _bg(gate) as held:
        assert held and gate.active == 1
        with gate.slot(1, deadline=time.time() + 0.5) as leg:
            assert leg                                   # not behind the pregen job
        with _bg(gate, cancel=lambda: True) as second:
            assert not second                            # pregen only fills idle slots


def test_a_waiting_real_leg_takes_the_free_slot_before_pregen():
    gate = FairSlots()
    order, ready = [], threading.Event()

    def pregen():
        with _bg(gate) as ok:
            order.append(("pregen", ok))

    def leg():
        ready.set()
        with gate.slot(1) as ok:
            order.append(("leg", ok))

    with gate.slot(1):                                   # a real leg holds the cap
        p = threading.Thread(target=pregen, daemon=True)
        p.start()
        with gate.condition:
            assert gate.condition.wait_for(lambda: len(gate.waiters) == 1, timeout=3)
        t = threading.Thread(target=leg, daemon=True)
        t.start()
        ready.wait(3)
        with gate.condition:
            assert gate.condition.wait_for(lambda: len(gate.waiters) == 2, timeout=3)
    t.join(5)
    p.join(5)
    assert order == [("leg", True), ("pregen", True)]


def test_pregen_caps_at_pregen_max_parallel():
    gate = FairSlots()
    with (gate.slot(5, background=True, owner=PREGEN_OWNER, owner_limit=1) as a,
          gate.slot(5, background=True, owner=PREGEN_OWNER, owner_limit=1,
                    deadline=time.time() + 0.3) as b):
        assert a and not b


def test_a_real_leg_generates_while_pregen_holds_every_slot(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, idx, _ = _stage(tmp_path, prov, store)
    st.cfg = replace(st.cfg, generation=replace(st.cfg.generation, max_parallel_jobs=1))
    st.capacity_wait_s = 0.3
    gate = st.__dict__.setdefault("_slots", FairSlots(clock=st.clock))
    with _bg(gate) as held:
        assert held
        res = _go(st, tmp_path)                          # would time out if it queued
    assert res.index["corpus_digest"] == idx["corpus_digest"]


def test_a_cancelled_background_build_yields_before_renting(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, _, _ = _stage(tmp_path, prov, store)
    st.cfg = replace(st.cfg, generation=replace(st.cfg.generation, max_parallel_jobs=1))
    gate = st.__dict__.setdefault("_slots", FairSlots(clock=st.clock))
    events = []
    st.observer = events.append
    (tmp_path / "repo").mkdir(exist_ok=True)
    with gate.slot(1), pytest.raises(GenerationYield):
        st.generate(round_id="77", hotkey=HK, uid=4, era=9, gen_ref="g@sha256:1",
                    repo_dir=tmp_path / "repo", seed=123, background=True, cancel=lambda: True)
    assert prov.launched == []
    assert "infra_fault" not in [e["stage"] for e in events]


# ── the trainer mixin: one corpus per key, the leg wins a waiting pregen ───


def _mixin_runner(tmp_path, build):
    from cascade.trainer.runner_generation import _GenerationMixin

    stage = SimpleNamespace(pod_files=lambda result: {"corpus.json": result})
    me = SimpleNamespace(cfg=load_chain_config(), work_root=tmp_path,
                         _generation_stage=lambda: stage, _build_leg_corpus=build)
    return me, _GenerationMixin


def test_the_leg_reuses_a_pregen_corpus_and_a_waiting_pregen_yields(tmp_path):
    builds = []

    def build(gen, seeds, era, **kw):
        builds.append(kw)
        if kw.get("background") and kw["cancel"]():
            raise GenerationYield("yield")
        return f"corpus-{len(builds)}"

    me, M = _mixin_runner(tmp_path, build)
    gen = SimpleNamespace(hotkey=HK, ref="r@1", uid=1)
    seeds, contract = SimpleNamespace(base_seed=9, generation_seed=1), SimpleNamespace(corpus_mode="shards")
    assert M._pregen_leg_corpus(me, gen, seeds) == "built"
    assert builds[0]["background"] is True
    assert M._leg_corpus(me, gen, seeds, contract) == {"corpus.json": "corpus-1"}
    assert len(builds) == 1                              # the leg took the pregen corpus
    assert M._pregen_leg_corpus(me, gen, seeds) == "cached"
    # a leg waiting for the key makes a still-queued pregen of it step aside
    other = SimpleNamespace(hotkey=HK, ref="r@2", uid=1)
    key = (HK, "9", "r@2", me.cfg.generator.min_channels, me.cfg.generator.max_channels)
    M._want_leg_corpus(me, key, +1)
    with pytest.raises(GenerationYield):
        M._pregen_leg_corpus(me, other, seeds)
    M._want_leg_corpus(me, key, -1)
    assert me._leg_corpus_wanted == {}


def test_pregen_skips_a_key_a_leg_is_building(tmp_path):
    me, M = _mixin_runner(tmp_path, lambda *a, **k: pytest.fail("must not build"))
    gen = SimpleNamespace(hotkey=HK, ref="r@1", uid=1)
    seeds = SimpleNamespace(base_seed=9, generation_seed=1)
    key = (HK, "9", "r@1", me.cfg.generator.min_channels, me.cfg.generator.max_channels)
    lock = me.__dict__.setdefault("_leg_corpus_locks", {}).setdefault(key, threading.Lock())
    with lock:
        assert M._pregen_leg_corpus(me, gen, seeds) == "busy"


# ── the scheduler pass ──────────────────────────────────────────────────────


class PregenOps(FakeOps):
    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.ready: set = set()               # (hotkey, base_seed) already stored
        self.built: list = []                 # (hotkey, base_seed, ref) per build
        self.gate: threading.Event | None = None
        self.exc: Exception | None = None
        self.cancels: list = []

    def pregen_ready(self, gen, era):
        return (gen.hotkey, era.base_seed) in self.ready

    def pregen_build(self, gen, era, *, cancel):
        self.built.append((gen.hotkey, era.base_seed, gen.ref))
        self.cancels.append(cancel)
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if cancel():
                    raise GenerationYield("seed epoch moved on")
        if self.exc is not None:
            raise self.exc
        return "built"


def _pregen_cfg(cfg, **kw):
    armed = _armed(cfg)
    g = replace(armed.generation, enabled=True, pregen_enabled=True, **kw)
    return replace(armed, generation=g)


def _setup(cfg, tmp_path, **kw):
    armed = _pregen_cfg(cfg, **kw)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock, ops=PregenOps(tmp_path, clock))
    client = FakeClient()
    block = armed.round.rolling_from_block + 5
    sched.tick(client, block)                 # the era exists; nothing queued yet
    return sched, ops, client, block


def _queue(ops, hk, block, *, reveal=True, ref=None):
    ref = ref or REF[hk]
    ops.queue().add(hk, ref, reveal_block=block)
    if reveal:
        ops.commits.append(_commit(hk, ref, block))


def _pass(sched, client, block):
    sched._intake_history = list(sched.ops.commitments(client))
    sched._pregen_tick(client, block, sched.clock())


def _drain(sched, timeout=5.0):
    end = time.time() + timeout
    while sched._pregen_threads and time.time() < end:
        time.sleep(0.01)
    assert not sched._pregen_threads


def test_pregen_starts_for_a_revealed_queued_entry_without_a_corpus(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    _queue(ops, "ALFA", block)
    _pass(sched, client, block)
    _drain(sched)
    seed = sched.state.current.base_seed
    assert ops.built == [("ALFA", seed, REF["ALFA"])]
    assert ops.queue().get("ALFA").status == "queued"   # no GPU admission happened
    _pass(sched, client, block)                          # built: never again
    _drain(sched)
    assert len(ops.built) == 1
    spend = json.loads((Path(tmp_path) / sched.PREGEN_LEDGER).read_text())
    assert spend == {str(seed): pytest.approx(sched._pregen_cost_usd())}


def test_pregen_skips_ready_unrevealed_in_flight_and_disabled(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    seed = sched.state.current.base_seed
    _queue(ops, "ALFA", block)
    ops.ready.add(("ALFA", seed))                        # corpus already stored
    _queue(ops, "BRAV", block, reveal=False)             # funded, not revealed
    _queue(ops, "CHAR", block)
    sched._threads["CHAR"] = threading.current_thread()  # a live leg builds its own
    _pass(sched, client, block)
    _drain(sched)
    assert ops.built == []
    sched._threads.clear()
    sched.cfg = replace(sched.cfg, generation=replace(sched.cfg.generation, pregen_enabled=False))
    _pass(sched, client, block)
    assert ops.built == []


def test_pregen_respects_reveal_order_and_max_parallel(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path, pregen_max_parallel=2)
    ops.gate = threading.Event()
    _queue(ops, "CHAR", block + 1)
    _queue(ops, "BRAV", block + 2)
    _queue(ops, "ALFA", block + 3)
    _pass(sched, client, block + 3)
    time.sleep(0.05)
    assert [b[0] for b in ops.built] == ["CHAR", "BRAV"]
    _pass(sched, client, block + 3)                      # cap still full; no repeats
    assert len(ops.built) == 2
    ops.gate.set()
    _drain(sched)
    _pass(sched, client, block + 3)
    _drain(sched)
    assert [b[0] for b in ops.built] == ["CHAR", "BRAV", "ALFA"]


def test_pregen_stops_at_the_epoch_budget(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    one = sched._pregen_cost_usd()
    sched.cfg = replace(sched.cfg, generation=replace(sched.cfg.generation,
                                                       pregen_max_usd_per_epoch=one * 1.5))
    _queue(ops, "ALFA", block + 1)
    _queue(ops, "BRAV", block + 2)
    _pass(sched, client, block + 2)
    _drain(sched)
    _pass(sched, client, block + 2)
    _drain(sched)
    assert [b[0] for b in ops.built] == ["ALFA"]
    assert sched.state.current.base_seed in sched._pregen_broke


def test_a_pregen_failure_never_fails_or_burns_the_entry(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    ops.exc = GenerationInfraError("no CPU capacity")
    _queue(ops, "ALFA", block)
    _pass(sched, client, block)
    _drain(sched)
    e = ops.queue().get("ALFA")
    assert e.status == "queued" and e.operator_faults == 0 and not e.operator_billed
    assert ops.burnt == []
    _pass(sched, client, block)                          # backing off
    _drain(sched)
    assert len(ops.built) == 1
    ops.exc = None
    sched.clock.t += sched.PREGEN_RETRY_SECONDS + 1
    _pass(sched, client, block)
    _drain(sched)
    assert len(ops.built) == 2
    # a generator fault is the leg's to judge: logged, entry untouched, not retried
    ops.exc = GenerationFault("boom")
    _queue(ops, "BRAV", block + 1)
    _pass(sched, client, block + 1)
    _drain(sched)
    sched.clock.t += sched.PREGEN_RETRY_MAX_SECONDS + 1
    _pass(sched, client, block + 1)
    _drain(sched)
    assert [b[0] for b in ops.built].count("BRAV") == 1
    assert ops.queue().get("BRAV").status == "queued" and ops.burnt == []


def test_pregen_uses_the_next_eras_seeds_when_the_entry_admits_there(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    era_start = sched.cfg.round.rolling_from_block
    late = era_start + EB * 4 - 1100                     # inside the last wall + margin
    sched.clock.t += (late - block) * R.BLOCK_SECONDS
    cur = sched.state.current
    _queue(ops, "ALFA", late)
    _pass(sched, client, late)                           # next era unknown: waits
    assert ops.built == []
    sched.state.next = replace(cur, index=cur.index + 1, start_block=era_start + EB * 4,
                               base_seed=cur.base_seed + 12345)
    _pass(sched, client, late)
    _drain(sched)
    assert ops.built == [("ALFA", cur.base_seed + 12345, REF["ALFA"])]


def test_a_new_seed_epoch_stops_waiting_pregen_and_refunds_it(cfg, tmp_path):
    sched, ops, client, block = _setup(cfg, tmp_path)
    ops.gate = threading.Event()
    old = sched.state.current.base_seed
    _queue(ops, "ALFA", block)
    _pass(sched, client, block)
    time.sleep(0.05)
    assert ops.built and not ops.cancels[0]()
    sched.state.current = replace(sched.state.current, base_seed=old + 1)
    _pass(sched, client, block)       # old seed gone: it steps aside; the new seed's starts
    end = time.time() + 5
    while (("ALFA", old, REF["ALFA"]) in sched._pregen_threads) and time.time() < end:
        time.sleep(0.01)
    assert ("ALFA", old, REF["ALFA"]) not in sched._pregen_threads
    spend = json.loads((Path(tmp_path) / sched.PREGEN_LEDGER).read_text())
    assert spend[str(old)] == 0.0                        # nothing was rented
    ops.gate.set()
    _drain(sched)
    assert ops.built == [("ALFA", old, REF["ALFA"]), ("ALFA", old + 1, REF["ALFA"])]
