"""Trainer settlement publish is write-ahead (T2) and hash-chained (V6).

A crash between the manifest publish and the state commit used to re-step
the boundary on restart and republish a DIFFERENT manifest (more legs
finished since) under the same round id. The exact signed text is now
persisted before the first byte goes out and republished verbatim; the
committed boundary records the sha256 of those bytes for the next
manifest's ``prev_manifest_sha256``."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.manifest import TrainedEntry, dump_manifest, load_manifest
from cascade.trainer import rolling as R
from cascade.trainer.rolling import LegOps, RollingScheduler

from .test_rolling import (
    EB,
    REF,
    Clock,
    FakeClient,
    FakeOps,
    _advance,
    _armed,
    _commit,
    _join,
    _ptr,
)


class _CrashOps(FakeOps):
    """publish_manifest uploads (recorded) and then 'crashes' while armed."""

    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.crash = 0
        self.texts: list[str] = []
        self.reports: list[tuple] = []

    def publish_manifest(self, manifest):
        self.texts.append(dump_manifest(manifest))
        super().publish_manifest(manifest)
        if self.crash:
            self.crash -= 1
            raise RuntimeError("trainer died after the upload")

    def publish_bench(self, round_id, created_block, entries, **kw):
        self.reports.append((round_id, [(e.trained_pointer, dict(vars(s))) for e, s in entries],
                             kw.get("rows_for")))
        return super().publish_bench(round_id, created_block, entries, **kw)


def _scheduler(cfg, tmp_path, clock, ops):
    runner = SimpleNamespace(cfg=cfg, work_root=tmp_path, promotion=None,
                             pool_provenance_fn=None,
                             _rolling_note_king_host=lambda era: None)
    return RollingScheduler(runner, ops, clock=clock)


def _setup(cfg, tmp_path, *, gate="off"):
    armed = _armed(cfg)
    if gate != "off":
        armed = replace(armed, scoring=replace(armed.scoring, bench_gate_mode=gate))
    clock = Clock()
    ops = _CrashOps(tmp_path, clock)
    sched = _scheduler(armed, tmp_path, clock, ops)
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100), _commit("BRAV", REF["BRAV"], b0 - 50)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    return armed, clock, ops, sched, era_start, b0


def test_crash_after_publish_republishes_the_same_bytes(cfg, tmp_path, monkeypatch):
    # the hash link is rollout-gated ([manifest] hash_link_from_block); armed here
    monkeypatch.setattr(R.RollingScheduler, "_hash_link_active", lambda self, b: True)
    armed, clock, ops, sched, era_start, b0 = _setup(cfg, tmp_path)
    client = FakeClient()
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    ops.crash = 1
    clock.t += (b1 + 3 - b0) * R.BLOCK_SECONDS
    with pytest.raises(RuntimeError, match="died"):
        sched.tick(client, b1 + 3)
    _join(sched)
    first = ops.texts[-1]
    assert sched.state.pending_publish is not None
    assert sched.state.last_settled_boundary < b1          # not committed
    # BRAV finishes before the restart: the old re-step would have added it
    q = ops.queue()
    q.add("BRAV", REF["BRAV"], reveal_block=b0 - 50)
    assert q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=b1 + EB,
                            era_index=sched.state.current.index, started_block=b1 + 1)
    entry = TrainedEntry("BRAV", 2, "challenger", REF["BRAV"], _ptr("BRAV-late"), "d", b1 + 1,
                         gpu_name="RTX 4090")
    sched.state.finished.append(R.FinishedLeg(
        hotkey="BRAV", uid=2, ref=REF["BRAV"], era_index=sched.state.current.index,
        started_block=b1 + 1, reveal_block=b0 - 50, entry=R._entry_to_json(entry)))
    sched._save()
    again = _scheduler(armed, tmp_path, clock, ops)       # a restart reads the state file
    assert again.state.pending_publish["manifest"] == first
    again.tick(client, b1 + 4)
    _join(again)
    assert ops.texts[-1] == first                         # byte-identical republish
    m = load_manifest(first)
    assert [e.miner_hotkey for e in m.entries] == ["KING", "ALFA"]
    st = again.state
    assert st.pending_publish is None and st.last_settled_boundary == b1
    assert st.last_published_round_id == m.round_id
    assert st.last_published_manifest_sha256 == hashlib.sha256(first.encode()).hexdigest()
    assert ops.queue().get("ALFA").status == "done"
    assert any(f.hotkey == "BRAV" for f in st.finished)    # waits for the next boundary
    # the next settlement chains to the exact bytes of the first
    b2 = b1 + EB
    _advance(clock, ops, again, client, from_block=b1 + 4, to_block=b2 + 3)
    m2 = load_manifest(ops.texts[-1])
    assert [e.miner_hotkey for e in m2.entries] == ["KING", "BRAV"]
    assert m2.prev_round_id == m.round_id
    assert m2.prev_manifest_sha256 == hashlib.sha256(first.encode()).hexdigest()
    assert b"prev_manifest_sha256" in m2.canonical_body()


def test_first_manifest_after_a_seeded_root_omits_the_hash_link(cfg, tmp_path):
    armed, clock, ops, sched, era_start, b0 = _setup(cfg, tmp_path)
    client = FakeClient()
    sched.tick(client, b0)
    _join(sched)
    _advance(clock, ops, sched, client, from_block=b0, to_block=era_start + EB + 3)
    m = load_manifest(ops.texts[-1])
    assert m.prev_round_id == "LEGACY" and m.prev_manifest_sha256 == ""
    assert b"prev_manifest_sha256" not in m.canonical_body()


def test_bench_first_report_is_republished_with_the_same_pairs(cfg, tmp_path):
    armed, clock, ops, sched, era_start, b0 = _setup(cfg, tmp_path, gate="shadow")
    client = FakeClient()
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    ops.crash = 1
    clock.t += (b1 + 3 - b0) * R.BLOCK_SECONDS
    with pytest.raises(RuntimeError):
        sched.tick(client, b1 + 3)
    _join(sched)
    rid, pairs, rows_for = ops.reports[-1]
    assert rows_for is None and pairs                       # the first publish
    pend = sched.state.pending_publish
    assert pend["bench_first"] is True
    sched.tick(client, b1 + 4)
    _join(sched)
    rid2, pairs2, rows_for2 = ops.reports[-1]
    assert (rid2, pairs2) == (rid, pairs)                   # the same report content
    assert rows_for2 == frozenset(p for p, _ in pairs)      # no rows added on a replay
    assert ops.texts[-1] == ops.texts[-2]
    assert sched.state.pending_publish is None


class _Store:
    def __init__(self):
        self.puts = []

    def put_text(self, key, text, **kw):
        self.puts.append((key, text))


def test_legops_signs_once_and_publishes_byte_exactly(monkeypatch):
    """sr25519 signatures are randomised: a re-sign would publish new bytes."""
    from cascade.shared.manifest import TrainingManifest

    store = _Store()
    hotkey = SimpleNamespace(sign=lambda body: os.urandom(64))
    runner = SimpleNamespace(wallet=SimpleNamespace(hotkey=hotkey),
                             manifest_store=lambda: store,
                             _publish_training_summary=lambda m: None,
                             publish=lambda m: pytest.fail("re-signing publish path used"))
    ops = LegOps(runner)
    m = TrainingManifest(round_id="7", created_block=2, contract_digest="1" * 64,
                         base_arch_digest="2" * 64, eval_dataset="d", entries=[])
    signed = ops.sign_manifest(m)
    assert signed.signature
    assert ops.sign_manifest(signed) is signed             # never re-signed
    ops.publish_manifest(signed)
    ops.publish_manifest(load_manifest(dump_manifest(signed)))
    texts = {t for _, t in store.puts}
    assert texts == {dump_manifest(signed)}
