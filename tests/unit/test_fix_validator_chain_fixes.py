"""Validator-chain fixes: promotion catch-up (V1), crash-safe Cascade /
activation state (V2), the manifest hash link (V6), per-entry ref faults +
deregistered reveal history (V7), and the pruned pool-snapshot index (V8)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from cascade.shared import activation as A
from cascade.shared import hippius
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
)
from cascade.shared.chain import Commitment
from cascade.shared.era import era_length_blocks, min_effective_era, settlement_era
from cascade.shared.hippius import StorageError
from cascade.shared.manifest import dump_manifest, load_manifest
from cascade.shared.promotion import (
    PromotedMember,
    PromotionRecord,
    dump_promotion_record,
    promotion_index_key,
    promotion_record_key,
)
from cascade.validator.cascade import (
    CascadeController,
    CascadeState,
    CascadeStateUnreadable,
    dumps,
    load_state,
)
from tests.unit.test_chain_poll import _STORE, _client, _Sub
from tests.unit.test_era_validator import (
    _DEFAULT_HISTORY,
    PTR_C,
    REF_C,
    REF_C2,
    REF_K,
    _armed,
    _manifest,
    _rollover,
    _runner,
    _scores,
    _scores6,
    _Store,
)
from tests.unit.test_pool_store import _FakeS3Store, _make_pool_tar

G1 = "metro-v1:trained:cascade/ckpt@sha256:" + "7" * 64
G2 = "metro-v1:trained:cascade/ckpt@sha256:" + "8" * 64
G3 = "metro-v1:trained:cascade/ckpt@sha256:" + "9" * 64


# ── V1: promotion catch-up ───────────────────────────────────────────────────

def _ledger(armed, n_block, gens=(1, 2, 3), *, latest=3):
    """Records for ``gens``, effective at eras N-2, N-1, N (N = n_block's
    era), each fired the earliest it may, benched in its own round."""
    era_n = settlement_era(armed.round, n_block)
    length = era_length_blocks(armed.round, era_n.start_block)
    ptrs = {1: G1, 2: G2, 3: G3}
    texts = {promotion_index_key(): json.dumps({"latest_generation": latest})}
    for g in gens:
        eff = era_n.index - 3 + g
        fired = (eff - 2) * length
        assert min_effective_era(armed.round, fired) <= eff
        rnd = f"r{g}"
        texts[promotion_record_key(g)] = dump_promotion_record(PromotionRecord(
            generation=g, king_hotkey="king_hk", fired_round=rnd, fired_block=fired,
            effective_era=eff, members=(PromotedMember(ptrs[g], "toto2-4m", rnd, 0.4),)))
        texts[bench_report_key(rnd)] = dump_bench_report(BenchReport(
            round_id=rnd, created_block=fired, entries=(BenchEntry(
                role="king", size="toto2-4m", miner_hotkey="hk", miner_uid=0,
                trained_pointer=ptrs[g], scores=_scores6(0.4)),)))
    return era_n, texts


def _fresh_cascade(armed, tmp_path):
    return CascadeController(
        reign_days=armed.scoring.cascade_reign_days, round_cfg=armed.round,
        state=CascadeState(king_hotkey="king_hk", reign_start_block=0),
        state_path=tmp_path / "cascade_state.json")


def _n_block(armed):
    eb = armed.round.epoch_blocks
    return _rollover(armed) + eb * 4 * 6 + eb


def test_joiner_at_generation_zero_catches_up_through_every_record(cfg, tmp_path):
    armed = _armed(cfg)
    n_block = _n_block(armed)
    era_n, texts = _ledger(armed, n_block)
    cas = _fresh_cascade(armed, tmp_path)
    v = _runner(armed, cascade=cas, store=_Store(texts))
    m = _manifest(armed, n_block, warm_start=G3, generation=3, member_index=0)
    assert v.check_manifest(m) is None
    # gens 1 and 2 (superseded, effective already) installed by the walk; 3 staged
    assert cas.state.generation == 2 and cas.state.pending_generation == 3
    assert not cas.state.clock_observed          # a caught-up clock never attests
    assert v.process_settlement(m, windows=[], base_seed=n_block) is not None
    assert cas.state.generation == 3 and cas.state.members == (G3,)
    assert load_state(tmp_path / "cascade_state.json").generation == 3


def test_catch_up_keeps_the_one_step_lag_for_an_earlier_era(cfg, tmp_path):
    armed = _armed(cfg)
    n_block = _n_block(armed)
    eb = armed.round.epoch_blocks
    _, texts = _ledger(armed, n_block)
    cas = _fresh_cascade(armed, tmp_path)
    v = _runner(armed, cascade=cas, store=_Store(texts))
    prev_block = n_block - eb * 4                 # era N-1: generation 2 in force
    assert v.check_manifest(_manifest(armed, prev_block, warm_start=G2, generation=2,
                                      member_index=0)) is None
    # gen 2 is effective and superseded by a known record: installed; 3 staged
    assert cas.state.generation == 2 and cas.state.pending_generation == 3
    # gen 3 (era N) is NOT in force yet
    assert v.check_manifest(_manifest(armed, prev_block, warm_start=G3, generation=3,
                                      member_index=0)).startswith("era_init_mismatch")


def test_missing_middle_record_stops_the_walk(cfg, tmp_path):
    armed = _armed(cfg)
    n_block = _n_block(armed)
    _, texts = _ledger(armed, n_block, gens=(1, 3))
    cas = _fresh_cascade(armed, tmp_path)
    v = _runner(armed, cascade=cas, store=_Store(texts))
    m3 = _manifest(armed, n_block, warm_start=G3, generation=3, member_index=0)
    assert v.check_manifest(m3).startswith("era_init_mismatch")
    assert cas.state.generation == 1 and cas.state.pending_generation == 0
    # the ledger names generation 1 (no skip to 3)
    assert v.check_manifest(_manifest(armed, n_block, warm_start=G1, generation=1,
                                      member_index=0)) is None


def test_unreadable_record_mid_walk_restores_the_start_state(cfg, tmp_path):
    armed = _armed(cfg)
    n_block = _n_block(armed)
    _, texts = _ledger(armed, n_block)

    class _Down(_Store):
        def get_text(self, key):
            if key == bench_report_key("r2"):
                raise StorageError("503")
            return super().get_text(key)

    cas = _fresh_cascade(armed, tmp_path)
    v = _runner(armed, cascade=cas, store=_Down(texts))
    with pytest.raises(StorageError):
        v.check_manifest(_manifest(armed, n_block, warm_start=G3, generation=3,
                                   member_index=0))
    assert cas.state == CascadeState(king_hotkey="king_hk", reign_start_block=0)
    assert load_state(tmp_path / "cascade_state.json") == cas.state


# ── V2: crash-safe state files ───────────────────────────────────────────────

def test_cascade_state_falls_back_to_backup_and_refuses_garbage(tmp_path):
    p = tmp_path / "cascade_state.json"
    assert load_state(p) == CascadeState()                       # absent ⇒ fresh
    ctl = CascadeController(reign_days=7, state_path=p)
    ctl.note_dethrone("kingA", block=100)
    ctl.note_promotion(generation=2, members=("a",), block=200)
    assert (tmp_path / "cascade_state.json.bak").is_file()
    p.write_text('{"king_hotkey": "kin', encoding="utf-8")       # torn main
    assert load_state(p).king_hotkey == "kingA"                  # the previous good copy
    (tmp_path / "cascade_state.json.bak").write_text("garbage", encoding="utf-8")
    with pytest.raises(CascadeStateUnreadable):
        load_state(p)
    p.write_text(dumps(ctl.state), encoding="utf-8")
    assert load_state(p).generation == 2


def test_activation_store_falls_back_to_backup_and_refuses_garbage(tmp_path):
    p = tmp_path / "activation_state.json"
    store = A.ActivationStore(p)
    assert store.load() == A.ActivationRecord()
    one = A.ActivationRecord(feature="f", lock_block=10, activation_block=20, source="tally")
    two = replace(one, last_checked_boundary=30)
    store.save(one)
    store.save(two)
    assert store.load() == two
    p.write_text("{not json", encoding="utf-8")
    assert store.load() == one                                   # .bak holds the previous record
    (tmp_path / "activation_state.json.bak").write_text("[]", encoding="utf-8")
    with pytest.raises(A.ActivationStateUnreadable):
        store.load()


# ── V6: the manifest hash link ───────────────────────────────────────────────

def test_old_manifest_bytes_are_unchanged_and_the_link_is_signed(cfg):
    armed = _armed(cfg)
    block = _rollover(armed) + armed.round.epoch_blocks
    old = _manifest(armed, block, prev_round_id="123")
    raw = dump_manifest(old)
    assert "prev_manifest_sha256" not in raw
    again = load_manifest(raw)
    assert again.prev_manifest_sha256 == ""
    assert dump_manifest(again) == raw
    assert again.canonical_body() == old.canonical_body()
    linked = replace(old, prev_manifest_sha256="ab" * 32)
    assert b'"prev_manifest_sha256":"' + b"ab" * 32 + b'"' in linked.canonical_body()
    assert load_manifest(dump_manifest(linked)) == linked


def test_hash_link_must_match_the_previous_handled_manifest(cfg):
    armed = _armed(cfg)
    eb = armed.round.epoch_blocks
    b1 = _rollover(armed) + eb
    b2 = b1 + eb
    raw1 = dump_manifest(_manifest(armed, b1))
    sha1 = hashlib.sha256(raw1.encode("utf-8")).hexdigest()
    r = _runner(armed)
    r.state = replace(r.state, last_handled_round_id=str(b1), last_handled_manifest_sha=sha1)
    assert r.check_manifest(_manifest(armed, b2, prev_round_id=str(b1))) is None  # unset ⇒ unchecked
    good = replace(_manifest(armed, b2, prev_round_id=str(b1)), prev_manifest_sha256=sha1)
    assert r.check_manifest(good) is None
    forked = replace(good, prev_manifest_sha256="0" * 64)
    assert r.check_manifest(forked).startswith("manifest_chain_broken")
    # no handled digest yet (fresh joiner): nothing to compare against
    r.state = replace(r.state, last_handled_manifest_sha=None)
    assert r.check_manifest(forked) is None


# ── V7: deregistered history + per-entry ref faults ──────────────────────────

def test_poll_can_keep_reveals_of_deregistered_hotkeys():
    foreign = "5ForeignHotkeyNotOnMetagraph"
    assert foreign in _STORE
    default = _client(_Sub()).poll_commitments(include_history=True)
    assert foreign not in {c.hotkey for c in default}
    kept = _client(_Sub()).poll_commitments(include_history=True, include_deregistered=True)
    gone = [c for c in kept if c.hotkey == foreign]
    assert len(gone) == 1 and gone[0].uid is None and gone[0].coldkey is None
    assert {c.uid for c in kept if c.hotkey != foreign} == {c.uid for c in default}


def _faults_cfg(armed, gate):
    return replace(armed, scoring=replace(armed.scoring, entry_fault_from_block=gate))


def test_bad_challenger_ref_excludes_only_that_entry_from_the_gate_block(cfg):
    armed = _armed(cfg)
    eb = armed.round.epoch_blocks
    block = _rollover(armed) + eb * 2
    bad = _manifest(armed, block, challengers=(("chal_hk", 1, REF_C2, PTR_C),
                                               ("x_hk", 2, REF_K, PTR_C)))
    # before the gate (and with it inert): the whole settlement is rejected
    for gate in (0, block + eb):
        r = _runner(_faults_cfg(armed, gate))
        assert r.check_manifest(bad).startswith("era_ref_mismatch")
        assert r.era_entry_faults(bad) == {}
        assert r.judged_manifest(bad) is bad
    r = _runner(_faults_cfg(armed, block))
    assert r.check_manifest(bad) is None
    faults = r.era_entry_faults(bad)
    assert list(faults) == ["chal_hk"] and faults["chal_hk"].startswith("era_ref_mismatch")
    judged = r.judged_manifest(bad)
    assert [e.miner_hotkey for e in judged.entries] == ["king_hk", "x_hk"]
    # a different manifest under the same round id is not covered by the cache
    assert r.era_entry_faults(replace(bad, created_block=bad.created_block + 1)) == {}
    # a bad KING ref still rejects the settlement
    bad_king = _manifest(armed, block, king=("king_hk", 0, REF_C, PTR_C))
    assert r.check_manifest(bad_king).startswith("era_ref_mismatch")


def test_deregistered_history_counts_only_from_the_gate_block(cfg):
    armed = _armed(cfg)
    eb = armed.round.epoch_blocks
    block = _rollover(armed) + eb * 2
    history = [c for c in _DEFAULT_HISTORY if c.hotkey != "chal_hk"] + [
        Commitment(uid=None, hotkey="chal_hk", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{REF_C}", commit_block=0)]
    m = _manifest(armed, block)
    pre = _runner(_faults_cfg(armed, 0), history=lambda: list(history))
    assert pre.check_manifest(m).startswith("era_ref_unbound")    # bit-identical to before
    post = _runner(_faults_cfg(armed, block), history=lambda: list(history))
    assert post.check_manifest(m) is None and post.era_entry_faults(m) == {}


def test_excluded_challenger_is_not_judged(cfg):
    armed = _faults_cfg(_armed(cfg), 1)
    block = _rollover(armed) + armed.round.epoch_blocks
    king = _scores(1.0, 0)
    m = _manifest(armed, block, challengers=(("chal_hk", 1, REF_C2, PTR_C),))
    r = _runner(armed, king_scores=king)
    assert r.check_manifest(m) is None
    assert r.judged_manifest(m).challenger_cohort() == []
    assert m.challenger_cohort() != [] and REF_C != REF_C2


# ── V8: a pruned pool index never pins the oldest survivor ───────────────────

def _publish(store, tmp_path, blk, **kw):
    hippius.publish_pool_snapshot(
        store, _make_pool_tar(tmp_path, f"p{blk}", phase=blk), effective_block=blk,
        as_of="d", n_series=5, context_length=128, horizon=16, **kw)


def test_pruned_index_raises_instead_of_pinning_the_oldest(tmp_path):
    store = _FakeS3Store()
    for i in range(1, 4):
        _publish(store, tmp_path, 7200 * i)
    index = hippius.read_pool_index(store)
    assert index.pruned_through == 0
    assert "pruned_through" not in json.loads(store.objects[hippius.POOL_INDEX_KEY])
    assert hippius.select_snapshot(index, 100).effective_block == 7200   # never trimmed: legacy floor
    for i in range(4, 7):
        _publish(store, tmp_path, 7200 * i, max_keep=3)
    index = hippius.read_pool_index(store)
    assert [m.effective_block for m in index] == [28800, 36000, 43200]
    assert index.pruned_through == 21600
    assert hippius.select_snapshot(index, 30000).effective_block == 28800
    with pytest.raises(hippius.SnapshotPruned):
        hippius.select_snapshot(index, 21600)
    assert issubclass(hippius.SnapshotPruned, StorageError)
    # a plain list (callers building their own) keeps the legacy floor
    assert hippius.select_snapshot(list(index), 21600).effective_block == 28800
    with pytest.raises(hippius.SnapshotPruned):
        hippius.select_snapshot(list(index), 21600, pruned_through=21600)
    # the marker survives later publishes that trim nothing
    _publish(store, tmp_path, 7200 * 7, max_keep=60)
    assert hippius.read_pool_index(store).pruned_through == 21600


def test_publish_default_keeps_sixty():
    import inspect

    assert inspect.signature(hippius.publish_pool_snapshot).parameters["max_keep"].default == 60
