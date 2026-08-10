"""Cascade's bench-score source order on the validator: the round's signed
bench report (post-publish artifact) → the manifest entry's in-entry
bench_scores (pre-report rounds) → absent (queued briefly for the late-landing
report, then written off — the round just contributes no bench numbers)."""

from __future__ import annotations

from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
)
from cascade.shared.manifest import BenchScores, TrainedEntry, TrainingManifest
from cascade.validator.cascade import CascadeController
from cascade.validator.loop import BENCH_REPORT_RETRY_ROUNDS, ValidatorRunner
from cascade.validator.state import genesis

REF = "alice/gen-a@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:cascade/ckpt-r777-king@sha256:" + "c" * 64

REPORT_SCORES = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)
MANIFEST_SCORES = BenchScores(
    gifteval_crps=0.99, gifteval_mase=0.99, boom_crps=0.99,
    boom_mase=0.99, time_crps=0.99, time_mase=0.99,
)


class _FakeStore:
    def __init__(self, texts=None):
        self.texts = dict(texts or {})

    def get_text(self, key):
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


def _manifest(cfg, *, round_id="777", with_manifest_scores=False):
    size = cfg.throne_contracts()[0].arch_preset
    return TrainingManifest(
        round_id=round_id, created_block=9000,
        contract_digest="c" * 64, base_arch_digest="a" * 64,
        eval_dataset="cascade-private-v1",
        entries=[TrainedEntry(
            miner_hotkey="king_hk", miner_uid=1, role="king", gen_ref=REF,
            trained_pointer=PTR, corpus_digest="d", train_block=9000, size=size,
            bench_scores=MANIFEST_SCORES if with_manifest_scores else None,
        )],
    )


def _report_text(cfg, pointer=PTR):
    size = cfg.throne_contracts()[0].arch_preset
    return dump_bench_report(BenchReport(
        round_id="777", created_block=9000,
        entries=(BenchEntry(role="king", size=size, miner_hotkey="king_hk",
                            miner_uid=1, trained_pointer=pointer, scores=REPORT_SCORES),),
    ))


def _runner(cfg, store):
    controller = CascadeController(reign_days=7)
    controller.note_dethrone("king_hk", block=9000)
    return ValidatorRunner(cfg=cfg, state=genesis("king_hk", 1), verify_signatures=False,
                           cascade=controller, bench_report_store=store)


def test_report_wins_over_manifest_scores(cfg):
    # Both sources present: the report is authoritative — even against a
    # manifest entry that still carries (different) in-entry numbers.
    store = _FakeStore({bench_report_key("777"): _report_text(cfg)})
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg, with_manifest_scores=True), now=1000.0)
    (rec,) = runner.cascade.state.checkpoints
    assert rec.checkpoint_id == PTR
    assert rec.gifteval_crps == REPORT_SCORES.gifteval_crps


def test_manifest_scores_are_the_fallback(cfg):
    # No report (older round, or the trainer's bench died): the manifest
    # entry's in-entry bench_scores still feed the reign log.
    runner = _runner(cfg, _FakeStore())
    runner._record_duel_checkpoints(_manifest(cfg, with_manifest_scores=True), now=1000.0)
    (rec,) = runner.cascade.state.checkpoints
    assert rec.gifteval_crps == MANIFEST_SCORES.gifteval_crps


def test_report_joins_on_pointer_not_role(cfg):
    # A report benching a DIFFERENT checkpoint must not be misattributed.
    other = "metro-v1:trained:hippius:cascade/ckpt-other@sha256:" + "0" * 64
    store = _FakeStore({bench_report_key("777"): _report_text(cfg, pointer=other)})
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg), now=1000.0)
    assert runner.cascade.state.checkpoints == ()            # nothing recorded…
    assert len(runner._pending_bench) == 1                   # …queued for the retry


def test_missing_both_queues_then_records_when_report_lands(cfg):
    store = _FakeStore()
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg), now=1000.0)
    assert runner.cascade.state.checkpoints == ()
    assert len(runner._pending_bench) == 1
    # The report lands between rounds (the bench runs post-publish); the next
    # round's cascade step drains the queue into the reign log.
    store.texts[bench_report_key("777")] = _report_text(cfg)
    runner._drain_pending_bench(now=2000.0)
    (rec,) = runner.cascade.state.checkpoints
    assert rec.checkpoint_id == PTR
    assert rec.gifteval_crps == REPORT_SCORES.gifteval_crps
    assert runner._pending_bench == []


def test_missing_report_expires_after_retry_budget(cfg):
    runner = _runner(cfg, _FakeStore())
    runner._record_duel_checkpoints(_manifest(cfg), now=1000.0)
    for _ in range(BENCH_REPORT_RETRY_ROUNDS):
        runner._drain_pending_bench(now=2000.0)
    # Written off: the round contributes no bench numbers, and Cascade simply
    # promotes over the reign rounds that have them.
    assert runner._pending_bench == []
    assert runner.cascade.state.checkpoints == ()


def test_pending_dropped_when_reign_ends(cfg):
    store = _FakeStore()
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg), now=1000.0)
    assert len(runner._pending_bench) == 1
    # Dethrone: the reign log was cleared at the re-crown, and the OLD reign's
    # checkpoint must never leak into the new reign's promotion pool — even if
    # its report shows up afterwards.
    runner.cascade.note_dethrone("new_king_hk", block=16200)
    store.texts[bench_report_key("777")] = _report_text(cfg)
    runner._drain_pending_bench(now=3000.0)
    assert runner._pending_bench == []
    assert runner.cascade.state.checkpoints == ()


def test_pending_dropped_when_promotion_recrowns_same_king(cfg):
    # An accepted promotion re-crowns the SAME king with a fresh clock and a
    # cleared log; a pre-promotion checkpoint whose report lands afterwards
    # must NOT drain into the new reign — it would corrupt the quality floor
    # the next promotion is verified against.
    store = _FakeStore()
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg), now=1000.0)
    assert len(runner._pending_bench) == 1
    runner.cascade.note_promotion(generation=1, members=(PTR,), block=16200)
    store.texts[bench_report_key("777")] = _report_text(cfg)
    runner._drain_pending_bench(now=3000.0)
    assert runner._pending_bench == []
    assert runner.cascade.state.checkpoints == ()


def test_unparseable_report_falls_back(cfg):
    store = _FakeStore({bench_report_key("777"): "{not json"})
    runner = _runner(cfg, store)
    runner._record_duel_checkpoints(_manifest(cfg, with_manifest_scores=True), now=1000.0)
    (rec,) = runner.cascade.state.checkpoints
    assert rec.gifteval_crps == MANIFEST_SCORES.gifteval_crps


def test_challenger_checkpoints_are_candidates_too(cfg):
    # DEC-CA-0013: both duel roles land in the reign log — the challenger's
    # checkpoint carries a different generator's data, the deepest diversity
    # the promoted set can draw on.
    size = cfg.throne_contracts()[0].arch_preset
    chal_ptr = "metro-v1:trained:hippius:cascade/ckpt-r777-chal@sha256:" + "e" * 64
    manifest = TrainingManifest(
        round_id="777", created_block=9000,
        contract_digest="c" * 64, base_arch_digest="a" * 64,
        eval_dataset="cascade-private-v1",
        entries=[
            TrainedEntry(miner_hotkey="king_hk", miner_uid=1, role="king",
                         gen_ref=REF, trained_pointer=PTR, corpus_digest="d",
                         train_block=9000, size=size),
            TrainedEntry(miner_hotkey="chal_hk", miner_uid=2, role="challenger",
                         gen_ref=REF, trained_pointer=chal_ptr, corpus_digest="d",
                         train_block=9000, size=size),
        ],
    )
    report = dump_bench_report(BenchReport(
        round_id="777", created_block=9000,
        entries=(
            BenchEntry(role="king", size=size, miner_hotkey="king_hk", miner_uid=1,
                       trained_pointer=PTR, scores=REPORT_SCORES),
            BenchEntry(role="challenger", size=size, miner_hotkey="chal_hk",
                       miner_uid=2, trained_pointer=chal_ptr, scores=REPORT_SCORES),
        ),
    ))
    runner = _runner(cfg, _FakeStore({bench_report_key("777"): report}))
    runner._record_duel_checkpoints(manifest, now=1000.0)
    assert [(r.checkpoint_id, r.role) for r in runner.cascade.state.checkpoints] == [
        (PTR, "king"), (chal_ptr, "challenger")]
