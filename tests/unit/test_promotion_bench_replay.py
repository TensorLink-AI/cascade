"""Promotion's boundary replay of the reign's PUBLISHED bench reports
(:meth:`TrainerRunner._replay_reign_bench_reports`)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cascade.shared.manifest import BenchScores
from cascade.trainer.loop import TrainerRunner

_BENCH = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)


class _FakeStore:
    def __init__(self):
        self.texts: dict[str, str] = {}
        self.reads: list[str] = []

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.texts[key] = text

    def get_text(self, key):
        self.reads.append(key)
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


@pytest.fixture()
def cascade_cfg(cfg):
    return replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=True))


def _runner(cfg, tmp_path):
    runner = TrainerRunner(cfg=cfg, base_trainer=None, work_root=tmp_path, use_sandbox=False)
    runner._manifest_store = _FakeStore()
    return runner


# ── boundary replay of published bench reports ───────────────────────────────
# An out-of-band report (a mop-up after a failed leg, a publish during trainer
# downtime) must still feed the candidate pool before a promotion fires.


class _FakePromotion:
    king_hotkey = "5King"

    def __init__(self):
        self.calls: list = []

    def record_bench(self, manifest, report):
        self.calls.append((manifest.round_id, tuple(e.role for e in report.entries)))
        return len(report.entries)


def _index_row(rid, esb, king):
    return {"round_id": rid, "status": "scored", "post_round_king_hotkey": king,
            "epoch_start_block": esb, "validator_hotkey": "v"}


def _stub_manifest_loader(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cascade.shared.manifest.load_manifest",
        lambda text: SimpleNamespace(round_id=json.loads(text)["round_id"],
                                     warm_start_ckpt=""))


def _seed_replay_store(store, *, signed_ok=True):
    from cascade.shared.bench_report import (
        BenchEntry,
        BenchReport,
        bench_report_key,
        dump_bench_report,
    )
    from cascade.shared.hippius import RECEIPT_INDEX_KEY, manifest_round_key

    store.texts[RECEIPT_INDEX_KEY] = json.dumps({"rounds": [
        _index_row("700", 100, "OldKing"),   # prior reign: outside the tail
        _index_row("776", 200, "5King"),     # reign round, report never published
        _index_row("777", 300, "5King"),     # reign round with an OOB report
    ]})
    store.texts[manifest_round_key("777")] = json.dumps({"round_id": "777"})
    report = BenchReport(round_id="777", created_block=5, entries=(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="a", miner_uid=1,
                   trained_pointer="ptr-k", scores=_BENCH),
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="b",
                   miner_uid=2, trained_pointer="ptr-c", scores=_BENCH),
    ))
    store.texts[bench_report_key("777")] = dump_bench_report(report)


def test_boundary_replay_ingests_out_of_band_bench_reports(cascade_cfg, tmp_path,
                                                           monkeypatch):
    # trainer_hotkey blanked: this test probes the WIRING (tail → fetch →
    # record_bench); the signature discipline gets its own test below.
    cfg = replace(cascade_cfg, manifest=replace(cascade_cfg.manifest,
                                                trainer_hotkey=""))
    runner = _runner(cfg, tmp_path)
    _stub_manifest_loader(monkeypatch)
    store = runner.manifest_store()
    _seed_replay_store(store)
    runner.promotion = _FakePromotion()

    runner._replay_reign_bench_reports()

    # 777 replayed; 776 (no report) skipped silently; 700 (prior reign) never
    # part of the tail. record_bench owns dedup/generation gating downstream.
    assert runner.promotion.calls == [("777", ("king", "challenger"))]
    from cascade.shared.hippius import manifest_round_key

    assert manifest_round_key("700") not in store.reads

    # idempotent to re-run at the next boundary (record_bench dedupes; the
    # replay itself must not error or grow state)
    runner._replay_reign_bench_reports()
    assert len(runner.promotion.calls) == 2


def test_boundary_replay_rejects_unsigned_reports_under_a_pinned_hotkey(
        cascade_cfg, tmp_path, monkeypatch):
    # With a pinned trainer hotkey, an unsigned/forged report replays NOTHING —
    # same discipline as the deploy-time backfill.
    cfg = replace(cascade_cfg, manifest=replace(cascade_cfg.manifest,
                                                trainer_hotkey="5Pinned"))
    runner = _runner(cfg, tmp_path)
    _stub_manifest_loader(monkeypatch)
    store = runner.manifest_store()
    _seed_replay_store(store)
    runner.promotion = _FakePromotion()

    runner._replay_reign_bench_reports()
    assert runner.promotion.calls == []


def test_boundary_replay_is_a_no_op_without_an_anchored_engine(
        cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path)
    runner.promotion = None
    runner._replay_reign_bench_reports()    # must not raise, must not read
    assert runner.manifest_store().reads == []


# ── DEC-CA-0036: funded challengers bench on their payer's pod, verified on the king's ──





















