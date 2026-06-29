"""Trainer round orchestration — train_one → manifest assembly with the GPU and
Hippius boundaries faked (no torch, no Hub, no S3)."""

from __future__ import annotations

import numpy as np

from metronome.shared.chain import Commitment
from metronome.shared.hippius import HubRef, HubUpload
from metronome.trainer import loop as loop_mod
from metronome.trainer.contract import TrainResult
from metronome.trainer.loop import TrainerRunner
from metronome.trainer.queue import SubmissionQueue

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
REF_OUT = "metronome/ckpt-out@sha256:" + "c" * 64


class _FakeStream:
    digest = "corpusdigest"
    n_series = 3
    total_points = 192

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def series(self):
        for _ in range(3):
            yield np.ones((1, 64))


class _FakeBaseTrainer:
    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
        for _ in stream:
            pass
        if logger:
            logger({"event": "step", "step": 1, "loss": 0.1})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000, train_seconds=1.0,
                           metrics={"final_loss": 0.1})


def _fake_upload(local_dir, repo, hub=None):
    return HubUpload(ref=HubRef.parse(REF_OUT), size_bytes=1)


def test_run_round_assembles_signed_ready_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub", _fake_upload)

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)

    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_B}", commit_block=6),
    ]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10, max_challengers=1)

    assert manifest.round_id == "1"
    king = manifest.entry_for_role("king")
    chal = manifest.entry_for_role("challenger")
    assert king.gen_ref == REF_A and chal.gen_ref == REF_B
    assert king.trained_pointer == f"metro-v1:trained:hippius:{REF_OUT}"
    assert king.corpus_digest == "corpusdigest"
    # contract/base-arch digests recorded once for the controlled-experiment gate
    assert manifest.contract_digest and manifest.base_arch_digest == cfg.training.base_arch_digest


def _patch_train_boundaries(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub", _fake_upload)


def test_run_round_skips_challenger_that_copies_the_king(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    # 'b' committed the king's exact generator ref — a copy. The round must train
    # only the king, with no challenger entry (the copy is filtered for free).
    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=6),
    ]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10, max_challengers=1)
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entry_for_role("challenger") is None


def test_run_round_with_queue_trains_then_dedups_next_round(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    queue = SubmissionQueue()
    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_B}", commit_block=6),
    ]
    # Round 1: challenger B is enqueued, selected, trained, and marked done.
    m1 = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10,
                          max_challengers=1, queue=queue)
    assert m1.entry_for_role("challenger").gen_ref == REF_B
    assert REF_B in queue.trained_refs

    # Round 2 (same reign): B already trained this reign ⇒ no challenger this time.
    m2 = runner.run_round(commits, king_hotkey="a", base_seed=2, block=20,
                          max_challengers=1, queue=queue)
    assert m2.entry_for_role("king").gen_ref == REF_A
    assert m2.entry_for_role("challenger") is None
