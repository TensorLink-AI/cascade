"""Trainer round orchestration — train_one → manifest assembly with the GPU and
Hippius boundaries faked (no torch, no IPFS, no S3)."""

from __future__ import annotations

import numpy as np

from metronome.shared.chain import Commitment
from metronome.shared.hippius import RegistryUpload
from metronome.trainer import loop as loop_mod
from metronome.trainer.contract import TrainResult
from metronome.trainer.loop import TrainerRunner

CID_A = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
CID_B = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
CID_OUT = "QmPChd2hVbrJ6bfo3WBcTW4iZnpHm8TEzWkLHmLpXhF68A"


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


def test_run_round_assembles_signed_ready_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_registry", lambda cid, dest, reg, **k: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(
        loop_mod, "upload_dir_to_registry",
        lambda local_dir, reg: RegistryUpload(cid=CID_OUT, tar_digest="deadbeef", size_bytes=1),
    )

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)

    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{CID_A}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{CID_B}", commit_block=6),
    ]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10, max_challengers=1)

    assert manifest.round_id == "1"
    king = manifest.entry_for_role("king")
    chal = manifest.entry_for_role("challenger")
    assert king.gen_cid == CID_A and chal.gen_cid == CID_B
    assert king.trained_pointer == f"metro-v1:trained:hippius:{CID_OUT}"
    assert king.tar_digest == "deadbeef"
    assert king.corpus_digest == "corpusdigest"
    # contract/base-arch digests recorded once for the controlled-experiment gate
    assert manifest.contract_digest and manifest.base_arch_digest == cfg.training.base_arch_digest
