"""Trainer round orchestration — heat screen → per-size final → manifest
assembly, with the GPU and Hippius boundaries faked (no torch, no Hub, no S3)."""

from __future__ import annotations

import numpy as np

from cascade.shared.chain import Commitment
from cascade.shared.hippius import HubRef, HubUpload
from cascade.trainer import loop as loop_mod
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner, resolve_commitments

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
REF_C = "carol/gen-c@sha256:" + "c" * 64
REF_D = "dave/gen-d@sha256:" + "d" * 64
REF_OUT = "cascade/ckpt-out@sha256:" + "e" * 64


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


def _patch_train_boundaries(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub", _fake_upload)


def _commit(uid, hotkey, ref, block):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


def test_run_round_trains_king_and_challenger_at_every_size(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    assert manifest.round_id == "1"
    sizes = sorted(s.arch_preset for s in cfg.training.final_sizes())
    assert len(sizes) >= 2  # chain.toml ships primary + a [[training.sizes]]
    # One (king, challenger) pair per configured size, each tagged with its size.
    assert sorted(e.size for e in manifest.entries_for_role("king")) == sizes
    assert sorted(e.size for e in manifest.entries_for_role("challenger")) == sizes
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entry_for_role("challenger").gen_ref == REF_B
    # contract/base-arch digests recorded once for the controlled-experiment gate
    assert manifest.contract_digest and manifest.base_arch_digest == cfg.training.base_arch_digest


def test_run_round_skips_challenger_that_copies_the_king(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    # 'b' committed the king's exact generator ref — a copy. The round trains only
    # the king (at every size), with no challenger entry (the copy is filtered).
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_A, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entries_for_role("challenger") == []


def test_heat_screens_field_down_to_one_finalist(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    # Three challengers, finalists = 1 (chain.toml). The cheapest heat score wins.
    scores = {"b": 0.9, "c": 0.2, "d": 0.5}
    seen: list[str] = []

    def screen(ckpt_dir, gen, base_seed):
        seen.append(gen.hotkey)
        return scores[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    assert sorted(seen) == ["b", "c", "d"]  # every challenger got a heat run
    chal = manifest.entry_for_role("challenger")
    assert chal.miner_hotkey == "c"          # lowest geomean advances
    # the single finalist is trained at every size in the final
    sizes = sorted(s.arch_preset for s in cfg.training.final_sizes())
    assert sorted(e.size for e in manifest.entries_for_role("challenger")) == sizes


def test_no_screen_fn_takes_lowest_uid_when_field_exceeds_finalists(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)  # no screen_fn
    commits = [_commit(0, "a", REF_A, 5), _commit(2, "c", REF_C, 7), _commit(1, "b", REF_B, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    # finalists = 1, no screener ⇒ field order (lowest UID first) → challenger b (uid 1)
    assert manifest.entry_for_role("challenger").miner_hotkey == "b"


def test_run_round_cutoff_excludes_late_commits(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    # Challenger 'b' committed at block 100, at/after the epoch boundary (50) ⇒
    # not eligible this round; only the king (committed at 5) trains.
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 100)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10, cutoff_block=50)
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entries_for_role("challenger") == []


def test_resolve_commitments_cutoff_is_strict_and_latest_eligible_wins():
    # b re-deploys: REF_B at block 5 (eligible) then REF_C at block 60 (late).
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 5), _commit(1, "b", REF_C, 60)]
    resolved = {r.hotkey: r.ref for r in resolve_commitments(commits, cutoff_block=50)}
    assert resolved == {"a": REF_A, "b": REF_B}  # late re-deploy ignored; pre-cutoff ref kept
    # exactly-at-boundary is excluded (strict <)
    assert all(r.hotkey != "x" for r in resolve_commitments([_commit(9, "x", REF_D, 50)], cutoff_block=50))
