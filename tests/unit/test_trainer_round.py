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


def test_run_round_trains_king_and_challenger_at_every_size(two_size_cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    cfg = two_size_cfg
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    assert manifest.round_id == "1"
    sizes = sorted(t.arch_preset for t in cfg.throne_contracts())
    assert len(sizes) == 2  # combined throne over both sizes
    # One (king, challenger) pair per throne size, each tagged with its size.
    assert sorted(e.size for e in manifest.entries_for_role("king")) == sizes
    assert sorted(e.size for e in manifest.entries_for_role("challenger")) == sizes
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entry_for_role("challenger").gen_ref == REF_B
    # contract/base-arch digests recorded once for the controlled-experiment gate
    assert manifest.contract_digest and manifest.base_arch_digest == cfg.training.base_arch_digest


def test_run_round_single_size_at_launch(cfg, tmp_path, monkeypatch):
    # Shipped config (20M disabled) ⇒ king + challenger trained at the 4M primary
    # only: exactly one entry per role, tagged with the primary preset.
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert [e.size for e in manifest.entries_for_role("king")] == [cfg.training.arch_preset]
    assert [e.size for e in manifest.entries_for_role("challenger")] == [cfg.training.arch_preset]


def test_sliding_window_screen_small_throne_big(two_size_cfg, tmp_path, monkeypatch):
    # The seam: screen at the small primary, train + judge the throne at the
    # bigger size ONLY (4M never appears in the manifest — it's just the screen).
    from dataclasses import replace

    _patch_train_boundaries(monkeypatch)
    cfg = replace(two_size_cfg, round=replace(two_size_cfg.round,
                  screen_size=two_size_cfg.training.arch_preset,
                  throne_sizes=("toto2-test-xl",)))
    # screen only matters when it has to choose; give it 3 challengers + a screener.
    def screen(ckpt_dir, gen, base_seed):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    # throne entries are the big size only — the 4M screen is internal, never published.
    assert {e.size for e in manifest.entries} == {"toto2-test-xl"}
    assert manifest.entry_for_role("challenger").miner_hotkey == "c"  # heat winner promoted


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
    # the single finalist is trained at every throne size
    sizes = sorted(t.arch_preset for t in cfg.throne_contracts())
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


def test_run_round_king_is_cutoff_exempt(cfg, tmp_path, monkeypatch):
    # The reigning king 'a' re-committed at block 100 (AT/AFTER the epoch boundary
    # 50); challenger 'b' is pre-cutoff. The king is exempt from the submission
    # cutoff — it must still be trained AS king, never silently replaced by the
    # challenger (which would make the validator reject the round king_resyncing).
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 100), _commit(1, "b", REF_B, 5)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=110, cutoff_block=50)
    assert manifest.entry_for_role("king").gen_ref == REF_A          # king kept, not swapped
    assert [e.gen_ref for e in manifest.entries_for_role("challenger")] == [REF_B]


def test_plan_round_never_silently_swaps_a_named_champion():
    from cascade.trainer.loop import ResolvedGenerator, plan_round

    field = [ResolvedGenerator(hotkey="b", uid=1, ref=REF_B)]
    # champion 'a' named + resolved cutoff-exempt, absent from the challenger field
    king_a = ResolvedGenerator(hotkey="a", uid=0, ref=REF_A)
    plan = plan_round(field, "a", king=king_a)
    assert plan.king.hotkey == "a"                       # champion wins
    assert [c.hotkey for c in plan.challengers] == ["b"]
    # genesis (no champion named) still promotes the lowest-UID interim king
    assert plan_round(field, None).king.hotkey == "b"


def test_resolve_commitments_cutoff_is_strict_and_latest_eligible_wins():
    # b re-deploys: REF_B at block 5 (eligible) then REF_C at block 60 (late).
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 5), _commit(1, "b", REF_C, 60)]
    resolved = {r.hotkey: r.ref for r in resolve_commitments(commits, cutoff_block=50)}
    assert resolved == {"a": REF_A, "b": REF_B}  # late re-deploy ignored; pre-cutoff ref kept
    # exactly-at-boundary is excluded (strict <)
    assert all(r.hotkey != "x" for r in resolve_commitments([_commit(9, "x", REF_D, 50)], cutoff_block=50))


def test_one_submission_per_hotkey_burns_challenger(cfg, tmp_path, monkeypatch):
    # chain.toml ships one_submission_per_hotkey = true: a challenger that competes
    # in one round is burned (persisted under work_root) and skipped in the next.
    _patch_train_boundaries(monkeypatch)
    assert cfg.round.one_submission_per_hotkey is True
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    m1 = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert m1.entries_for_role("challenger")  # 'b' competed this round
    # Same field next epoch ⇒ 'b' already used its one submission ⇒ king-only round.
    m2 = runner.run_round(commits, king_hotkey="a", base_seed=2, block=20)
    assert m2.entries_for_role("challenger") == []


def test_one_submission_per_hotkey_off_recompetes(cfg, tmp_path, monkeypatch):
    from dataclasses import replace
    _patch_train_boundaries(monkeypatch)
    cfg = replace(cfg, round=replace(cfg.round, one_submission_per_hotkey=False))
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    m1 = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    m2 = runner.run_round(commits, king_hotkey="a", base_seed=2, block=20)
    assert m1.entries_for_role("challenger") and m2.entries_for_role("challenger")  # re-competes


def test_run_round_remote_heat_dispatches_to_pod(cfg, tmp_path, monkeypatch):
    # With remote_hosts set, the HEAT trains on the pod (dispatch) — not locally —
    # at the cheap heat budget + a per-challenger repo, then screens the fetched
    # checkpoints. Proves the wallet-safe split can run the heat on remote GPUs.
    from dataclasses import replace

    from cascade.shared.manifest import TrainedEntry, format_trained_pointer
    import cascade.trainer.remote as remote_mod

    _patch_train_boundaries(monkeypatch)  # patches fetch_from_hub → returns dest
    dispatched = []

    class _FakeDisp:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block,
                     arch_preset=None, train_hours=None, repo_suffix=""):
            dispatched.append({"hotkey": hotkey, "role": role, "arch_preset": arch_preset,
                               "train_hours": train_hours, "repo_suffix": repo_suffix})
            return TrainedEntry(
                miner_hotkey=hotkey, miner_uid=uid, role=role, gen_ref=gen_ref,
                trained_pointer=format_trained_pointer(REF_OUT), corpus_digest="d",
                train_block=block, gpu_name="", size=arch_preset or cfg.training.arch_preset,
            )

    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _FakeDisp)

    def screen(ckpt_dir, gen, base_seed):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]  # 'c' is best

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen,
                           remote_hosts=[object()], trainer_spec="m:C")
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    # the 3 challengers were heat-trained on the pod at the cheap budget + unique repos
    heat = [d for d in dispatched if d["train_hours"] is not None]
    assert len(heat) == 3
    assert all(d["train_hours"] == cfg.round.heat_train_hours for d in heat)
    assert sorted(d["repo_suffix"] for d in heat) == ["-heat-u1", "-heat-u2", "-heat-u3"]
    # heat winner ('c') promoted; the final dispatch carries no heat overrides
    assert manifest.entry_for_role("challenger").miner_hotkey == "c"
    assert any(d["role"] == "king" and d["train_hours"] is None for d in dispatched)
