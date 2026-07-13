"""Trainer round orchestration — heat screen → per-size final → manifest
assembly, with the GPU and Hippius boundaries faked (no torch, no Hub, no S3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cascade.shared.chain import Commitment
from cascade.shared.hippius import HubRef, HubUpload
from cascade.shared.manifest import dump_manifest, load_manifest
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


def _fake_upload(local_dir, repo, hub=None, *, hf_repo=None, hf_token=None):
    return HubUpload(ref=HubRef.parse(REF_OUT), size_bytes=1)


def _patch_train_boundaries(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)


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
    def screen(ckpt_dir, gen, base_seed, block=None):
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

    def screen(ckpt_dir, gen, base_seed, block=None):
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


def test_heat_complete_marker_written_when_heat_settles(cfg, tmp_path, monkeypatch):
    """The provisioner's heat-teardown signal: once the field is screened and
    finalists chosen, ``work_root/<round_id>/heat_complete.json`` appears with
    the settled field — heat pods are safe to release from that moment."""
    _patch_train_boundaries(monkeypatch)
    scores = {"b": 0.9, "c": 0.2, "d": 0.5}
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False,
                           screen_fn=lambda ckpt_dir, gen, base_seed, block=None: scores[gen.hotkey])
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    marker = json.loads((tmp_path / "1" / "heat_complete.json").read_text())
    assert marker == {"round_id": "1", "screened": 3, "finalists": ["c"]}
    assert not (tmp_path / "1" / "heat_complete.json.tmp").exists()  # atomic publish


def test_heat_complete_marker_written_even_without_a_screen(cfg, tmp_path, monkeypatch):
    """Field ≤ finalists ⇒ no screening runs, but the heat stage is still
    settled (no heat dispatch can follow) — the marker must appear anyway."""
    _patch_train_boundaries(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    runner.run_round(commits, king_hotkey="a", base_seed=7, block=10)

    marker = json.loads((tmp_path / "7" / "heat_complete.json").read_text())
    assert marker == {"round_id": "7", "screened": 1, "finalists": ["b"]}


def test_heat_records_informational_standings(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    # Same field as above: cheapest (c) advances, everyone else is screened.
    scores = {"b": 0.9, "c": 0.2, "d": 0.5}

    def screen(ckpt_dir, gen, base_seed, block=None):
        return scores[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    heat = manifest.heat
    assert heat is not None
    assert heat.finalists == 1
    by_hk = {e.hotkey: e for e in heat.entrants}
    assert set(by_hk) == {"b", "c", "d"}
    # ranked cheapest-first; relative to the best (c), never the raw scores
    assert by_hk["c"].rank == 1 and by_hk["c"].status == "advanced"
    assert by_hk["c"].rel_score == 1.0
    assert by_hk["d"].rank == 2 and by_hk["d"].status == "screened"
    assert by_hk["d"].rel_score == pytest.approx(2.5)     # 0.5 / 0.2
    assert by_hk["b"].rank == 3 and by_hk["b"].status == "screened"
    assert by_hk["b"].rel_score == pytest.approx(4.5)     # 0.9 / 0.2
    # survives serialisation to the wire (rides the manifest, unsigned)
    assert load_manifest(dump_manifest(manifest)).heat == heat


def test_heat_none_when_no_screen_runs(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    # One challenger, finalists = 1 ⇒ it fits without a screen; no standings to show.
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=lambda *a: 0.0)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert manifest.heat is None
    # a heat-less manifest carries no "heat" key at all (byte-compatible wire)
    assert "heat" not in json.loads(dump_manifest(manifest))


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

    import cascade.trainer.remote as remote_mod
    from cascade.shared.manifest import TrainedEntry, format_trained_pointer

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

    def screen(ckpt_dir, gen, base_seed, block=None):
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


def test_burn_happens_after_heat_not_at_entry(cfg, tmp_path, monkeypatch):
    # A round that dies MID-HEAT (pod fleet lost, trainer crash) must not consume
    # anyone's one lifetime submission: the burn is persisted only after the heat
    # stage completes. The retried round then re-admits the same field.
    _patch_train_boundaries(monkeypatch)
    assert cfg.round.one_submission_per_hotkey is True
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6)]

    def _boom(*a, **k):
        raise RuntimeError("fleet died mid-heat")

    monkeypatch.setattr(runner, "_run_heat", _boom)
    with pytest.raises(RuntimeError, match="mid-heat"):
        runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert not (tmp_path / cfg.round.submissions_db_path).exists()  # nobody burned

    monkeypatch.undo()
    _patch_train_boundaries(monkeypatch)  # undo() dropped the boundary patches too
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=2, block=20)
    assert manifest.entries_for_role("challenger")  # 'b' still had its shot
    assert (tmp_path / cfg.round.submissions_db_path).exists()  # …and is burned now


def test_heat_and_final_contracts_use_scaled_guard(cfg, tmp_path, monkeypatch):
    # The heat trains under for_hours(heat_train_hours): token budget AND the hard
    # wall-clock guard scale to the cheap budget (a staller costs minutes, not the
    # final's max_train_seconds). The final keeps the pinned contract guard.
    _patch_train_boundaries(monkeypatch)
    contracts = []

    class _Recorder(_FakeBaseTrainer):
        def train(self, stream, contract, **kw):
            contracts.append(contract)
            return super().train(stream, contract, **kw)

    def screen(ckpt_dir, gen, base_seed, block=None):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_Recorder(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    heat = [c for c in contracts if c.target_train_hours == cfg.round.heat_train_hours]
    final = [c for c in contracts if c.target_train_hours == cfg.training.target_train_hours]
    assert len(heat) == 3 and len(final) == 2  # 3 screened; king + finalist finals
    expected_guard = max(
        int(round(cfg.round.heat_guard_factor * cfg.round.heat_train_hours * 3600)),
        cfg.round.heat_guard_floor_seconds,
    )
    assert all(c.max_train_seconds == expected_guard for c in heat)
    assert all(c.train_tokens == c.tokens_for_hours(cfg.round.heat_train_hours) for c in heat)
    assert all(c.max_train_seconds == cfg.training.max_train_seconds for c in final)


def test_screen_receives_epoch_boundary_block(cfg, tmp_path, monkeypatch):
    # The screener gets the round's epoch-boundary block (cutoff_block), not the
    # current height, so a daily-snapshot pool picks the validator's snapshot.
    _patch_train_boundaries(monkeypatch)
    seen_blocks = []

    def screen(ckpt_dir, gen, base_seed, block=None):
        seen_blocks.append(block)
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    runner.run_round(commits, king_hotkey="a", base_seed=1, block=60, cutoff_block=50)
    assert seen_blocks == [50, 50, 50]


def test_remote_dispatch_retries_once_on_next_host(cfg, tmp_path, monkeypatch):
    # Rented pods churn: every dispatch (heat AND final, king included) that fails
    # once is retried on the next host instead of dropping the challenger's only
    # slot or aborting the round. Two hosts ⇒ the retry lands on the other box.
    import cascade.trainer.remote as remote_mod
    from cascade.shared.manifest import TrainedEntry, format_trained_pointer

    _patch_train_boundaries(monkeypatch)
    host_a, host_b = object(), object()
    calls: dict[tuple, list] = {}
    failed_once: set[tuple] = set()

    class _FlakyDisp:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block,
                     arch_preset=None, train_hours=None, repo_suffix=""):
            key = (hotkey, role, train_hours is not None)
            calls.setdefault(key, []).append(host)
            if key not in failed_once:
                failed_once.add(key)
                raise RuntimeError("pod flake")
            return TrainedEntry(
                miner_hotkey=hotkey, miner_uid=uid, role=role, gen_ref=gen_ref,
                trained_pointer=format_trained_pointer(REF_OUT), corpus_digest="d",
                train_block=block, gpu_name="", size=arch_preset or cfg.training.arch_preset,
            )

    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _FlakyDisp)

    def screen(ckpt_dir, gen, base_seed, block=None):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen,
                           remote_hosts=[host_a, host_b], trainer_spec="m:C")
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    # every dispatch survived its first failure via a retry on the OTHER host
    assert manifest.entry_for_role("king").gen_ref == REF_A
    assert manifest.entry_for_role("challenger").miner_hotkey == "c"
    for key, hosts_seen in calls.items():
        assert len(hosts_seen) == 2, key
        assert hosts_seen[0] is not hosts_seen[1], key


def test_reload_remote_hosts_per_round(cfg, tmp_path):
    # The elastic-fleet seam: hosts TOML re-read per round; missing/empty file ⇒
    # local round; the provisioner writing the file brings the fleet up without a
    # trainer restart, and emptying it tears the fleet down again.
    hosts_path = tmp_path / "hosts.toml"
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           remote_hosts_path=hosts_path, hosts_wait_seconds=0)
    runner._reload_remote_hosts()
    assert runner.remote_hosts is None                      # no file yet ⇒ local

    hosts_path.write_text('[[host]]\nname = "pod-a"\nhost = "1.2.3.4"\n', encoding="utf-8")
    runner._reload_remote_hosts()
    assert [h.name for h in runner.remote_hosts] == ["pod-a"]

    hosts_path.write_text("", encoding="utf-8")             # fleet torn down
    runner._reload_remote_hosts()
    assert runner.remote_hosts is None


def test_plan_payload_counts_the_real_eligible_field(cfg, tmp_path):
    # --plan-only runs the round's own eligibility pipeline (dedup + burn filter),
    # so the provisioner sizes pods off what the heat will actually train.
    from cascade.trainer.main import _plan_payload

    class _StubClient:
        def current_block(self):
            return 3 * cfg.round.epoch_blocks + 100

        def poll_commitments(self):
            return [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
                    _commit(2, "c", REF_A, 7)]   # 'c' copies the king's ref → deduped

        def highest_incentive_hotkey(self):
            return "a"

    payload = _plan_payload(cfg, _StubClient(), tmp_path)
    assert payload["king"] == "a"
    assert payload["resolved"] == 3
    assert payload["challengers"] == 1               # only 'b' survives dedup
    assert payload["eligible_challengers"] == 1
    assert payload["next_boundary_block"] == 4 * cfg.round.epoch_blocks
    assert payload["blocks_to_boundary"] == cfg.round.epoch_blocks - 100

    # burned hotkeys drop out of the eligible count
    (tmp_path / cfg.round.submissions_db_path).write_text(json.dumps(["b"]), encoding="utf-8")
    assert _plan_payload(cfg, _StubClient(), tmp_path)["eligible_challengers"] == 0


def test_screen_block_derived_when_cutoff_omitted(cfg, tmp_path, monkeypatch):
    # Direct callers may omit cutoff_block; the screener must still get the
    # derived epoch boundary (never None, which a bucket pool reads as "newest").
    _patch_train_boundaries(monkeypatch)
    seen_blocks = []

    def screen(ckpt_dir, gen, base_seed, block=None):
        seen_blocks.append(block)
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen)
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    block = 2 * cfg.round.epoch_blocks + 123
    runner.run_round(commits, king_hotkey="a", base_seed=1, block=block)
    assert seen_blocks == [2 * cfg.round.epoch_blocks] * 3


def test_stage_tagged_hosts_split_heat_from_final(cfg, tmp_path, monkeypatch):
    # The cheap-GPU seam: hosts tagged stage="heat" serve only the screen
    # trainings (a cheaper SKU class), stage="final" only the king/finalist runs
    # (the SKU the validator's gpu_name gate pairs). Untagged = both.
    from types import SimpleNamespace

    import cascade.trainer.remote as remote_mod
    from cascade.shared.manifest import TrainedEntry, format_trained_pointer

    _patch_train_boundaries(monkeypatch)
    cheap_a = SimpleNamespace(name="a6000-1", stage="heat")
    cheap_b = SimpleNamespace(name="a6000-2", stage="heat")
    big = SimpleNamespace(name="l40-1", stage="final")
    dispatched: list[tuple[str, str, bool]] = []

    class _FakeDisp:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block,
                     arch_preset=None, train_hours=None, repo_suffix=""):
            dispatched.append((host.name, role, train_hours is not None))
            return TrainedEntry(
                miner_hotkey=hotkey, miner_uid=uid, role=role, gen_ref=gen_ref,
                trained_pointer=format_trained_pointer(REF_OUT), corpus_digest="d",
                train_block=block, gpu_name="", size=arch_preset or cfg.training.arch_preset,
            )

    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _FakeDisp)

    def screen(ckpt_dir, gen, base_seed, block=None):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen,
                           remote_hosts=[cheap_a, cheap_b, big], trainer_spec="m:C")
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    heat_hosts = {name for name, _, is_heat in dispatched if is_heat}
    final_hosts = {name for name, _, is_heat in dispatched if not is_heat}
    assert heat_hosts <= {"a6000-1", "a6000-2"}          # heats never on the L40
    assert final_hosts == {"l40-1"}                      # final never on the A6000s
    assert manifest.entry_for_role("challenger").miner_hotkey == "c"


def test_stage_filter_falls_back_when_no_host_matches(cfg, tmp_path):
    from types import SimpleNamespace

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           remote_hosts=[SimpleNamespace(name="l40", stage="final")])
    assert [h.name for h in runner._hosts_for("final")] == ["l40"]
    # nothing tagged for the heat ⇒ use the whole fleet rather than strand the stage
    assert [h.name for h in runner._hosts_for("heat")] == ["l40"]
    # legacy untagged host objects serve both stages
    legacy = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           remote_hosts=[object(), object()])
    assert len(legacy._hosts_for("heat")) == 2 and len(legacy._hosts_for("final")) == 2


def test_heat_dispatch_uses_tight_ssh_timeout(cfg, tmp_path, monkeypatch):
    # The outer SSH timeout is the only bound on a fully wedged pod; heats cap
    # it at scaled-guard + 30min instead of the 6h final default.
    from types import SimpleNamespace

    import cascade.trainer.remote as remote_mod
    from cascade.shared.manifest import TrainedEntry, format_trained_pointer

    _patch_train_boundaries(monkeypatch)
    timeouts: list[tuple[bool, int]] = []

    class _FakeDisp:
        def __init__(self, *, trainer_spec, timeout_seconds):
            self.timeout_seconds = timeout_seconds

        def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block,
                     arch_preset=None, train_hours=None, repo_suffix=""):
            timeouts.append((train_hours is not None, self.timeout_seconds))
            return TrainedEntry(
                miner_hotkey=hotkey, miner_uid=uid, role=role, gen_ref=gen_ref,
                trained_pointer=format_trained_pointer(REF_OUT), corpus_digest="d",
                train_block=block, gpu_name="", size=arch_preset or cfg.training.arch_preset,
            )

    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _FakeDisp)

    def screen(ckpt_dir, gen, base_seed, block=None):
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, screen_fn=screen,
                           remote_hosts=[SimpleNamespace(name="p", stage="any")],
                           trainer_spec="m:C")
    commits = [_commit(0, "a", REF_A, 5), _commit(1, "b", REF_B, 6),
               _commit(2, "c", REF_C, 7), _commit(3, "d", REF_D, 8)]
    runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    heat_guard = cfg.screen_contract().for_hours(
        cfg.round.heat_train_hours,
        guard_factor=cfg.round.heat_guard_factor,
        guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
    ).max_train_seconds
    heat_timeouts = {t for is_heat, t in timeouts if is_heat}
    final_timeouts = {t for is_heat, t in timeouts if not is_heat}
    assert heat_timeouts == {heat_guard + 1800}          # 5400 + 1800 on chain.toml
    assert final_timeouts == {runner.remote_timeout_seconds}
