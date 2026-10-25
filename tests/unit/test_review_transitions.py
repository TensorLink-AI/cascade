"""Exercise the recovered contract at process and era boundaries."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.shared.manifest import TrainingManifest, dump_manifest
from cascade.shared.promotion import PromotedMember, PromotionRecord

from .test_review_failure_boundaries import _repo


def test_local_score_materializes_and_consumes_the_v2_shards(cfg, tmp_path, monkeypatch):
    from cascade.miner.score import score_generator
    from cascade.miner.training_context import with_contract

    from .test_miner_score import _FakeTrainer

    trainer = _FakeTrainer()
    monkeypatch.setattr("cascade.trainer.main._load_trainer", lambda _: trainer)
    # The data-generation and shard-reading path is real; only the GPU work
    # and evaluation are replaced at their public interfaces.
    from cascade.eval.scoring import WindowScore

    monkeypatch.setattr("cascade.validator.evaluator.evaluate_checkpoint", lambda ckpt, windows, **kw: [
        WindowScore(str(i), 1.0, np.full(9, 0.5), 5.0) for i in range(len(windows))])
    cfg = with_contract(cfg, replace(cfg.training.v2_contract, ref_throughput_tokens_per_s=1,
                                     target_train_hours=1))
    cfg = replace(cfg, generation=replace(cfg.generation, target_points=32768,
                                         window_seconds=10, shard_points=8192))
    result = score_generator(_repo(tmp_path / "gen", "(8, 256)"), cfg, n_windows=3,
                             train_hours=1, cache_dir=tmp_path / "work")
    assert result.n_series > 0 and len(result.corpus_digest) == 64
    assert np.isfinite(result.geomean) and result.geomean > 0
    assert trainer.calls == [{}]
    assert not list((tmp_path / "work").glob("ckpt-*"))


def test_v2_baseline_requires_a_full_policy_bound_proof(cfg, tmp_path, monkeypatch):
    from cascade.trainer.launch_preflight import proof_path, qualify, ready

    cfg = replace(cfg, manifest=replace(cfg.manifest, validator_hotkey="fixture-validator"),
                  generator=replace(cfg.generator, corpus_target_points=2048, corpus_n_series=4))
    repo = _repo(tmp_path / "gen", "(8, 256)")
    ref = "fixtures/baseline@sha256:" + "a" * 64
    monkeypatch.setattr("cascade.shared.hippius.fetch_from_hub",
                        lambda ref, directory, hub: shutil.copytree(repo, directory, dirs_exist_ok=True))
    assert not ready(cfg, ref, tmp_path / "state")
    proof = qualify(cfg, ref, tmp_path / "state")
    assert proof["verification"]["scope"] == "full"
    assert ready(cfg, ref, tmp_path / "state")
    shared = replace(cfg, manifest=replace(cfg.manifest,
                        validator_hotkey=cfg.manifest.trainer_hotkey))
    assert ready(shared, ref, tmp_path / "state")
    # A later crowned generator can use the established launch proof.
    assert ready(cfg, ref.replace("a" * 64, "b" * 64), tmp_path / "state")
    changed = replace(cfg, generation=replace(cfg.generation, child_memory_mb=1024))
    assert not ready(changed, ref, tmp_path / "state")
    # A static/quick proof cannot authorize a paid baseline leg.
    (tmp_path / "state" / "v2-launch.json").unlink()
    proof["verification"]["scope"] = "quick"
    proof_path(tmp_path / "state", ref).write_text(json.dumps(proof))
    assert not ready(cfg, ref, tmp_path / "state")


def test_late_or_future_passing_check_does_not_fire_a_promotion(cfg, tmp_path):
    from .test_dec55_trainer_promotion import KING_PTR, _Runner, _v2

    cfg, gate, length = _v2(cfg)
    boundary = gate + 4 * length + 900
    check = {"result": "pass", "round_id": "r-check", "king_pointer": KING_PTR,
             "block": boundary}
    late = _Runner(cfg, tmp_path / "late", (check, boundary, "r-check"))
    late._promotion_boundary_step("KING", boundary + 3 * length, "late", effective_era=99)
    assert not late.published and late.promotion.generation == 0
    future = _Runner(cfg, tmp_path / "future", (check, boundary, "r-check"))
    future._promotion_boundary_step("KING", boundary - 900, "future", effective_era=99)
    assert not future.published and future.promotion.generation == 0


def test_unattesting_validator_needs_a_timely_hash_linked_promotion_notice(cfg):
    from cascade.shared.era import min_effective_era
    from cascade.validator.king_check import KingCheckMixin

    boundary = 100 * cfg.round.epoch_blocks * cfg.round.era_settlements
    effective = min_effective_era(cfg.round, boundary)
    rec = PromotionRecord(1, "king", "notice", boundary,
                          (PromotedMember("checkpoint", "m", "r", 0.5),), effective_era=effective)
    notice = TrainingManifest("notice", boundary, "c", "d", "e",
                              promotion_notices={"1": hashlib.sha256(rec.canonical_body()).hexdigest()})
    current = TrainingManifest("current", (effective + 1) * cfg.round.epoch_blocks * cfg.round.era_settlements,
                               "c", "d", "e", prev_round_id="notice",
                               prev_manifest_sha256=hashlib.sha256(dump_manifest(notice).encode()).hexdigest())
    runner = SimpleNamespace(cfg=cfg, verify_signatures=False, _promotion_check_manifest=current,
                             _lane_b_load_settlement=lambda rid: notice)
    assert KingCheckMixin._v2_notice_reason(runner, rec) is None
    runner._promotion_check_manifest = replace(current, prev_manifest_sha256="f" * 64)
    assert "hash differs" in KingCheckMixin._v2_notice_reason(runner, rec)
    runner._promotion_check_manifest = replace(current, prev_manifest_sha256="")
    assert "predecessor hashes" in KingCheckMixin._v2_notice_reason(runner, rec)
    # The trainer repeats the notice until the effective era: a late copy above
    # the timely announcement does not make the promotion late.
    runner._promotion_check_manifest = replace(current, promotion_notices=notice.promotion_notices)
    assert KingCheckMixin._v2_notice_reason(runner, rec) is None
    bare = replace(notice, promotion_notices=None)
    runner._promotion_check_manifest = replace(
        current, promotion_notices=notice.promotion_notices,
        prev_manifest_sha256=hashlib.sha256(dump_manifest(bare).encode()).hexdigest())
    runner._lane_b_load_settlement = lambda rid: bare
    assert "too late" in KingCheckMixin._v2_notice_reason(runner, rec)


def test_trainer_keeps_its_prior_init_until_a_promotion_notice_really_publishes(cfg):
    from cascade.shared.era import min_effective_era
    from cascade.shared.hippius import manifest_round_key
    from cascade.shared.manifest import sign_manifest
    from cascade.shared.promotion import (
        dump_promotion_record,
        promotion_record_key,
        sign_promotion_record,
    )
    from cascade.trainer.rolling import LegOps
    from cascade.trainer.runner_promotion import _PromotionMixin

    from .test_marketcast_validator import TRAINER_KP

    length = cfg.round.epoch_blocks * cfg.round.era_settlements
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=2 * length),
                  manifest=replace(cfg.manifest, trainer_hotkey=TRAINER_KP.ss58_address))
    block = (cfg.v2.random_init_eras + 5) * length
    record = sign_promotion_record(PromotionRecord(1, "king", "check", block,
        (PromotedMember("checkpoint", "m", "r", 0.5),),
        effective_era=min_effective_era(cfg.round, block)), TRAINER_KP)
    texts = {promotion_record_key(1): dump_promotion_record(record)}
    store = SimpleNamespace(get_text=lambda key: texts[key])
    engine = SimpleNamespace(generation=1, members=record.members, pending_record=record,
                              unpublished_record=lambda: engine.pending_record)
    runner = SimpleNamespace(cfg=cfg, promotion=engine, manifest_store=lambda: store,
                              _rolling_latest_round_id=lambda: "notice")
    ops = LegOps(runner)
    assert ops.promotion_generations() == (0, [], 0)
    assert _PromotionMixin._promotion_notices(runner, block) == {}
    engine.pending_record = None
    # A missing generation object cannot hold unrelated A settlements.
    text = texts.pop(promotion_record_key(1))
    assert _PromotionMixin._promotion_notices(runner, block) == {}
    from cascade.shared.hippius import StorageError

    with pytest.raises(StorageError, match="promotion publication/notice unavailable"):
        ops.promotion_generations()  # ambiguity must not silently select generation zero
    texts[promotion_record_key(1)] = text
    texts[manifest_round_key("notice")] = dump_manifest(sign_manifest(
        TrainingManifest("notice", block, "c", "d", "e"), TRAINER_KP))
    assert ops.promotion_generations() == (0, [], 0)
    notice = sign_manifest(TrainingManifest("notice", block, "c", "d", "e",
        promotion_notices={"1": hashlib.sha256(record.canonical_body()).hexdigest()}), TRAINER_KP)
    texts[manifest_round_key("notice")] = dump_manifest(notice)
    assert ops.promotion_generations() == (1, [("checkpoint", "m")], record.effective_era)


def test_calendar_activation_discards_a_prepared_legacy_era(cfg, tmp_path):
    from cascade.shared.era import era_for_block
    from cascade.trainer.rolling import FinishedLeg

    from .test_rolling import Clock, FakeClient, _armed, _sched

    old = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(old, tmp_path, clock)
    gate = old.round.rolling_from_block + 2 * old.round.epoch_blocks * old.round.era_settlements
    previous = era_for_block(old.round, gate - 1)
    future = era_for_block(old.round, gate)
    sched.state.current = sched._new_era_state(FakeClient(), previous)
    stale = sched._new_era_state(FakeClient(), future)
    sched.state.next = stale
    ops.queue().add("ALFA", "alfa/gen@sha256:" + "1" * 64, reveal_block=1)
    sched.state.finished.append(FinishedLeg("ALFA", 1, "ref", future.index, 1, 1, {}, None, ""))
    new = replace(old, v2=replace(old.v2, from_block=gate),
                   training=replace(old.training, v2_from_block=gate))
    sched.update_config(new)
    assert sched.state.current.index == previous.index and sched.state.next is None
    assert stale._superseded and not sched.state.finished
    assert ops.retired == [future.index]
    assert ops.queue().get("ALFA").operator_billed
    sched._roll_era(FakeClient(), future, gate)
    assert sched.state.current is not stale
    assert sched.state.current.epoch_start == future.index


def test_recovered_promotion_invalidates_an_era_prepared_with_the_old_init(cfg, tmp_path):
    from cascade.shared.era import era_for_block, v2_first_warm_era

    from .test_dec55_trainer_promotion import _v2
    from .test_rolling import Clock, FakeClient, _sched

    cfg, _, length = _v2(cfg)
    sched, ops = _sched(cfg, tmp_path, Clock())
    future = era_for_block(cfg.round, v2_first_warm_era(cfg) * length)
    current = era_for_block(cfg.round, future.start_block - 1)
    sched.state.current = sched._new_era_state(FakeClient(), current)
    stale = sched._new_era_state(FakeClient(), future)
    sched.state.next = stale
    ops.gens = (1, [("new-init", "moe")], future.index)
    sched._sync_generations()
    assert sched.state.current.index == current.index
    assert sched.state.next is None and stale._superseded
    assert ops.retired == [future.index]
    replacement = sched._new_era_state(FakeClient(), future)
    assert replacement.generation == 1 and replacement.warm_start_ckpt == "new-init"
    # Also repair a crash between saving the generation and invalidating the era.
    sched.state.next = stale
    sched._sync_generations()
    assert sched.state.next is None


def test_timed_out_acceptance_preserves_qualifications_and_era_init(tmp_path, monkeypatch):
    from cascade.trainer.lane_b_driver import LaneBDriver, LaneBDriverState

    driver = LaneBDriver.__new__(LaneBDriver)
    driver.cfg = SimpleNamespace(lane_b=SimpleNamespace(acceptance_at_era=lambda _: True),
                                 round=SimpleNamespace(ops_acceptance_timeout_seconds=10))
    driver._lock = threading.RLock()
    driver.st = LaneBDriverState()
    driver.st.pending_merges = [{"fixture": "qualified, unpaid"}]
    driver._acceptance_jobs = {(1, "base"): {"done": threading.Event(), "started": 1}}
    saves = []
    driver.save = lambda: saves.append(json.dumps(driver.st.merged_inits))
    monkeypatch.setattr("cascade.trainer.lane_b_driver.time.monotonic", lambda: 20)
    assert driver.prepare_init_for(2, "base", tmp_path, era_start=20) == ("base", [])
    assert driver.st.pending_merges == [{"fixture": "qualified, unpaid"}]
    assert len(driver._acceptance_jobs) == 1 and saves
    # Completion of the timed-out worker cannot change this era's init.
    job = driver._acceptance_jobs[(1, "base")]
    job["done"].set()
    job["result"] = ("unadopted-merged-checkpoint", [{"fixture": "merge"}])
    assert driver.prepare_init_for(2, "base", tmp_path, era_start=20) == ("base", [])
    assert driver.st.pending_merges == [{"fixture": "qualified, unpaid"}]


def test_lane_c_confirms_only_the_final_holder_and_labels_the_runner_up():
    from cascade.shared.lane_c import LaneCState
    from cascade.validator.lane_c import LaneCJudgement, apply_lane_c

    from .test_marketcast_validator import MC, _entry

    def judgement(hotkey, lcb, phase):
        return LaneCJudgement(_entry(hotkey), True, "verdict", "pass", gain_lcb=lcb,
                              gain=0.1, base_pointer=MC.base_pointer, phase=phase,
                              full_pay={"tier": 0.05, "full_eras": 10})

    rows = [judgement("carol", 0.06, "confirm"), judgement("alice", 0.08, "confirm")]
    state = LaneCState()
    wins = apply_lane_c(state, rows, round_id="r3", settle_block=100)
    assert [w.hotkey for w in wins] == ["alice"]
    assert [r.outcome for r in rows] == ["below holder LCB", "confirmed"]
    assert state.slots["mkt:returns"].hotkey == "alice"
    rows = [judgement("alice", 0.08, "judge"), judgement("carol", 0.06, "judge")]
    apply_lane_c(LaneCState(), rows, round_id="r2", settle_block=90)
    assert [r.outcome for r in rows] == ["pending", "not_selected"]


def test_marketcast_audit_refuses_two_confirmed_rows_on_one_target(cfg, tmp_path):
    import copy

    from cascade.audit.checks import check_lane_c

    from .test_marketcast_validator import BLOCK, EB, _rc, _three_rounds

    flow = _three_rounds(cfg, tmp_path)
    rows = list(flow.rows[2])
    winner = next(row for row in rows if row.get("outcome") == "confirmed")
    forged = copy.deepcopy(winner)
    forged["hotkey"] = "different-miner"
    rows.append(forged)
    check = check_lane_c(_rc("1003", BLOCK + 2 * EB, rows), flow.r.cfg)
    assert check.status == "FAIL" and "two confirmed rows" in check.detail
