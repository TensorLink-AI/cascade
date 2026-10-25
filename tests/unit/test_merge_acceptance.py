"""Qualification is unpaid; only compatible, independently verified merges earn."""

from __future__ import annotations

import copy
import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.shared.lane_b import DomainWin, LaneBState, lane_b_payouts  # noqa: E402
from cascade.shared.manifest import (  # noqa: E402
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    load_manifest,
)
from cascade.trainer.ckpt_digest import save_tensors_hashed  # noqa: E402
from cascade.trainer.expert_lane import _moe_names, settle_leg  # noqa: E402
from cascade.trainer.lane_b_driver import LaneBDriver  # noqa: E402
from cascade.validator.era_gate import EraGateMixin  # noqa: E402
from cascade.validator.expert_lane import apply_judgements  # noqa: E402
from cascade.validator.lane_b import LaneBMixin, _LaneBRound  # noqa: E402
from cascade.validator.merge_acceptance import judge_acceptance, verify_acceptance  # noqa: E402

from .test_lane_b_driver import GEN, Ops  # noqa: E402
from .test_lane_b_merge_state import LC  # noqa: E402
from .test_moe_trainer import _train  # noqa: E402
from .test_validator_expert_lane import _scores, _ws  # noqa: E402

LB = parse_lane_b({"from_block": 1, "domains": ["energy", "web"],
                   "acceptance_from_block": 1, "bootstrap_B": 300,
                   "min_target_windows": 20, "min_guard_windows": 20})


def _scores_for(energy=1.0, web=1.0):
    return _ws(_scores({"energy": energy, "web": web}), True)


class MergeOps(Ops):
    def __init__(self, tmp, king):
        super().__init__(tmp, {})
        self.base = load_file(str(king / "weights.safetensors"))
        self.names = _moe_names(king)
        self.fail_scoring = False

    def merge_scores(self, pointer, block):
        if self.fail_scoring:
            raise OSError("GPU unavailable")
        state = load_file(str(self.fetch(pointer) / "weights.safetensors"))
        n = self.names[0][0]
        edited = {r for r in (1, 2, 3) if not torch.equal(state[n][r], self.base[n][r])}
        # Each delta works alone. Rows 1+2 conflict; rows 1+3 are compatible.
        return _scores_for(1.05 if {1, 2} <= edited else (0.9 if 1 in edited else 1.0),
                           0.9 if edited & {2, 3} else 1.0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("acceptance-king")
    _train(tmp / "king", LC, budget=600)
    return tmp / "king"


def _setup(tmp_path, checkpoint):
    import shutil

    ops = MergeOps(tmp_path, checkpoint)
    king = ops._ptr(checkpoint)
    drv = LaneBDriver(SimpleNamespace(lane_b=LB), ops, tmp_path / "driver.json")
    ops.cfg = drv.cfg
    qualified = []
    for row, domain in ((1, "energy"), (2, "web"), (3, "web")):
        directory = tmp_path / f"leg-{row}"
        shutil.copytree(checkpoint, directory)
        state = {n: t.clone() for n, t in ops.base.items()}
        state[ops.names[0][0]][row] += 0.01
        save_tensors_hashed(state, directory / "weights.safetensors")
        rebased = settle_leg(checkpoint, checkpoint, directory, {0: [row]},
                             tmp_path / f"rebased-{row}", refuse_nonfinite=True)
        pointer = ops._ptr(rebased)
        win = DomainWin(domain, f"miner-{row}", row, str(10 + row), 10 + row,
                        GEN, 0.1, pointer, king, {"0": [row]},
                        tier=0.05, full_eras=10, commit_block=row,
                        targets=(domain,), qualified_block=10 + row)
        qualified.append(win)
        drv.st.pending_merges.append({"rebased_pointer": pointer, "king_pointer": king,
                                      "selection": {"0": [row]}, "round_id": win.round_id,
                                      "qualified_wins": [dataclasses.asdict(win)]})
    return drv, ops, king, qualified


class Runner(EraGateMixin, LaneBMixin):
    def __init__(self, cfg, ops, qualifications, path):
        self.ops = ops
        self.cfg = SimpleNamespace(
            lane_b=LB, round=dataclasses.replace(cfg.round, epoch_blocks=10,
                epoch_blocks_prev=0, epoch_activation_block=0, era_settlements=1),
            eval=dataclasses.replace(cfg.eval, scored_horizons=(), n_windows=120),
            training=SimpleNamespace(at_block=lambda b: SimpleNamespace(primary_size=LC)))
        self._lane_b_state = LaneBState(qualifications=list(qualifications))
        self._merged_init_ok = set()
        self._lane_b_window_source = SimpleNamespace(windows_for_round=lambda *a, **kw: [])
        self.path = path

    def _settlement_cfg(self, block):
        return nullcontext()

    def _fetch_checkpoint_dir(self, entry):
        return self.ops.fetch(entry.trained_pointer)

    def _evaluate(self, entry, windows, contract):
        return self.ops.merge_scores(entry.trained_pointer, 20)

    def _lane_b_state_path(self):
        return self.path


def _manifest(pointer, king, merges):
    return TrainingManifest(
        round_id="30", created_block=30, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", GEN, king, "3" * 64, 21)],
        warm_start_ckpt=pointer, warm_start_base=king, warm_start_merges=tuple(merges),
        era={"index": 2, "start_block": 20, "seed_block": 10})


def test_incremental_merge_rejects_conflict_and_accepts_later_candidate(tmp_path, checkpoint, cfg):
    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    # Deliberately scrambled pending order must not affect the result.
    drv.st.pending_merges.reverse()
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    assert [m["selection"] for m in merges] == [{"0": [1]}, {"0": [3]}]
    assert drv.st.pending_merges == []
    state = load_file(str(ops.fetch(pointer) / "weights.safetensors"))
    n = ops.names[0][0]
    assert torch.equal(state[n][2], ops.base[n][2])  # rejected rows never enter main
    for row in (1, 3):
        assert not torch.equal(state[n][row], ops.base[n][row])
    assert drv.init_for(2, king, tmp_path / "work", era_start=20) == (pointer, merges)
    runner = Runner(cfg, ops, qualified, tmp_path / "lane-b.json")
    manifest = _manifest(pointer, king, merges)
    assert load_manifest(dump_manifest(manifest)).canonical_body() == manifest.canonical_body()
    assert runner._verify_merged_init(manifest, strict=True) is None
    assert runner._lane_b().wins == {}  # a read-only gate cannot start rewards
    acceptances = verify_acceptance(runner, manifest)
    runner._lane_b_commit([_LaneBRound("30", 30, [], [], acceptances=acceptances)])
    st = runner._lane_b()
    assert {w.hotkey for w in st.wins.values()} == {"miner-1", "miner-3"}
    assert {w.merge_block for w in st.wins.values()} == {20}
    assert {w.round_id for w in st.wins.values()} == {"11", "13"}
    assert lane_b_payouts(st, block=19, era_blocks=10, cfg=LB, n_uids=10) == []
    assert all(w["age"] == 0 for w in lane_b_payouts(
        st, block=20, era_blocks=10, cfg=LB, n_uids=10))
    # Repeated seed-epoch manifests, restarts and out-of-order folding cannot
    # restart the reward/decay clock or apply the delta twice.
    restored = LaneBState.from_json(st.to_json())
    assert restored == st
    for merge, wins, block in acceptances:
        restored.accept(merge, wins, block + 10)
    assert {w.merge_block for w in restored.wins.values()} == {20}


def test_transient_acceptance_failure_keeps_all_pending(tmp_path, checkpoint):
    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    original = copy.deepcopy(drv.st.pending_merges)
    ops.fail_scoring = True
    with pytest.raises(OSError, match="GPU unavailable"):
        drv.init_for(2, king, tmp_path / "work", era_start=20)
    assert drv.st.pending_merges == original and drv.st.merged_inits == {}


def test_qualified_leg_is_unpaid_and_does_not_replace_holder(tmp_path, checkpoint):
    from cascade.validator.expert_lane import judge_manifest

    from .test_validator_expert_lane import _manifest as expert_manifest

    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    m = expert_manifest({0: [1]})
    entry = dataclasses.replace(m.entries[0], base_king_pointer=king,
                                rebased_pointer=drv.st.pending_merges[0]["rebased_pointer"])
    m = dataclasses.replace(m, king_pointer=king, entries=(entry,))
    js = judge_manifest(m, king_pointer=king, king_scores=_scores_for(),
                        evaluate=lambda p: _scores_for(.9), fetch_dir=ops.fetch,
                        cfg=LB, seed="test", strict=True)
    holder = DomainWin("energy", "old-holder", 9, "old", 1, full_eras=2, tier=.01)
    st = LaneBState(wins={"energy": holder})
    wins = apply_judgements(st, js, round_id="11", settle_block=11, qualify_only=True)
    assert wins and st.qualifications == wins and st.wins == {"energy": holder}
    assert {w["hotkey"] for w in lane_b_payouts(
        st, block=19, era_blocks=10, cfg=LB, n_uids=10)} == {"old-holder"}


def test_original_anchor_blocks_accumulated_small_regressions():
    qualified = [DomainWin("web", "miner", 1, "10", 10, targets=("web",))]
    wins, reason = judge_acceptance(
        qualified, anchor_scores=_scores_for(), before_scores=_scores_for(1.008),
        candidate_scores=_scores_for(1.016, .9), public_rows={}, cfg=LB, seed="anchor")
    assert not wins and "regression" in reason["reason"]


@pytest.mark.parametrize("tamper", ["missing", "before", "candidate", "tier", "order"])
def test_validator_refuses_forged_acceptance(tmp_path, checkpoint, cfg, tamper):
    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    forged = copy.deepcopy(merges)
    if tamper == "missing":
        forged[0].pop("acceptance")
    elif tamper == "before":
        forged[1]["acceptance"]["before"] = king
    elif tamper == "candidate":
        forged[1]["acceptance"]["candidate"] = king
    elif tamper == "tier":
        forged[0]["acceptance"]["wins"][0]["full_eras"] = 999
    else:
        forged.reverse()
    runner = Runner(cfg, ops, qualified, tmp_path / "state.json")
    reason = runner._verify_merged_init(_manifest(pointer, king, forged), strict=True)
    assert reason and "era_init_mismatch" in reason
    assert not runner._lane_b().wins


def test_settlement_only_qualifies_and_persists_pending_evidence(tmp_path, checkpoint, cfg):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import FinishedExpertLeg

    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    leg_pointer = drv.st.pending_merges[0]["rebased_pointer"]
    drv.st.pending_merges = []
    ops.effect = {"energy": .9}
    leg = OpenLeg(LaneBSubmission("alice", 7, GEN, "energy", 5), {0: [1]}, king, 1)
    drv.st.finished = [dataclasses.asdict(FinishedExpertLeg(
        leg=dataclasses.asdict(leg), leg_pointer=leg_pointer, corpus_digest="c", train_block=19))]
    drv._routing_stale = lambda *a: (False, {})
    js = drv.settle(round_id="21", block=21, era_index=1,
                    king_pointer=king, work_dir=tmp_path / "settle")
    assert js and js[0].passed
    assert drv.st.pending_merges[0]["qualified_wins"][0]["qualified_block"] == 21
    runner = Runner(cfg, ops, [], tmp_path / "state.json")
    runner._lane_b_commit([_LaneBRound("21", 21, [], js)])
    assert not runner._lane_b().wins and runner._lane_b().qualifications


def test_missing_acceptance_and_failed_combination_never_pay(tmp_path, checkpoint, cfg):
    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    assert any(o["outcome"] == "acceptance_rejected" and o["hotkey"] == "miner-2"
               for o in drv.st.outcomes)
    runner = Runner(cfg, ops, qualified, tmp_path / "state.json")
    assert runner._verify_merged_init(_manifest(pointer, king, merges), strict=True) is None
    # A formerly verified pointer cannot bypass a changed merge declaration.
    forged = copy.deepcopy(merges)
    forged[0].pop("acceptance")
    assert runner._verify_merged_init(_manifest(pointer, king, forged), strict=True)
    assert not runner._lane_b().wins


def test_required_public_guard_cannot_pass_with_no_usable_items(tmp_path, checkpoint):
    _, _, _, qualified = _setup(tmp_path, checkpoint)
    required = dataclasses.replace(LB, require_public=("gift",), min_public_items=10)
    empty = {"gift": {"status": "ok", "rows": []}}
    wins, _ = judge_acceptance(
        qualified[:1], anchor_scores=_scores_for(), before_scores=_scores_for(),
        candidate_scores=_scores_for(.9, 1.),
        public_rows={"anchor": empty, "before": empty, "candidate": empty},
        cfg=required, seed="empty-required")
    assert wins == []


def test_incomplete_acceptance_benchmarks_preserve_qualified_work(tmp_path, checkpoint):
    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    drv.cfg = SimpleNamespace(lane_b=dataclasses.replace(LB, require_public=("gift",)))
    ops.bench_rows = lambda directory: {"gift": {"status": "error", "rows": []}}
    before = copy.deepcopy(drv.st.pending_merges)
    with pytest.raises(OSError, match="public"):
        drv.init_for(2, king, tmp_path / "work", era_start=20)
    assert drv.st.pending_merges == before
    assert drv.st.merged_inits == {} and drv.st.outcomes == []


def test_no_two_accepted_miners_replace_same_slot_in_one_batch(tmp_path, checkpoint):
    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    # Remove the conflicting candidate, then make both remaining legs target
    # energy: the later one must wait rather than merge without any paid time.
    drv.st.pending_merges.pop(1)
    later = drv.st.pending_merges[1]
    later["qualified_wins"][0]["domain"] = "energy"
    later["qualified_wins"][0]["targets"] = ("energy",)
    _, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    assert len(merges) == 1 and len(drv.st.pending_merges) == 1


def test_audit_binds_payment_to_accepted_marginal_gain(tmp_path, checkpoint, cfg):
    from cascade.audit.checks import check_lane_b

    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    runner = Runner(cfg, ops, qualified, tmp_path / "state.json")
    manifest = _manifest(pointer, king, merges)
    for merge, wins, block in verify_acceptance(runner, manifest):
        runner._lane_b().accept(merge, wins, block)
    import json

    receipt = SimpleNamespace(
        epoch_start_block=30, manifest=json.loads(manifest.canonical_body()), lane_b=[],
        lane_b_pay={"wins": lane_b_payouts(
            runner._lane_b(), block=30, era_blocks=10, cfg=LB, n_uids=10)})
    assert check_lane_b(receipt, runner.cfg).status == "PASS"
    receipt.lane_b_pay["wins"][0]["gain_lcb"] = .99
    assert check_lane_b(receipt, runner.cfg).status == "FAIL"
    receipt.lane_b_pay["wins"][0]["merge_block"] = 11
    assert check_lane_b(receipt, runner.cfg).status == "FAIL"


def test_acceptance_activation_keeps_an_existing_seed_epoch_init(cfg):
    from cascade.validator.merge_acceptance import acceptance_block

    cfg = dataclasses.replace(cfg, round=dataclasses.replace(
        cfg.round, epoch_blocks=10, epoch_blocks_prev=0, epoch_activation_block=0,
        era_settlements=1))
    manifest = TrainingManifest("30", 30, "a", "b", "c",
        era={"index": 2, "start_block": 20, "seed_block": 0})
    assert acceptance_block(cfg, manifest) == 10
    lb = parse_lane_b({"acceptance_from_block": 20})
    assert not lb.acceptance_at_era(acceptance_block(cfg, manifest))
    assert not lb.acceptance_at_settlement(19)
    assert lb.acceptance_at_settlement(20) and lb.acceptance_at_settlement(21)


@pytest.mark.parametrize("value", [-1, True, "soon", 1.5])
def test_invalid_acceptance_activation_refused(value):
    with pytest.raises(ValueError, match="acceptance_from_block"):
        parse_lane_b({"acceptance_from_block": value})


def test_acceptance_source_can_be_recovered_without_starting_rewards(tmp_path, checkpoint, cfg):
    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    runner = Runner(cfg, ops, [], tmp_path / "state.json")
    calls = []
    runner._epoch_start_block = lambda manifest: int(manifest.round_id)
    runner._lane_b_load_settlement = lambda rid: TrainingManifest(
        rid, int(rid), "a", "b", "c", era={"index": 1, "start_block": 10, "seed_block": 0})

    def recover(rid):
        calls.append(rid)
        for win in qualified:
            if win.round_id == rid:
                runner._lane_b().qualify(win)

    runner._lane_b_fold_on_demand = recover
    assert runner._verify_merged_init(_manifest(pointer, king, merges), strict=True) is None
    assert calls == ["11", "13"] and not runner._lane_b().wins


def test_repeated_seed_epoch_manifest_keeps_acceptance_block(tmp_path, checkpoint, cfg):
    drv, ops, king, qualified = _setup(tmp_path, checkpoint)
    pointer, merges = drv.init_for(2, king, tmp_path / "work", era_start=20)
    runner = Runner(cfg, ops, qualified, tmp_path / "state.json")
    manifest = _manifest(pointer, king, merges)
    first = verify_acceptance(runner, manifest)
    for merge, wins, block in first:
        runner._lane_b().accept(merge, wins, block)
    later = dataclasses.replace(manifest, round_id="40", created_block=40,
        era={"index": 3, "start_block": 30, "seed_block": 10})
    assert {block for _, _, block in verify_acceptance(runner, later)} == {20}
    assert {w.merge_block for w in runner._lane_b().wins.values()} == {20}


def test_all_rejected_candidates_leave_main_checkpoint_unchanged(tmp_path, checkpoint):
    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    ops.merge_scores = lambda *args: _scores_for()  # every marginal gain is zero
    assert drv.init_for(2, king, tmp_path / "work", era_start=20) == (king, [])
    assert not drv.st.pending_merges
    assert len(drv.st.outcomes) == 3
    assert all(o["outcome"] == "acceptance_rejected" for o in drv.st.outcomes)


def test_status_does_not_report_accepted_contributions_as_dropped(tmp_path, checkpoint):
    from cascade.shared.v2_status import lane_b_view

    drv, _, king, _ = _setup(tmp_path, checkpoint)
    drv.init_for(2, king, tmp_path / "work", era_start=20)
    view = lane_b_view(LB, dataclasses.asdict(drv.st), block=20)
    assert not view["per_domain"]["energy"]["dropped"]
    assert [o["outcome"] for o in view["per_domain"]["web"]["dropped"]] == ["acceptance_rejected"]


def test_acceptance_is_nonblocking_and_staging_does_not_consume_queue(tmp_path, checkpoint):
    import threading

    from cascade.trainer.merge_acceptance import MergeAcceptancePending
    from cascade.trainer.rolling import RollingScheduler

    drv, ops, king, _ = _setup(tmp_path, checkpoint)
    drv.save()
    entered, release = threading.Event(), threading.Event()
    real = ops.merge_scores

    def slow(*args):
        entered.set()
        assert release.wait(10)
        return real(*args)

    ops.merge_scores = slow
    try:
        with pytest.raises(MergeAcceptancePending):
            drv.prepare_init_for(2, king, tmp_path / "work", era_start=20)
        assert entered.wait(5)
        assert len(drv.st.pending_merges) == 3 and not drv.st.merged_inits
        assert RollingScheduler._chain_step(
            SimpleNamespace(), "roll", lambda: drv.prepare_init_for(
                2, king, tmp_path / "work", era_start=20)) is False
    finally:
        release.set()
    assert drv._acceptance_jobs[(2, king)]["done"].wait(10)
    assert len(drv.st.pending_merges) == 3 and not drv.st.merged_inits
    restarted = LaneBDriver(drv.cfg, ops, drv.path)
    assert len(restarted.st.pending_merges) == 3
    pointer, merges = drv.prepare_init_for(2, king, tmp_path / "work", era_start=20)
    assert pointer != king and len(merges) == 2 and not drv.st.pending_merges
    assert drv.prepare_init_for(2, king, tmp_path / "work", era_start=20) == (pointer, merges)


def test_obsolete_background_job_does_not_spend_pending_deltas(tmp_path, checkpoint):
    from cascade.trainer.merge_acceptance import MergeAcceptancePending

    drv, _, king, _ = _setup(tmp_path, checkpoint)
    for era, block in ((2, 20), (3, 30)):
        with pytest.raises(MergeAcceptancePending):
            drv.prepare_init_for(era, king, tmp_path / "work", era_start=block)
        assert drv._acceptance_jobs[(era, king)]["done"].wait(10)
        assert len(drv.st.pending_merges) == 3 and not drv.st.merged_inits
    _, merges = drv.prepare_init_for(3, king, tmp_path / "work", era_start=30)
    assert {m["acceptance"]["block"] for m in merges} == {30}
    assert "2" not in drv.st.merged_inits


def test_failed_adoption_save_leaves_queue_and_staged_result_retryable(tmp_path, checkpoint, monkeypatch):
    from cascade.trainer.merge_acceptance import MergeAcceptancePending

    drv, _, king, _ = _setup(tmp_path, checkpoint)
    with pytest.raises(MergeAcceptancePending):
        drv.prepare_init_for(2, king, tmp_path / "work", era_start=20)
    assert drv._acceptance_jobs[(2, king)]["done"].wait(10)
    real = drv.save

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(drv, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        drv.prepare_init_for(2, king, tmp_path / "work", era_start=20)
    assert len(drv.st.pending_merges) == 3 and not drv.st.merged_inits
    monkeypatch.setattr(drv, "save", real)
    _, merges = drv.prepare_init_for(2, king, tmp_path / "work", era_start=20)
    assert len(merges) == 2 and not drv.st.pending_merges


def test_scheduler_does_not_start_plain_init_while_acceptance_is_running(cfg, tmp_path, monkeypatch):
    from cascade.shared.era import era_for_block
    from cascade.trainer import rolling
    from cascade.trainer.merge_acceptance import MergeAcceptancePending

    from .test_rolling import Clock, FakeClient, _armed, _ptr, _sched

    armed = dataclasses.replace(_armed(cfg), lane_b=LB)
    sched, _ = _sched(armed, tmp_path, Clock())
    king = _ptr("member")
    monkeypatch.setattr(rolling, "era_init", lambda *a: (1, 0, king, "size"))

    class Waiting:
        def init_for(self, *a, **kw):
            raise AssertionError("must use nonblocking prepare")

        def prepare_init_for(self, *a, **kw):
            raise MergeAcceptancePending("benchmark running")

    sched.lane_b = Waiting()
    era = era_for_block(armed.round, armed.round.rolling_from_block)
    with pytest.raises(MergeAcceptancePending):
        sched._new_era_state(FakeClient(), era)
    assert sched.state.current is None and sched.state.next is None
