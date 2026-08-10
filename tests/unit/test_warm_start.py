"""Cascade warm-start consumption + envelope gate (DEC-CA-0005/0004/0012).

The trainer's promotion engine writes the pointer file (the live generation's
member set); the trainer reads it, trains every matching-size run this round
from the epoch-rotation member, and stamps the pin onto the signed manifest;
each validator then VERIFIES the declaration — any live member passes, an
unseen init must be justified by a trainer-signed PromotionRecord that survives
the envelope (provenance + quality floor + ripeness + set cap) — instead of
re-deriving the selection.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
)
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from cascade.shared.promotion import (
    PromotedMember,
    PromotionRecord,
    dump_promotion_record,
    promotion_index_key,
    promotion_record_key,
)
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost, worker_argv
from cascade.validator.cascade import CascadeController, CascadeState
from cascade.validator.loop import ValidatorRunner

REF = "alice/metro-gen@sha256:" + "a" * 64
REF_T = "cascade/ckpt-r1-king-toto2-4m@sha256:" + "b" * 64
REF_T2 = "cascade/ckpt-r2-chal-toto2-4m@sha256:" + "c" * 64
PTR = format_trained_pointer(REF_T)
PTR2 = format_trained_pointer(REF_T2)

DAY = 7_200  # blocks; the controller runs with round_cfg=None in these tests


# ── trainer: reading the promoted-init pointer ───────────────────────────────


def _trainer(cfg, tmp_path, ws_path):
    return TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                         warm_start_path=ws_path)


def test_no_pointer_path_means_random_init(cfg, tmp_path):
    assert _trainer(cfg, tmp_path, None)._load_warm_start() is None


def test_absent_pointer_file_means_random_init(cfg, tmp_path):
    assert _trainer(cfg, tmp_path, tmp_path / "nope.json")._load_warm_start() is None


def test_legacy_single_pointer_yields_ref_and_size(cfg, tmp_path):
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({"checkpoint_id": PTR, "size": "toto2-22m"}), encoding="utf-8")
    assert _trainer(cfg, tmp_path, p)._load_warm_start() == (PTR, "toto2-22m")


def test_pointer_file_without_size_defaults_to_primary(cfg, tmp_path):
    # Pointer files written before the size field existed default to the
    # primary arch preset (what the benchmark sidecar scores).
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({"checkpoint_id": PTR}), encoding="utf-8")
    ref, size = _trainer(cfg, tmp_path, p)._load_warm_start()
    assert ref == PTR and size == cfg.training.primary_size.arch_preset


def test_member_set_rotates_by_epoch_index(cfg, tmp_path):
    # DEC-CA-0012: a multi-member pointer file rotates deterministically —
    # epoch N trains from members[N % len(members)].
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({
        "generation": 2,
        "members": [
            {"checkpoint_id": PTR, "size": "toto2-4m"},
            {"checkpoint_id": PTR2, "size": "toto2-4m"},
        ],
        "checkpoint_id": PTR, "size": "toto2-4m",   # legacy mirror
    }), encoding="utf-8")
    t = _trainer(cfg, tmp_path, p)
    assert t._load_warm_start(epoch_index=0) == (PTR, "toto2-4m")
    assert t._load_warm_start(epoch_index=1) == (PTR2, "toto2-4m")
    assert t._load_warm_start(epoch_index=2) == (PTR, "toto2-4m")
    assert t._load_warm_start() == (PTR, "toto2-4m")  # no index ⇒ first member


def test_engine_wins_over_pointer_file(cfg, tmp_path):
    # With the promotion engine wired, it is the single source of the
    # allocation policy — the pointer file (which the engine itself writes) is
    # only the fallback for engine-less runs.
    from cascade.shared.promotion import PromotedMember
    from cascade.trainer.promotion import TrainerPromotion

    eng = TrainerPromotion(reign_threshold=5, k_max=2, quality_epsilon=0.05)
    eng.members = (PromotedMember(PTR, "toto2-4m", "r1", 1.0),
                   PromotedMember(PTR2, "", "r2", 1.01))
    t = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                      warm_start_path=None, promotion=eng)
    assert t._load_warm_start(epoch_index=0) == (PTR, "toto2-4m")
    # An empty member size falls back to the primary preset.
    assert t._load_warm_start(epoch_index=1) == (
        PTR2, cfg.training.primary_size.arch_preset)
    # Random-init era: engine has no members.
    eng.members = ()
    assert t._load_warm_start(epoch_index=0) is None


def test_broken_pointer_file_raises_never_falls_back(cfg, tmp_path):
    # DEC-CA-0005: once a promotion is live, a round must never silently train
    # from random init — a live-but-unusable pointer aborts the round.
    p = tmp_path / "ws.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _trainer(cfg, tmp_path, p)._load_warm_start()
    p.write_text(json.dumps({"checkpoint_id": "not-a-pointer"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _trainer(cfg, tmp_path, p)._load_warm_start()
    p.write_text(json.dumps({"score": 0.5}), encoding="utf-8")  # no checkpoint_id
    with pytest.raises(RuntimeError):
        _trainer(cfg, tmp_path, p)._load_warm_start()
    p.write_text(json.dumps({"members": [{"size": "toto2-4m"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _trainer(cfg, tmp_path, p)._load_warm_start()


# ── validator: the envelope gate ─────────────────────────────────────────────


class _Store:
    """Duck-typed manifest-bucket store: get_text raises on a missing key."""

    def __init__(self, texts: dict | None = None):
        self.texts = dict(texts or {})

    def get_text(self, key: str) -> str:
        return self.texts[key]


def _entry(role, uid, ptr=PTR):
    return TrainedEntry(f"hk{uid}", uid, role, REF, ptr, "d", 10)


def _manifest(cfg, *, warm_start_ckpt="", created_block=10):
    return TrainingManifest(
        round_id="1", created_block=created_block,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=[_entry("king", 0), _entry("challenger", 1)],
        warm_start_ckpt=warm_start_ckpt,
        warm_start_size="toto2-4m" if warm_start_ckpt else "",
    )


def _validator(cfg, tmp_path, *, cascade: CascadeController | None,
               store: _Store | None = None):
    ws = tmp_path / "warm_start_init.json"
    cfg = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    return ValidatorRunner(cfg=cfg, verify_signatures=False, cascade=cascade,
                           bench_report_store=store or _Store())


def _scores(v: float) -> BenchScores:
    return BenchScores(gifteval_crps=v, gifteval_mase=v, boom_crps=v,
                       boom_mase=v, time_crps=v, time_mase=v)


def _bench_report_text(round_id: str, scored: dict[str, float]) -> str:
    entries = tuple(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="hk", miner_uid=0,
                   trained_pointer=ptr, scores=_scores(v))
        for ptr, v in scored.items()
    )
    return dump_bench_report(BenchReport(round_id=round_id, created_block=10,
                                         entries=entries))


def _promotion_store(members: list[tuple[str, float]], *, generation=1,
                     bench_round="r1") -> _Store:
    """A store holding a promotion record + the bench report that scores its
    members (the on-demand provenance/quality source)."""
    record = PromotionRecord(
        generation=generation, king_hotkey="hk0", fired_round=bench_round,
        fired_block=0,
        members=tuple(PromotedMember(checkpoint_id=p, size="toto2-4m",
                                     source_round=bench_round, score=v)
                      for p, v in members),
    )
    return _Store({
        promotion_index_key(): json.dumps({"latest_generation": generation}),
        promotion_record_key(generation): dump_promotion_record(record),
        bench_report_key(bench_round): _bench_report_text(
            bench_round, dict(members)),
    })


def test_gate_off_when_cascade_disabled(cfg, tmp_path):
    # Pure KOTH ignores the field entirely (even a pinned manifest passes).
    r = _validator(cfg, tmp_path, cascade=None)
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None


def test_random_init_passes_in_generation_zero(cfg, tmp_path):
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5))
    assert r.check_manifest(_manifest(cfg)) is None


def test_random_init_rejected_once_promotion_is_live(cfg, tmp_path):
    # The heart of DEC-CA-0005: a trainer that silently fell back to random
    # init after a promotion must be rejected, not scored.
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, generation=1, members=(PTR,)))
    r = _validator(cfg, tmp_path, cascade=ctl)
    reason = r.check_manifest(_manifest(cfg))
    assert reason is not None and "warm_start_missing" in reason


def test_live_member_passes_without_any_fetch(cfg, tmp_path):
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, generation=1, members=(PTR, PTR2)))
    r = _validator(cfg, tmp_path, cascade=ctl)
    # ANY live member passes — rotation/allocation across members is trainer
    # policy, not consensus.
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR2)) is None


def test_unseen_pin_without_record_rejected(cfg, tmp_path):
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5))
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR))
    assert reason is not None and "warm_start_mismatch" in reason


def test_verified_promotion_record_is_adopted(cfg, tmp_path):
    store = _promotion_store([(PTR, 1.0), (PTR2, 1.02)], generation=1)
    ctl = CascadeController(reign_days=5)
    r = _validator(cfg, tmp_path, cascade=ctl, store=store)
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None
    # Accepted: generation + member set installed, clock re-crowned at the
    # manifest's epoch block.
    assert ctl.state.generation == 1
    assert ctl.state.members == (PTR, PTR2)
    # Subsequent rounds pass on the fast path (no fetch).
    r2 = _validator(cfg, tmp_path, cascade=ctl)
    assert r2.check_manifest(_manifest(cfg, warm_start_ckpt=PTR2)) is None


def test_stale_record_generation_rejected(cfg, tmp_path):
    # A replayed old record can never roll the live set back.
    store = _promotion_store([(PTR2, 1.0)], generation=1)
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, generation=1, members=(PTR,)))
    r = _validator(cfg, tmp_path, cascade=ctl, store=store)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR2))
    assert reason is not None and "stale or replayed" in reason


def test_pin_not_in_record_rejected(cfg, tmp_path):
    store = _promotion_store([(PTR, 1.0)], generation=1)
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5), store=store)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR2))
    assert reason is not None and "not a member" in reason


def test_member_set_over_top_k_rejected(cfg, tmp_path):
    # cfg ships cascade_top_k = 3; a 4-member record breaks the envelope.
    members = [(format_trained_pointer(f"cascade/ckpt-{i}@sha256:" + "d" * 64), 1.0)
               for i in range(4)]
    store = _promotion_store(members, generation=1)
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5), store=store)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=members[0][0]))
    assert reason is not None and "warm_start_promotion_invalid" in reason


def test_member_above_quality_floor_rejected(cfg, tmp_path):
    # cfg ships cascade_quality_epsilon = 0.05: a member 20% off the best is
    # outside the envelope — diversity can't be bought with a worse init.
    store = _promotion_store([(PTR, 1.0), (PTR2, 1.2)], generation=1)
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5), store=store)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR))
    assert reason is not None and "warm_start_member_below_floor" in reason


def test_member_without_bench_numbers_rejected(cfg, tmp_path):
    # Provenance fails CLOSED: an init nobody can score must not become the
    # field's floor.
    record = PromotionRecord(
        generation=1, king_hotkey="hk0", fired_round="r1", fired_block=0,
        members=(PromotedMember(checkpoint_id=PTR, size="toto2-4m",
                                source_round="r1", score=1.0),),
    )
    store = _Store({
        promotion_index_key(): json.dumps({"latest_generation": 1}),
        promotion_record_key(1): dump_promotion_record(record),
        # no bench report for r1
    })
    r = _validator(cfg, tmp_path, cascade=CascadeController(reign_days=5), store=store)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR))
    assert reason is not None and "warm_start_member_unverifiable" in reason


def test_premature_promotion_rejected_when_clock_can_attest(cfg, tmp_path):
    # An anchored, generation≥1 validator enforces ripeness on the +1
    # transition: reign 5 rounds, promotion declared 2 rounds in → early.
    store = _promotion_store([(PTR2, 1.0)], generation=2)
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=0, generation=1, members=(PTR,)))
    r = _validator(cfg, tmp_path, cascade=ctl, store=store)
    reason = r.check_manifest(
        _manifest(cfg, warm_start_ckpt=PTR2, created_block=2 * DAY + 10))
    assert reason is not None and "warm_start_promotion_early" in reason
    # …and accepted once the reign clock ripened.
    assert r.check_manifest(
        _manifest(cfg, warm_start_ckpt=PTR2, created_block=5 * DAY + 10)) is None
    assert ctl.state.generation == 2 and ctl.state.members == (PTR2,)


def test_out_of_reign_member_rejected_when_clock_can_attest(cfg, tmp_path):
    # Provenance is reign-scoped for an attesting validator: a member whose
    # signed bench report predates the current reign's anchor (stale trainer
    # state, fabricated source_round) fails, even though the numbers verify.
    store = _promotion_store([(PTR2, 1.0)], generation=2)  # report created_block=10
    ctl = CascadeController(reign_days=5, state=CascadeState(
        king_hotkey="hk0", reign_start_block=5 * DAY, generation=1, members=(PTR,)))
    r = _validator(cfg, tmp_path, cascade=ctl, store=store)
    reason = r.check_manifest(
        _manifest(cfg, warm_start_ckpt=PTR2, created_block=10 * DAY + 10))
    assert reason is not None and "warm_start_member_out_of_reign" in reason


def test_bootstrap_validator_skips_unverifiable_ripeness(cfg, tmp_path):
    # A fresh validator (generation 0, clock measuring only its own uptime)
    # cannot distinguish "promotion too early" from "my clock started late" —
    # it verifies provenance/quality/signature and adopts.
    store = _promotion_store([(PTR, 1.0)], generation=3)
    ctl = CascadeController(reign_days=5)
    r = _validator(cfg, tmp_path, cascade=ctl, store=store)
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None
    assert ctl.state.generation == 3


# ── migration: pre-DEC-CA-0012 pointer files ─────────────────────────────────


def test_legacy_single_pointer_file_is_grandfathered(cfg, tmp_path):
    # An upgraded validator whose old Cascade installed a winner keeps
    # accepting it as generation 1 — no round of warm_start_mismatch at deploy.
    ws = tmp_path / "warm_start_init.json"
    ws.write_text(json.dumps({"checkpoint_id": PTR, "size": "toto2-4m"}), encoding="utf-8")
    ctl = CascadeController(reign_days=5)
    cfg2 = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    r = ValidatorRunner(cfg=cfg2, verify_signatures=False, cascade=ctl,
                        bench_report_store=_Store())
    assert r.check_manifest(_manifest(cfg2, warm_start_ckpt=PTR)) is None
    assert ctl.state.generation == 1 and ctl.state.members == (PTR,)


def test_trainer_written_member_file_is_adopted(cfg, tmp_path):
    # On the owner box the pointer file is the trainer engine's member set.
    ws = tmp_path / "warm_start_init.json"
    ws.write_text(json.dumps({
        "generation": 2,
        "members": [{"checkpoint_id": PTR}, {"checkpoint_id": PTR2}],
        "checkpoint_id": PTR,
    }), encoding="utf-8")
    ctl = CascadeController(reign_days=5)
    cfg2 = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    r = ValidatorRunner(cfg=cfg2, verify_signatures=False, cascade=ctl,
                        bench_report_store=_Store())
    assert r.check_manifest(_manifest(cfg2, warm_start_ckpt=PTR2)) is None
    assert ctl.state.generation == 2 and ctl.state.members == (PTR, PTR2)


def test_unreadable_pin_state_fails_closed(cfg, tmp_path):
    ws = tmp_path / "warm_start_init.json"
    cfg2 = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    ws.write_text("{corrupt", encoding="utf-8")
    r = ValidatorRunner(cfg=cfg2, verify_signatures=False,
                        cascade=CascadeController(reign_days=5),
                        bench_report_store=_Store())
    reason = r.check_manifest(_manifest(cfg2, warm_start_ckpt=PTR))
    assert reason is not None and "warm_start_state_unreadable" in reason


# ── remote worker plumbing ───────────────────────────────────────────────────


def test_worker_argv_carries_warm_start_ref():
    argv = worker_argv(
        RemoteHost(name="box", host="1.2.3.4", remote_python="/venv/python"),
        gen_ref=REF, uid=3, hotkey="hkX", role="king",
        base_seed=99, block=12, trainer_spec="m:C", warm_start_ref=PTR,
    )
    assert argv[argv.index("--warm-start-ref") + 1] == PTR


def test_worker_argv_omits_warm_start_by_default():
    argv = worker_argv(
        RemoteHost(name="box", host="1.2.3.4", remote_python="/venv/python"),
        gen_ref=REF, uid=3, hotkey="hkX", role="king",
        base_seed=99, block=12, trainer_spec="m:C",
    )
    assert "--warm-start-ref" not in argv
