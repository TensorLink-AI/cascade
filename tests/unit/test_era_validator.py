"""DEC-CA-0043 on the validator: the era envelope, the manifest chain walk,
ref binding at train_block, tenure in blocks, and the audit round-trip.

Every test is framed around the rollout invariant: before ROLLOVER the
validator and its receipts are byte-identical to the ungated code; from it
the envelope fails closed and the audit replays each settlement under its
own block."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cascade.audit import checks as C
from cascade.eval.scoring import WindowScore
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    BenchScores,
    bench_report_key,
    dump_bench_report,
)
from cascade.shared.chain import Commitment, seed_from_block_hash
from cascade.shared.era import min_effective_era, settlement_era
from cascade.shared.hippius import ObjectNotFound, StorageError, manifest_round_key
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
)
from cascade.shared.promotion import (
    PromotedMember,
    PromotionRecord,
    dump_promotion_record,
    promotion_index_key,
    promotion_record_key,
)
from cascade.validator import loop as L
from cascade.validator.cascade import CascadeController, CascadeState
from cascade.validator.loop import ValidatorRunner, chain_manifests, ref_as_of
from cascade.validator.state import ChampionState, dumps, genesis, loads

REF_K = "alice/gen@sha256:" + "a" * 64
REF_C = "bob/gen@sha256:" + "b" * 64
REF_C2 = "bob/gen@sha256:" + "e" * 64
PTR_K = format_trained_pointer("cascade/ckpt@sha256:" + "c" * 64)
PTR_C = format_trained_pointer("cascade/ckpt@sha256:" + "d" * 64)
PTR_C2 = format_trained_pointer("cascade/ckpt@sha256:" + "f" * 64)
M1A = format_trained_pointer("cascade/ckpt@sha256:" + "1" * 64)
M1B = format_trained_pointer("cascade/ckpt@sha256:" + "2" * 64)
M2A = format_trained_pointer("cascade/ckpt@sha256:" + "3" * 64)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _scores(scale, seed, n=300):
    rng = np.random.default_rng(seed)
    return [WindowScore(series_id=str(i), mase=float(rng.uniform(0.5, 1.5) * scale),
                        qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                        abs_target=float(rng.uniform(5.0, 10.0))) for i in range(n)]


def _noisy(base, factor, seed):
    rng = np.random.default_rng(seed)
    return [WindowScore(s.series_id, s.mase * (j := factor * float(rng.uniform(0.6, 1.4))),
                        s.qloss_per_q * j, s.abs_target) for s in base]


def _armed(cfg, *, settlements=4, rollover_eras=2):
    """cfg with every DEC-CA-0043 gate at one boundary-aligned rollover."""
    eb = cfg.round.epoch_blocks
    rollover = eb * settlements * rollover_eras
    round_cfg = replace(cfg.round, rolling_from_block=rollover, era_settlements=settlements)
    scoring = replace(cfg.scoring, era_king_from_block=rollover,
                      tenure_blocks_from_block=rollover,
                      margin_warmup_blocks=cfg.scoring.margin_warmup_rounds * eb,
                      cascade_reign_blocks=cfg.scoring.cascade_reign_days * eb)
    return replace(cfg, round=round_cfg, scoring=scoring)


def _rollover(cfg):
    return cfg.scoring.era_king_from_block


def _era_stamp(cfg, block, *, generation=0, member_index=0):
    era = settlement_era(cfg.round, block)
    return replace(era, generation=generation, member_index=member_index).to_json()


def _manifest(cfg, block, *, era=True, king=("king_hk", 0, REF_K, PTR_K),
              challengers=(("chal_hk", 1, REF_C, PTR_C),), train_block=None,
              warm_start="", prev_round_id="", round_id=None, gpus=("", ""),
              generation=0, member_index=0, stamp=None):
    tb = train_block if train_block is not None else block - 1   # legs finish before the boundary
    entries = [TrainedEntry(king[0], king[1], "king", king[2], king[3], "d", tb,
                            gpu_name=gpus[0])]
    for i, (hk, uid, ref, ptr) in enumerate(challengers):
        entries.append(TrainedEntry(hk, uid, "challenger", ref, ptr, "d", tb,
                                    gpu_name=gpus[1], duel_rank=i))
    return TrainingManifest(
        round_id=round_id or str(block), created_block=block + 5,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset, entries=entries,
        warm_start_ckpt=warm_start, warm_start_size="toto2-4m" if warm_start else "",
        era=(stamp if stamp is not None else
             (_era_stamp(cfg, block, generation=generation, member_index=member_index)
              if era else None)),
        prev_round_id=prev_round_id,
    )


# The reveal history the era gate binds refs against (fail closed without one):
# every fixture hotkey committed its fixture ref at block 0.
_DEFAULT_HISTORY = [
    Commitment(uid=0, hotkey="king_hk", coldkey=None,
               payload=f"metro-v1:gen:hippius:{REF_K}", commit_block=0),
    Commitment(uid=1, hotkey="chal_hk", coldkey=None,
               payload=f"metro-v1:gen:hippius:{REF_C}", commit_block=0),
    Commitment(uid=2, hotkey="x_hk", coldkey=None,
               payload=f"metro-v1:gen:hippius:{REF_K}", commit_block=0),
]


def _runner(cfg, *, king_scores=None, chal=None, state=None, cascade=None, store=None,
            history=None):
    king_scores = king_scores or _scores(1.0, 0)
    chal = chal or {}
    history = history or (lambda: list(_DEFAULT_HISTORY))

    def fake_eval(entry, windows):
        if entry.role == "king":
            return king_scores
        return chal.get(entry.miner_hotkey, _noisy(king_scores, 1.0, 5))

    return ValidatorRunner(cfg=cfg, state=state or genesis("king_hk", 0),
                           evaluate_fn=fake_eval, verify_signatures=False,
                           cascade=cascade, bench_report_store=store,
                           commitment_history_fn=history)


class _Store:
    def __init__(self, texts=None):
        self.texts = dict(texts or {})

    def get_text(self, key):
        if key not in self.texts:
            raise ObjectNotFound(key)
        return self.texts[key]


class _Chain:
    def __init__(self, hashes):
        self.hashes = hashes

    def block_hash(self, block):
        return self.hashes[int(block)]


# ── bit-identity before ROLLOVER ─────────────────────────────────────────────

def test_pre_rollover_validator_and_receipt_are_byte_identical(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    block = _rollover(armed) - eb * 3          # a pre-gate boundary
    strong = {"chal_hk": _noisy(_scores(1.0, 0), 0.6, 22)}
    outs, receipts = [], []
    for c in (cfg, armed):
        r = _runner(c, chal=strong)
        m = _manifest(c, block, era=False)
        out = r.process_settlement(m, windows=[], base_seed=block)
        receipt = r.build_round_receipt(m, base_seed=block, epoch_start_block=block,
                                        epoch_block_hash="0x" + "ab" * 32,
                                        outcome=out, windows=[])
        outs.append(out)
        receipts.append(receipt)
    assert receipts[0].canonical_body() == receipts[1].canonical_body()
    assert outs[0].result.lcb == outs[1].result.lcb
    assert b"era_start_block" not in receipts[1].canonical_body()
    # A stray era stamp before the gate is ignored, never gated on.
    stray = _manifest(armed, block, era=True)
    assert _runner(armed).check_manifest(stray) is None
    assert C.check_era(receipts[1], armed).status == C.PASS


# ── the envelope, validator → receipt → audit ─────────────────────────────────

def test_era_settlement_round_trips_validator_receipt_audit_and_fails_with_gate_off(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    block = _rollover(armed) + eb              # second settlement of the era
    era = settlement_era(armed.round, block)
    seed_hash = "0x" + "cd" * 32
    era_seed = seed_from_block_hash(seed_hash)
    r = _runner(armed, chal={"chal_hk": _noisy(_scores(1.0, 0), 0.6, 22)})
    m = _manifest(armed, block)
    out = r.process_settlement(m, windows=[], base_seed=block)
    assert out is not None and out.transition.dethroned
    receipt = r.build_round_receipt(m, base_seed=block, epoch_start_block=block,
                                    epoch_block_hash="0x" + "ab" * 32,
                                    outcome=out, windows=[], era_base_seed=era_seed)
    assert receipt.era_start_block == era.start_block
    assert receipt.era_base_seed == era_seed
    # training seeds are the ERA's, base_seed stays the boundary's
    from cascade.trainer.contract import RoundSeeds
    assert receipt.training_seed == RoundSeeds.derive(era_seed, armed.training).training_seed
    assert receipt.base_seed == block
    assert C.check_round_seeds(receipt, armed).status == C.PASS
    assert C.check_era(receipt, armed).status == C.WARN            # no chain
    chain = _Chain({era.seed_block: seed_hash})
    assert C.check_era(receipt, armed, chain).status == C.PASS, C.check_era(receipt, armed, chain).detail
    bad_chain = _Chain({era.seed_block: "0x" + "ef" * 32})
    assert C.check_era(receipt, armed, bad_chain).status == C.FAIL
    # gate off ⇒ the same receipt FAILS: era context recorded where none may be
    off = replace(armed, scoring=replace(armed.scoring, era_king_from_block=0))
    assert C.check_era(receipt, off).status == C.FAIL
    assert C.check_round_seeds(receipt, off).status == C.FAIL
    # and the loaded receipt keeps the fields
    from cascade.shared.receipt import dump_receipt, load_receipt
    again = load_receipt(dump_receipt(receipt))
    assert again.era_start_block == era.start_block and again.era_base_seed == era_seed
    assert again.canonical_body() == receipt.canonical_body()


def test_era_stamp_is_verified_against_the_grid(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    block = _rollover(armed) + eb * 2
    r = _runner(armed)
    assert r.check_manifest(_manifest(armed, block)) is None
    assert r.check_manifest(_manifest(armed, block, era=False)).startswith("era_missing")
    wrong = dict(_era_stamp(armed, block))
    wrong["seed_block"] = wrong["start_block"]
    assert r.check_manifest(_manifest(armed, block, stamp=wrong)).startswith("era_mismatch")
    wrong = dict(_era_stamp(armed, block))
    wrong["index"] += 1
    assert r.check_manifest(_manifest(armed, block, stamp=wrong)).startswith("era_mismatch")
    assert r.check_manifest(_manifest(armed, block, stamp={"index": 1})).startswith("era_malformed")
    # entries must have trained inside the era's window [seed_block, era_end)
    era = settlement_era(armed.round, block)
    early = _manifest(armed, block, train_block=era.seed_block - 1)
    assert r.check_manifest(early).startswith("era_entry_out_of_window")
    pretrained = _manifest(armed, block, train_block=era.seed_block)   # pre-train window
    assert r.check_manifest(pretrained) is None


def test_gpu_same_sku_fallback_lifted_only_after_the_gate(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    assert not cfg.training.expected_gpu
    before = _manifest(armed, _rollover(armed) - eb, era=False, gpus=("NVIDIA L40S", "NVIDIA H100"))
    assert _runner(armed).check_manifest(before).startswith("gpu_mismatch")
    after = _manifest(armed, _rollover(armed) + eb, gpus=("NVIDIA L40S", "NVIDIA H100"))
    assert _runner(armed).check_manifest(after) is None


# ── init resolution across a generation switch ───────────────────────────────

def _scores6(v):
    return BenchScores(gifteval_crps=v, gifteval_mase=v, boom_crps=v,
                       boom_mase=v, time_crps=v, time_mase=v)


def _promo_store(members, *, generation, effective_era, fired_block, bench_round="r9"):
    record = PromotionRecord(
        generation=generation, king_hotkey="king_hk", fired_round=bench_round,
        fired_block=fired_block, effective_era=effective_era,
        members=tuple(PromotedMember(checkpoint_id=p, size="toto2-4m",
                                     source_round=bench_round, score=v) for p, v in members))
    report = BenchReport(round_id=bench_round, created_block=fired_block, entries=tuple(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="hk", miner_uid=0,
                   trained_pointer=p, scores=_scores6(v)) for p, v in members))
    return _Store({
        promotion_index_key(): json.dumps({"latest_generation": generation}),
        promotion_record_key(generation): dump_promotion_record(record),
        bench_report_key(bench_round): dump_bench_report(report),
    })


def _cascade(armed, tmp_path, *, generation, members):
    return CascadeController(
        reign_days=armed.scoring.cascade_reign_days, round_cfg=armed.round,
        state=CascadeState(king_hotkey="king_hk", reign_start_block=0,
                           generation=generation, members=tuple(members)),
        state_path=tmp_path / "cascade_state.json")


def test_generation_switch_lands_on_the_announced_era_only(cfg, tmp_path):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    n_block = _rollover(armed) + eb * 4 * 3 + eb      # era n's first settlement (3 eras past rollover)
    era_n = settlement_era(armed.round, n_block)
    prev_block = n_block - eb                         # last settlement of era n−1 (= era n's start)
    era_prev = settlement_era(armed.round, prev_block)
    fired = era_prev.start_block - eb * 4 * 2         # fired two eras earlier
    assert min_effective_era(armed.round, fired) <= era_n.index
    members1 = (M1A, M1B)

    def _validator(store):
        cas = _cascade(armed, tmp_path, generation=1, members=members1)
        return _runner(armed, cascade=cas, store=store)

    # (a) effective_era too early ⇒ the record is ignored: a manifest trained
    #     from the new set fails, the old set still passes
    early = _promo_store([(M2A, 0.4)], generation=2, effective_era=era_prev.index,
                         fired_block=era_prev.start_block - 1)
    v = _validator(early)
    m_new = _manifest(armed, n_block, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m_new).startswith("era_init_mismatch")
    assert v.cascade.state.generation == 1 and v.cascade.state.pending_generation == 0

    # (b) a valid record: era n−1 trains on the OLD init, era n on the NEW one
    good = _promo_store([(M2A, 0.4)], generation=2, effective_era=era_n.index,
                        fired_block=fired)
    v = _validator(good)
    old_member = members1[era_prev.index % 2]
    m_prev = _manifest(armed, prev_block, warm_start=old_member, generation=1,
                       member_index=era_prev.index % 2)
    assert v.check_manifest(m_prev) is None
    # the pre-train window already knows the next era's init (staged)
    assert v.cascade.state.pending_generation == 2
    assert v.cascade.state.pending_effective_era == era_n.index
    assert v.cascade.state.generation == 1
    # a manifest on the new init BEFORE the era ⇒ mixed init, rejected
    m_prev_new = _manifest(armed, prev_block, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m_prev_new).startswith("era_init_mismatch")
    v.state = replace(v.state, last_handled_round_id=None)
    # era n: the new generation is installed when its first settlement is
    # HANDLED (the gate is pure: it judges against the staged ledger)
    m_n = _manifest(armed, n_block, warm_start=M2A, generation=2, member_index=0)
    assert v.check_manifest(m_n) is None
    assert v.cascade.state.generation == 1 and v.cascade.state.pending_generation == 2
    assert v.process_settlement(m_n, windows=[], base_seed=n_block) is not None
    assert v.cascade.state.generation == 2 and v.cascade.state.members == (M2A,)
    assert v.cascade.state.pending_generation == 0
    assert v.check_manifest(m_n) is None                 # same verdict after the install
    # (c) an era-n manifest still on the old init fails the envelope
    v2 = _validator(good)
    assert v2.check_manifest(_manifest(armed, n_block, warm_start=old_member, generation=1,
                                       member_index=era_prev.index % 2)).startswith("era_init_mismatch")
    # (d) stamp generation right, checkpoint wrong ⇒ mixed init
    v3 = _validator(good)
    assert v3.check_manifest(_manifest(armed, n_block, warm_start=M1A, generation=2,
                                       member_index=0)).startswith("era_init_mismatch")


def test_member_rotation_follows_the_era_index(cfg, tmp_path):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    cas = _cascade(armed, tmp_path, generation=1, members=(M1A, M1B))
    v = _runner(armed, cascade=cas, store=_Store())
    for k in range(4):
        block = _rollover(armed) + eb * 4 * k + eb
        era = settlement_era(armed.round, block)
        want = (M1A, M1B)[era.index % 2]
        assert v.check_manifest(_manifest(armed, block, warm_start=want, generation=1,
                                          member_index=era.index % 2)) is None
        other = (M1A, M1B)[(era.index + 1) % 2]
        assert v.check_manifest(_manifest(armed, block, warm_start=other, generation=1,
                                          member_index=(era.index + 1) % 2)
                                ).startswith("era_init_mismatch")


# ── dethrone mid-era ─────────────────────────────────────────────────────────

def test_dethrone_mid_era_adopts_the_winner_pointer_and_keeps_the_era(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    b1 = _rollover(armed) + eb
    era = settlement_era(armed.round, b1)
    king = _scores(1.0, 0)
    r = _runner(armed, king_scores=king, chal={"chal_hk": _noisy(king, 0.6, 22),
                                               "x_hk": _noisy(king, 1.3, 33)})
    out = r.process_settlement(_manifest(armed, b1), windows=[], base_seed=b1)
    assert out.transition.dethroned
    assert r.state.king_hotkey == "chal_hk"
    assert r.state.king_pointer == PTR_C
    assert r.state.king_since_block == b1
    assert r.state.era_index == era.index
    r.state = replace(r.state, last_handled_round_id=str(b1))
    # next settlement, same era: the winner's checkpoint IS the era king
    b2 = b1 + eb
    m2 = _manifest(armed, b2, king=("chal_hk", 1, REF_C, PTR_C),
                   challengers=(("x_hk", 2, REF_K, PTR_K),), prev_round_id=str(b1))
    out2 = r.process_settlement(m2, windows=[], base_seed=b2)
    assert out2 is not None and not out2.transition.dethroned
    assert r.state.era_index == era.index and r.state.king_pointer == PTR_C
    # another checkpoint of the same generator cannot be substituted
    m2b = _manifest(armed, b2, king=("chal_hk", 1, REF_C, PTR_C2),
                    challengers=(("x_hk", 2, REF_K, PTR_K),), prev_round_id=str(b1))
    assert r.check_manifest(m2b).startswith("era_king_pointer_mismatch")
    # a NEW era adopts the era's first king leg — when the settlement is
    # HANDLED, never in the gate: a transient after the gate leaves the
    # pointer unadopted, so a legitimately re-published manifest carrying a
    # different first-king-leg pointer is still judged.
    b3 = era.start_block + eb * 4 + eb            # next era's first settlement
    r.state = replace(r.state, last_handled_round_id=str(b2))
    m3 = _manifest(armed, b3, king=("chal_hk", 1, REF_C, PTR_C2),
                   challengers=(("x_hk", 2, REF_K, PTR_K),), prev_round_id=str(b2))
    assert r.check_manifest(m3) is None
    assert r.state.era_index == era.index and r.state.king_pointer == PTR_C
    good_eval = r.evaluate_fn

    def boom(entry, windows):
        raise RuntimeError("eval pod down")
    r.evaluate_fn = boom
    with pytest.raises(RuntimeError):
        r.process_settlement(m3, windows=[], base_seed=b3)
    assert r.state.era_index == era.index and r.state.king_pointer == PTR_C
    m3b = _manifest(armed, b3, king=("chal_hk", 1, REF_C, PTR_K),
                    challengers=(("x_hk", 2, REF_K, PTR_K),), prev_round_id=str(b2))
    assert r.check_manifest(m3b) is None
    r.evaluate_fn = good_eval
    assert r.process_settlement(m3b, windows=[], base_seed=b3) is not None
    assert r.state.era_index == era.index + 1 and r.state.king_pointer == PTR_K
    assert r.check_manifest(m3).startswith("era_king_pointer_mismatch")


def test_restart_mid_era_restores_era_index_and_king_pointer():
    st = ChampionState(king_hotkey="k", king_uid=0, tenure_rounds=2,
                       king_since_block=9108000, king_pointer=PTR_C, era_index=633)
    again = loads(dumps(st))
    assert again == st
    legacy = loads('{"king_hotkey": "k", "king_uid": 0, "tenure_rounds": 1}')
    assert legacy.king_since_block is None and legacy.king_pointer == "" and legacy.era_index is None
    assert "king_since_block" not in dumps(legacy)


# ── the manifest chain ───────────────────────────────────────────────────────

def test_chain_walk_catches_up_through_missed_settlements_and_latest_alone_wedges(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    king = _scores(1.0, 0)
    bA, bB, bC = (_rollover(armed) + eb * i for i in (0, 1, 2))
    mA = _manifest(armed, bA)
    mB = _manifest(armed, bB, prev_round_id=str(bA))                      # dethrone here
    mC = _manifest(armed, bC, prev_round_id=str(bB), king=("chal_hk", 1, REF_C, PTR_C),
                   challengers=(("x_hk", 2, REF_K, PTR_K),))
    store = _Store({manifest_round_key(m.round_id): dump_manifest(m) for m in (mA, mB, mC)})
    latest = dump_manifest(mC)
    # walked oldest-first back to the last handled settlement
    walk = chain_manifests(store, latest, str(bA))
    walked = list(walk.manifests)
    assert walk.reached
    assert [load_manifest(raw).round_id for raw, _ in walked] == [str(bB), str(bC)]
    assert chain_manifests(store, latest, None).manifests == ((latest, walked[-1][1]),)
    assert chain_manifests(store, latest, str(bB)).manifests == ((latest, walked[-1][1]),)
    # a validator that handled A, was down for B, and sees C:
    r = _runner(armed, king_scores=king, chal={"chal_hk": _noisy(king, 0.6, 22),
                                               "x_hk": _noisy(king, 1.3, 33)})
    r.state = replace(r.state, last_handled_round_id=str(bA))
    assert r._pending_manifests(store, latest, str(bA)) == walked
    # jumping to latest (the old code path) wedges on the envelope
    assert r.check_manifest(mC).startswith("manifest_chain_broken")
    # walking the chain: B dethrones, C is accepted with the new era king
    outB = r.process_settlement(mB, windows=[], base_seed=bB)
    assert outB.transition.dethroned and r.state.king_hotkey == "chal_hk"
    r.state = replace(r.state, last_handled_round_id=str(bB))
    assert r.check_manifest(mC) is None
    outC = r.process_settlement(mC, windows=[], base_seed=bC)
    assert outC is not None and r.state.king_hotkey == "chal_hk"
    # pre-rollover: latest alone, as before
    pre = _manifest(armed, _rollover(armed) - eb, era=False)
    raw = dump_manifest(pre)
    assert r._pending_manifests(store, raw, "whatever")[0][0] == raw


def test_chain_walk_never_latches_past_a_hole_or_the_depth_cap(cfg):
    """A walk that does not reach the last handled settlement handles NOTHING:
    judging the oldest collected manifest would fail its chain check, latch
    past the gap (a dethrone in it lost) and diverge this validator's throne
    from the fleet's. The loop retries next poll, deeper."""
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    blocks = [_rollover(armed) + eb * i for i in range(6)]
    ms = [_manifest(armed, b, prev_round_id=str(blocks[i - 1]) if i else "LEGACY")
          for i, b in enumerate(blocks)]
    store = _Store({manifest_round_key(m.round_id): dump_manifest(m) for m in ms[2:]})
    latest = dump_manifest(ms[-1])
    # depth cap
    walk = chain_manifests(store, latest, "unknown", max_depth=3)
    assert not walk.reached
    assert [load_manifest(raw).round_id for raw, _ in walk.manifests] == [str(b) for b in blocks[3:]]
    # a hole at ms[1]
    walk = chain_manifests(store, latest, str(blocks[0]))
    assert not walk.reached
    # the runner handles nothing, keeps its position and doubles the depth
    r = _runner(armed)
    r.state = replace(r.state, last_handled_round_id=str(blocks[0]))
    assert r.chain_walk_depth == L.CHAIN_WALK_DEPTH
    assert r._pending_manifests(store, latest, str(blocks[0])) == []
    assert r.chain_walk_depth == 2 * L.CHAIN_WALK_DEPTH
    assert r.state.last_handled_round_id == str(blocks[0])
    # the hole is filled: the walk reaches, everything is handled in order
    store.texts[manifest_round_key(ms[1].round_id)] = dump_manifest(ms[1])
    pending = r._pending_manifests(store, latest, str(blocks[0]))
    assert [load_manifest(raw).round_id for raw, _ in pending] == [str(b) for b in blocks[1:]]
    assert r.chain_walk_depth == L.CHAIN_WALK_DEPTH
    # the chain's ROOT (the first settlement links to the unchained last
    # legacy round): a validator whose position predates it walks to the
    # legacy round and handles it too
    full = _Store({manifest_round_key(m.round_id): dump_manifest(m) for m in ms})
    legacy = _manifest(armed, _rollover(armed) - eb, era=False, round_id="LEGACY")
    full.texts[manifest_round_key("LEGACY")] = dump_manifest(legacy)
    walk = chain_manifests(full, latest, "OLDER")
    assert walk.reached
    assert [load_manifest(raw).round_id for raw, _ in walk.manifests] == \
        ["LEGACY"] + [str(b) for b in blocks]
    # the depth never exceeds CHAIN_WALK_MAX_DEPTH
    r.chain_walk_depth = L.CHAIN_WALK_MAX_DEPTH
    assert r._pending_manifests(store, latest, "unknown") == []
    assert r.chain_walk_depth == L.CHAIN_WALK_MAX_DEPTH


def test_ref_binding_fails_closed_without_a_history_provider(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    block = _rollover(armed) + eb
    bare = ValidatorRunner(cfg=armed, state=genesis("king_hk", 0),
                           evaluate_fn=lambda e, w: [], verify_signatures=False)
    assert bare.commitment_history_fn is None
    assert bare.check_manifest(_manifest(armed, block)).startswith("era_ref_unverifiable")
    # pre-gate: never consulted
    assert bare.check_manifest(_manifest(armed, block - eb * 2, era=False)) is None


# ── ref binding at train_block ───────────────────────────────────────────────

def test_refs_are_judged_as_of_train_block_not_the_boundary(cfg):
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    block = _rollover(armed) + eb
    tb = block - 100                                   # leg started before the boundary
    history = [
        Commitment(uid=0, hotkey="king_hk", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{REF_K}", commit_block=tb - 500),
        Commitment(uid=1, hotkey="chal_hk", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{REF_C}", commit_block=tb - 10),
        # re-commit between leg start and settlement
        Commitment(uid=1, hotkey="chal_hk", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{REF_C2}", commit_block=tb + 50),
    ]
    assert ref_as_of(history, "chal_hk", tb) == REF_C
    assert ref_as_of(history, "chal_hk", tb + 50) == REF_C2
    assert ref_as_of(history, "chal_hk", tb - 11) is None
    r = _runner(armed, history=lambda: history)
    judged = _manifest(armed, block, train_block=tb)
    assert r.check_manifest(judged) is None
    rebound = _manifest(armed, block, train_block=tb,
                        challengers=(("chal_hk", 1, REF_C2, PTR_C),))
    assert r.check_manifest(rebound).startswith("era_ref_mismatch")
    nothing = _manifest(armed, block, train_block=tb,
                        challengers=(("nobody", 3, REF_C2, PTR_C),))
    assert r.check_manifest(nothing).startswith("era_ref_unbound")
    # a chain failure is a transient, not a verdict
    def boom():
        raise RuntimeError("rpc down")
    with pytest.raises(RuntimeError):
        _runner(armed, history=boom).check_manifest(judged)
    # pre-gate: never consulted
    assert _runner(armed, history=boom).check_manifest(
        _manifest(armed, _rollover(armed) - eb, era=False)) is None


# ── tenure in blocks ─────────────────────────────────────────────────────────

def test_first_settlement_after_the_grid_switch_keeps_the_kings_decayed_margin(cfg):
    eb_new = cfg.round.epoch_blocks // 4
    eb_old = cfg.round.epoch_blocks
    rollover = eb_old * 8
    base = replace(cfg, round=replace(cfg.round, epoch_blocks=eb_new, epoch_blocks_prev=eb_old,
                                      epoch_activation_block=rollover, rolling_from_block=rollover,
                                      era_settlements=4),
                   scoring=replace(cfg.scoring, era_king_from_block=rollover,
                                   tenure_blocks_from_block=rollover,
                                   margin_warmup_blocks=cfg.scoring.margin_warmup_rounds * eb_old))
    king = _scores(1.0, 0)
    weak = {"chal_hk": _noisy(king, 1.2, 9)}
    # a king with 14 rounds of tenure on the old grid, crowning block unknown
    state = ChampionState(king_hotkey="king_hk", king_uid=0, tenure_rounds=14)
    r = _runner(base, king_scores=king, chal=weak, state=state)
    out = r.process_settlement(_manifest(base, rollover), windows=[], base_seed=rollover)
    assert out.king_tenure_rounds == 56                       # 14 × 3600 / 900
    assert out.result.margin == pytest.approx(cfg.scoring.win_margin_end)
    receipt = r.build_round_receipt(_manifest(base, rollover), base_seed=rollover,
                                    epoch_start_block=rollover, epoch_block_hash="0x" + "ab" * 32,
                                    outcome=out, windows=[], era_base_seed=5)
    assert receipt.verdict.params["margin_warmup_rounds"] == cfg.scoring.margin_warmup_rounds * 4
    assert C.check_koth_params(receipt, base).status == C.PASS
    assert C.check_verdict(receipt, base).status == C.PASS
    # the anchor is imputed ONCE and persisted: the next settlement counts
    # 57, not 60 (re-imputing from the advanced counter slid it back a whole
    # old-grid round per settlement — tenure grew 4× per settlement)
    assert r.state.king_since_block == rollover - 14 * eb_old
    assert r.state.tenure_rounds == 15
    out_next = r.process_settlement(_manifest(base, rollover + eb_new), windows=[],
                                    base_seed=rollover + eb_new)
    assert out_next.king_tenure_rounds == 57
    assert r.state.king_since_block == rollover - 14 * eb_old
    # the same king one settlement BEFORE the switch: the counter, as today —
    # and nothing is anchored before the gate
    r2 = _runner(base, king_scores=king, chal=weak, state=state)
    out2 = r2.process_settlement(_manifest(base, rollover - eb_old, era=False), windows=[],
                            base_seed=rollover - eb_old)
    assert out2.king_tenure_rounds == 14
    assert r2.state.king_since_block is None


def test_unreadable_promotion_evidence_is_a_transient_for_the_era_gate(cfg, tmp_path):
    # An unreadable ledger is not evidence of a bad init: the era gate raises
    # (poll loop retries, no receipt) instead of judging against the stale
    # ledger and latching era_init_mismatch.
    armed = _armed(cfg)
    eb = cfg.round.epoch_blocks
    n_block = _rollover(armed) + eb * 4 * 3 + eb
    era_n = settlement_era(armed.round, n_block)
    era_prev = settlement_era(armed.round, n_block - eb)
    fired = era_prev.start_block - eb * 4 * 2
    assert min_effective_era(armed.round, fired) <= era_n.index
    good = _promo_store([(M2A, 0.4)], generation=2, effective_era=era_n.index,
                        fired_block=fired)
    m_n = _manifest(armed, n_block, warm_start=M2A, generation=2, member_index=0)

    class _Down(_Store):
        def __init__(self, texts, key):
            super().__init__(texts)
            self.key = key

        def get_text(self, key):
            if key == self.key:
                raise StorageError(f"s3_get_failed: {key}: 503")
            return super().get_text(key)

    for key in (promotion_index_key(), promotion_record_key(2), bench_report_key("r9")):
        cas = _cascade(armed, tmp_path, generation=1, members=(M1A, M1B))
        v = _runner(armed, cascade=cas, store=_Down(good.texts, key))
        with pytest.raises(StorageError):
            v.check_manifest(m_n)
        assert cas.state.generation == 1 and cas.state.pending_generation == 0
    # Readable: the same manifest is accepted.
    cas = _cascade(armed, tmp_path, generation=1, members=(M1A, M1B))
    v = _runner(armed, cascade=cas, store=good)
    assert v.check_manifest(m_n) is None
    assert cas.state.pending_generation == 2
