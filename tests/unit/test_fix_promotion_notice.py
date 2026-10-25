"""The DEC-CA-0055 promotion notice as a CONSENSUS rule, after the review of
2026-10-25: a record whose notice is finally late is never effective for any
node — every validator's catch-up walk skips it and continues to the next
generation, the trainer's ledger keeps the previous init and proceeds to the
next generation on the same rule; a record that cannot be judged yet holds;
the trainer never publishes a record a settlement late (a failed publish at
the fire boundary retracts the fire); one body per generation."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.era import (
    EraSpec,
    era_for_block,
    era_length_blocks,
    min_effective_era,
    v2_first_warm_era,
)
from cascade.shared.hippius import ObjectNotFound, StorageError
from cascade.shared.manifest import TrainingManifest, dump_manifest
from cascade.shared.promotion import PromotedMember, PromotionRecord
from cascade.validator.cascade import CascadeController, CascadeState
from cascade.validator.era_gate import EraGateMixin
from cascade.validator.king_check import NOTICE_LATE, KingCheckMixin
from cascade.validator.lane_b import LaneBTransient


def _cfg(cfg):
    L = era_length_blocks(cfg.round, 0)
    gate = 10_000 * L
    return replace(cfg, v2=replace(cfg.v2, from_block=gate, king_ladder=True),
                   training=replace(cfg.training, v2_from_block=gate),
                   scoring=replace(cfg.scoring, era_king_from_block=gate)), gate, L


def _digest(rec):
    return hashlib.sha256(rec.canonical_body()).hexdigest()


class Chain:
    """Trainer-side chain builder: manifests hash-linked oldest → newest."""

    def __init__(self, signer=None):
        self.by_id: dict[str, TrainingManifest] = {}
        self.order: list[str] = []
        self.signer = signer

    def add(self, rid, created_block, notices=None):
        from cascade.shared.manifest import sign_manifest

        prev = self.order[-1] if self.order else ""
        prev_sha = (hashlib.sha256(dump_manifest(self.by_id[prev]).encode()).hexdigest()
                    if prev else "")
        m = TrainingManifest(rid, int(created_block), "c", "d", "e", prev_round_id=prev,
                             prev_manifest_sha256=prev_sha,
                             promotion_notices=dict(notices) if notices else None)
        if self.signer is not None:
            m = sign_manifest(m, self.signer)
        self.by_id[rid] = m
        self.order.append(rid)
        return m

    def load(self, rid):
        return self.by_id[rid]

    def head(self):
        return self.by_id[self.order[-1]]


def _notices(cfg, record, block):
    """runner_promotion._promotion_notices for one published record."""
    if int(record.effective_era) <= era_for_block(cfg.round, block).index:
        return {}
    return {str(record.generation): _digest(record)}


def _geometry(cfg, L):
    e = v2_first_warm_era(cfg) + 10               # well past the random-init window
    return cfg.round.epoch_blocks, e, e * L


def _record(cfg, gen, boundary, *, head_offset=5, ckpt="ckpt-king"):
    return PromotionRecord(gen, "KING", f"check-{boundary}", boundary + head_offset,
                           (PromotedMember(ckpt, "m", "r", 0.5),),
                           effective_era=min_effective_era(cfg.round, boundary))


def _two_generations(cfg, L, *, gen1_notice_from_k=4, signer=None):
    """Era e = [S, S+4g). Gen 1 fires at the 3rd settlement (k = 3); its
    notice first appears at ``gen1_notice_from_k`` (4 = the flush failed at
    the fire boundary ⇒ announced in era e+1 ⇒ late). Gen 2 fires at k = 13
    with a timely notice; the head is gen 2's effective era's first
    settlement."""
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    rec2 = _record(cfg, 2, S + 13 * g, ckpt="ckpt-king-2")
    chain = Chain(signer)
    for k in range(1, 22):
        b = S + k * g
        notices = {}
        if k >= gen1_notice_from_k:
            notices.update(_notices(cfg, rec1, b + 5))
        if k >= 13:
            notices.update(_notices(cfg, rec2, b + 5))
        chain.add(f"r{k}", b + 5, notices or None)
    era = era_for_block(cfg.round, chain.head().created_block)
    assert era.index == rec2.effective_era
    return rec1, rec2, chain, EraSpec(index=era.index, start_block=era.start_block,
                                      seed_block=era.start_block - L)


class _Probe(EraGateMixin, KingCheckMixin):
    pass


def _validator(cfg, chain, head, records, *, generation=0, members=()):
    p = _Probe()
    p.cfg = cfg
    p.verify_signatures = False
    p.cascade = CascadeController(reign_days=1.0, state=CascadeState(
        king_hotkey="KING", reign_start_block=0, generation=generation,
        members=tuple(members)))
    p.state = SimpleNamespace(king_check_pass=None, king_hotkey="KING")
    p._promotion_check_manifest = head
    p._lane_b_load_settlement = chain.load
    g = cfg.round.epoch_blocks
    p._epoch_start_block = lambda m: int(m.created_block) // g * g
    p._fetch_promotion_record = lambda gen, missing_ok=False: records.get(gen)
    p._fetch_latest_promotion_record = lambda: records[max(records)]
    p._verify_members = lambda *a, **k: None
    p._v2_lineage_size = lambda record: "m"
    return p


# ── the wedge: a late gen N no longer stops the walk at N−1 ──────────────────


def test_late_generation_is_skipped_and_the_next_timely_one_installs(cfg):
    cfg, gate, L = _cfg(cfg)
    rec1, rec2, chain, era = _two_generations(cfg, L)
    assert KingCheckMixin._v2_notice_reason(
        _validator(cfg, chain, chain.head(), {}), rec1) == NOTICE_LATE
    v = _validator(cfg, chain, chain.head(), {1: rec1, 2: rec2})
    v._refresh_pending_promotion(era)
    st = v.cascade.state
    # gen 1 skipped (never installed), gen 2 staged: the ledger for its
    # effective era names generation 2 — what the trainer's manifest declares
    assert (st.generation, st.members) == (0, ())
    assert (st.pending_generation, st.pending_members) == (2, ("ckpt-king-2",))
    assert EraGateMixin._era_ledger(v, era) == (2, ("ckpt-king-2",))
    # idempotent: a retry of the gate re-derives the same ledger
    v._refresh_pending_promotion(era)
    assert EraGateMixin._era_ledger(v, era) == (2, ("ckpt-king-2",))
    # handled: gen 2 installs, the skip of gen 1 is history
    EraGateMixin._install_era_promotion(v, replace(
        chain.head(), era=replace(era, generation=2, member_index=0).to_json()))
    st = v.cascade.state
    assert (st.generation, st.members, st.pending_generation) == (2, ("ckpt-king-2",), 0)
    assert EraGateMixin._era_ledger(v, era) == (2, ("ckpt-king-2",))


def test_skip_marker_keeps_the_live_init_until_the_next_record(cfg):
    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    chain = Chain()
    for k in range(1, 10):
        b = S + k * g
        chain.add(f"r{k}", b + 5, _notices(cfg, rec1, b + 5) if k >= 4 else None)
    head = chain.head()
    era_h = era_for_block(cfg.round, head.created_block)
    era = EraSpec(index=era_h.index, start_block=era_h.start_block, seed_block=era_h.start_block - L)
    v = _validator(cfg, chain, head, {1: rec1})
    v._refresh_pending_promotion(era)
    st = v.cascade.state
    # the marker: generation 1 is dead, nothing installed, the init stays gen 0
    assert (st.generation, st.members) == (0, ())
    assert (st.pending_generation, st.pending_members) == (1, ())
    assert EraGateMixin._skipped_generation(st) == 1 and EraGateMixin._staged_generation(st) == 0
    assert EraGateMixin._era_ledger(v, era) == (0, ())
    assert v._v2_warm_eras() == ()                   # no seed epoch from a dead record
    # the marker is persisted and never installed at any later settlement
    assert v.cascade.state == CascadeController(reign_days=1.0, state=st).state
    EraGateMixin._install_era_promotion(v, replace(
        head, era=replace(era, generation=0, member_index=0).to_json()))
    assert v.cascade.state.generation == 0
    # a dethrone during the notice era keeps it (DEC-CA-0055 keep rule)
    v.cascade.note_dethrone("5New", block=head.created_block, keep_pending=v._v2_pending_kept())
    assert EraGateMixin._skipped_generation(v.cascade.state) == 1
    # no further record ⇒ the refresh is a no-op (no re-walk of the chain)
    v._lane_b_load_settlement = lambda rid: pytest.fail("the dead record is re-judged")
    v._refresh_pending_promotion(era)
    assert EraGateMixin._skipped_generation(v.cascade.state) == 1


def test_a_record_that_cannot_be_judged_yet_holds_instead_of_skipping(cfg):
    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    chain = Chain()
    for k in range(1, 4):
        chain.add(f"r{k}", S + k * g + 5, _notices(cfg, rec1, S + k * g + 5) if k == 3 else None)
    # a validator behind the trainer gates r2 (created before the fire): not
    # announced YET — rejected without a marker, re-judged at the next gate
    era2 = era_for_block(cfg.round, chain.load("r2").created_block)
    v = _validator(cfg, chain, chain.load("r2"), {1: rec1})
    v._refresh_pending_promotion(EraSpec(index=era2.index, start_block=era2.start_block,
                                         seed_block=era2.start_block - L))
    assert v.cascade.state.pending_generation == 0
    # the fire boundary's manifest carries the notice: staged
    v._promotion_check_manifest = chain.load("r3")
    v._refresh_pending_promotion(EraSpec(index=era2.index, start_block=era2.start_block,
                                         seed_block=era2.start_block - L))
    assert (v.cascade.state.pending_generation, v.cascade.state.pending_members) == (1, ("ckpt-king",))


def test_catch_up_stamped_manifests_make_the_notice_late_for_every_node(cfg):
    """A trainer that settles boundaries late stamps created_block = the
    boundary (< the fire block = head): the walk never sees those notices,
    the first counted announcement is a settlement later — late. Validator
    walk and trainer probe agree (both skip)."""
    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    chain = Chain()
    for k in range(1, 10):
        b = S + k * g
        chain.add(f"r{k}", b, _notices(cfg, rec1, b) if k >= 3 else None)
    assert chain.load("r3").created_block < rec1.fired_block
    for rid in ("r4", "r9"):
        assert KingCheckMixin._v2_notice_reason(
            _validator(cfg, chain, chain.load(rid), {}), rec1) == NOTICE_LATE
    head = chain.head()
    era_h = era_for_block(cfg.round, head.created_block)
    v = _validator(cfg, chain, head, {1: rec1})
    v._refresh_pending_promotion(EraSpec(index=era_h.index, start_block=era_h.start_block,
                                         seed_block=era_h.start_block - L))
    assert EraGateMixin._skipped_generation(v.cascade.state) == 1


# ── fix 4: an unreadable chain manifest is a classified hold ─────────────────


def test_unreadable_notice_chain_manifest_is_a_storage_error(cfg):
    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    chain = Chain()
    for k in range(1, 7):
        chain.add(f"r{k}", S + k * g + 5, _notices(cfg, rec1, S + k * g + 5) if k >= 3 else None)

    def load(rid):
        if rid == "r4":
            raise LaneBTransient("manifests/round-r4.json unreadable: 503")
        return chain.load(rid)

    probe = SimpleNamespace(cfg=cfg, verify_signatures=False,
                            _promotion_check_manifest=chain.head(), _lane_b_load_settlement=load)
    with pytest.raises(StorageError, match="promotion_notice_unverifiable"):
        KingCheckMixin._v2_notice_reason(probe, rec1)
    # ... and the walk restores its state for the retry (the gate stays pure)
    head = chain.head()
    era_h = era_for_block(cfg.round, head.created_block)
    v = _validator(cfg, chain, head, {1: rec1})
    v._lane_b_load_settlement = load
    before = v.cascade.state
    with pytest.raises(StorageError):
        v._refresh_pending_promotion(EraSpec(index=era_h.index, start_block=era_h.start_block,
                                             seed_block=era_h.start_block - L))
    assert v.cascade.state == before


# ── fix 5: one body per generation ───────────────────────────────────────────


def test_a_staged_generation_republished_with_other_members_holds(cfg):
    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec1 = _record(cfg, 1, S + 3 * g)
    chain = Chain()
    for k in range(1, 5):
        chain.add(f"r{k}", S + k * g + 5, _notices(cfg, rec1, S + k * g + 5) if k >= 3 else None)
    era_h = era_for_block(cfg.round, chain.head().created_block)
    era = EraSpec(index=era_h.index, start_block=era_h.start_block, seed_block=era_h.start_block - L)
    records = {1: rec1}
    v = _validator(cfg, chain, chain.head(), records)
    v._refresh_pending_promotion(era)
    assert v.cascade.state.pending_members == ("ckpt-king",)
    records[1] = replace(rec1, members=(PromotedMember("ckpt-other", "m", "r", 0.5),))
    with pytest.raises(StorageError, match="promotion_record_conflict"):
        v._refresh_pending_promotion(era)
    assert v.cascade.state.pending_members == ("ckpt-king",)      # the staged body stands


def test_trainer_refuses_to_overwrite_a_published_record_with_another_body():
    from cascade.shared.promotion import (
        PromotionRecordConflict,
        dump_promotion_record,
        load_promotion_record,
        promotion_record_key,
        publish_promotion_record,
    )

    class _Store:
        def __init__(self):
            self.texts = {}

        def put_text(self, key, text, content_type="", acl=None):
            self.texts[key] = text

        def get_text(self, key):
            if key not in self.texts:
                raise ObjectNotFound(key)
            return self.texts[key]

    store = _Store()
    rec = PromotionRecord(1, "KING", "r3", 300, (PromotedMember("ckpt", "m", "r", 0.5),),
                          effective_era=7)
    publish_promotion_record(store, dump_promotion_record(rec), 1)
    # the same body again (a retry whose acknowledgement was lost) is fine
    publish_promotion_record(store, dump_promotion_record(replace(rec, signature="sig")), 1)
    with pytest.raises(PromotionRecordConflict):
        publish_promotion_record(store, dump_promotion_record(replace(rec, fired_block=301)), 1)
    assert load_promotion_record(store.get_text(promotion_record_key(1))).fired_block == 300
    # a malformed object at the key is replaced, a store read failure refuses
    store.texts[promotion_record_key(1)] = "{not json"
    publish_promotion_record(store, dump_promotion_record(rec), 1)
    assert load_promotion_record(store.get_text(promotion_record_key(1))).fired_block == 300


# ── the trainer side: same rule, same outcome ────────────────────────────────


def _trainer_probe(cfg, chain, store_texts, engine):
    from cascade.trainer.rolling import LegOps

    def get_text(key):
        if key not in store_texts:
            raise ObjectNotFound(key)
        return store_texts[key]

    runner = SimpleNamespace(cfg=cfg, promotion=engine,
                             manifest_store=lambda: SimpleNamespace(get_text=get_text),
                             _rolling_latest_round_id=lambda: chain.order[-1])
    return LegOps(runner)


def test_trainer_ledger_skips_the_late_generation_and_installs_the_next(cfg):
    from cascade.shared.hippius import manifest_round_key
    from cascade.shared.promotion import (
        dump_promotion_record,
        promotion_record_key,
        sign_promotion_record,
    )

    from .test_marketcast_validator import TRAINER_KP

    cfg, gate, L = _cfg(cfg)
    cfg = replace(cfg, manifest=replace(cfg.manifest, trainer_hotkey=TRAINER_KP.ss58_address))
    rec1, rec2, chain, era = _two_generations(cfg, L, signer=TRAINER_KP)
    rec1, rec2 = (sign_promotion_record(r, TRAINER_KP) for r in (rec1, rec2))
    texts = {promotion_record_key(1): dump_promotion_record(rec1),
             promotion_record_key(2): dump_promotion_record(rec2)}
    for rid in chain.order:
        texts[manifest_round_key(rid)] = dump_manifest(chain.load(rid))
    # gen 1 is the engine's live generation: never effective ⇒ (0, [], 0),
    # cached (the chain is not re-walked), the ledger keeps generation 0
    engine = SimpleNamespace(generation=1, members=rec1.members, pending_record=None,
                             unpublished_record=lambda: None)
    ops = _trainer_probe(cfg, chain, texts, engine)
    assert ops.promotion_generations() == (0, [], 0)
    ops.r.manifest_store = lambda: pytest.fail("the dead record is re-judged")
    assert ops.promotion_generations() == (0, [], 0)
    # gen 2 fires later with a timely notice: installed at ITS effective era
    engine2 = SimpleNamespace(generation=2, members=rec2.members, pending_record=None,
                              unpublished_record=lambda: None)
    assert _trainer_probe(cfg, chain, texts, engine2).promotion_generations() == (
        2, [("ckpt-king-2", "m")], rec2.effective_era)
    # and the validator's ledger for that era names the same generation
    v = _validator(cfg, chain, chain.head(), {1: rec1, 2: rec2})
    v._refresh_pending_promotion(era)
    assert EraGateMixin._era_ledger(v, era) == (2, ("ckpt-king-2",))


def test_flush_failure_at_the_fire_boundary_retracts_the_fire(cfg, tmp_path, caplog):
    """The 3rd-settlement fire whose record cannot be published: not fired
    (the engine returns to its pre-fire state) instead of announced in the
    next era. Re-read at the next boundary of the same era, it fires with
    that boundary's head block; a boundary in the next era refuses it."""
    from cascade.shared.promotion import promotion_record_key

    from .test_dec55_trainer_promotion import KING_PTR, _Runner, _v2

    cfg, gate, L = _v2(cfg)
    g = cfg.round.epoch_blocks
    S = (v2_first_warm_era(cfg) + 6) * L
    boundary = S + 3 * g
    check = {"result": "pass", "round_id": "r-check", "king_pointer": KING_PTR, "block": boundary}
    r = _Runner(cfg, tmp_path / "t", (check, boundary, "r-check"))
    outage = {"on": True}

    def publish(record):
        if outage["on"]:
            raise StorageError("s3 put failed: 503")
        r.published.append(record)
        r._store.texts[promotion_record_key(record.generation)] = "published"

    r._publish_promotion_record = publish
    texts = r._store.texts

    def get_text(key):
        if key not in texts:
            raise ObjectNotFound(key)
        return texts[key]

    r._store.get_text = get_text
    with caplog.at_level(logging.ERROR, logger="cascade.trainer"):
        r._promotion_boundary_step("KING", boundary + 5, "r3", effective_era=99)
    assert "retracted" in caplog.text
    eng = r.promotion
    assert (eng.generation, eng.members, eng.pending_record, eng.last_king_check) == (0, (), None, "")
    assert not r.published and not (tmp_path / "t" / "p.json").exists()
    # the next boundary (same era): the store is back ⇒ fires with its head
    outage["on"] = False
    later = boundary + g if era_for_block(cfg.round, boundary + g).index == era_for_block(
        cfg.round, boundary).index else boundary + 5 + 1
    r._promotion_boundary_step("KING", later, "r4", effective_era=99)
    assert len(r.published) == 1 and r.published[0].fired_block == later
    assert r.published[0].effective_era == min_effective_era(cfg.round, boundary)
    assert eng.generation == 1 and eng.pending_record is None


def test_retract_leaves_a_record_that_did_land_published(cfg, tmp_path):
    """The put landed but its acknowledgement was lost: the pending record is
    marked published (same body), nothing is retracted."""
    from cascade.shared.promotion import dump_promotion_record, promotion_record_key

    from .test_dec55_trainer_promotion import KING_PTR, _Runner, _v2

    cfg, gate, L = _v2(cfg)
    g = cfg.round.epoch_blocks
    boundary = (v2_first_warm_era(cfg) + 6) * L + 3 * g
    check = {"result": "pass", "round_id": "r-check", "king_pointer": KING_PTR, "block": boundary}
    r = _Runner(cfg, tmp_path / "t", (check, boundary, "r-check"))

    def publish(record):
        r._store.texts[promotion_record_key(record.generation)] = dump_promotion_record(record)
        raise StorageError("index put failed after the record landed")

    r._publish_promotion_record = publish
    r._promotion_boundary_step("KING", boundary + 5, "r3", effective_era=99)
    assert r.promotion.generation == 1 and r.promotion.pending_record is None


def test_notice_falls_back_to_the_copy_this_trainer_published(cfg):
    from cascade.trainer.runner_promotion import _PromotionMixin

    cfg, gate, L = _cfg(cfg)
    g, e, S = _geometry(cfg, L)
    rec = _record(cfg, 1, S + 3 * g)
    engine = SimpleNamespace(generation=1, unpublished_record=lambda: None)
    def get_text(key):
        raise StorageError("503")

    store = SimpleNamespace(get_text=get_text)
    runner = SimpleNamespace(cfg=cfg, promotion=engine, manifest_store=lambda: store)
    assert _PromotionMixin._promotion_notices(runner, rec.fired_block) == {}
    runner.__dict__["_last_published_promotion"] = rec
    assert _PromotionMixin._promotion_notices(runner, rec.fired_block) == {"1": _digest(rec)}
    # another generation's copy never stands in
    runner.__dict__["_last_published_promotion"] = replace(rec, generation=2)
    assert _PromotionMixin._promotion_notices(runner, rec.fired_block) == {}
