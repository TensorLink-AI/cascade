"""W-IMG lane A: one worker image per v2 seed epoch, chosen by the trainer,
checked by validators and the audit without any fixed pin (plan rev6.24 (e))."""

from __future__ import annotations

from dataclasses import replace

import pytest

from .helpers import IMG_X, IMG_Y, receipt, stamp, v2_setup


def _era(m):
    from cascade.shared.era import EraSpec

    return EraSpec.from_json(m.era)


# ── the validator gate ──────────────────────────────────────────────────────


def test_epoch_where_king_and_challengers_ran_one_image_passes(tmp_path):
    r, m, v2c, _ = v2_setup(tmp_path, rule=True)
    good = stamp(m, v2c)
    assert len(good.entries) >= 2              # a king and at least one challenger
    assert r.check_manifest(good) is None
    # the same with a declared contract naming another image (no pin: the
    # settlement's own image is the epoch's)
    assert r.check_manifest(stamp(m, v2c, body_image=IMG_Y)) is None


def test_mixed_images_inside_one_epoch_are_refused(tmp_path):
    r, m, v2c, _ = v2_setup(tmp_path, rule=True)
    mixed = stamp(m, v2c, images={1: IMG_Y})   # a challenger on another image
    reason = r.check_manifest(mixed)
    assert reason is not None and reason.startswith("era_image_mixed"), reason


def test_before_activation_mixed_images_keep_todays_rule(tmp_path):
    """Release-then-activate: with ``image_epoch_from_block`` = 0 the gate is
    today's DEC-CA-0067 check (an image-only difference is accepted)."""
    from cascade.shared.era import image_epoch_rule_at

    r, m, v2c, cfg = v2_setup(tmp_path, rule=False)
    assert cfg.v2.image_epoch_from_block == 0
    assert not image_epoch_rule_at(cfg, int(m.created_block), _era(m))
    assert r.check_manifest(stamp(m, v2c, images={1: IMG_Y})) is None


def test_image_changed_mid_epoch_is_refused_and_a_new_epoch_passes(tmp_path):
    """The validator records the epoch's image at its first handled
    settlement; a later settlement of the SAME epoch on another image is
    refused, while a settlement of the NEXT seed epoch may carry a new image
    with no validator config change."""
    r, m, v2c, cfg = v2_setup(tmp_path, rule=True)
    cfg = replace(cfg, v2=replace(cfg.v2, king_ladder=True))
    r.cfg = cfg
    epoch = r._init_epoch_start(_era(m))
    assert epoch is not None
    first = stamp(m, v2c, body_image=IMG_X)
    assert r.check_manifest(first) is None
    r._adopt_era_king(first)
    assert r.state.epoch_init["image"] == IMG_X
    changed = r.check_manifest(stamp(m, v2c, body_image=IMG_Y))
    assert changed is not None and changed.startswith("era_image_changed"), changed
    # the stored image belongs to an EARLIER seed epoch: the new image passes
    r.state = replace(r.state, epoch_init={**r.state.epoch_init, "epoch": int(epoch) - 1})
    assert r.check_manifest(stamp(m, v2c, body_image=IMG_Y)) is None


def test_rule_starts_only_at_a_seed_epoch_boundary(tmp_path):
    from cascade.shared.era import (
        image_epoch_rule_at,
        image_epoch_rule_for,
        seed_epoch_start_block,
    )

    _, m, _, cfg = v2_setup(tmp_path, rule=True)
    era = _era(m)
    start = seed_epoch_start_block(cfg.round, era)
    length = cfg.round.epoch_blocks * cfg.round.era_settlements
    assert start == int(era.start_block)       # the first v2 era opens the epoch
    assert image_epoch_rule_at(cfg, int(m.created_block), era)
    # a block INSIDE the epoch never splits it: the epoch keeps today's rule
    late = replace(cfg, v2=replace(cfg.v2, image_epoch_from_block=start + length))
    assert not image_epoch_rule_for(late, start)
    assert image_epoch_rule_for(late, start + length)
    # every era of one seed epoch maps to the same start (the stamp's seed block)
    later = replace(era, index=era.index + 1, start_block=era.start_block + length)
    assert seed_epoch_start_block(cfg.round, later) == start


# ── the audit (receipt replay) ──────────────────────────────────────────────


def test_audit_image_epoch_per_receipt_and_across_receipts(tmp_path):
    from cascade.audit.checks import check_image_epoch, check_image_epoch_chain

    _, m, v2c, cfg = v2_setup(tmp_path, rule=True)
    block = int(m.created_block)
    assert check_image_epoch(receipt(stamp(m, v2c), block), cfg).status == "PASS"
    assert check_image_epoch(receipt(stamp(m, v2c, images={1: IMG_Y}), block),
                             cfg).status == "FAIL"
    a = receipt(stamp(m, v2c, body_image=IMG_X), block)
    b = receipt(stamp(m, v2c, body_image=IMG_Y), block)
    assert check_image_epoch_chain([a, a], cfg).status == "PASS"
    assert check_image_epoch_chain([a, b], cfg).status == "FAIL"     # changed mid-epoch
    # the next seed epoch (a later seed block in the era stamp) may change it
    length = cfg.round.epoch_blocks * cfg.round.era_settlements
    nxt = dict(b.manifest)
    nxt["era"] = {**dict(nxt["era"]), "seed_block": int(nxt["era"]["seed_block"]) + length,
                  "start_block": int(nxt["era"]["start_block"]) + length,
                  "index": int(nxt["era"]["index"]) + 1}
    b2 = replace_ns(b, manifest=nxt, epoch_start_block=block + length)
    assert check_image_epoch_chain([a, b2], cfg).status == "PASS"


def replace_ns(ns, **kw):
    from types import SimpleNamespace

    return SimpleNamespace(**{**vars(ns), **kw})


def test_audit_row_joins_only_when_configured(tmp_path):
    """Existing receipts replay byte-identically: before the activation the
    Tier-0 run carries no image-epoch row and the row itself reports the rule
    not in force."""
    from cascade.audit.checks import _image_epoch_configured, check_image_epoch

    _, m, v2c, cfg = v2_setup(tmp_path, rule=False)
    assert not _image_epoch_configured(cfg)
    row = check_image_epoch(receipt(stamp(m, v2c, images={1: IMG_Y}), int(m.created_block)), cfg)
    assert row.status == "PASS" and "not in force" in row.detail
    _, _, _, on = v2_setup(tmp_path / "on", rule=True)
    assert _image_epoch_configured(on)


# ── the trainer chooses the image per seed epoch ────────────────────────────


def _armed_after(cfg):
    from cascade.shared.config.chain_config import bind_image_after

    v2 = replace(cfg.v2, train_image_digest_after=IMG_Y,
                 funded_pod_image_after=f"repo@{IMG_Y}",
                 worker_code_fingerprint_after="c" * 64,
                 funded_sku_wall_measured_image_after="")
    return bind_image_after(replace(cfg, v2=v2))


def test_trainer_binds_a_new_image_at_the_next_seed_epoch_only(tmp_path):
    from cascade.shared.era import round_pins_at
    from cascade.trainer.image_epoch import (
        apply_binding,
        load_binding,
        resolve_binding,
        save_binding,
    )

    _, m, _, cfg = v2_setup(tmp_path, rule=True)
    cfg = _armed_after(cfg)
    length = cfg.round.epoch_blocks * cfg.round.era_settlements
    now = int(m.created_block)
    now_era = now // length
    # no seed epoch scheduled far enough ahead: unbound, every era keeps X
    assert resolve_binding(cfg, [now_era + 1], now, None) is None
    assert cfg.training.at_block(now_era * length + 10 * length).train_image_digest != IMG_Y
    # a warm start two eras ahead opens the next seed epoch: bound there
    b = resolve_binding(cfg, [now_era + 1, now_era + 3], now, None)
    assert b == {"image": IMG_Y, "block": (now_era + 3) * length}
    path = tmp_path / "image_epoch_state.json"
    save_binding(path, b)
    assert load_binding(path) == b
    # one-way: a later poll keeps the stored block for the same image
    assert resolve_binding(cfg, [now_era + 5], now + 2 * length, b) == b
    bound = apply_binding(cfg, b)
    assert bound.v2.image_after_epoch_block == b["block"]
    before = (now_era + 2) * length
    assert bound.training.at_block(before).train_image_digest != IMG_Y
    assert bound.training.at_block(b["block"]).train_image_digest == IMG_Y
    assert round_pins_at(bound, before) is bound.round
    assert round_pins_at(bound, b["block"]).funded_pod_image == f"repo@{IMG_Y}"


def test_without_w_img_the_switch_block_is_unchanged(tmp_path):
    """Byte-identical before activation: the after release still runs from
    ``image_decoupled_from_block`` (DEC-CA-0067), nothing else."""
    from cascade.shared.era import image_after_from, image_decoupled_from

    _, _, _, cfg = v2_setup(tmp_path, rule=False)
    cfg = _armed_after(cfg)
    assert image_after_from(cfg) == image_decoupled_from(cfg) > 0
    c = cfg.training.v2_contract
    assert c.train_image_after_block == image_decoupled_from(cfg)


@pytest.mark.parametrize("w_img", [False, True])
def test_contract_digests_are_unchanged(tmp_path, w_img):
    """The W-IMG keys are not contract terms: the shipped contract digests
    are the same with or without them."""
    from cascade.shared.manifest import contract_digest

    _, _, v2c, cfg = v2_setup(tmp_path, rule=w_img)
    assert bool(cfg.v2.image_epoch_from_block) is w_img
    assert contract_digest(cfg.training.v2_contract) == contract_digest(v2c)
    assert contract_digest(cfg.training.at_block(int(cfg.v2.from_block))) == contract_digest(
        v2c.at_block(int(cfg.v2.from_block)))
