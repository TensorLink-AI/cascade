"""W-IMG lanes B, C, D: the image each leg / eval ran is recorded in the
signed record (logged, never enforced), dropped at its default so every
existing record keeps its bytes (plan rev6.24 (e))."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from .helpers import IMG_Y, base_cfg

PTR = "metro-v1:trained:hippius:cascade/ckpt@sha256:" + "c" * 64
REF = "alice/recipe@sha256:" + "a" * 64


def test_image_records_start_at_the_activation_block(tmp_path):
    from cascade.shared.era import image_record_at

    cfg = base_cfg(tmp_path)
    assert not image_record_at(cfg, 10**9)                       # off: nothing recorded
    on = replace(cfg, v2=replace(cfg.v2, image_epoch_from_block=3600 * 1000))
    assert not image_record_at(on, 3600 * 1000 - 1)
    assert image_record_at(on, 3600 * 1000)


def _expert(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    return ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=REF, domain="energy",
                       selection={0: [1]}, exclude={}, base_king_pointer=PTR,
                       rebased_pointer=PTR, corpus_digest="d", train_block=1, commit_block=1,
                       **kw)


def test_lane_b_expert_entry_records_its_image_drop_when_default():
    from cascade.shared.expert_manifest import ExpertManifest, load_expert_manifest

    plain = _expert()
    stamped = _expert(train_image_digest=IMG_Y)
    m0 = ExpertManifest(round_id="1", created_block=1, king_pointer=plain.rebased_pointer,
                        lineage_tokens=0, entries=(plain,))
    m1 = replace(m0, entries=(stamped,))
    assert "train_image_digest" not in json.loads(m0.canonical_body())["entries"][0]
    assert json.loads(m1.canonical_body())["entries"][0]["train_image_digest"] == IMG_Y
    assert load_expert_manifest(m1.to_json()).entries[0].train_image_digest == IMG_Y
    assert load_expert_manifest(m0.to_json()).canonical_body() == m0.canonical_body()


def test_lane_c_entry_records_its_image_drop_when_default():
    from cascade.shared.lane_c_manifest import LaneCEntry, LaneCManifest, load_lane_c_manifest

    ptr = PTR
    base = dict(miner_hotkey="hk", miner_uid=1, target="mkt:ohlcv", recipe_ref=REF,
                trained_pointer=ptr, train_block=1)
    plain = LaneCEntry(**base)
    stamped = LaneCEntry(**base, train_image_digest=IMG_Y)
    m0 = LaneCManifest(round_id="1", era_index=1, created_block=1, base_pointer=ptr,
                       base_round_id="1", pool_block=0, entries=(plain,))
    m1 = replace(m0, entries=(stamped,))
    assert "train_image_digest" not in json.loads(m0.canonical_body())["entries"][0]
    assert json.loads(m1.canonical_body())["entries"][0]["train_image_digest"] == IMG_Y
    assert load_lane_c_manifest(m1.to_json()).entries[0].train_image_digest == IMG_Y
    assert load_lane_c_manifest(m0.to_json()).canonical_body() == m0.canonical_body()


def test_lane_d_report_records_the_eval_image_only_from_activation(tmp_path):
    from cascade.validator.lane_d import EVAL_IMAGE_EXTENSION
    from cascade.validator.lane_d_wiring import LaneDInputs, _eval_identity_of

    cfg = base_cfg(tmp_path)
    ident = {"git_head": "f" * 40, "package_hash": "e" * 64}
    inp = SimpleNamespace(eval_identity=ident)
    assert _eval_identity_of(cfg, inp, 10**9) is None            # before: not recorded
    on = replace(cfg, v2=replace(cfg.v2, image_epoch_from_block=3600))
    assert _eval_identity_of(on, inp, 3600) == ident
    assert _eval_identity_of(on, SimpleNamespace(eval_identity=None), 3600) is None
    assert EVAL_IMAGE_EXTENSION == "eval_image"
    assert "eval_identity" in LaneDInputs.__dataclass_fields__


def test_lane_d_settlement_threads_the_identity_into_extensions():
    """The report writer puts the identity under ``extensions`` (open, never
    consensus) and only when given one."""
    import inspect

    from cascade.validator import lane_d

    sig = inspect.signature(lane_d.settle_lane_d)
    assert sig.parameters["eval_identity"].default is None
    src = inspect.getsource(lane_d.settle_lane_d)
    assert "EVAL_IMAGE_EXTENSION" in src and "if eval_identity else {}" in src
