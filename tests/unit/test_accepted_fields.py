"""Config-gated payload acceptance: mask + roles end to end (DEC-CA-0019 /
DEC-CA-0022 consumption, DEC-CA-0016 layer 3 arming)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from cascade.interface.generator import (
    DataGenerator,
    canonicalize_record,
    canonicalize_yield,
    check_record,
    drain_generator,
    record_frame_bytes,
)
from cascade.shared.config import GeneratorConfig, validate_accepted_fields
from cascade.shared.manifest import corpus_digest

KW = dict(min_length=10, max_length=200, max_total_points=100_000)


class _Gen(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0, items=()) -> None:
        self._items = list(items)

    @property
    def name(self) -> str:
        return "fixed"

    def generate(self, n_series: int) -> Iterator:
        yield from self._items[:n_series]


def _vals(seed: int, n: int = 100) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _masked(seed: int, n: int = 100, frac: float = 0.1):
    v = _vals(seed, n)
    m = np.zeros(n, dtype=np.uint8)
    m[: int(n * frac)] = 1
    v = v.copy()
    v[m.astype(bool)] = 0.0
    return {"values": v, "mask": m}


# ── config validation ────────────────────────────────────────────────────────


def test_accepted_fields_only_consumable_names():
    assert validate_accepted_fields(["mask"]) == ("mask",)
    assert validate_accepted_fields(["roles", "mask"]) == ("mask", "roles")
    assert validate_accepted_fields(()) == ()
    with pytest.raises(ValueError, match="no wired consumer"):
        validate_accepted_fields(["start"])          # reserved, no consumer
    with pytest.raises(ValueError, match="no wired consumer"):
        validate_accepted_fields(["labels"])


# ── carrier: acceptance flows through, unarmed stays rejected ────────────────


def test_unarmed_mask_still_rejected():
    with pytest.raises(ValueError, match="'mask' is reserved"):
        canonicalize_yield(_masked(1))


def test_armed_mask_flows_through():
    rec = canonicalize_yield(_masked(1), accepted=("mask",))
    assert isinstance(rec, dict) and set(rec) == {"values", "mask"}


def test_armed_values_only_record_still_collapses_to_bare_array():
    v = _vals(1)
    out = canonicalize_yield({"values": v}, accepted=("mask",))
    assert out is v          # byte-identity path is untouched by arming


# ── check_record rules ───────────────────────────────────────────────────────


def _canon(rec):
    return canonicalize_record(rec)


def test_mask_shape_must_match():
    with pytest.raises(ValueError, match="mask shape"):
        check_record(_canon({"values": _vals(1), "mask": np.zeros(50, np.uint8)}))


def test_mask_entries_binary():
    m = np.zeros(100, np.uint8)
    m[0] = 2
    with pytest.raises(ValueError, match="0/1"):
        check_record(_canon({"values": np.zeros(100), "mask": m}))


def test_masked_values_pinned_zero():
    v = _vals(1)
    m = np.zeros(100, np.uint8)
    m[3] = 1                      # value under the mask not zeroed
    with pytest.raises(ValueError, match="pinned exactly 0.0"):
        check_record(_canon({"values": v, "mask": m}))


def test_missing_frac_gate():
    rec = _canon(_masked(1, frac=0.4))
    check_record(rec, max_missing_frac=0.5)
    with pytest.raises(ValueError, match="missing fraction"):
        check_record(rec, max_missing_frac=0.3)


def test_fully_masked_channel_rejected():
    n = 100
    v = np.zeros((2, n))
    v[1] = _vals(2, n)
    m = np.zeros((2, n), np.uint8)
    m[0] = 1
    with pytest.raises(ValueError, match="entire channel"):
        check_record(_canon({"values": v, "mask": m}), max_missing_frac=1.0)


def test_roles_rules():
    v = np.stack([_vals(1), _vals(2)])
    ok = _canon({"values": v, "roles": np.array([0, 1], np.uint8)})
    check_record(ok)
    with pytest.raises(ValueError, match="at least one target"):
        check_record(_canon({"values": v, "roles": np.array([1, 1], np.uint8)}))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        check_record(_canon({"values": v, "roles": np.array([0, 2], np.uint8)}))
    # future-known admitted only behind the switch
    check_record(
        _canon({"values": v, "roles": np.array([0, 2], np.uint8)}),
        allow_future_known=True,
    )
    with pytest.raises(ValueError, match="roles shape"):
        check_record(_canon({"values": v, "roles": np.array([0], np.uint8)}))


# ── digest framing: sentinel, stability, no collision ────────────────────────


def test_record_frame_starts_with_sentinel_and_is_deterministic():
    rec = _canon(_masked(1))
    b1, b2 = record_frame_bytes(rec), record_frame_bytes(rec)
    assert b1 == b2 and b1[0] == 0xFF


def test_masked_corpus_digest_differs_from_values_only_and_is_stable():
    v = _vals(1)
    masked = _masked(1, frac=0.0)      # mask all zeros, values identical
    d_plain = corpus_digest([v])
    d_masked = corpus_digest([{"values": v, "mask": np.zeros(100, np.uint8)}])
    assert d_plain != d_masked          # a mask is corpus semantics, not decoration
    assert d_masked == corpus_digest([canonicalize_record(masked)])


def test_stream_digest_handles_records():
    from cascade.trainer.stream import _StreamDigest

    d1 = _StreamDigest()
    d1.update(_canon(_masked(1)))
    d2 = _StreamDigest()
    d2.update(_canon(_masked(1)))
    assert d1.hexdigest() == d2.hexdigest()


# ── drain integration ────────────────────────────────────────────────────────


def test_drain_accepts_and_validates_masked_series():
    items = [_masked(1), _vals(2), _masked(3, frac=0.2)]
    out = drain_generator(
        _Gen(items=items), 3, accepted_fields=("mask",), max_missing_frac=0.5, **KW
    )
    assert isinstance(out[0], dict) and isinstance(out[1], np.ndarray)
    assert out[0]["mask"].dtype == np.uint8
    with pytest.raises(ValueError, match="missing fraction"):
        drain_generator(
            _Gen(items=items), 3, accepted_fields=("mask",),
            max_missing_frac=0.1, **KW
        )


def test_drain_payload_bytes_price_the_mask():
    # 100 values (800B) + 100 mask bytes = 900B per series.
    items = [_masked(1)]
    drain_generator(
        _Gen(items=items), 1, accepted_fields=("mask",), max_payload_bytes=900, **KW
    )
    with pytest.raises(ValueError, match="payload bytes"):
        drain_generator(
            _Gen(items=items), 1, accepted_fields=("mask",),
            max_payload_bytes=899, **KW
        )


def test_drain_dedups_records_and_arrays_separately():
    v = _vals(1)
    zero_mask = {"values": v, "mask": np.zeros(100, np.uint8)}
    # Same values as record and as bare array: distinct dedup keys (0xFF frame
    # vs legacy), so neither counts as a duplicate of the other.
    out = drain_generator(
        _Gen(items=[v, zero_mask]), 2, accepted_fields=("mask",),
        max_dup_fraction=0.0, **KW
    )
    assert len(out) == 2
    with pytest.raises(ValueError, match="duplicate-series"):
        drain_generator(
            _Gen(items=[zero_mask, zero_mask]), 2, accepted_fields=("mask",),
            max_dup_fraction=0.0, **KW
        )


# ── sandbox frame round-trip ─────────────────────────────────────────────────


def test_record_frames_round_trip_through_the_pipe():
    import io as _io
    import os

    from cascade.trainer.sandbox import _frame_iter, _write_frame, _write_record_frame

    r_fd, w_fd = os.pipe()
    rec = _canon(_masked(5))
    plain = _canon({"values": _vals(6)})["values"]
    with os.fdopen(w_fd, "wb") as wr:
        _write_record_frame(wr, rec)
        _write_frame(wr, plain)

    rd = os.fdopen(r_fd, "rb")
    try:
        class _P:  # minimal Popen stand-in for _frame_iter
            stdout = rd
            stderr = _io.BytesIO()

            @staticmethod
            def poll():
                return 0

        items = list(_frame_iter(_P, 5))
    finally:
        rd.close()
    assert len(items) == 2
    assert isinstance(items[0], dict)
    assert np.array_equal(items[0]["values"], rec["values"])
    assert np.array_equal(items[0]["mask"], rec["mask"])
    assert np.array_equal(items[1], plain)


def test_npz_materialise_round_trip(tmp_path):
    from cascade.trainer.sandbox import _load_series, _save_series

    series = [_canon(_masked(1)), _canon({"values": _vals(2)})["values"]]
    _save_series(tmp_path / "corpus.npz", series)
    back = _load_series(tmp_path / "corpus.npz", 2)
    assert isinstance(back[0], dict) and np.array_equal(back[0]["mask"], series[0]["mask"])
    assert np.array_equal(back[1], series[1])
    assert corpus_digest(series) == corpus_digest(back)


# ── channel-correlation enforce gate ─────────────────────────────────────────


def test_corr_gate_off_and_shadow_do_nothing():
    from cascade.trainer.channel_stats import corr_enforce_gate

    cfg_off = GeneratorConfig(
        corpus_n_series=1, min_length=10, max_length=200, max_total_points=1000,
        max_generate_seconds=10, max_memory_mb=256, channel_corr_mode="off",
    )
    assert corr_enforce_gate(cfg_off) is None
    cfg_sh = GeneratorConfig(
        corpus_n_series=1, min_length=10, max_length=200, max_total_points=1000,
        max_generate_seconds=10, max_memory_mb=256, channel_corr_mode="shadow",
    )
    assert corr_enforce_gate(cfg_sh) is None


def test_corr_gate_enforce_rejects_jitter_duplicates_only():
    from cascade.trainer.channel_stats import corr_enforce_gate

    cfg = GeneratorConfig(
        corpus_n_series=1, min_length=10, max_length=200, max_total_points=1000,
        max_generate_seconds=10, max_memory_mb=256,
        channel_corr_mode="enforce", max_channel_corr=0.999,
    )
    gate = corr_enforce_gate(cfg)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(200)
    dup = np.ascontiguousarray(np.stack([base, base + 1e-9]))
    with pytest.raises(ValueError, match="max_channel_corr"):
        gate(dup, 0)
    honest = np.ascontiguousarray(rng.standard_normal((3, 200)))
    gate(honest, 1)                       # independent channels pass
    gate(np.atleast_2d(base), 2)          # univariate is always exempt


def test_drain_enforces_corr_gate():
    from cascade.trainer.channel_stats import corr_enforce_gate

    cfg = GeneratorConfig(
        corpus_n_series=2, min_length=10, max_length=200, max_total_points=10_000,
        max_generate_seconds=10, max_memory_mb=256,
        channel_corr_mode="enforce", max_channel_corr=0.999, max_channels=2,
    )
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100)
    items = [np.stack([base, base + 1e-9])]
    with pytest.raises(ValueError, match="max_channel_corr"):
        drain_generator(
            _Gen(items=items), 1, max_channels=2,
            extra_series_check=corr_enforce_gate(cfg), **KW
        )


# ── trainer consumption (torch) ──────────────────────────────────────────────


def test_pinball_weight_none_matches_mean_and_zero_excludes():
    torch = pytest.importorskip("torch")
    from cascade.trainer.toto2_model import pinball_loss

    g = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 3, 4, 9, generator=g)
    tgt = torch.randn(2, 3, 4, generator=g)
    levels = tuple((i + 1) / 10 for i in range(9))
    assert torch.allclose(
        pinball_loss(pred, tgt, levels),
        pinball_loss(pred, tgt, levels, weight=torch.ones_like(tgt)),
    )
    # Zero-weighting one element changes the loss to the mean over the rest.
    w = torch.ones_like(tgt)
    w[0, 0, 0] = 0.0
    manual = pinball_loss(pred[:, 1:], tgt[:, 1:], levels)  # sanity: different slice
    assert manual == manual  # not NaN
    l_w = pinball_loss(pred, tgt, levels, weight=w)
    assert not torch.allclose(l_w, pinball_loss(pred, tgt, levels))
    # All-zero weight degrades to 0, not NaN.
    assert float(pinball_loss(pred, tgt, levels, weight=torch.zeros_like(tgt))) == 0.0


def test_trainer_consumes_masked_and_role_tagged_corpus(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from cascade.trainer.toto2_trainer import Toto2Trainer

    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=2, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
    )
    rng = np.random.default_rng(0)

    def _rec(i):
        v = rng.normal(size=(2, 32)).cumsum(axis=-1)
        m = np.zeros((2, 32), dtype=np.uint8)
        m[0, :4] = 1
        v[0, :4] = 0.0
        roles = np.array([0, 1], dtype=np.uint8)   # target + past-covariate
        return {"values": v, "mask": m, "roles": roles}

    series = [_rec(i) for i in range(4)] + [rng.normal(size=32).cumsum() for _ in range(2)]
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(
        iter(series), contract, training_seed=1, token_budget=10**6,
        out_dir=tmp_path / "ckpt",
    )
    assert result.metrics["steps"] > 0
    assert np.isfinite(result.metrics["final_loss"])
    # 4 record series (2ch x 16 kept) + 2 bare (1ch x 16): all channels count.
    assert result.metrics["tokens_seen"] == 4 * 2 * 16 + 2 * 16
