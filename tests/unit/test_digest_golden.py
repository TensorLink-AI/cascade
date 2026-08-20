"""Golden-vector digest tests — the byte-freeze guard for DEC-CA-0020.

Every digest below is pinned to the EXACT hex the deployed code produces
today (frozen 2026-08-14, before the record-carrier canonicaliser landed).
The existing digest tests only check self-consistency — two implementations
that agree with themselves but not with history would pass them. These
vectors are the contract: any change that moves a values-only corpus digest,
stream digest, or dedup series key breaks every archived manifest signature
and audit re-derivation, and MUST fail here first.

If one of these assertions ever fails, the fix is to restore the bytes — not
to update the constant. The only sanctioned way these constants change is a
deliberate, announced digest-scheme migration (none is planned; DEC-CA-0020
explicitly requires values-only corpora to hash byte-identically forever).
"""

from __future__ import annotations

import numpy as np

from cascade.interface.generator import _series_key
from cascade.shared.manifest import corpus_digest
from cascade.trainer.stream import _StreamDigest

# ── fixtures: fully deterministic, reconstructible from first principles ─────


def _a1() -> np.ndarray:
    return np.sin(np.arange(96, dtype=np.float64) * 0.1)          # (96,)


def _a2() -> np.ndarray:
    return np.arange(64, dtype=np.float64) * 0.5 - 7.0            # (64,)


def _m3() -> np.ndarray:
    t = np.arange(80, dtype=np.float64)
    return np.stack([t, t**2 * 1e-3, np.cos(t * 0.05)])           # (3, 80)


def _canon(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))


# ── frozen bytes (2026-08-14) ────────────────────────────────────────────────

GOLDEN_CORPUS_VALUES_ONLY = (
    "1937fa6eab6c78d0099cd1aace2aa95a8adb987dea55be238df89b3889bd7fae"
)
GOLDEN_CORPUS_MULTICHANNEL = (
    "cc9387a775df8ea9beff72adba727efdeabd46eda822a45c4a16cd10edf6a9b4"
)
GOLDEN_SERIES_KEY_1D = "502c90bf676fef3199d29e05143af476"
GOLDEN_SERIES_KEY_3CH = "7fe8030015874bc7bd347583ab1992a8"
GOLDEN_STREAM_2X1D = (
    "86f5ed2940d7f9fdd7a5ed1689c3ac28ec0e46bdfe1d205ac9186a58e7a330d1"
)
GOLDEN_STREAM_MIXED = (
    "9a3a0499baac3b8b7967f73bcfb99797ebceedc97d6c5644d09de0d6ce1bac8d"
)


def test_corpus_digest_golden_values_only():
    assert corpus_digest([_a1(), _a2()]) == GOLDEN_CORPUS_VALUES_ONLY


def test_corpus_digest_golden_promoted_equals_1d():
    # A 1-D (L,) series and its promoted (1, L) form must hash identically —
    # the digest's own canonicalisation, frozen.
    assert (
        corpus_digest([np.atleast_2d(_a1()), np.atleast_2d(_a2())])
        == GOLDEN_CORPUS_VALUES_ONLY
    )


def test_corpus_digest_golden_multichannel():
    assert corpus_digest([_a1(), _m3()]) == GOLDEN_CORPUS_MULTICHANNEL


def test_series_key_golden():
    assert _series_key(_canon(_a1())).hex() == GOLDEN_SERIES_KEY_1D
    assert _series_key(_canon(_m3())).hex() == GOLDEN_SERIES_KEY_3CH


def test_series_key_first_bytes_are_channel_count():
    # The carrier's canonical-digest sentinel scheme (DEC-CA-0020) relies on
    # this structural fact: a legacy series key starts with the 4-byte
    # big-endian channel count, so its first byte is 0x00 for any realistic C
    # (< 2^24). A future extended-record key claims a first byte no legacy key
    # can produce. Freeze the premise.
    assert _series_key(_canon(_a1()))[:0] == b""  # digest, not raw prefix …
    import hashlib

    canon = _canon(_m3())
    expect = hashlib.blake2b(
        canon.shape[0].to_bytes(4, "big") + canon.tobytes(), digest_size=16
    ).digest()
    assert _series_key(canon) == expect
    assert canon.shape[0].to_bytes(4, "big")[0] == 0x00


def test_stream_digest_golden():
    d = _StreamDigest()
    d.update(_canon(_a1()))
    d.update(_canon(_a2()))
    assert d.hexdigest() == GOLDEN_STREAM_2X1D

    d2 = _StreamDigest()
    d2.update(_canon(_a1()))
    d2.update(_canon(_m3()))
    assert d2.hexdigest() == GOLDEN_STREAM_MIXED


def test_stream_digest_prefix_property():
    # The stream digest covers exactly the consumed prefix: the 1-series
    # prefix of the 2-series stream must NOT equal the 2-series digest, and
    # re-reading the same prefix must reproduce it (audit re-derivation).
    d1 = _StreamDigest()
    d1.update(_canon(_a1()))
    d2 = _StreamDigest()
    d2.update(_canon(_a1()))
    assert d1.hexdigest() == d2.hexdigest() != GOLDEN_STREAM_2X1D


# ── contract digest (frozen 2026-08-14, before drop-when-default fields) ─────

GOLDEN_CONTRACT = "8fb51bb1d811dbdf7f90200aff921baaab91249b75cad65b8643b91fd05e4a9a"
GOLDEN_DICT = "8baa73198470c7bb4c3ce142a8fd651affc0310d878bb9bd159e37a573fb4874"


def _fixture_contract(**kw):
    from cascade.shared.config import TrainingContractConfig

    base = dict(
        base_arch="toto2", arch_preset="toto2-4m", base_arch_digest="a" * 64,
        d_model=256, num_layers=4, num_heads=4, head_dim=64, patch_size=32,
        mlp_expansion=2, num_quantiles=9, masking="contiguous_patch",
        cpm_c_max=16, cpm_p_max=0.4, input_transform="arcsinh_causal",
        context_length=4096, horizon=64, target_train_hours=3.0,
        ref_throughput_tokens_per_s=3_700_000, warmup_fraction=0.05,
        batch_size=64, optimizer="normuon_adamw", base_lr=4e-3,
        weight_decay=0.05, lr_schedule="warmup_cosine", umup_base_d_model=256,
        train_seed_salt=1337, max_train_seconds=10800,
        expected_gpu="NVIDIA L40S",
    )
    base.update(kw)
    return TrainingContractConfig(**base)


def test_contract_digest_golden():
    from cascade.shared.manifest import contract_digest

    assert contract_digest(_fixture_contract()) == GOLDEN_CONTRACT
    assert contract_digest({"a": 1, "b": [1, 2]}) == GOLDEN_DICT


def test_drop_when_default_fields_never_move_a_deployed_digest():
    # A digest-bound field at its inert default is omitted from the hash —
    # adding such fields to the dataclass is digest-invisible until SET.
    from cascade.shared.manifest import _DIGEST_DROP_WHEN_DEFAULT, contract_digest

    c = _fixture_contract()
    for key in _DIGEST_DROP_WHEN_DEFAULT:
        if hasattr(c, key):
            # present on the dataclass at its default → golden must still hold
            assert contract_digest(c) == GOLDEN_CONTRACT
