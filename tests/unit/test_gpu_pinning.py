"""Byte-exact GPU pinning — the validator's matched-hardware gate and the
gpu_name manifest round-trip."""

from __future__ import annotations

from dataclasses import replace

from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
)
from cascade.validator.loop import ValidatorRunner

REF = "alice/metro-gen@sha256:" + "a" * 64
REF_T = "cascade/ckpt-r1-king@sha256:" + "b" * 64


def _entry(role, uid, gpu):
    return TrainedEntry(f"hk{uid}", uid, role, REF, format_trained_pointer(REF_T),
                        "d", 10, gpu_name=gpu)


def _manifest(cfg, king_gpu, chal_gpu):
    return TrainingManifest(
        round_id="1", created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=[_entry("king", 0, king_gpu), _entry("challenger", 1, chal_gpu)],
    )


def _runner(cfg):
    return ValidatorRunner(cfg=cfg, verify_signatures=False)


def test_config_default_expected_gpu_is_empty(cfg):
    assert cfg.training.expected_gpu == ""


def test_matching_gpus_pass(cfg):
    assert _runner(cfg).check_manifest(_manifest(cfg, "NVIDIA H100", "NVIDIA H100")) is None


def test_different_gpus_rejected(cfg):
    reason = _runner(cfg).check_manifest(_manifest(cfg, "NVIDIA H100", "NVIDIA A100"))
    assert reason is not None and "gpu_mismatch" in reason


def test_empty_gpu_names_do_not_trip_gate(cfg):
    # legacy/CPU entries without a recorded GPU still pass (no pin, nothing to compare)
    assert _runner(cfg).check_manifest(_manifest(cfg, "", "")) is None


def test_pinned_gpu_enforced(cfg):
    pinned = replace(cfg, training=replace(cfg.training, expected_gpu="NVIDIA H100 80GB HBM3"))
    runner = _runner(pinned)
    assert runner.check_manifest(_manifest(pinned, "NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3")) is None
    # right hardware on both, but not the pinned SKU → rejected
    bad = runner.check_manifest(_manifest(pinned, "NVIDIA A100", "NVIDIA A100"))
    assert bad is not None and "gpu_mismatch" in bad
    # pinned but an entry has no recorded GPU → rejected
    missing = runner.check_manifest(_manifest(pinned, "NVIDIA H100 80GB HBM3", ""))
    assert missing is not None


def test_manifest_roundtrips_gpu_name(cfg):
    again = load_manifest(dump_manifest(_manifest(cfg, "NVIDIA H100", "NVIDIA H100")))
    assert again.entry_for_role("king").gpu_name == "NVIDIA H100"


# ── per-size pins (DEC-CA-0027 size-conditional silicon) ─────────────────────

H100 = "NVIDIA H100 80GB HBM3"


def _sized_entry(role, uid, gpu, size):
    return TrainedEntry(f"hk{uid}", uid, role, REF, format_trained_pointer(REF_T),
                        "d", 10, gpu_name=gpu, size=size)


def _sized_cfg(cfg):
    from cascade.shared.config import SizeSpec

    big = SizeSpec(
        arch_preset="toto2-313m", base_arch_digest="c" * 64, d_model=1024,
        num_layers=16, num_heads=16, mlp_expansion=2,
        ref_throughput_tokens_per_s=100_000, expected_gpu=H100,
    )
    return replace(cfg, training=replace(
        cfg.training, expected_gpu="NVIDIA L40S", extra_sizes=(big,)))


def _sized_manifest(cfg, entries):
    return TrainingManifest(
        round_id="1", created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=entries,
    )


def test_per_size_pins_each_size_judged_against_its_own_silicon(cfg):
    # The DEC-CA-0027 deadlock regression: a 313M duel on H100 beside a 4M
    # lineage on L40S must PASS — each size against its own pin — while the
    # old whole-manifest rule rejected any manifest not uniformly on the base
    # pin.
    sized = _sized_cfg(cfg)
    runner = _runner(sized)
    good = _sized_manifest(sized, [
        _sized_entry("king", 0, "NVIDIA L40S", ""),
        _sized_entry("challenger", 1, "NVIDIA L40S", ""),
        _sized_entry("king", 0, H100, "toto2-313m"),
        _sized_entry("challenger", 1, H100, "toto2-313m"),
    ])
    assert runner.check_manifest(good) is None

    # The large size on the WRONG silicon is rejected, naming the size.
    bad = runner.check_manifest(_sized_manifest(sized, [
        _sized_entry("king", 0, "NVIDIA L40S", ""),
        _sized_entry("king", 0, "NVIDIA H100 PCIe", "toto2-313m"),
    ]))
    assert bad is not None and "gpu_mismatch" in bad and "toto2-313m" in bad

    # And the base size still enforces the base pin.
    bad2 = runner.check_manifest(_sized_manifest(sized, [
        _sized_entry("king", 0, H100, ""),
        _sized_entry("king", 0, H100, "toto2-313m"),
    ]))
    assert bad2 is not None and "gpu_mismatch" in bad2
