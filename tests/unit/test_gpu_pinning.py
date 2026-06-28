"""Byte-exact GPU pinning — the validator's matched-hardware gate and the
gpu_name manifest round-trip."""

from __future__ import annotations

from dataclasses import replace

from metronome.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
)
from metronome.validator.loop import ValidatorRunner

CID = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
CID_T = "QmPChd2hVbrJ6bfo3WBcTW4iZnpHm8TEzWkLHmLpXhF68A"


def _entry(role, uid, gpu):
    return TrainedEntry(f"hk{uid}", uid, role, CID, format_trained_pointer(CID_T),
                        "d", 10, tar_digest="t", gpu_name=gpu)


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
