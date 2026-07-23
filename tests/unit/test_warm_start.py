"""Cascade warm-start consumption (DEC-CA-0005/0004): the trainer's pointer
read, the validator's signed-pin gate, and the remote worker plumbing.

The pointer file is what a fired Cascade installs (``warm_start_init_path``);
the trainer reads it, trains every matching-size run from the pinned checkpoint,
and stamps the pin onto the signed manifest; each validator then requires the
manifest's pin to equal the init its OWN deterministic promotion installed.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost, worker_argv
from cascade.validator.cascade import CascadeController
from cascade.validator.loop import ValidatorRunner

REF = "alice/metro-gen@sha256:" + "a" * 64
REF_T = "cascade/ckpt-r1-king-toto2-4m@sha256:" + "b" * 64
PTR = format_trained_pointer(REF_T)


# ── trainer: reading the promoted-init pointer ───────────────────────────────


def _trainer(cfg, tmp_path, ws_path):
    return TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                         warm_start_path=ws_path)


def test_no_pointer_path_means_random_init(cfg, tmp_path):
    assert _trainer(cfg, tmp_path, None)._load_warm_start() is None


def test_absent_pointer_file_means_random_init(cfg, tmp_path):
    assert _trainer(cfg, tmp_path, tmp_path / "nope.json")._load_warm_start() is None


def test_pointer_file_yields_ref_and_size(cfg, tmp_path):
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


# ── validator: the signed-pin gate ───────────────────────────────────────────


def _entry(role, uid):
    return TrainedEntry(f"hk{uid}", uid, role, REF, PTR, "d", 10)


def _manifest(cfg, *, warm_start_ckpt=""):
    return TrainingManifest(
        round_id="1", created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=[_entry("king", 0), _entry("challenger", 1)],
        warm_start_ckpt=warm_start_ckpt,
        warm_start_size="toto2-4m" if warm_start_ckpt else "",
    )


def _validator(cfg, tmp_path, *, cascade: bool, installed: str | None = None):
    """A runner with the warm-start pin file under tmp; ``installed`` writes the
    pointer a fired Cascade would have installed."""
    ws = tmp_path / "warm_start_init.json"
    cfg = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    if installed is not None:
        ws.write_text(json.dumps({"checkpoint_id": installed}), encoding="utf-8")
    ctl = CascadeController(reign_days=7) if cascade else None
    return ValidatorRunner(cfg=cfg, verify_signatures=False, cascade=ctl)


def test_gate_off_when_cascade_disabled(cfg, tmp_path):
    # Pure KOTH ignores the field entirely (even a pinned manifest passes).
    r = _validator(cfg, tmp_path, cascade=False)
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None


def test_random_init_passes_before_any_promotion(cfg, tmp_path):
    r = _validator(cfg, tmp_path, cascade=True)
    assert r.check_manifest(_manifest(cfg)) is None


def test_unexpected_pin_rejected_before_any_promotion(cfg, tmp_path):
    r = _validator(cfg, tmp_path, cascade=True)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR))
    assert reason is not None and "warm_start_mismatch" in reason


def test_matching_pin_passes_after_promotion(cfg, tmp_path):
    r = _validator(cfg, tmp_path, cascade=True, installed=PTR)
    assert r.check_manifest(_manifest(cfg, warm_start_ckpt=PTR)) is None


def test_random_init_rejected_once_promotion_is_live(cfg, tmp_path):
    # The heart of DEC-CA-0005: a trainer that silently fell back to random
    # init after a promotion must be rejected, not scored.
    r = _validator(cfg, tmp_path, cascade=True, installed=PTR)
    reason = r.check_manifest(_manifest(cfg))
    assert reason is not None and "warm_start_mismatch" in reason


def test_stale_pin_rejected(cfg, tmp_path):
    other = format_trained_pointer("cascade/ckpt-r0-king-toto2-4m@sha256:" + "c" * 64)
    r = _validator(cfg, tmp_path, cascade=True, installed=PTR)
    reason = r.check_manifest(_manifest(cfg, warm_start_ckpt=other))
    assert reason is not None and "warm_start_mismatch" in reason


def test_unreadable_pin_state_fails_closed(cfg, tmp_path):
    ws = tmp_path / "warm_start_init.json"
    cfg2 = replace(cfg, validator=replace(cfg.validator, warm_start_init_path=str(ws)))
    ws.write_text("{corrupt", encoding="utf-8")
    r = ValidatorRunner(cfg=cfg2, verify_signatures=False,
                        cascade=CascadeController(reign_days=7))
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
