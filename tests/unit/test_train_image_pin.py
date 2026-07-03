"""[training] train_image_digest — the runtime-image pin for FINAL runs.

The pin is part of the byte-exact re-derivation contract (alongside
``expected_gpu``): it is folded into ``contract_digest``, and a trainer/worker
refuses a final run when its runtime (``CASCADE_TRAIN_IMAGE_DIGEST``, injected
at pod launch) doesn't carry the pinned image digest.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.manifest import contract_digest
from cascade.trainer.contract import (
    TRAIN_IMAGE_DIGEST_ENV,
    TrainImageMismatch,
    assert_train_image,
)

SHA = "sha256:" + "ab" * 32
OTHER_SHA = "sha256:" + "cd" * 32
PINNED_REF = f"ghcr.io/tensorlink-ai/cascade-worker@{SHA}"


def test_train_image_digest_folded_into_contract_digest(cfg):
    base = contract_digest(cfg.training)
    assert base != contract_digest(replace(cfg.training, train_image_digest=SHA))


def test_loader_reads_train_image_digest(cfg):
    # The shipped chain.toml leaves the pin empty (unpinned).
    assert cfg.training.train_image_digest == ""


def test_for_size_inherits_image_pin(two_size_cfg):
    training = replace(two_size_cfg.training, train_image_digest=SHA)
    for size in training.all_sizes():
        assert size.train_image_digest == SHA


def test_unpinned_contract_never_refuses(cfg, monkeypatch):
    monkeypatch.delenv(TRAIN_IMAGE_DIGEST_ENV, raising=False)
    assert_train_image(cfg.training)  # no pin ⇒ no check


def test_pinned_contract_refuses_without_runtime_digest(cfg, monkeypatch):
    monkeypatch.delenv(TRAIN_IMAGE_DIGEST_ENV, raising=False)
    pinned = replace(cfg.training, train_image_digest=PINNED_REF)
    with pytest.raises(TrainImageMismatch, match="unset"):
        assert_train_image(pinned)


def test_pinned_contract_refuses_mismatched_runtime(cfg, monkeypatch):
    monkeypatch.setenv(TRAIN_IMAGE_DIGEST_ENV, OTHER_SHA)
    pinned = replace(cfg.training, train_image_digest=PINNED_REF)
    with pytest.raises(TrainImageMismatch, match="not the contracted"):
        assert_train_image(pinned)


@pytest.mark.parametrize("pin", [SHA, PINNED_REF])
@pytest.mark.parametrize("runtime", [SHA, PINNED_REF, SHA.upper()])
def test_matching_runtime_passes_bare_or_full_ref(cfg, monkeypatch, pin, runtime):
    # Bare digest and full digest-pinned ref are equivalent on both sides
    # (comparison is on the sha256), and case is normalised.
    monkeypatch.setenv(TRAIN_IMAGE_DIGEST_ENV, runtime)
    assert_train_image(replace(cfg.training, train_image_digest=pin))


def test_undigested_pin_is_a_loud_config_error(cfg, monkeypatch):
    monkeypatch.setenv(TRAIN_IMAGE_DIGEST_ENV, SHA)
    bad = replace(cfg.training, train_image_digest="ghcr.io/x/worker:latest")
    with pytest.raises(TrainImageMismatch, match="no\\s+sha256"):
        assert_train_image(bad)


def test_local_final_refuses_mismatched_runtime(cfg, monkeypatch, tmp_path):
    """The local-training final path enforces the pin before any GPU work."""
    from cascade.trainer.loop import TrainerRunner

    monkeypatch.setenv(TRAIN_IMAGE_DIGEST_ENV, OTHER_SHA)
    pinned = replace(cfg, training=replace(cfg.training, train_image_digest=SHA))
    runner = TrainerRunner(cfg=pinned, base_trainer=object(), work_root=tmp_path)
    with pytest.raises(TrainImageMismatch):
        runner._train_final([], seeds=None, block=0)
