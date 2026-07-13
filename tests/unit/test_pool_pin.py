"""The eval-pool pin gate: a pinned manifest must match the validator's own
deterministic snapshot selection, unpinned manifests keep legacy behaviour,
and an unverifiable pin rejects loudly (never scores on unproven data)."""

from __future__ import annotations

import types

from cascade.shared.manifest import TrainingManifest
from cascade.validator.loop import ValidatorRunner

KEY = "pool/snapshots/block-1000.tar"
SHA = "e" * 64


def _manifest(*, key: str = "", sha: str = "") -> TrainingManifest:
    return TrainingManifest(
        round_id="7", created_block=1000,
        contract_digest="c" * 64, base_arch_digest="a" * 64,
        eval_dataset="cascade-private-v1",
        eval_pool_key=key, eval_pool_sha256=sha,
    )


def _source(key: str, sha: str):
    return types.SimpleNamespace(
        provenance_for_round=lambda seed, *, block=None: (key, sha)
    )


def test_unpinned_manifest_passes_any_source():
    assert ValidatorRunner.check_pool_pin(_manifest(), _source("x", "y"), block=1000) is None
    # even a source with no provenance hook at all
    assert ValidatorRunner.check_pool_pin(_manifest(), object(), block=1000) is None


def test_matching_pin_passes():
    m = _manifest(key=KEY, sha=SHA)
    assert ValidatorRunner.check_pool_pin(m, _source(KEY, SHA), block=1000) is None


def test_mismatched_pin_rejects():
    m = _manifest(key=KEY, sha=SHA)
    reason = ValidatorRunner.check_pool_pin(m, _source(KEY, "f" * 64), block=1000)
    assert reason is not None and reason.startswith("pool_pin_mismatch")
    reason = ValidatorRunner.check_pool_pin(m, _source("pool/other.tar", SHA), block=1000)
    assert reason is not None and reason.startswith("pool_pin_mismatch")


def test_pin_without_verifiable_provenance_rejects():
    m = _manifest(key=KEY, sha=SHA)
    # no provenance hook on the source
    reason = ValidatorRunner.check_pool_pin(m, object(), block=1000)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
    # source resolved no snapshot
    reason = ValidatorRunner.check_pool_pin(m, _source("", ""), block=1000)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
    # provenance lookup blew up (unreadable index) — reject, never crash
    def boom(seed, *, block=None):
        raise RuntimeError("index unreadable")
    reason = ValidatorRunner.check_pool_pin(
        m, types.SimpleNamespace(provenance_for_round=boom), block=1000)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
