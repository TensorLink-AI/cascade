"""The eval-pool pin gate: a pinned manifest must match the validator's own
deterministic snapshot selection, unpinned manifests keep legacy behaviour,
and an unverifiable pin rejects loudly (never scores on unproven data)."""

from __future__ import annotations

import types

import pytest

from cascade.shared.hippius import StorageError
from cascade.shared.manifest import TrainingManifest
from cascade.validator.loop import POOL_PIN_READ_GRACE_SECONDS, ValidatorRunner

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
    # source resolved no snapshot (index genuinely ABSENT) → "resolved no snapshot"
    reason = ValidatorRunner.check_pool_pin(m, _source("", ""), block=1000)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
    assert "resolved no snapshot" in reason
    # provenance lookup blew up with an UNEXPECTED error (a bug, not a storage
    # read failure) — reject, never crash, and report the distinct failure.
    def boom(seed, *, block=None):
        raise RuntimeError("unexpected bug")
    reason = ValidatorRunner.check_pool_pin(
        m, types.SimpleNamespace(provenance_for_round=boom), block=1000)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
    assert "provenance lookup failed" in reason


def test_pin_read_failure_propagates_for_retry():
    # An UNREADABLE index (StorageError: auth/network/5xx) is a transient, not
    # a verdict: it must escape the gate so the live loop can retry within the
    # grace window instead of latching a reject — a 30-second bucket blip must
    # not cost the validator the whole round.
    m = _manifest(key=KEY, sha=SHA)

    def unreadable(seed, *, block=None):
        raise StorageError("s3_get_failed: pool/index.json: 403")

    with pytest.raises(StorageError):
        ValidatorRunner.check_pool_pin(
            m, types.SimpleNamespace(provenance_for_round=unreadable), block=1000)


def test_pin_read_grace_retries_then_rejects():
    runner = ValidatorRunner(cfg=None)
    err = StorageError("s3_get_failed: pool/index.json: 403")
    t0 = 1000.0
    # Within grace: None → the caller skips the cycle (no latch, no receipt).
    assert runner._pool_pin_read_failed("7", err, now=t0) is None
    assert runner._pool_pin_read_failed(
        "7", err, now=t0 + POOL_PIN_READ_GRACE_SECONDS - 1) is None
    # Grace expired: the terminal reject reason, carrying the honest cause.
    reason = runner._pool_pin_read_failed("7", err, now=t0 + POOL_PIN_READ_GRACE_SECONDS)
    assert reason is not None and reason.startswith("pool_pin_unverifiable")
    assert "persistently" in reason
    # The round was forgotten on expiry: a later failure (e.g. after a
    # re-publish of the same round) starts a FRESH grace window.
    assert runner._pool_pin_read_failed("7", err, now=t0 + 9000.0) is None
    # Independent rounds keep independent clocks.
    assert runner._pool_pin_read_failed("8", err, now=t0 + 9000.0) is None
