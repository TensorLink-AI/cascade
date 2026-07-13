"""Receipt publication path: scores threaded out of process_round, receipt
assembly in the validator runner, and the S3 publish/read helpers — the
integration test on the fake-round harness (no chain, no torch, no boto3)."""

from __future__ import annotations

import types

import numpy as np
import pytest

import cascade.shared.hippius as hippius
from cascade.eval.scoring import WindowScore
from cascade.shared.chain import Commitment, seed_from_block_hash
from cascade.shared.manifest import (
    TrainingManifest,
    contract_digest,
)
from cascade.shared.receipt import load_receipt
from cascade.trainer.contract import RoundSeeds
from cascade.validator.loop import ValidatorRunner, participants_from_commitments
from cascade.validator.state import genesis

from .receipt_fixture import (
    BLOCK_HASH,
    CREATED_BLOCK,
    EPOCH_START,
    GEN_REF_CHAL,
    GEN_REF_KING,
    make_manifest,
)


def _scores(scale, seed, n=300):
    rng = np.random.default_rng(seed)
    return [
        WindowScore(
            series_id=f"w{i}",
            mase=float(rng.uniform(0.5, 1.5) * scale),
            qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
            abs_target=float(rng.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _runner(cfg, evaluate_fn):
    return ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0),
                           evaluate_fn=evaluate_fn, verify_signatures=False)


def _strong_eval():
    king = _scores(1.0, 0)
    chal = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
            for s in king]
    return (lambda e, w: king if e.role == "king" else chal), king, chal


# ── scores threaded out of process_round ─────────────────────────────────────


def test_process_round_threads_per_entry_scores(cfg):
    base_seed = seed_from_block_hash(BLOCK_HASH)
    eval_fn, king, chal = _strong_eval()
    runner = _runner(cfg, eval_fn)
    outcome = runner.process_round(make_manifest(cfg, base_seed=base_seed), [], base_seed)
    assert outcome is not None
    roles = [(e.role, e.size) for e in outcome.entry_scores]
    assert roles == [("king", cfg.training.arch_preset), ("challenger", cfg.training.arch_preset)]
    king_rec = outcome.entry_scores[0]
    assert king_rec.hotkey == "king_hk" and king_rec.uid == 0
    assert len(king_rec.scores) == len(king)
    # exact scoring order and values survive
    assert [s.series_id for s in king_rec.scores] == [s.series_id for s in king]
    assert king_rec.scores[0].mase == king[0].mase


def test_process_round_threads_scores_per_size(two_size_cfg):
    from .test_validator_round import _multi_manifest

    eval_fn, king, _ = _strong_eval()
    runner = _runner(two_size_cfg, eval_fn)
    m = _multi_manifest(two_size_cfg, sizes=("toto2-4m", "toto2-test-xl"))
    outcome = runner.process_round(m, [], 7)
    assert [(e.role, e.size) for e in outcome.entry_scores] == [
        ("king", "toto2-4m"), ("challenger", "toto2-4m"),
        ("king", "toto2-test-xl"), ("challenger", "toto2-test-xl"),
    ]
    # pooled decision saw all sizes; records carry each size separately
    assert outcome.result.n_windows == 2 * len(king)


# ── participant resolution ────────────────────────────────────────────────────


def test_participants_keep_latest_precutoff_commit_per_hotkey():
    commits = [
        Commitment(0, "a", None, f"metro-v1:gen:hippius:{GEN_REF_KING}", 100),
        Commitment(0, "a", None, f"metro-v1:gen:hippius:{GEN_REF_CHAL}", 200),   # later wins
        Commitment(1, "b", None, f"metro-v1:gen:hippius:{GEN_REF_CHAL}", EPOCH_START),  # at cutoff ⇒ out
        Commitment(2, "c", None, "garbage", 50),                                  # unparseable ⇒ out
    ]
    parts = participants_from_commitments(commits, cutoff_block=EPOCH_START)
    assert len(parts) == 1
    assert parts[0].hotkey == "a" and parts[0].gen_ref == GEN_REF_CHAL
    assert parts[0].commit_block == 200


# ── receipt assembly on the runner ────────────────────────────────────────────


def test_build_round_receipt_scored_matches_outcome(cfg):
    base_seed = seed_from_block_hash(BLOCK_HASH)
    eval_fn, king, chal = _strong_eval()
    runner = _runner(cfg, eval_fn)
    manifest = make_manifest(cfg, base_seed=base_seed)
    windows = [types.SimpleNamespace(series_id=f"w{i}") for i in range(8)]
    outcome = runner.process_round(manifest, windows, base_seed)

    receipt = runner.build_round_receipt(
        manifest, base_seed=base_seed,
        epoch_start_block=EPOCH_START, epoch_block_hash=BLOCK_HASH,
        outcome=outcome, windows=windows,
        pool_provenance=("pool@ref", "sha256:" + "e" * 64),
        reward_uids=(1,), weights=(0.0, 1.0), validator_hotkey="5Val",
    )
    assert receipt.status == "scored"
    assert receipt.base_seed == base_seed
    seeds = RoundSeeds.derive(base_seed, cfg.training)
    assert receipt.generation_seed == seeds.generation_seed
    assert receipt.training_seed == seeds.training_seed
    assert receipt.manifest["round_id"] == manifest.round_id
    assert receipt.eval_context.window_ids == tuple(f"w{i}" for i in range(8))
    assert receipt.eval_context.pool_ref == "pool@ref"
    assert receipt.entry_scores == outcome.entry_scores
    v = receipt.verdict
    assert v.challenger_wins_round == outcome.result.challenger_wins_round
    assert v.dethroned == outcome.transition.dethroned
    assert v.king_hotkey == runner.state.king_hotkey
    assert v.bootstrap_seed == str(base_seed)
    assert receipt.weights == (0.0, 1.0)


def test_build_round_receipt_rejected_carries_reason(cfg):
    base_seed = seed_from_block_hash(BLOCK_HASH)
    runner = _runner(cfg, lambda e, w: [])
    manifest = make_manifest(cfg, base_seed=base_seed)
    receipt = runner.build_round_receipt(
        manifest, base_seed=base_seed,
        epoch_start_block=EPOCH_START, epoch_block_hash=BLOCK_HASH,
        reject_reason="signature_invalid",
    )
    assert receipt.status == "rejected"
    assert receipt.reject_reason == "signature_invalid"
    assert receipt.entry_scores == () and receipt.verdict is None


def test_build_round_receipt_scored_requires_outcome(cfg):
    base_seed = seed_from_block_hash(BLOCK_HASH)
    runner = _runner(cfg, lambda e, w: [])
    with pytest.raises(ValueError, match="outcome"):
        runner.build_round_receipt(
            make_manifest(cfg, base_seed=base_seed), base_seed=base_seed,
            epoch_start_block=EPOCH_START, epoch_block_hash=BLOCK_HASH,
        )


# ── end-to-end publish on the fake-round harness ─────────────────────────────


class _FakeHotkey:
    ss58_address = "5FakeValidatorHotkey"

    def sign(self, body: bytes) -> bytes:
        return b"SIG:" + body[:8]


class _FakeClient:
    """The minimal client surface _publish_round_receipt touches."""

    def __init__(self):
        self._wallet = types.SimpleNamespace(hotkey=_FakeHotkey())

    def block_hash(self, block):
        assert block == EPOCH_START  # epoch boundary derived from created_block
        return BLOCK_HASH

    def poll_commitments(self):
        return [
            Commitment(0, "king_hk", None, f"metro-v1:gen:hippius:{GEN_REF_KING}", 100),
            Commitment(1, "chal_hk", None, f"metro-v1:gen:hippius:{GEN_REF_CHAL}", 200),
        ]

    def wallet(self):
        return self._wallet


def test_published_receipt_matches_round_outcome(cfg, monkeypatch):
    """The A-workstream acceptance path: run a fake round, publish, and the
    receipt on 'S3' reproduces the outcome exactly."""
    published: dict[str, str] = {}
    publish_hotkeys: list[str] = []

    def fake_publish(store, text, round_id, *, validator_hotkey=""):
        publish_hotkeys.append(validator_hotkey)
        return published.setdefault(round_id, text)

    monkeypatch.setattr(hippius, "publish_receipt", fake_publish)

    base_seed = seed_from_block_hash(BLOCK_HASH)
    eval_fn, king, chal = _strong_eval()
    runner = _runner(cfg, eval_fn)
    manifest = make_manifest(cfg, base_seed=base_seed)
    windows = [types.SimpleNamespace(series_id=f"w{i}") for i in range(4)]
    outcome = runner.process_round(manifest, windows, base_seed)

    window_source = types.SimpleNamespace(
        provenance_for_round=lambda seed, *, block=None: ("pool@ref", "sha256:" + "e" * 64)
    )
    runner._publish_round_receipt(
        _FakeClient(), manifest, base_seed,
        outcome=outcome, windows=windows, window_source=window_source,
        reward_uids=(1,), weights=(0.0, 1.0, 0.0),
    )

    assert manifest.round_id in published
    receipt = load_receipt(published[manifest.round_id])
    assert receipt.status == "scored"
    assert receipt.epoch_start_block == EPOCH_START
    assert receipt.epoch_block_hash == BLOCK_HASH
    assert receipt.validator_hotkey == "5FakeValidatorHotkey"
    # published under the validator's own receipts/<hotkey>/ prefix
    assert publish_hotkeys == ["5FakeValidatorHotkey"]
    assert receipt.signature is not None  # signed with the (fake) wallet
    assert [p.hotkey for p in receipt.participants] == ["king_hk", "chal_hk"]
    assert receipt.eval_context.pool_ref == "pool@ref"
    assert receipt.verdict.dethroned == outcome.transition.dethroned
    assert receipt.entry_scores == outcome.entry_scores
    assert receipt.weights == (0.0, 1.0, 0.0)
    # embedded manifest is the gated manifest, verbatim
    assert receipt.load_embedded_manifest().entries[0].gen_ref == GEN_REF_KING


def test_published_rejection_receipt(cfg, monkeypatch):
    published: dict[str, str] = {}
    monkeypatch.setattr(
        hippius, "publish_receipt",
        lambda store, text, round_id, **kw: published.setdefault(round_id, text),
    )
    base_seed = seed_from_block_hash(BLOCK_HASH)
    runner = _runner(cfg, lambda e, w: [])
    # a manifest that fails the contract gate
    good = make_manifest(cfg, base_seed=base_seed)
    bad = TrainingManifest(
        round_id=good.round_id, created_block=CREATED_BLOCK,
        contract_digest="0" * 64, base_arch_digest=good.base_arch_digest,
        eval_dataset=good.eval_dataset, entries=good.entries,
    )
    reason = runner.check_manifest(bad)
    assert reason is not None and reason.startswith("contract_digest_mismatch")

    runner._publish_round_receipt(_FakeClient(), bad, base_seed, reject_reason=reason)
    receipt = load_receipt(published[bad.round_id])
    assert receipt.status == "rejected"
    assert receipt.reject_reason == reason
    assert contract_digest(cfg.training) in receipt.reject_reason


def test_receipt_failure_never_raises(cfg, monkeypatch):
    """A receipt hiccup must never disturb the round (weights/state are done)."""
    def boom(store, text, round_id, **kw):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(hippius, "publish_receipt", boom)
    base_seed = seed_from_block_hash(BLOCK_HASH)
    runner = _runner(cfg, lambda e, w: [])
    manifest = make_manifest(cfg, base_seed=base_seed)
    # must not raise
    runner._publish_round_receipt(_FakeClient(), manifest, base_seed,
                                  reject_reason="signature_invalid")


# ── S3 receipt helpers ───────────────────────────────────────────────────────


class _FakeS3Store:
    def __init__(self):
        self.objects: dict[str, str] = {}
        self.acls: dict[str, str | None] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.objects[key] = text
        self.acls[key] = acl

    def get_text(self, key):
        # Mirror the real S3Store: a missing object is a StorageError, not KeyError.
        if key not in self.objects:
            raise hippius.StorageError(f"s3_get_failed: {key}: missing")
        return self.objects[key]


class _ScopedS3Store(_FakeS3Store):
    """Writes allowed only under one validator's prefix — models the
    owner-issued credential scoped to ``receipts/<hotkey>/*``."""

    def __init__(self, allow_prefix: str):
        super().__init__()
        self.allow_prefix = allow_prefix

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        if not key.startswith(self.allow_prefix):
            raise hippius.StorageError(f"s3_put_failed: {key}: AccessDenied")
        super().put_text(key, text, content_type=content_type, acl=acl)


def test_publish_and_read_receipt_keys():
    store = _FakeS3Store()
    key = hippius.publish_receipt(store, '{"round_id":"42"}', "42")
    assert key == "receipts/round-42.json"
    # receipts are the audit-facing artefact: written world-readable
    assert store.acls[key] == "public-read"
    assert store.acls["receipts/latest.json"] == "public-read"
    assert hippius.read_receipt(store, "42") == '{"round_id":"42"}'
    assert hippius.read_latest_receipt(store) == '{"round_id":"42"}'
    # a second round moves latest but keeps the old round readable
    hippius.publish_receipt(store, '{"round_id":"43"}', "43")
    assert hippius.read_latest_receipt(store) == '{"round_id":"43"}'
    assert hippius.read_receipt(store, "42") == '{"round_id":"42"}'


HK = "5ValidatorHotkey"


def test_publish_prefixed_mirrors_legacy_keys():
    store = _FakeS3Store()
    key = hippius.publish_receipt(store, '{"round_id":"42"}', "42", validator_hotkey=HK)
    # authoritative copies under the validator's own prefix, world-readable
    assert key == f"receipts/{HK}/round-42.json"
    assert store.acls[key] == "public-read"
    assert store.acls[f"receipts/{HK}/latest.json"] == "public-read"
    # broad (owner) credential ⇒ the legacy shared keys stay fed for old readers
    assert store.objects["receipts/round-42.json"] == '{"round_id":"42"}'
    assert store.objects["receipts/latest.json"] == '{"round_id":"42"}'
    assert hippius.read_receipt(store, "42", HK) == '{"round_id":"42"}'
    assert hippius.read_latest_receipt(store, HK) == '{"round_id":"42"}'


def test_publish_with_scoped_key_skips_legacy_mirror():
    store = _ScopedS3Store(f"receipts/{HK}/")
    key = hippius.publish_receipt(store, '{"round_id":"42"}', "42", validator_hotkey=HK)
    # the prefix-scoped credential publishes cleanly under its own namespace…
    assert key == f"receipts/{HK}/round-42.json"
    assert key in store.objects and f"receipts/{HK}/latest.json" in store.objects
    # …and the denied legacy mirror is skipped, not fatal
    assert "receipts/round-42.json" not in store.objects
    assert "receipts/latest.json" not in store.objects


def test_read_receipt_falls_back_to_legacy_key():
    store = _FakeS3Store()
    # a pre-prefix receipt exists only at the legacy shared key
    hippius.publish_receipt(store, '{"round_id":"41"}', "41")
    assert hippius.read_receipt(store, "41", HK) == '{"round_id":"41"}'
    assert hippius.read_latest_receipt(store, HK) == '{"round_id":"41"}'
