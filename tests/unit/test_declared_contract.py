"""Declared-contract gating (DEC-CA-0036).

The property under test is operational, not numerical: a trainer-side change
that does NOT touch the locked terms must keep scoring on a validator whose
``chain.toml`` still says the old thing. That is the whole point — image
re-pins and recipe fixes ship the same day, and validator restarts are reserved
for actual scoring-rule changes.

The negative half matters just as much: the locked terms still gate, a body
that disagrees with its own digest is rejected, and with the block gate off
nothing changes at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.config import SizeSpec, load_chain_config
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    contract_payload,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
    locked_contract_terms,
)
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

CID = "alice/gen@sha256:" + "a" * 64
CID2 = "cascade/ckpt@sha256:" + "b" * 64

# Blocks are compared on the EPOCH grid (``_epoch_start_block`` floors
# ``created_block`` to it), so these have to be far enough apart that flooring
# keeps them on opposite sides of the gate.
DECLARED_FROM = 7_200
EPOCH_BLOCK = 720_000        # floors well past the gate
EARLY_BLOCK = 100            # floors to 0, i.e. before it


def _size(preset, **kw):
    """A SizeSpec with the boilerplate filled in."""
    return SizeSpec(arch_preset=preset, base_arch_digest="e" * 64, d_model=512,
                    num_layers=8, num_heads=8, mlp_expansion=2,
                    ref_throughput_tokens_per_s=185_000, **kw)


@pytest.fixture
def armed(cfg):
    """``cfg`` with declared-contract gating armed from ``DECLARED_FROM``."""
    return replace(cfg, scoring=replace(cfg.scoring,
                                        declared_contract_from_block=DECLARED_FROM))


def _manifest(cfg, *, trained_under=None, block=EPOCH_BLOCK, body=...):
    """A well-formed two-entry manifest.

    ``trained_under`` is the contract the TRAINER ran (defaults to ``cfg``'s) —
    the knob that lets a test put the trainer ahead of the validator. ``body``
    overrides the published contract body (``None`` = publish none, i.e. a
    pre-DEC-CA-0036 trainer).
    """
    training = trained_under if trained_under is not None else cfg.training
    if body is ...:
        body = contract_payload(training)
    entries = [
        TrainedEntry("king_hk", 0, "king", CID, format_trained_pointer(CID2), "d", 10),
        TrainedEntry("chal_hk", 1, "challenger", CID, format_trained_pointer(CID2), "d", 10),
    ]
    return TrainingManifest(
        round_id="1",
        created_block=block,
        contract_digest=contract_digest(training),
        base_arch_digest=training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=entries,
        contract_body=body,
    )


def _gate(cfg, manifest):
    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0),
                             evaluate_fn=lambda e, w: [], verify_signatures=False)
    return runner.check_manifest(manifest)


# ── the point of the whole change ────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    # The original motivation: re-pin the training runtime image.
    ("train_image_digest", "sha256:" + "c" * 64),
    # DEC-CA-0018-shaped recipe edits.
    ("lr_schedule", "warmup_cosine"),
    ("base_lr", 8e-3),
    ("optimizer", "adamw"),
    ("warmup_fraction", 0.10),
    ("batch_size", 128),
    # DEC-CA-0035/0033-class recipe knobs (values deliberately differ from the
    # shipped armed cut so the digest actually moves against the fixture).
    ("ema_decay", 0.99),
    ("warm_lr_scale", 0.25),
    ("weight_decay", 1e-3),
    # Compute budget / economics.
    ("target_train_hours", 6.0),
    ("ref_throughput_tokens_per_s", 90_000),
    ("max_train_seconds", 999),
    # Corpus feed + submission surface.
    ("corpus_mode", "cache_reuse"),
    ("accepted_fields", ("mask",)),
])
def test_trainer_side_change_scores_on_an_un_upgraded_validator(armed, field, value):
    # The trainer moves; the validator's chain.toml does NOT. Before DEC-CA-0036
    # this was contract_digest_mismatch on every round until all six external
    # validators restarted.
    moved = replace(armed.training, **{field: value})
    assert contract_digest(moved) != contract_digest(armed.training), (
        f"{field} must be digest-bound for this test to mean anything"
    )
    assert _gate(armed, _manifest(armed, trained_under=moved)) is None


def test_declared_body_travels_through_sign_and_parse(cfg):
    # The body has to survive the wire, or a validator recomputes a different
    # canonical_body and the signature check fails before the gate ever runs.
    m = _manifest(cfg)
    round_tripped = load_manifest(dump_manifest(m))
    # JSON has no tuples, so ``extra_sizes`` comes back a list. What must survive
    # is the SIGNED bytes and the digest — contract_payload is idempotent across
    # that round trip precisely so the re-hash below holds.
    assert round_tripped.canonical_body() == m.canonical_body()
    assert contract_digest(round_tripped.contract_body) == round_tripped.contract_digest
    assert locked_contract_terms(round_tripped.contract_body) == \
        locked_contract_terms(cfg.training)


def test_absent_body_leaves_canonical_bytes_untouched(cfg):
    # Drop-when-unset: a manifest that publishes no body must hash exactly as it
    # did before the field existed, so archived signatures stay valid.
    with_body = _manifest(cfg)
    without = replace(with_body, contract_body=None)
    assert b"contract_body" not in without.canonical_body()
    assert b"contract_body" in with_body.canonical_body()


# ── what still gates ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("base_arch_digest", "f" * 64),
    ("arch_preset", "toto2-22m"),
    ("expected_gpu", "NVIDIA H100 80GB HBM3"),
    # The fifth locked term (the read the original draft missed): the receipt
    # assembler derives the round's published training_seed from the LOCAL
    # salt, so a declared salt change would break audit replay.
    ("train_seed_salt", 4242),
])
def test_locked_terms_still_reject(armed, field, value):
    moved = replace(armed.training, **{field: value})
    reason = _gate(armed, _manifest(armed, trained_under=moved))
    assert reason is not None
    # base_arch_digest has its own dedicated gate that fires first; the others
    # are caught by the locked projection.
    assert "contract_locked_mismatch" in reason or "base_arch_digest_mismatch" in reason


def test_body_that_disagrees_with_its_own_digest_is_rejected(armed):
    # A trainer cannot declare one contract and publish another: the body is
    # what the round is audited against, so it must re-hash to the digest.
    m = _manifest(armed)
    lying = replace(m, contract_body={**m.contract_body, "base_lr": 1.0})
    reason = _gate(armed, lying)
    assert reason is not None and "contract_body_mismatch" in reason


def test_per_size_gpu_pin_is_locked_but_size_recipe_is_not(armed):
    base = _size("toto2-22m", expected_gpu="L40S")
    pinned = replace(armed.training, extra_sizes=(base,))
    armed = replace(armed, training=pinned)

    # Same size, same GPU pin, different width ⇒ declared, not gated.
    recipe = replace(pinned, extra_sizes=(replace(base, d_model=768),))
    assert _gate(armed, _manifest(armed, trained_under=recipe)) is None

    # Same size, DIFFERENT GPU pin ⇒ locked, rejected.
    silicon = replace(pinned, extra_sizes=(replace(base, expected_gpu="H100"),))
    reason = _gate(armed, _manifest(armed, trained_under=silicon))
    assert reason is not None and "contract_locked_mismatch" in reason


def test_extra_size_declaration_order_does_not_matter(armed):
    a = _size("toto2-22m")
    b = _size("toto2-113m", d_ff=1024)
    assert (locked_contract_terms(replace(armed.training, extra_sizes=(a, b)))
            == locked_contract_terms(replace(armed.training, extra_sizes=(b, a))))


# ── the block gate itself ────────────────────────────────────────────────────

def test_gate_off_keeps_strict_equality(cfg):
    # declared_contract_from_block = 0 (the shipped mainnet value): a moved
    # contract must still be rejected, body or no body.
    moved = replace(cfg.training, train_image_digest="sha256:" + "c" * 64)
    reason = _gate(cfg, _manifest(cfg, trained_under=moved))
    assert reason is not None and "contract_digest_mismatch" in reason


def test_rounds_before_the_block_replay_strict(armed):
    moved = replace(armed.training, train_image_digest="sha256:" + "c" * 64)
    early = _manifest(armed, trained_under=moved, block=EARLY_BLOCK)
    reason = _gate(armed, early)
    assert reason is not None and "contract_digest_mismatch" in reason


def test_body_less_manifest_falls_back_to_strict(armed):
    # A trainer that predates the field cannot be waved through just because the
    # block gate is armed — there is nothing to verify against.
    moved = replace(armed.training, train_image_digest="sha256:" + "c" * 64)
    reason = _gate(armed, _manifest(armed, trained_under=moved, body=None))
    assert reason is not None and "contract_digest_mismatch" in reason

    # ...and an unmoved contract still passes on that path.
    assert _gate(armed, _manifest(armed, body=None)) is None


# ── the loader (the landed-without-parsing defect class — see OPSLOG) ────────

def test_declared_contract_from_block_loader_round_trip(tmp_path, cfg):
    # A knob that exists on the dataclass but is not parsed by
    # load_chain_config arms as a silent no-op — this bit three PRs in one
    # stack. Round-trip the repo's own chain.toml with the key injected.
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "chain.toml"
    text = src.read_text()
    assert "declared_contract_from_block" in text, "chain.toml must document the key"
    text = re.sub(r"^declared_contract_from_block\s*=.*$",
                  "declared_contract_from_block = 123456",
                  text, count=1, flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    loaded = load_chain_config(p)
    assert loaded.scoring.declared_contract_from_block == 123456


# ── audit ────────────────────────────────────────────────────────────────────

def test_audit_reports_declared_drift_without_failing(armed):
    from cascade.audit.checks import check_contract_declaration, check_contract_digest

    moved = replace(armed.training, train_image_digest="sha256:" + "c" * 64,
                    lr_schedule="warmup_cosine")
    m = _manifest(armed, trained_under=moved)

    class _Receipt:
        epoch_start_block = EPOCH_BLOCK
        manifest = {"contract_digest": m.contract_digest,
                    "contract_body": m.contract_body}

    assert check_contract_digest(_Receipt(), armed).status == "PASS"
    drift = check_contract_declaration(_Receipt(), armed)
    # Recorded, not gated — but never silent.
    assert drift.status == "WARN"
    assert "train_image_digest" in drift.detail and "lr_schedule" in drift.detail


def test_audit_replays_a_round_against_its_own_contract(armed):
    """The provenance fix: re-pinning chain.toml must not change the verdict on
    an ALREADY-published round. Under the old local-config comparison it did."""
    from cascade.audit.checks import check_contract_digest

    m = _manifest(armed)                       # trained under today's contract

    class _Receipt:
        epoch_start_block = EPOCH_BLOCK
        manifest = {"contract_digest": m.contract_digest,
                    "contract_body": m.contract_body}

    # Now the operator re-pins the image locally, as a routine worker rebuild.
    repinned = replace(armed, training=replace(armed.training,
                                               train_image_digest="sha256:" + "d" * 64))
    assert check_contract_digest(_Receipt(), repinned).status == "PASS"


def test_tier2_replay_contract_comes_from_the_body(armed):
    """contract_for_replay reconstructs the round's own contract — including
    sizes and image pin — and falls back to local config outside the regime."""
    from cascade.audit.rederive import contract_for_replay

    # ema_decay 0.0 ⇒ digest-dropped from the published body (the round
    # predates the EMA arming, whatever the auditor's file says today).
    trained = replace(
        armed.training,
        train_image_digest="sha256:" + "c" * 64,
        base_lr=8e-3,
        ema_decay=0.0,
        extra_sizes=(_size("toto2-22m", expected_gpu="L40S"),),
    )
    m = _manifest(armed, trained_under=trained)

    class _Receipt:
        epoch_start_block = EPOCH_BLOCK
        manifest = {"contract_digest": m.contract_digest,
                    "contract_body": m.contract_body}

    replay = contract_for_replay(_Receipt(), armed)
    assert replay.train_image_digest == trained.train_image_digest
    assert replay.base_lr == trained.base_lr
    assert [s.arch_preset for s in replay.extra_sizes] == ["toto2-22m"]
    assert contract_digest(replay) == m.contract_digest
    # A digest-dropped key absent from the body reads as the INERT default,
    # even when the auditor's local config has armed it (as the shipped
    # 2026-08-27 file does: ema_decay 0.999).
    assert armed.training.ema_decay != 0.0
    assert replay.ema_decay == 0.0

    # Outside the declared regime the local contract is the authority.
    class _EarlyReceipt:
        epoch_start_block = EARLY_BLOCK
        manifest = dict(_Receipt.manifest)

    assert contract_for_replay(_EarlyReceipt(), armed) is armed.training
