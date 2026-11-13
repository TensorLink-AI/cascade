"""DEC-CA-0067 after the merge with main's money-safety releases (#417–#419):

* pre-switch byte identity: a manifest built with a switch SCHEDULED for a
  later era is byte-identical (body and signature) to one built with no
  switch at all, round-trips through the parser unchanged, and the
  validator stamps no ``image_decoupled_block`` on receipts before lock-in
  (the frozen ``round_receipt_v3.json`` golden still holds);
* the note byte budget: both owner options — (a) the typed forfeit /
  margin-v2 blocks, (b) retiring the UNLOCKED pre-launch burn vote — the
  guard that refuses to retire a LOCKED one, and the startup WARNING for any
  segment the 128-byte field would drop;
* the money gates read the leg's era pins (``leg_round``) before the corpus;
* an ssh failure (the fingerprint probe included) is infra: never a miner
  fault, never an image refusal, never a lane fault.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.era import era_length_blocks, image_decoupled_era

from .test_image_decoupled_contract import (
    REPO,
    _leg,
    _live_records,
    _trainer,
    _untyped_blocks,
)

V2B = "v2b-69630f92"


# ── pre-switch byte identity (manifests, signatures, receipts) ───────────────


def test_pre_switch_manifest_and_signature_are_byte_identical_with_a_later_switch(cfg, tmp_path):
    from cascade.shared.manifest import dump_manifest, load_manifest, sign_manifest

    from .test_manifest_signing import _FakeWallet

    sched, era, contract, start = _trainer(cfg, tmp_path, on=False)
    plain = sched._build_manifest(era, _leg(contract, start), [], "50000", start,
                                  pin_block=start)
    later = start + era_length_blocks(sched.cfg.round, start)
    sched.cfg = replace(sched.cfg, v2=replace(sched.cfg.v2, image_decoupled_from_block=later))
    assert not image_decoupled_era(sched.cfg, start) and image_decoupled_era(sched.cfg, later)
    armed = sched._build_manifest(era, _leg(contract, start), [], "50000", start,
                                  pin_block=start)
    assert armed.canonical_body() == plain.canonical_body()
    assert b"numerics_contract_digest" not in armed.canonical_body()
    s_plain = sign_manifest(plain, _FakeWallet())
    s_armed = sign_manifest(armed, _FakeWallet())
    assert s_armed.signature == s_plain.signature
    back = load_manifest(dump_manifest(s_armed))
    # what an OLD validator re-serialises (verify_signature) is the signed body
    assert back.canonical_body() == s_armed.canonical_body() == plain.canonical_body()


def test_pre_lock_receipts_carry_no_switch_and_match_the_v3_golden():
    from cascade.shared.receipt import dump_receipt
    from cascade.validator.loop import ValidatorRunner

    from .receipt_fixture import make_scored_receipt

    release = load_chain_config(REPO / "chain.toml")
    untyped = replace(release, v2=replace(release.v2, image_decoupled_from_block=0))
    runner = ValidatorRunner(cfg=untyped)
    assert runner.image_decoupled_block == 0                    # nothing locked, nothing typed
    receipt, _, _ = make_scored_receipt()
    stamped = replace(receipt, image_decoupled_block=runner.image_decoupled_block)
    golden = (REPO / "tests" / "fixtures" / "round_receipt_v3.json").read_text(encoding="utf-8")
    assert dump_receipt(stamped) == golden
    assert b"image_decoupled_block" not in stamped.canonical_body()


# ── the note byte budget: option (a) typed blocks, option (b) retire prelaunch ──


def _retired(cfg):
    """Option (b): ``[v2] prelaunch_burn = 0`` (the unlocked vote retired)."""
    return replace(cfg, v2=replace(cfg.v2, prelaunch_burn=0.0))


def test_option_b_drops_the_prelaunch_segment_and_option_a_still_needed(tmp_path):
    rec, fr, mr = _live_records()
    untyped = _untyped_blocks(tmp_path)
    assert A.prelaunch_burn_feature(untyped) is not None
    assert A.prelaunch_burn_feature(_retired(untyped)) is None
    note = A.own_signal_payload(_retired(untyped), rec, fr, mr, A.ActivationRecord())
    assert "prelaunch-" not in note and len(note.encode()) <= A.NOTE_MAX_BYTES
    # (b) ALONE does not fit a locked v2b beside the untyped forfeit / margin
    # segments: the validator still refuses (mainnet needs option (a))
    why = A.v2b_note_problem(_retired(untyped), rec, fr, mr, head_block=9_420_000)
    assert "TRIM" in why and "prelaunch_burn = 0" in why
    # (a) alone fits (the release chain.toml), and (a)+(b) fits with more room
    release = load_chain_config(REPO / "chain.toml")
    assert A.v2b_note_problem(release, rec, head_block=9_420_000) == ""
    assert A.v2b_note_problem(_retired(release), rec, head_block=9_420_000) == ""
    open_vote = replace(release, v2=replace(release.v2, image_decoupled_from_block=0))
    both = A.own_signal_payload(_retired(open_vote), rec, None, None, A.ActivationRecord(), None,
                                None, A.ActivationRecord(feature=V2B, lock_block=9504000,
                                                         activation_block=9507600,
                                                         source="tally"))
    assert f":{V2B}:9504000:9507600:" in both + ":" and "prelaunch-" not in both


def test_retiring_a_locked_prelaunch_vote_is_refused():
    release = load_chain_config(REPO / "chain.toml")
    locked = A.ActivationRecord(feature="prelaunch-95-r2", lock_block=9400000,
                                activation_block=9400900, source="tally")
    assert A.prelaunch_retirement_problem(_retired(release), A.ActivationRecord()) == ""
    assert A.prelaunch_retirement_problem(_retired(release), locked).startswith(
        "[v2] prelaunch_burn = 0")
    assert A.prelaunch_retirement_problem(release, locked) == ""       # not retired


def _startup(cfg, tmp_path, prelaunch: A.ActivationRecord | None = None):
    from cascade.validator.loop import ValidatorRunner

    rec, fr, mr = _live_records()
    A.ActivationStore(tmp_path / "act.json").save(rec)
    A.ActivationStore(tmp_path / "activation_forfeit_state.json").save(fr)
    A.ActivationStore(tmp_path / "activation_margin_v2_state.json").save(mr)
    if prelaunch is not None:
        A.ActivationStore(tmp_path / "activation_prelaunch_burn_state.json").save(prelaunch)
    runner = ValidatorRunner(cfg=cfg, activation_store=A.ActivationStore(tmp_path / "act.json"))

    def offline():
        raise OSError("chain offline")

    runner._activation_startup(SimpleNamespace(current_block=offline, hotkey_ss58=lambda: "v1"))
    return runner


def test_validator_refuses_to_start_when_a_locked_prelaunch_vote_is_retired(tmp_path):
    cfg = _retired(load_chain_config(REPO / "chain.toml"))
    locked = A.ActivationRecord(feature="prelaunch-95-r2", lock_block=9400000,
                                activation_block=9400900, source="tally")
    with pytest.raises(RuntimeError, match="refusing to start.*prelaunch_burn = 0"):
        _startup(cfg, tmp_path, locked)


def test_validator_starts_with_an_unlocked_prelaunch_vote_retired(tmp_path):
    unlocked = A.ActivationRecord(feature="prelaunch-95-r2", last_checked_boundary=9423000)
    _startup(_retired(load_chain_config(REPO / "chain.toml")), tmp_path, unlocked)


def test_startup_warns_for_every_segment_the_note_would_trim(tmp_path, caplog):
    # v2b typed (so the v2b refusal is not what fires) on a deployment that
    # did not type the forfeit / margin-v2 blocks: the prelaunch vote drops
    cfg = _untyped_blocks(tmp_path)
    cfg = replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=9504000))
    rec, fr, mr = _live_records()
    assert A.v2b_note_problem(cfg, rec, fr, mr) == ""
    # (the newer DEC-CA-0068 "lad-" segment goes first; the prelaunch one follows)
    assert "prelaunch-95-r2" in A.note_trimmed_segments(cfg, rec, fr, mr)
    with caplog.at_level(logging.WARNING, logger="cascade.validator"):
        _startup(cfg, tmp_path)
    assert any("STARTUP" in r.getMessage() and "'prelaunch-95-r2'" in r.getMessage()
               for r in caplog.records)
    # the release config trims nothing: no startup warning
    release = load_chain_config(REPO / "chain.toml")
    assert A.note_trimmed_segments(release, rec) == []


def test_note_parts_refactor_keeps_the_note_bytes(tmp_path):
    """own_signal_payload is built from _own_note_parts: same note as the
    trim of the parts, for every shape the tests above use."""
    rec, fr, mr = _live_records()
    for cfg in (_untyped_blocks(tmp_path), load_chain_config(REPO / "chain.toml")):
        for rf, rm in ((fr, mr), (None, None)):
            note = A.own_signal_payload(cfg, rec, rf, rm, A.ActivationRecord())
            head, extras, tail = A._own_note_parts(cfg, rec, rf, rm, A.ActivationRecord())
            dropped = A.note_trimmed_segments(cfg, rec, rf, rm, A.ActivationRecord())
            kept = [s for s in extras if s.feature not in dropped]
            assert note == A.format_signals(head, *kept, *tail)


# ── merge overlap: the money gates read the leg's era pins ───────────────────


def test_pre_corpus_verdict_reads_the_legs_era_pins(monkeypatch):
    from cascade.funding.queue import OPERATOR_BLOCKED_CLASS
    from cascade.trainer import leg_pins, money_guard

    def unqualified(runner):
        raise leg_pins.ImageAfterUnqualified("after image not qualified (rent): no record")

    monkeypatch.setattr(leg_pins, "leg_round", unqualified)
    runner, gen = SimpleNamespace(cfg=None), SimpleNamespace(hotkey="hk")
    why, cls = money_guard.would_rent_refusal(runner, gen, operator=True)
    assert cls == OPERATOR_BLOCKED_CLASS and "not qualified" in why
    # a post-switch leg's wall gate is judged on the AFTER pins it will rent
    after = SimpleNamespace(funded_sku_wall_measured_image="sha256:" + "c" * 64,
                            funded_pod_image="ghcr.io/x/w@sha256:" + "d" * 64)
    monkeypatch.setattr(leg_pins, "leg_round", lambda runner: after)
    why, cls = money_guard.would_rent_refusal(runner, gen, operator=False)
    assert cls == OPERATOR_BLOCKED_CLASS and "measured on image" in why


# ── finding 4 under #419's fault classification ──────────────────────────────


@pytest.mark.parametrize("msg", ["fingerprint probe: ssh: connect: Connection timed out",
                                 "remote final on lane failed (rc=255) ssh transport"])
def test_ssh_failure_is_infra_never_miner_never_an_image_refusal(msg):
    from cascade.trainer.faults import _lane_fault, _transport_failure, classify_leg_exception
    from cascade.trainer.leg_pins import LegImageMismatch, worker_refused_image
    from cascade.trainer.remote import RemoteDispatchError

    err = RemoteDispatchError(msg, returncode=255)
    assert _transport_failure(err) and not _lane_fault(err)    # the dead-pod drain owns it
    assert not worker_refused_image(err) and not isinstance(err, LegImageMismatch)
    miner_fault, error_class, _ = classify_leg_exception(
        SimpleNamespace(), SimpleNamespace(hotkey="hk", ref="r"), SimpleNamespace(base_seed=1),
        err, rc=255, text=str(err), stalled_before=False, shards_leg=False)
    # infra: the payer leg's generic handler never flips it operator-billed
    # (#419 — only an operator-class verdict is a make-good)
    assert (miner_fault, error_class) == (False, "infra")
