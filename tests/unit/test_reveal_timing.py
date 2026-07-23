"""Timed-reveal protection: the deploy-side reveal delay must land a submission's
timelock reveal strictly INSIDE its round's window — after the commit, before the
epoch-boundary cutoff (eligibility gates on the reveal block, so a reveal at/after
the boundary silently costs the miner the round) — and the trainer's same-ref
dedup must award a duplicated ref to the earliest reveal, not the lowest UID."""

from __future__ import annotations

import argparse

import pytest

from cascade.shared.chain import blocks_until_boundary_reveal
from cascade.trainer.loop import ResolvedGenerator, plan_round

EPOCH = 7200
MARGIN = 25


def _boundary_after(block: int) -> int:
    return (block // EPOCH + 1) * EPOCH


# ── blocks_until_boundary_reveal ──────────────────────────────────────────────


@pytest.mark.parametrize("current", [0, 1, 3600, EPOCH - MARGIN - 1, EPOCH, EPOCH + 5000])
def test_timed_reveal_lands_inside_the_window(current):
    delay = blocks_until_boundary_reveal(current, EPOCH, MARGIN)
    target = current + delay
    boundary = _boundary_after(current)
    assert delay >= 1                      # strictly after the commit
    assert target == boundary - MARGIN     # exactly the margin before the cutoff
    assert target < boundary               # eligible: reveal strictly pre-boundary


def test_inside_the_margin_floors_to_reveal_now():
    # Committing within the margin (or in the sliver past the target) must not
    # compute a zero/negative delay — reveal now; residual exposure < margin.
    boundary = EPOCH
    for current in (boundary - MARGIN, boundary - 5, boundary - 1):
        assert blocks_until_boundary_reveal(current, EPOCH, MARGIN) == 1


def test_commit_exactly_at_a_boundary_targets_the_next_one():
    delay = blocks_until_boundary_reveal(EPOCH, EPOCH, MARGIN)
    assert EPOCH + delay == 2 * EPOCH - MARGIN


def test_next_epoch_skips_the_imminent_boundary():
    current = 100
    delay = blocks_until_boundary_reveal(current, EPOCH, MARGIN, next_epoch=True)
    assert current + delay == 2 * EPOCH - MARGIN


def test_zero_margin_targets_one_block_before_the_boundary_is_not_allowed():
    # margin 0 targets the boundary itself — the reveal would be INELIGIBLE
    # (cutoff is strict); the function permits margin_blocks=0 arithmetically
    # but the target then equals the boundary, which the caller must not want.
    # Document the arithmetic so the default margin stays the protection.
    delay = blocks_until_boundary_reveal(10, EPOCH, 0)
    assert 10 + delay == EPOCH  # == boundary ⇒ would miss the round


@pytest.mark.parametrize(
    ("epoch", "margin", "min_blocks"),
    [(0, 0, 1), (100, 100, 1), (100, -1, 1), (100, 10, 0)],
)
def test_invalid_config_is_rejected(epoch, margin, min_blocks):
    with pytest.raises(ValueError):
        blocks_until_boundary_reveal(50, epoch, margin, min_blocks=min_blocks)


# ── deploy CLI resolution ─────────────────────────────────────────────────────


def _deploy_args(**over):
    base = {"blocks_until_reveal": None, "reveal_now": False, "next_epoch": False}
    base.update(over)
    return argparse.Namespace(**base)


def test_cli_default_is_the_timed_reveal(cfg, capsys):
    from cascade.miner.cli import _resolve_blocks_until_reveal

    current = 1000
    delay = _resolve_blocks_until_reveal(_deploy_args(), cfg, current)
    epoch, margin = cfg.round.epoch_blocks, cfg.round.reveal_margin_blocks
    assert current + delay == (current // epoch + 1) * epoch - margin
    assert "timed reveal" in capsys.readouterr().out


def test_cli_explicit_and_reveal_now_override(cfg):
    from cascade.miner.cli import _resolve_blocks_until_reveal

    assert _resolve_blocks_until_reveal(_deploy_args(blocks_until_reveal=7), cfg, 1000) == 7
    assert _resolve_blocks_until_reveal(_deploy_args(reveal_now=True), cfg, 1000) == 1


def test_cli_conflicting_reveal_flags_are_rejected(tmp_path):
    from cascade.miner import cli as cli_mod

    common = ["deploy", str(tmp_path), "--wallet-name", "w", "--wallet-hotkey", "h"]
    assert cli_mod.main([*common, "--reveal-now", "--next-epoch"]) == 2
    assert cli_mod.main([*common, "--reveal-now", "--blocks-until-reveal", "5"]) == 2
    assert cli_mod.main([*common, "--next-epoch", "--blocks-until-reveal", "5"]) == 2
    assert cli_mod.main([*common, "--hub-repo", "a/b", "--hub-namespace", "a"]) == 2


def test_fresh_hub_repo_is_namespaced_and_unpredictable():
    from cascade.miner.cli import _fresh_hub_repo

    r1, r2 = _fresh_hub_repo("acct"), _fresh_hub_repo("acct")
    assert r1.startswith("acct/gen-") and r2.startswith("acct/gen-")
    assert r1 != r2


# ── same-ref dedup: earliest reveal wins ─────────────────────────────────────

REF_X = "alice/gen-x@sha256:" + "a" * 64
REF_Y = "bob/gen-y@sha256:" + "b" * 64


def _rg(hotkey, uid, ref, reveal_block):
    return ResolvedGenerator(hotkey=hotkey, uid=uid, ref=ref, reveal_block=reveal_block)


def test_duplicated_ref_goes_to_the_earliest_reveal_not_the_lowest_uid():
    original = _rg("alice", uid=9, ref=REF_X, reveal_block=100)
    copier = _rg("mallory", uid=1, ref=REF_X, reveal_block=200)  # lower UID, later reveal
    king = _rg("king", uid=0, ref=REF_Y, reveal_block=50)
    plan = plan_round([king, original, copier], "king")
    assert [c.hotkey for c in plan.challengers] == ["alice"]


def test_duplicated_ref_ties_break_by_uid():
    a = _rg("a", uid=3, ref=REF_X, reveal_block=100)
    b = _rg("b", uid=2, ref=REF_X, reveal_block=100)
    king = _rg("king", uid=0, ref=REF_Y, reveal_block=50)
    plan = plan_round([king, a, b], "king")
    assert [c.hotkey for c in plan.challengers] == ["b"]


def test_config_carries_the_reveal_margin(cfg):
    assert cfg.round.reveal_margin_blocks == 25


# ── reveal-status verdicts ───────────────────────────────────────────────────


def test_reveal_verdict_on_time_is_eligible_and_quiet():
    from cascade.miner.cli import _reveal_verdict

    # revealed exactly at boundary − margin, boundary not yet passed
    missed, report = _reveal_verdict(
        reveal_block=EPOCH - MARGIN, current_block=EPOCH - 10,
        epoch_blocks=EPOCH, margin_blocks=MARGIN, expect_boundary=EPOCH,
    )
    assert not missed
    assert f"locking at block {EPOCH}" in report
    assert "MISSED" not in report and "exceeds" not in report


def test_reveal_verdict_flags_a_missed_boundary_loudly():
    from cascade.miner.cli import _reveal_verdict

    # targeted the boundary at EPOCH but the reveal jittered past it
    missed, report = _reveal_verdict(
        reveal_block=EPOCH + 2, current_block=EPOCH + 40,
        epoch_blocks=EPOCH, margin_blocks=MARGIN, expect_boundary=EPOCH,
    )
    assert missed
    assert "MISSED" in report
    assert f"locking at block {2 * EPOCH}" in report      # auto-rolls to next round
    assert "one-submission budget" in report               # miss ≠ burned submission


def test_reveal_verdict_notes_exposure_beyond_the_margin():
    from cascade.miner.cli import _reveal_verdict

    # a --reveal-now style early reveal: public far longer than the margin
    missed, report = _reveal_verdict(
        reveal_block=100, current_block=200,
        epoch_blocks=EPOCH, margin_blocks=MARGIN,
    )
    assert not missed and "exceeds" in report


def test_pending_timelock_parses_commitment_of_record():
    """Live shape 2026-07-15: CommitmentOf → {deposit, block, info.fields:
    [{TimelockEncrypted: {encrypted, reveal_round}}]}. While pending, the
    revealed store still shows the PREVIOUS submission — reveal-status must
    surface the pending commit instead of gaslighting the miner."""
    from types import SimpleNamespace

    from cascade.miner.cli import _pending_timelock

    record = {"deposit": 0, "block": 7561939,
              "info": {"fields": [{"TimelockEncrypted":
                                   {"encrypted": "0xabcd", "reveal_round": 19938211}}]}}

    class _Substrate:
        def query(self, module, storage_function, params):
            assert (module, storage_function) == ("Commitments", "CommitmentOf")
            return SimpleNamespace(value=record)

    client = SimpleNamespace(netuid=259, subtensor=lambda: SimpleNamespace(substrate=_Substrate()))
    assert _pending_timelock(client, "5FBi...") == (7561939, 19938211)

    # no pending record (post-reveal): None, quietly
    class _Empty:
        def query(self, *a, **k):
            return SimpleNamespace(value=None)

    client = SimpleNamespace(netuid=259, subtensor=lambda: SimpleNamespace(substrate=_Empty()))
    assert _pending_timelock(client, "5FBi...") is None
