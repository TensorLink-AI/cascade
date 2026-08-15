"""Tests for scripts/replay_margin_decay.py — the margin-decay replay harness.

The harness re-judges signed receipts under candidate tenure-decay schedules
using only recorded fields (params, tenure, LCBs) — no bootstrap re-run — so
these tests build receipts with hand-picked verdicts and check the flip logic,
the consistency gate, and the floor guardrail.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cascade.eval.koth import KothParams
from cascade.shared.receipt import (
    VerdictRecord,
    build_receipt,
    dump_receipt,
    load_receipt,
)
from tests.unit.receipt_fixture import (
    BLOCK_HASH,
    EPOCH_START,
    make_manifest,
    make_scored_receipt,
)

_SPEC = importlib.util.spec_from_file_location(
    "replay_margin_decay",
    Path(__file__).resolve().parents[2] / "scripts" / "replay_margin_decay.py",
)
replay = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = replay  # dataclasses resolve types via sys.modules
_SPEC.loader.exec_module(replay)

PARAMS = KothParams(
    win_margin_start=0.02, win_margin_end=0.02, margin_warmup_rounds=0,
    min_windows=200, bootstrap_B=500, bootstrap_alpha=0.05, dethrone_cp=1,
)


class _Salt:
    train_seed_salt = 1337


def _seeds():
    from cascade.shared.chain import seed_from_block_hash
    from cascade.trainer.contract import RoundSeeds

    base = seed_from_block_hash(BLOCK_HASH)
    return base, RoundSeeds.derive(base, _Salt())


def make_receipt_with_verdict(
    *, lcb: float | None, margin: float, tenure: int, won: bool,
    inconclusive: bool = False, round_id: str = "1", cohort_k: int = 0,
    cohort_lcbs: dict | None = None, params: KothParams = PARAMS,
    validator_hotkey: str = "",
):
    base_seed, seeds = _seeds()
    manifest = make_manifest(None, base_seed=base_seed)
    verdict = VerdictRecord(
        params=dict(dataclasses.asdict(params)),
        bootstrap_seed=str(base_seed),
        king_tenure_rounds=tenure,
        lcb=lcb,
        margin=margin,
        challenger_wins_round=won,
        inconclusive=inconclusive,
        n_windows=256,
        king_geomean=1.0,
        chal_geomean=0.9,
        gift_lcb=None,
        gift_gate_passed=None,
        dethroned=won,
        note="test",
        king_hotkey="chal_hk" if won else "king_hk",
        king_uid=1 if won else 0,
        cohort_k=cohort_k,
        cohort_lcbs=cohort_lcbs,
    )
    receipt = build_receipt(
        round_id=round_id, status="scored", epoch_start_block=EPOCH_START,
        epoch_block_hash=BLOCK_HASH, base_seed=base_seed, seeds=seeds,
        manifest=manifest, verdict=verdict, validator_hotkey=validator_hotkey,
    )
    # Round-trip through JSON so the tests exercise exactly what the harness
    # reads off disk.
    return load_receipt(dump_receipt(receipt))


def test_loss_flips_to_win_once_decay_undercuts_lcb():
    # Flat 2% margin held this challenger out at lcb=1.2%; a floor of 0.8%
    # reached by tenure 8 lets it through at tenure 9.
    r = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False)
    row = replay.replay_receipt(r, replay.Schedule(start=0.02, end=0.008, warmup=8))
    assert row is not None and row.flipped and row.won_candidate
    assert row.margin_candidate == pytest.approx(0.008)


def test_slow_warmup_makes_it_a_near_miss_not_a_flip():
    # Same round, warmup 16: at tenure 9 the margin is 0.02 + 9/16·(0.008−0.02)
    # = 0.01325 > lcb — no flip, but within the near-miss band.
    r = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False)
    row = replay.replay_receipt(r, replay.Schedule(start=0.02, end=0.008, warmup=16))
    assert row is not None and not row.flipped
    assert row.margin_candidate == pytest.approx(0.01325)
    assert row.near_miss


def test_decay_never_revokes_a_recorded_win():
    r = make_receipt_with_verdict(lcb=0.05, margin=0.02, tenure=3, won=True)
    for end in (0.005, 0.008, 0.010):
        for warmup in (8, 12, 16):
            row = replay.replay_receipt(r, replay.Schedule(0.02, end, warmup))
            assert row is not None and row.won_candidate and not row.flipped


def test_fresh_king_margin_unchanged_at_tenure_zero():
    r = make_receipt_with_verdict(lcb=0.015, margin=0.02, tenure=0, won=False)
    row = replay.replay_receipt(r, replay.Schedule(start=0.02, end=0.005, warmup=8))
    assert row is not None
    assert row.margin_candidate == pytest.approx(0.02)
    assert not row.flipped


def test_inconclusive_and_rejected_rounds_are_skipped():
    r = make_receipt_with_verdict(lcb=None, margin=0.02, tenure=4, won=False,
                                  inconclusive=True)
    assert replay.replay_receipt(r, replay.Schedule(0.02, 0.008, 8)) is None


def test_cohort_round_reports_new_clearers():
    # DEC-CA-0012 cohort: recorded LCBs are already at alpha/k; the margin is
    # the same bar for every member. Decay lets "a" (0.9%) through, not "b".
    r = make_receipt_with_verdict(
        lcb=0.009, margin=0.02, tenure=10, won=False,
        cohort_k=2, cohort_lcbs={"a": 0.009, "b": 0.001},
    )
    row = replay.replay_receipt(r, replay.Schedule(start=0.02, end=0.008, warmup=8))
    assert row is not None and row.flipped and row.won_candidate
    assert row.new_clearers == ("a",)


def test_margin_consistency_gate_rejects_tampered_receipts():
    # Recorded margin that the recorded params + tenure cannot reproduce.
    r = make_receipt_with_verdict(lcb=0.012, margin=0.017, tenure=9, won=False)
    with pytest.raises(ValueError, match="not replayable"):
        replay.replay_receipt(r, replay.Schedule(0.02, 0.008, 8))


def test_floor_guardrail_refuses_zero_or_negative_end():
    with pytest.raises(SystemExit, match="guardrail"):
        replay.build_schedules(0.02, [0.0], [8])
    with pytest.raises(SystemExit, match="guardrail"):
        replay.build_schedules(0.02, [-0.005], [8])


def test_ramp_schedules_are_refused():
    with pytest.raises(SystemExit, match="decay"):
        replay.build_schedules(0.02, [0.03], [8])


def test_grid_report_over_mixed_rounds(tmp_path):
    rounds = [
        make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False,
                                  round_id="10"),
        make_receipt_with_verdict(lcb=0.05, margin=0.02, tenure=0, won=True,
                                  round_id="11"),
        make_receipt_with_verdict(lcb=None, margin=0.02, tenure=1, won=False,
                                  inconclusive=True, round_id="12"),
    ]
    schedules = replay.build_schedules(0.02, [0.008], [8, 16])
    report = replay.replay_all(rounds, schedules)
    fast = report["start=0.02 end=0.008 warmup=8"]
    slow = report["start=0.02 end=0.008 warmup=16"]
    assert fast["n_decided"] == 2 and fast["n_flips"] == 1
    assert fast["n_flips_win_to_loss"] == 0
    assert slow["n_flips"] == 0 and slow["n_near_misses"] == 1


def test_load_receipts_from_disk_dedupes_and_reports_problems(tmp_path):
    r = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False,
                                  round_id="7")
    (tmp_path / "round-7.json").write_text(dump_receipt(r), encoding="utf-8")
    sub = tmp_path / "dup"
    sub.mkdir()
    (sub / "round-7.json").write_text(dump_receipt(r), encoding="utf-8")
    (tmp_path / "round-8.json").write_text("{not json", encoding="utf-8")
    receipts, problems = replay.load_receipts([tmp_path])
    assert [x.round_id for x in receipts] == ["7"]
    assert len(problems) == 1 and "round-8" in problems[0]


def test_multi_validator_trees_conflict_unless_filtered(tmp_path):
    # A cache holding two validators' receipts for the SAME round must not be
    # replayed on path-sort order — around a scoring-rule seam the two can
    # record opposite verdicts. Unfiltered: the round is dropped with a loud
    # problem; --validator-hotkey selects one trail.
    r_a = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False,
                                    round_id="7", validator_hotkey="hkA")
    r_b = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=True,
                                    round_id="7", validator_hotkey="hkB")
    for hk, r in (("hkA", r_a), ("hkB", r_b)):
        d = tmp_path / hk
        d.mkdir()
        (d / "round-7.json").write_text(dump_receipt(r), encoding="utf-8")
    receipts, problems = replay.load_receipts([tmp_path])
    assert receipts == []
    assert any("multiple validators" in p and "hkA" in p and "hkB" in p
               for p in problems)
    only_a, problems_a = replay.load_receipts([tmp_path], filter_hotkey="hkA")
    assert [r.validator_hotkey for r in only_a] == ["hkA"] and problems_a == []


def test_fixture_receipt_replays_cleanly():
    # The repo's real fixture (a genuine evaluate_round verdict) goes through
    # the whole harness path without tripping the consistency gate.
    receipt, _, _ = make_scored_receipt(None)
    receipt = load_receipt(dump_receipt(receipt))
    row = replay.replay_receipt(receipt, replay.Schedule(0.02, 0.008, 12))
    assert row is not None and row.won_recorded and row.won_candidate


def test_cli_end_to_end(tmp_path, capsys):
    d = tmp_path / "receipts"
    d.mkdir()
    r = make_receipt_with_verdict(lcb=0.012, margin=0.02, tenure=9, won=False,
                                  round_id="42")
    (d / "round-42.json").write_text(dump_receipt(r), encoding="utf-8")
    out = tmp_path / "report.json"
    rc = replay.main([str(d), "--ends", "0.008", "--warmups", "8", "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "FLIP round=42" in text
    assert "first-order replay" in text
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["start=0.02 end=0.008 warmup=8"]["n_flips"] == 1
