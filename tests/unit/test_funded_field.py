"""[round] funded_mode: field selection, settle marking, and the cadence floor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue
from cascade.shared.config import RoundConfig, validate_funded_mode
from cascade.trainer.loop import TrainerRunner

REF = "ns/gen@sha256:" + "a" * 64


def _challenger(hotkey: str, ref: str = REF, reveal_block: int = 0) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey, uid=1, ref=ref, reveal_block=reveal_block)


def _runner(tmp_path, **round_kw):
    """A minimal stand-in: the funded methods touch only cfg.round + work_root."""
    rnd = RoundConfig(funded_queue_path="funded_queue.json", **round_kw)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path,
                           _funded_field={}, _funded_leg_failures={})
    for name in ("_funded_queue", "_filter_funded_challengers", "_settle_funded",
                 "_skip_unfunded_round", "_submissions_path", "_payer_vault",
                 "_reconcile_funded_pods", "_record_funded_failure"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    return fake


def _queue(tmp_path) -> FundedQueue:
    return FundedQueue(tmp_path / "funded_queue.json")


def test_off_mode_touches_nothing(tmp_path):
    runner = _runner(tmp_path, funded_mode="off")
    field = [_challenger("hkA"), _challenger("hkB")]
    assert runner._filter_funded_challengers(field) == field
    assert not (tmp_path / "funded_queue.json").exists()
    assert not runner._skip_unfunded_round("r1")


def test_shadow_mode_reports_but_never_selects_or_mutates(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    q.add("hkC", REF, reveal_block=20)
    q.mark_in_round(["hkC"])                 # e.g. left over from a required era
    runner = _runner(tmp_path, funded_mode="shadow")
    field = [_challenger("hkA"), _challenger("hkB")]
    assert runner._filter_funded_challengers(field) == field
    assert _queue(tmp_path).get("hkA").status == "queued"   # untouched
    # Strictly read-only: shadow must not recover/expire — the observer never
    # changes the observed (audit 2026-08-29).
    assert _queue(tmp_path).get("hkC").status == "in_round"


def test_required_selects_by_queue_seniority_capped(tmp_path):
    q = _queue(tmp_path)
    q.add("hkLate", REF, reveal_block=300)
    q.add("hkEarly", REF, reveal_block=100)
    q.add("hkMid", REF, reveal_block=200)
    runner = _runner(tmp_path, funded_mode="required", finalists=1, max_finalists=2)
    field = [_challenger(hk) for hk in ("hkLate", "hkEarly", "hkMid", "hkUnfunded")]
    kept = runner._filter_funded_challengers(field)
    # finalist_cap = max(1, 2) = 2: two earliest-revealed funded entries enter.
    assert [c.hotkey for c in kept] == ["hkEarly", "hkMid"]
    reread = _queue(tmp_path)
    assert reread.get("hkEarly").status == "in_round"
    assert reread.get("hkMid").status == "in_round"
    assert reread.get("hkLate").status == "queued"          # waits, unburned


def test_required_ref_mismatch_is_terminal(tmp_path):
    # A re-revealed ref can never match again — leaving it queued would hold
    # the skip-floor open and bill a king leg per boundary forever.
    q = _queue(tmp_path)
    q.add("hkA", "ns/old@sha256:" + "b" * 64, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    kept = runner._filter_funded_challengers([_challenger("hkA", ref=REF)])
    assert kept == []
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.last_error_class) == ("failed", "ref_mismatch")
    # …and the miner can immediately fund the new ref.
    assert _queue(tmp_path).add("hkA", REF, 11) == "queued"


def test_required_burned_hotkey_is_terminal(tmp_path):
    import json

    q = _queue(tmp_path)
    q.add("hkBurned", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    (tmp_path / "trainer_submissions.json").write_text(json.dumps(["hkBurned"]))
    # The burn filter already removed the challenger from the field; the
    # funded entry must die too, not squat in the depth count.
    assert runner._filter_funded_challengers([_challenger("hkOther")]) == []
    entry = _queue(tmp_path).get("hkBurned")
    assert (entry.status, entry.last_error_class) == ("failed", "burned")


def test_required_funded_but_not_revealed_waits(tmp_path):
    q = _queue(tmp_path)
    q.add("hkGhost", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    assert runner._filter_funded_challengers([_challenger("hkOther")]) == []
    assert _queue(tmp_path).get("hkGhost").status == "queued"


def test_torn_round_recovers_then_settle_marks_done(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = [_challenger("hkA")]
    assert [c.hotkey for c in runner._filter_funded_challengers(field)] == ["hkA"]
    # Crash before settle: entry is stuck in_round. The NEXT round's filter
    # recovers it and re-selects — never burned, never dropped.
    kept = runner._filter_funded_challengers(field)
    assert [c.hotkey for c in kept] == ["hkA"]
    # Settle from the duel outcome: a trained (judged) entry goes done; a
    # re-fund then starts fresh.
    runner._settle_funded([(field[0], "challenger")],
                          [SimpleNamespace(hotkey="hkA", role="challenger")])
    assert _queue(tmp_path).get("hkA").status == "done"
    assert _queue(tmp_path).add("hkA", REF, 10) == "queued"


def test_settle_only_touches_in_round_entries(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    runner._funded_field = {"hkA": REF}                      # claimed but never flipped
    runner._settle_funded([(_challenger("hkA"), "challenger")],
                          [SimpleNamespace(hotkey="hkA", role="challenger")])
    assert _queue(tmp_path).get("hkA").status == "queued"


def test_settle_requeues_infra_failed_leg_unburning_the_entry(tmp_path):
    # The 2026-09-01 live finding: a duel leg lost to operator infra must NOT
    # consume the paid entry — it re-queues with one bounded attempt burned.
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._record_funded_failure("hkA", "remote challenger failed (rc=255)",
                                  miner_fault=False, error_class="infra", burn=True)
    runner._settle_funded([(field[0], "challenger")], [])    # no manifest entry
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.attempts, entry.last_error_class) == ("queued", 1, "infra")


def test_settle_fails_miner_fault_leg_terminally(tmp_path):
    # Worker rc=3 = the miner's own submission was rejected: their entry fails
    # visibly (fund again after fixing), never silently re-queues forever.
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._record_funded_failure("hkA", "miner submission rejected: corpus",
                                  miner_fault=True, error_class="generator", burn=False)
    runner._settle_funded([(field[0], "challenger")], [])
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.last_error_class) == ("failed", "generator")


def test_settle_unknown_failure_defaults_to_bounded_infra_requeue(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._settle_funded([(field[0], "challenger")], [])    # no record at all
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.attempts) == ("queued", 1)


def test_skip_unfunded_round_floor(tmp_path):
    import time

    runner = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=True)
    assert runner._skip_unfunded_round("r1")                 # empty queue: skip
    FundedQueue(tmp_path / "funded_queue.json", clock=time.time).add(
        "hkA", REF, reveal_block=10)
    assert not runner._skip_unfunded_round("r1")
    # Without the flag a required round still runs (king-only if unfunded).
    runner2 = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=False)
    assert not runner2._skip_unfunded_round("r1")


def test_skip_floor_expires_dead_entries(tmp_path):
    # An entry far older than the payer-key TTL cannot hold the boundary open:
    # the skip path expires it terminally and then skips — no perpetual
    # king-leg drain from one never-enterable fund (audit 2026-08-29).
    FundedQueue(tmp_path / "funded_queue.json", clock=lambda: 1000.0).add(
        "hkStale", REF, reveal_block=10)     # funded_at ≈ 1970 vs the real now
    runner = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=True)
    assert runner._skip_unfunded_round("r1")
    entry = _queue(tmp_path).get("hkStale")
    assert (entry.status, entry.last_error_class) == ("failed", "funding_expired")


def test_validate_funded_mode():
    assert validate_funded_mode("shadow") == "shadow"
    with pytest.raises(ValueError):
        validate_funded_mode("on")


def test_round_config_defaults_are_inert():
    rnd = RoundConfig()
    assert rnd.funded_mode == "off"
    assert rnd.skip_unfunded_rounds is False
    assert rnd.max_rounds_per_day == 1
