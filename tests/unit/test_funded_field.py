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
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    fake._funded_queue = TrainerRunner._funded_queue.__get__(fake)
    fake._filter_funded_challengers = TrainerRunner._filter_funded_challengers.__get__(fake)
    fake._mark_funded_done = TrainerRunner._mark_funded_done.__get__(fake)
    fake._skip_unfunded_round = TrainerRunner._skip_unfunded_round.__get__(fake)
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
    runner = _runner(tmp_path, funded_mode="shadow")
    field = [_challenger("hkA"), _challenger("hkB")]
    assert runner._filter_funded_challengers(field) == field
    assert _queue(tmp_path).get("hkA").status == "queued"   # untouched


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


def test_required_ref_mismatch_waits(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", "ns/old@sha256:" + "b" * 64, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    kept = runner._filter_funded_challengers([_challenger("hkA", ref=REF)])
    assert kept == []                                        # funded a different reveal
    assert _queue(tmp_path).get("hkA").status == "queued"


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
    # Settle: the consumed entry goes done; a re-fund then starts fresh.
    runner._mark_funded_done(field)
    assert _queue(tmp_path).get("hkA").status == "done"
    assert _queue(tmp_path).add("hkA", REF, 10) == "queued"


def test_mark_done_only_touches_in_round_entries(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    runner._mark_funded_done([_challenger("hkA")])           # never selected
    assert _queue(tmp_path).get("hkA").status == "queued"


def test_skip_unfunded_round_floor(tmp_path):
    runner = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=True)
    assert runner._skip_unfunded_round("r1")                 # empty queue: skip
    _queue(tmp_path).add("hkA", REF, reveal_block=10)
    assert not runner._skip_unfunded_round("r1")
    # Without the flag a required round still runs (king-only if unfunded).
    runner2 = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=False)
    assert not runner2._skip_unfunded_round("r1")


def test_validate_funded_mode():
    assert validate_funded_mode("shadow") == "shadow"
    with pytest.raises(ValueError):
        validate_funded_mode("on")


def test_round_config_defaults_are_inert():
    rnd = RoundConfig()
    assert rnd.funded_mode == "off"
    assert rnd.skip_unfunded_rounds is False
    assert rnd.max_rounds_per_day == 1
