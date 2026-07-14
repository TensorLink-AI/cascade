"""Cascade — king-reign promotion: trigger, selection, action, persistence.

The wall-clock reign clock, per-reign checkpoint log, lowest-score selection, and
the vacate-throne action are exercised here on the pure controller (no I/O beyond
an optional temp state file). Time is passed explicitly as ``now`` so nothing
depends on the wall clock. Checkpoints are scored on six public-benchmark numbers
(GIFT-Eval / BOOM / TIME CRPS+MASE).
"""

from __future__ import annotations

import math

import pytest

from cascade.validator.cascade import (
    CascadeController,
    CascadeState,
    CheckpointRecord,
    cascade_score,
    crown,
    dumps,
    geomean,
    load_state,
    loads,
    reign_days,
    select_winner,
    should_cascade,
    vacate,
)

DAY = 86_400.0


def _ckpt(cid, gc, gm, tc, tm, ts, *, bc=1.0, bm=1.0) -> CheckpointRecord:
    """A scored checkpoint. BOOM defaults to 1.0 so tests that only vary GIFT-Eval
    and TIME keep BOOM out of the comparison."""
    return CheckpointRecord.scored(
        cid, gifteval_crps=gc, gifteval_mase=gm, boom_crps=bc, boom_mase=bm,
        time_crps=tc, time_mase=tm, timestamp=ts,
    )


def _record(ctl, cid, *, gc, gm, tc, tm, now, bc=1.0, bm=1.0):
    return ctl.record_checkpoint(
        cid, gifteval_crps=gc, gifteval_mase=gm, boom_crps=bc, boom_mase=bm,
        time_crps=tc, time_mase=tm, now=now,
    )


# ── the geomean score ────────────────────────────────────────────────────────


def test_geomean_is_nth_root_of_product():
    assert geomean(1.0, 1.0, 1.0) == 1.0
    assert math.isclose(geomean(2.0, 2.0, 2.0, 2.0), 2.0)
    assert math.isclose(geomean(0.5, 0.8, 0.4, 0.9), (0.5 * 0.8 * 0.4 * 0.9) ** 0.25)


def test_cascade_score_is_geomean_of_six():
    vals = (0.5, 0.8, 0.6, 0.7, 0.4, 0.9)
    assert math.isclose(cascade_score(*vals), math.prod(vals) ** (1.0 / 6))


def test_geomean_clamps_zero_and_negative():
    # A zero (or spurious negative) eval must not zero-out or NaN the product.
    v = geomean(0.0, 1.0, 1.0)
    assert v > 0.0 and math.isfinite(v)
    assert math.isfinite(geomean(-1.0, 1.0, 1.0))


def test_record_score_matches_cascade_score():
    r = _ckpt("c", 0.5, 0.8, 0.4, 0.9, 0.0, bc=0.6, bm=0.7)
    assert math.isclose(r.score, cascade_score(0.5, 0.8, 0.6, 0.7, 0.4, 0.9))


# ── trigger: the wall-clock reign clock ──────────────────────────────────────


def test_reign_clock_counts_days_since_crown():
    st = crown(CascadeState(), king_hotkey="k", now=1000.0)
    assert reign_days(st, now=1000.0) == 0.0
    assert math.isclose(reign_days(st, now=1000.0 + 3 * DAY), 3.0)


def test_reign_clock_is_none_when_throne_vacant():
    assert reign_days(CascadeState(), now=1000.0) is None


def test_dethrone_resets_the_clock():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", now=0.0)
    _record(ctl, "a", gc=1, gm=1, tc=1, tm=1, now=DAY)
    # 6 days in, not ripe.
    assert ctl.cascade_check(now=6 * DAY) is None
    # A new king dethrones on day 6 → clock resets; the old reign's log is cleared.
    ctl.note_dethrone("kingB", now=6 * DAY)
    assert ctl.state.king_hotkey == "kingB"
    assert ctl.state.checkpoints == ()
    assert math.isclose(reign_days(ctl.state, now=6 * DAY), 0.0)


def test_should_cascade_requires_clock_and_a_checkpoint():
    st = crown(CascadeState(), king_hotkey="k", now=0.0)
    # Ripe clock but empty log → not a cascade (nothing to promote).
    assert not should_cascade(st, now=10 * DAY, reign_days_threshold=7)
    st = CascadeState(king_hotkey="k", reign_start=0.0, checkpoints=(_ckpt("a", 1, 1, 1, 1, 0.0),))
    assert not should_cascade(st, now=6 * DAY, reign_days_threshold=7)  # too soon
    assert should_cascade(st, now=7 * DAY, reign_days_threshold=7)      # ripe + a checkpoint


# ── selection: lowest score wins, earliest breaks ties ───────────────────────


def test_select_winner_picks_lowest_score():
    st = CascadeState(
        king_hotkey="k",
        reign_start=0.0,
        checkpoints=(
            _ckpt("hi", 0.5, 0.8, 0.4, 0.9, 1.0),
            _ckpt("lo", 0.4, 0.7, 0.3, 0.8, 2.0),
            _ckpt("mid", 0.45, 0.75, 0.35, 0.85, 3.0),
        ),
    )
    assert select_winner(st).checkpoint_id == "lo"


def test_boom_can_flip_the_winner():
    # Two checkpoints identical on GIFT-Eval and TIME; BOOM decides.
    st = CascadeState(
        king_hotkey="k",
        reign_start=0.0,
        checkpoints=(
            _ckpt("boom-bad", 0.5, 0.5, 0.5, 0.5, 1.0, bc=0.9, bm=0.9),
            _ckpt("boom-good", 0.5, 0.5, 0.5, 0.5, 2.0, bc=0.2, bm=0.2),
        ),
    )
    assert select_winner(st).checkpoint_id == "boom-good"


def test_select_winner_breaks_ties_by_earliest():
    st = CascadeState(
        king_hotkey="k",
        reign_start=0.0,
        checkpoints=(
            _ckpt("late", 1.0, 1.0, 1.0, 1.0, 5.0),
            _ckpt("early", 1.0, 1.0, 1.0, 1.0, 2.0),
        ),
    )
    assert select_winner(st).checkpoint_id == "early"


def test_select_winner_none_when_empty():
    assert select_winner(CascadeState(king_hotkey="k", reign_start=0.0)) is None


# ── action: fire the cascade, install, vacate ────────────────────────────────


def test_cascade_fires_at_threshold_and_selects_best():
    installed: list[CheckpointRecord] = []
    ctl = CascadeController(reign_days=7, install_fn=installed.append)
    ctl.note_dethrone("kingA", now=0.0)
    _record(ctl, "worse", gc=0.6, gm=0.9, tc=0.5, tm=1.0, now=DAY)
    _record(ctl, "best", gc=0.4, gm=0.7, tc=0.3, tm=0.8, now=2 * DAY)

    assert ctl.cascade_check(now=6.99 * DAY) is None      # clock not ripe
    event = ctl.cascade_check(now=7 * DAY)
    assert event is not None
    assert event.old_king == "kingA"
    assert math.isclose(event.reign_days, 7.0)
    assert event.winner.checkpoint_id == "best"
    # Installed as-is, exactly once.
    assert [r.checkpoint_id for r in installed] == ["best"]


def test_cascade_vacates_throne_and_resets_clock():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", now=0.0)
    _record(ctl, "c", gc=1, gm=1, tc=1, tm=1, now=DAY)
    ctl.cascade_check(now=7 * DAY)
    # Throne vacated: no king, clock stopped, reign log cleared.
    assert ctl.state == CascadeState()
    assert reign_days(ctl.state, now=8 * DAY) is None
    # A vacant throne never fires again until a new king is crowned.
    assert ctl.cascade_check(now=100 * DAY) is None


def test_cascade_holds_when_clock_ripe_but_no_checkpoint():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", now=0.0)
    # Reigned well past the threshold but produced no scored checkpoint → hold.
    assert ctl.cascade_check(now=30 * DAY) is None
    assert ctl.state.king_hotkey == "kingA"  # still reigning


def test_record_checkpoint_ignored_when_throne_vacant():
    ctl = CascadeController(reign_days=7)
    # No king crowned yet.
    assert _record(ctl, "c", gc=1, gm=1, tc=1, tm=1, now=0.0) is None
    assert ctl.state.checkpoints == ()


def test_install_failure_leaves_reign_intact_for_retry():
    def _boom(_winner: CheckpointRecord) -> None:
        raise RuntimeError("registry down")

    ctl = CascadeController(reign_days=7, install_fn=_boom)
    ctl.note_dethrone("kingA", now=0.0)
    _record(ctl, "c", gc=1, gm=1, tc=1, tm=1, now=DAY)
    with pytest.raises(RuntimeError):
        ctl.cascade_check(now=7 * DAY)
    # The throne was NOT vacated — a failed install is retried next round.
    assert ctl.state.king_hotkey == "kingA"
    assert len(ctl.state.checkpoints) == 1


def test_new_reign_after_cascade_can_fire_again():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", now=0.0)
    _record(ctl, "a", gc=1, gm=1, tc=1, tm=1, now=DAY)
    ctl.cascade_check(now=7 * DAY)          # fires, throne vacated
    # A fresh king is crowned; its own reign clock starts from the crown instant.
    ctl.note_dethrone("kingB", now=8 * DAY)
    _record(ctl, "b", gc=0.5, gm=0.5, tc=0.5, tm=0.5, now=9 * DAY)
    assert ctl.cascade_check(now=8 * DAY + 6 * DAY) is None
    ev = ctl.cascade_check(now=8 * DAY + 7 * DAY)
    assert ev is not None and ev.old_king == "kingB" and ev.winner.checkpoint_id == "b"


# ── persistence: the reign clock survives restarts ───────────────────────────


def test_state_round_trips_through_json():
    st = CascadeState(
        king_hotkey="k",
        reign_start=1234.5,
        checkpoints=(
            _ckpt("a", 0.5, 0.8, 0.4, 0.9, 10.0, bc=0.6, bm=0.7),
            _ckpt("b", 0.4, 0.7, 0.3, 0.8, 20.0, bc=0.5, bm=0.6),
        ),
    )
    assert loads(dumps(st)) == st


def test_empty_state_round_trips():
    assert loads(dumps(CascadeState())) == CascadeState()


def test_controller_persists_and_reloads(tmp_path):
    path = tmp_path / "cascade_state.json"
    ctl = CascadeController(reign_days=7, state_path=path)
    ctl.note_dethrone("kingA", now=1000.0)
    _record(ctl, "a", gc=0.4, gm=0.7, tc=0.3, tm=0.8, now=1000.0 + DAY)
    # Simulate a restart: rebuild the controller from the persisted file.
    reloaded = CascadeController(reign_days=7, state=load_state(path), state_path=path)
    assert reloaded.state.king_hotkey == "kingA"
    assert reloaded.state.reign_start == 1000.0
    assert [c.checkpoint_id for c in reloaded.state.checkpoints] == ["a"]
    # And the resumed clock still fires at the original crown instant.
    ev = reloaded.cascade_check(now=1000.0 + 7 * DAY)
    assert ev is not None and ev.winner.checkpoint_id == "a"


def test_load_state_missing_file_is_fresh(tmp_path):
    assert load_state(tmp_path / "nope.json") == CascadeState()


def test_load_state_corrupt_file_is_fresh(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_state(p) == CascadeState()


def test_vacate_is_empty_state():
    assert vacate() == CascadeState()


# ── config toggle: warm-start on/off ─────────────────────────────────────────


def test_cascade_toggle_wires_controller(tmp_path):
    from cascade.shared.config import load_chain_config
    from cascade.validator.loop import build_runner

    base = load_chain_config("chain.toml")
    # Off (mainnet default) ⇒ no controller wired (pure KOTH).
    assert base.scoring.cascade_enabled is False
    runner_off = build_runner(chain_toml=None)  # DEFAULT_CHAIN_TOML == chain.toml
    assert runner_off.cascade is None

    # On ⇒ controller wired (synthetic toml; the shipped testnet file toggles
    # this deliberately over time — deferred 2026-07-13 — so don't assert it).
    runner_on = build_runner(chain_toml=_write_toml_with_cascade(tmp_path, enabled=True))
    assert runner_on.cascade is not None
    assert runner_on.cascade.reign_days == 7


def _write_toml_with_cascade(tmp_path, *, enabled: bool):
    """Copy chain.toml into tmp with cascade_enabled flipped, so build_runner's
    persisted-state paths land in tmp rather than the repo root."""
    import re
    from pathlib import Path

    text = Path("chain.toml").read_text(encoding="utf-8")
    text = re.sub(r"cascade_enabled\s*=\s*\w+",
                  f"cascade_enabled      = {'true' if enabled else 'false'}", text)
    # Redirect the persisted state files under tmp.
    text = re.sub(r'cascade_state_db_path\s*=\s*"[^"]*"',
                  f'cascade_state_db_path      = "{tmp_path / "cascade_state.json"}"', text)
    text = re.sub(r'warm_start_init_path\s*=\s*"[^"]*"',
                  f'warm_start_init_path       = "{tmp_path / "warm_start_init.json"}"', text)
    p = tmp_path / "chain.toml"
    p.write_text(text, encoding="utf-8")
    return p
