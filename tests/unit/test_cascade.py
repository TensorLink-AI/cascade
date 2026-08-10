"""Cascade validator half — reign clock, reign log, promotion adoption,
persistence.

Under propose-and-verify (DEC-CA-0013) the validator never fires or selects a
promotion: the controller tracked here maintains the block-anchored reign clock
(the envelope's ripeness predicate), the reign's benched-checkpoint log (the
envelope's provenance/quality evidence), and the accepted generation + member
set. The clock is driven by explicit ``block`` values (7200 blocks = 1 day);
wall-clock ``now`` only stamps records. Checkpoints are scored on six
public-benchmark numbers (GIFT-Eval / BOOM / TIME CRPS+MASE).
"""

from __future__ import annotations

import math

from cascade.validator.cascade import (
    BLOCKS_PER_DAY,
    CascadeController,
    CascadeState,
    CheckpointRecord,
    best_score,
    cascade_score,
    crown,
    dumps,
    geomean,
    load_state,
    loads,
    log_record_for,
    reign_days,
)

# The reign clock counts ROUNDS. With round_cfg=None the divisor is a fixed
# 7200-block round, so these block arithmetic constants are unchanged.
DAY = BLOCKS_PER_DAY  # one 7200-block round


def _ckpt(cid, gc, gm, tc, tm, ts, *, bc=1.0, bm=1.0, role="") -> CheckpointRecord:
    """A scored checkpoint. BOOM defaults to 1.0 so tests that only vary GIFT-Eval
    and TIME keep BOOM out of the comparison."""
    return CheckpointRecord.scored(
        cid, gifteval_crps=gc, gifteval_mase=gm, boom_crps=bc, boom_mase=bm,
        time_crps=tc, time_mase=tm, timestamp=ts, role=role,
    )


def _record(ctl, cid, *, gc, gm, tc, tm, now, bc=1.0, bm=1.0, role=""):
    return ctl.record_checkpoint(
        cid, gifteval_crps=gc, gifteval_mase=gm, boom_crps=bc, boom_mase=bm,
        time_crps=tc, time_mase=tm, now=now, role=role,
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


# ── the block-anchored reign clock ───────────────────────────────────────────


def test_reign_clock_counts_days_since_crown():
    st = crown(CascadeState(), king_hotkey="k", block=1000)
    assert reign_days(st, block=1000) == 0.0
    assert math.isclose(reign_days(st, block=1000 + 3 * DAY), 3.0)


def test_reign_clock_is_none_when_throne_vacant():
    assert reign_days(CascadeState(), block=1000) is None


def test_dethrone_resets_the_clock_and_clears_the_log():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    _record(ctl, "a", gc=1, gm=1, tc=1, tm=1, now=1.0)
    assert not ctl.is_ripe(block=6 * DAY)
    # A new king dethrones on day 6 → clock resets; the old reign's log is cleared.
    ctl.note_dethrone("kingB", block=6 * DAY)
    assert ctl.state.king_hotkey == "kingB"
    assert ctl.state.checkpoints == ()
    assert reign_days(ctl.state, block=6 * DAY) == 0.0


def test_is_ripe_at_threshold():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    assert not ctl.is_ripe(block=7 * DAY - 1)
    assert ctl.is_ripe(block=7 * DAY)


def test_is_ripe_false_when_vacant_or_unanchored():
    assert not CascadeController(reign_days=7).is_ripe(block=100 * DAY)
    ctl = CascadeController(
        reign_days=7, state=CascadeState(king_hotkey="k", reign_start_block=None))
    assert not ctl.is_ripe(block=100 * DAY)


def test_unanchored_reign_reanchors_instead_of_reading_ripe():
    """The stale-state regression (DEC-CA-0005): a persisted reign with no block
    anchor (legacy wall-clock state) must re-anchor at the observed round's
    block — never read as instantly ripe off stale state."""
    ctl = CascadeController(
        reign_days=7,
        state=CascadeState(
            king_hotkey="kingA", reign_start_block=None,
            checkpoints=(_ckpt("a", 1, 1, 1, 1, 0.0),),
        ),
    )
    ctl.observe_round(block=50_000)
    assert ctl.state.reign_start_block == 50_000
    assert ctl.state.checkpoints != ()                    # log kept
    assert not ctl.is_ripe(block=50_000 + 6 * DAY)        # not ripe vs the NEW anchor
    assert ctl.is_ripe(block=50_000 + 7 * DAY)


def test_observe_round_is_a_noop_when_anchored():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=1000)
    ctl.observe_round(block=9999)
    assert ctl.state.reign_start_block == 1000


# ── the reign log: envelope evidence ─────────────────────────────────────────


def test_best_score_and_lookup():
    st = CascadeState(
        king_hotkey="k",
        reign_start_block=0,
        checkpoints=(
            _ckpt("hi", 0.5, 0.8, 0.4, 0.9, 1.0),
            _ckpt("lo", 0.4, 0.7, 0.3, 0.8, 2.0),
        ),
    )
    assert math.isclose(best_score(st), st.checkpoints[1].score)
    assert log_record_for(st, "hi") is st.checkpoints[0]
    assert log_record_for(st, "nope") is None
    assert best_score(CascadeState(king_hotkey="k", reign_start_block=0)) is None


def test_record_checkpoint_ignored_when_throne_vacant():
    ctl = CascadeController(reign_days=7)
    # No king crowned yet.
    assert _record(ctl, "c", gc=1, gm=1, tc=1, tm=1, now=0.0) is None
    assert ctl.state.checkpoints == ()


def test_record_checkpoint_logs_both_roles_and_dedupes():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    assert _record(ctl, "k1", gc=1, gm=1, tc=1, tm=1, now=1.0, role="king") is not None
    assert _record(ctl, "c1", gc=1, gm=1, tc=1, tm=1, now=1.0, role="challenger") is not None
    # The pending-bench re-probe can offer a round twice — second offer is a no-op.
    assert _record(ctl, "k1", gc=1, gm=1, tc=1, tm=1, now=2.0, role="king") is None
    assert [(r.checkpoint_id, r.role) for r in ctl.state.checkpoints] == [
        ("k1", "king"), ("c1", "challenger")]


# ── accepted promotions: generation + members ────────────────────────────────


def test_note_promotion_recrowns_same_king_and_installs_members():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    _record(ctl, "a", gc=1, gm=1, tc=1, tm=1, now=1.0)
    ctl.note_promotion(generation=1, members=("a", "b"), block=7 * DAY)
    # King persists (DEC-CA-0004); clock restarts at the acceptance block; the
    # log clears for the fresh reign; the generation + member set install.
    assert ctl.state.king_hotkey == "kingA"
    assert ctl.state.reign_start_block == 7 * DAY
    assert ctl.state.checkpoints == ()
    assert ctl.state.generation == 1
    assert ctl.state.members == ("a", "b")
    assert not ctl.is_ripe(block=7 * DAY + 6 * DAY)
    assert ctl.is_ripe(block=14 * DAY)


def test_promotion_survives_a_dethrone():
    """A promotion outlives the reign that produced it: the field keeps training
    from the live member set no matter who holds the throne."""
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    ctl.note_promotion(generation=1, members=("a",), block=7 * DAY)
    ctl.note_dethrone("kingB", block=8 * DAY)
    assert ctl.state.generation == 1
    assert ctl.state.members == ("a",)
    assert ctl.state.king_hotkey == "kingB"


def test_can_verify_ripeness_needs_an_observed_anchor():
    ctl = CascadeController(reign_days=7)
    assert not ctl.can_verify_ripeness()          # fresh state: no anchor
    # A WATCHED dethrone verdict is an observed transition — the clock can
    # attest even the first (generation 0 → 1) promotion.
    ctl.note_dethrone("kingA", block=0)
    assert ctl.can_verify_ripeness()
    # An ADOPTION (validator joined/restarted mid-reign) is not.
    ctl2 = CascadeController(reign_days=7)
    ctl2.note_dethrone("kingA", block=0, observed=False)
    assert not ctl2.can_verify_ripeness()
    # ...but an accepted promotion re-arms attestation for the next reign.
    ctl2.note_promotion(generation=1, members=("a",), block=7 * DAY)
    assert ctl2.can_verify_ripeness()
    # A legacy re-anchor only measures its own uptime.
    ctl3 = CascadeController(
        reign_days=7, state=CascadeState(king_hotkey="k", reign_start_block=None))
    ctl3.observe_round(block=1000)
    assert not ctl3.can_verify_ripeness()


def test_adopt_member_set_grandfathers_recorded_generation():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    ctl.adopt_member_set(generation=2, members=("a", "b"))
    assert ctl.state.generation == 2
    assert ctl.state.members == ("a", "b")
    # Once past the random-init era the shim never runs again.
    ctl.adopt_member_set(generation=9, members=("z",))
    assert ctl.state.generation == 2 and ctl.state.members == ("a", "b")


def test_adopt_legacy_pointer_grandfathers_generation_one():
    ctl = CascadeController(reign_days=7)
    ctl.note_dethrone("kingA", block=0)
    ctl.adopt_legacy_pointer("old-winner")
    assert ctl.state.generation == 1
    assert ctl.state.members == ("old-winner",)
    # Once past the random-init era the shim never runs again.
    ctl.adopt_legacy_pointer("other")
    assert ctl.state.members == ("old-winner",)


# ── persistence: clock, log, and accepted promotion survive restarts ──────────


def test_state_round_trips_through_json():
    sized = CheckpointRecord.scored(
        "c", gifteval_crps=0.5, gifteval_mase=0.8, boom_crps=0.6, boom_mase=0.7,
        time_crps=0.4, time_mase=0.9, timestamp=30.0, size="toto2-4m", role="challenger",
    )
    st = CascadeState(
        king_hotkey="k",
        reign_start_block=12345,
        checkpoints=(
            _ckpt("a", 0.5, 0.8, 0.4, 0.9, 10.0, bc=0.6, bm=0.7),
            _ckpt("b", 0.4, 0.7, 0.3, 0.8, 20.0, bc=0.5, bm=0.6),
            sized,
        ),
        generation=3,
        members=("a", "b"),
        clock_observed=True,
    )
    again = loads(dumps(st))
    assert again == st
    assert again.checkpoints[2].size == "toto2-4m"
    assert again.checkpoints[2].role == "challenger"


def test_empty_state_round_trips():
    assert loads(dumps(CascadeState())) == CascadeState()


def test_pre_promotion_state_loads_at_generation_zero():
    """State files from before the promotion fields default to the random-init
    era (the validator loop's adoption shim upgrades them)."""
    st = loads('{"king_hotkey": "k", "reign_start_block": 5, "checkpoints": []}')
    assert st.generation == 0 and st.members == ()


def test_controller_persists_and_reloads(tmp_path):
    path = tmp_path / "cascade_state.json"
    ctl = CascadeController(reign_days=7, state_path=path)
    ctl.note_dethrone("kingA", block=1000)
    _record(ctl, "a", gc=0.4, gm=0.7, tc=0.3, tm=0.8, now=1.0)
    ctl.note_promotion(generation=2, members=("a",), block=1000 + 7 * DAY)
    # Simulate a restart: rebuild the controller from the persisted file.
    reloaded = CascadeController(reign_days=7, state=load_state(path), state_path=path)
    assert reloaded.state.king_hotkey == "kingA"
    assert reloaded.state.generation == 2
    assert reloaded.state.members == ("a",)
    # The resumed clock still measures from the acceptance block.
    assert reloaded.state.reign_start_block == 1000 + 7 * DAY
    assert reloaded.is_ripe(block=1000 + 14 * DAY)


def test_legacy_wallclock_state_loads_unanchored(tmp_path):
    """A state file written by the wall-clock era (reign_start in epoch seconds,
    no reign_start_block) keeps its king and log but loads UNANCHORED — the
    clock re-anchors at the next observed round instead of reading ripe off a
    stale wall-clock value (the 2026-07-20 immediate-fire)."""
    p = tmp_path / "cascade_state.json"
    p.write_text(
        '{"king_hotkey": "kingA", "reign_start": 1752300000.0, "checkpoints": []}',
        encoding="utf-8",
    )
    st = load_state(p)
    assert st.king_hotkey == "kingA"
    assert st.reign_start_block is None


def test_load_state_missing_file_is_fresh(tmp_path):
    assert load_state(tmp_path / "nope.json") == CascadeState()


def test_load_state_corrupt_file_is_fresh(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_state(p) == CascadeState()


# ── config toggle: warm-start on/off ─────────────────────────────────────────


def test_cascade_toggle_wires_controller(tmp_path):
    from cascade.shared.config import load_chain_config
    from cascade.validator.loop import build_runner

    base = load_chain_config("chain.toml")
    # Shipped ARMED (2026-08-05). Config assert only — building a runner from
    # the repo toml with cascade on would persist reign state into the repo
    # root, so both runner checks below use synthetic tomls under tmp.
    assert base.scoring.cascade_enabled is True
    assert base.scoring.cascade_top_k == 3
    assert math.isclose(base.scoring.cascade_quality_epsilon, 0.05)

    # Off ⇒ no controller wired (pure KOTH).
    runner_off = build_runner(chain_toml=_write_toml_with_cascade(tmp_path, enabled=False))
    assert runner_off.cascade is None

    # On ⇒ controller wired.
    runner_on = build_runner(chain_toml=_write_toml_with_cascade(tmp_path, enabled=True))
    assert runner_on.cascade is not None
    assert runner_on.cascade.reign_days == 5  # cascade_reign_rounds, lowered at arming


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


# ── epoch-relative reign clock ───────────────────────────────────────────────


def test_reign_is_counted_in_rounds_not_wall_clock_days():
    """The threshold means "survived N challenges". Halving the round must
    halve the wall-clock a reign needs, not double the challenges required."""
    from cascade.shared.config import RoundConfig
    from cascade.validator.cascade import reign_rounds

    st = crown(CascadeState(), king_hotkey="k", block=0)
    slow = RoundConfig(epoch_blocks=7200)
    fast = RoundConfig(epoch_blocks=3600)

    seven_slow_rounds = 7 * 7200
    assert math.isclose(reign_rounds(st, seven_slow_rounds, slow), 7.0)
    # Same wall-clock at the faster cadence is twice as many challenges.
    assert math.isclose(reign_rounds(st, seven_slow_rounds, fast), 14.0)
    # ...and 7 challenges now takes half the wall clock.
    assert math.isclose(reign_rounds(st, 7 * 3600, fast), 7.0)


def test_reign_rounds_splits_at_a_cadence_activation():
    """A reign spanning the switch counts the rounds that actually ran on each
    side, not the whole span at either length."""
    from cascade.shared.config import RoundConfig
    from cascade.validator.cascade import reign_rounds

    cfg = RoundConfig(epoch_blocks=3600, epoch_blocks_prev=7200,
                      epoch_activation_block=100_000)
    st = crown(CascadeState(), king_hotkey="k", block=100_000 - 2 * 7200)
    # 2 rounds at 7200 before the switch, then 3 at 3600 after it.
    got = reign_rounds(st, 100_000 + 3 * 3600, cfg)
    assert math.isclose(got, 5.0)


def test_reign_rounds_defaults_to_the_historical_divisor():
    from cascade.validator.cascade import reign_rounds

    st = crown(CascadeState(), king_hotkey="k", block=0)
    assert math.isclose(reign_rounds(st, 7 * BLOCKS_PER_DAY, None), 7.0)
