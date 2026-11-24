"""W2, plan rev6.23 (owner 2026-11-24): ``[lane_d] court_weights`` (one
weight per court slot, ``1 + reward_prior_kings`` of them, each >= 0,
summing to 1 within 1e-9) replaces the geometric ``king_decay``, which is
retired (``RETIRED_KEYS["lane_d"]``: old files load, the value is ignored);
the crown bar decays from ``d_margin_start`` to ``d_margin`` over
``d_margin_warmup_settlements``. Lane A's ``[scoring] king_decay`` is a
different, live key and is untouched."""

from __future__ import annotations

import pytest

from .helpers import ARM, SHIPPED, field, load_ok, overrides_refusal, raw_with, refused


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_court_weights_are_80_15_5(name):
    ld = load_ok(raw_with(name)).lane_d
    assert tuple(field(ld, "court_weights")) == (0.80, 0.15, 0.05)
    assert field(ld, "reward_prior_kings") == 2
    assert not hasattr(ld, "king_decay"), "[lane_d] king_decay is retired (rev6.23)"


@pytest.mark.parametrize("name", SHIPPED)
def test_lane_d_king_decay_is_retired_and_ignored(name):
    from cascade.shared.config import RETIRED_KEYS

    assert "king_decay" in RETIRED_KEYS.get("lane_d", frozenset()), \
        "[lane_d] king_decay is not in RETIRED_KEYS['lane_d']"
    old = load_ok(raw_with(name, {"king_decay": 0.25}), " with the retired king_decay")
    new = load_ok(raw_with(name))
    assert old.lane_d == new.lane_d


@pytest.mark.parametrize("name", SHIPPED)
def test_lane_a_king_decay_is_untouched(name):
    """``[scoring] king_decay`` still parses and moves lane A's config."""
    raw = raw_with(name)
    raw["scoring"] = {**raw["scoring"], "king_decay": 0.25}
    assert load_ok(raw).scoring.king_decay == 0.25


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("rpk", [0, 1, 3])
def test_reward_prior_kings_alone_is_a_length_mismatch(name, rpk):
    assert refused(raw_with(name, {"reward_prior_kings": rpk})), \
        f"reward_prior_kings = {rpk} with 3 court_weights loaded"
    w = [1.0] + [0.0] * rpk
    ld = load_ok(raw_with(name, {"reward_prior_kings": rpk, "court_weights": w})).lane_d
    assert tuple(ld.court_weights) == tuple(w)


def test_court_weights_and_the_bar_keys_are_overlay_settable():
    for ov in ({"court_weights": [0.7, 0.2, 0.1]}, {"d_margin_start": 0.05},
               {"d_margin_warmup_settlements": 16},
               {"reward_prior_kings": 1, "court_weights": [0.9, 0.1]}):
        assert overrides_refusal({"lane_d": {**ARM, **ov}}) is None, ov


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_crown_bar_defaults(name):
    ld = load_ok(raw_with(name)).lane_d
    got = tuple(getattr(ld, k, None)
                for k in ("d_margin_start", "d_margin", "d_margin_warmup_settlements"))
    assert got == (0.03, 0.01, 32), f"crown bar keys parsed as {got}"
    assert callable(getattr(ld, "crown_bar", None)), "LaneDConfig has no crown_bar(tenure)"
    assert ld.crown_bar(0) == 0.03 and ld.crown_bar(32) == 0.01 and ld.crown_bar(99) == 0.01
    assert ld.crown_bar(16) == pytest.approx(0.02, abs=1e-15)
