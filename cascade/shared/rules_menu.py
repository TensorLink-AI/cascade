"""The rule menu (DEC-CA-0075, W-CMP): pre-implemented verdict TEST
FAMILIES and PAY CURVES a lane (or a lane-C competition spec) selects by
name in its config, so the owner can change "the test or the formula" by a
signed overlay (DEC-CA-0057) without a validator release.

Every entry is code that ships in this release; an overlay only PICKS one.
A name this release does not know, or one a lane cannot run, is refused at
load (the overlay is refused before any node signals it; on a lane-C spec
the competition is refused and its share burns —
:mod:`cascade.shared.competition_spec`).

Test families — how a settlement's family of looks is corrected:

* ``plain_lcb`` — each entry's gain LCB at ``alpha`` (no correction);
* ``bonferroni`` — each entry at ``alpha / k`` (``k`` = its family size as
  the lane counts it: lane B a leg's own targets, lane C the weighted owner
  family of a target, red-team F6);
* ``maxt`` — the joint max-T bootstrap over a cohort sharing a reference
  (:func:`cascade.eval.bootstrap.cohort_maxt_lcbs`; lane D's crown test).

Pay curves — what a confirmed win is paid over time (the curve is RECORDED
on the win when it differs from the lane's legacy one, so a later overlay
never changes a promise already made):

* ``tiered_decay`` — full pay for the win's recorded ``full_eras``, then
  exponential decay (``half_life_rounds``) to ``floor`` until the TTL
  (:func:`cascade.eval.expert_lane.reward_multiplier`; lane B legacy);
* ``fixed_term`` — full pay for the win's recorded ``full_eras`` (lane C:
  the DEC-CA-0073 5–20 day gain-scaled duration), then the slot ends;
* ``indefinite_until_beaten`` — full pay until a better entry replaces the
  holder; no TTL.

``LANE_SUPPORT`` is the honest compatibility matrix of THIS release: the
combinations each lane's verdict / payout code actually runs. Anything
outside it is refused at load rather than silently ignored.
"""

from __future__ import annotations

TEST_FAMILIES: tuple[str, ...] = ("plain_lcb", "maxt", "bonferroni")
PAY_CURVES: tuple[str, ...] = ("tiered_decay", "fixed_term", "indefinite_until_beaten")

# Today's rules, per lane — the defaults; every default is byte-identical.
LANE_DEFAULTS: dict[str, tuple[str, str]] = {
    "lane_b": ("bonferroni", "tiered_decay"),
    "lane_c": ("bonferroni", "fixed_term"),
    "lane_d": ("maxt", "indefinite_until_beaten"),
}

# What each lane's code runs in this release (family set, curve set).
LANE_SUPPORT: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "lane_b": (frozenset({"plain_lcb", "bonferroni"}), frozenset(PAY_CURVES)),
    "lane_c": (frozenset({"plain_lcb", "bonferroni"}), frozenset(PAY_CURVES)),
    "lane_d": (frozenset({"maxt"}), frozenset({"indefinite_until_beaten"})),
}


class MenuError(ValueError):
    """An unknown or unsupported menu choice (typed)."""


def check_choice(lane: str, *, test_family: str, pay_curve: str) -> tuple[str, str]:
    """``(test_family, pay_curve)`` if ``lane`` runs them, else raise
    :class:`MenuError` naming the menu and the lane's support."""
    fams, curves = LANE_SUPPORT[lane]
    if test_family not in TEST_FAMILIES:
        raise MenuError(f"[{lane}] test_family {test_family!r} is not on the menu "
                        f"{list(TEST_FAMILIES)}")
    if pay_curve not in PAY_CURVES:
        raise MenuError(f"[{lane}] pay_curve {pay_curve!r} is not on the menu {list(PAY_CURVES)}")
    if test_family not in fams:
        raise MenuError(f"[{lane}] test_family {test_family!r}: this release runs "
                        f"{sorted(fams)} in {lane}")
    if pay_curve not in curves:
        raise MenuError(f"[{lane}] pay_curve {pay_curve!r}: this release runs "
                        f"{sorted(curves)} in {lane}")
    return test_family, pay_curve


def parse_lane_menu(lane: str, raw: dict) -> tuple[str, str]:
    """``test_family`` / ``pay_curve`` of a lane table (absent = the lane's
    legacy rule). Raises ``ValueError`` (a :class:`MenuError`)."""
    df, dc = LANE_DEFAULTS[lane]
    f, c = raw.get("test_family", df), raw.get("pay_curve", dc)
    if not isinstance(f, str) or not isinstance(c, str):
        raise MenuError(f"[{lane}] test_family / pay_curve must be strings")
    return check_choice(lane, test_family=f, pay_curve=c)


# ── test families ───────────────────────────────────────────────────────────


def family_alpha(family: str, alpha: float, k: int) -> float:
    """The per-entry level of a Bonferroni-shaped lane: ``alpha`` for
    ``plain_lcb``, ``alpha / k`` for ``bonferroni`` (``k`` >= 1).
    ``maxt`` has no per-entry level (joint test: :func:`maxt_lcbs`)."""
    if family == "plain_lcb":
        return float(alpha)
    if family == "bonferroni":
        return float(alpha) / max(1, int(k))
    raise MenuError(f"test family {family!r} has no per-entry level")


def family_size(family: str, k: int) -> int:
    """The divisor a Bonferroni-shaped lane applies (1 = none)."""
    return max(1, int(k)) if family == "bonferroni" else 1


def maxt_lcbs(king, members, **kw):
    """The ``maxt`` family: joint max-T LCBs of ``members`` against
    ``king`` (:func:`cascade.eval.bootstrap.cohort_maxt_lcbs`, same args)."""
    from ..eval.bootstrap import cohort_maxt_lcbs

    return cohort_maxt_lcbs(king, members, **kw)


# ── pay curves ──────────────────────────────────────────────────────────────


def recorded_curve(win, lane: str) -> str:
    """The curve a win is paid on: the one recorded at the win, else the
    lane's LEGACY curve (never the config now in force)."""
    c = str(getattr(win, "curve", "") or "")
    return c or LANE_DEFAULTS[lane][1]


def curve_record(curve: str, lane: str) -> str:
    """What a new win records: ``""`` for the lane's legacy curve (the win
    stays byte-identical), else the curve name."""
    if curve not in PAY_CURVES:
        raise MenuError(f"pay curve {curve!r} is not on the menu")
    return "" if curve == LANE_DEFAULTS[lane][1] else curve


def curve_expired(curve: str, age_eras: int, *, full_eras: int | None, ttl_eras: int = 0) -> bool:
    """Whether a win of ``age_eras`` has stopped paying on ``curve``."""
    age = int(age_eras)
    if curve == "indefinite_until_beaten":
        return False
    if curve == "fixed_term":
        if full_eras is None:
            raise MenuError("a fixed_term win carries no full_eras")
        return age >= int(full_eras)
    if curve == "tiered_decay":
        return int(ttl_eras or 0) > 0 and age >= int(ttl_eras)
    raise MenuError(f"pay curve {curve!r} is not on the menu")


def curve_multiplier(curve: str, age_eras: int, *, full_eras: int | None,
                     half_life_rounds: float = 1.0, floor: float = 0.1) -> float:
    """A live (not expired) win's multiplier on ``curve``."""
    if curve in ("indefinite_until_beaten", "fixed_term"):
        return 1.0
    if curve == "tiered_decay":
        from ..eval.expert_lane import reward_multiplier

        if full_eras is None:
            raise MenuError("a tiered_decay win carries no full_eras")
        return reward_multiplier(int(age_eras), full_eras=int(full_eras),
                                 half_life_rounds=float(half_life_rounds), floor=float(floor))
    raise MenuError(f"pay curve {curve!r} is not on the menu")


def menu() -> dict:
    """The menu as the overlay policy publishes it."""
    return {"test_families": list(TEST_FAMILIES), "pay_curves": list(PAY_CURVES),
            "lane_defaults": {k: {"test_family": f, "pay_curve": c}
                              for k, (f, c) in sorted(LANE_DEFAULTS.items())},
            "lane_support": {k: {"test_families": sorted(f), "pay_curves": sorted(c)}
                             for k, (f, c) in sorted(LANE_SUPPORT.items())}}


__all__ = ["LANE_DEFAULTS", "LANE_SUPPORT", "PAY_CURVES", "TEST_FAMILIES", "MenuError",
           "check_choice", "curve_expired", "curve_multiplier", "curve_record",
           "family_alpha", "family_size", "maxt_lcbs", "menu", "parse_lane_menu",
           "recorded_curve"]
