"""WB11: normalised hashes catch TB11 evasions at every series level (plan
§12b.1 LB1 "robust to scale, offset and rounding", §4 TB11).

A suite window copied as ``round(a·x + b, 3)`` (any ``a != 0``, so a negated
copy too) or cast to float32 must hit the normalised set at series levels
0.4, 7, 100 and 5000, enough to be ``is_contaminated`` under the WB10 default
rule; unrelated series at the same levels must stay clean. Schema 1 (z-score
at 4 significant figures) fails this at small levels: rounding to 3 dp moves
a z-score by up to ~6e-3 at level 0.4, which flips its 4th significant figure
in nearly every window. Contract: README.md "Normalised matching at every
scale"."""

from __future__ import annotations

import functools

import pytest

from .helpers import (
    MIN_HITS,
    ROUND_DP,
    ROUND_STEP,
    SCALE_ITEMS,
    build,
    fixture_suites,
    match,
    mod,
    scale_suites,
    sub_windows,
    to_float32,
    unrelated_series,
    window_count,
    window_std,
)

# Minimum share of a copy's sub-windows that must hit the normalised set.
# - rounded copies: 0.9. The tolerance needed is bounded (rounding step <= 4 %
#   of the copy's window std, checked below), so a correct scheme hits every
#   window; 0.9 leaves room for a builder's probe cap, and a 40-point fragment
#   (9 windows) still gives >= 8 hits, well above min_hits["normalised"] = 3.
# - float32 casts: 1.0. float32 moves a z-score by <= ~1e-5 at these levels,
#   orders of magnitude inside any tolerance that meets the rounded cases.
ROUNDED_FLOOR = 0.9
FLOAT32_FLOOR = 1.0
# The rounded cases are only promised while the rounding step is small against
# the copy's spread: ROUND_STEP <= |a| * window std / MIN_SPREAD_STEPS for every
# window. Below that, rounding destroys the shape itself (a=1e-6 rounds every
# value to the same 3-dp number), and no normalisation can recover it.
MIN_SPREAD_STEPS = 25

# (case, a, b, rounded?): a·x + b, then rounded to 3 dp or cast to float32.
CASES = (
    ("tb11", 3.7, 100.0, True),            # plan TB11: x3.7, +100, 3 dp
    ("negated", -2.5, 7.0, True),          # a < 0: the z-score flips sign
    ("large_offset", 12.0, -1.0e4, True),
    ("rounded_only", 1.0, 0.0, True),      # the series itself at 3 dp
    ("float32", 1.0, 0.0, False),
    ("tb11_float32", 3.7, 100.0, False),
)

# Unrelated series per level for the false-positive check: 30 x 17 windows x 4
# levels = 2040 sub-windows, all of which must miss. Zero hits in 2040 bounds
# the per-window false-positive rate below 3/2040 ~ 1.5e-3 (95 %, rule of
# three); a series needs 3 hits to be refused.
UNRELATED_PER_LEVEL = 30


@functools.lru_cache(maxsize=1)
def _artefact() -> dict:
    return build(scale_suites())


def _copy(values, a: float, b: float, rounded: bool) -> list[float]:
    if rounded:
        return [round(a * x + b, ROUND_DP) for x in values]
    return [to_float32(a * x + b) for x in values]


def _min_spread(values, a: float) -> float:
    return min(abs(a) * window_std(w) for _, w in sub_windows(values))


def _check_copy(art: dict, suite: str, item: str, values, case) -> None:
    name, a, b, rounded = case
    if rounded:
        spread = _min_spread(values, a)
        assert spread >= ROUND_STEP * MIN_SPREAD_STEPS, (
            f"fixture precondition: {item} {name}: smallest |a|·window std {spread:.4g} is "
            f"under {MIN_SPREAD_STEPS} rounding steps; the case is outside the promise")
    cp = _copy(values, a, b, rounded)
    assert cp != list(values), (item, name)
    n = window_count(cp)
    floor = ROUNDED_FLOOR if rounded else FLOAT32_FLOOR
    r = match(art, cp)
    assert r["exact_hits"] == 0, (item, name, r)
    assert r["normalised_hits"] >= floor * n, (
        f"{item} ({name}: a={a}, b={b}, {'3 dp' if rounded else 'float32'}): "
        f"{r['normalised_hits']}/{n} windows hit the normalised set, need >= {floor:.0%}")
    first = r["first"]
    assert first is not None and first["kind"] == "normalised", (item, name, r)
    assert first["suite"] == suite, (item, name, first)
    assert first["window_id"].startswith(item + "@"), (item, name, first)
    assert mod().is_contaminated(art, cp) is True, (item, name, r)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("spec", SCALE_ITEMS, ids=[f"level{s[2]:g}" for s in SCALE_ITEMS])
def test_evasion_copy_is_caught_at_every_level(spec, case):
    suite, item, _level, _scale = spec
    values = scale_suites()[suite]["series"][item]
    _check_copy(_artefact(), suite, item, values, case)


# The base fixture's series at levels 0.4, 100 and 5000 (TB11 only): the same
# promise holds for the artefact every other WB11 test builds.
BASE_ITEMS = (("boom", "boom/svc-a/cpu"), ("gift", "gift/m4_hourly/0"),
              ("gift", "gift/electricity/7"))


@pytest.mark.parametrize("suite, item", BASE_ITEMS, ids=[i for _, i in BASE_ITEMS])
def test_tb11_copy_of_a_base_fixture_series_is_caught(suite, item):
    values = fixture_suites()[suite]["series"][item]
    _check_copy(build(), suite, item, values, CASES[0])


@pytest.mark.parametrize("spec", SCALE_ITEMS, ids=[f"level{s[2]:g}" for s in SCALE_ITEMS])
def test_unrelated_series_at_the_same_level_stay_clean(spec):
    _suite, _item, level, scale = spec
    art = _artefact()
    windows = 0
    for k in range(UNRELATED_PER_LEVEL):
        s = unrelated_series(level, scale, k)
        windows += window_count(s)
        r = match(art, s)
        assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, (level, k, r)
        assert r["first"] is None, (level, k, r)
        assert mod().is_contaminated(art, s, MIN_HITS) is False, (level, k, r)
        # also their rounded copies (a rounded clean series is still clean)
        rc = _copy(s, 3.7, 100.0, True)
        assert match(art, rc)["normalised_hits"] == 0, (level, k)
    assert windows == UNRELATED_PER_LEVEL * window_count(scale_suites()[_suite]["series"][_item])
