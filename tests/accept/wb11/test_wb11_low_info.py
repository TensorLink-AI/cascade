"""WB11 (RV6 B5): low-information windows are never contamination evidence.

Real GIFT/BOOM data has zero runs (solar at night), sparse 0/1 counters,
exact index counters and values sitting at a level plus float noise. Schema 2
as first built hashed all of them, so 1 KB of zeros, 32 ``x = 0`` lines,
one-hot list literals or a ``linspace`` ramp in a benign generator came back
contaminated. The rule (README "Low-information windows",
``helpers.low_information``): such a window is in neither set at build time,
and a low-information submission window is never counted as a hit. Real suite
windows are still detected. Contract: README.md."""

from __future__ import annotations

import functools

import pytest

from .helpers import (
    WINDOW,
    benign_inputs,
    build,
    exact_hash,
    expected_exact,
    fixture_suites,
    hazard_suites,
    informative_count,
    low_information,
    match,
    mod,
    sub_windows,
)

INFORMATIVE_ITEMS = (("gift", "gift/solar/0"), ("boom", "boom/spikes/0"),
                     ("time", "time/trend/0"))
LOW_INFO_ITEMS = (("boom", "boom/counter/0"), ("time", "time/ramp/0"), ("fev", "fev/still/0"))


@functools.lru_cache(maxsize=1)
def _artefact() -> dict:
    return build(hazard_suites())


def _values(suite: str, item: str) -> list[float]:
    return list(hazard_suites()[suite]["series"][item])


def test_fixture_preconditions():
    """The hazard fixture has what the tests need: low-information windows in
    every hazard item and informative ones in the others."""
    for suite, item in LOW_INFO_ITEMS:
        vals = _values(suite, item)
        assert all(low_information(w) for _, w in sub_windows(vals)), item
    for suite, item in INFORMATIVE_ITEMS:
        vals = _values(suite, item)
        assert informative_count(vals) >= 9, (item, informative_count(vals))
    solar = _values("gift", "gift/solar/0")
    assert any(low_information(w) for _, w in sub_windows(solar)), "no all-zero window"
    for name, vals in benign_inputs().items():
        assert all(low_information([float(x) for x in w]) for _, w in sub_windows(vals)), name


def test_low_information_windows_are_in_neither_set():
    a = _artefact()
    want = expected_exact(hazard_suites())
    assert a["exact"] == want, (
        f"exact set differs from the low-information-filtered reference: "
        f"{len(set(a['exact']) - set(want))} extra, {len(set(want) - set(a['exact']))} missing")
    for _suite, item in LOW_INFO_ITEMS:
        for kind in ("exact", "normalised"):
            ids = [wid for s, wid in a[kind].values() if wid.startswith(item + "@")]
            assert ids == [], f"{kind} set holds low-information windows of {item}: {ids[:3]}"
    zero = exact_hash([0.0] * WINDOW)
    assert zero not in a["exact"], "the all-zero window is in the exact set"
    for s, wid in a["normalised"].values():
        item, off = wid.rsplit("@", 1)
        w = hazard_suites()[s]["series"][item][int(off):int(off) + WINDOW]
        assert not low_information(w), f"normalised set holds a low-information window {wid}"


@pytest.mark.parametrize("name", sorted(benign_inputs()))
def test_benign_streams_are_clean(name):
    """1 KB of zero f64s, 32 ``x = 0`` constants, three one-hot list literals,
    a linspace ramp, ``range(48)`` and a value plus float noise: 0 hits, not
    contaminated, against suites that carry zero runs, spikes and ramps."""
    vals = benign_inputs()[name]
    r = match(_artefact(), vals)
    assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, (name, r)
    assert r["first"] is None, (name, r)
    assert mod().is_contaminated(_artefact(), vals) is False, (name, r)


def test_low_information_submission_windows_are_ignored_at_lookup():
    """Even an artefact that holds a low-information window's hash (one built
    before the rule, or by hand) gives no hit for it."""
    a = build(fixture_suites())
    a = dict(a, exact=dict(a["exact"]))
    for w in ([0.0] * WINDOW, [float(i) for i in range(WINDOW)],
              [1.0] + [0.0] * (WINDOW - 1)):
        a["exact"][exact_hash(w)] = ["time", "time/flat/0@0"]
    for vals in ([0.0] * 40, [float(i) for i in range(40)], [1.0] + [0.0] * 40):
        r = match(a, vals)
        assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, (vals[:3], r)
        assert mod().is_contaminated(a, vals) is False, (vals[:3], r)


@pytest.mark.parametrize("suite, item", INFORMATIVE_ITEMS, ids=[i for _, i in INFORMATIVE_ITEMS])
def test_real_suite_windows_are_still_detected(suite, item):
    """Every informative window of the hazard suites still hits exact and
    normalised, and the series is contaminated."""
    vals = _values(suite, item)
    r = match(_artefact(), vals)
    n = informative_count(vals)
    assert r["exact_hits"] == n and r["normalised_hits"] == n, (item, n, r)
    assert r["first"]["suite"] == suite and r["first"]["window_id"].startswith(item + "@"), r
    assert mod().is_contaminated(_artefact(), vals) is True, (item, r)


def test_scaled_copy_of_a_spiky_suite_series_is_still_caught():
    """A scaled and shifted copy of the noisy spike train (the shape one-hot
    lists resemble) still hits normalised on its informative windows: the
    lookup skips the one-hot submission, not the noisy suite shape."""
    vals = _values("boom", "boom/spikes/0")
    cp = [3.7 * x + 100.0 for x in vals]
    r = match(_artefact(), cp)
    assert r["exact_hits"] == 0, r
    assert r["normalised_hits"] >= 0.9 * informative_count(vals), r
    assert mod().is_contaminated(_artefact(), cp) is True, r
