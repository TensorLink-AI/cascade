"""WB11: the lookup API WB13's intake check calls (plan §12b.1 LB1).

``match_series`` counts the submission sub-windows (every offset) that hit the
exact and the normalised set; ``is_contaminated`` applies
``contamination_min_hits`` (1 exact or 3 normalised). A scaled + shifted or
float32-rounded copy of a suite window hits the normalised set but not the
exact set. Contract: README.md."""

from __future__ import annotations

import struct

import pytest

from .helpers import MIN_HITS, WINDOW, build, clean_series, fixture_suites, match, mod

SERIES = ("gift", "gift/m4_hourly/0")       # 80 points
NORMALISED_FLOOR = 0.9      # share of windows a scaled + shifted copy must hit
# float32 moves a z-score by ~1e-6 here, far inside schema 2's tolerance
# (README "Normalised matching at every scale"): every window must hit. v1's
# 4-significant-figure rounding lost about 1 window in 7 to this.
FLOAT32_FLOOR = 1.0


def _suite_series(suite: str = SERIES[0], item: str = SERIES[1]) -> list[float]:
    return list(fixture_suites()[suite]["series"][item])


def test_clean_series_has_no_hits():
    r = match(build(), clean_series())
    assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, r
    assert r["first"] is None, r
    assert mod().is_contaminated(build(), clean_series()) is False


def test_embedded_window_is_found_with_suite_and_window_id():
    a = build()
    seg = _suite_series()[20:20 + WINDOW + 2]          # 3 sub-windows: offsets 20, 21, 22
    sub = clean_series(50) + seg + clean_series(40, seed="tail")
    r = match(a, sub)
    assert r["exact_hits"] == 3, r
    assert r["normalised_hits"] == 3, r
    first = r["first"]
    assert first["kind"] == "exact", first
    assert first["suite"] == "gift", first
    assert first["window_id"] == "gift/m4_hourly/0@20", first
    assert first["at"] == 50, first
    assert mod().is_contaminated(a, sub) is True


def test_less_than_one_window_is_not_a_hit():
    a = build()
    seg = _suite_series()[10:10 + WINDOW - 1]
    r = match(a, clean_series(30) + seg + clean_series(30, seed="tail"))
    assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, r


def test_scaled_and_shifted_copy_hits_normalised_not_exact():
    a = build()
    seg = _suite_series()[5:5 + 40]                    # 9 sub-windows
    copy_ = [7.5 * x + 1234.5 for x in seg]
    r = match(a, copy_)
    assert r["exact_hits"] == 0, r
    assert r["normalised_hits"] >= NORMALISED_FLOOR * 9, r
    assert r["first"]["kind"] == "normalised" and r["first"]["suite"] == "gift", r
    assert mod().is_contaminated(a, copy_) is True


def test_float32_rounded_copy_hits_normalised_not_exact():
    a = build()
    seg = _suite_series()
    f32 = [struct.unpack("<f", struct.pack("<f", x))[0] for x in seg]
    assert f32 != seg
    n = len(seg) - WINDOW + 1
    r = match(a, f32)
    assert r["exact_hits"] == 0, r
    assert r["normalised_hits"] >= FLOAT32_FLOOR * n, (r, n)
    assert mod().is_contaminated(a, f32) is True


def test_offset_copy_and_int_inputs():
    a = build()
    seg = _suite_series("boom", "boom/svc-b/mem")      # 40 points, 9 windows
    r = match(a, [x - 1.0e6 for x in seg])
    assert r["exact_hits"] == 0 and r["normalised_hits"] >= NORMALISED_FLOOR * 9, r
    flat = [3] * 40                                    # ints equal to the flat suite values
    r = match(a, flat)
    assert r["exact_hits"] == 9 and r["normalised_hits"] == 0, r


def test_constant_submission_never_hits_normalised():
    a = build()
    r = match(a, [5.0] * 60)
    assert r["exact_hits"] == 0 and r["normalised_hits"] == 0, r


def test_nan_in_submission_skips_only_its_windows():
    a = build()
    seg = _suite_series()[:40]                          # offsets 0..8
    seg[3] = float("nan")                               # kills offsets 0..3
    r = match(a, seg)
    assert r["exact_hits"] == 5, r
    assert r["first"]["at"] == 4 and r["first"]["window_id"] == "gift/m4_hourly/0@4", r


@pytest.mark.parametrize("hits, min_hits, want", [
    ({"exact": 1, "normalised": 1}, MIN_HITS, True),
    ({"exact": 0, "normalised": 3}, MIN_HITS, True),
    ({"exact": 0, "normalised": 2}, MIN_HITS, False),
    ({"exact": 1, "normalised": 1}, {"exact": 2, "normalised": 5}, False),
    ({"exact": 0, "normalised": 5}, (("exact", 2), ("normalised", 5)), True),
])
def test_is_contaminated_applies_min_hits(hits, min_hits, want):
    """``hits`` is built as a submission: a scaled copy of a 31+n-point segment
    (normalised only) after an exact copy of a 31+e-point segment."""
    a = build()
    s = _suite_series()
    sub = []
    if hits["exact"]:
        sub += s[:WINDOW - 1 + hits["exact"]]
        sub += clean_series(40, seed="gap")
    if hits["normalised"] > hits["exact"]:
        sub += [2.0 * x - 9.0 for x in s[40:40 + WINDOW - 1 + hits["normalised"] - hits["exact"]]]
    r = match(a, sub)
    assert r["exact_hits"] == hits["exact"], r
    assert r["normalised_hits"] == hits["normalised"], r
    assert mod().is_contaminated(a, sub, min_hits) is want, (r, min_hits)


def test_is_contaminated_default_is_the_wb10_default():
    """With no ``min_hits`` it is ``contamination_min_hits``'s default, 1 / 3."""
    from cascade.shared.config.v2 import LaneBConfig

    lb = LaneBConfig()
    assert dict(lb.contamination_min_hits) == MIN_HITS
    a = build()
    s = _suite_series()
    two = [2.0 * x - 9.0 for x in s[:WINDOW + 1]]        # 2 normalised windows
    assert mod().is_contaminated(a, two) is False
    assert mod().is_contaminated(a, two, lb.contamination_min_hits) is False
    three = [2.0 * x - 9.0 for x in s[:WINDOW + 2]]
    assert mod().is_contaminated(a, three) is True


def test_lookup_works_on_a_loaded_artefact(tmp_path):
    m = mod()
    a = build()
    p = tmp_path / "hashset-v1.json"
    m.write_artefact(a, p)
    b = m.load_artefact(p)
    seg = _suite_series()[:40]
    assert m.match_series(b, seg) == m.match_series(a, seg)
