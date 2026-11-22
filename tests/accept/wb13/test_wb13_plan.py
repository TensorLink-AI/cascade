"""WB13: the plan rows these tests pin, verbatim, and the fixture's sanity.

These tests pass on main. They make sure the plan text the contract quotes
hasn't drifted, and that the WB11 hash set (on main) really sees the fixture
windows, so a WB13 failure can only be about WB13.

The WB13 row (§3.0 lane B track), verbatim:

    | WB13 | Intake contamination check (LB1): AST literals, `config.json`,
    non-code bytes vs the hash set; typed `benchmark_series_embedded`;
    `suite_target_max_repo_mb` (D38) | receipt field `contamination:
    {hashset_version, result, suite, window_id}` | WB11 | ci |
    `scripts/laned/accept WB13` (TB10–TB12) |

TB10–TB12 (§4), verbatim:

    | TB10 | **(rev4)** Generator embedding an exact GIFT/BOOM/TIME/FEV test
    window as a literal | Refused at intake, `benchmark_series_embedded`,
    before any compute |
    | TB11 | **(rev4)** Same window scaled ×3.7, offset +100 and rounded to
    3 dp, in `config.json` | Refused (normalised match) |
    | TB12 | **(rev4)** A clean generator; and a hash-set version change |
    Passes; the receipt records the hash-set version used |
"""

from __future__ import annotations

from . import helpers as h

WB13_ROW = (
    "| WB13 | Intake contamination check (LB1): AST literals, `config.json`, non-code bytes "
    "vs the hash set; typed `benchmark_series_embedded`; `suite_target_max_repo_mb` (D38) "
    "| receipt field `contamination: {hashset_version, result, suite, window_id}` | WB11 "
    "| ci | `scripts/laned/accept WB13` (TB10–TB12) |")
TB_ROWS = (
    "| TB10 | **(rev4)** Generator embedding an exact GIFT/BOOM/TIME/FEV test window as a "
    "literal | Refused at intake, `benchmark_series_embedded`, before any compute |",
    "| TB11 | **(rev4)** Same window scaled ×3.7, offset +100 and rounded to 3 dp, in "
    "`config.json` | Refused (normalised match) |",
    "| TB12 | **(rev4)** A clean generator; and a hash-set version change | Passes; the "
    "receipt records the hash-set version used |",
)
LB1_LINES = (
    "  - numeric literals in `.py` files (AST walk, all number constants and list",
    "  - every value in `config.json`;",
    "  - any non-code bytes (decoded as float32/float64/int arrays).",
    "  - A match on ≥ `contamination_min_hits` sub-windows (recommended 1 exact or",
    "- **When.** At intake, **before any compute**. The typed refusal is",
    "  `benchmark_series_embedded`, with the suite and window id in the receipt",
)
D38_DECIDED = "| D38 | decided | (a) 8 MB `suite_target_max_repo_mb` | O | owner (default), 2026-11-21 |"
TB_MAP = "| TB10–TB12 | WB11, WB13 |"


def _plan_lines() -> set[str]:
    return set(h.PLAN.read_text(encoding="utf-8").splitlines())


def test_plan_rows_pinned():
    lines = _plan_lines()
    for row in (WB13_ROW, *TB_ROWS, *LB1_LINES, D38_DECIDED, TB_MAP):
        assert row in lines, f"plan drifted; re-read and re-pin: {row}"


def test_shipped_suite_target_cap_is_d38_a():
    """D38 (a), decided: 8 MB for suite-target entries."""
    from cascade.shared.config import load_chain_config

    lb = load_chain_config(h.CHAIN_TOML).lane_b
    assert lb.suite_target_max_repo_mb == 8
    assert lb.contamination_hashset == ""      # unarmed on main until the owner pins WB11


def test_fixture_windows_are_in_the_hash_set():
    """The WB11 lookup sees every fixture embedding the WB13 tests use:
    TB10's exact window, TB11's scaled and rounded copy, the shifted copy,
    the float32 copy of ``F32_ITEM`` and the integer flat series. A clean series has no hit."""
    m = h.wb11()
    art = h.artefact(1)
    seg = h.segment()
    exact = m.match_series(art, seg)
    assert exact["exact_hits"] == h.SEG_LEN - 31
    assert exact["first"]["window_id"] == h.SEG_WINDOW_ID
    for name, vals in (("tb11", h.tb11(seg)), ("shifted", h.shifted(seg)),
                       ("float32", h.float32_copy(h.f32_source()))):
        r = m.match_series(art, vals)
        assert r["exact_hits"] == 0, name
        assert m.is_contaminated(art, vals), (name, r)
        assert r["first"]["suite"] == "gift", (name, r)
    flat = m.match_series(art, [h.FLAT_VALUE] * 40)
    assert flat["exact_hits"] == 9 and flat["first"]["window_id"] == h.FLAT_WINDOW_ID
    assert m.match_series(art, h.clean_series())["first"] is None
    assert m.match_series(art, h.segment()[:31])["first"] is None


def test_fixture_tb11_margin():
    """TB11 needs ≥ 3 normalised hits (the WB10 default). The fixture
    window gives a clear margin. README "TB11 and series scale" explains why
    this depends on the series' scale under the WB11 schema-1 hash."""
    r = h.wb11().match_series(h.artefact(1), h.tb11(h.segment()))
    assert r["normalised_hits"] >= 5, r
