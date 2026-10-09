"""Dashboard benchmark tiles: a "Best checkpoint so far" row beside the king's.

The king retrains every era on a different init member, so its "beats Toto2"
tags flip era to era (2026-10-07: TIME CRPS 0.57009 → 0.57156 against the
reference 0.57046). The best row only moves when a checkpoint genuinely beats
it. Its pick must match the stakeholder page's "best so far" card: lowest
combined score (geomean of the six) across the loaded reports. Pure text.
"""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "cascade" / "website" / "index.html"


def _render_train_stats() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return re.search(r"function renderTrainStats\(.*?\n\}\n", html, re.S).group(0)


def test_both_benchmark_rows_render_with_labels():
    fn = _render_train_stats()
    assert "Current king &mdash; latest era" in fn
    assert "Best checkpoint so far" in fn
    # one shared renderer, so both rows carry the same six tiles and Toto2 tags
    assert fn.count("function benchRow(") == 1
    assert "benchRow(lastK.e.six" in fn and "benchRow(best.e.six" in fn


def test_best_is_the_lowest_combined_score_king_or_challenger():
    fn = _render_train_stats()
    # selection over every entry (no role filter), lowest geomean wins
    assert "if(e.geo!=null&&(!best||e.geo<best.geo)) best={geo:e.geo,e:e,row:row};" in fn


def test_best_row_shows_the_kings_gap():
    fn = _render_train_stats()
    assert "vs best" in fn and "Current king: " in fn


def test_row_heading_spans_the_grid():
    html = PAGE.read_text(encoding="utf-8")
    assert re.search(r"\.train-tiles \.row-h \{[^}]*grid-column:1 / -1", html)
