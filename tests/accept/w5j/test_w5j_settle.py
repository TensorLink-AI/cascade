"""W5j through ``settle_lane_d``: the judged windows are the cohort's draw at
the settlement's boundary block hash, identical for KING_D and every member,
capped by ``max_windows_per_settlement``, and recorded in the signed report."""

from __future__ import annotations

from .helpers import HASH_A, HASH_B, HORIZON, lane_d_cfg, ref_for
from .settle_helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    SECOND_UID,
    draw_cells,
    entry,
    forecasts,
    house,
    records,
    settle,
    write_pool,
)

EIGHT = {f"src{i}": 8 for i in range(8)}      # 64 eligible windows, 8 sources


def _run(tmp_path, block_hash, *, names=("alice",), cap=40):
    cfg = lane_d_cfg(max_windows_per_settlement=cap)
    recs = records(EIGHT)
    pool = write_pool(tmp_path / f"pool-{block_hash[-6:]}", recs)
    es = [entry(n) for n in names]
    h = house()
    fc = forecasts(recs, [e["digest"] for e in es], h["digest"])
    uids = dict(zip((e["hotkey"] for e in es), (CHALLENGER_UID, SECOND_UID), strict=False))
    out = settle(cfg, BOUNDARY_1, pool, es, h, fc, uids,
                 tmp_path / f"state-{block_hash[-6:]}.json", block_hash)
    assert out.report is not None, "armed settlement wrote no report"
    return out.report, recs, es, h, cfg


def _judged(report) -> list[str]:
    return sorted(w["window_id"] for w in report["windows"])


def test_w5j_settlement_judges_the_draw(tmp_path):
    report, recs, _es, _h, cfg = _run(tmp_path, HASH_A)
    ref = ref_for(draw_cells(recs), cfg, HASH_A, max_windows=40)
    want = [w["window_id"] for w in ref["windows"]]
    got = _judged(report)
    assert len(got) <= 40, f"{len(got)} windows judged above max_windows_per_settlement 40"
    assert got == want, "the judged windows are not the seeded per-bin draw"


def test_w5j_report_records_the_draw(tmp_path):
    report, *_ = _run(tmp_path, HASH_A)
    d = report.get("draw")
    assert isinstance(d, dict), "the report has no draw record"
    assert d.get("boundary_block_hash") == HASH_A
    assert d.get("seed") == f"{HASH_A}:lane_d_windows"
    assert d.get("max_windows_per_settlement") == 40
    for w in report["windows"]:
        for f in ("timeframe", "bin", "h", "origin", "kind", "embargo_window"):
            assert f in w, f"report window {w['window_id']} lacks {f!r}"
        assert (w["timeframe"], w["bin"], w["kind"], w["h"]) == ("H", "forward", "forward",
                                                                 HORIZON)
        assert w["window_id"] == f"{w['window_id'].rsplit(':', 2)[0]}:{w['origin']}:h{HORIZON}"
        assert w["embargo_window"] is False


def test_w5j_king_and_every_member_score_the_identical_draw(tmp_path):
    report, _recs, es, h, _cfg = _run(tmp_path, HASH_A, names=("alice", "bob"))
    judged = set(_judged(report))
    by_digest: dict[str, set[str]] = {}
    for r in report["rows"]:
        by_digest.setdefault(r["digest"], set()).add(r["window_id"])
    members = [e["digest"] for e in es]
    assert set(members) <= set(by_digest), "a member has no rows"
    for d in members:
        assert by_digest[d] == judged, f"member {d[:8]} scored a different window set"
    kings = [d for d in by_digest if d not in members]
    assert kings, "no KING_D rows"
    for d in kings:
        assert by_digest[d] == judged, f"bar {d[:8]} scored a different window set"
    assert h["digest"] in by_digest


def test_w5j_another_block_hash_draws_another_set(tmp_path):
    a, *_ = _run(tmp_path, HASH_A)
    b, *_ = _run(tmp_path, HASH_B)
    assert _judged(a) != _judged(b), "the judged set does not depend on the block hash"
