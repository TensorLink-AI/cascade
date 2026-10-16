"""Best-bench pins in the dashboard-facing ``receipts/index.json``.

The scoreboard's best-so-far card and headline read the lowest six-score
geomean across the bench reports of rounds the rolling index lists; the
round holding the best per size is pinned so it never ages out (2026-10-16:
the headline dropped 98.4 % → 98.2 % when u133's report left the scan).
Presentational only — fake S3 store, no chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cascade.shared.hippius as hippius
from cascade.shared.bench_report import bench_report_key, publish_bench_report
from cascade.trainer.promotion import reign_tail

REPO = Path(__file__).resolve().parents[2]
SIZE = "toto2-4m"


class _Store:
    def __init__(self):
        self.objects: dict[str, str] = {}
        self.puts = 0

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.objects[key] = text
        self.puts += 1

    def get_text(self, key):
        if key not in self.objects:
            raise hippius.StorageError(f"s3_get_failed: {key}: missing")
        return self.objects[key]


def _six(g: float) -> dict:
    return {k: g for k in hippius._SIX_KEYS}


def _report(*geomeans: float, size: str = SIZE) -> dict:
    return {"entries": [{"role": "challenger", "size": size, "scores": _six(g)}
                        for g in geomeans]}


def _row(rid: str, block: int, *, vk: str = "v1", status: str = "scored",
         king: str = "kingA") -> dict:
    return {"round_id": rid, "epoch_start_block": block, "validator_hotkey": vk,
            "status": status, "post_round_king_hotkey": king}


def _seed(store: _Store, rows: list[dict], **header) -> None:
    store.objects[hippius.RECEIPT_INDEX_KEY] = json.dumps(
        {"schema": hippius.RECEIPT_INDEX_SCHEMA, "rounds": rows, **header})


def _index(store: _Store) -> dict:
    return json.loads(store.objects[hippius.RECEIPT_INDEX_KEY])


def _rids(store: _Store) -> list[str]:
    return [r["round_id"] for r in _index(store)["rounds"]]


# ── geomean ──────────────────────────────────────────────────────────────────


def test_six_geomean_is_the_pages_plain_geomean():
    s = {"gifteval_crps": 0.5, "gifteval_mase": 0.8, "boom_crps": 0.4,
         "boom_mase": 0.6, "time_crps": 0.55, "time_mase": 0.75}
    want = (0.5 * 0.8 * 0.4 * 0.6 * 0.55 * 0.75) ** (1 / 6)
    assert abs(hippius.bench_six_geomean(s) - want) < 1e-12
    assert hippius.bench_six_geomean({**s, "boom_crps": 0.0}) is None
    assert hippius.bench_six_geomean({**s, "time_mase": float("nan")}) is None
    assert hippius.bench_six_geomean({k: v for k, v in s.items() if k != "boom_mase"}) is None


# ── note_best_bench ─────────────────────────────────────────────────────────


def test_pin_moves_only_when_beaten_and_marks_every_validators_row():
    st = _Store()
    _seed(st, [_row("r1", 100), _row("r1", 100, vk="v2"), _row("r2", 200)])
    hippius.note_best_bench(st, "r1", _report(0.60, 0.59), reach_back=False)
    idx = _index(st)
    assert idx["best_bench"][SIZE] == {"round_id": "r1", "geomean": pytest.approx(0.59)}
    r1 = [r for r in idx["rounds"] if r["round_id"] == "r1"]
    assert len(r1) == 2 and all(r["pinned"] == "best_bench" for r in r1)
    assert all(r["best_bench"] == {SIZE: pytest.approx(0.59)} for r in r1)

    puts = st.puts
    hippius.note_best_bench(st, "r2", _report(0.595), reach_back=False)  # worse
    assert st.puts == puts  # unchanged pin ⇒ no rewrite

    hippius.note_best_bench(st, "r2", _report(0.58), reach_back=False)   # better
    idx = _index(st)
    assert idx["best_bench"][SIZE]["round_id"] == "r2"
    assert {r["round_id"] for r in idx["rounds"] if r.get("pinned")} == {"r2"}
    assert all("best_bench" not in r for r in idx["rounds"] if r["round_id"] == "r1")


def test_pins_are_per_size():
    st = _Store()
    _seed(st, [_row("r1", 100), _row("r2", 200)])
    hippius.note_best_bench(st, "r1", _report(0.59), reach_back=False)
    hippius.note_best_bench(st, "r2", _report(0.70, size="toto2-22m"), reach_back=False)
    best = _index(st)["best_bench"]
    assert best[SIZE]["round_id"] == "r1" and best["toto2-22m"]["round_id"] == "r2"


def test_a_rejected_round_never_pins():
    st = _Store()
    _seed(st, [_row("r1", 100), _row("rx", 200, status="rejected")])
    hippius.note_best_bench(st, "r1", _report(0.60), reach_back=False)
    hippius.note_best_bench(st, "rx", _report(0.50), reach_back=False)
    assert _index(st)["best_bench"][SIZE]["round_id"] == "r1"


def test_a_round_not_listed_yet_pins_and_its_rows_get_marked_on_landing():
    st = _Store()
    _seed(st, [_row("r1", 100)])
    hippius.note_best_bench(st, "r9", _report(0.55), reach_back=False)
    assert _index(st)["best_bench"][SIZE]["round_id"] == "r9"
    hippius.update_receipt_index(st, _row("r9", 900))
    r9 = next(r for r in _index(st)["rounds"] if r["round_id"] == "r9")
    assert r9["pinned"] == "best_bench" and r9["best_bench"] == {SIZE: pytest.approx(0.55)}


def test_reach_back_seeds_from_listed_reports_once():
    st = _Store()
    _seed(st, [_row("old", 100), _row("mid", 200), _row("bad", 300, status="rejected")])
    st.objects[bench_report_key("old")] = json.dumps(_report(0.595))
    st.objects[bench_report_key("mid")] = json.dumps(_report(0.60))
    st.objects[bench_report_key("bad")] = json.dumps(_report(0.40))   # rejected: ignored
    hippius.note_best_bench(st, "new", _report(0.598))
    assert _index(st)["best_bench"][SIZE] == {"round_id": "old", "geomean": pytest.approx(0.595)}


def test_unreadable_or_absent_index_is_never_written():
    st = _Store()
    assert hippius.note_best_bench(st, "r1", _report(0.5)) == {}
    assert hippius.RECEIPT_INDEX_KEY not in st.objects


# ── update_receipt_index ────────────────────────────────────────────────────


def test_trim_never_evicts_the_pinned_round():
    st = _Store()
    _seed(st, [_row(f"r{i}", 100 + i) for i in range(5)])
    hippius.note_best_bench(st, "r0", _report(0.59), reach_back=False)
    for i in range(5, 9):
        hippius.update_receipt_index(st, _row(f"r{i}", 100 + i), max_keep=4)
    assert _rids(st) == ["r0", "r5", "r6", "r7", "r8"]  # pin + the 4 newest
    assert _index(st)["best_bench"][SIZE]["round_id"] == "r0"


def test_old_release_writer_dropping_the_header_is_healed_from_row_marks():
    st = _Store()
    _seed(st, [_row("r1", 100), _row("r2", 200)])
    hippius.note_best_bench(st, "r1", _report(0.59), reach_back=False)
    doc = _index(st)
    doc.pop("best_bench")              # an old-release peer rebuilds the header
    _seed(st, doc["rounds"])
    hippius.update_receipt_index(st, _row("r3", 300))
    assert _index(st)["best_bench"][SIZE] == {"round_id": "r1", "geomean": pytest.approx(0.59)}


# ── the window ──────────────────────────────────────────────────────────────


def test_window_drops_only_pinned_rows_older_than_the_window():
    rows = [{**_row("p", 10), "pinned": "best_bench"}, _row("a", 100),
            {**_row("q", 150), "pinned": "best_bench"}, _row("b", 200)]
    assert [r["round_id"] for r in hippius.receipt_index_window(rows)] == ["a", "q", "b"]
    assert hippius.receipt_index_window([]) == []


def test_reign_tail_ignores_an_old_pinned_row():
    rows = [{**_row("p", 10, king="kingA"), "pinned": "best_bench"},
            _row("a", 100, king="kingA"), _row("b", 200, king="kingA")]
    king, start, rids = reign_tail(rows)
    assert king == "kingA" and start == 100 and rids == ["a", "b"]


# ── publish hook ────────────────────────────────────────────────────────────


def test_publish_bench_report_offers_the_report_to_the_pins():
    st = _Store()
    _seed(st, [_row("r1", 100)])
    publish_bench_report(st, json.dumps(_report(0.59)), "r1")
    assert _index(st)["best_bench"][SIZE]["round_id"] == "r1"


def test_publish_bench_report_survives_a_failing_pin():
    class _NoRead(_Store):
        def get_text(self, key):
            raise RuntimeError("boom")

    st = _NoRead()
    assert publish_bench_report(st, json.dumps(_report(0.59)), "r1") == bench_report_key("r1")
    assert bench_report_key("r1") in st.objects


# ── dashboards ──────────────────────────────────────────────────────────────


def test_both_dashboards_split_old_pins_out_of_the_window_and_still_scan_them():
    for page in ("stakeholders.html", "index.html"):
        html = (REPO / "cascade" / "website" / page).read_text(encoding="utf-8")
        assert "splitPinned(collapseByRound(d))" in html, page
        assert 'r.pinned==="best_bench"' in html.replace(" ", ""), page
    sh = (REPO / "cascade" / "website" / "stakeholders.html").read_text(encoding="utf-8")
    scan = sh[sh.index("function scanBench"):sh.index("function scanBench") + 900]
    assert "pinnedOldScored()" in scan
    view = sh[sh.index("function benchView"):sh.index("function benchView") + 2400]
    assert "pinnedOldScored().concat(rs)" in view and "if(isOld) return;" in view
