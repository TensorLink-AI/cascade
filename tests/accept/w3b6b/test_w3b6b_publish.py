"""W3b6b, publish: W8b's step PUTs every bound bench record at
``bench_v/lane_d/<boundary>/<sha256>.json`` (the persisted bytes, through its
existing store) BEFORE the report; a failed PUT is ``publish_failed`` and the
court doesn't move. Lane A's ``bench_v`` keys are untouched. Contract:
README.md."""

from __future__ import annotations

import hashlib
import json

from . import helpers as h

w8b = h.w8b


def _rig(tmp_path, **kw):
    cfg = w8b.chain_cfg(tmp_path / "v")
    return w8b.Rig(tmp_path, cfg, w8b.fixture(tmp_path), **kw)


def _bound(report: dict) -> list[str]:
    return [d for r in h.gate_results(report) for role in h.ROLES
            for d in [(r.get("rows") or {}).get(role)] if d]


def test_bound_records_are_published_content_addressed_before_the_report(tmp_path):
    rig = _rig(tmp_path)
    rig.step(w8b.B1)
    rig.step(w8b.B2)
    rep = rig.published_report(w8b.B2)
    bound = _bound(rep)
    assert bound, "W3b6b: the crowning settlement binds its gate records"
    report_at = rig.store.puts.index(w8b.wiring().report_key(w8b.B2))
    for d in bound:
        key = h.rows_key(w8b.B2, d)
        assert key in rig.store.objects, f"{key} is published"
        text = rig.store.objects[key]
        assert hashlib.sha256(text.encode("ascii")).hexdigest() == d
        assert text.encode("ascii") == (h.bench_rows_dir(rig.state_path) / f"{d}.json").read_bytes(), \
            "the persisted bytes, byte for byte"
        assert rig.store.puts.index(key) < report_at, "records land before the report"
    json.loads(rig.store.objects[h.rows_key(w8b.B2, bound[0])])


def test_settlement_without_a_gate_publishes_no_record(tmp_path):
    rig = _rig(tmp_path)
    rig.step(w8b.B1)
    assert not [k for k in rig.store.objects if k.startswith(h.ROWS_PREFIX + "/")]


def test_failed_record_put_is_publish_failed_and_the_court_holds(tmp_path):
    rig = _rig(tmp_path)
    rig.step(w8b.B1)
    before = rig.step.court_for(w8b.B1)
    rig.store.fail_put = lambda key: key.startswith(h.ROWS_PREFIX + "/")
    try:
        rig.step(w8b.B2)
    except Exception as e:  # noqa: BLE001 — the code is checked below
        w8b.assert_wiring_error(e, "publish_failed")
    else:
        raise AssertionError("W3b6b: a failed bench record PUT must be publish_failed")
    assert w8b.wiring().report_key(w8b.B2) not in rig.store.objects, \
        "no report is published while its records are not"
    assert rig.step.court_for(w8b.B2) is None, "an unpublished settlement reached the court"
    assert rig.step.court_for(w8b.B1) == before
    rig.store.fail_put = None
    rig.step(w8b.B2)
    rep = rig.published_report(w8b.B2)
    assert _bound(rep) and all(h.rows_key(w8b.B2, d) in rig.store.objects for d in _bound(rep)), \
        "the retried tick publishes the saved records (catch-up path)"


def test_lane_a_bench_keys_are_untouched(tmp_path):
    rig = _rig(tmp_path)
    for b in (w8b.B1, w8b.B2, w8b.B3):
        rig.step(b)
    assert _bound(rig.published_report(w8b.B2)), "W3b6b: records are bound"
    lane_a = [k for k in rig.store.objects
              if k.startswith("bench_v/") and not k.startswith(h.ROWS_PREFIX + "/")]
    assert lane_a == [], "nothing under lane A's bench_v/round-* or bench_v/rows/*"
