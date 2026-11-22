"""W7a x W5b: the settlement at `[lane_d] until_block` produces a report that
validates against `lane-d-report/1` (plan rev6 §1.5, §3.2 invariant 11, T18).

W5b closes every open entry at `until_block` with status `closed` and reason
`lane_closed`, in `entries[].reason` and in `skipped[].reason`. That reason is
typed (invariant 11), so the schema's `entry_reason` and `skip_reason` enums
must carry it. The fixture is W5b's `until_block` settlement, reused from
tests/accept/w5b/helpers.py (imported in the body, synthetic chain, no torch).

On main the test fails on an `assert`: `validate_report` refuses
`lane_closed` (`bad_value`), which is not yet in the schema's enums.
"""

from __future__ import annotations


def _closing_settlement(tmp_path):
    """W5b's T18 `until_block` chain: two entries settled at BOUNDARY_1, then
    the closing settlement at `until_block` (W5b README "T18 `until_block`")."""
    from tests.accept.w5b import helpers as w5b

    until = w5b.ARM_BLOCK + w5b.ERA
    cfg = w5b.lane_d_cfg(until_block=until)
    assert cfg.active(until) and not cfg.active(until + w5b.SETTLEMENT)
    h = w5b.house()
    first, second = w5b.entry("first"), w5b.entry("second", reveal_ts=w5b.T0 + 3600)
    four = {"src0": 3, "src1": 3, "src2": 3, "src3": 3}
    small = w5b.snapshot_records("a", four, 0)
    big = w5b.snapshot_records("b", {"src0": 6, "src1": 6, "src2": 6, "src3": 6}, 2)
    pa = w5b.write_pool(tmp_path / "pool_a", small, "w5b-a")
    pb = w5b.write_pool(tmp_path / "pool_b", big, "w5b-b")
    fc = w5b.forecasts(small + big, [first["digest"], second["digest"]], h["digest"])
    uid_of = {first["hotkey"]: w5b.CHALLENGER_UID, second["hotkey"]: w5b.SECOND_UID}
    state_path = tmp_path / "state" / "lane_d_state.json"
    s1 = w5b.settle(cfg, w5b.BOUNDARY_1, pa, [first], h, fc, uid_of, state_path)
    closing = w5b.settle(cfg, until, pb, [first, second], h, fc, uid_of, state_path)
    return s1, closing, (first["digest"], second["digest"])


def test_e2e_until_block_closing_report_validates(tmp_path):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    s1, closing, digests = _closing_settlement(tmp_path)
    assert s1.report is not None and closing.report is not None
    validate_report(s1.report)
    r = closing.report
    # the fixture really is the closing settlement (W5b T18)
    listed = {e["digest"]: e for e in r["entries"]}
    for dg in digests:
        assert (listed[dg]["status"], listed[dg]["reason"]) == ("closed", "lane_closed")
    assert {s["digest"]: s["reason"] for s in r["skipped"]} \
        == dict.fromkeys(digests, "lane_closed")
    assert r["cohorts"] == [] and r["judged"] == [] and r["rows"] == []
    try:
        validate_report(r)
    except ReportSchemaError as exc:
        raise AssertionError(
            f"the until_block report does not validate: {exc.code} at {exc.field}: {exc}"
        ) from exc
