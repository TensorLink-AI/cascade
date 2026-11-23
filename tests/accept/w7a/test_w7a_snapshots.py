"""W7a (RV4 correction, for W5b #581): the report names every snapshot a
judged window's version came from (plan rev6 §3.0 W5b row, §1.5, §3.2
invariant 4, §14.3 item 2; review evidence/rv4.md, W5b BLOCKING items).

Two **required** fields (README "Decisions pinned here"):

- `pool.snapshots`: `[{snapshot, sha256, first_seen_roots}]`, each snapshot
  once (by name and by sha256). `pool.snapshot` / `pool.sha256` stay the
  newest snapshot's and it is always listed. `pool.first_seen_roots` is the
  sorted union of the listed snapshots' roots.
- `windows[].snapshot_sha256`: the sha256 of the snapshot whose version was
  scored, always one of `pool.snapshots[].sha256`.

Which snapshot a window's version comes from is W5b's rule; these tests pin
the shape and the structural rules, on reports spliced from a real
`settle_lane_d` report (one snapshot, then a two-snapshot union). On main
they fail on an `assert`: the fields are absent from the report, and the
closed schema refuses them (`unknown_field`).
"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    POOL_SNAPSHOT_FIELDS,
    ROOTS,
    assert_valid,
    loser_then_crown,
    with_snapshots,
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_snapshots")


def _older(scratch) -> dict:
    """Snapshot 1 as an older member of BOUNDARY_2's union."""
    p1 = loser_then_crown(scratch)["s1"].report["pool"]
    return {"snapshot": p1["snapshot"], "sha256": p1["sha256"],
            "first_seen_roots": list(ROOTS[1])}


def _one(scratch) -> dict:
    return with_snapshots(loser_then_crown(scratch)["s2"].report)


def _two(scratch) -> dict:
    return with_snapshots(loser_then_crown(scratch)["s2"].report, older=_older(scratch))


def _refused(report, code, field):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as info:
        validate_report(report)
    got = (info.value.code, info.value.field)
    assert got == (code, field), f"refused {got}, want {(code, field)}"


# ── what settle_lane_d emits ────────────────────────────────────────────────


def test_e2e_report_lists_its_snapshots_and_each_windows_version(scratch):
    built = loser_then_crown(scratch)
    for s in (built["s1"], built["s2"]):
        pool = s.report["pool"]
        snaps = pool.get("snapshots")
        assert isinstance(snaps, list), "report.pool.snapshots is missing"
        assert {"snapshot": pool["snapshot"], "sha256": pool["sha256"]} in [
            {"snapshot": x["snapshot"], "sha256": x["sha256"]} for x in snaps], \
            "the newest snapshot (pool.snapshot / pool.sha256) is not listed"
        assert len({x["snapshot"] for x in snaps}) == len(snaps)
        assert len({x["sha256"] for x in snaps}) == len(snaps)
        union = sorted({r for x in snaps for r in x["first_seen_roots"]})
        assert pool["first_seen_roots"] == union
        listed = {x["sha256"] for x in snaps}
        assert s.report["windows"], "the rich chain judges windows"
        for w in s.report["windows"]:
            assert w.get("snapshot_sha256") in listed, \
                f"{w['window_id']}: snapshot_sha256 {w.get('snapshot_sha256')!r} not listed"
        assert_valid(s.report)


# ── a report carrying them validates ────────────────────────────────────────


def test_single_snapshot_report_validates(scratch):
    assert_valid(_one(scratch))


def test_two_snapshot_union_report_validates(scratch):
    r = _two(scratch)
    assert len(r["pool"]["snapshots"]) == 2
    assert r["windows"][0]["snapshot_sha256"] != r["pool"]["sha256"]
    assert_valid(r)


# ── required, closed ────────────────────────────────────────────────────────


def test_pool_snapshots_is_required(scratch):
    r = _one(scratch)
    del r["pool"]["snapshots"]
    _refused(r, "missing_field", "pool.snapshots")


def test_window_snapshot_sha256_is_required(scratch):
    r = _one(scratch)
    del r["windows"][0]["snapshot_sha256"]
    _refused(r, "missing_field", "windows[0].snapshot_sha256")


@pytest.mark.parametrize("name", POOL_SNAPSHOT_FIELDS)
def test_every_snapshot_field_is_required(scratch, name):
    r = _two(scratch)
    del r["pool"]["snapshots"][1][name]
    _refused(r, "missing_field", f"pool.snapshots[1].{name}")


def test_extra_key_in_a_snapshot_item_is_refused(scratch):
    r = _two(scratch)
    r["pool"]["snapshots"][0]["horizon"] = 64
    _refused(r, "unknown_field", "pool.snapshots[0].horizon")
    r = _two(scratch)
    r["pool"]["snapshots"][1]["note"] = "newest"
    _refused(r, "unknown_field", "pool.snapshots[1].note")


# ── values ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["abc", "Z" * 64, "A" * 64, "0x" + "a" * 64, "a" * 63, ""])
def test_bad_snapshot_sha256_is_refused(scratch, bad):
    r = _two(scratch)
    r["pool"]["snapshots"][0]["sha256"] = bad
    _refused(r, "bad_value", "pool.snapshots[0].sha256")


def test_bad_window_snapshot_sha256_is_refused(scratch):
    r = _one(scratch)
    r["windows"][0]["snapshot_sha256"] = "not-a-digest"
    _refused(r, "bad_value", "windows[0].snapshot_sha256")
    r = _one(scratch)
    r["windows"][0]["snapshot_sha256"] = None
    _refused(r, "bad_type", "windows[0].snapshot_sha256")


def test_snapshot_types(scratch):
    r = _two(scratch)
    r["pool"]["snapshots"] = {"w7a-2": r["pool"]["sha256"]}
    _refused(r, "bad_type", "pool.snapshots")
    r = _two(scratch)
    r["pool"]["snapshots"][0]["snapshot"] = 1
    _refused(r, "bad_type", "pool.snapshots[0].snapshot")
    r = _two(scratch)
    r["pool"]["snapshots"][0]["first_seen_roots"] = ["not-a-root"]
    _refused(r, "bad_value", "pool.snapshots[0].first_seen_roots[0]")


def test_duplicate_snapshot_is_refused(scratch):
    """Each snapshot is listed once: a repeated name or sha256 is refused at
    the later item."""
    r = _two(scratch)
    r["pool"]["snapshots"].append(copy.deepcopy(r["pool"]["snapshots"][0]))
    _refused(r, "bad_value", "pool.snapshots[2].snapshot")
    r = _two(scratch)
    r["pool"]["snapshots"][1]["sha256"] = r["pool"]["snapshots"][0]["sha256"]
    _refused(r, "bad_value", "pool.snapshots[1].sha256")


def test_window_must_name_a_listed_snapshot(scratch):
    r = _one(scratch)
    r["windows"][0]["snapshot_sha256"] = _older(scratch)["sha256"]
    _refused(r, "bad_value", "windows[0].snapshot_sha256")


def test_newest_snapshot_must_be_listed(scratch):
    """`pool.snapshot` / `pool.sha256` (the newest, the seed's) are listed."""
    r = _two(scratch)
    r["pool"]["snapshots"] = r["pool"]["snapshots"][:1]
    for w in r["windows"]:
        w["snapshot_sha256"] = r["pool"]["snapshots"][0]["sha256"]
    _refused(r, "bad_value", "pool.snapshots")
