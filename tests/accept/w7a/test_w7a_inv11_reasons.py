"""W7a, invariant 11: `pending`, `hold` and `burn` always carry a typed
reason enum, and the report states the exact reason (plan rev6 §3.2
invariant 11, §1.2 A8 queue selection and coldkey resolution, §1.5 B8, B10,
§1.4 D21).

Each case settles real entries through `settle_lane_d` (synthetic chain, no
torch) and asserts the exact reason on the judged set, the skip list, the
entry records and the court slots. On main they fail at the import of
`validate_report` (PATHS) or at an `assert` (main does not list voided
entries as skips, keeps a capped entry's reason at `pending_floor` and has no
`burn_reason`).
"""

from __future__ import annotations

import pytest

from .helpers import (
    BOUNDARY_1,
    BOUNDARY_2,
    CHALLENGER_UID,
    ENTRY_REASONS,
    ENTRY_STATUSES,
    READY_BLOCK,
    REVEAL_BLOCK,
    SKIP_REASONS,
    assert_valid,
    build_chain,
    canonical_rows_sha256,
    coldkey,
    entry,
    fixture_key,
    lane_d_cfg,
    loser_then_crown,
    settle,
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_inv11")


def _check_typed(report):
    """Generic invariant 11 over one report."""
    assert_valid(report)
    for e in report["entries"]:
        assert e["status"] in ENTRY_STATUSES, e
        assert e["reason"] in ENTRY_REASONS, e
    for s in report["skipped"]:
        assert s["reason"] in SKIP_REASONS, s
    for slot in report["pay"]["court"]:
        assert (slot["uid"] is None) == (slot["burn_reason"] is not None), slot
    judged = set(report["judged"])
    assert judged == {v["digest"] for v in report["verdicts"]}
    assert not judged & {s["digest"] for s in report["skipped"]}


def test_inv11_caps_and_voids_are_skips_with_exact_reasons(scratch, tmp_path):
    chain = build_chain(scratch / "caps", n_entries=5)
    a_dg, b_dg, d_dg, e_dg, f_dg = chain["digests"]
    a = entry(a_dg, "a", coldkey=coldkey("c1"), reveal_offset=0)
    b = entry(b_dg, "b", coldkey=coldkey("c1"), reveal_offset=60)          # same coldkey
    d = entry(d_dg, "d", coldkey=coldkey("c2"), reveal_offset=120)         # no UID (B8)
    e = entry(e_dg, "e", coldkey=None, reveal_offset=180)                  # unresolved (A8)
    f = entry(f_dg, "f", coldkey=coldkey("c3"), reveal_offset=240,
              ready_block=REVEAL_BLOCK - 8000)                              # ttl 7200 (B10)
    uid_of = {a["hotkey"]: CHALLENGER_UID, b["hotkey"]: 5, e["hotkey"]: 7, f["hotkey"]: 2}
    cfg = lane_d_cfg(max_judged_per_coldkey_per_settlement=1)
    s = settle(chain, cfg, tmp_path / "lane_d_state.json", fixture_key("signer"),
               boundary=BOUNDARY_1, snap=1, entries=[a, b, d, e, f], uid_of=uid_of)
    r = s.report
    _check_typed(r)
    assert r["judged"] == [a_dg]
    skips = {x["digest"]: x["reason"] for x in r["skipped"]}
    assert len(skips) == len(r["skipped"]), "one skip per entry"
    assert skips == {b_dg: "coldkey_cap_settlement", d_dg: "deregistered",
                     e_dg: "coldkey_unresolved", f_dg: "ready_expired"}
    ents = {x["digest"]: x for x in r["entries"]}
    assert (ents[b_dg]["status"], ents[b_dg]["reason"]) == ("pending", "coldkey_cap_settlement")
    assert ents[d_dg]["reason"] == "deregistered"
    assert ents[e_dg]["reason"] == "coldkey_unresolved" and ents[e_dg]["coldkey"] is None
    assert ents[f_dg]["reason"] == "ready_expired"
    assert ents[a_dg]["coldkey"] == ents[b_dg]["coldkey"] == coldkey("c1")


def test_inv11_floor_miss_is_pending_floor_with_no_rows(scratch, tmp_path):
    chain = build_chain(scratch / "floor")
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    cfg = lane_d_cfg(min_windows=1000)
    s = settle(chain, cfg, tmp_path / "lane_d_state.json", fixture_key("signer"),
               boundary=BOUNDARY_1, snap=1, entries=[e], uid_of={e["hotkey"]: CHALLENGER_UID})
    r = s.report
    _check_typed(r)
    assert r["judged"] == [] and r["verdicts"] == [] and r["rows"] == []
    assert r["rows_sha256"] == canonical_rows_sha256([])
    assert r["skipped"] == [{"digest": e["digest"], "reason": "pending_floor"}]
    (ent,) = r["entries"]
    assert (ent["status"], ent["reason"]) == ("pending", "pending_floor")


def test_inv11_house_king_slot_burns_with_house_king(scratch):
    built = loser_then_crown(scratch / "chain")
    r = built["s1"].report
    _check_typed(r)
    assert [(c["uid"], c["burn_reason"]) for c in r["pay"]["court"]] == [(None, "house_king")]
    assert built["s1"].pay["court_uids"] == [None]


def test_inv11_deregistered_king_slot_burns_with_deregistered(scratch):
    built = loser_then_crown(scratch / "chain")
    _check_typed(built["s2"].report)
    # A third settlement after the crowned king's hotkey lost its UID (B8).
    s3 = settle(built["chain"], built["cfg"], built["state"], built["signer"],
                boundary=BOUNDARY_2 + 900, snap=2, entries=[built["entry"]], uid_of={})
    r = s3.report
    _check_typed(r)
    (slot,) = r["pay"]["court"]
    assert slot["digest"] == built["entry"]["digest"]
    assert (slot["uid"], slot["burn_reason"]) == (None, "deregistered")
    assert s3.pay["court_uids"] == [None]
    assert r["judged"] == [] and r["skipped"] == []


def test_inv11_fixture_blocks_are_consistent():
    """The fixture itself: the TTL case really exceeds W2's default
    `ready_ttl_blocks` and the others do not."""
    cfg = lane_d_cfg()
    assert REVEAL_BLOCK - READY_BLOCK <= cfg.ready_ttl_blocks < 8000
