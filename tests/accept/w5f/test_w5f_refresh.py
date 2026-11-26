"""W5f house refresh through the real `settle_lane_d` (plan rev6 §1.4, §2
items 1-2, §4 T10; README.md has the contract).

Every state comes from real settlements on fixture pools:

- **House-held refresh.** While the house holds KING_D, a new pointer moves
  the throne to the new checkpoint at that boundary (no duel: the house
  only replaces itself). The slot keeps burning (`house_king`).
- **Miner-held refresh.** While a miner holds KING_D, the new house is the
  second bar for every miner cohort from that boundary, and the new house
  enters as a challenger against KING_D on forward windows.
- **T10.** Two real miner crowns (X, then Y) build a court. The pointer is
  swapped to a stronger H1: H1 clears against Y at its first verdict and
  retakes KING_D at once (plan rev6.11 (b): no confirmation duel, for the
  house as for a miner). The king slot burns; Y and X, the prior miner
  kings, keep their tails; the house is never a prior king.

On main the miner-held refresh never judges the house, so the challenger and
T10 tests fail on an assertion; the house-held and second-bar rules are
already on main and W5f owns them as its assertions.
"""

from __future__ import annotations

import pytest

from .helpers import (
    H0,
    H1,
    H2,
    T_H1,
    T_H2,
    UIDS,
    Chain,
    X,
    Y,
    body,
    crown_x,
    fixture_key,
    house,
    miner_court,
    read_state,
    rows_of,
    verdict_of,
)

UID_X, UID_Y = UIDS[X["hotkey"]], UIDS[Y["hotkey"]]
H1W = house("house-1w", T_H1)         # a WEAKER swapped pointer


def _house_record(h: dict) -> dict:
    return {"digest": h["digest"], "hotkey": None, "house": True, "reveal_ts": h["reveal_ts"]}


def _audit(report, cfg):
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import PASS
    from cascade.audit.lane_d import check_lane_d

    res = check_lane_d(report, cfg, signer_hotkey=fixture_key("signer").ss58_address)
    assert res.status == PASS, f"check_lane_d {res.status}: {res.detail}"


# ── the house holds KING_D: the throne moves with the pointer ───────────────


@pytest.fixture(scope="module")
def house_held(tmp_path_factory):
    """boundary(1): genesis H0, no entries; boundary(2): the pointer swaps
    to H1; boundary(3): the same pointer again; boundary(4): H2."""
    c = Chain(tmp_path_factory.mktemp("w5f-house-held") / "chain")
    c.run(1, [], H0)
    states = {1: c.state()}
    for n, h in ((2, H1), (3, H1), (4, H2)):
        c.run(n, [], h)
        states[n] = c.state()
    return c, states


def test_house_held_refresh_moves_the_throne_at_that_boundary(house_held):
    c, states = house_held
    st = states[2]
    assert st["house"] == {"digest": H1["digest"], "reveal_ts": T_H1}, \
        "the state must hold the new house digest and its T"
    k = st["king"]
    assert (k["digest"], k["house"], k["hotkey"], k["reveal_ts"]) == \
        (H1["digest"], True, None, T_H1)
    assert st["prior_kings"] == [], "the house is never a prior king"
    r = c.reports[2]
    assert r["house"] == _house_record(H1), "the report must record the house digest used"
    assert r["king_after"] == _house_record(H1)
    assert r["crowned"] is None, "a house refresh is not a crowning"


def test_house_held_refresh_slot_burns(house_held):
    c, _ = house_held
    for n in (1, 2, 3, 4):
        assert c.pays[n]["court_uids"] == [None], f"boundary {n}: the house slot must burn"
        (slot,) = c.reports[n]["pay"]["court"]
        assert slot["hotkey"] is None and slot["burn_reason"] == "house_king"
    assert c.reports[4]["pay"]["court"][0]["digest"] == H2["digest"]


def test_house_held_refresh_is_idempotent(house_held):
    c, states = house_held
    for key in ("house", "king", "prior_kings", "entries"):
        assert states[3][key] == states[2][key], \
            f"re-applying the same pointer changed the state's {key}"
    assert c.reports[3]["house"] == c.reports[2]["house"]
    assert states[4]["king"]["digest"] == H2["digest"] and states[4]["king"]["house"]
    assert states[4]["house"] == {"digest": H2["digest"], "reveal_ts": T_H2}


def test_house_held_refresh_reports_pass_the_audit(house_held):
    c, _ = house_held
    for n in (1, 2, 3, 4):
        _audit(c.reports[n], c.cfg)


# ── a miner holds KING_D: a weaker pointer is a bar and a losing challenger ─


@pytest.fixture(scope="module")
def weak_swap(tmp_path_factory):
    """X crowned; at boundary(3) the pointer swaps to a WEAK H1w and Y
    (good) is judged against KING_D = X with H1w as the second bar, and is
    crowned at that first verdict (rev6.11 (b)); boundary(4) judges no
    challenger."""
    h0 = H0
    c = Chain(tmp_path_factory.mktemp("w5f-weak-swap") / "chain")
    crown_x(c)
    c.model(H1W["digest"], "bad", "house-1w")
    c.model(Y["digest"], "good", "y")
    c.run(3, [X, Y], H1W, candidates=[Y["digest"]],
          references=[X["digest"], H1W["digest"], h0["digest"]])
    s3 = c.state()
    c.run(4, [X, Y], H1W, candidates=[Y["digest"]],
          references=[X["digest"], H1W["digest"], h0["digest"]])
    return c, s3


def _cohort_of(report, dg) -> dict:
    found = [co for co in report["cohorts"] if dg in co["members"]]
    assert len(found) == 1, f"{dg[:12]} is not in exactly one cohort"
    return found[0]


def test_miner_held_refresh_new_house_is_no_second_bar(weak_swap):
    """DEC-CA-0076 (owner 2026-11-24, laned:tests-correction): after the
    first miner crown the house is no bar — a refreshed house is recorded
    but Y is judged against KING_D alone (was: the new house as the second
    bar, W5c change 3)."""
    c, s3 = weak_swap
    r = c.reports[3]
    assert r["house"] == _house_record(H1W)
    assert s3["house"] == {"digest": H1W["digest"], "reveal_ts": T_H1}
    assert r["king_before"]["digest"] == X["digest"] and not r["king_before"]["house"], \
        "a miner KING_D keeps its throne on a house refresh"
    assert r["crowned"] == Y["digest"]
    assert s3["king"]["digest"] == Y["digest"] and not s3["king"]["house"]
    co = _cohort_of(r, Y["digest"])
    assert co["king_digest"] == X["digest"]
    assert "house_digest" not in co, "the house is a second bar after a miner crown"
    assert not rows_of(r, H1W["digest"]), "the house was scored after a miner crown"
    assert not rows_of(r, H0["digest"])
    v = verdict_of(r, Y["digest"])
    assert v is not None and "house" not in v
def test_miner_held_refresh_new_house_enters_no_challenger(weak_swap):
    """DEC-CA-0076 (laned:tests-correction; was §2 item 2, the house duels a
    miner KING_D): a refreshed house is never judged after a miner crown."""
    c, s3 = weak_swap
    for n in (3, 4):
        assert verdict_of(c.reports[n], H1W["digest"]) is None, \
            f"boundary {n}: the refreshed house was judged against a miner KING_D"
        assert not c.reports[n]["king_after"]["house"], f"boundary {n}: the house took KING_D"
    assert "house_entry" not in s3
def test_miner_held_refresh_weak_house_never_enters_the_court(weak_swap):
    c, _ = weak_swap
    # rev6.11 (b): Y is crowned at boundary(3), its first verdict.
    assert c.pays[3]["court_uids"] == [UID_Y, UID_X]
    st = c.state()
    assert st["king"]["digest"] == Y["digest"], "fixture: Y did not crown over X and H1w"
    assert [p["digest"] for p in st["prior_kings"]] == [X["digest"]]
    assert c.pays[4]["court_uids"] == [UID_Y, UID_X]
    for n in (3, 4):
        assert all(s["digest"] != H1W["digest"] for s in c.reports[n]["pay"]["court"])


def test_miner_held_refresh_reports_pass_the_audit(weak_swap):
    c, _ = weak_swap
    for n in (3, 4):
        _audit(c.reports[n], c.cfg)


# ── T10: a stronger pointer retakes KING_D from a miner court ───────────────


@pytest.fixture(scope="module")
def t10(tmp_path_factory):
    """boundary(1)..(4): X then Y crowned (court [Y, X]); boundary(5): the
    pointer swaps to the stronger H1, which retakes at its first verdict
    (rev6.11 (b)); a restart copy is taken after boundary(5); boundary(6):
    the same pointer, the house holds KING_D; boundary(7): the pointer moves
    on to H2 while the house holds KING_D."""
    base = tmp_path_factory.mktemp("w5f-t10")
    c = Chain(base / "chain")
    miner_court(c)
    refs = [Y["digest"], X["digest"], H0["digest"]]
    c.model(H1["digest"], "sharp", "house-1")
    c.run(5, [X, Y], H1, candidates=[H1["digest"]], references=refs)
    s5 = c.state()
    restart = c.copy_to(base / "restart")
    c.run(6, [X, Y], H1, candidates=[H1["digest"]], references=refs)
    s6 = c.state()
    c.model(H2["digest"], "sharp", "house-2")
    c.run(7, [X, Y], H2, candidates=[H2["digest"]], references=[H1["digest"], *refs])
    return {"chain": c, "s5": s5, "s6": s6, "restart": restart, "refs": refs}


def test_t10_stronger_house_never_retakes(t10):
    """DEC-CA-0076 (owner 2026-11-24, laned:tests-correction; was T10: a
    stronger house retakes KING_D): the stronger H1 is recorded as the house
    and never judged; Y keeps KING_D and the court stays the miner kings."""
    c, s5 = t10["chain"], t10["s5"]
    r = c.reports[5]
    assert r["house"] == _house_record(H1)
    assert verdict_of(r, H1["digest"]) is None, "the house challenged a miner KING_D"
    assert r["king_before"]["digest"] == Y["digest"]
    assert r["king_after"] == r["king_before"], "the house retook KING_D"
    assert r["crowned"] is None
    k = s5["king"]
    assert (k["digest"], k["house"]) == (Y["digest"], False)
    assert c.pays[5]["court_uids"] == [UID_Y, UID_X]
def test_t10_the_house_never_holds_king_d_after_a_miner_crown(t10):
    """DEC-CA-0076 (laned:tests-correction): boundaries 5-7, the house
    refreshed twice (H1, then H2), never KING_D and never judged."""
    c = t10["chain"]
    for n in (5, 6, 7):
        r = c.reports[n]
        assert not r["king_after"]["house"], f"boundary {n}: the house holds KING_D"
        assert verdict_of(r, r["house"]["digest"]) is None
    assert "house_entry" not in t10["s5"] and "house_entry" not in t10["s6"]
def test_t10_the_court_is_the_miner_kings_with_no_house_slot(t10):
    """DEC-CA-0076 (laned:tests-correction; was T10's burning house slot):
    the court is KING_D then the prior miner kings, every slot paid."""
    c, s6 = t10["chain"], t10["s6"]
    assert [p["digest"] for p in s6["prior_kings"]] == [X["digest"]]
    assert c.pays[6]["court_uids"] == [UID_Y, UID_X]
    court = c.reports[6]["pay"]["court"]
    assert [(s["digest"], s["uid"], s["burn_reason"]) for s in court] == \
        [(Y["digest"], UID_Y, None), (X["digest"], UID_X, None)]
    assert c.reports[6]["pay"]["share"] == c.cfg.share
def test_t10_a_later_house_refresh_keeps_the_miner_court(t10):
    """boundary(7): H2 is recorded as the house; KING_D and the court are
    untouched (DEC-CA-0076, laned:tests-correction)."""
    c = t10["chain"]
    st = c.state()
    assert st["king"]["digest"] == Y["digest"] and not st["king"]["house"]
    assert st["house"] == {"digest": H2["digest"], "reveal_ts": T_H2}
    assert [p["digest"] for p in st["prior_kings"]] == [X["digest"]]
    assert c.pays[7]["court_uids"] == [UID_Y, UID_X]
    assert verdict_of(c.reports[7], H2["digest"]) is None
def test_t10_reports_pass_the_audit(t10):
    c = t10["chain"]
    for n in (5, 6, 7):
        _audit(c.reports[n], c.cfg)


def test_t10_report_verifies_with_the_validator_key(t10):
    pytest.importorskip("bittensor.sp_core")
    from cascade.validator.lane_d_report import verify_report

    for n in (5, 6, 7):
        assert verify_report(t10["chain"].reports[n], fixture_key("signer").ss58_address)


def test_t10_retake_is_restart_safe(t10):
    """A restart after boundary(5) (the state file and its reports copied, a
    fresh call) settles boundary(6) to the same state and report body."""
    c, restart = t10["chain"], t10["restart"]
    s = c.run(6, [X, Y], H1, candidates=[H1["digest"]], references=t10["refs"],
              state_path=restart)
    assert body(s.report) == body(c.reports[6])
    assert s.pay == c.pays[6]
    assert read_state(restart) == t10["s6"]


def test_t10_no_house_entry_is_ever_registered(t10):
    """DEC-CA-0076 (laned:tests-correction; was: the house entry is judged
    once): no settlement judges the house after a miner crown."""
    c = t10["chain"]
    assert [n for n in (5, 6, 7) if verdict_of(c.reports[n], H1["digest"]) is not None
            or verdict_of(c.reports[n], H2["digest"]) is not None] == []
def test_t10_exact_copy_of_the_house_is_still_refused_at_dedup(tmp_path):
    """T2 alongside T10: a miner entry whose digest is the swapped house's is
    refused (`duplicate_digest`), never judged."""
    c = Chain(tmp_path / "chain")
    crown_x(c)
    copy_entry = {**Y, "digest": H1["digest"]}
    c.model(H1["digest"], "sharp", "house-1")
    c.run(3, [X, copy_entry], H1, candidates=[H1["digest"]],
          references=[X["digest"], H0["digest"]])
    skipped = [(s["digest"], s["reason"]) for s in c.reports[3]["skipped"]]
    assert (H1["digest"], "duplicate_digest") in skipped, \
        "a resubmitted house digest must be refused at dedup"
    assert all(Y["hotkey"] != e["hotkey"] or e["status"] == "closed"
               for e in c.reports[3]["entries"]), "the copy must be closed, never judged"
