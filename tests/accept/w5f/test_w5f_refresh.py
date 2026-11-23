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
  swapped to a stronger H1: H1 first-clears against Y, confirms on the next
  disjoint set and retakes KING_D. The king slot burns; Y and X, the prior
  miner kings, keep their tails; the house is never a prior king.

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
    """X crowned at boundary(2); at boundary(3) the pointer swaps to a WEAK
    H1w and Y (good) is judged against KING_D = X; boundary(4) confirms Y."""
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


def test_miner_held_refresh_new_house_is_the_second_bar_from_that_boundary(weak_swap):
    c, s3 = weak_swap
    r = c.reports[3]
    assert r["house"] == _house_record(H1W)
    assert s3["house"] == {"digest": H1W["digest"], "reveal_ts": T_H1}
    assert s3["king"]["digest"] == X["digest"] and not s3["king"]["house"], \
        "a miner KING_D keeps its throne on a house refresh"
    co = _cohort_of(r, Y["digest"])
    assert co["king_digest"] == X["digest"]
    assert co.get("house_digest") == H1W["digest"], \
        "Y must be judged against the NEW house as the second bar"
    assert rows_of(r, H1W["digest"]) >= set(co["window_ids"])
    assert not rows_of(r, H0["digest"]), "the old house must not be scored after the swap"
    v = verdict_of(r, Y["digest"])
    assert v is not None and v["house"]["digest"] == H1W["digest"]


def test_miner_held_refresh_new_house_enters_as_a_challenger(weak_swap):
    """§1.4 "It enters at that settlement" / §2 item 2: the house duels a
    miner KING_D forward-only. A weaker house loses and never retakes."""
    c, _ = weak_swap
    v = verdict_of(c.reports[3], H1W["digest"])
    assert v is not None, "the refreshed house was not judged against the miner KING_D"
    assert v["king_digest"] == X["digest"]
    assert v["cleared"] is False, "a weaker house must not clear the miner KING_D"
    for n in (3, 4):
        st_king = c.reports[n]["king_after"]
        assert not st_king["house"], f"boundary {n}: a weaker house took KING_D"


def test_miner_held_refresh_weak_house_never_enters_the_court(weak_swap):
    c, _ = weak_swap
    assert c.pays[3]["court_uids"] == [UID_X]
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
    """boundary(1)..(4): X then Y crowned (court [Y, X]); boundary(5)/(6):
    the pointer swaps to the stronger H1 and retakes; a restart copy is taken
    after boundary(5); boundary(7): the pointer moves on to H2 while the
    house holds KING_D."""
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


def test_t10_stronger_house_first_clears_and_is_not_yet_king(t10):
    c, s5 = t10["chain"], t10["s5"]
    r = c.reports[5]
    assert r["house"] == _house_record(H1)
    v = verdict_of(r, H1["digest"])
    assert v is not None, "the swapped house was not judged against the miner KING_D"
    assert v["king_digest"] == Y["digest"] and v["stage"] == "first"
    assert v["cleared"] is True, v
    co = r["cohorts"][v["cohort"]]
    assert co["members"] == [H1["digest"]], "the house challenger is judged in its own cohort"
    assert co["king_digest"] == Y["digest"]
    assert "house_digest" not in co, "the house is never its own second bar"
    assert "house" not in v
    assert rows_of(r, H1["digest"]) == set(co["window_ids"])
    assert r["king_after"]["digest"] == Y["digest"] and not r["king_after"]["house"], \
        "the mandatory confirmation duel applies to the house too"
    assert s5["king"]["digest"] == Y["digest"]
    assert c.pays[5]["court_uids"] == [UID_Y, UID_X]


def test_t10_house_retakes_after_confirmation(t10):
    c, s6 = t10["chain"], t10["s6"]
    r = c.reports[6]
    v = verdict_of(r, H1["digest"])
    assert v is not None, "the house's confirmation duel did not run"
    assert v["king_digest"] == Y["digest"] and v["stage"] == "confirm"
    assert v["cleared"] is True, v
    assert not (set(r["cohorts"][v["cohort"]]["window_ids"])
                & set(c.reports[5]["cohorts"][verdict_of(c.reports[5],
                                                        H1["digest"])["cohort"]]["window_ids"])), \
        "the house's confirmation set shares a window with its first-clear set"
    assert r["king_before"]["digest"] == Y["digest"]
    assert r["king_after"] == _house_record(H1), "T10: the house retakes KING_D"
    assert r["house"] == _house_record(H1)
    k = s6["king"]
    assert (k["digest"], k["house"], k["hotkey"]) == (H1["digest"], True, None)


def test_t10_slot_burns_and_prior_miner_tails_keep_paying(t10):
    c, s6 = t10["chain"], t10["s6"]
    assert [p["digest"] for p in s6["prior_kings"]] == [Y["digest"], X["digest"]], \
        "the dethroned miner king must become the newest prior king"
    assert c.pays[6]["court_uids"] == [None, UID_Y, UID_X], \
        "T10: the king slot burns and both prior miner tails keep paying"
    court = c.reports[6]["pay"]["court"]
    assert court[0]["digest"] == H1["digest"] and court[0]["hotkey"] is None
    assert court[0]["burn_reason"] == "house_king"
    assert [(s["digest"], s["uid"], s["burn_reason"]) for s in court[1:]] == \
        [(Y["digest"], UID_Y, None), (X["digest"], UID_X, None)]
    assert c.reports[6]["pay"]["share"] == c.cfg.share


def test_t10_house_held_refresh_after_the_retake_keeps_the_tails(t10):
    """boundary(7): H2 replaces H1 in place; H1 is never a prior king."""
    c = t10["chain"]
    st = c.state()
    assert st["king"]["digest"] == H2["digest"] and st["king"]["house"]
    assert st["house"] == {"digest": H2["digest"], "reveal_ts": T_H2}
    assert [p["digest"] for p in st["prior_kings"]] == [Y["digest"], X["digest"]], \
        "a house-held refresh must not touch the miner tails"
    assert all(p["digest"] not in (H0["digest"], H1["digest"]) for p in st["prior_kings"])
    assert c.pays[7]["court_uids"] == [None, UID_Y, UID_X]
    assert c.reports[7]["king_after"] == _house_record(H2)
    assert verdict_of(c.reports[7], H2["digest"]) is None, \
        "a house-held refresh replaces the house in place: no duel"


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


def test_t10_house_entry_is_registered_once(t10):
    """The same pointer at boundary(5) and (6) is one house entry: its first
    clear is kept, never restarted by re-applying the pointer."""
    c = t10["chain"]
    v6 = verdict_of(c.reports[6], H1["digest"])
    assert v6 is not None and v6["confirmation"]["first_set_sha256"] == \
        c.reports[5]["cohorts"][verdict_of(c.reports[5], H1["digest"])["cohort"]]["set_sha256"]
    assert v6["confirmation"]["first_king_digest"] == Y["digest"]


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
