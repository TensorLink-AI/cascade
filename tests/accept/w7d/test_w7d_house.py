"""W7d replay of the W5f house challenger (plan rev6 §1.4 "The house", §2
items 1-2, §4 T10, §14.3; README.md "House-challenger chains").

Every chain is real `settle_lane_d` runs (`house_helpers.HOUSE_CHAINS`,
built with the W5f helpers). The replay must MATCH each honest chain: a new
house that challenges a miner KING_D alone in its own cohort (`members ==
[house]`, `king_digest` = KING_D, no `house_digest`), whether it loses or
clears and retakes KING_D at its first verdict (plan rev6.11 (b): no
confirmation duel; the miner king the newest prior king, the slot burns
`house_king`, `crowned` null), a miner copy of the
house closed `duplicate_digest` (at registration, or an open entry the new
pointer names), and a later miner dethroning the house.

Forgeries are re-signed by the validator's own key, their receipts re-issued:
a house cohort with an extra member, a retake with `crowned` set, a retake
that leaves the dethroned miner king out of the prior kings, and a
`duplicate_digest` close of an entry that is not a copy. Each is a MISMATCH
at its field path; every earlier settlement still MATCHes.

On main the replay judges every cohort as miner members (the house is not a
listed entry) and gives the house a second bar against itself, so the honest
chains are MISMATCH and every case here fails on an assertion; the fixture
facts pass.
"""

from __future__ import annotations

import math

import pytest

from .helpers import describe, replay, replay_module
from .house_helpers import (
    COPY,
    H1,
    H1W,
    HOUSE_CHAINS,
    UIDS,
    C,
    X,
    Y,
    Z,
    doctored,
    entry_index,
    has_path,
    house_cohort_index,
    king_record,
    signer_key,
    verdict_index,
)

# rev6.11 (b): the house retakes at its first verdict.
RETAKES = (("house_retakes", 5), ("copy_at_swap", 3), ("open_copy", 4))


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    root = tmp_path_factory.mktemp("w7d-house")
    return {name: build(root / name) for name, build in HOUSE_CHAINS.items()}


def _house_record(h: dict) -> dict:
    return {"digest": h["digest"], "hotkey": None, "house": True, "reveal_ts": h["reveal_ts"]}


def _all_match(c, **kw):
    mod = replay_module()
    res = replay(c.pub, **kw)
    assert len(res) == len(c.pub.boundaries), describe(res)
    assert [r.boundary for r in res] == c.pub.boundaries
    assert all(r.status == mod.MATCH for r in res), ("the honest chain", describe(res))
    return res


# ── the fixture facts (pass on main: they check the producer's chains) ──────


def test_fixture_house_loses(chains):
    c = chains["house_loses"]
    r3 = c.reports[3]
    i = house_cohort_index(r3)
    assert i is not None, "fixture: the weak house was not judged in its own cohort"
    co = r3["cohorts"][i]
    assert co["king_digest"] == X["digest"] and "house_digest" not in co
    v = r3["verdicts"][verdict_index(r3, H1W["digest"])]
    assert (v["stage"], v["cleared"], v["outcome"]) == ("first", False, "lost")
    yc = r3["cohorts"][r3["verdicts"][verdict_index(r3, Y["digest"])]["cohort"]]
    assert yc["house_digest"] == H1W["digest"], "fixture: H1w is not Y's second bar"
    assert r3["crowned"] == Y["digest"], "fixture: Y is crowned at its first verdict"
    r4 = c.reports[4]
    assert house_cohort_index(r4) is None, "fixture: a lost house was judged again"
    assert r4["crowned"] is None and r4["king_after"]["digest"] == Y["digest"]


def test_fixture_house_retakes(chains):
    c = chains["house_retakes"]
    r5, r6 = c.reports[5], c.reports[6]
    v5 = r5["verdicts"][verdict_index(r5, H1["digest"])]
    assert (v5["stage"], v5["cleared"], v5["outcome"]) == ("first", True, "crowned")
    assert r5["crowned"] is None and r5["king_after"] == _house_record(H1)
    assert [p["digest"] for p in r5["prior_kings"]] == [Y["digest"], X["digest"]]
    assert r5["pay"]["court"][0]["burn_reason"] == "house_king"
    assert verdict_index(r6, H1["digest"]) is None, "fixture: the house was judged again"
    assert r6["king_after"] == _house_record(H1)
    r7, r8 = c.reports[7], c.reports[8]
    assert r7["crowned"] == Z["digest"] and r7["king_before"] == _house_record(H1)
    assert [p["digest"] for p in r7["prior_kings"]] == [Y["digest"], X["digest"]], \
        "fixture: a crown over the house pushed it onto the prior kings"
    assert r8["crowned"] is None and r8["king_after"]["digest"] == Z["digest"]


def test_fixture_copies_are_closed_duplicate_digest(chains):
    r = chains["copy_at_swap"].reports[3]
    e = r["entries"][entry_index(r, COPY["digest"])]
    assert (e["status"], e["reason"]) == ("closed", "duplicate_digest")
    assert r["verdicts"][verdict_index(r, H1["digest"])]["stage"] == "first"
    c = chains["open_copy"]
    r3, r4 = c.reports[3], c.reports[4]
    # b3 has no gate inputs for C, so its first clear is held (rev6.11 (b)).
    assert r3["verdicts"][verdict_index(r3, C["digest"])]["outcome"] == "cleared"
    e3 = r3["entries"][entry_index(r3, C["digest"])]
    assert (e3["status"], e3["reason"]) == ("hold", "bench_hold")
    e = r4["entries"][entry_index(r4, C["digest"])]
    assert (e["status"], e["reason"]) == ("closed", "duplicate_digest")
    v = r4["verdicts"][verdict_index(r4, H1["digest"])]
    assert v["stage"] == "first", "fixture: the house inherited the copy's first clear"
    assert all(r.get("held_from") is None for r in (r4.get("gate") or {}).get("results", [])), \
        "fixture: the house inherited the copy's hold"
    assert r4["king_after"] == _house_record(H1)
    assert c.reports[5]["king_after"] == _house_record(H1)


# ── honest chains MATCH ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(HOUSE_CHAINS))
def test_honest_house_chain_matches_from_files(chains, name):
    _all_match(chains[name])


@pytest.mark.parametrize("name", sorted(HOUSE_CHAINS))
def test_honest_house_chain_matches_with_states_and_metagraphs(chains, name):
    c = chains[name]
    _all_match(c, states=dict(c.pub.states), uid_of=dict(c.pub.uid_of))


@pytest.mark.parametrize("name", sorted(HOUSE_CHAINS))
def test_honest_house_chain_matches_from_a_mapping(chains, name):
    c = chains[name]
    _all_match(c, reports=c.pub.reports())


def _at(res, c, n):
    """The result for boundary(n) (the chains settle boundary 1, 2, ...)."""
    assert res[n - 1].boundary == c.pub.boundaries[n - 1]
    return res[n - 1]


def test_replay_derives_a_losing_house_challenger(chains):
    c = chains["house_loses"]
    res = _all_match(c, uid_of=dict(c.pub.uid_of))
    r3 = _at(res, c, 3)
    assert dict(r3.outcomes) == {Y["digest"]: "crowned", H1W["digest"]: "lost"}
    assert r3.crowned == Y["digest"] and r3.king_after == Y["digest"]
    assert [(d, u) for d, u, _ in r3.pay] == [(Y["digest"], UIDS[Y["hotkey"]]),
                                             (X["digest"], UIDS[X["hotkey"]])]
    r4 = _at(res, c, 4)
    assert dict(r4.outcomes) == {}, "a lost house or a crowned king is not judged again"
    assert r4.crowned is None and r4.king_after == Y["digest"]
    assert [(d, u) for d, u, _ in r4.pay] == [(Y["digest"], UIDS[Y["hotkey"]]),
                                             (X["digest"], UIDS[X["hotkey"]])]


def test_replay_derives_the_house_first_clear_and_retake(chains):
    c = chains["house_retakes"]
    res = _all_match(c, uid_of=dict(c.pub.uid_of))
    r5 = _at(res, c, 5)
    assert dict(r5.outcomes) == {H1["digest"]: "crowned"}
    assert dict(r5.gate) == {H1["digest"]: "pass"}
    assert r5.crowned is None, "a retake is not a miner crowning"
    assert r5.king_after == H1["digest"]
    assert [(d, u) for d, u, _ in r5.pay] == [(H1["digest"], None),
                                             (Y["digest"], UIDS[Y["hotkey"]]),
                                             (X["digest"], UIDS[X["hotkey"]])]
    assert math.isclose(sum(s for _, _, s in r5.pay), c.cfg.share, rel_tol=1e-12)
    shares = [s for _, _, s in r5.pay]
    assert shares[0] > shares[1] > shares[2] > 0
    r6 = _at(res, c, 6)
    assert dict(r6.outcomes) == {}, "the house is not judged after its retake"
    assert r6.crowned is None and r6.king_after == H1["digest"]


def test_replay_derives_a_miner_dethroning_the_house(chains):
    c = chains["house_retakes"]
    res = _all_match(c, uid_of=dict(c.pub.uid_of))
    r7 = _at(res, c, 7)
    assert dict(r7.outcomes) == {Z["digest"]: "crowned"}
    assert r7.crowned == Z["digest"] and r7.king_after == Z["digest"]
    assert [(d, u) for d, u, _ in r7.pay] == [(Z["digest"], UIDS[Z["hotkey"]]),
                                             (Y["digest"], UIDS[Y["hotkey"]]),
                                             (X["digest"], UIDS[X["hotkey"]])]
    r8 = _at(res, c, 8)
    assert dict(r8.outcomes) == {}
    assert r8.crowned is None and r8.king_after == Z["digest"]


@pytest.mark.parametrize("name,n", [("copy_at_swap", 3), ("open_copy", 4)])
def test_replay_derives_a_retake_after_a_copy_closed(chains, name, n):
    c = chains[name]
    res = _all_match(c, uid_of=dict(c.pub.uid_of))
    r = _at(res, c, n)
    assert r.crowned is None and r.king_after == H1["digest"]
    assert dict(r.outcomes) == {H1["digest"]: "crowned"}
    assert [(d, u) for d, u, _ in r.pay] == [(H1["digest"], None),
                                            (X["digest"], UIDS[X["hotkey"]])]
    # rev6.11 (b): the retake is the house's first verdict; it was never
    # judged before, and is not judged after.
    before = _at(res, c, n - 1)
    assert H1["digest"] not in dict(before.outcomes) or name == "open_copy"
    after = _at(res, c, n + 1)
    assert dict(after.outcomes) == {} and after.king_after == H1["digest"]


# ── forgeries are MISMATCH at their path ────────────────────────────────────


def _mismatch_at(c, n, mutate, path):
    mod = replay_module()
    _all_match(c, uid_of=dict(c.pub.uid_of))
    reports, receipts = doctored(c.pub, n, mutate, key=signer_key())
    res = replay(c.pub, reports=reports, receipts=receipts, uid_of=dict(c.pub.uid_of))
    assert len(res) == len(c.pub.boundaries), describe(res)
    for r in res[:n - 1]:
        assert r.status == mod.MATCH, ("an earlier settlement stopped matching", describe(res))
    r = res[n - 1]
    assert r.status == mod.MISMATCH, describe(res)
    assert has_path(r, path), (path, r.mismatches)
    for r in res[n:]:
        assert (r.status, r.reason) == (mod.UNVERIFIABLE, "prior_unverified"), describe(res)


@pytest.mark.parametrize("name,n,extra", [("house_retakes", 5, "x"), ("house_loses", 3, "y")])
def test_house_cohort_with_an_extra_member_is_a_mismatch(chains, name, n, extra):
    c = chains[name]
    i = house_cohort_index(c.reports[n])
    assert i is not None
    add = {"x": X, "y": Y}[extra]["digest"]

    def mutate(rep):
        co = rep["cohorts"][i]
        co["members"] = [add, *co["members"]]

    _mismatch_at(c, n, mutate, f"cohorts[{i}].members")


@pytest.mark.parametrize("name,n", RETAKES)
def test_retake_with_crowned_set_is_a_mismatch(chains, name, n):
    def mutate(rep):
        rep["crowned"] = rep["house"]["digest"]

    _mismatch_at(chains[name], n, mutate, "crowned")


@pytest.mark.parametrize("name,n", RETAKES)
def test_retake_without_the_miner_king_in_the_prior_kings_is_a_mismatch(chains, name, n):
    c = chains[name]
    before = [king_record(p) for p in c.reports[n - 1]["prior_kings"]]
    keep = {p["digest"] for p in before}

    def mutate(rep):
        # Self-consistent: the court drops the dethroned miner king too.
        rep["prior_kings"] = [dict(p) for p in before]
        court = rep["pay"]["court"]
        rep["pay"]["court"] = [court[0], *[s for s in court[1:] if s["digest"] in keep]]

    _mismatch_at(c, n, mutate, "prior_kings")


@pytest.mark.parametrize("name,n", [("house_retakes", 5), ("house_loses", 3),
                                    ("copy_at_swap", 3)])
def test_fake_duplicate_digest_on_a_non_copy_is_a_mismatch(chains, name, n):
    """X (KING_D or a prior king, no verdict here) is not a copy of the house."""
    c = chains[name]
    i = entry_index(c.reports[n], X["digest"])

    def mutate(rep):
        rep["entries"][i].update(status="closed", reason="duplicate_digest")
        rep["skipped"] = [*rep["skipped"], {"digest": X["digest"],
                                            "reason": "duplicate_digest"}]

    _mismatch_at(c, n, mutate, f"entries[{i}]")
