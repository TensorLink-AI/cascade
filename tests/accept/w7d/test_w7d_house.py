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

import pytest

from .helpers import describe, replay, replay_module
from .house_helpers import (
    COPY,
    HOUSE_CHAINS,
    UIDS,
    X,
    Y,
    Z,
    doctored,
    entry_index,
    has_path,
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


def test_dec0076_no_house_challenger_after_a_miner_crown(chains):
    """DEC-CA-0076 (owner 2026-11-24, laned:tests-correction): the W5f house
    challenger and T10 retake are retired. In every house chain, once a
    miner has held KING_D no settlement judges the house and KING_D is never
    the house again; a miner copy of the house is still closed
    `duplicate_digest`. (Replaces the house-loses / house-retakes fixture
    facts and their replay derivations.)"""
    for name, c in sorted(chains.items()):
        miner_seen = False
        for n in sorted(c.reports):
            r = c.reports[n]
            if miner_seen:
                assert verdict_index(r, r["house"]["digest"]) is None, \
                    f"{name} b{n}: the house was judged after a miner crown"
                assert not r["king_after"]["house"], f"{name} b{n}: the house retook KING_D"
            miner_seen = miner_seen or not r["king_after"]["house"]
    r = chains["copy_at_swap"].reports[3]
    e = r["entries"][entry_index(r, COPY["digest"])]
    assert (e["status"], e["reason"]) == ("closed", "duplicate_digest")








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





@pytest.mark.parametrize("name,n", RETAKES)
def test_retake_with_crowned_set_is_a_mismatch(chains, name, n):
    def mutate(rep):
        rep["crowned"] = rep["house"]["digest"]

    _mismatch_at(chains[name], n, mutate, "crowned")





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
