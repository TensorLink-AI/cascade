"""W9: the crown chain and the recorded inputs, end to end in chain mode
(plan rev6.11 (b): the crown at the first verdict, no confirmation duel; §1.2 "record the
coldkey early and carry it in the signed report", §1.5 "`checks.
derive_weights` and `check_lane_d` re-derive the lane D weights from the
receipt chain"; README "Chain mode").

W7a (RV7 B2, `test_w7a_audit_provenance.py`) pins the chain-mode keyword
`check_lane_d(..., prior_reports=[...])` and the first-clear / `held_from`
evidence. W9 adds what only the chain shows: each settlement starts where
the previous signed report ended (`king_before`, the house), and an entry's
recorded inputs never change between reports. Every doctored report here is
self-consistent on its own (the single-report audit PASSes it); only the
chain refutes it.

On main `check_lane_d` takes no `prior_reports` (an `assert`).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-chain"))


def _reports(run) -> dict:
    return {n: copy.deepcopy(run[n].report) for n in ("s1", "s2", "s3", "s3d")}


def test_honest_chain_passes_in_chain_mode(run):
    r = _reports(run)
    h.assert_pass(r["s1"], run["cfg"], "b1 (genesis)", prior=[])
    h.assert_pass(r["s2"], run["cfg"], "b2 after b1", prior=[r["s1"]])
    h.assert_pass(r["s3"], run["cfg"], "b3 after b1, b2", prior=[r["s1"], r["s2"]])
    h.assert_pass(r["s3d"], run["cfg"], "b3 (X deregistered) after b1, b2",
                  prior=[r["s2"], r["s1"]])


def test_the_crown_chain_crowns_at_the_first_verdict(run):
    """rev6.11 (b): X is crowned at b1, its first verdict (no confirmation
    duel); b2 and b3 judge nothing. On main X first-clears at b1
    (`confirm_pending`) and is crowned at b2."""
    h.check_crown_chain_shape(run)


def test_a_settlement_that_undoes_the_crown_fails(run):
    """b3 restated as if b1 never crowned X: KING_D the house before and
    after, no prior kings, the house slot unpaid (0.25 burns instead of
    paying X). Its verdict-free body is a no-crown report; b2's signed
    `king_after` (X) refutes it."""
    r = _reports(run)
    forged = h.resign(h.undo_crown(r["s3"], r["s1"]))
    h.assert_fail(forged, run["cfg"], "b3's king_before is not b2's king_after",
                  prior=[r["s1"], r["s2"]])


@pytest.mark.parametrize("field,value", [
    ("coldkey", h.w7a.coldkey("w9-other-coldkey")),
    ("ready_block", h.w7a.READY_BLOCK + 5),
    ("context", 8192),
])
def test_an_entrys_recorded_inputs_never_change_between_reports(run, field, value):
    """X's coldkey (as recorded at `ready`, D22 option b), `ready_block` and
    `context` are inputs: b2 restating any of them differs from b1's record
    of the same digest."""
    r = _reports(run)
    x = run["entry"]["digest"]
    assert h.entry_rec(r["s1"], x)[field] != value
    forged = copy.deepcopy(r["s2"])
    h.entry_rec(forged, x)[field] = value
    forged = h.resign(forged)
    h.assert_pass(forged, run["cfg"], f"fixture: b2 with X's {field} restated, alone")
    h.assert_fail(forged, run["cfg"], f"X's {field} differs from b1's record",
                  prior=[r["s1"]])


def test_the_house_and_its_t_carry_over_unless_refreshed(run):
    """b2 restates the house H0's T (and every copy of it: `house`, and
    `king_before` and any cohort's king T where they name the house) a day
    earlier. The house digest did not change, so its T is b1's."""
    r = _reports(run)
    forged = copy.deepcopy(r["s2"])
    t = h.HOUSE_T - 86_400
    forged["house"]["reveal_ts"] = t
    if forged["king_before"]["digest"] == forged["house"]["digest"]:
        forged["king_before"]["reveal_ts"] = t
    for c in forged["cohorts"]:
        if c["king_digest"] == forged["house"]["digest"]:
            c["king_reveal_ts"] = t
    forged = h.resign(forged)
    h.assert_pass(forged, run["cfg"], "fixture: b2 with the house's T restated, alone")
    h.assert_fail(forged, run["cfg"], "the house's T differs from b1's", prior=[r["s1"]])
