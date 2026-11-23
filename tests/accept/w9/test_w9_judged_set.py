"""W9: `check_lane_d` re-derives the judged set and every selection skip
(plan rev6 §1.2 "(rev3) Enforcement": "The judged set and every skip reason
are structured report fields. `check_lane_d` re-derives the judged set";
the caps `max_judged_per_coldkey_per_settlement` and
`max_entries_per_settlement`; README "Judged set").

Each case starts from an honest signed report (it must PASS), changes one
signed input or output, re-signs it with the validator's key and expects a
FAIL. On main `check_lane_d` never reads `judged`, `skipped` or an entry's
`coldkey`, so every doctored report PASSes (an `assert`).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def caps(tmp_path_factory):
    return h.caps(tmp_path_factory.mktemp("w9-caps"))


def _case(caps, name):
    run = caps[name]
    return copy.deepcopy(run["report"]), run["cfg"], caps["digests"], caps["coldkeys"]


def test_honest_cap_reports_pass(caps):
    for name in ("cold", "glob"):
        report, cfg, _, _ = _case(caps, name)
        h.assert_pass(report, cfg, f"honest {name} cap report")


# ── the per-coldkey cap ─────────────────────────────────────────────────────


def test_capped_entry_whose_coldkey_had_room_fails(caps):
    """B's coldkey moved to an unused one: FIFO now selects B (and caps no
    one), so its `coldkey_cap_settlement` skip is a lie."""
    report, cfg, d, k = _case(caps, "cold")
    h.entry_rec(report, d["b"])["coldkey"] = k["k3"]
    h.assert_fail(h.resign(report), cfg, "a capped entry whose coldkey had room")


def test_judged_entry_over_its_coldkey_cap_fails(caps):
    """C moved onto A's coldkey: with a cap of 1, the later of the two (C)
    must have been skipped, yet it is judged."""
    report, cfg, d, k = _case(caps, "cold")
    h.entry_rec(report, d["c"])["coldkey"] = k["k1"]
    h.assert_fail(h.resign(report), cfg, "a judged entry over its coldkey cap")


def test_judged_entry_without_a_coldkey_fails(caps):
    """A null coldkey is `coldkey_unresolved` at registration: never judged."""
    report, cfg, d, _ = _case(caps, "cold")
    h.entry_rec(report, d["a"])["coldkey"] = None
    h.assert_fail(h.resign(report), cfg, "a judged entry with coldkey null")


def test_capped_entry_with_another_reason_fails(caps):
    """B is over the cap, so its typed reason is `coldkey_cap_settlement`,
    not `pending_floor` (invariant 11: the exact reason)."""
    report, cfg, d, _ = _case(caps, "cold")
    h.set_reason(report, d["b"], "pending_floor")
    h.assert_fail(h.resign(report), cfg, "a capped entry relabelled pending_floor")


def test_open_entry_missing_from_judged_and_skipped_fails(caps):
    """B is still open (`pending`) but the report drops its skip: an open
    entry the settlement considered is neither judged nor skipped."""
    report, cfg, d, _ = _case(caps, "cold")
    report["skipped"] = [s for s in report["skipped"] if s["digest"] != d["b"]]
    assert h.entry_rec(report, d["b"])["status"] == "pending"
    h.assert_fail(h.resign(report), cfg, "an open entry silently omitted")


def test_judged_lists_a_digest_without_a_verdict_fails(caps):
    report, cfg, d, _ = _case(caps, "cold")
    report["judged"] = [*report["judged"], d["b"]]
    h.assert_fail(h.resign(report), cfg, "judged names a skipped entry with no verdict")


def test_judged_must_equal_the_verdicts(caps):
    report, cfg, d, _ = _case(caps, "cold")
    report["judged"] = [x for x in report["judged"] if x != d["c"]]
    h.assert_fail(h.resign(report), cfg, "judged omits a judged member")


# ── the global cap and FIFO by T ────────────────────────────────────────────


def test_global_cap_skips_the_latest_reveal(caps):
    """C's reveal moved before B's: FIFO by T now selects A and C and caps B,
    so the signed judged set (A, B) and C's skip are wrong."""
    report, cfg, d, _ = _case(caps, "glob")
    h.entry_rec(report, d["c"])["reveal_ts"] = h.T0 + 30
    h.assert_fail(h.resign(report), cfg, "the global cap skipped an earlier reveal")


def test_global_cap_skip_reason_is_exact(caps):
    report, cfg, d, _ = _case(caps, "glob")
    h.set_reason(report, d["c"], "pending_floor")
    h.assert_fail(h.resign(report), cfg, "a globally capped entry relabelled pending_floor")


# ── registration rules re-derivable from the signed entry record ────────────
# (the block clock: `ready_block` / `reveal_block` against `ready_ttl_blocks`,
# B10; the armed contexts, §1.1). Only the rules the producer's registration
# applies (`_register`) are pinned, so the audit never refutes an honest
# settlement.


@pytest.mark.parametrize("field,value,what", [
    ("ready_block", h.w7a.REVEAL_BLOCK - 7200 - 1,
     "revealed more than ready_ttl_blocks after ready (ready_expired)"),
    ("context", 2048, "with a context the config does not arm (context_not_armed)"),
])
def test_judged_entry_breaking_a_block_clock_rule_fails(caps, field, value, what):
    report, cfg, d, _ = _case(caps, "cold")
    assert int(cfg.ready_ttl_blocks) == 7200, "fixture: the default ready_ttl_blocks"
    h.entry_rec(report, d["a"])[field] = value
    h.assert_fail(h.resign(report), cfg, f"a judged entry {what}")
