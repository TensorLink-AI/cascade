"""2026-11-26: a lane B make-good approval never matched its leg.

The request is filed under the generator ref (``gen.ref``) while the lane B
queue entry carries ``"<domain>|<generator ref>"``; ``approval_on_file`` checks
with the entry's ref, so ``_bound`` compared the two raw and every approved
lane B make-good stayed blocked ("paused after 3 operator-side failures")."""
from __future__ import annotations

import importlib

GEN = "thomas05101990/gen-574ca46dc640@sha256:72e839e5b15f22eb2949ff5c81655045f35ca3204c7c5152fc3cb031c96fa8f3"
HK = "5GMzsPiUhy6gtqU5GxeUgzJ4yyQnJ5iyx8EUdSgZD2riSDgn"
Q = "lane_b_queue.json"
FUNDED_AT = 1795652216.2198212


def _approvals():
    return importlib.import_module("cascade.funding.operator_approvals")


def _approved_store(tmp_path):
    m = _approvals()
    store = m.OperatorApprovals(tmp_path / "operator_approvals.json")
    assert store.request(Q, HK, ref=GEN, attempt_id="a1", kind="make_good",
                         funded_at=FUNDED_AT, sku="8xH200")
    store.decide(HK, approve=True, attempt_id="a1", max_cost_usd=115.0, queue_name=Q)
    return store


def test_a_lane_b_entry_ref_matches_its_approved_request(tmp_path):
    store = _approved_store(tmp_path)
    # the check the trainer runs for a lane B leg: the queue entry's ref
    verdict = store.check(Q, HK, ref=f"energy|{GEN}", attempt_id="a2", kind="",
                          funded_at=FUNDED_AT)
    assert verdict == "approved", verdict
    # the bare generator ref (lane A, the request side) still matches
    assert store.check(Q, HK, ref=GEN, attempt_id="a2", kind="",
                       funded_at=FUNDED_AT) == "approved"
    # ...but another generator or another funding never matches
    other = GEN.replace("574ca46dc640", "000000000000")
    assert store.check(Q, HK, ref=f"energy|{other}", attempt_id="a2", kind="",
                       funded_at=FUNDED_AT) == "none"
    assert store.check(Q, HK, ref=f"energy|{GEN}", attempt_id="a2", kind="",
                       funded_at=FUNDED_AT + 1) == "none"
