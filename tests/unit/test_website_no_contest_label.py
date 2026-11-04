"""A king-only settlement is a designed no-contest, not a rejection (owner,
2026-11-04: "rename rounds where there is no competitor to not say rejected").
The validator's receipt keeps status="rejected" / reason="no_king_challenger_pair"
(audit, bootstrap and archived signatures branch on it); only the miner
dashboard's LABEL changes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "cascade/website/index.html"


def test_dashboard_labels_a_king_only_round_as_no_contest():
    html = PAGE.read_text()
    # the reason table and the helpers exist
    assert 'var NO_CONTEST_REASONS={no_king_challenger_pair:"No challenger this round — the king holds"};' in html
    assert "function isNoContest(r)" in html and "function reasonLabel(r)" in html
    # the verdict badge: no-contest first, then real rejections, then the scored verdicts
    assert re.search(r'function verdictClass\(r\)\{ if\(isNoContest\(r\)\)return"nocontest"; if\(r\.status==="rejected"\)return"rejected";', html)
    assert re.search(r'function verdictLabel\(r\)\{ if\(isNoContest\(r\)\)return"No contest"; if\(r\.status==="rejected"\)return"Rejected";', html)
    # the muted style covers the new badge class
    assert ".badge.inconclusive, .badge.rejected, .badge.nocontest {" in html
    # the Reason cell and the "newer settlement" note never print the raw machine reason
    assert 'esc(reasonLabel(r))' in html
    assert 'esc(r.reject_reason||"--")' not in html
    assert 'String(later.reject_reason)' not in html
    assert 'isNoContest(later)?"no contest ("+reasonLabel(later).toLowerCase()+")"' in html


def test_label_logic_matches_the_receipt_statuses():
    """Pure re-implementation of the page's helpers over the receipt shapes
    the validator publishes."""
    reasons = {"no_king_challenger_pair": "No challenger this round — the king holds"}

    def is_no_contest(r):
        return r.get("status") == "rejected" and str(r.get("reject_reason") or "") in reasons

    def label(r):
        if is_no_contest(r):
            return "No contest"
        if r.get("status") == "rejected":
            return "Rejected"
        if r.get("inconclusive"):
            return "Inconclusive"
        return ("Dethroned" if r.get("dethroned") else "Challenger wins") if r.get("challenger_wins_round") else "King holds"

    def reason(r):
        k = str(r.get("reject_reason") or "")
        return reasons.get(k, k or "--")

    king_only = {"status": "rejected", "reject_reason": "no_king_challenger_pair"}
    resync = {"status": "rejected", "reject_reason": "king_resyncing"}
    assert label(king_only) == "No contest" and reason(king_only) == "No challenger this round — the king holds"
    assert label(resync) == "Rejected" and reason(resync) == "king_resyncing"      # real rejections unchanged
    assert label({"status": "scored", "challenger_wins_round": False}) == "King holds"
    assert label({"status": "scored", "challenger_wins_round": True, "dethroned": True}) == "Dethroned"
    assert label({"status": "scored", "inconclusive": True}) == "Inconclusive"
    assert reason({"status": "rejected"}) == "--"
