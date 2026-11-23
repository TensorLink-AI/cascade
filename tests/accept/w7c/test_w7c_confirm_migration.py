"""W7c (plan rev6.11 (b), W7c row): recovery of a testnet `confirm_pending`
state after the confirmation duel is dropped.

Lane D was never armed on mainnet, but a testnet `lane_d_state.json` written
by the old code may hold an entry in `confirm_pending` (a first clear
waiting for its confirmation duel). On load at the first settlement after
the release:

- if the entry's `confirm.king_digest` still equals KING_D, it becomes a
  **crown candidate on its first-clear set**: A10 and the §1.3b gate decide
  it, with no second duel;
- otherwise it is `pending` / `pending_floor`: the first clear is discarded
  (the old D20 (a) outcome) and the windows it was judged on stay used, so
  it is never re-judged on them.

Either way the migration is tagged `confirm_state_migrated` in the signed
report of that settlement (and only that one), and the receipt's prior /
posterior digests bind the migration: the prior is the legacy state as
loaded, the posterior the migrated state written. README.md "Migration of a
`confirm_pending` state (rev6.11)" has the contract.

The legacy state is built from a real boundary(1) run (X `pending_floor`)
with X's record rewritten into the exact shape the old code wrote at a first
clear (`lane_d.py` on main @ 315d240: `status` / `reason` `confirm_pending`,
`confirm = {first_set_sha256, king_digest, king_reveal_ts, rule, boundary}`,
`frozen_set = None`, `used_windows` = the first-clear set).

On main the old code runs the confirmation duel instead: snapshot 2 alone is
below the floors, so nothing is crowned and the entry stays
`confirm_pending` (assertion failures).
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    DAY,
    FLOOR_WINDOWS,
    HOUSE_T,
    Chain,
    W7cChain,
    boundary,
    canonical,
    digest,
    lane_d_cfg,
    receipt_of,
    sha,
    state_sha,
    string_paths,
    window_id,
    write_canonical,
)

TAG = "confirm_state_migrated"


def _first_clear_ids(chain: Chain) -> list[str]:
    """The legacy first-clear set: every window of snapshot 1."""
    return sorted(window_id(r["series_id"], r["target_ts"]) for r in chain.records[0])


def _legacy_state(chain: Chain, path, *, king_matches: bool) -> dict:
    """Settle boundary(1) (X `pending_floor`), then rewrite X's record into
    the old code's `confirm_pending` shape and write it canonically."""
    chain.settle(1, path)
    state = json.loads(path.read_bytes())
    x = chain.x["digest"]
    assert state["entries"][x]["status"] == "pending", "fixture: X waits on the floors at b1"
    ids = _first_clear_ids(chain)
    if king_matches:
        kd, kts = state["king"]["digest"], int(state["king"]["reveal_ts"])
    else:
        kd, kts = digest("w7c a king that no longer holds KING_D"), HOUSE_T - DAY
    state["entries"][x].update(
        status="confirm_pending", reason="confirm_pending", used_windows=ids, frozen_set=None,
        confirm={"first_set_sha256": sha("\n".join(ids).encode()), "king_digest": kd,
                 "king_reveal_ts": kts, "rule": "restart", "boundary": boundary(1)})
    write_canonical(path, state)
    return state


def _settle_b2(chain: Chain, path, legacy: dict):
    try:
        return chain.settle(2, path, expect_prior_sha256=state_sha(legacy))
    except (TypeError, ValueError) as e:
        raise AssertionError(f"the migrating settlement was refused: {e!r}") from e


def _tag_paths(report: dict) -> list[str]:
    body = {k: v for k, v in report.items() if k != "signature"}
    return string_paths(body, TAG)


def _binds(settlement, legacy: dict, path) -> None:
    from cascade.validator.lane_d_state import verify_receipt

    rec = receipt_of(settlement)
    assert isinstance(rec, dict), "the migrating settlement returns a receipt"
    assert rec["prior_state_sha256"] == state_sha(legacy), \
        "the receipt's prior is the legacy state as loaded"
    assert rec["posterior_state_sha256"] == sha(path.read_bytes())
    verify_receipt(rec, report=settlement.report, state=json.loads(path.read_bytes()))


def test_rev611_confirm_pending_against_the_current_king_is_crowned_on_its_first_clear(tmp_path):
    chain = W7cChain(tmp_path)
    path = chain.state_path("legacy")
    legacy = _legacy_state(chain, path, king_matches=True)
    x = chain.x["digest"]
    hosts = len(chain.hosts)

    s2 = _settle_b2(chain, path, legacy)
    assert s2.report["crowned"] == x, (
        "a confirm_pending entry whose first-clear king is still KING_D is a crown "
        "candidate on its first-clear set (A10 + gate, no second duel)")
    assert any(h.calls for h in chain.hosts[hosts:]), "the §1.3b gate decides the candidate"
    state = json.loads(path.read_bytes())
    assert state["king"]["digest"] == x and state["entries"][x]["status"] == "crowned"
    assert "confirm_pending" not in canonical(state).decode(), \
        "no confirm_pending survives the migration"
    assert _tag_paths(s2.report), f"the signed report tags the migration ({TAG})"
    _binds(s2, legacy, path)

    s3 = chain.settle(3, path)
    assert s3.report["king_before"]["digest"] == x and s3.report["crowned"] is None
    assert not _tag_paths(s3.report), "the migration runs once"


def test_rev611_confirm_pending_against_another_king_returns_to_pending_floor(tmp_path):
    chain = W7cChain(tmp_path)
    path = chain.state_path("legacy")
    legacy = _legacy_state(chain, path, king_matches=False)
    x = chain.x["digest"]

    s2 = _settle_b2(chain, path, legacy)
    assert s2.report["crowned"] is None, "a first clear against another king is discarded"
    state = json.loads(path.read_bytes())
    rec = state["entries"][x]
    assert (rec["status"], rec["reason"]) == ("pending", "pending_floor"), \
        (rec["status"], rec["reason"])
    assert set(_first_clear_ids(chain)) <= set(rec["used_windows"]), \
        "the windows of the discarded first clear stay used (never re-judged)"
    assert "confirm_pending" not in canonical(state).decode()
    assert _tag_paths(s2.report), f"the signed report tags the migration ({TAG})"
    _binds(s2, legacy, path)

    s3 = chain.settle(3, path)
    assert not _tag_paths(s3.report), "the migration runs once"


@pytest.mark.parametrize("king_matches", [True, False], ids=["same_king", "king_changed"])
def test_rev611_the_migration_is_deterministic(tmp_path, king_matches):
    """Two validators migrating the same legacy state give byte-identical
    states and equal receipts (consensus: external validators re-derive it)."""
    chain = W7cChain(tmp_path)
    out = []
    for run in ("a", "b"):
        path = chain.state_path(run)
        legacy = _legacy_state(chain, path, king_matches=king_matches)
        s2 = _settle_b2(chain, path, legacy)
        out.append((sha(path.read_bytes()), receipt_of(s2)))
    assert out[0] == out[1]
    assert _tag_paths(s2.report), f"the signed report tags the migration ({TAG})"


def test_rev611_a_fresh_state_never_carries_confirm_pending(tmp_path):
    """The new code never writes `confirm_pending`: a clean run's states and
    reports hold no such status at any boundary."""
    chain = W7cChain(tmp_path)
    path = chain.state_path("clean")
    for n in (1, 2, 3):
        s = chain.settle(n, path)
        assert "confirm_pending" not in path.read_text(), f"state at boundary({n})"
        assert not string_paths(s.report, "confirm_pending"), f"report at boundary({n})"
    assert s.report["king_before"]["digest"] == chain.x["digest"]


def test_rev611_confirm_on_king_change_is_a_parsed_legacy_key(tmp_path):
    """`[lane_d] confirm_on_king_change` stays parsed (the shipped chain.toml
    carries it) and has no effect: it leaves the report's config echo, so
    both values give byte-identical reports, states and receipts."""
    chain = W7cChain(tmp_path)
    out = {}
    for rule in ("restart", "original"):
        path = chain.state_path(f"rule-{rule}")
        cfg = lane_d_cfg(confirm_on_king_change=rule, min_windows=FLOOR_WINDOWS)
        out[rule] = [(sha(canonical({k: v for k, v in s.report.items() if k != "signature"})),
                      receipt_of(s), sha(path.read_bytes()))
                     for s in (chain.settle(n, path, cfg=cfg) for n in (1, 2, 3))]
    assert out["restart"] == out["original"]
