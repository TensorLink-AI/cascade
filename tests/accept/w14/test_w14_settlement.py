"""W14: `absent_settlement` (`cascade/validator/lane_d.py`), the trigger.

Plan W14 row: "Today a missing lane D report makes satellites re-assert the
old vector for one era (`stale_after_blocks = 3600`) and then burn the whole
vector". §5 kill switch: "Stop publishing ⇒ only lane D burns" is what W14
makes true. Every chain is W7c's real `settle_lane_d` fixture. README.md
"Trigger" and "Why an absent record can't suppress a real crown".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from .helpers import (
    ABSENT_KEYS,
    REPO,
    REQUIRED_REASONS,
    Chain,
    absent,
    boundary,
    dir_snapshot,
    genesis_sha,
    lane_d_cfg,
    receipt_error_code,
    settle_b4,
    unarmed_cfg,
    w7c,
)


def _no_report_at(state_path, n: int) -> bool:
    from cascade.validator.lane_d_state import report_path

    return not report_path(state_path, boundary(n)).exists()


# ── invariant 1 ─────────────────────────────────────────────────────────────


def test_inv1_absent_settlement_unchanged_when_unarmed(tmp_path):
    """Unarmed, an absent settlement is nothing: no report, no pay, no
    receipt, and no file is written."""
    chain = Chain(tmp_path)
    sp = chain.state_path("u")
    before = dir_snapshot(sp.parent)
    s = absent(chain, 1, sp, cfg=unarmed_cfg())
    assert (s.report, s.pay, s.receipt) == (None, None, None)
    assert dir_snapshot(sp.parent) == before


def test_inv1_absent_settlement_armed_nonempty_lane_d_effect(tmp_path):
    from cascade.validator.lane_d_state import ABSENT_STATUS

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s = absent(chain, 1, sp)
    assert s.report is None
    assert s.receipt is not None and s.receipt["status"] == ABSENT_STATUS
    assert s.pay is not None
    assert float(s.pay["share"]) == float(lane_d_cfg().share) == 0.25
    assert list(s.pay["court_uids"]) == []


# ── the record a settlement produces ────────────────────────────────────────


def test_absent_from_genesis_binds_the_genesis_digest_and_writes_nothing(tmp_path):
    chain = Chain(tmp_path)
    sp = chain.state_path("g")
    before = dir_snapshot(sp.parent)
    s = absent(chain, 1, sp)
    rec = s.receipt
    assert sorted(rec) == sorted(ABSENT_KEYS)
    assert rec["boundary"] == boundary(1)
    assert rec["reason"] == "report_absent"
    assert rec["prior_state_sha256"] == rec["posterior_state_sha256"] == genesis_sha(chain)
    assert dir_snapshot(sp.parent) == before
    assert not sp.exists()


@pytest.mark.parametrize("reason", REQUIRED_REASONS)
def test_absent_after_a_settlement_binds_the_loaded_state(tmp_path, reason):
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1 = chain.settle(1, sp)
    before = dir_snapshot(sp.parent)
    s2 = absent(chain, 2, sp, reason=reason)
    post1 = s1.receipt["posterior_state_sha256"]
    assert s2.receipt["prior_state_sha256"] == s2.receipt["posterior_state_sha256"] == post1
    assert s2.receipt["reason"] == reason
    assert w7c.sha(sp.read_bytes()) == post1
    assert dir_snapshot(sp.parent) == before, "an absent settlement writes no file"
    assert _no_report_at(sp, 2)
    verify_receipt(s2.receipt, report=None, state=json.loads(sp.read_bytes()))
    verify_receipt_chain([s1.receipt, s2.receipt])


def test_an_untyped_reason_is_refused_and_writes_nothing(tmp_path):
    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    chain.settle(1, sp)
    before = dir_snapshot(sp.parent)
    with pytest.raises(ValueError):
        absent(chain, 2, sp, reason="operator_said_so")
    assert dir_snapshot(sp.parent) == before


def test_a_recall_is_idempotent(tmp_path):
    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    chain.settle(1, sp)
    a = absent(chain, 2, sp)
    before = dir_snapshot(sp.parent)
    b = absent(chain, 2, sp)
    assert (a.receipt, a.pay, a.report) == (b.receipt, b.pay, b.report)
    assert dir_snapshot(sp.parent) == before


# ── it never replaces a judgement ───────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2])
def test_absent_never_relabels_a_settled_boundary(tmp_path, n):
    """Once a boundary is settled with a signed report, it can't be marked
    absent (that would hide the crown at boundary(2))."""
    from cascade.validator.lane_d_state import LaneDStateError

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    chain.run(sp, 2)
    before = dir_snapshot(sp.parent)
    with pytest.raises(LaneDStateError) as e:
        absent(chain, n, sp)
    assert receipt_error_code(e) in {"stale_boundary", "report_conflict"}
    assert dir_snapshot(sp.parent) == before


def test_absent_never_replaces_a_persisted_report(tmp_path, monkeypatch):
    """A crash after boundary(2)'s crowning report landed and before the
    state did: the report must be adopted (W7c recovery), never replaced by
    an absent record."""
    from cascade.validator.lane_d_state import LaneDStateError, report_path

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    chain.settle(1, sp)
    with monkeypatch.context() as m:
        w7c.crash_on_rename(m, sp)
        with pytest.raises(OSError):
            chain.settle(2, sp)
    assert report_path(sp, boundary(2)).is_file()
    before = dir_snapshot(sp.parent)
    with pytest.raises(LaneDStateError) as e:
        absent(chain, 2, sp)
    assert receipt_error_code(e) == "report_conflict"
    assert dir_snapshot(sp.parent) == before


def test_absent_fails_closed_on_lost_state(tmp_path):
    """A lost state after a persisted settlement is never genesis, so an
    absent record can't restart lane D from the house and drop KING_D."""
    from cascade.validator.lane_d_state import LaneDStateError

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    chain.run(sp, 2)
    sp.unlink()
    before = dir_snapshot(sp.parent)
    with pytest.raises(LaneDStateError) as e:
        absent(chain, 3, sp)
    assert receipt_error_code(e) == "state_missing"
    assert dir_snapshot(sp.parent) == before


def test_absent_checks_the_callers_expected_prior(tmp_path):
    from cascade.validator.lane_d_state import LaneDStateError

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1, s2 = chain.run(sp, 2)
    before = dir_snapshot(sp.parent)
    with pytest.raises(LaneDStateError) as e:
        absent(chain, 3, sp, expect_prior_sha256=s1.receipt["posterior_state_sha256"])
    assert receipt_error_code(e) == "state_digest_mismatch"
    assert dir_snapshot(sp.parent) == before
    ok = absent(chain, 3, sp, expect_prior_sha256=s2.receipt["posterior_state_sha256"])
    assert ok.receipt["prior_state_sha256"] == s2.receipt["posterior_state_sha256"]


# ── it defers, never erases ─────────────────────────────────────────────────


def test_absent_defers_a_pending_crown_never_erases_it(tmp_path):
    """X waits on the floors at boundary(1) (pending); boundary(2) is absent;
    X is still pending and boundary(3) is its first verdict, which crowns it
    (rev6.11 (b): no confirmation duel)."""
    from cascade.validator.lane_d_state import verify_receipt_chain

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1 = chain.settle(1, sp)
    a2 = absent(chain, 2, sp)
    s3 = chain.settle(3, sp)
    assert s3.report["crowned"] == chain.x["digest"]
    assert list(s3.pay["court_uids"]) == [chain.uid_of[chain.x["hotkey"]]]
    assert s3.receipt["prior_state_sha256"] == a2.receipt["posterior_state_sha256"]
    verify_receipt_chain([s1.receipt, a2.receipt, s3.receipt])


def test_absent_after_a_crown_keeps_the_king(tmp_path):
    """X crowned at boundary(2); boundary(3) is absent (lane D's share
    burns); the state still holds X as KING_D and boundary(4) pays X."""
    from cascade.validator.lane_d_state import verify_receipt_chain

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1, s2 = chain.run(sp, 2)
    a3 = absent(chain, 3, sp)
    assert list(a3.pay["court_uids"]) == []
    state = json.loads(sp.read_bytes())
    assert state["king"]["digest"] == chain.x["digest"]
    s4 = settle_b4(chain, sp)
    assert s4.report["king_before"]["digest"] == chain.x["digest"]
    assert list(s4.pay["court_uids"]) == [chain.uid_of[chain.x["hotkey"]]]
    verify_receipt_chain([s1.receipt, s2.receipt, a3.receipt, s4.receipt])


def test_a_forged_absent_cannot_drop_a_crown(tmp_path):
    """An absent record binding a state from before the crown breaks the
    chain both ways: against the crowning receipt before it and against the
    next real settlement after it."""
    from cascade.validator.lane_d_state import make_absent_receipt, verify_receipt_chain

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1, s2 = chain.run(sp, 2)
    forged = make_absent_receipt(boundary=boundary(3), reason="report_absent",
                                 state_sha256=s1.receipt["posterior_state_sha256"])
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([s1.receipt, s2.receipt, forged])
    assert receipt_error_code(e) == "chain_break"
    s4 = settle_b4(chain, sp)
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([s1.receipt, forged, s4.receipt])
    assert receipt_error_code(e) == "chain_break"


@pytest.mark.parametrize("order", ["repeated", "reversed"])
def test_absent_records_obey_boundary_order(tmp_path, order):
    from cascade.validator.lane_d_state import verify_receipt_chain

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1 = chain.settle(1, sp)
    a2 = absent(chain, 2, sp)
    seq = [s1.receipt, a2.receipt, a2.receipt] if order == "repeated" else [a2.receipt,
                                                                            s1.receipt]
    with pytest.raises(ValueError) as e:
        verify_receipt_chain(seq)
    assert receipt_error_code(e) == "boundary_order"


# ── determinism ─────────────────────────────────────────────────────────────


def test_absent_records_are_deterministic_across_runs(tmp_path):
    pools = tmp_path / "pools"
    recs = []
    for run in ("r1", "r2"):
        chain = Chain(tmp_path / run, pools_dir=pools)
        sp = chain.state_path(run)
        s1 = chain.settle(1, sp)
        recs.append((s1.receipt, absent(chain, 2, sp).receipt))
    assert recs[0] == recs[1]


_CHILD = """
import json, sys
from pathlib import Path
from tests.accept.w14.helpers import Chain, absent
chain = Chain(Path(sys.argv[1]))
s = absent(chain, 1, chain.state_path("child"), reason=sys.argv[2])
print(json.dumps(s.receipt, sort_keys=True))
"""


@pytest.mark.parametrize("seed", ["0", "4242"])
def test_absent_records_are_independent_of_the_hash_seed(tmp_path, seed):
    chain = Chain(tmp_path / "here")
    here = absent(chain, 1, chain.state_path("here"), reason="infra_deferred").receipt
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-P", "-c", _CHILD, str(tmp_path / "child"),
                          "infra_deferred"], cwd=str(REPO), env=env, capture_output=True,
                         text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == here
