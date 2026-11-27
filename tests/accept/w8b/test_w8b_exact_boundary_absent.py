"""2026-11-27 lane D settlement robustness (W8b wiring).

1. ``court_for`` / ``genesis_for`` answer from the settlement at EXACTLY the
   vector's boundary. The audit (``cascade.audit.checks.lane_d_pay``) pays
   only a record at the settlement's own boundary and a restarted node, a
   satellite and CI hold no older court; a running node that carried the
   last court (or the house genesis) forward paid what every other node
   burned. With no record at ``b`` a running node and a fresh node now push
   the same vector.
2. ``LaneDStep.abandon(b)``: the worker moved past ``b`` unsettled ⇒ the W14
   absent record (``infra_deferred``) is published at ``b`` and becomes the
   latest pointer, the next settlement extends it, and the audit's fetch +
   absent check verify it (CI receipt replay no longer skips the round).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from . import helpers as h

B4 = h.ARM_BLOCK + 4 * h.SETTLEMENT


def _nonzero(vec) -> dict[int, float]:
    return {i: w for i, w in enumerate(vec) if w}


@pytest.mark.parametrize("settled, missing", [((h.B1,), h.B2), ((h.B1, h.B2), h.B3)],
                         ids=["house-genesis-carried", "court-carried"])
def test_no_record_at_b_is_the_same_vector_on_a_running_and_a_fresh_node(tmp_path, settled,
                                                                         missing):
    cfg = h.chain_cfg(tmp_path / "v", lanes_bc=False)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    for b in settled:
        rig.step(b)
    running = rig.step
    rig.inputs.fail = True
    with pytest.raises(Exception):  # noqa: B017 — inputs_unavailable (test_w8b_fail_closed)
        running(missing)
    fresh = rig.restart()
    with pytest.raises(Exception):  # noqa: B017 — the fresh node cannot settle it either
        fresh(missing)
    assert running.court_for(missing) is None, \
        "a running node carried an older court forward to a boundary with no settlement"
    assert running.genesis_for(missing) is False, \
        "a running node carried the house genesis forward to a boundary with no settlement"
    assert fresh.court_for(missing) is None and fresh.genesis_for(missing) is False
    at = missing + 5
    vec_running, r = h.vector_at(cfg, at, step=running)
    vec_fresh, _ = h.vector_at(cfg, at, step=fresh)
    assert h.close(vec_running, vec_fresh), \
        ("running and fresh nodes diverge at a boundary with no lane D record",
         _nonzero(vec_running), _nonzero(vec_fresh))
    assert h.close(vec_running, h.want_vector(r, cfg, missing, [])), \
        "no lane D record at b: lane D's whole share burns (the audit's expectation)"


# ── abandon: the absent infra_deferred record ───────────────────────────────


class HouseInputs(h.Inputs):
    """The inputs source with its cheap ``house()`` (the signed pointer)."""

    def house(self):
        return dict(self.fx["chain"]["house"])


def _abandon(step):
    fn = getattr(step, "abandon", None)
    assert callable(fn), "LaneDStep.abandon(boundary) is missing: an orphan leaves no receipt"
    return fn


def _rig(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    fx = h.fixture(tmp_path)
    return h.Rig(tmp_path / "rig", cfg, fx, inputs=HouseInputs(fx))


def test_abandon_publishes_the_absent_record_and_the_next_settlement_extends_it(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt_chain

    w = h.wiring()
    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    b2 = rig.published_receipt(h.B2)
    calls_before = len(rig.inputs.calls)
    rec = _abandon(rig.step)(h.B3)
    assert rec is not None, "no absent record was published for the abandoned boundary"
    assert len(rig.inputs.calls) == calls_before, "abandon re-read the settlement inputs"
    pub = rig.published_receipt(h.B3)
    assert pub == rec
    assert pub["status"] == "absent" and pub["reason"] == "infra_deferred"
    assert pub["boundary"] == h.B3
    assert pub["prior_state_sha256"] == pub["posterior_state_sha256"] \
        == b2["posterior_state_sha256"], "an absent record moves no lane D state"
    assert json.loads(rig.store.objects[w.RECEIPT_LATEST_KEY]) == pub
    assert w.report_key(h.B3) not in rig.store.objects, "an absent settlement has no report"
    assert rig.step.court_for(h.B3) is None, "an absent settlement pays no court"
    rig.step(B4)
    b4 = rig.published_receipt(B4)
    verify_receipt_chain([rig.published_receipt(h.B1), b2, pub, b4])


def test_the_audit_verifies_the_published_absent_record(tmp_path):
    from cascade.audit.lane_d import check_lane_d_absent
    from cascade.audit.main import fetch_lane_d_record

    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    assert _abandon(rig.step)(h.B3) is not None
    record = fetch_lane_d_record(SimpleNamespace(epoch_start_block=h.B3, lane_d=None),
                                 rig.store)
    assert record is not None and record["report"] is None, \
        "the replay found no lane D record at the abandoned boundary"
    res = check_lane_d_absent(record["receipt"], rig.cfg.lane_d, boundary=h.B3,
                              previous_posterior_sha256=rig.published_receipt(h.B2)[
                                  "posterior_state_sha256"])
    assert res.status == "PASS", res.detail


def test_abandon_never_overwrites_a_settled_or_later_boundary(tmp_path):
    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    abandon = _abandon(rig.step)
    before = dict(rig.store.objects)
    assert abandon(h.B2) is None, "a settled boundary was recorded absent"
    assert abandon(h.B1) is None, "an older boundary was recorded absent"
    assert rig.store.objects == before


def test_only_the_producing_node_records_an_orphan(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path), with_inputs=False)
    before = dict(rig.store.objects)
    assert _abandon(rig.step)(h.B1) is None
    assert rig.store.objects == before, "a node that produces no lane D published a record"


def test_abandon_never_raises(tmp_path):
    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.store.fail_put = lambda _k: True
    assert _abandon(rig.step)(h.B2) is None
