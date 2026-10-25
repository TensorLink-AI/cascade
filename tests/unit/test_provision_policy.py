"""Rental policy — pod counts, teardown, and the epoch/TTL arithmetic.

All pure arithmetic: no clock, no chain, no provider API (the loop injects the
observed world)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.provision.policy import ProvisionPolicy, pods_for_slots, teardown_due

# ── pods_for_slots ───────────────────────────────────────────────────────────


def test_pods_for_slots_ceils_and_clamps():
    assert pods_for_slots(1, 1, 1) == 1
    assert pods_for_slots(3, 2, 4) == 2
    assert pods_for_slots(9, 2, 4) == 4               # clamped at max_pods
    assert pods_for_slots(0, 2, 4) == 0               # nothing to serve
    assert pods_for_slots(3, 2, 0) == 0               # max_pods 0 = stage unmanaged


# ── teardown_due: the eval arm + the TTL backstop ────────────────────────────


def test_eval_pod_dies_on_receipt_or_newer_manifest():
    assert teardown_due("eval", receipt_seen=True, rented_at=0.0, now=100.0, ttl_hours=24.0)
    assert teardown_due("eval", newer_manifest=True, rented_at=0.0, now=100.0, ttl_hours=24.0)
    assert not teardown_due("eval", rented_at=0.0, now=100.0, ttl_hours=24.0)


def test_ttl_backstop_fires_for_every_stage_regardless_of_signals():
    for stage in ("heat", "final", "eval"):
        assert teardown_due(stage, rented_at=0.0, now=24 * 3600.0, ttl_hours=24.0)
        assert not teardown_due(stage, rented_at=0.0, now=24 * 3600.0 - 1, ttl_hours=24.0)


def test_eval_signals_never_kill_leftover_fleet_pods():
    # heat/final pods are only ledger leftovers of the retired per-round
    # fleet: the TTL alone reaps them.
    for stage in ("heat", "final"):
        assert not teardown_due(stage, receipt_seen=True, newer_manifest=True,
                                rented_at=0.0, now=100.0, ttl_hours=24.0)


def test_teardown_rejects_unknown_stage():
    with pytest.raises(ValueError):
        teardown_due("any", rented_at=0.0, now=0.0, ttl_hours=24.0)


# ── cadence change: the epoch and the TTL track the activation block ─────────


def _loop_with_schedule(**over):
    """A ProvisionerLoop with only the fields the cadence helpers touch."""
    from cascade.provision.loop import ProvisionerLoop

    base = dict(
        policy=ProvisionPolicy(max_spend_per_round=500.0, ttl_epochs=1),
        providers={}, chain_client=None, render=None,
        hosts_path=Path("/tmp/h"), work_root=Path("/tmp"), state_path=Path("/tmp/s.json"),
        epoch_blocks=3600, epoch_blocks_prev=7200, epoch_activation_block=8726400,
    )
    base.update(over)
    return ProvisionerLoop(**base)


def test_ttl_takes_the_longer_epoch_while_a_change_is_pending():
    """A too-short TTL reaps live pods mid-epoch; a too-long one only delays a
    backstop that receipt-latched teardown reaches first. Must err long."""
    pending = _loop_with_schedule()
    assert pending.ttl_hours == 24.0
    settled = _loop_with_schedule(epoch_blocks_prev=0, epoch_activation_block=0)
    assert settled.ttl_hours == 12.0
