"""Fleet policy — trigger timing, slot-based sizing, budget breaker, teardown.

All pure arithmetic: no clock, no chain, no provider API (the loop injects the
observed world). Sizing is exercised at both the testnet (900-block) and
mainnet (7200-block) epoch shapes."""

from __future__ import annotations

import pytest

from cascade.provision.policy import (
    FleetPlan,
    ProvisionPolicy,
    StageFleet,
    StagePolicy,
    should_trigger,
    size_fleet,
    teardown_due,
    within_budget,
)


def _policy(*, heat_gpus=8, heat_max=4, final_gpus=2, final_max=4,
            slot_overhead=1.3, margin=25, max_spend=25.0, ttl_epochs=1):
    return ProvisionPolicy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=heat_gpus,
                         max_pods=heat_max, providers=("lium", "shadeform"),
                         max_price_hr=4.0, slot_overhead=slot_overhead),
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=final_gpus,
                          max_pods=final_max, providers=("lium", "shadeform"),
                          max_price_hr=3.0),
        trigger_margin_blocks=margin,
        max_spend_per_round=max_spend,
        ttl_epochs=ttl_epochs,
    )


# ── should_trigger ───────────────────────────────────────────────────────────


def test_trigger_inside_margin_testnet_900():
    # 900-block testnet epoch: boundary at 900; 20 blocks out is inside a 25 margin.
    assert should_trigger(880, 900, 25, None)


def test_no_trigger_outside_margin_testnet_900():
    assert not should_trigger(870, 900, 25, None)          # 30 blocks out


def test_trigger_exactly_at_margin_boundary():
    assert should_trigger(875, 900, 25, None)               # exactly 25 out


def test_no_retrigger_for_already_provisioned_round():
    # The upcoming round is keyed by its boundary block (here 900): a 30s poll
    # loop stays inside the margin for many iterations and must rent ONCE.
    assert not should_trigger(880, 900, 25, already_provisioned_round=900)
    # …but a stale key from the PREVIOUS round does not block this one.
    assert should_trigger(880, 900, 25, already_provisioned_round=0)


def test_on_boundary_targets_next_epoch():
    # At block 900 the upcoming boundary is 1800 — a full epoch away, no trigger.
    assert not should_trigger(900, 900, 25, None)


def test_trigger_mainnet_7200():
    assert should_trigger(7180, 7200, 25, None)
    assert not should_trigger(7100, 7200, 25, None)
    assert not should_trigger(7180, 7200, 25, already_provisioned_round=7200)
    # Next epoch, same margin: 14400 is a fresh round key.
    assert should_trigger(14380, 7200, 25, already_provisioned_round=7200)


def test_trigger_rejects_bad_epoch_blocks():
    with pytest.raises(ValueError):
        should_trigger(10, 0, 25, None)


# ── size_fleet: heat (slot-based, multi-GPU pods) ────────────────────────────


def test_heat_sized_by_slots_not_pods():
    # 200 challengers × 0.5h × 1.3 overhead / 21h window = 6.19 → 7 slots →
    # one 8-GPU pod covers it.
    plan = size_fleet(200, 2, 0.5, 24.0, 3.0, _policy())
    assert plan.heat == StageFleet(pods=1, gpus_per_pod=8, slots=7)


def test_heat_pods_scale_with_field_and_clamp_at_max():
    # 2000 challengers → 61.9 → 62 slots → ⌈62/8⌉ = 8 pods, clamped to max 4.
    plan = size_fleet(2000, 2, 0.5, 24.0, 3.0, _policy(heat_max=4))
    assert plan.heat.pods == 4
    # The demand is still recorded: a clamped fleet runs more serial waves.
    assert plan.heat.slots == 62
    unclamped = size_fleet(2000, 2, 0.5, 24.0, 3.0, _policy(heat_max=16))
    assert unclamped.heat.pods == 8


def test_no_heat_pods_when_field_fits_in_final():
    # Everyone advances (n_eligible <= finalists): nothing to screen, rent nothing.
    for n in (0, 1, 2):
        plan = size_fleet(n, 2, 0.5, 24.0, 3.0, _policy())
        assert plan.heat == StageFleet(pods=0, gpus_per_pod=8, slots=0)


def test_heat_window_floors_at_heat_hours():
    # final_hours consumes the whole epoch → window floors at heat_hours so a
    # slot still fits one serial run: 10 × 0.5 × 1.3 / 0.5 = 13 slots.
    plan = size_fleet(10, 1, 0.5, 3.0, 3.0, _policy(heat_max=16))
    assert plan.heat.slots == 13
    assert plan.heat.pods == 2                              # ⌈13/8⌉


def test_heat_overhead_pads_slots():
    lean = size_fleet(200, 2, 0.5, 24.0, 3.0, _policy(slot_overhead=1.0))
    padded = size_fleet(200, 2, 0.5, 24.0, 3.0, _policy(slot_overhead=2.0))
    assert padded.heat.slots > lean.heat.slots


def test_testnet_shape_small_field():
    # Testnet: 900 blocks ≈ 3h epoch, 0.1h heats, 0.25h final, field of 6.
    plan = size_fleet(6, 1, 0.1, 3.0, 0.25, _policy())
    assert plan.heat.pods == 1 and plan.heat.slots == 1     # 6×0.1×1.3/2.75 → 1


# ── size_fleet: final (king + finalists on ONE pod by default) ───────────────


def test_final_default_is_one_multi_gpu_pod():
    # 1 king + 1 finalist = 2 slots on one 2-GPU pod: the validator's
    # expected_gpu pairing is satisfied by construction (same physical box).
    plan = size_fleet(20, 1, 0.5, 24.0, 3.0, _policy(final_gpus=2))
    assert plan.final == StageFleet(pods=1, gpus_per_pod=2, slots=2)


def test_final_single_gpu_falls_back_to_pod_per_run():
    plan = size_fleet(20, 2, 0.5, 24.0, 3.0, _policy(final_gpus=1))
    assert plan.final == StageFleet(pods=3, gpus_per_pod=1, slots=3)


def test_final_intermediate_shape_takes_ceil_pods():
    plan = size_fleet(20, 3, 0.5, 24.0, 3.0, _policy(final_gpus=2))
    assert plan.final == StageFleet(pods=2, gpus_per_pod=2, slots=4)


def test_final_rented_even_with_empty_field():
    # The king always trains — a round with zero challengers still needs a final pod.
    plan = size_fleet(0, 1, 0.5, 24.0, 3.0, _policy())
    assert plan.final.pods == 1


def test_size_fleet_rejects_bad_inputs():
    with pytest.raises(ValueError):
        size_fleet(-1, 1, 0.5, 24.0, 3.0, _policy())
    with pytest.raises(ValueError):
        size_fleet(5, 1, 0.0, 24.0, 3.0, _policy())
    with pytest.raises(ValueError):
        size_fleet(5, 1, 0.5, 0.0, 3.0, _policy())


# ── within_budget (worst-case circuit breaker) ───────────────────────────────


def _plan(heat_pods=3, final_pods=1):
    return FleetPlan(heat=StageFleet(heat_pods, 8, 20),
                     final=StageFleet(final_pods, 2, 2))


def test_budget_bills_every_pod_for_full_ttl():
    ok, usd = within_budget(_plan(), {"heat": 0.5, "final": 2.0},
                            max_spend=100.0, ttl_hours=24.0)
    assert usd == pytest.approx(3 * 0.5 * 24 + 1 * 2.0 * 24)  # 84.0
    assert ok


def test_budget_refuses_over_cap():
    ok, usd = within_budget(_plan(), {"heat": 0.5, "final": 2.0},
                            max_spend=25.0, ttl_hours=24.0)
    assert not ok and usd == pytest.approx(84.0)


def test_budget_ignores_stage_without_offer():
    # No heat offer ⇒ the loop is not renting heat this round ⇒ it cannot spend.
    ok, usd = within_budget(_plan(), {"final": 2.0}, max_spend=50.0, ttl_hours=24.0)
    assert ok and usd == pytest.approx(48.0)


def test_budget_ignores_empty_stage():
    ok, usd = within_budget(_plan(heat_pods=0), {"heat": 0.5, "final": 2.0},
                            max_spend=50.0, ttl_hours=24.0)
    assert ok and usd == pytest.approx(48.0)


# ── teardown_due (per-stage signals + TTL backstop) ──────────────────────────


def test_heat_pod_dies_on_marker_while_final_runs():
    assert teardown_due("heat", heat_marker_seen=True, manifest_seen=False,
                        rented_at=0.0, now=100.0, ttl_hours=24.0)
    assert not teardown_due("final", heat_marker_seen=True, manifest_seen=False,
                            rented_at=0.0, now=100.0, ttl_hours=24.0)


def test_manifest_kills_both_stages():
    for stage in ("heat", "final"):
        assert teardown_due(stage, heat_marker_seen=False, manifest_seen=True,
                            rented_at=0.0, now=100.0, ttl_hours=24.0)


def test_ttl_backstop_fires_regardless_of_signals():
    for stage in ("heat", "final"):
        assert teardown_due(stage, heat_marker_seen=False, manifest_seen=False,
                            rented_at=0.0, now=24 * 3600.0, ttl_hours=24.0)


def test_no_teardown_before_any_signal():
    for stage in ("heat", "final"):
        assert not teardown_due(stage, heat_marker_seen=False, manifest_seen=False,
                                rented_at=0.0, now=24 * 3600.0 - 1, ttl_hours=24.0)


def test_teardown_rejects_unknown_stage():
    with pytest.raises(ValueError):
        teardown_due("any", heat_marker_seen=False, manifest_seen=False,
                     rented_at=0.0, now=0.0, ttl_hours=24.0)
