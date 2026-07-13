"""Pure fleet policy for the per-round provisioner — sizing, triggering, budget.

Everything here is arithmetic over plain data: no clock, no chain, no provider
API. The service loop (``cascade.provision.loop``) feeds these functions the
observed world (current block, the trainer's ``--plan-only`` field count, live
offer prices) and acts on the returned plan; keeping the decisions pure makes
every sizing/teardown/budget rule unit-testable without a cloud account.

The shape of the problem: a round is one ``epoch_blocks`` window. Shortly
before the boundary (once timed reveals have landed and the field is countable)
the provisioner rents two fleets —

* a **heat** fleet of cheap-SKU pods that screen-trains the whole eligible
  field at ``heat_train_hours`` each, sized so the heat finishes in time for
  the final; and
* a **final** fleet on the pinned SKU where the king and each finalist train
  the full budget. The default shape is ONE multi-GPU pod: the validator's
  ``expected_gpu`` pairing is trivially satisfied when every final run reports
  the same physical machine's GPU, and a single box removes cross-pod variance.

Both fleets are bounded by ``max_pods`` per stage and by a hard
``max_spend_per_round`` circuit breaker computed at worst case (every pod
billed for the full TTL) — a runaway round can cost at most that number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FleetPlan",
    "ProvisionPolicy",
    "StageFleet",
    "StagePolicy",
    "should_trigger",
    "size_fleet",
    "teardown_due",
    "within_budget",
]


@dataclass(frozen=True)
class StagePolicy:
    """The owner's per-stage rental knobs (one for the heat, one for the final).

    ``sku`` is the exact device string (``nvidia-smi --query-gpu=name`` output,
    e.g. ``"NVIDIA L40S"``) the health gate asserts on every GPU of every pod —
    exact, because ``L40`` and ``L40S`` are different silicon and the final's
    ``expected_gpu`` pin is byte-compared. ``gpus_per_pod`` is the pod shape
    (8 for an 8× cluster); each GPU becomes one ``hosts.toml`` entry with its
    own ``cuda_device``. ``providers`` is the marketplace priority order — the
    first with capacity wins, and a stage is never split across providers.
    ``max_price_hr`` (USD, per pod-hour) rejects overpriced offers before the
    round-level budget breaker even runs. ``slot_overhead`` pads the heat's
    slot math for real-world drag (image pulls, corpus build, checkpoint
    fetch, one flaky-pod retry); it is meaningless for the final, whose pod
    count is fixed by ``1 + finalists``, so leave it at the default there.
    """

    sku: str
    gpus_per_pod: int
    max_pods: int
    providers: tuple[str, ...]
    max_price_hr: float
    slot_overhead: float = 1.3
    # The marketplace's name for the same silicon when it differs from the
    # nvidia-smi device string ("A6000" on lium vs "NVIDIA RTX A6000" on the
    # pod). Empty = same as ``sku``. The health gate ALWAYS asserts ``sku``.
    market_sku: str = ""

    @property
    def marketplace_sku(self) -> str:
        return self.market_sku or self.sku


@dataclass(frozen=True)
class ProvisionPolicy:
    """The whole per-round rental policy: two stage shapes plus round-level caps.

    ``trigger_margin_blocks`` is how many blocks before the epoch boundary the
    provisioner counts the field and rents — it must sit inside the trainer's
    reveal margin (the field only settles once timed reveals land) and be small
    enough that pods don't idle-bill for hours before the round starts.
    ``max_spend_per_round`` (USD) is the worst-case circuit breaker (see
    :func:`within_budget`). ``ttl_epochs`` is the hard pod lifetime backstop in
    epochs: even if every teardown signal is missed (trainer crash, storage
    outage), a pod dies ``ttl_epochs`` epochs after it was rented.
    """

    heat: StagePolicy
    final: StagePolicy
    trigger_margin_blocks: int
    max_spend_per_round: float
    ttl_epochs: int = 1


@dataclass(frozen=True)
class StageFleet:
    """One stage's sized fleet: ``pods`` rented boxes of ``gpus_per_pod`` GPUs.

    ``slots`` is the stage's computed GPU-slot *demand* (heat: enough parallel
    slots to screen the field in the available window; final: ``1 +
    finalists``). When ``max_pods`` clamps the pod count, ``slots`` can exceed
    ``pods × gpus_per_pod`` — the stage still completes, in more serial waves
    (the trainer round-robins jobs over however many hosts exist).
    """

    pods: int
    gpus_per_pod: int
    slots: int


@dataclass(frozen=True)
class FleetPlan:
    """The sized rental for one round: a heat fleet and a final fleet."""

    heat: StageFleet
    final: StageFleet


def should_trigger(
    block: int,
    epoch_blocks: int,
    margin_blocks: int,
    already_provisioned_round: int | None,
) -> bool:
    """True when it is time to provision the *upcoming* round.

    The upcoming round is keyed by its boundary block (``next_boundary = (block
    // epoch_blocks + 1) × epoch_blocks`` — also what ``--plan-only`` reports as
    ``next_boundary_block``). We trigger inside the last ``margin_blocks``
    blocks of the current epoch, and only once per round:
    ``already_provisioned_round`` is the boundary block of the last round this
    provisioner rented for (``None`` on a fresh start), so a 30s poll loop that
    stays inside the margin for many iterations still rents exactly one fleet.

    Triggering *late* (near the boundary) is deliberate: with timed reveals the
    eligible field is only countable ~reveal-margin blocks before the boundary,
    and the trainer's ``--hosts-wait-seconds`` covers pod boot time after the
    round starts — so there is no payoff to renting earlier, only idle billing.
    """
    if epoch_blocks <= 0:
        raise ValueError(f"epoch_blocks must be positive; got {epoch_blocks}")
    next_boundary = (block // epoch_blocks + 1) * epoch_blocks
    if next_boundary - block > margin_blocks:
        return False
    return already_provisioned_round != next_boundary


def size_fleet(
    n_eligible: int,
    finalists: int,
    heat_hours: float,
    epoch_hours: float,
    final_hours: float,
    policy: ProvisionPolicy,
) -> FleetPlan:
    """Size both fleets off the revealed field — SLOT-based for multi-GPU pods.

    **Heat**: the field is ``n_eligible`` challengers, each needing one
    ``heat_hours`` screen-train. All of it must finish before the final needs
    its ``final_hours`` at the end of the ``epoch_hours`` round, so the heat's
    available window is ``epoch_hours − final_hours`` (floored at
    ``heat_hours`` so at least one serial run always fits per slot). The
    parallel GPU slots needed are then

        heat_slots = ⌈ n_eligible × heat_hours × slot_overhead / window ⌉

    and pods = ⌈slots / gpus_per_pod⌉, clamped to ``[0, max_pods]``. When the
    whole field already fits in the final (``n_eligible <= finalists``) there
    is nothing to screen — everyone advances — so **no heat pods at all**.

    **Final**: exactly ``1 + finalists`` GPU slots (the king plus each
    finalist). The default shape is ONE pod with ``gpus_per_pod >= 1 +
    finalists``: every final run lands on the same physical box, so the
    validator's ``expected_gpu`` pairing (king and challenger report the same
    GPU) is satisfied by construction. With ``gpus_per_pod = 1`` this falls
    back to ``1 + finalists`` single-GPU pods (the pre-provisioner shape);
    in-between shapes take ``⌈slots / gpus_per_pod⌉`` pods. ``max_pods`` clamps
    here too — a clamped final still completes (the trainer round-robins),
    just serially.
    """
    if n_eligible < 0 or finalists < 0:
        raise ValueError("n_eligible and finalists must be non-negative")
    if heat_hours <= 0 or epoch_hours <= 0 or final_hours < 0:
        raise ValueError("heat_hours/epoch_hours must be positive, final_hours >= 0")

    n_to_screen = n_eligible if n_eligible > finalists else 0
    if n_to_screen > 0:
        window = max(epoch_hours - final_hours, heat_hours)
        heat_slots = math.ceil(n_to_screen * heat_hours * policy.heat.slot_overhead / window)
        heat_pods = _clamp(math.ceil(heat_slots / policy.heat.gpus_per_pod),
                           0, policy.heat.max_pods)
    else:
        heat_slots, heat_pods = 0, 0

    final_slots = 1 + finalists
    # max_pods = 0 means "stage unmanaged": the operator serves it with static
    # hand-rented pods (hosts.toml static entries), so the provisioner rents none.
    if policy.final.max_pods == 0:
        final_pods = 0
    else:
        final_pods = _clamp(math.ceil(final_slots / policy.final.gpus_per_pod),
                            1, policy.final.max_pods)

    return FleetPlan(
        heat=StageFleet(pods=heat_pods, gpus_per_pod=policy.heat.gpus_per_pod,
                        slots=heat_slots),
        final=StageFleet(pods=final_pods, gpus_per_pod=policy.final.gpus_per_pod,
                         slots=final_slots),
    )


def within_budget(
    plan: FleetPlan,
    offers_by_stage: dict[str, float],
    max_spend: float,
    ttl_hours: float,
) -> tuple[bool, float]:
    """The round's spend circuit breaker, computed at WORST case.

    ``offers_by_stage`` maps ``"heat"`` / ``"final"`` to the chosen offer's
    hourly USD price per pod. The projection bills **every pod for the full
    ``ttl_hours``** — not the expected stage duration — because the TTL is the
    only teardown guarantee that needs no cooperating signal (marker, manifest,
    even the provisioner's own state file can be lost). Whatever goes wrong,
    the round cannot cost more than the projection this function approved.

    A stage missing from ``offers_by_stage`` contributes nothing: no offer
    means the loop is not renting that stage this round (provider outage →
    smaller fleet), so it cannot spend. Returns ``(ok, projected_usd)`` so the
    caller can log the number either way.
    """
    projected = 0.0
    for stage, fleet in (("heat", plan.heat), ("final", plan.final)):
        price = offers_by_stage.get(stage)
        if price is None or fleet.pods <= 0:
            continue
        projected += fleet.pods * float(price) * float(ttl_hours)
    return (projected <= max_spend, projected)


def teardown_due(
    stage: str,
    *,
    heat_marker_seen: bool,
    manifest_seen: bool,
    rented_at: float,
    now: float,
    ttl_hours: float,
) -> bool:
    """Whether a pod of ``stage`` should be terminated NOW.

    Per-stage, cheapest-signal-first teardown is the whole point of the
    provisioner: **heat** pods die on the trainer's ``heat_complete.json``
    marker (once the field is screened and finalists chosen, no heat dispatch
    can occur for the rest of the round — see
    ``TrainerRunner._mark_heat_complete``) while the final still runs;
    **final** pods die when the round's manifest publishes (the round is over).
    A published manifest also kills any heat pod the marker missed — the round
    being over subsumes the heat being over.

    The TTL is the hard backstop for BOTH stages: ``rented_at``/``now`` are
    seconds on the same (injected) clock, and once ``ttl_hours`` have elapsed
    the pod dies regardless of signals — a crashed trainer or an unreadable
    manifest store must never turn into an eternally-billing pod.
    """
    if stage not in ("heat", "final"):
        raise ValueError(f"stage must be 'heat' or 'final'; got {stage!r}")
    if now - rented_at >= ttl_hours * 3600.0:
        return True
    if manifest_seen:
        return True
    return stage == "heat" and heat_marker_seen


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))
