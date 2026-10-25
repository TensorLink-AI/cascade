"""Pure rental policy for the provisioner — stage shapes, pod counts, teardown.

Everything here is arithmetic over plain data: no clock, no chain, no provider
API. The service loop (``cascade.provision.loop``) feeds these functions the
observed world and acts on the answer; keeping the decisions pure makes every
rule unit-testable without a cloud account.

The provisioner's one rental stage is the validator's **eval** pod (one box,
usually one GPU) for its heavy evals (GIFT-Eval gate, cascade bench). It is
manifest-triggered — rented when a round's manifest publishes — and dies when
that round's receipt appears (or a newer manifest makes it moot, or the TTL).
Absent config (``policy.eval is None`` / ``max_pods = 0``) the stage does not
exist and validators eval locally. The trainer rents its own pods (payer legs,
the era king) — see :mod:`cascade.provision.funded`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProvisionPolicy",
    "SkuCandidate",
    "StagePolicy",
    "pods_for_slots",
    "teardown_due",
]


@dataclass(frozen=True)
class SkuCandidate:
    """One rentable (SKU, shape, price) option for a stage.

    ``sku`` is the exact nvidia-smi device string (the health gate asserts it
    on the pod that was ACTUALLY rented); ``market_sku`` the marketplace's
    alias when it differs; ``gpus_per_pod`` the pod shape to rent;
    ``max_price_hr`` the per-pod-hour cap for THIS candidate (a 4× box
    legitimately costs more than a 1× one).
    """

    sku: str
    market_sku: str = ""
    gpus_per_pod: int = 1
    max_price_hr: float = 0.0

    @property
    def marketplace_sku(self) -> str:
        return self.market_sku or self.sku


@dataclass(frozen=True)
class StagePolicy:
    """The owner's rental knobs for one stage (today: the eval pod).

    ``sku`` is the exact device string (``nvidia-smi --query-gpu=name`` output,
    e.g. ``"NVIDIA L40S"``) the health gate asserts on every GPU of every pod —
    exact, because ``L40`` and ``L40S`` are different silicon and the final's
    ``expected_gpu`` pin is byte-compared. ``gpus_per_pod`` is the pod shape
    (8 for an 8× cluster); each GPU becomes one ``hosts.toml`` entry with its
    own ``cuda_device``. ``providers`` is the marketplace priority order — the
    first with capacity wins, and a stage is never split across providers.
    ``max_price_hr`` (USD, per pod-hour) rejects overpriced offers before the
    budget breaker even runs.
    """

    sku: str
    gpus_per_pod: int
    max_pods: int
    providers: tuple[str, ...]
    max_price_hr: float
    # The marketplace's name for the same silicon when it differs from the
    # nvidia-smi device string ("A6000" on lium vs "NVIDIA RTX A6000" on the
    # pod). Empty = same as ``sku``. The health gate ALWAYS asserts ``sku``.
    market_sku: str = ""
    # Ordered SKU fallbacks tried AFTER the primary (sku/gpus_per_pod/
    # max_price_hr above): the first candidate × provider with capacity for
    # the whole stage wins; a stage never mixes candidates.
    candidates: tuple[SkuCandidate, ...] = ()

    @property
    def marketplace_sku(self) -> str:
        return self.market_sku or self.sku

    @property
    def sku_candidates(self) -> tuple[SkuCandidate, ...]:
        """The primary shape first, then the configured fallbacks, in order."""
        primary = SkuCandidate(sku=self.sku, market_sku=self.market_sku,
                               gpus_per_pod=self.gpus_per_pod,
                               max_price_hr=self.max_price_hr)
        return (primary, *self.candidates)


@dataclass(frozen=True)
class ProvisionPolicy:
    """The rental policy: the eval stage plus the service-level caps.

    ``max_spend_per_round`` (USD) is the worst-case circuit breaker for one
    rental (every pod billed for the full TTL). ``ttl_epochs`` is the hard pod
    lifetime backstop in epochs: even if every teardown signal is missed
    (validator silent, storage outage), a pod dies ``ttl_epochs`` epochs after
    it was rented. ``eval`` is ``None`` (no [provisioner.eval] table) or
    ``max_pods = 0`` when no eval pod is ever rented.
    """

    max_spend_per_round: float
    ttl_epochs: int = 1
    eval: StagePolicy | None = None


def pods_for_slots(slots: int, gpus_per_pod: int, max_pods: int) -> int:
    """Pods needed to serve ``slots`` GPU-slots at a candidate's shape.

    ``max_pods`` clamps (a clamped stage still completes, in serial waves);
    ``max_pods == 0`` means the stage is unmanaged — always 0 pods. Zero slots
    rents nothing regardless of shape.
    """
    if slots <= 0 or max_pods <= 0:
        return 0
    return min(math.ceil(slots / max(1, gpus_per_pod)), max_pods)


def teardown_due(
    stage: str,
    *,
    rented_at: float,
    now: float,
    ttl_hours: float,
    receipt_seen: bool = False,
    newer_manifest: bool = False,
) -> bool:
    """Whether a pod of ``stage`` should be terminated NOW.

    **eval** pods are rented when a round's manifest publishes (the
    validator's heavy evals start then) and die when that round's receipt
    appears (``receipt_seen`` — no more offloaded evals are coming) or a
    NEWER manifest supersedes the round they served (``newer_manifest``).

    The TTL is the hard backstop for EVERY stage: ``rented_at``/``now`` are
    seconds on the same (injected) clock, and once ``ttl_hours`` have elapsed
    the pod dies regardless of signals — a silent validator or an unreadable
    store must never turn into an eternally-billing pod. ``heat``/``final``
    pods are only ever ledger leftovers of the retired per-round fleet: the
    TTL alone reaps them.
    """
    if stage not in ("heat", "final", "eval"):
        raise ValueError(f"stage must be 'heat', 'final', or 'eval'; got {stage!r}")
    if now - rented_at >= ttl_hours * 3600.0:
        return True
    return stage == "eval" and (receipt_seen or newer_manifest)
