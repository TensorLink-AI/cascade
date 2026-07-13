"""Crash-safe rental ledger — what the provisioner owns, on disk, at all times.

The one unforgivable failure mode for a provisioner is the **leaked pod**: a
box that keeps billing after the process that rented it died. The defence is a
write-ahead ledger: every instance is recorded here (atomically, tmp + rename)
the moment it is launched, *before* anything else happens to it, so a restart
can always resume teardown from disk. The complement is :func:`reconcile`,
which catches the opposite hole — a pod the provider created but whose record
we failed to write (crash between the API call and the save): anything live
and tagged ``cascade-`` that the ledger does not own is an orphan to kill.

Everything here is pure functions over plain data plus two tiny I/O helpers
(:func:`load_state` / :func:`save_state`); time enters only as caller-supplied
ISO strings, so tests inject their own clock.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = [
    "PodInstance",
    "RoundState",
    "add_instance",
    "drop_instances",
    "instances_for_stage",
    "load_state",
    "owned_ids",
    "reconcile",
    "save_state",
]


@dataclass(frozen=True)
class PodInstance:
    """One rented pod: where it lives, what stage it serves, when its TTL started.

    ``rented_at_iso`` (UTC ISO-8601, from the injected clock) is the TTL
    anchor — teardown math (:func:`cascade.provision.policy.teardown_due`)
    runs off it, so it must be recorded at launch, not at first health check.
    """

    provider: str
    instance_id: str
    stage: str                       # "heat" | "final"
    rented_at_iso: str


@dataclass(frozen=True)
class RoundState:
    """Everything the provisioner owns for one round, JSON-serialisable.

    ``round_id`` is the round key the provisioner rents under (the upcoming
    boundary block at trigger time — it need not equal the trainer's
    base-seed round id, which is only knowable at the boundary).
    ``published`` records that ``hosts.toml`` was written for this fleet, so a
    restart mid-round knows whether the trainer may already be dispatching.
    """

    round_id: str
    instances: tuple[PodInstance, ...] = field(default=())
    published: bool = False


# ── pure transforms (the loop composes these, then saves) ────────────────────


def add_instance(state: RoundState, inst: PodInstance) -> RoundState:
    """The state with ``inst`` appended (immutably — callers save the result)."""
    return replace(state, instances=(*state.instances, inst))


def drop_instances(state: RoundState, instance_ids: set[str]) -> RoundState:
    """The state with every instance in ``instance_ids`` removed (post-teardown)."""
    return replace(
        state,
        instances=tuple(i for i in state.instances if i.instance_id not in instance_ids),
    )


def instances_for_stage(state: RoundState, stage: str) -> tuple[PodInstance, ...]:
    """The owned instances serving ``stage``."""
    return tuple(i for i in state.instances if i.stage == stage)


def owned_ids(state: RoundState) -> set[str]:
    """Every instance id the ledger owns (reconcile's 'ours' set)."""
    return {i.instance_id for i in state.instances}


def reconcile(owned_ids: set[str], tagged_live_ids: set[str]) -> list[str]:
    """Orphans: live pods tagged ``cascade-`` that this ledger does NOT own.

    ``tagged_live_ids`` is what the provider reports as live under our naming
    tag; anything there but not in ``owned_ids`` was rented by a run of this
    provisioner that lost its record (crash between launch and save, a deleted
    state file) — kill it. The reverse difference (owned but not live) needs
    no action: the pod is already gone and ``terminate`` is idempotent anyway.

    Deliberately restricted to *tagged* ids so a shared provider account's
    unrelated pods are never touched. Returns a sorted list for stable logs.
    """
    return sorted(tagged_live_ids - owned_ids)


# ── disk I/O (atomic: a watcher/restart never reads a torn ledger) ───────────


def save_state(path: Path | str, state: RoundState) -> None:
    """Write the ledger atomically (tmp + ``os.replace``).

    Called after *every* mutation, launch first — the record must hit disk
    before the pod is relied on, or a crash leaks it (reconcile is the
    backstop, but only for pods the provider tags; belt and braces).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round_id": state.round_id,
        "published": state.published,
        "instances": [
            {
                "provider": i.provider,
                "instance_id": i.instance_id,
                "stage": i.stage,
                "rented_at_iso": i.rented_at_iso,
            }
            for i in state.instances
        ],
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def load_state(path: Path | str) -> RoundState | None:
    """Read the ledger back, or ``None`` when there is none (fresh start).

    A corrupt file raises — silently starting fresh over an unreadable ledger
    is exactly how pods leak; the operator should look before the service
    proceeds (the orphan reconciler would eventually catch tagged pods, but
    ``None``-on-corruption would make that the *primary* mechanism).
    """
    p = Path(path)
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return RoundState(
        round_id=str(raw["round_id"]),
        published=bool(raw.get("published", False)),
        instances=tuple(
            PodInstance(
                provider=str(i["provider"]),
                instance_id=str(i["instance_id"]),
                stage=str(i["stage"]),
                rented_at_iso=str(i["rented_at_iso"]),
            )
            for i in raw.get("instances", [])
        ),
    )
