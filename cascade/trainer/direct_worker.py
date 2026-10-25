"""Budget-exact worker driver for an explicitly supplied, resumable gang backend.

The backend owns tensors and distributed collectives. It must return all-rank
routing counts after all accumulation passes and encode full continuation state.
No legacy train_moe checkpoint is treated as a complete distributed snapshot.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

from ..shared.direct_moe import digest

COMPONENTS = {"weights", "optimizer", "scheduler", "router", "rng", "sampler", "generator"}


class GangBackend(Protocol):
    world_size: int
    def restore(self, states: dict[int, dict[str, bytes]]) -> None: ...
    def step(self, plan: dict) -> dict[int, list[list[int]]]: ...
    def balance(self, global_loads: list[list[int]]) -> None: ...
    def snapshot(self) -> dict[int, dict[str, bytes]]: ...


def global_routing_loads(rank_loads, world_size):
    if sorted(rank_loads) != list(range(world_size)):
        raise ValueError("routing counts missing a rank")
    first = rank_loads[0]
    if not first or any(not layer for layer in first):
        raise ValueError("empty routing counts")
    out = [[0 for _ in layer] for layer in first]
    for layers in rank_loads.values():
        if len(layers) != len(out) or any(len(a) != len(b) for a, b in zip(layers, out, strict=True)):
            raise ValueError("routing count geometry mismatch")
        for i, layer in enumerate(layers):
            for e, n in enumerate(layer):
                if type(n) is not int or n < 0:
                    raise ValueError("invalid routing assignment count")
                out[i][e] += n
    return out


def run_worker(controller, job, schedule, backend: GangBackend, *, now, checkpoint_every=1,
               max_steps=None, micro_batch=1, expert=False):
    """Run/resume at verified optimizer boundaries. `now` is an injected clock.

    Exact tail masks/data order and gradient normalisation are part of the backend
    contract; a profile cannot be qualified without checking those invariants.
    """
    if checkpoint_every < 1:
        raise ValueError("invalid checkpoint cadence")
    j = controller.ledger.state["jobs"][job]
    if j["state"] != "running" or backend.world_size != j["terms"]["world_size"]:
        raise ValueError("worker gang mismatch/not running")
    if expert != (j["terms"]["role"] == "expert"):
        raise ValueError("expert freeze mode mismatch")
    cohort = controller.ledger.state["cohorts"][j["terms"]["cohort"]]["contract"]
    if digest(cohort["schedule"]) != digest(schedule.__dict__):
        raise ValueError("worker schedule mismatch")
    cp = controller.resume_manifest(job)
    consumed = 0
    if cp is not None:
        states = {int(rank): {kind: controller.objects.get(obj["uri"]) for kind, obj in parts.items()}
                  for rank, parts in cp["ranks"].items()}
        backend.restore(states)
        consumed = cp["consumed"]
    remaining = j["terms"]["budget"] - consumed
    if remaining == 0:
        return cp
    steps = schedule.steps(cohort["exposure"] + consumed, remaining,
                           world_size=backend.world_size, micro_batch=micro_batch)
    for index, plan in enumerate(steps, 1):
        # Backend.step sums every local microbatch before returning rank loads.
        loads = backend.step(plan)
        global_loads = global_routing_loads(loads, backend.world_size)
        if not expert:
            backend.balance(global_loads)
        consumed += plan["valid_values"]
        should_stop = consumed == j["terms"]["budget"] or (max_steps is not None and index >= max_steps)
        if index % checkpoint_every == 0 or should_stop:
            states = backend.snapshot()
            if sorted(states) != list(range(backend.world_size)):
                raise ValueError("snapshot missing a rank")
            ranks = {}
            for rank, parts in states.items():
                if set(parts) != COMPONENTS:
                    raise ValueError("snapshot lacks complete continuation state")
                ranks[str(rank)] = {}
                for kind, body in parts.items():
                    checksum = hashlib.sha256(body).hexdigest()
                    # Remote store adapter owns its URI scheme; no controller disk.
                    uri = f"{controller.objects.prefix}{checksum}"
                    controller.objects.put(uri, body)
                    ranks[str(rank)][kind] = {"uri": uri, "sha256": checksum}
            cp = dict(job=job, fence=j["fence"], contract=digest(j["terms"]), kind="continuation",
                      accumulation_boundary=True, consumed=consumed, ranks=ranks)
            controller.checkpoint(job, cp, now=now())
        if should_stop:
            return cp
    return cp
