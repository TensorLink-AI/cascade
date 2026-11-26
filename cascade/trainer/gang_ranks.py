"""How many data-parallel ranks a pyramid leg trains on, and the guard that
refuses a short gang. Torch-free: :mod:`.pyramid_dp` feeds it the CUDA device
count, tests feed it the dispatch env.

A leg's ranks are every CUDA device its process can see — one rule for lane A
and a lane B expert leg, for a pod dispatched with an explicit device list
(funded legs: ``CUDA_VISIBLE_DEVICES=0,1,…,7``) or as the whole pod
(operator legs: ``cuda_device=None``, no mask, every GPU visible).

2026-11-26 (lane B expert leg 5GMzsP…, operator 8xRTX4090): the worker image
pinned for the leg (rc15, main 5e3fade) still carried
``world = gang_size(trainer.device) if slices is None else 1`` — an expert leg
trained on GPU 0 while 7 GPUs idled, ~8x slower, and was killed 10 minutes in
by the throughput watchdog. :func:`check_full_gang` turns that class of bug
(any path that builds fewer ranks than visible GPUs) into an immediate,
named failure instead of a silent slowdown.
"""

from __future__ import annotations

import contextlib
import os

GANG_ENV = "CASCADE_TRAIN_GPUS"
VISIBLE_ENV = "CUDA_VISIBLE_DEVICES"


class GangShortfall(RuntimeError):
    """A multi-GPU leg started with fewer training ranks than visible GPUs."""


def visible_gpu_count(physical: int, environ=None) -> int:
    """GPUs a process sees under ``environ``'s ``CUDA_VISIBLE_DEVICES`` on a
    box with ``physical`` GPUs: unset = all of them, ``""`` = none, else the
    listed ordinals (capped at ``physical``)."""
    env = os.environ if environ is None else environ
    physical = max(0, int(physical))
    if VISIBLE_ENV not in env:
        return physical
    listed = [d for d in str(env[VISIBLE_ENV]).split(",") if d.strip()]
    return min(physical, len(listed))


def gang_override(environ=None) -> int | None:
    """``CASCADE_TRAIN_GPUS`` as an int, or ``None`` when unset / unreadable."""
    env = os.environ if environ is None else environ
    with contextlib.suppress(TypeError, ValueError):
        return int(env[GANG_ENV]) if GANG_ENV in env else None
    return None


def leg_ranks(device, visible: int, environ=None) -> int:
    """Training ranks for a leg on ``device`` with ``visible`` CUDA devices:
    every visible device for a plain ``"cuda"`` device (``CASCADE_TRAIN_GPUS``
    may lower it), else 1. Never depends on the leg's lane or role."""
    if str(device) != "cuda" or int(visible) < 1:
        return 1
    n = int(visible)
    override = gang_override(environ)
    if override is not None:
        n = min(n, override)
    return max(1, n)


def check_full_gang(world: int, visible: int, environ=None, *, device="cuda") -> None:
    """Fail fast when a CUDA leg on a multi-GPU pod trains on fewer ranks than
    the GPUs it can see. An explicit ``CASCADE_TRAIN_GPUS`` (a deliberate
    smaller gang, e.g. the bitwise qualification) is honoured, never refused."""
    if not str(device).startswith("cuda") or int(visible) <= 1 or int(world) >= int(visible):
        return
    if gang_override(environ) is not None:
        return
    raise GangShortfall(
        f"pyramid: {int(world)} training rank(s) on a pod with {int(visible)} visible GPUs "
        f"(device={device!s}, {VISIBLE_ENV}="
        f"{(os.environ if environ is None else environ).get(VISIBLE_ENV, '<unset>')!r}) — "
        f"refusing a ~{int(visible) // max(1, int(world))}x slower leg; every leg (lane A or "
        f"a lane B expert leg) must train on every visible GPU (set {GANG_ENV} to opt into "
        f"a smaller gang deliberately)")
