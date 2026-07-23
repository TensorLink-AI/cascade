"""Eval-window source — a rotating, private, per-round held-out set.

The KOTH comparison is only a *controlled* measurement if the king's model and
the challenger's model are scored on the same windows; it is only *contamination
resistant* if those windows are not a fixed public benchmark a generator can
distribution-match. cascade resolves both with a single rule:

    windows_for_round(round_seed) → a seeded slice of a private pool

* **Same within a round.** King and challenger are scored on the identical slice
  (one ``round_seed`` per round → one selection), so the eval cancels and the
  comparison is paired.
* **Agreed across validators.** ``round_seed`` is the round's chain block hash,
  so every honest validator draws the byte-identical slice and they converge on
  the same KOTH verdict.
* **Rotated across rounds.** A different ``round_seed`` permutes the pool
  differently, so the scored slice rotates — there is no fixed eval set to overfit
  (the TIME-benchmark philosophy: fresh data each time).

The pool itself is owner-controlled and **private** (``[eval] window_pool``).
Loading it — pulling the held-out corpus and slicing it into context/target
windows — is the integrator boundary; this module owns
the deterministic, validator-agreeing *selection and rotation*, which is pure and
unit-tested. :func:`build_windows_from_series` is the reference cutter the loader
can use to turn raw held-out series into :class:`EvalWindow` s.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..eval.window import EvalWindow


def _seed_to_int(seed: int | str) -> int:
    """Stable seed → 64-bit int (blake2b), matching the eval bootstrap so the
    same ``round_seed`` string maps identically everywhere a validator uses it."""
    if isinstance(seed, int):
        return seed
    h = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


@runtime_checkable
class WindowSource(Protocol):
    """Produces the eval windows for one round, deterministic in ``round_seed``.

    ``block`` (the round's epoch-boundary block number) is an optional selector
    for sources that rotate a daily snapshot (the bucket pool); a static pool
    ignores it. Keeping it keyword-only + defaulted preserves the simple
    ``windows_for_round(seed, n)`` call for static sources.
    """

    def windows_for_round(
        self, round_seed: int | str, n_windows: int, *, block: int | None = None
    ) -> list[EvalWindow]:
        ...


@dataclass(frozen=True)
class RotatingWindowSource:
    """Select a rotating slice of a fixed private pool, seeded by the round.

    The pool is the materialised private held-out set (injected; the loader is a
    boundary). Selection is a seeded permutation prefix: deterministic in
    ``round_seed`` (so all validators agree and king/challenger pair), and
    rotating across rounds (so no fixed slice can be overfit). When the pool has
    fewer than ``n_windows`` entries the whole pool is returned in a seeded order
    and the KOTH ``min_windows`` gate decides whether the round is conclusive.
    """

    pool: tuple[EvalWindow, ...]
    # Where the pool came from, for the public round receipt: ``(ref, digest)``
    # — the Hub ``repo@digest`` and its OCI digest for a static pool. ("", "")
    # for an injected/test pool.
    provenance: tuple[str, str] = ("", "")

    def __post_init__(self) -> None:
        if not self.pool:
            raise ValueError("RotatingWindowSource pool is empty")

    def provenance_for_round(self, round_seed: int | str, *, block: int | None = None) -> tuple[str, str]:
        """``(pool_ref, pool_digest)`` recorded in the round receipt. A static
        pool's provenance is round-independent (``block`` ignored)."""
        return self.provenance

    def windows_for_round(
        self, round_seed: int | str, n_windows: int, *, block: int | None = None
    ) -> list[EvalWindow]:
        # ``block`` is only meaningful for a rotating daily snapshot (bucket
        # pool); a static pool draws its slice from ``round_seed`` alone.
        if n_windows <= 0:
            raise ValueError(f"n_windows must be positive; got {n_windows}")
        rng = np.random.default_rng(_seed_to_int(round_seed))
        perm = rng.permutation(len(self.pool))
        take = min(n_windows, len(self.pool))
        return [self.pool[i] for i in perm[:take]]


def build_windows_from_series(
    series: list[np.ndarray],
    *,
    context_length: int,
    horizon: int,
    metadata: list[dict] | dict | None = None,
    id_prefix: str = "w",
) -> list[EvalWindow]:
    """Cut held-out series into context/target :class:`EvalWindow` s.

    Each series contributes one window: the last ``horizon`` steps are the target
    and up to ``context_length`` steps before them are the history. Accepts 1-D
    ``(L,)`` or 2-D ``(C, L)`` series (1-D becomes a single channel). A series too
    short to yield ``horizon`` targets plus at least one context step is skipped.

    This is the reference cutter for a :class:`WindowSource` loader; it is pure
    and deterministic so the resulting pool is identical for every validator.
    """
    if context_length <= 0 or horizon <= 0:
        raise ValueError("context_length and horizon must be positive")
    out: list[EvalWindow] = []
    for i, raw in enumerate(series):
        arr = np.atleast_2d(np.asarray(raw, dtype=np.float64))   # (C, L)
        length = arr.shape[-1]
        if length < horizon + 1:
            continue
        ctx = min(context_length, length - horizon)
        history = arr[:, length - horizon - ctx : length - horizon]
        target = arr[:, length - horizon :]
        if metadata is None:
            md: dict = {}
        elif isinstance(metadata, dict):
            md = metadata
        else:
            md = metadata[i]
        out.append(
            EvalWindow(series_id=f"{id_prefix}{i}", history=history, target=target, metadata=md)
        )
    return out
