"""Eval-pool **builder** — the owner-side tooling that produces the held-out
corpus the validator scores on.

:mod:`metronome.validator.pool` is the *consumer*: it fetches a Hippius registry
CID, loads its ``.npy`` / ``.npz`` series + ``metadata.json``, and slices them
into :class:`~metronome.eval.window.EvalWindow` s. This package is the *producer*
of exactly that directory layout.

The design follows the three anti-gaming levers the eval depends on
(``OPEN_QUESTIONS.md`` #6):

* **Privacy** — the pool is owner-controlled and never published as a named
  public benchmark a generator could distribution-match.
* **Freshness** — sources harvest *recent, real* data up to an ``as_of`` cutoff,
  and the operator re-harvests periodically so each pinned pool rotates in time.
  Data that did not exist at generator-submission time cannot be memorised.
* **Breadth** — multiple real-world domains (weather, web traffic, …) at
  sub-daily frequency, so the only way to score well is to forecast generally,
  not to match one distribution.

Pieces:

* :mod:`.source` — the pluggable :class:`DataSource` interface, the
  :class:`HarvestedSeries` record a source yields, and a mockable
  :class:`HttpFetcher` so the live sources are testable offline.
* :mod:`.builder` — the deterministic core: clean (gap-fill / drop), validate
  (length, channels, degeneracy), dedup, and write the on-disk pool in the exact
  format :mod:`metronome.validator.pool` reads back.
* :mod:`.sources` — concrete sources (``openmeteo``, ``wikimedia``,
  ``synthetic``) and a name registry.
* :mod:`.cli` — ``metronome-pool build [--upload]``.
"""

from __future__ import annotations

from .builder import BuildSummary, PoolBuildConfig, SeriesRecord, build_pool, prepare_series
from .source import DataSource, HarvestContext, HarvestedSeries, HarvestError, HttpFetcher

__all__ = [
    "BuildSummary",
    "PoolBuildConfig",
    "SeriesRecord",
    "build_pool",
    "prepare_series",
    "DataSource",
    "HarvestContext",
    "HarvestedSeries",
    "HarvestError",
    "HttpFetcher",
]
