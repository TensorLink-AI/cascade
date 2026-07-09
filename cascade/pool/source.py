"""Data-source interface for the eval-pool builder.

A :class:`DataSource` turns a remote, real-world feed into a stream of
:class:`HarvestedSeries`. The network call is injected as a ``fetch`` callable so
sources are unit-testable against canned API JSON with no live endpoint — the
default :class:`HttpFetcher` is the only thing that touches the network, and it
is the seam the tests replace.

Sources yield *raw* series (whatever the API returns, gaps and all); all
cleaning, validation, length normalisation, and de-duplication happen once in
:mod:`.builder`, so a new source only has to map an endpoint to arrays plus a
pandas-style ``freq`` (which fixes MASE seasonality downstream).
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


class HarvestError(RuntimeError):
    """A source could not fetch or parse its data."""


# A JSON fetcher: ``fetch(url, params) -> parsed JSON``. Injected into sources so
# the network is a single mockable boundary.
FetchJson = Callable[[str, "dict[str, Any] | None"], Any]


@dataclass(frozen=True)
class HarvestContext:
    """Inputs a source needs to scope its pull.

    ``as_of`` is the freshness cutoff (harvest data up to this date); ``span_days``
    is how much recent history to request. ``context_length`` / ``horizon`` mirror
    ``[eval]`` so a source can size its request to cover at least one full window,
    and ``max_series`` caps how many series a single source contributes.
    """

    as_of: dt.date
    span_days: int = 210
    context_length: int = 4096
    horizon: int = 64
    max_series: int = 10_000


@dataclass(frozen=True)
class HarvestedSeries:
    """One raw series emitted by a source, before cleaning/validation.

    Attributes:
        series_id: globally-unique id (sources namespace it, e.g.
            ``"openmeteo__tokyo__temperature_2m"``). Becomes the on-disk filename
            stem and the ``metadata.json`` key, so it must survive sanitisation
            to a stable, unique value.
        values: ``(L,)`` or ``(C, L)``; may contain NaN/None-derived gaps.
        freq: pandas-style frequency (``"H"``, ``"D"``, …) driving MASE
            seasonality via :func:`cascade.eval.seasonality.get_seasonality`.
        domain: coarse bucket (``"weather"``, ``"web_traffic"``, …) used for
            per-domain caps and provenance.
        seasonal_period: optional explicit override; if ``None`` it is derived
            from ``freq`` at build time.
        source: optional upstream feed id (e.g. a tsbench-forge catalog id).
            Written into pool ``metadata.json`` as the cluster key the KOTH
            cluster bootstrap resamples — series from one feed are correlated,
            so they must move together. ``None`` (the default) leaves each
            series its own cluster.
        attrs: free-form provenance (lat/lon, article title, …).
    """

    series_id: str
    values: np.ndarray
    freq: str
    domain: str
    seasonal_period: int | None = None
    source: str | None = None
    attrs: dict = field(default_factory=dict)


@runtime_checkable
class DataSource(Protocol):
    """A named producer of real-world series for the held-out pool."""

    name: str

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        """Yield :class:`HarvestedSeries`, calling ``fetch`` for any network I/O."""
        ...


@dataclass
class HttpFetcher:
    """Default JSON fetcher over stdlib ``urllib`` with bounded retries.

    Stdlib-only (no ``requests`` dependency) and synchronous. Honours the
    standard ``HTTP(S)_PROXY`` environment (``urllib`` reads ``getproxies()``),
    so it works behind an operator proxy without extra config. Retries on
    transient transport errors and 5xx with exponential backoff; other 4xx fail
    fast.

    Rate limiting (HTTP 429) gets its own, more patient budget: many-call
    sources (Open-Meteo's global grid is ~250 requests) trip per-window limits,
    and a 1-2s backoff just re-trips them. On 429 the fetcher honours a
    ``Retry-After`` header when present, else backs off geometrically up to
    ``max_backoff``, for up to ``rate_limit_retries`` attempts — separate from
    the ``retries`` budget so a 429 storm doesn't exhaust the transient-error
    allowance in a couple of seconds.
    """

    timeout: float = 30.0
    retries: int = 3
    backoff: float = 1.5
    rate_limit_retries: int = 6
    rate_limit_backoff: float = 2.0
    max_backoff: float = 60.0
    user_agent: str = "cascade-pool/1 (+https://github.com/TensorLink-AI/cascade)"
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    @staticmethod
    def _retry_after(e: urllib.error.HTTPError) -> float | None:
        """Seconds from a ``Retry-After`` header (integer form only), or None."""
        raw = e.headers.get("Retry-After") if e.headers else None
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None  # HTTP-date form: fall back to geometric backoff

    def __call__(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last: Exception | None = None
        attempt = 0          # transient (5xx / transport) budget
        rl_attempt = 0       # rate-limit (429) budget, separate
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 — fixed https hosts
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429:
                    if rl_attempt >= self.rate_limit_retries:
                        break
                    wait = self._retry_after(e)
                    if wait is None:
                        wait = min(self.max_backoff, self.rate_limit_backoff ** rl_attempt)
                    rl_attempt += 1
                    self._sleep(min(wait, self.max_backoff))
                    continue
                # Any other 4xx won't improve on retry.
                if e.code < 500:
                    raise HarvestError(f"http_{e.code} for {url}") from e
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last = e
            attempt += 1
            if attempt >= self.retries:
                break
            self._sleep(self.backoff ** attempt)
        raise HarvestError(f"fetch_failed after {self.retries} tries: {url}: {last}")


def daterange(as_of: dt.date, span_days: int) -> tuple[str, str]:
    """``(start, end)`` ISO dates spanning ``span_days`` back from ``as_of``."""
    start = as_of - dt.timedelta(days=max(1, span_days))
    return start.isoformat(), as_of.isoformat()
