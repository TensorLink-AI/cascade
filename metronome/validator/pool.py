"""Private eval-window **pool loader** — the integrator boundary from
OPEN_QUESTIONS.md #6, now wired to Hippius.

:mod:`metronome.validator.windows` owns the deterministic, validator-agreeing
*selection and rotation* of windows; this module owns the other half: pulling the
owner-controlled, held-out series pool and slicing it into :class:`EvalWindow` s.

The pool is referenced by ``[eval] window_pool`` as a Hippius Hub **ref**
(``repo@digest``) — the owner uploads the held-out corpus to the registry with
``upload_dir_to_hub`` and pins the ref here. It is private (not a public
benchmark) and refreshed periodically so it stays contamination-resistant. The
directory behind the ref holds one or more array files:

* ``*.npy``            — a single ``(L,)`` or ``(C, L)`` series each.
* ``*.npz``            — many arrays under arbitrary keys, each a series.
* ``metadata.json``    — optional ``{series_id: {freq / seasonal_period: ...}}``
  used to drive MASE seasonality (matched to a window by its source filename/key).

Every series contributes one window (last ``horizon`` steps = target, up to
``context_length`` before = history) via the pure cutter
:func:`metronome.validator.windows.build_windows_from_series`, so the resulting
pool is byte-identical for every validator that fetches the same ref.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..shared.config import ChainConfig
from ..shared.hippius import HubConfig, HubRef, fetch_from_hub, is_hub_ref
from .windows import RotatingWindowSource, build_windows_from_series

log = logging.getLogger("metronome.validator")


class PoolError(RuntimeError):
    """The eval pool could not be loaded or sliced."""


def _load_series_dir(d: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load every ``*.npy`` / ``*.npz`` array under ``d`` into a series list.

    Returns ``(series, source_ids)`` in a **stable sorted order** (by filename,
    then by key within an ``.npz``) so the pool is identical across validators.
    """
    series: list[np.ndarray] = []
    ids: list[str] = []
    for p in sorted(d.rglob("*.npy")):
        arr = np.load(p, allow_pickle=False)
        series.append(np.asarray(arr, dtype=np.float64))
        ids.append(p.stem)
    for p in sorted(d.rglob("*.npz")):
        with np.load(p, allow_pickle=False) as npz:
            for key in sorted(npz.files):
                series.append(np.asarray(npz[key], dtype=np.float64))
                ids.append(f"{p.stem}:{key}")
    return series, ids


def load_pool(
    cfg: ChainConfig,
    *,
    cache_dir: Path | str | None = None,
    hub: HubConfig | None = None,
) -> RotatingWindowSource:
    """Fetch the private pool ref from the Hippius Hub registry and build the
    window source. Raises :class:`PoolError` on a missing/empty/malformed pool.

    Window geometry comes from ``[eval]`` (``context_length`` / ``horizon``),
    which the config pins equal to ``[training]`` so trained models fit the
    windows.
    """
    ref = cfg.eval.window_pool
    if not ref or not is_hub_ref(ref):
        raise PoolError(
            f"[eval] window_pool must be a Hippius Hub ref (repo@digest); got {ref!r}. "
            "Upload the held-out pool with upload_dir_to_hub and pin its ref."
        )
    hub = hub or HubConfig.from_storage(cfg.storage)
    dest = Path(cache_dir or "./_eval_pool") / HubRef.parse(ref).digest.replace(":", "-")
    try:
        fetch_from_hub(ref, dest, hub)
    except Exception as e:  # noqa: BLE001
        raise PoolError(f"pool_fetch_failed: {e}") from e

    return window_source_from_dir(dest, cfg, label=f"ref={ref}")


def window_source_from_dir(dest: Path, cfg: ChainConfig, *, label: str) -> RotatingWindowSource:
    """Load a pool directory (``.npy``/``.npz`` + optional ``metadata.json``) into
    a :class:`RotatingWindowSource`. Shared by the static-CID and bucket loaders."""
    series, ids = _load_series_dir(dest)
    if not series:
        raise PoolError(f"pool {label} contained no .npy/.npz series under {dest}")

    metadata_p = dest / "metadata.json"
    md_map: dict = {}
    if metadata_p.is_file():
        try:
            md_map = json.loads(metadata_p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("ignoring unreadable pool metadata.json: %s", e)
    metadata = [md_map.get(sid, {}) for sid in ids]

    windows = build_windows_from_series(
        series,
        context_length=cfg.eval.context_length,
        horizon=cfg.eval.horizon,
        metadata=metadata,
        id_prefix="",
    )
    if not windows:
        raise PoolError(
            f"pool {label} had {len(series)} series but none were long enough for "
            f"horizon={cfg.eval.horizon}+context (need >= horizon+1 steps)"
        )
    log.info("loaded eval pool %s series=%d windows=%d", label, len(series), len(windows))
    return RotatingWindowSource(pool=tuple(windows))


# ───────────────────────── daily bucket-published pool ──────────────────────


class BucketWindowSource:
    """A :class:`~metronome.validator.windows.WindowSource` backed by the daily
    snapshot bucket (``[storage] pool_bucket`` + ``pool/index.json``).

    Per round it (re-)reads the owner-controlled index and selects the snapshot
    whose ``effective_round`` is the greatest ``<= round_id`` — the **same**
    deterministic choice on every validator, so async polling cannot diverge.
    The chosen snapshot is fetched once (sha256-verified), cached, and reused;
    its :class:`RotatingWindowSource` then draws the rotating slice for the round.

    ``round_seed`` carries the round id (the live loop passes ``int(round_id)``),
    which is both the snapshot selector and the rotation seed.
    """

    def __init__(self, cfg: ChainConfig, store: object, *, cache_dir: Path | str | None = None,
                 max_cached: int = 3) -> None:
        self.cfg = cfg
        self.store = store
        self.cache_dir = Path(cache_dir or "./_eval_pool")
        self.max_cached = max_cached
        self._cache: dict[str, RotatingWindowSource] = {}

    def _ensure_snapshot(self, meta) -> RotatingWindowSource:
        src = self._cache.get(meta.sha256)
        if src is not None:
            return src
        from ..shared.hippius import fetch_pool_snapshot

        dest = self.cache_dir / f"snapshot-{meta.effective_round}-{meta.sha256[:12]}"
        try:
            fetch_pool_snapshot(self.store, meta, dest)
        except Exception as e:  # noqa: BLE001
            raise PoolError(f"snapshot_fetch_failed (round>={meta.effective_round}): {e}") from e
        src = window_source_from_dir(dest, self.cfg, label=f"snapshot@{meta.effective_round}")
        if len(self._cache) >= self.max_cached:
            self._cache.pop(next(iter(self._cache)))  # evict oldest inserted
        self._cache[meta.sha256] = src
        return src

    def windows_for_round(self, round_seed, n_windows):
        from ..shared.hippius import read_pool_index, select_snapshot

        try:
            round_id = int(round_seed)
        except (TypeError, ValueError) as e:
            raise PoolError(f"bucket pool needs an integer round id; got {round_seed!r}") from e

        index = read_pool_index(self.store)
        meta = select_snapshot(index, round_id)
        if meta is None:
            raise PoolError(
                f"no eval-pool snapshot published in {self.cfg.storage.pool_bucket}; "
                "run `metronome-pool publish` from the owner orchestrator"
            )
        return self._ensure_snapshot(meta).windows_for_round(round_seed, n_windows)


def load_bucket_pool(
    cfg: ChainConfig, *, cache_dir: Path | str | None = None, store: object | None = None
) -> BucketWindowSource:
    """Build a :class:`BucketWindowSource` for ``[storage] pool_bucket``."""
    if not cfg.storage.pool_bucket:
        raise PoolError("[storage] pool_bucket is empty; cannot load a bucket pool")
    if store is None:
        from ..shared.hippius import pool_s3_store

        store = pool_s3_store(cfg.storage)
    return BucketWindowSource(cfg, store, cache_dir=cache_dir)
