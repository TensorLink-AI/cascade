"""Private eval-window **pool loader** — the integrator boundary from
OPEN_QUESTIONS.md #6, now wired to Hippius.

:mod:`metronome.validator.windows` owns the deterministic, validator-agreeing
*selection and rotation* of windows; this module owns the other half: pulling the
owner-controlled, held-out series pool and slicing it into :class:`EvalWindow` s.

The pool is referenced by ``[eval] window_pool`` as a Hippius registry **CID**
(the owner uploads the held-out corpus to the registry with ``upload_dir_to_registry``
and pins the CID here). It is private (not a public benchmark) and refreshed
periodically so it stays contamination-resistant. The directory behind the CID
holds one or more array files:

* ``*.npy``            — a single ``(L,)`` or ``(C, L)`` series each.
* ``*.npz``            — many arrays under arbitrary keys, each a series.
* ``metadata.json``    — optional ``{series_id: {freq / seasonal_period: ...}}``
  used to drive MASE seasonality (matched to a window by its source filename/key).

Every series contributes one window (last ``horizon`` steps = target, up to
``context_length`` before = history) via the pure cutter
:func:`metronome.validator.windows.build_windows_from_series`, so the resulting
pool is byte-identical for every validator that fetches the same CID.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..shared.config import ChainConfig
from ..shared.hippius import RegistryConfig, fetch_from_registry, is_cid
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
    registry: RegistryConfig | None = None,
) -> RotatingWindowSource:
    """Fetch the private pool CID from the Hippius registry and build the window
    source. Raises :class:`PoolError` on a missing/empty/malformed pool.

    Window geometry comes from ``[eval]`` (``context_length`` / ``horizon``),
    which the config pins equal to ``[training]`` so trained models fit the
    windows.
    """
    cid = cfg.eval.window_pool
    if not cid or not is_cid(cid):
        raise PoolError(
            f"[eval] window_pool must be a Hippius registry CID; got {cid!r}. "
            "Upload the held-out pool with upload_dir_to_registry and pin its CID."
        )
    reg = registry or RegistryConfig.from_storage(cfg.storage)
    dest = Path(cache_dir or "./_eval_pool") / cid
    try:
        fetch_from_registry(cid, dest, reg)
    except Exception as e:  # noqa: BLE001
        raise PoolError(f"pool_fetch_failed: {e}") from e

    series, ids = _load_series_dir(dest)
    if not series:
        raise PoolError(f"pool {cid} contained no .npy/.npz series under {dest}")

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
            f"pool {cid} had {len(series)} series but none were long enough for "
            f"horizon={cfg.eval.horizon}+context (need >= horizon+1 steps)"
        )
    log.info("loaded eval pool cid=%s series=%d windows=%d", cid, len(series), len(windows))
    return RotatingWindowSource(pool=tuple(windows))
