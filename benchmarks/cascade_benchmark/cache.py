"""Disk cache for checkpoint-independent baseline metrics.

The Seasonal-Naive baseline a suite normalizes against depends only on the eval
data, not on the checkpoint being scored — so it is computed once and reused
across every round and every checkpoint. GIFT-Eval and BOOM ship *vendored*
baselines; TIME has none, so its per-task Seasonal-Naive metrics are cached here
(pure stdlib, no gluonts/timebench, so it is unit-testable on its own).

Location: ``CASCADE_BENCH_TIME_BASELINE_CACHE`` if set, else
``~/.cache/cascade_benchmark/time_snaive``. ``CASCADE_BENCH_NO_CACHE`` disables
it. Every operation is best-effort: a cache miss/failure just means recompute,
never an error.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path


def baseline_cache_dir() -> Path | None:
    """The cache directory, created on first use, or ``None`` when caching is
    disabled or the directory can't be created (→ recompute every time)."""
    if os.environ.get("CASCADE_BENCH_NO_CACHE", "").strip().lower() in ("1", "true", "yes"):
        return None
    d = os.environ.get("CASCADE_BENCH_TIME_BASELINE_CACHE") or str(
        Path.home() / ".cache" / "cascade_benchmark" / "time_snaive"
    )
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        return Path(d)
    except Exception:  # noqa: BLE001 — cache is best-effort
        return None


def cache_key(cache_dir: Path, name: str, term: str, pred_len, n_q: int) -> Path:
    """Cache file for one task's baseline. Keyed by the fields that fix the TIME
    data + grid (dataset name, term, prediction length, quantile-grid size); the
    data for a released config is immutable, so this key is stable across rounds."""
    raw = f"{name}__{term}__pl{pred_len}__q{n_q}"
    return cache_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", raw) + ".json")


def load_baseline(cache_dir: Path | None, name, term, pred_len, n_q: int) -> dict | None:
    """Cached baseline metric dict for one task, or ``None`` on miss/unreadable."""
    if cache_dir is None:
        return None
    p = cache_key(cache_dir, name, term, pred_len, n_q)
    if not p.is_file():
        return None
    try:
        return {k: float(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except Exception:  # noqa: BLE001 — a corrupt cache entry just misses
        return None


def store_baseline(cache_dir: Path | None, name, term, pred_len, n_q: int, metrics: dict) -> None:
    """Persist a task's baseline metric dict (best-effort — never raises)."""
    if cache_dir is None:
        return
    # A cache write must never fail a task.
    with contextlib.suppress(Exception):
        cache_key(cache_dir, name, term, pred_len, n_q).write_text(
            json.dumps(metrics), encoding="utf-8"
        )
