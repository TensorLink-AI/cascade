"""Multi-horizon calibration telemetry — LOG-ONLY, never touches the verdict.

Design: docs/notes/2026-08-31-multi-horizon-eval.md. The scored duel stays at
``[eval] horizon``; this module measures how the round's duel checkpoints
perform at the OTHER rungs of the planned 16/64/256/720 ladder so the scored
flip (a separate, coordinated consensus release) can be calibrated on real
per-(horizon, domain) candidate spread instead of guesses.

Everything here is deterministic in the round seed and pure enough to unit
test: the draw is an even-by-domain, eligibility-filtered, seeded permutation
per domain; the scoring reuses the exact production scorer
(:func:`cascade.eval.scoring` via ``evaluate_checkpoint``), so the telemetry
numbers are commensurable with the real verdict metric. Long horizons rely on
the checkpoint wrapper's own rollout — the same path the public bench sidecar
uses for GIFT-Eval medium/long terms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .windows import EvalWindow, horizon_draw

log = logging.getLogger("cascade.validator.calib")

# Minimum history a calibration window must keep in front of its target —
# matching the scored draw's implicit floor is not required (this is
# telemetry), but a forecast off near-zero context measures nothing.
MIN_CONTEXT = 64


def load_pool_series(pool_dir: Path) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """Raw ``(series, ids, metadata)`` from a pool snapshot directory —
    the same layout :mod:`cascade.validator.pool` loads (``*.npy``/``*.npz``
    + optional ``metadata.json``)."""
    from .pool import _load_series_dir

    series, ids = _load_series_dir(Path(pool_dir))
    md_map: dict = {}
    mf = Path(pool_dir) / "metadata.json"
    if mf.is_file():
        try:
            md_map = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — telemetry never raises for metadata
            log.warning("calib: unreadable pool metadata.json (%s); domains blank", e)
    return series, ids, [md_map.get(sid, {}) for sid in ids]


def calib_draw(
    series: list[np.ndarray],
    metadata: list[dict],
    *,
    horizon: int,
    n_windows: int,
    context_length: int,
    seed: int,
) -> list[EvalWindow]:
    """Even-by-domain, eligibility-filtered, seeded window draw for one rung.

    Thin wrapper over :func:`cascade.validator.windows.horizon_draw` — the
    draw is shared with the scored path so the two can never drift — keeping
    the calibration id namespace (``calib-h<h>-s<i>``) distinct from scored
    draws. Deterministic in ``seed`` — identical for every checkpoint scored
    this round, so comparisons stay paired.
    """
    return horizon_draw(
        series, metadata, horizon=int(horizon), n_windows=n_windows,
        context_length=context_length, seed=seed, min_context=MIN_CONTEXT,
        id_prefix=f"calib-h{int(horizon)}-",
    )


def _geo(scores) -> float | None:
    from ..eval.scoring import global_geomean

    return round(global_geomean(list(scores)), 6) if scores else None


def run_horizon_calibration(
    checkpoints: dict[str, Path],
    pool_dir: Path,
    *,
    horizons: tuple[int, ...],
    n_windows: int,
    num_samples: int,
    context_length: int,
    seed: int,
    device: str = "cpu",
    evaluate_fn=None,
) -> dict:
    """Score every checkpoint at every rung; return the telemetry document.

    ``checkpoints`` maps a label (``"king"``, ``"challenger-u91"``, …) to a
    local checkpoint directory. ``evaluate_fn(ckpt_dir, windows, num_samples,
    device)`` is injectable for tests; the default is the production
    :func:`cascade.validator.evaluator.evaluate_checkpoint`.
    """
    if evaluate_fn is None:
        from .evaluator import evaluate_checkpoint

        def evaluate_fn(d, w, ns, dev):  # noqa: ANN001
            return evaluate_checkpoint(d, w, num_samples=ns, device=dev)

    series, _ids, metadata = load_pool_series(pool_dir)
    out: dict = {"seed": seed, "n_windows": n_windows,
                 "num_samples": num_samples, "horizons": {}}
    for h in horizons:
        windows = calib_draw(series, metadata, horizon=int(h),
                             n_windows=n_windows, context_length=context_length,
                             seed=seed + int(h))
        if not windows:
            log.warning("calib h=%d: no eligible series; rung skipped", h)
            continue
        rung: dict = {"n_windows": len(windows), "entries": {}}
        king_scores = None
        for label, ckpt in checkpoints.items():
            try:
                scores = evaluate_fn(Path(ckpt), windows, int(num_samples), device)
            except Exception as e:  # noqa: BLE001 — one bad leg must not lose the rung
                log.warning("calib h=%d %s failed: %s", h, label, e)
                continue
            by_dom: dict[str, list] = {}
            for s in scores:
                by_dom.setdefault(s.domain or "?", []).append(s)
            entry = {"geomean": _geo(scores),
                     "domains": {d: _geo(v) for d, v in sorted(by_dom.items())}}
            if label == "king":
                king_scores = scores
            elif king_scores is not None and len(scores) == len(king_scores):
                # Paired per-window MASE deltas (challenger − king): the spread
                # statistic the admission mask + margin recalibration need.
                deltas = [c.mase - k.mase
                          for c, k in zip(scores, king_scores, strict=True)]
                entry["mase_delta_vs_king"] = {
                    "mean": round(float(np.mean(deltas)), 6),
                    "std": round(float(np.std(deltas)), 6),
                    "win_frac": round(float(np.mean([d < 0 for d in deltas])), 4),
                }
            rung["entries"][label] = entry
            log.info("calib h=%d %s: geomean=%s", h, label, entry["geomean"])
        out["horizons"][str(h)] = rung
    return out
