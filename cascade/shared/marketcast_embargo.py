"""MarketCast embargo contamination check (DEC-CA-0073 guard (b)).

Every MarketCast horizon is judged on windows drawn from the embargo — the
``[marketcast] embargo_eval_days`` before a leg's dispatch, bars no training
leg reads (a recipe's ``data.end`` is >= ``embargo_days`` before its commit).
The trainer controls the real data; the one way a miner can still carry
embargoed prices into training is to embed them in the generator tree. This
module closes that hole the way lane B's LB1 closes it for benchmark series:
the tree is scanned (parsed, NEVER executed —
:func:`cascade.interface.contamination_scan.scan_repo`) for numeric
sequences matching the embargoed bars, exactly (float64 bytes) or after
z-normalisation (the WB11 tolerant z-grid, so a rescaled / shifted / rounded
copy still matches), on 32-step sub-windows.

The reference set is built from the signed eval snapshot the main validator
publishes (``pool_mkt``) over ``(dispatch − embargo_eval_days, min(snapshot
end, dispatch)]``: for every asset, the close price and its log return
(basis points) of the hourly bank and of each native bank of at least
:data:`SCAN_MIN_BAR_MINUTES` minutes. Only digests are kept (a set of 12-byte
prefixes), so the set stays tens of MB on the trainer box.

The receipt rides the trainer-signed ``marketcast`` record of the leg
(``embargo_scan``); validators judge an entry only with a passing one
(:func:`cascade.validator.marketcast.embargo_scan_problem`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config.marketcast import COARSE_BAR_MINUTES, EMBARGO_RULE, duration_minutes

# the finest bank scanned: 15-minute bars (5m / 1m would multiply the set
# 3x / 12x; a 32-step 15m sub-window already spans 8 h of embargoed path)
SCAN_MIN_BAR_MINUTES = 15
SCAN_SUITE = "mkt_embargo"
SCAN_ARTEFACT_VERSION = 1
_PREFIX = 12                       # bytes of each sha256 kept


class _DigestSet(Mapping):
    """``{hex digest: [suite, id]}`` as the WB11 matcher reads it, holding
    only 12-byte digest prefixes (every hit names the one reference set)."""

    def __init__(self) -> None:
        self._s: set[bytes] = set()

    def add(self, hexdigest: str) -> None:
        self._s.add(bytes.fromhex(hexdigest)[:_PREFIX])

    def get(self, key, default=None):
        try:
            hit = bytes.fromhex(str(key))[:_PREFIX] in self._s
        except ValueError:
            return default
        return [SCAN_SUITE, "embargo"] if hit else default

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return len(self._s)


def _banks(snap) -> list[tuple[str, int, int]]:
    """``(bank dir, t0, bar_minutes)`` of every scanned bank of ``snap``."""
    from ..pool.marketcast import native_dir

    out = []
    c = snap.meta.get("coarse")
    if c:
        out.append(("coarse", int(c["t0"]), int(c.get("bar_minutes", COARSE_BAR_MINUTES))))
    for iv, rec in sorted((snap.meta.get("native") or {}).items()):
        bm = int(rec.get("bar_minutes", duration_minutes(iv)))
        if bm >= SCAN_MIN_BAR_MINUTES:
            out.append((native_dir(iv), int(rec["t0"]), bm))
    return out


def embargo_series(snap, lo: float, hi: float) -> dict[str, np.ndarray]:
    """``{item: values}``: per asset and scanned bank, the closes of the bars
    lying wholly in ``(lo, hi]`` and their log returns (bp)."""
    out: dict[str, np.ndarray] = {}
    for cls, syms in sorted((snap.meta.get("assets") or {}).items()):
        for asset in syms:
            for bank, t0, bm in _banks(snap):
                close = snap.bars(cls, asset, bank=bank).get("close")
                if close is None:
                    continue
                x = np.asarray(close, dtype=np.float64)
                starts = float(t0) + 60.0 * bm * np.arange(x.size)
                keep = (starts >= float(lo)) & (starts + 60.0 * bm <= float(hi))
                if not keep.any():
                    continue
                px = x[keep]
                out[f"{cls}/{asset}/{bank}/close"] = px
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[f"{cls}/{asset}/{bank}/ret_bp"] = np.diff(np.log(px)) * 1e4
    return out


def embargo_artefact(series: Mapping[str, np.ndarray], *, window: int | None = None) -> dict:
    """A WB11-shaped artefact (``version``, ``window``, ``exact``,
    ``normalised``) over ``series`` that :func:`~cascade.interface.
    contamination_scan.scan_repo` matches a tree against. Windows with a
    missing value and low-information windows never enter (as in WB11)."""
    from ..eval.contamination import WINDOW, exact_hash, is_low_information, normalised_hash

    n = int(window or WINDOW)
    exact, norm = _DigestSet(), _DigestSet()
    for item in sorted(series):
        vals = [float(v) for v in np.asarray(series[item], dtype=np.float64)]
        for off in range(0, len(vals) - n + 1):
            w = vals[off:off + n]
            if not all(math.isfinite(v) for v in w) or is_low_information(w):
                continue
            exact.add(exact_hash(w))
            nh = normalised_hash(w)
            if nh is not None:
                norm.add(nh)
    return {"version": SCAN_ARTEFACT_VERSION, "window": n, "exact": exact, "normalised": norm,
            "suites": {SCAN_SUITE: {}}}


def scan_receipt(tree: Path | str | None, snap, mc, *, train_time: float,
                 min_hits=None) -> dict:
    """The ``embargo_scan`` receipt of one leg: ``result`` ``"no_generator"``
    (no tree to scan), ``"clean"``, ``"embedded"`` (+ ``window_id`` /
    ``kind``) or ``"budget_exceeded"`` (refused like embedded: a tree too big
    to scan never passes), with the snapshot digest and the scanned span."""
    lo = float(train_time) - int(mc.embargo_eval_days) * 86400.0
    hi = min(float(snap.end), float(train_time))
    rec = {"rule": EMBARGO_RULE, "snapshot": str(snap.digest), "lo": lo, "hi": hi}
    if tree is None:
        return {**rec, "result": "no_generator"}
    from ..interface import contamination_scan as cs

    art = embargo_artefact(embargo_series(snap, lo, hi))
    got = cs.scan_repo(Path(tree), art, min_hits=min_hits)
    out = {**rec, "result": str(got["result"])}
    if got.get("window_id"):
        out["window_id"] = str(got["window_id"])
    return out


__all__ = ["SCAN_MIN_BAR_MINUTES", "embargo_artefact", "embargo_series", "scan_receipt"]
