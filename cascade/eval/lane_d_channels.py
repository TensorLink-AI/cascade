"""Lane D multivariate windows: which channels a ``(C, L)`` window is scored
on (LANE_D_PLAN rev6.18 A4, F2; DEC-CA-0041).

A lane D window is either univariate (a 1-D history and target, today's
pool) or multichannel: a 2-D ``(C, L)`` history with a ``(C, horizon)``
target, forecast jointly by one model call. Before any model sees it, a
multichannel window is reduced to the channels it is scored on:

1. **empty channels drop** (:func:`empty_channels`): a channel whose history
   is entirely non-finite (the all-NaN columns RMISC's file merge leaves) is
   removed first;
2. **the cap** (:func:`pick_channels`): of the remaining channels, at most
   ``max_channels`` (``[generator] max_channels``, read from the config, 32
   at launch) are kept. A wider series keeps the ``max_channels`` channels
   with the smallest ``sha256(seed ‖ ":" ‖ index)``, where ``seed =
   sha256(boundary_block_hash ‖ series_id ‖ "lane_d_channels")``, listed
   ascending: the same for every model, re-derivable by anyone holding the
   block hash. Without a block hash an over-cap series is not scorable
   (fail closed: no pick);
3. **one channel left** is univariate: the window is scored exactly as a
   1-D window (same forecast batch, same ``score_window`` path), so a
   univariate result is bit-identical whatever the storage shape.

:func:`channel_plan` computes the pick of every multichannel window id from
pool windows in pin order (oldest first; the first version of an id decides,
as the box's eval-host request does), and :func:`apply_pick` slices a window
version to its pick. The signed report records each multichannel window's
:func:`channel_record` (W7a) and the audit re-derives it (W9).

Pure numpy + hashlib: no torch, no I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

CHANNEL_SEED_TAG = "lane_d_channels"


@dataclass(frozen=True)
class ChannelPick:
    """One multichannel window's channel selection. ``channels`` empty means
    the window cannot be scored (no non-empty channel, or an over-cap series
    without a block hash)."""

    series_id: str
    n_raw: int
    empty: tuple[int, ...]
    channels: tuple[int, ...]

    @property
    def scorable(self) -> bool:
        return bool(self.channels)

    @property
    def n_channels(self) -> int:
        """The scored channel count: 1 scores as a univariate window."""
        return len(self.channels)


def is_multichannel(history) -> bool:
    """A window is multichannel iff its stored history is 2-D ``(C, L)`` with
    C ≥ 2 (a ``(1, L)`` history is univariate)."""
    return np.ndim(history) == 2 and int(np.shape(history)[0]) >= 2


def empty_channels(history) -> tuple[int, ...]:
    """The channels of a ``(C, L)`` history that hold no finite value."""
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"a multichannel history is (C, L); got {h.shape}")
    return tuple(int(c) for c in np.flatnonzero(~np.isfinite(h).any(axis=1)))


def channel_seed(block_hash: str, series_id: str) -> str:
    """``sha256(block_hash ‖ series_id ‖ "lane_d_channels")`` hex."""
    return hashlib.sha256(f"{block_hash}{series_id}{CHANNEL_SEED_TAG}".encode()).hexdigest()


def pick_channels(block_hash: str | None, series_id: str, candidates: Iterable[int],
                  max_channels: int) -> tuple[int, ...]:
    """At most ``max_channels`` of ``candidates``, ascending. All of them when
    they fit; else the ``max_channels`` with the smallest
    ``sha256(channel_seed ‖ ":" ‖ index)`` (ties impossible: indices are
    distinct). An over-cap set without a ``block_hash`` picks nothing."""
    cand = sorted({int(c) for c in candidates})
    cap = int(max_channels)
    if cap < 1:
        raise ValueError(f"max_channels must be >= 1; got {max_channels!r}")
    if len(cand) <= cap:
        return tuple(cand)
    if block_hash is None:
        return ()
    seed = channel_seed(str(block_hash), str(series_id))
    ranked = sorted(cand, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    return tuple(sorted(ranked[:cap]))


def window_pick(w: Mapping, block_hash: str | None, max_channels: int) -> ChannelPick | None:
    """The pick of one multichannel window (``None`` for a univariate one)."""
    if not is_multichannel(w["history"]):
        return None
    h = np.asarray(w["history"], dtype=np.float64)
    empty = empty_channels(h)
    keep = [c for c in range(h.shape[0]) if c not in set(empty)]
    sid = str(w.get("series_id") or w["window_id"])
    return ChannelPick(series_id=sid, n_raw=int(h.shape[0]), empty=empty,
                       channels=pick_channels(block_hash, sid, keep, max_channels))


def channel_plan(windows: Iterable[Mapping], block_hash: str | None,
                 max_channels: int) -> dict[str, ChannelPick]:
    """``{window_id: ChannelPick}`` for every multichannel window id in
    ``windows`` (pool windows in pin order, oldest first): the first version
    of an id decides its pick."""
    out: dict[str, ChannelPick] = {}
    seen: set[str] = set()
    for w in windows:
        wid = w["window_id"]
        if not wid or wid in seen:
            continue
        seen.add(wid)
        p = window_pick(w, block_hash, max_channels)
        if p is not None:
            out[wid] = p
    return out


def apply_pick(w: Mapping, pick: ChannelPick | None) -> dict | None:
    """``w`` reduced to ``pick``: univariate windows (``pick is None``) come
    back unchanged (the same dict); a pick of one channel gives 1-D history
    and target; more gives ``(k, L)`` / ``(k, horizon)``. ``None`` when the
    version cannot carry the pick (not 2-D, another channel count, an empty
    pick)."""
    if pick is None:
        if is_multichannel(w["history"]):
            return None
        if np.ndim(w["history"]) == 2 or np.ndim(w["target"]) == 2:
            # A (1, L) series stored 2-D is the univariate window.
            return {**w, "history": np.asarray(w["history"], dtype=np.float64).reshape(-1),
                    "target": np.asarray(w["target"], dtype=np.float64).reshape(-1)}
        return w
    h = np.asarray(w["history"], dtype=np.float64)
    t = np.asarray(w["target"], dtype=np.float64)
    if (not pick.scorable or h.ndim != 2 or t.ndim != 2
            or h.shape[0] != pick.n_raw or t.shape[0] != pick.n_raw):
        return None
    idx = list(pick.channels)
    if len(idx) == 1:
        return {**w, "history": h[idx[0]].copy(), "target": t[idx[0]].copy()}
    return {**w, "history": h[idx].copy(), "target": t[idx].copy()}


def channels_sha256(channels: Iterable[int]) -> str:
    """sha256 hex of the canonical JSON list of the scored channel indices."""
    return hashlib.sha256(json.dumps([int(c) for c in channels],
                                     separators=(",", ":")).encode()).hexdigest()


def channel_record(pick: ChannelPick) -> dict:
    """The W7a per-window fields of a multichannel window (absent on a
    univariate window, so univariate reports are unchanged)."""
    return {"series_id": pick.series_id, "n_channels_raw": int(pick.n_raw),
            "empty_channels": [int(c) for c in pick.empty],
            "n_channels": int(pick.n_channels),
            "channels": [int(c) for c in pick.channels],
            "channels_sha256": channels_sha256(pick.channels)}


def forecast_shape(pick: ChannelPick | None, horizon: int, n_levels: int) -> tuple[int, ...]:
    """The forecast a window is scored from: ``(horizon, n_levels)`` for a
    univariate (or one-channel) window, else ``(k, horizon, n_levels)``."""
    if pick is None or pick.n_channels == 1:
        return (int(horizon), int(n_levels))
    return (pick.n_channels, int(horizon), int(n_levels))


__all__ = ["CHANNEL_SEED_TAG", "ChannelPick", "apply_pick", "channel_plan", "channel_record",
           "channel_seed", "channels_sha256", "empty_channels", "forecast_shape",
           "is_multichannel", "pick_channels", "window_pick"]
