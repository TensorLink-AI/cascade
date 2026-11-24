"""F2 (plan rev6.18 A4, DEC-CA-0041): a multichannel lane D chain fixture,
shared by the W5d, W5e, W7a and W9 acceptance tests.

It is W7a's synthetic chain (`helpers.build_chain`: two pool_d snapshots,
quantile forecasts, no torch) with the series of chosen sources stored as
``(C, L)`` histories and ``(C, horizon)`` targets. The channel pick is
re-stated here from the plan, never imported from the producer:

* a channel whose history holds no finite value is dropped first;
* at most ``[generator] max_channels`` channels are kept (read from the
  repo's ``chain.toml``, 32 at launch); a wider series keeps the
  ``max_channels`` with the smallest ``sha256(seed + ":" + index)``, ``seed =
  sha256(boundary_block_hash + series_id + "lane_d_channels")``, ascending;
* one channel left is a univariate window.

The eval host's forecasts follow the pick: ``(horizon, 9)`` for a one-channel
window, ``(k, horizon, 9)`` for k channels. `settle_mv` asserts that
`settle_lane_d` takes `max_channels` before calling it, so on a main without
F2 every caller fails on an assertion. Nothing here imports a module in PATHS
at import time.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import tomllib
from pathlib import Path

import numpy as np

from . import helpers as w7a

CHANNEL_SEED_TAG = "lane_d_channels"
CHANNEL_FIELDS = ("series_id", "n_channels_raw", "empty_channels", "n_channels", "channels",
                  "channels_sha256")
BOUNDARY_OF = {1: w7a.BOUNDARY_1, 2: w7a.BOUNDARY_2}


def repo_max_channels() -> int:
    """``[generator] max_channels`` of the repo's chain.toml (never hardcoded)."""
    with open(w7a.REPO / "chain.toml", "rb") as f:
        return int(tomllib.load(f)["generator"]["max_channels"])


def spec_pick(block_hash, series_id: str, candidates, max_channels: int) -> list[int]:
    cand = sorted({int(c) for c in candidates})
    if len(cand) <= int(max_channels):
        return cand
    if block_hash is None:
        return []
    seed = hashlib.sha256(f"{block_hash}{series_id}{CHANNEL_SEED_TAG}".encode()).hexdigest()
    ranked = sorted(cand, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    return sorted(ranked[:int(max_channels)])


def channels_sha256(channels) -> str:
    return hashlib.sha256(json.dumps([int(c) for c in channels],
                                     separators=(",", ":")).encode()).hexdigest()


def _channel_rows(base: np.ndarray, n: int) -> np.ndarray:
    """``n`` channels of one series at distinct scales and phases."""
    t = np.arange(base.size, dtype=np.float64)
    return np.stack([base * (1.0 + 0.15 * c) + 3.0 * c + np.cos(2 * np.pi * t / 12 + c)
                     for c in range(n)])


def build_mv_chain(tmp: Path, *, channels: dict[str, int], empty: dict | None = None,
                   zero_target: dict | None = None, one_row: bool = False,
                   max_channels: int | None = None) -> dict:
    """W7a's chain with every series of ``channels``' sources stored ``(C, L)``.

    ``empty`` ({source: channel indices}) makes those channels all-NaN;
    ``zero_target`` ({source: channel indices}) makes those channels' targets
    all zero. ``one_row``: instead, EVERY series is stored ``(1, L)`` /
    ``(1, horizon)`` with W7a's own values (a univariate pool in 2-D form).
    The forecasts follow each window's spec pick at the snapshot's boundary.
    """
    empty, zero_target = dict(empty or {}), dict(zero_target or {})
    cap = repo_max_channels() if max_channels is None else int(max_channels)
    digests = [w7a.text_digest("w7a entry 0")]
    house = w7a.text_digest("w7a house")
    loser = w7a.text_digest("w7a loser")
    rng = np.random.default_rng(7)
    pools, forecasts, picks = {}, {}, {}
    for snap in (1, 2):
        bh = w7a.gate_block_hash(BOUNDARY_OF[snap])
        series = w7a._series(snap, 100 + snap, rich=True)
        targets, fc = {}, {d: {} for d in [*digests, house, loser]}
        picks[snap] = {}
        for s in series:
            sd = float(np.std(s["history"]))
            target = np.resize(s["history"][-24:], w7a.HORIZON) + rng.normal(0, 0.3, w7a.HORIZON)
            wid = w7a.window_id(s["series_id"], s["target_ts"])
            noise = rng.normal(0, 0.05 * sd, w7a.HORIZON)
            n = 1 if one_row else int(channels.get(s["source"], 1))
            if one_row or n == 1:
                hist, tgt = s["history"], target
                fc_good = w7a.quantiles_around(target + noise, 0.3 * sd)
                fc_house = w7a.quantiles_around(target + 3.0 * sd, 0.3 * sd)
                fc_loser = w7a.quantiles_around(target + 6.0 * sd, 0.3 * sd)
                if one_row:
                    s["history"], targets[s["series_id"]] = hist[None, :], tgt[None, :]
                else:
                    targets[s["series_id"]] = tgt
                for d in digests:
                    fc[d][wid] = fc_good
                fc[house][wid], fc[loser][wid] = fc_house, fc_loser
                continue
            hist = _channel_rows(s["history"], n)
            tgt = _channel_rows(target, n)
            for c in zero_target.get(s["source"], ()):
                tgt[c] = 0.0
            for c in empty.get(s["source"], ()):
                hist[c] = np.nan
                tgt[c] = np.nan
            s["history"], targets[s["series_id"]] = hist, tgt
            keep = [c for c in range(n) if np.isfinite(hist[c]).any()]
            pick = spec_pick(bh, s["series_id"], keep, cap)
            picks[snap][wid] = pick

            def q(shift: float, pick=pick, tgt=tgt, hist=hist, noise=noise) -> np.ndarray:
                out = np.stack([w7a.quantiles_around(
                    tgt[c] + shift * float(np.std(hist[c])) + 0.01 * noise,
                    0.3 * float(np.std(hist[c]))) for c in pick])
                return out[0] if len(pick) == 1 else out

            for d in digests:
                fc[d][wid] = q(0.0)
            fc[house][wid], fc[loser][wid] = q(3.0), q(6.0)
        pools[snap] = w7a._write_pool(tmp / f"pool_d_{snap}", snap, series, targets, rich=True)
        forecasts[snap] = fc
    return {"pools": pools, "forecasts": forecasts, "picks": picks, "digests": digests,
            "loser": loser, "house": {"digest": house, "reveal_ts": w7a.HOUSE_T},
            "max_channels": cap}


def settle_mv(chain: dict, state_path: Path, *, boundary: int, snap: int, entries: list[dict],
              uid_of: dict, cfg=None, max_channels="repo", block_hash="gate"):
    """`settle_lane_d` with the W5e gate inputs and ``max_channels`` (the
    repo chain.toml's unless given; ``None`` = the legacy call).
    ``block_hash=None`` drops the boundary hash (and so the gate)."""
    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    assert "max_channels" in params, (
        "F2 (rev6.18 A4): settle_lane_d(..., max_channels=) takes [generator] max_channels, "
        "the channel cap of a multichannel window")
    kw = w7a.gate_inputs(boundary, chain["house"]["digest"])
    if block_hash is None:
        kw = {k: v for k, v in kw.items() if k != "boundary_block_hash"}
    return settle_lane_d(cfg=cfg or w7a.lane_d_cfg(), boundary=boundary,
                         pool_dir=chain["pools"][snap], entries=[dict(e) for e in entries],
                         house=dict(chain["house"]), forecasts=chain["forecasts"][snap],
                         uid_of=dict(uid_of), state_path=state_path,
                         signer=w7a.fixture_key("signer"),
                         max_channels=(chain["max_channels"] if max_channels == "repo"
                                       else max_channels), **kw)


def crowned_mv(tmp: Path, **build) -> dict:
    """One challenger settled at BOUNDARY_1 on snapshot 1 of a multichannel
    chain (it clears and is crowned through the gate, as on W7a's chain)."""
    chain = build_mv_chain(tmp / "chain", **build)
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    uid_of = {e["hotkey"]: w7a.CHALLENGER_UID}
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    s1 = settle_mv(chain, state, boundary=w7a.BOUNDARY_1, snap=1, entries=[e], uid_of=uid_of)
    return {"chain": chain, "entry": e, "uid_of": uid_of, "state": state, "s1": s1}


def windows_of(report: dict) -> dict:
    return {w["window_id"]: w for w in report["windows"]}


def mv_window_ids(chain: dict, snap: int = 1) -> set[str]:
    return set(chain["picks"][snap])


def per_channel_score(q, target, history, seasonal_period: int) -> tuple[float, float]:
    """The DEC-CA-0041 per-window score from the UNIVARIATE scorer: each
    channel scored alone, CRPS averaged over channels with scale, MASE over
    all channels."""
    from cascade.eval.lane_d_scorer import score_window

    q, y, h = np.asarray(q), np.asarray(target), np.asarray(history)
    if q.ndim == 2:
        return score_window(q, y, h, seasonal_period)
    crps, mase = [], []
    for c in range(q.shape[0]):
        if float(np.abs(y[c]).sum()) > 0.0:
            crps.append(score_window(q[c], y[c], h[c], seasonal_period)[0])
        mase.append(_mase(q[c][:, 4], y[c], h[c], seasonal_period))
    return float(np.mean(crps)), float(np.mean(mase))


def _mase(point, y, h, sp: int) -> float:
    from cascade.eval.mase import mase

    return float(mase(np.asarray(point), np.asarray(y), np.asarray(h), max(1, int(sp))))
