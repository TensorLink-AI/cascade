"""Shared spec and fixture builders for the W5c cohort tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): §1.3 "Cohorts (several entries
in one settlement)":

1. The cohort is every entry that freezes this settlement.
2. The window set is the intersection eligible for max(T over cohort ∪ king).
3. If the intersection misses the floors, the latest-T entries are deferred to
   the next settlement until it meets them.
4. The cohort is then judged with max-T on that shared set.

plus T9 ("Three entries freezing in one settlement with different T:
intersection set; latest-T deferred if below floors; max-T crowns the best")
and the accepted Proposed design change 3 ("a miner challenger must also
clear `d_margin` against the current house on the same set; the house ranks
first on ties"). README.md states the contract.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

HOUR = 3600
DAY = 86_400
HORIZON = 64                     # §1.1: lane D scores h = 64 only
HISTORY = 256
MARGIN_H = 6.0                   # forward_margin_hours (recommended)
T0 = 1_790_000_000               # the earliest challenger's reveal time (unix s)
HOUSE_T = T0 - 2 * DAY           # the genesis house: earlier than every entry

# Three challengers with different T, one day apart (T9).
T_A, T_B, T_C = T0, T0 + DAY, T0 + 2 * DAY
T_LATE = T0 + 5 * DAY            # later than every window of the cohort pool


def bound(t: int) -> int:
    """§1.3 forward bound max(T_E, T_K) + forward_margin, for T >= the king's."""
    return int(t + MARGIN_H * HOUR)


# Era length 3600 blocks (900 x 4); ARM_BLOCK is an era boundary.
ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
BOUNDARY_2 = ARM_BLOCK + 2 * SETTLEMENT
BOUNDARY_3 = ARM_BLOCK + 3 * SETTLEMENT

SOURCES = ("src0", "src1", "src2", "src3")
FEED_LAG_HOURS = 1.0
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied.
    Floors are lowered so small fixtures can meet them (24 windows, 4
    sources); `max_source_share` keeps 0.25 (every fixture group splits evenly
    over the 4 sources, so the cap never drops a window)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def window_id(series_id: str, target_ts) -> str:
    """§1.3 window identity `<series_id>:<target_start_ts>:h64`."""
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def record(series_id: str, source: str, start: int) -> dict:
    """One pool_d/1 series record with hourly targets from `start`. Both clocks
    follow `start`; its frontier holds one earlier snapshot taken 1 h before
    `start` with data up to 2 h before it. So for every bound below
    `start - 1 h` the window passes every §1.3 rule (S0 present, no
    backfill), and for every bound at or after `start` it is `before_margin`."""
    ts = [int(start + i * HOUR) for i in range(HORIZON)]
    return {"series_id": series_id, "source": source, "feed": f"feed-{source}",
            "freq": "H", "seasonal_period": 24, "target_ts": ts,
            "first_seen_ts": [t + 1800 for t in ts], "first_target_ts": ts[0],
            "end_ts": ts[-1], "frontier": [[start - HOUR, start - 2 * HOUR]]}


def group(tag: str, per_source: int, start: int) -> list[dict]:
    """`per_source` windows for each of the 4 sources, targets from `start`,
    one minute apart (all inside the same §1.3 band)."""
    out = []
    for k in range(per_source):
        for j, src in enumerate(SOURCES):
            i = k * len(SOURCES) + j
            out.append(record(f"{tag}-{src}-{k}", src, start + i * 60))
    return out


def band_start(t: int) -> int:
    """A target start 3 h after T's bound: eligible for every T' <= t only
    when the next band starts at least a day later."""
    return bound(t) + 3 * HOUR


def wids(records) -> list[str]:
    return sorted(window_id(r["series_id"], r["target_ts"]) for r in records)


def target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def write_pool(directory: Path, records: list[dict], snapshot: str) -> Path:
    """Write one pool_d/1 snapshot: every series names a feed listed in the
    top-level `feeds` block (measured lag under the limit, not a forecast
    product) and carries a `frontier` (W5a fails closed without them)."""
    directory.mkdir(parents=True, exist_ok=True)
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False}
             for r in records}
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "feeds": dict(sorted(feeds.items())), "series": records}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in records:
        h, y = target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


# Forecast qualities: (median noise in sd units, quantile spread in sd units,
# median offset in sd units). "sharp" beats "good" beats "fair" beats "bad".
QUALITY = {"sharp": (0.005, 0.1, 0.0), "good": (0.02, 0.2, 0.0),
           "fair": (0.05, 0.3, 0.0), "bad": (0.0, 0.3, 3.0)}


def model_forecasts(records, quality: str, name: str, *, skip=()) -> dict:
    """`{window_id: (64, 9) quantiles}` for one model of the given quality,
    seeded by `name`; windows in `skip` have no forecast (absent, not
    malformed: the eval host returned nothing for them)."""
    noise, spread, offset = QUALITY[quality]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = {}
    gone = set(skip)
    for r in records:
        _, y = target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        wid = window_id(r["series_id"], r["target_ts"])
        med = y + offset * sd + rng.normal(0, noise * sd, HORIZON)
        if wid not in gone:
            out[wid] = _quantiles(med, spread * sd)
    return out


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def set_sha(window_ids) -> str:
    """The repo's window-set digest (`cohorts[].set_sha256` convention)."""
    return hashlib.sha256("\n".join(sorted(window_ids)).encode()).hexdigest()


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w5c-{name}")


def entry(name: str, *, reveal_ts: int, ready_block: int = ARM_BLOCK + 40,
          reveal_block: int = ARM_BLOCK + 60) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5c {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": int(ready_block), "reveal_block": int(reveal_block),
            "reveal_ts": int(reveal_ts)}


def house(name: str = "house", reveal_ts: int = HOUSE_T) -> dict:
    return {"digest": digest(f"w5c {name}"), "reveal_ts": int(reveal_ts)}


def uids(*entries) -> dict:
    return {e["hotkey"]: 10 + i for i, e in enumerate(entries)}


def settle(cfg, boundary, pool_dir, entries, house_rec, fc, uid_of, state_path, prior=()):
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), **kw)


def read_state(path) -> dict:
    return json.loads(Path(path).read_text())


def rows_of(report, dg) -> set[str]:
    return {r["window_id"] for r in report["rows"] if r["digest"] == dg}


def verdict_of(report, dg) -> dict | None:
    for v in report["verdicts"]:
        if v["digest"] == dg:
            return v
    return None


def skipped_reasons(report, dg) -> list[str]:
    return [s["reason"] for s in report["skipped"] if s["digest"] == dg]
