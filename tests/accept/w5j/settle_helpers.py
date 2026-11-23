"""pool_d/1 fixtures and the settle call for the W5j settlement tests (the
same shapes as the W5b fixtures: hourly series whose both clocks follow the
bound, listed feeds, a frontier in S0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .helpers import ARM_BLOCK, HORIZON

HOUR = 3600
DAY = 86_400
HISTORY = 256
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY
MARGIN_H = 6.0
BOUND = T0 + int(MARGIN_H * HOUR)
SETTLEMENT = 900
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
CHALLENGER_UID = 6
SECOND_UID = 7

SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)


def hourly(start: int) -> list[int]:
    return [int(start + i * HOUR) for i in range(HORIZON)]


def window_id(series_id: str, target_ts) -> str:
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def record(series_id: str, source: str, start: int) -> dict:
    ts = hourly(start)
    return {"series_id": series_id, "source": source, "feed": source,
            "freq": "H", "seasonal_period": 24,
            "target_ts": ts, "first_seen_ts": [t + 1800 for t in ts],
            "first_target_ts": ts[0], "end_ts": ts[-1],
            "frontier": [[BOUND + 2 * HOUR, BOUND + HOUR]]}


def records(counts: dict[str, int]) -> list[dict]:
    out, i = [], 0
    for src, n in counts.items():
        for k in range(n):
            out.append(record(f"j-{src}-{k}", src, BOUND + 3 * HOUR + i * HOUR))
            i += 1
    return out


def target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def write_pool(directory: Path, recs: list[dict], snapshot: str = "w5j") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    feeds = {f: {"mirror_lag_hours": 1.0, "forecast_product": False}
             for f in sorted({r["feed"] for r in recs})}
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "feeds": feeds, "series": recs}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in recs:
        h, y = target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def forecasts(recs, challengers, house: str) -> dict:
    fc = {d: {} for d in [*challengers, house]}
    rng = np.random.default_rng(7)
    for r in recs:
        _, y = target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        wid = window_id(r["series_id"], r["target_ts"])
        for d in challengers:
            fc[d][wid] = _quantiles(y + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
        fc[house][wid] = _quantiles(y + 3.0 * sd, 0.3 * sd)
    return fc


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair.create_from_uri(f"//w5j-{name}")


def entry(name: str, *, reveal_ts: int = T0) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5j {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": ARM_BLOCK + 40, "reveal_block": ARM_BLOCK + 60,
            "reveal_ts": int(reveal_ts)}


def house() -> dict:
    return {"digest": digest("w5j house"), "reveal_ts": HOUSE_T}


def settle(cfg, boundary, pool_dir, entries, house_rec, fc, uid_of, state_path, block_hash):
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), boundary_block_hash=block_hash)


def draw_cells(recs: list[dict]) -> list[dict]:
    """The draw's input for these one-window series (every point eligible
    and passing at e = 0)."""
    members = [{"series_id": r["series_id"], "source": r["source"], "ts": list(r["target_ts"]),
                "eligible": [True] * HORIZON, "e0": [True] * HORIZON} for r in recs]
    return [{"timeframe": "H", "bin": "forward", "kind": "forward",
             "h_lo": HORIZON, "h_hi": HORIZON, "series": members}]
