"""Shared spec and fixture builders for the W5b union + freeze tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): §1.3 "Evidence accumulates,
then freezes" (union over every daily pool_d snapshot since T; freeze at the
first settlement meeting `min_windows` / `min_clusters`; judged once on the
cohort's intersection set, which is a subset of the frozen set), §1.3 Cohorts,
the W5b row (rev5: `max_source_share`, deterministic drop), T18, T18e, §17.4
D5 / D29 / B14. README.md states the module contract.

Nothing here imports a module in PATHS at import time, so on main every test
fails inside its own body: at its first import of
`cascade.validator.lane_d_frozen` (a `ModuleNotFoundError` for a path in
PATHS) or at an `assert`.
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
T0 = 1_790_000_000               # the challenger's reveal time (unix s)
HOUSE_T = T0 - 2 * DAY           # the house's T: earlier, so max(T_E, T_K) = T0
MARGIN_H = 6.0                   # forward_margin_hours (recommended)
BOUND = T0 + int(MARGIN_H * HOUR)
FEED_LAG_HOURS = 1.0             # measured mirror lag, under max_mirror_lag_hours

# Era length 3600 blocks (900 x 4); ARM_BLOCK is an era boundary.
ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
BOUNDARY_2 = ARM_BLOCK + 2 * SETTLEMENT
# The last settlement of the first armed era (the settlement AT an era
# boundary belongs to the era before it) and the first one of the next era.
ERA_LAST = ARM_BLOCK + ERA
NEXT_ERA_FIRST = ARM_BLOCK + ERA + SETTLEMENT

CHALLENGER_UID = 6
SECOND_UID = 7

SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied.
    Floors are lowered so small fixtures can meet them (24 windows, 4
    sources); `max_source_share` keeps its recommended 0.25 unless set."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def window_id(series_id: str, target_ts) -> str:
    """§1.3 window identity `<series_id>:<target_start_ts>:h64`."""
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def hourly(start: int) -> list[int]:
    return [int(start + i * HOUR) for i in range(HORIZON)]


def record(series_id: str, source: str, start: int) -> dict:
    """One forward-eligible pool_d/1 series record for bound BOUND: both clocks
    after the bound, S0 (the first snapshot after the bound) holds the series
    with a frontier before the target, and its feed (named after its source)
    is listed in the pool's ``feeds`` block (see ``write_pool``)."""
    ts = hourly(start)
    return {"series_id": series_id, "source": source, "feed": source,
            "freq": "H", "seasonal_period": 24,
            "target_ts": ts, "first_seen_ts": [t + 1800 for t in ts],
            "first_target_ts": ts[0], "end_ts": ts[-1],
            "frontier": [[BOUND + 2 * HOUR, BOUND + HOUR]]}


def trap(kind: str, series_id: str, source: str) -> dict:
    """A record §1.3 refuses at bound BOUND (W5a's reasons)."""
    if kind == "before_margin":
        old = hourly(BOUND - 70 * HOUR)
        rec = record(series_id, source, BOUND + 3 * HOUR)
        rec.update(target_ts=old, first_seen_ts=[t + 1800 for t in old],
                   first_target_ts=old[0], end_ts=old[-1])
        return rec
    if kind == "publish_ahead":
        rec = record(series_id, source, BOUND + 3 * HOUR)
        rec["first_seen_ts"] = [BOUND - HOUR] * HORIZON
        return rec
    raise ValueError(kind)


def snapshot_records(tag: str, counts: dict[str, int], day: int) -> list[dict]:
    """`counts[source]` forward-eligible windows per source, targets starting
    `day` days after the bound (one hour apart)."""
    out, i = [], 0
    for src, n in counts.items():
        for k in range(n):
            out.append(record(f"{tag}-{src}-{k}", src, BOUND + 3 * HOUR + day * DAY + i * HOUR))
            i += 1
    return out


def wids(records) -> set[str]:
    return {window_id(r["series_id"], r["target_ts"]) for r in records}


def target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def feeds_block(records: list[dict]) -> dict:
    """The top-level ``feeds`` block: every feed the records name (``feed``,
    else ``source``), with a measured lag under the 6 h limit and not a
    forecast product. W5a fails closed, so an unlisted feed would be refused
    ``unlisted_feed``; listing every feed keeps eligibility on the clocks and
    the frontier alone."""
    names = sorted({r.get("feed") or r["source"] for r in records})
    return {f: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False, "embargo_hours": 0.0}
            for f in names}


def write_pool(directory: Path, records: list[dict], snapshot: str) -> Path:
    """Write one pool_d/1 snapshot (W1.5 README format + the W5a additions:
    a ``feeds`` block listing every feed, and a ``frontier`` per series)."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "feeds": feeds_block(records), "series": records}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in records:
        h, y = target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def quantiles_around(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def forecasts(records, challengers, house: str) -> dict:
    """Challengers close to the truth, the house 3 sd off: a scored challenger
    clears, so a judged entry is visible in state as a held first clear
    (`hold` / `bench_hold`: rev6.11 (b) crowns at the first verdict, and the
    legacy call has no gate inputs, so W5e holds it)."""
    fc = {d: {} for d in [*challengers, house]}
    rng = np.random.default_rng(7)
    for r in records:
        _, y = target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        wid = window_id(r["series_id"], r["target_ts"])
        for d in challengers:
            fc[d][wid] = quantiles_around(y + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
        fc[house][wid] = quantiles_around(y + 3.0 * sd, 0.3 * sd)
    return fc


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
    return keypair_class().create_from_uri(f"//w5b-{name}")


def entry(name: str, *, reveal_ts: int = T0, ready_block: int = ARM_BLOCK + 40,
          reveal_block: int = ARM_BLOCK + 60) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5b {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": int(ready_block), "reveal_block": int(reveal_block),
            "reveal_ts": int(reveal_ts)}


def house(name: str = "house") -> dict:
    return {"digest": digest(f"w5b {name}"), "reveal_ts": HOUSE_T}


def settle(cfg, boundary, pool_dir, entries, house_rec, fc, uid_of, state_path, prior=()):
    """`settle_lane_d` on the newest snapshot `pool_dir`, with `prior` the
    earlier retained snapshots (README: `prior_pool_dirs`)."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {}
    if prior:
        import cascade.validator.lane_d_frozen  # noqa: F401

        kw["prior_pool_dirs"] = [Path(p) for p in prior]
    return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(boundary), cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), **kw)


def capped_counts(counts: dict[str, int], max_share: float) -> dict[str, int]:
    """Oracle for the minimum-drop cap (README "The cap"): every source keeps
    min(n_s, Q) windows, Q the largest integer <= max n_s with
    Q <= max_share * sum_s min(n_s, Q)."""
    top = max(counts.values())
    for q in range(top, -1, -1):
        if q <= max_share * sum(min(n, q) for n in counts.values()):
            return {s: min(n, q) for s, n in counts.items()}
    raise AssertionError("unreachable: Q = 0 always satisfies the cap")


def per_source(window_ids, source_of: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in window_ids:
        out[source_of[w]] = out.get(source_of[w], 0) + 1
    return out
