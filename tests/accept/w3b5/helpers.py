"""Shared spec and fixture builders for the W3b5 acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6) §1.3b:

  (rev3) Always-run low-frequency check (A10). Alongside the drawn suite,
  every crowning candidate also runs a paired non-inferiority check on a
  low-frequency slice of the private pool.
  - The slice: `[lane_d] lowfreq_freqs`, recommended daily and slower.
    Forward-only rarely reaches these series (§1.7).
  - The test: CRPS and MASE `LCB_α ≥ −tau_public` against both KING_D and
    the house, on the same window set, scored by the lane D scorer.
  - On failure: `lowfreq_regression` (typed), and the candidate isn't
    crowned.
  - Limit (§9): the private pool isn't forward-only, so a contaminated model
    passes this check more easily. It catches honest regressions only.

§1.7: "Daily-or-slower series almost never qualify, which is why A10's
low-frequency check uses the private pool instead."

§4 T4b: "A model that regresses only on low-frequency series | Blocked by
`lowfreq_regression`".

README.md states the contract. Nothing here imports a module in PATHS at
import time.

**The low-frequency forecasts are built so the expected LCB is exact.** On a
low-frequency window every quantile level sits *above* the target by
``offset_k * d`` (``d > 0`` one scalar per window), so the pinball loss is
linear in the offsets: CRPS ∝ Σ_k (1 − τ_k)·offset_k and MASE ∝ the median
offset. Two models built this way have the same CRPS ratio and the same MASE
ratio on every window, so any paired bootstrap of the relative improvement
(mean relative difference, geomean ratio, cluster or not) is degenerate at
``1 − ratio``. That is the LCB the tests expect (`expected_lcbs`), to 1e-6.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

HOUR = 3600
DAY = 86_400
HORIZON = 64
HISTORY = 256
MARGIN_H = 6.0
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY           # the genesis house: earlier than every entry
T_FIRST = T0                     # entries judged against the genesis house
T_LATER = T0 + DAY               # an entry judged against a miner KING_D

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA


def boundary(n: int) -> int:
    """The n-th settlement boundary after arming (n >= 1)."""
    return ARM_BLOCK + n * SETTLEMENT


SOURCES = ("src0", "src1", "src2", "src3")
LOWFREQ_SOURCES = ("lf0", "lf1")
FEED_LAG_HOURS = 1.0
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN = 4
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)
LOWFREQ_RESULTS = ("pass", "fail", "no_windows", "not_candidate")
REFS = ("king_d", "house")
METRICS = ("crps", "mase")


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied
    (floors lowered to 24 windows / 4 sources; the 0.25 source cap never
    drops a window because every hourly group splits evenly; `tau_public`
    and `lowfreq_freqs` keep their W2 defaults unless overridden)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


# ── pool_d fixtures ─────────────────────────────────────────────────────────


def bound(t: int) -> int:
    return int(t + MARGIN_H * HOUR)


def band_start(t: int) -> int:
    return bound(t) + 3 * HOUR


def window_id(series_id: str, target_ts) -> str:
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def hourly_record(series_id: str, source: str, start: int) -> dict:
    """One hourly pool_d/1 series, forward-eligible for every bound below
    `start - 1 h`."""
    ts = [int(start + i * HOUR) for i in range(HORIZON)]
    return {"series_id": series_id, "source": source, "feed": f"feed-{source}",
            "freq": "H", "seasonal_period": 24, "target_ts": ts,
            "first_seen_ts": [t + 1800 for t in ts], "first_target_ts": ts[0],
            "end_ts": ts[-1], "frontier": [[start - HOUR, start - 2 * HOUR]]}


def hourly_group(tag: str, start: int, per_source: int = 6) -> list[dict]:
    out = []
    for k in range(per_source):
        for j, src in enumerate(SOURCES):
            i = k * len(SOURCES) + j
            out.append(hourly_record(f"{tag}-{src}-{k}", src, start + i * 60))
    return out


_FREQ_STEP = {"D": DAY, "W": 7 * DAY, "M": 30 * DAY, "ME": 30 * DAY, "MS": 30 * DAY,
              "Q": 91 * DAY, "A": 365 * DAY, "Y": 365 * DAY}
_FREQ_PERIOD = {"D": 7, "W": 52, "M": 12, "ME": 12, "MS": 12, "Q": 4, "A": 1, "Y": 1}


def lowfreq_records(tag: str, n: int, freq: str = "D") -> list[dict]:
    """`n` low-frequency series of the private pool, alternating over
    `LOWFREQ_SOURCES`. Their targets are years old, so the §1.3 forward rule
    refuses every one of them (`before_margin`): they can never enter a
    frozen set or a duel, only the A10 slice (§1.7)."""
    step = _FREQ_STEP[freq]
    out = []
    for i in range(n):
        src = LOWFREQ_SOURCES[i % len(LOWFREQ_SOURCES)]
        start = T0 - 3 * 365 * DAY - HORIZON * step + i * 3 * DAY
        ts = [int(start + k * step) for k in range(HORIZON)]
        out.append({"series_id": f"{tag}-{src}-{i}", "source": src, "feed": f"feed-{src}",
                    "freq": freq, "seasonal_period": _FREQ_PERIOD[freq], "target_ts": ts,
                    "first_seen_ts": [t + 1800 for t in ts], "first_target_ts": ts[0],
                    "end_ts": ts[-1], "frontier": [[start - HOUR, start - 2 * HOUR]]})
    return out


def wids(records) -> list[str]:
    return sorted(window_id(r["series_id"], r["target_ts"]) for r in records)


def target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def write_pool(directory: Path, records: list[dict], snapshot: str) -> Path:
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


def snapshot(tmp_path: Path, n: int, t: int, lowfreq: list[dict] = ()) -> tuple[Path, list[dict]]:
    """Snapshot `n`: 24 fresh hourly windows over 4 sources, eligible for
    every T <= t, a day after snapshot n - 1's band, plus the given
    low-frequency series. Returns (pool dir, hourly records)."""
    recs = hourly_group(f"s{n}", band_start(t) + (n - 1) * DAY)
    return write_pool(tmp_path / f"pool{n}", recs + list(lowfreq), f"w3b5-pool{n}"), recs


# ── forecasts ───────────────────────────────────────────────────────────────

# Hourly (duel) quality: (noise sd, spread, median offset), in target sd units.
QUALITY = {"sharp": (0.005, 0.1, 0.0), "good": (0.02, 0.2, 0.0),
           "fair": (0.05, 0.3, 0.0), "bad": (0.0, 0.3, 3.0)}


def hourly_forecasts(records, quality: str, name: str) -> dict:
    noise, spread, offset = QUALITY[quality]
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    out = {}
    for r in records:
        _, y = target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        med = y + offset * sd + rng.normal(0, noise * sd, HORIZON)
        out[window_id(r["series_id"], r["target_ts"])] = np.stack(
            [med + spread * sd * z for z in _Z], axis=-1)
    return out


def flat(c: float) -> tuple[float, float, float]:
    """Every quantile level `c * d` above the target."""
    return (float(c), float(c), float(c))


def lowfreq_forecasts(records, spec: tuple[float, float, float], *, jitter: float = 0.0,
                      name: str = "") -> dict:
    """Low-frequency forecasts: levels below the median at `lo * d`, the
    median at `mid * d`, levels above at `hi * d` over the target
    (`spec = (lo, mid, hi)`, all > 0). `jitter` multiplies the window's
    offsets by `exp(N(0, jitter))`, seeded by `name` and the window, so the
    ratio to another model varies per window (a non-degenerate bootstrap)."""
    lo, mid, hi = spec
    assert min(spec) > 0, "every level must sit above the target (linear pinball)"
    out = {}
    for r in records:
        wid = window_id(r["series_id"], r["target_ts"])
        _, y = target_values(r["series_id"], r["target_ts"][0])
        d = 0.5 * float(np.std(y)) + 0.1
        if jitter:
            seed = int(hashlib.sha256(f"{name}|{wid}".encode()).hexdigest()[:8], 16)
            d *= float(np.exp(np.random.default_rng(seed).normal(0.0, jitter)))
        offs = [lo] * MEDIAN + [mid] + [hi] * (len(SCORING_LEVELS) - MEDIAN - 1)
        out[wid] = np.stack([y + o * d for o in offs], axis=-1)
    return out


def _crps_weight(spec) -> float:
    lo, mid, hi = spec
    offs = [lo] * MEDIAN + [mid] + [hi] * (len(SCORING_LEVELS) - MEDIAN - 1)
    return sum((1.0 - t) * o for t, o in zip(SCORING_LEVELS, offs, strict=True))


def expected_lcbs(candidate_spec, ref_spec) -> dict:
    """The exact paired LCB of the candidate against one reference on the
    slice: `1 - ratio` per metric (see the module docstring)."""
    return {"crps": 1.0 - _crps_weight(candidate_spec) / _crps_weight(ref_spec),
            "mase": 1.0 - candidate_spec[1] / ref_spec[1]}


def merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for dg, fc in d.items():
            out.setdefault(dg, {}).update(fc)
    return out


# ── entries, keys, reads ────────────────────────────────────────────────────


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w3b5-{name}")


def entry(name: str, *, reveal_ts: int, after: int = 0) -> dict:
    """An entry readied and revealed just after settlement `after`."""
    base = ARM_BLOCK + after * SETTLEMENT
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w3b5 {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": base + 40, "reveal_block": base + 60, "reveal_ts": int(reveal_ts)}


def house(name: str = "house-0", reveal_ts: int = HOUSE_T) -> dict:
    return {"digest": digest(f"w3b5 {name}"), "reveal_ts": int(reveal_ts)}


def uids(*entries) -> dict:
    return {e["hotkey"]: 10 + i for i, e in enumerate(entries)}


def read_state(path) -> dict:
    return json.loads(Path(path).read_text())


def verdict_of(report, dg) -> dict | None:
    for v in report["verdicts"]:
        if v["digest"] == dg:
            return v
    return None


def entry_of(report, dg) -> dict:
    (e,) = [e for e in report["entries"] if e["digest"] == dg]
    return e


def settle(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path, prior=()):
    """The real `settle_lane_d` with main's keywords (no gate inputs: the
    A10 check runs whether or not the W3b gate is wired)."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(cfg=cfg, boundary=b, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=copy.deepcopy(fc), uid_of=dict(uid_of),
                         state_path=state_path, signer=fixture_key("signer"), **kw)


def check_lowfreq_block(block: dict, cfg) -> None:
    """The A10 record of a crowning candidate that was evaluated: `result`
    `pass`/`fail`, the config's `freqs` and `tau_public`, and `lcb` =
    `{king_d, house} x {crps, mase}` finite floats, with `fail` iff some LCB
    is below `-tau_public`."""
    assert block["result"] in ("pass", "fail"), \
        f"the A10 check was not evaluated for a crowning candidate: {block!r}"
    assert block["freqs"] == [str(f) for f in cfg.lowfreq_freqs]
    assert block["tau_public"] == float(cfg.tau_public)
    lcb = block.get("lcb")
    assert isinstance(lcb, dict) and set(lcb) == set(REFS), \
        f"lowfreq.lcb must hold one entry per reference {REFS}: {lcb!r}"
    for ref in REFS:
        assert isinstance(lcb[ref], dict) and set(lcb[ref]) == set(METRICS), lcb
        for m in METRICS:
            v = lcb[ref][m]
            assert isinstance(v, float) and np.isfinite(v), f"lcb[{ref}][{m}] = {v!r}"
    worst = min(lcb[r][m] for r in REFS for m in METRICS)
    want = "fail" if worst < -float(cfg.tau_public) else "pass"
    assert block["result"] == want, \
        f"result {block['result']!r} but the worst LCB is {worst} vs -tau {-cfg.tau_public}"
