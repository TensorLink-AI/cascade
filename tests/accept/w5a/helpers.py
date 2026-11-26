"""Shared spec and fixture builders for the W5a forward-eligibility tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): §1.3 (the rev3 two-clock rule,
`max_mirror_lag_hours`, the refused v2 snapshot rule), Proposed design
changes 1 and 2 (ACCEPTED rev5: present in S0 with `end_ts`; every target ts
> end_ts(S0); series absent from S0 start their clock at first appearance;
publish-ahead feeds), T8 and §3.2 invariant 4. README.md states the module
contract.

Nothing here imports a module in PATHS, so on main every test fails inside
its own body: at its first import of `cascade.validator.lane_d_forward` (a
`ModuleNotFoundError` for a path in PATHS) or at an `assert`.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# The v2 snapshot-diff rule (W1's report-only measurement; never for verdicts).
REPO_SCRIPTS = REPO / "scripts" / "laned_w1_measure.py"

HOUR = 3600
HORIZON = 64                     # §1.1: lane D scores h = 64 only
HISTORY = 256
T0 = 1_790_000_000               # an entry's reveal time (unix s)
MARGIN_H = 6.0                   # forward_margin_hours (recommended)
LAG_H = 6.0                      # max_mirror_lag_hours (recommended)

# The precedence order of the typed refusal reasons (README "Reasons").
# RV3 BLOCKING 1: forward-only fails closed. A feed missing from the pool's
# `feeds` block (or a pool with no `feeds`) is `unlisted_feed`, checked right
# after the clocks are recorded (without the entry neither forecast_product
# nor the lag can be judged). A series whose `frontier` is absent or null has
# no evidence of being in S0: `absent_from_s0`, at rule 6's place.
REASONS = ("missing_clock", "unlisted_feed", "forecast_product", "over_lag",
           "before_margin", "publish_ahead", "late_mirror", "absent_from_s0", "backfill")

SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)

_ABSENT = object()                # "the series records no frontier key"


# rev6.18 F4 (A3): these fixtures pin the live-only rule: the public half off
# (public_weight = 0, a spec setting) and no per-window ratio clamp in effect
# (a floor far below any fixture ratio), so their pre-F4 numbers stand. F4's
# own tests (tests/accept/w5e/test_w5e_combined.py) pin the combined rule.
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": 2700 * 3600, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": LAG_H,
           "min_windows": 12, "min_clusters": 3, "bootstrap_B": 200}
    return parse_lane_d({**raw, **LIVE_ONLY, **over}, era_blocks_at=lambda _block: 3600)


def window_id(series_id: str, target_ts) -> str:
    """§1.3 window identity `<series_id>:<target_start_ts>:h64`."""
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def hourly(start: int) -> list[int]:
    return [int(start + i * HOUR) for i in range(HORIZON)]


def spec(series_id: str, source: str, target_ts, first_seen_ts, *, frontier=_ABSENT,
         feed: str | None = None, end_ts=_ABSENT, freq: str = "H") -> dict:
    """One pool_d/1 series record (README "pool_d/1 additions")."""
    rec = {"series_id": series_id, "source": source, "freq": freq, "seasonal_period": 24,
           "target_ts": [int(t) for t in target_ts],
           "first_seen_ts": [int(t) for t in first_seen_ts]}
    rec["first_target_ts"] = rec["target_ts"][0] if rec["target_ts"] else None
    rec["end_ts"] = (rec["target_ts"][-1] if rec["target_ts"] else None) \
        if end_ts is _ABSENT else end_ts
    if frontier is None:              # an explicit JSON null
        rec["frontier"] = None
    elif frontier is not _ABSENT:
        rec["frontier"] = [[int(a), int(e)] for a, e in frontier]
    if feed is not None:
        rec["feed"] = feed
    return rec


# ── one window per T8 case, built against a bound = wall(max T) + margin ─────
#
# Each builder makes exactly ONE rule fail (the case's reason) and keeps the
# timeline physically consistent: a frontier record's end_ts never exceeds its
# snapshot's as_of time.


def ok_frontier(bound: int) -> list[list[int]]:
    """S0 = the first snapshot after the bound; its frontier precedes the target."""
    return [[bound + 2 * HOUR, bound + 1 * HOUR]]


def case(kind: str, bound: int, series_id: str, source: str) -> dict:
    start = bound + 3 * HOUR
    ts = hourly(start)
    seen = [t + 1800 for t in ts]
    if kind == "eligible":
        return spec(series_id, source, ts, seen, frontier=ok_frontier(bound))
    if kind == "no_frontier":       # no frontier key: no evidence of S0 (RV3)
        return spec(series_id, source, ts, seen)
    if kind == "null_frontier":     # "frontier": null, the same as absent (RV3)
        return spec(series_id, source, ts, seen, frontier=None)
    if kind == "unlisted_feed":     # its feed has no entry in `feeds` (RV3)
        return spec(series_id, source, ts, seen, frontier=ok_frontier(bound),
                    feed="unlisted")
    if kind == "before_margin":     # both clocks before the bound
        old = hourly(bound - 70 * HOUR)
        return spec(series_id, source, old, [t + 1800 for t in old],
                    frontier=ok_frontier(bound))
    if kind == "late_mirror":       # values public before T reached the mirror late
        old = hourly(bound - 70 * HOUR)
        return spec(series_id, source, old, [bound + HOUR + i for i in range(HORIZON)],
                    frontier=[[bound + 2 * HOUR, bound - 80 * HOUR]])
    if kind == "publish_ahead":     # future-stamped values the mirror saw before T
        return spec(series_id, source, ts, [bound - HOUR] * HORIZON,
                    frontier=ok_frontier(bound))
    if kind == "over_lag":
        return spec(series_id, source, ts, seen, frontier=ok_frontier(bound), feed="laggy")
    if kind == "forecast_product":
        return spec(series_id, source, ts, seen, frontier=ok_frontier(bound),
                    feed="forecasts")
    if kind == "backfill":          # behind the frontier S0 already had
        return spec(series_id, source, ts, [ts[-1] + 2 * HOUR] * HORIZON,
                    frontier=[[ts[-1] + HOUR, ts[-1]]])
    if kind == "absent_from_s0":    # in no snapshot after the bound before this one
        return spec(series_id, source, ts, seen, frontier=[])
    raise ValueError(kind)


TRAP_KINDS = ("before_margin", "late_mirror", "publish_ahead", "over_lag",
              "forecast_product", "backfill", "absent_from_s0", "unlisted_feed")
# Series that fail closed on missing S0 evidence (RV3): refused `absent_from_s0`.
FAIL_CLOSED_KINDS = ("no_frontier", "null_frontier")
# The traps today's two-clock `forward_eligible` (W1.5) already refuses.
CLOCK_TRAPS = ("before_margin", "late_mirror", "publish_ahead")


def feeds_block(sources, embargo_hours: float = 0.0) -> dict:
    """Measured lags: every source within the limit, `laggy` over it,
    `forecasts` a forecast product. `unlisted` is never listed.

    rev6.10/rev6.11: every listed feed pins its effective `embargo_hours`
    (W6b). The default 0 is the rev3 rule, so every fixture built before the
    embargo keeps its verdicts byte for byte (README "Embargo")."""
    out = {s: {"mirror_lag_hours": 1.0, "forecast_product": False} for s in sources}
    out["laggy"] = {"mirror_lag_hours": LAG_H + 1.0, "forecast_product": False}
    out["forecasts"] = {"mirror_lag_hours": 1.0, "forecast_product": True}
    for f in out.values():
        f["embargo_hours"] = float(embargo_hours)
    return out


# ── rev6.10 (a) / rev6.11 (e): the embargo-aware rule ───────────────────────

EMBARGO_H = 48.0                  # [lane_d] embargo_hours default (decided)

# rev6.12 (a): the embargo applies to EVERY cadence (no sub-hourly zero); the
# fixtures sweep these pool `freq` strings to show `freq` plays no part.
CADENCE_FREQS = ("10s", "30S", "min", "5min", "15min", "30min", "H", "8H", "D", "W")


def with_embargo(cfg, hours: float):
    """`cfg` with `embargo_hours` set (W2e adds the real key; until then a
    read-only view). The test never reads the attribute itself."""
    import dataclasses

    if any(f.name == "embargo_hours" for f in dataclasses.fields(cfg)):
        return dataclasses.replace(cfg, embargo_hours=float(hours))

    class _View:
        def __init__(self, base, e):
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "embargo_hours", float(e))

        def __getattr__(self, name):
            return getattr(self._base, name)

    return _View(cfg, hours)


def oracle_embargo(rec: dict, feeds: dict | None, cfg_embargo: float):
    """README "Embargo": the effective e of a series (every `freq` alike,
    rev6.12), or None when the feed is listed but its `embargo_hours` is
    missing or not a non-negative number (fail closed, `unlisted_feed`).
    Re-derived here from the fixture record, never from the producer."""
    feed = (feeds or {}).get(rec.get("feed") or rec["source"])
    if not isinstance(feed, dict):
        return 0.0                             # unlisted: refused before e matters
    v = feed.get("embargo_hours")
    if isinstance(v, bool) or not isinstance(v, int | float) or not v >= 0:
        return None
    return min(float(v), float(cfg_embargo))


def oracle_fresh(rec: dict, feeds: dict | None, b: float, cfg) -> bool:
    """rev6.12 terms: a fresh window passes the e = 0 rule on every point."""
    return oracle_reason(rec, feeds, b, cfg) is None


def oracle_embargo_reason(rec: dict, feeds: dict | None, b: float, cfg,
                          cfg_embargo: float) -> str | None:
    """The rev6.10 rule: the rev3 oracle at the feed bound `b - e`, with an
    invalid feed `embargo_hours` refused as `unlisted_feed`."""
    e = oracle_embargo(rec, feeds, cfg_embargo)
    if e is None:
        why = oracle_reason(rec, feeds, b, cfg)
        return why if why == "missing_clock" else "unlisted_feed"
    return oracle_reason(rec, feeds, b - HOUR * e, cfg)


def target_values(series_id: str) -> tuple[np.ndarray, np.ndarray]:
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(series_id))
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def write_pool(directory: Path, series: list[dict], *, feeds: dict | None = None,
               snapshot: str = "w5a") -> Path:
    """Write one pool_d/1 snapshot (W1.5 README format + the W5a additions)."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "series": series}
    if feeds is not None:
        meta["feeds"] = feeds
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for s in series:
        h, y = target_values(s["series_id"])
        hist[s["series_id"]], tgt[s["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def quantiles_around(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w5a-{name}")


def oracle_reason(rec: dict, feeds: dict | None, bound: float, cfg) -> str | None:
    """The §1.3 rule re-derived here from the raw fixture record (independent
    of the producer), with the README precedence order."""
    ts, seen = rec["target_ts"], rec["first_seen_ts"]
    if (len(ts) != HORIZON or len(seen) != HORIZON or rec.get("end_ts") is None
            or int(rec["end_ts"]) != ts[-1]):
        return "missing_clock"
    name = rec.get("feed") or rec["source"]
    if name not in (feeds or {}):       # unlisted feed, or no feeds block (RV3)
        return "unlisted_feed"
    feed = feeds[name]
    if feed.get("forecast_product"):
        return "forecast_product"
    lag = feed.get("mirror_lag_hours")
    if lag is not None and float(lag) > float(cfg.max_mirror_lag_hours):
        return "over_lag"
    data_ok, seen_ok = min(ts) > bound, min(seen) > bound
    if not data_ok and not seen_ok:
        return "before_margin"
    if not seen_ok:
        return "publish_ahead"
    if not data_ok:
        return "late_mirror"
    if rec.get("frontier") is None:     # absent or null: no S0 evidence (RV3)
        return "absent_from_s0"
    after = [e for a, e in sorted(rec["frontier"]) if a > bound]
    if not after:
        return "absent_from_s0"
    if min(ts) <= after[0]:
        return "backfill"
    return None


def bound_of(t_entry: int, t_king: int, cfg) -> float:
    return float(max(t_entry, t_king)) + HOUR * float(cfg.forward_margin_hours)


# ── gate inputs (W5e fail-closed, RV5) ──────────────────────────────────────
# Once W5e wires the §1.3b gate into `settle_lane_d`, an ARMED settlement that
# would crown without a bench session and a boundary block hash does not crown
# (`bench_hold`). The crowning fixtures therefore pass a healthy bench and the
# boundary's block hash WHEN `settle_lane_d` accepts them (`gate_inputs`); on
# a main without the keywords they make the legacy call, unchanged. The bench
# shape follows `tests/accept/w5e/helpers.py` (`passing_bench`,
# `settle_crowning`): recorded per-item CRPS/MASE behind a real
# `lane_d_bench.SweepSession`.

GATE_KEYWORDS = ("boundary_block_hash", "sentinel", "bench")
BENCH_ITEMS = 40


def bench_rows(suite: str, factor: float = 1.0) -> list[dict]:
    """Recorded per-item CRPS/MASE for one suite, times `factor` (above 1 =
    worse). Deterministic, with no noise: equal factors are an exact tie."""
    import hashlib

    out = []
    for i in range(BENCH_ITEMS):
        h = hashlib.sha256(f"w5a bench {suite} {i}".encode()).digest()
        out.append({"item": f"{suite}/task-{i:03d}", "crps": (0.3 + h[0] / 255.0) * factor,
                    "mase": (0.8 + h[1] / 255.0) * factor})
    return out


class HealthyBench:
    """A stand-in for the bench pod (an external service). Every digest but
    the current house returns the same rows (a tie, never a regression, so a
    miner KING_D never blocks) and the house is 10% worse on every item, so
    every crowning candidate passes the paired gate on any drawn suite and
    the duel alone decides each outcome."""

    def __init__(self, house_digest: str):
        self.house_digest = house_digest
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok",
                        "rows": bench_rows(s, 1.0 if d == self.house_digest else 0.9)}
                    for s in suites} for d in digests}


def gate_block_hash(boundary) -> str:
    """A fixture boundary block hash (0x + 64 lowercase hex)."""
    import hashlib

    return "0x" + hashlib.sha256(f"w5a block {int(boundary)}".encode()).hexdigest()


def gate_inputs(boundary, house_digest: str) -> dict:
    """The W5e gate keywords `settle_lane_d` accepts (none on today's main):
    the boundary's block hash, no §16 sentinel (W-S1 unarmed) and a healthy
    bench session."""
    import inspect

    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    if "bench" not in params and "boundary_block_hash" not in params:
        return {}
    kw: dict = {}
    if "boundary_block_hash" in params:
        kw["boundary_block_hash"] = gate_block_hash(boundary)
    if "sentinel" in params:
        kw["sentinel"] = None
    if "bench" in params:
        from cascade.validator.lane_d_bench import SweepSession

        kw["bench"] = SweepSession(f"w5a-sess-{int(boundary)}", HealthyBench(house_digest))
    return kw
