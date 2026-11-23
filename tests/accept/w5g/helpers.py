"""Shared spec and fixture builders for the W5g acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6):

- §10.4 fault table: "Bench pod down before the candidate's sweep | Crowning
  held; CRIT after 1; after `max_bench_holds` (4) the candidate re-queues
  unjudged. Deliberately not lane A's never-timeout hold".
- §10.4: "Candidate sweep error, partial, NaN, missing metric |
  `checkpoint_fault`: entry refused; evidence kept".
- §1.3b: a candidate whose sweep errors, returns partial, NaN or a missing
  metric is a **`checkpoint_fault`, the miner's fault** (T17).
- §3.2 invariant 11: "`pending`, `hold` and `burn` always carry a typed
  reason enum, and tests assert the exact reason. (W5g, W7a)".
- W5g row: "Holds, `checkpoint_fault`, typed reason enums | W5e | ci |
  `scripts/laned/accept W5g` (asserts the exact reason enum per case)".

README.md states the contract. The fixtures follow W5e's (tests/accept/w5e):
real `settle_lane_d` calls over real `pool_d/1` snapshots; the bench pod is
the one external service (`BenchHost`, recorded per-item rows behind a real
`lane_d_bench.SweepSession`). Nothing here imports a module in PATHS at
import time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
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
T_X = T0                         # the challenger

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA


def boundary(n: int) -> int:
    """The n-th settlement boundary after arming (n >= 1)."""
    return ARM_BLOCK + n * SETTLEMENT


SOURCES = ("src0", "src1", "src2", "src3")
FEED_LAG_HOURS = 1.0
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)

GATE_SUITES = ("gift", "boom", "time", "fev")
BENCH_ITEMS = 40


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied
    (floors lowered to 24 windows / 4 sources)."""
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


def record(series_id: str, source: str, start: int) -> dict:
    """One pool_d/1 series record, eligible for every bound below `start - 1 h`."""
    ts = [int(start + i * HOUR) for i in range(HORIZON)]
    return {"series_id": series_id, "source": source, "feed": f"feed-{source}",
            "freq": "H", "seasonal_period": 24, "target_ts": ts,
            "first_seen_ts": [t + 1800 for t in ts], "first_target_ts": ts[0],
            "end_ts": ts[-1], "frontier": [[start - HOUR, start - 2 * HOUR]]}


def group(tag: str, per_source: int, start: int) -> list[dict]:
    out = []
    for k in range(per_source):
        for j, src in enumerate(SOURCES):
            i = k * len(SOURCES) + j
            out.append(record(f"{tag}-{src}-{k}", src, start + i * 60))
    return out


def target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def write_pool(directory: Path, records: list[dict], snapshot_name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False, "embargo_hours": 0.0}
             for r in records}
    meta = {"schema": "pool_d/1", "snapshot": snapshot_name, "horizon": HORIZON,
            "feeds": dict(sorted(feeds.items())), "series": records}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in records:
        h, y = target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def snapshot(tmp_path: Path, n: int, t: int = T_X) -> tuple[Path, list[dict]]:
    """Snapshot `n` (n >= 1): 24 fresh windows over 4 sources, eligible for
    every T <= t, in a band a day after snapshot n - 1's. Snapshots never
    share a window id."""
    recs = group(f"s{n}", 6, band_start(t) + (n - 1) * DAY)
    return write_pool(tmp_path / f"pool{n}", recs, f"w5g-pool{n}"), recs


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


QUALITY = {"sharp": (0.005, 0.1, 0.0), "good": (0.02, 0.2, 0.0),
           "fair": (0.05, 0.3, 0.0), "bad": (0.0, 0.3, 3.0)}


def model_forecasts(records, quality: str, name: str) -> dict:
    noise, spread, offset = QUALITY[quality]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = {}
    for r in records:
        _, y = target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        med = y + offset * sd + rng.normal(0, noise * sd, HORIZON)
        out[window_id(r["series_id"], r["target_ts"])] = _quantiles(med, spread * sd)
    return out


def forecasts(recs, models: dict) -> dict:
    """`{digest: forecasts}` from `{digest: (quality, name)}`."""
    return {dg: model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def block_hash(tag: str) -> str:
    """A fixture boundary block hash (0x + 64 hex)."""
    return "0x" + hashlib.sha256(f"w5g block {tag}".encode()).hexdigest()


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w5g-{name}")


def entry(name: str, *, reveal_ts: int, after: int = 0) -> dict:
    """An entry readied and revealed just after settlement `after`."""
    base = ARM_BLOCK + after * SETTLEMENT
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5g {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": base + 40, "reveal_block": base + 60, "reveal_ts": int(reveal_ts)}


def house(name: str = "house-0", reveal_ts: int = HOUSE_T) -> dict:
    return {"digest": digest(f"w5g {name}"), "reveal_ts": int(reveal_ts)}


def uids(*entries) -> dict:
    return {e["hotkey"]: 10 + i for i, e in enumerate(entries)}


def read_state(path) -> dict:
    return json.loads(Path(path).read_text())


def verdict_of(report, dg) -> dict | None:
    for v in report["verdicts"]:
        if v["digest"] == dg:
            return v
    return None


def rows_of(report, dg) -> set[str]:
    return {r["window_id"] for r in report["rows"] if r["digest"] == dg}


def entry_of(report, dg) -> dict:
    (e,) = [e for e in report["entries"] if e["digest"] == dg]
    return e


def skips_of(report, dg) -> list[str]:
    return [s["reason"] for s in report["skipped"] if s["digest"] == dg]


def report_digest(report) -> str:
    from cascade.validator.lane_d_state import report_digest as _rd

    return _rd(report)


# ── settlement calls ────────────────────────────────────────────────────────


def settle(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path, *,
           hash_=None, bench=None, prior=()):
    """`settle_lane_d` with the W5e gate inputs (`bench=None` or
    `hash_=None` = the gate cannot run: fail closed, `bench_hold`)."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(cfg=cfg, boundary=b, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), boundary_block_hash=hash_,
                         sentinel=None, bench=bench, **kw)


# ── the bench host (an external service: recorded suite outputs) ────────────


def _base(suite: str) -> list[tuple[str, float, float]]:
    rng = random.Random(f"w5g base {suite}")
    return [(f"{suite}/task-{i:03d}", math.exp(rng.gauss(-0.7, 0.6)),
             math.exp(rng.gauss(0.1, 0.4))) for i in range(BENCH_ITEMS)]


def suite_rows(suite: str, tag: str, factor: float = 1.0, *, nan: bool = False) -> list[dict]:
    """Per-item CRPS/MASE: the suite's base values times `factor` (above 1 =
    worse), with 1% per-item noise seeded by `tag`. `nan` makes the first
    item's CRPS NaN (a fault on that suite)."""
    rng = random.Random(f"w5g noise {suite} {tag}")
    out = []
    for item, c, m in _base(suite):
        out.append({"item": item, "crps": c * factor * math.exp(rng.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(rng.gauss(0.0, 0.01))})
    if nan:
        out[0]["crps"] = float("nan")
    return out


def bench_table(spec: dict) -> dict:
    """`{digest: {suite: rows}}` from `{digest: factor | "nan"}` over every
    gate suite (`"nan"` = a faulty row set on every suite)."""
    out = {}
    for dg, f in spec.items():
        out[dg] = {s: (suite_rows(s, dg, 1.0, nan=True) if f == "nan"
                       else suite_rows(s, dg, float(f))) for s in GATE_SUITES}
    return out


class BenchHost:
    """A stand-in for the bench pod: returns the recorded suite outputs for
    the digests it is asked for and logs every sweep call."""

    def __init__(self, table: dict):
        self.table = {d: dict(s) for d, s in table.items()}
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok", "rows": copy.deepcopy(self.table.get(d, {})[s])}
                    for s in suites if s in self.table.get(d, {})} for d in digests}

    def swept_digests(self) -> set[str]:
        return {d for ds, _ in self.calls for d in ds}


class DownHost:
    """The bench pod is down: every sweep raises (infrastructure)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, digests, suites) -> dict:
        self.calls += 1
        raise ConnectionError("w5g fixture: bench host unreachable")


def bench_session(host, session_id: str):
    from cascade.validator.lane_d_bench import SweepSession

    return SweepSession(session_id, host)


# Bench modes for a gated settlement of candidate `x` (KING_D = house `h`).
HOLD_MODES = ("no_bench", "no_hash", "host_down", "reference_fault")


def gate_inputs(mode: str, x: str, h: str, tag: str):
    """`(hash_, bench, host)` for one gated settlement.

    Hold modes (each a `bench_hold`, infrastructure): `no_bench` (no bench
    session), `no_hash` (no boundary block hash), `host_down` (the sweep
    raises), `reference_fault` (the house's rows are NaN). Decided modes:
    `pass` (the candidate beats both references), `regress` (it is worse:
    `gate_regression`), `candidate_fault` (its rows are NaN:
    `checkpoint_fault`)."""
    hash_ = block_hash(tag)
    if mode == "no_bench":
        return hash_, None, None
    if mode == "host_down":
        host = DownHost()
        return hash_, bench_session(host, f"sess-{tag}"), host
    spec = {"pass": {x: 0.9, h: 1.0}, "regress": {x: 1.10, h: 1.0},
            "candidate_fault": {x: "nan", h: 1.0},
            "reference_fault": {x: 0.9, h: "nan"},
            "no_hash": {x: 0.9, h: 1.0}}[mode]
    host = BenchHost(bench_table(spec))
    return (None if mode == "no_hash" else hash_), bench_session(host, f"sess-{tag}"), host


class Flow:
    """Challenger X against the genesis house H0. Boundary 1 lists no entry
    (nothing is judged). Boundary 2: X enters and clears at its first verdict
    on snapshot 2 and is gated with `mode2` (plan rev6.11 (b): a first clear
    is the crowning candidate; no confirmation duel). Later boundaries re-use
    snapshot 2 (no fresh window, so a held X can only be re-gated) unless a
    fresh snapshot is asked for."""

    def __init__(self, tmp_path: Path, *, mode2: str, cfg=None, extra=(), x_quality="sharp"):
        self.tmp = Path(tmp_path)
        self.cfg = cfg or lane_d_cfg()
        self.h0 = house()
        self.x = entry("x", reveal_ts=T_X)
        self.extra = list(extra)
        self.entries = [self.x, *self.extra]
        self.path = self.tmp / "state" / "lane_d_state.json"
        self.models = {self.h0["digest"]: ("bad", "house-0"),
                       self.x["digest"]: (x_quality, "x")}
        for e in self.extra:
            self.models[e["digest"]] = ("sharp", e["digest"][:8])
        self.uid_of = uids(*self.entries)
        self.pools: dict[int, tuple[Path, list[dict]]] = {}
        self.reports: dict[int, dict] = {}
        self.hosts: dict[int, object] = {}
        self.n = 0
        self.settle_at(1, "pass", snap=1, entries=[])
        assert self.reports[1]["judged"] == [], "fixture: boundary 1 lists no entry"
        self.settle_at(2, mode2, snap=2)
        v = verdict_of(self.reports[2], self.x["digest"])
        assert v is not None and v["stage"] == "first" and v["cleared"] is True, \
            f"fixture: X did not clear at its first verdict at boundary 2: {v}"

    def pool(self, snap: int):
        if snap not in self.pools:
            self.pools[snap] = snapshot(self.tmp, snap)
        return self.pools[snap]

    def settle_at(self, n: int, mode: str, *, snap: int = 2, entries=None):
        pool_dir, recs = self.pool(snap)
        hash_, bench, host = gate_inputs(mode, self.x["digest"], self.h0["digest"], f"b{n}")
        listed = self.entries if entries is None else entries
        out = settle(self.cfg, boundary(n), pool_dir, listed, self.h0,
                     forecasts(recs, self.models), self.uid_of, self.path,
                     hash_=hash_, bench=bench)
        self.reports[n] = out.report
        self.hosts[n] = host
        self.n = n
        return out.report

    def next(self, mode: str, *, snap: int = 2, entries=None):
        return self.settle_at(self.n + 1, mode, snap=snap, entries=entries)

    def state_entry(self) -> dict:
        return read_state(self.path)["entries"][self.x["digest"]]

    def king(self) -> str:
        return read_state(self.path)["king"]["digest"]
