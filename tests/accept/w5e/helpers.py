"""Shared spec and fixture builders for the W5e acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6) §1.3 "Verdict":

- The statistic is a paired cluster bootstrap of the CRPS/MASE geomean ratio,
  clusters = source. The challenger wins if `LCB_α ≥ d_margin` and the point
  gain is strictly positive.
- (rev6.11 (b), owner 2026-11-23) The confirmation duel is DROPPED: a crown
  is decided at the FIRST verdict, first clear (max-T LCB against KING_D)
  ∧ A10 ∧ the §1.3b gate. There is no `confirm` stage, no `confirm_pending`
  state and no D20 king-change rule; `confirm_on_king_change` is a legacy
  key, parsed and ignored. The stage leaves the producer, the audit
  (`cascade/audit/lane_d.py`) and the replay
  (`cascade/satellite/lane_d_replay.py`) together (PATHS, orchestrator
  2026-11-23), so the honest first-verdict crowns these fixtures build are
  audited and replayed as crowns.

§1.3b: one gate suite per crowning, drawn from the settlement boundary's
block hash over `gate_suites` minus the era's §16 sentinel; paired on one
host in one sweep against KING_D and the house; the sentinel is swept and
recorded, never gates. §1.5: every verdict publishes `rows_sha256`, the
sha256 of the exact scored per-window rows (canonical serialisation).

README.md states the contract. Nothing here imports a module in PATHS at
import time; `settle_gated` asserts the W5e keywords exist before calling,
so on main a gate test fails on an AssertionError inside its body.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
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
T_X = T0                         # the first challenger
T_Y = T0 + DAY                   # a later challenger (judged against a miner king)

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


# rev6.18 F4 (A3): these fixtures pin the live-only rule: the public half off
# (public_weight = 0, a spec setting) and no per-window ratio clamp in effect
# (a floor far below any fixture ratio), so their pre-F4 numbers stand. F4's
# own tests (tests/accept/w5e/test_w5e_combined.py) pin the combined rule.
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table with `over` applied
    (floors lowered to 24 windows / 4 sources; the 0.25 source cap never
    drops a window because every fixture group splits evenly)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **LIVE_ONLY, **over}, era_blocks_at=lambda _block: ERA)


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
    # rev6.10 (a): every listed feed pins its effective `embargo_hours` (W6b;
    # W5a fails a feed without it closed). 0 = the rev3 rule, so these
    # fixtures keep their verdicts under the embargo-aware W5a.
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False,
                         "embargo_hours": 0.0}
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


def snapshot(tmp_path: Path, n: int, t: int = T_X) -> tuple[Path, list[dict]]:
    """Snapshot `n` (n >= 1): 24 fresh windows over 4 sources, eligible for
    every T <= t, in a band a day after snapshot n - 1's. Snapshots never
    share a window id."""
    recs = group(f"s{n}", 6, band_start(t) + (n - 1) * DAY)
    return write_pool(tmp_path / f"pool{n}", recs, f"w5e-pool{n}"), recs


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


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def set_sha(window_ids) -> str:
    return hashlib.sha256("\n".join(sorted(window_ids)).encode()).hexdigest()


def block_hash(tag: str) -> str:
    """A fixture boundary block hash (0x + 64 hex)."""
    return "0x" + hashlib.sha256(f"w5e block {tag}".encode()).hexdigest()


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w5e-{name}")


def entry(name: str, *, reveal_ts: int, after: int = 0) -> dict:
    """An entry readied and revealed just after settlement `after`."""
    base = ARM_BLOCK + after * SETTLEMENT
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5e {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": base + 40, "reveal_block": base + 60, "reveal_ts": int(reveal_ts)}


def house(name: str = "house-0", reveal_ts: int = HOUSE_T) -> dict:
    return {"digest": digest(f"w5e {name}"), "reveal_ts": int(reveal_ts)}


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


# ── settlement calls ────────────────────────────────────────────────────────


def settle(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path, prior=()):
    """`settle_lane_d` with main's keywords only (no gate inputs): the legacy
    call. Armed, it may only be used where no member clears (rev6.11: a
    first clear is a crowning candidate; RV5: an armed crowning without the gate inputs is `bench_hold`);
    unarmed it returns `report=None, pay=None` as on main."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(cfg=cfg, boundary=b, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), **kw)


GATE_KEYWORDS = ("boundary_block_hash", "sentinel", "bench")


def settle_gated(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path, *,
                 hash_, bench, sentinel=None, prior=()):
    """`settle_lane_d` with the W5e gate inputs: the boundary's block hash
    (the gate draw), the era's §16 sentinel (`None` = W-S1 unarmed) and the
    bench host session (`lane_d_bench.SweepSession`)."""
    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    missing = [k for k in GATE_KEYWORDS if k not in params]
    assert not missing, (f"settle_lane_d takes no {missing}: the W3b gate is not wired into "
                         "the settlement (W5e)")
    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(cfg=cfg, boundary=b, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), boundary_block_hash=hash_,
                         sentinel=sentinel, bench=bench, **kw)


# ── the bench host (an external service: recorded suite outputs) ────────────


def _base(suite: str) -> list[tuple[str, float, float]]:
    rng = random.Random(f"w5e base {suite}")
    return [(f"{suite}/task-{i:03d}", math.exp(rng.gauss(-0.7, 0.6)),
             math.exp(rng.gauss(0.1, 0.4))) for i in range(BENCH_ITEMS)]


def suite_rows(suite: str, tag: str, factor: float = 1.0, *, nan: bool = False) -> list[dict]:
    """Per-item CRPS/MASE: the suite's base values times `factor` (above 1 =
    worse), with 1% per-item noise seeded by `tag`. `nan` makes the first
    item's CRPS NaN (a fault on that suite)."""
    rng = random.Random(f"w5e noise {suite} {tag}")
    out = []
    for item, c, m in _base(suite):
        out.append({"item": item, "crps": c * factor * math.exp(rng.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(rng.gauss(0.0, 0.01))})
    if nan:
        out[0]["crps"] = float("nan")
    return out


def bench_table(spec: dict) -> dict:
    """`{digest: {suite: rows}}` from `{digest: factor | {suite: factor | "nan"}}`
    over every gate suite; a suite factor `"nan"` is a faulty row set, a
    suite factor `None` leaves the suite out (not swept)."""
    out = {}
    for dg, f in spec.items():
        per = f if isinstance(f, dict) else {}
        default = 1.0 if isinstance(f, dict) else float(f)
        out[dg] = {}
        for s in GATE_SUITES:
            v = per.get(s, default)
            if v is None:
                continue
            out[dg][s] = (suite_rows(s, dg, 1.0, nan=True) if v == "nan"
                          else suite_rows(s, dg, float(v)))
    return out


class BenchHost:
    """A stand-in for the bench pod (an external service): returns the
    recorded suite outputs for the digests it is asked for and logs every
    sweep call as `(digests, suites)`."""

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

    def swept_suites(self) -> set[str]:
        return {s for _, ss in self.calls for s in ss}


def bench_session(host: BenchHost, session_id: str):
    from cascade.validator.lane_d_bench import SweepSession

    return SweepSession(session_id, host)


def passing_bench(candidates, references, session_id: str):
    """A healthy bench session on which every `candidates` digest beats every
    `references` digest (KING_D, the house, a first-clear king) on every gate
    suite, so a gated crowning candidate passes."""
    spec = {**{d: 1.0 for d in references}, **{d: 0.9 for d in candidates}}
    return bench_session(BenchHost(bench_table(spec)), session_id)


def settle_crowning(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path, *,
                    candidates, references, tag: str, prior=()):
    """An ARMED settlement that may crown, so it carries the gate inputs (RV5:
    an armed crowning without them fails closed). The bench passes every
    `candidates` digest; `tag` names the block hash and the session."""
    return settle_gated(cfg, b, pool_dir, entries, house_rec, fc, uid_of, state_path,
                        hash_=block_hash(tag),
                        bench=passing_bench(candidates, references, f"sess-{tag}"),
                        prior=prior)


def drawn_suite(cfg, hash_: str, sentinel=None) -> str:
    from cascade.validator.lane_d_gate import gate_suite_for

    return gate_suite_for(hash_, gate_suites=cfg.gate_suites, gate_weights=cfg.gate_weights,
                          sentinel=sentinel)


def expected_gate_record(cfg, b: int, hash_: str, sentinel=None) -> dict:
    from cascade.validator.lane_d_gate import gate_record

    return gate_record(cfg, int(b), hash_, sentinel=sentinel)
