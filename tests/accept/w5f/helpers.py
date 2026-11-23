"""Shared spec and fixture builders for the W5f acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6):

- §1.4 "The house (our lane A king)": the house is the checkpoint digest
  lane A currently treats as king. It's refreshed whenever that digest
  changes. It enters at that settlement, and its T is the block at which
  that digest was published. If the house holds KING_D, the lane D king slot
  goes unpaid (D21 DECIDED: burn). Prior miner kings keep their decaying
  tails. The house is never in the court and never a prior king.
- §2 item 1-2: the house enters whenever KING_A's digest changes. The house
  can retake KING_D when a better lane A king beats it on forward windows.
  The lane D king slot then burns.
- §4 T10: "House pointer swapped to a stronger checkpoint | House retakes;
  slot burns; prior miner tails keep paying". §4: house changes are
  simulated by swapping the house pointer.
- §1.5 "State": `lane_d_state.json` holds KING_D, the court, the house
  digest and T.

README.md states the contract. Nothing here imports a module in PATHS at
import time, so on main every test is collected and fails inside its body.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from statistics import NormalDist

import numpy as np

HOUR = 3600
DAY = 86_400
HORIZON = 64
HISTORY = 256
MARGIN_H = 6.0
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY           # H0, the genesis house
T_X = T0                         # miner X (crowned first)
T_Y = T0 + DAY                   # miner Y (crowned second)
T_H1 = T0 + 2 * DAY              # H1, the first swapped house pointer
T_H2 = T0 + 3 * DAY              # H2, a later house pointer
T_LATE = T_H2                    # every fixture window is eligible for every T above

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
    (floors lowered to 24 windows / 4 sources; 2 prior kings, decay 0.5)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200,
           "reward_prior_kings": 2, "king_decay": 0.5}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def unarmed_cfg():
    """The same table with lane D never active (`from_block = 0`)."""
    from cascade.shared.config.lane_d import parse_lane_d

    return parse_lane_d({}, era_blocks_at=lambda _block: ERA)


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
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False, "embargo_hours": 0.0}
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


def snapshot(root: Path, n: int) -> tuple[Path, list[dict]]:
    """Snapshot `n` (n >= 1): 24 fresh windows over 4 sources, eligible for
    every fixture T (<= T_LATE), in a band a day after snapshot n - 1's.
    Snapshots never share a window id."""
    recs = group(f"s{n}", 6, band_start(T_LATE) + (n - 1) * DAY)
    return write_pool(root / f"pool{n}", recs, f"w5f-pool{n}"), recs


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


# Model quality ladder: sharp > good > fair > bad.
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


def block_hash(tag: str) -> str:
    """A fixture boundary block hash (0x + 64 hex)."""
    return "0x" + hashlib.sha256(f"w5f block {tag}".encode()).hexdigest()


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w5f-{name}")


def entry(name: str, *, reveal_ts: int, after: int = 0) -> dict:
    """A miner entry readied and revealed just after settlement `after`."""
    base = ARM_BLOCK + after * SETTLEMENT
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w5f {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": base + 40, "reveal_block": base + 60, "reveal_ts": int(reveal_ts)}


def house(name: str, reveal_ts: int) -> dict:
    """The verified house record settle_lane_d takes: `{digest, reveal_ts}`."""
    return {"digest": digest(f"w5f {name}"), "reveal_ts": int(reveal_ts)}


H0 = house("house-0", HOUSE_T)
H1 = house("house-1", T_H1)
H2 = house("house-2", T_H2)
X = entry("x", reveal_ts=T_X, after=0)
Y = entry("y", reveal_ts=T_Y, after=2)
UIDS = {X["hotkey"]: 11, Y["hotkey"]: 12}


def read_state(path) -> dict:
    return json.loads(Path(path).read_text())


def verdict_of(report, dg) -> dict | None:
    for v in report["verdicts"]:
        if v["digest"] == dg:
            return v
    return None


def rows_of(report, dg) -> set[str]:
    return {r["window_id"] for r in report["rows"] if r["digest"] == dg}


def reports_dir(state_path) -> Path:
    p = Path(state_path)
    return p.parent / f"{p.stem}_reports"


def tree_bytes(directory: Path) -> dict:
    """`{relative path: bytes}` of every file under `directory` (none if absent)."""
    d = Path(directory)
    if not d.exists():
        return {}
    return {str(f.relative_to(d)): f.read_bytes() for f in sorted(d.rglob("*")) if f.is_file()}


def body(report: dict) -> dict:
    """The report without its (randomized sr25519) signature."""
    return {k: v for k, v in report.items() if k != "signature"}


# ── the bench host (an external service: recorded suite outputs) ────────────


def _base(suite: str) -> list[tuple[str, float, float]]:
    rng = random.Random(f"w5f base {suite}")
    return [(f"{suite}/task-{i:03d}", math.exp(rng.gauss(-0.7, 0.6)),
             math.exp(rng.gauss(0.1, 0.4))) for i in range(BENCH_ITEMS)]


def suite_rows(suite: str, tag: str, factor: float) -> list[dict]:
    rng = random.Random(f"w5f noise {suite} {tag}")
    return [{"item": item, "crps": c * factor * math.exp(rng.gauss(0.0, 0.01)),
             "mase": m * factor * math.exp(rng.gauss(0.0, 0.01))}
            for item, c, m in _base(suite)]


class BenchHost:
    """A stand-in for the bench pod: returns recorded suite outputs for the
    digests it is asked for and logs every sweep call."""

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok", "rows": copy.deepcopy(self.table[d][s])}
                    for s in suites if s in self.table.get(d, {})} for d in digests}


def passing_bench(candidates, references, session_id: str):
    """A healthy bench session on which every `candidates` digest beats every
    `references` digest on every gate suite."""
    from cascade.validator.lane_d_bench import SweepSession

    spec = {**{d: 1.0 for d in references}, **{d: 0.9 for d in candidates}}
    table = {d: {s: suite_rows(s, d, f) for s in GATE_SUITES} for d, f in spec.items()}
    return SweepSession(session_id, BenchHost(table))


# ── settlement calls ────────────────────────────────────────────────────────


def settle(cfg, b, pool_dir, entries, house_rec, fc, state_path, *, prior=(),
           candidates=(), references=(), tag=None, uid_of=None):
    """`settle_lane_d` through the real settlement. With `tag` the call
    carries the W5e gate inputs: the block hash named by `tag` and a bench
    session on which every `candidates` digest beats every `references`
    digest (so any crowning or retake the rules allow can pass the gate)."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    if tag is not None:
        kw.update(boundary_block_hash=block_hash(tag),
                  bench=passing_bench(candidates, references, f"sess-{tag}"))
    return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(b), cfg=cfg, boundary=b, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=copy.deepcopy(house_rec),
                         forecasts=fc, uid_of=dict(UIDS if uid_of is None else uid_of),
                         state_path=state_path, signer=fixture_key("signer"), **kw)


class Chain:
    """A lane D settlement chain in `root`: one fresh snapshot per boundary,
    forecasts for every registered model on it, every report kept."""

    def __init__(self, root: Path, cfg=None):
        self.root = Path(root)
        self.cfg = cfg or lane_d_cfg()
        self.path = self.root / "state" / "lane_d_state.json"
        self.models: dict[str, tuple[str, str]] = {}
        self.reports: dict[int, dict] = {}
        self.pays: dict[int, dict] = {}
        self.pools: dict[int, tuple[Path, list[dict]]] = {}

    def model(self, dg: str, quality: str, name: str) -> None:
        self.models[dg] = (quality, name)

    def pool(self, n: int) -> tuple[Path, list[dict]]:
        if n not in self.pools:
            self.pools[n] = snapshot(self.root / "pools", n)
        return self.pools[n]

    def forecasts(self, recs) -> dict:
        return {dg: model_forecasts(recs, q, name) for dg, (q, name) in self.models.items()}

    def run(self, n: int, entries, house_rec, *, candidates=(), references=(),
            state_path=None):
        """Settle boundary(n) on snapshot n, with the gate inputs (a bench on
        which `candidates` beat `references`)."""
        pool_dir, recs = self.pool(n)
        s = settle(self.cfg, boundary(n), pool_dir, entries, house_rec, self.forecasts(recs),
                   state_path or self.path, candidates=candidates, references=references,
                   tag=f"{self.root.name}-b{n}")
        if state_path is None:
            self.reports[n], self.pays[n] = s.report, s.pay
        return s

    def state(self) -> dict:
        return read_state(self.path)

    def copy_to(self, root: Path) -> Path:
        """A restart: the state file and its persisted reports, copied as an
        operator snapshot would be, into `root`. Returns the new state path."""
        new = Path(root) / "state" / "lane_d_state.json"
        new.parent.mkdir(parents=True)
        shutil.copyfile(self.path, new)
        src = reports_dir(self.path)
        if src.exists():
            shutil.copytree(src, reports_dir(new))
        return new


def crown_x(chain: Chain) -> None:
    """boundary(1): X (fair) clears against the genesis house H0 (bad) at its
    first verdict and, through the gate, is crowned (plan rev6.11 (b): no
    confirmation duel); boundary(2): X holds KING_D, nothing is judged."""
    chain.model(H0["digest"], "bad", "house-0")
    chain.model(X["digest"], "fair", "x")
    chain.run(1, [X], H0)
    chain.run(2, [X], H0, candidates=[X["digest"]], references=[H0["digest"]])
    st = chain.state()
    assert st["king"]["digest"] == X["digest"] and not st["king"]["house"], \
        "fixture: X was not crowned"


def crown_y(chain: Chain) -> None:
    """boundary(3): Y (good) clears KING_D = X and the house bar H0 at its
    first verdict and is crowned (rev6.11 (b)); boundary(4) judges nothing:
    Y is KING_D and X its prior miner king."""
    chain.model(Y["digest"], "good", "y")
    chain.run(3, [X, Y], H0)
    chain.run(4, [X, Y], H0, candidates=[Y["digest"]],
              references=[X["digest"], H0["digest"]])
    st = chain.state()
    assert st["king"]["digest"] == Y["digest"] and not st["king"]["house"], \
        "fixture: Y was not crowned"
    assert [p["digest"] for p in st["prior_kings"]] == [X["digest"]], \
        "fixture: X is not Y's prior king"


def miner_court(chain: Chain) -> None:
    """boundary(1)..(4): two real miner crowns, KING_D = Y, prior = [X]."""
    crown_x(chain)
    crown_y(chain)
