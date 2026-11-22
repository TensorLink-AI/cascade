"""Shared spec and fixture builders for the W7c acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6) §1.5 "State":

- A new `lane_d_state.json` holds KING_D, the court, the house digest and T,
  pending entries with their eligible-window progress, frozen sets, ...
- Its digest is in the receipt's prior and posterior digests (DEC-CA-0058),
  and the report digest is bound into the receipt as the bench source is.
- It's protected like `cascade_state.json`: snapshot before every restart,
  never hand-edited.

§3.2 invariant 5: **Restarts are idempotent.** No re-score, no double crown.
T11: Validator restart mid-judgement -> No re-score, no double crown.

RV5 (evidence/rv5.md, W7b gaps for W7c): (a) `verify_report` binds no
expected boundary, so an old report replays unless the boundary is checked;
(b) sr25519 signatures are randomized, so every digest W7c binds must hash
`canonical_body`, never the signature bytes; (c) `json.loads` keeps the last
of duplicate keys, so duplicate keys must be refused when loading.

The chain: one challenger X (sharp forecasts) and the genesis house H0 (bad
forecasts). Boundary 1 judges X on snapshot 1: a first clear,
`confirm_pending`. Boundary 2 confirms X on snapshot 2 (disjoint) and crowns
it. Boundary 3 judges nothing (no open entry) and only advances the state.
The pool snapshots are written once per test and shared by every run of
that test, so two runs see the same pool bytes.

README.md states the contract. Nothing here imports a module in PATHS at
import time; every builder is called from test bodies only.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]

HOUR = 3600
DAY = 86_400
HORIZON = 64
HISTORY = 256
MARGIN_H = 6.0
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY
T_X = T0

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
N_SNAPSHOTS = 3


def boundary(n: int) -> int:
    """The n-th settlement boundary after arming (n >= 1)."""
    return ARM_BLOCK + n * SETTLEMENT


SOURCES = ("src0", "src1", "src2", "src3")
FEED_LAG_HOURS = 1.0
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)
GATE_SUITES = ("gift", "boom", "time", "fev")
BENCH_ITEMS = 40

# ── the pinned contract (README "Contract") ─────────────────────────────────

RECEIPT_KEYS = ("schema", "boundary", "prior_state_sha256", "posterior_state_sha256",
                "report_sha256")
LAST_REPORT_KEYS = ("boundary", "rows_sha256", "report_sha256", "prior_state_sha256")
STATE_ERROR_CODES = ("bad_json", "duplicate_key", "not_canonical", "unknown_schema",
                     "state_digest_mismatch", "stale_boundary", "report_conflict")
RECEIPT_ERROR_CODES = ("not_object", "unknown_schema", "missing_field", "unknown_field",
                       "bad_value", "boundary_mismatch", "report_mismatch", "state_mismatch",
                       "boundary_order", "chain_break")


def canonical(obj) -> bytes:
    """README "Canonical bytes", re-derived here independently."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_sha(state: dict) -> str:
    return sha(canonical(state))


def body_sha(report: dict) -> str:
    """sha256 of the report's canonical body (every field but `signature`)."""
    return sha(canonical({k: v for k, v in report.items() if k != "signature"}))


def is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


# ── config ──────────────────────────────────────────────────────────────────


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table (floors 24 windows / 4
    sources, as in the W5e fixture)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def unarmed_cfg():
    """`[lane_d]` armed only far after every fixture boundary."""
    return lane_d_cfg(from_block=ARM_BLOCK + 100 * SETTLEMENT)


# ── pool_d fixtures (trimmed copy of tests/accept/w5e/helpers.py) ───────────


def _bound(t: int) -> int:
    return int(t + MARGIN_H * HOUR)


def _band_start(t: int) -> int:
    return _bound(t) + 3 * HOUR


def window_id(series_id: str, target_ts) -> str:
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def _record(series_id: str, source: str, start: int) -> dict:
    ts = [int(start + i * HOUR) for i in range(HORIZON)]
    return {"series_id": series_id, "source": source, "feed": f"feed-{source}",
            "freq": "H", "seasonal_period": 24, "target_ts": ts,
            "first_seen_ts": [t + 1800 for t in ts], "first_target_ts": ts[0],
            "end_ts": ts[-1], "frontier": [[start - HOUR, start - 2 * HOUR]]}


def _group(tag: str, per_source: int, start: int) -> list[dict]:
    out = []
    for k in range(per_source):
        for j, src in enumerate(SOURCES):
            i = k * len(SOURCES) + j
            out.append(_record(f"{tag}-{src}-{k}", src, start + i * 60))
    return out


def _target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(f"{series_id}:{start}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON, dtype=np.float64)
    full = 20.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return full[:HISTORY], full[HISTORY:]


def _write_pool(directory: Path, records: list[dict], snapshot: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False}
             for r in records}
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "feeds": dict(sorted(feeds.items())), "series": records}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in records:
        h, y = _target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def _snapshot_records(n: int) -> list[dict]:
    """Snapshot n: 24 fresh windows over 4 sources, eligible for T_X, a day
    after snapshot n - 1's band. Snapshots never share a window id."""
    return _group(f"s{n}", 6, _band_start(T_X) + (n - 1) * DAY)


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


QUALITY = {"sharp": (0.005, 0.1, 0.0), "bad": (0.0, 0.3, 3.0)}


def _model_forecasts(records, quality: str, name: str) -> dict:
    noise, spread, offset = QUALITY[quality]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = {}
    for r in records:
        _, y = _target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        med = y + offset * sd + rng.normal(0, noise * sd, HORIZON)
        out[window_id(r["series_id"], r["target_ts"])] = _quantiles(med, spread * sd)
    return out


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
    return keypair_class().create_from_uri(f"//w7c-{name}")


def block_hash(tag: str) -> str:
    return "0x" + hashlib.sha256(f"w7c block {tag}".encode()).hexdigest()


# ── the bench host (the one external service: recorded suite outputs) ───────


def _suite_rows(suite: str, tag: str, factor: float) -> list[dict]:
    base = random.Random(f"w7c base {suite}")
    noise = random.Random(f"w7c noise {suite} {tag}")
    out = []
    for i in range(BENCH_ITEMS):
        c, m = math.exp(base.gauss(-0.7, 0.6)), math.exp(base.gauss(0.1, 0.4))
        out.append({"item": f"{suite}/task-{i:03d}",
                    "crps": c * factor * math.exp(noise.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(noise.gauss(0.0, 0.01))})
    return out


class BenchHost:
    """A stand-in for the bench pod: returns recorded per-item rows (a factor
    below 1 = better than the references) and logs every sweep call."""

    def __init__(self, factors: dict):
        self.table = {d: {s: _suite_rows(s, d, f) for s in GATE_SUITES}
                      for d, f in factors.items()}
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok", "rows": copy.deepcopy(self.table.get(d, {})[s])}
                    for s in suites if s in self.table.get(d, {})} for d in digests}


GATE_KEYWORDS = ("boundary_block_hash", "sentinel", "bench")


def settle_takes_gate_inputs() -> bool:
    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    return all(k in params for k in GATE_KEYWORDS)


class CountingForecasts(dict):
    """The eval host's forecasts, counting every read (a re-score reads them)."""

    def __init__(self, data: dict):
        super().__init__(data)
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)

    def __contains__(self, key):
        self.reads += 1
        return super().__contains__(key)

    def __iter__(self):
        self.reads += 1
        return super().__iter__()

    def items(self):
        self.reads += 1
        return super().items()

    def keys(self):
        self.reads += 1
        return super().keys()

    def values(self):
        self.reads += 1
        return super().values()


# ── the chain ───────────────────────────────────────────────────────────────


class Chain:
    """The fixture chain. `pools_dir` holds the snapshots (written once and
    shared by every run of a test); each run has its own state path."""

    def __init__(self, root: Path, pools_dir: Path | None = None):
        self.root = Path(root)
        self.pools_dir = Path(pools_dir) if pools_dir is not None else self.root / "pools"
        self.records = [_snapshot_records(n) for n in range(1, N_SNAPSHOTS + 1)]
        self.pools = []
        for n, recs in enumerate(self.records, start=1):
            d = self.pools_dir / f"pool{n}"
            if not (d / "tgt.npz").is_file():
                _write_pool(d, recs, f"w7c-pool{n}")
            self.pools.append(d)
        self.house = {"digest": digest("w7c house-0"), "reveal_ts": HOUSE_T}
        self.x = {"hotkey": fixture_key("x").ss58_address, "digest": digest("w7c x"),
                  "context": 4096, "coldkey_at_ready": fixture_key("x-cold").ss58_address,
                  "ready_block": ARM_BLOCK + 40, "reveal_block": ARM_BLOCK + 60,
                  "reveal_ts": T_X}
        every = [r for recs in self.records for r in recs]
        self.forecasts = {self.x["digest"]: _model_forecasts(every, "sharp", "x"),
                          self.house["digest"]: _model_forecasts(every, "bad", "house-0")}
        self.uid_of = {self.x["hotkey"]: 11}
        self.signer = fixture_key("signer")
        self.hosts: list[BenchHost] = []

    def state_path(self, run: str) -> Path:
        p = self.root / run / "state" / "lane_d_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def settle(self, n: int, state_path: Path, *, cfg=None, forecasts=None):
        """`settle_lane_d` at boundary(n) on snapshot n (earlier snapshots
        retained). The gate inputs (a passing bench session) are passed
        whenever `settle_lane_d` takes them: an armed crowning without them
        fails closed (W5e RV5)."""
        from cascade.validator.lane_d import settle_lane_d

        kw = {}
        if n > 1:
            kw["prior_pool_dirs"] = list(self.pools[:n - 1])
        if settle_takes_gate_inputs():
            from cascade.validator.lane_d_bench import SweepSession

            host = BenchHost({self.house["digest"]: 1.0, self.x["digest"]: 0.9})
            self.hosts.append(host)
            kw.update(boundary_block_hash=block_hash(f"b{n}"), sentinel=None,
                      bench=SweepSession(f"w7c-sess-b{n}", host))
        fc = self.forecasts if forecasts is None else forecasts
        return settle_lane_d(cfg=cfg if cfg is not None else lane_d_cfg(), boundary=boundary(n),
                             pool_dir=self.pools[n - 1], entries=[dict(self.x)],
                             house=dict(self.house), forecasts=fc, uid_of=dict(self.uid_of),
                             state_path=state_path, signer=self.signer, **kw)

    def run(self, state_path: Path, upto: int) -> list:
        return [self.settle(n, state_path) for n in range(1, upto + 1)]


def receipt_of(settlement):
    """The settlement's receipt record (`None` when the field is absent, so a
    test fails on its assertion, not on an AttributeError)."""
    return getattr(settlement, "receipt", None)


def read_bytes(path) -> bytes | None:
    p = Path(path)
    return p.read_bytes() if p.is_file() else None


def write_canonical(path, state: dict) -> None:
    Path(path).write_bytes(canonical(state))


def crash_on_rename(monkeypatch, target) -> None:
    """Simulate a crash at the rename that would land `target`: `os.replace`
    (and `os.rename`) onto `target` raise; every other rename proceeds."""
    want = Path(target).resolve()

    def wrap(real):
        def boom(src, dst, *a, **k):
            if Path(dst).resolve() == want:
                raise OSError(f"w7c simulated crash before {want.name} landed")
            return real(src, dst, *a, **k)
        return boom

    monkeypatch.setattr(os, "replace", wrap(os.replace))
    monkeypatch.setattr(os, "rename", wrap(os.rename))


def chain_fingerprint(chain: Chain, state_path: Path, upto: int) -> dict:
    """Receipts, report body digests and state file digests of a clean run."""
    out = {"receipts": [], "bodies": [], "states": []}
    for n in range(1, upto + 1):
        s = chain.settle(n, state_path)
        out["receipts"].append(receipt_of(s))
        out["bodies"].append(body_sha(s.report))
        out["states"].append(sha(state_path.read_bytes()))
    return out
