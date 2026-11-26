"""Shared spec and fixture builders for the W7d acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6):

- §3.0 W7d: "Satellite replay: an **independent** implementation, no imports
  from producer eligibility, verdict or weight code (invariant 12)".
- §3.2 invariant 7: "Replay compares structured fields only."
- §3.2 invariant 12: "Audit and satellite-replay code never import the
  producer's eligibility, verdict or weight functions." (W7d, W9;
  import-graph check)
- §14.3 item 1: "Satellites verify the signature, check that the parameters
  stated in the report equal `config_at(boundary)` from the adopted overlay,
  check structural rules, re-run the bootstrap from signed rows, and derive
  weights." Item 4: "Structured comparisons only."
- §1.5: "`checks.derive_weights` and `check_lane_d` re-derive the lane D
  weights from the receipt chain."
- rev6.11 (b): every crown is decided at the first verdict (first clear ∧
  A10 ∧ gate); the replay re-derives no confirmation stage, record or D20
  rule, and a held first clear is `hold` / `bench_hold` (status / reason;
  its verdict outcome is `cleared`, never `crowned`).

Every honest chain is a sequence of real `settle_lane_d` runs on a synthetic
pool (numpy only, no torch), published the way W7c persists it: the signed
report files under `<state stem>_reports/lane_d_report_<boundary>.json`, the
receipt records, and (optionally) the state snapshot bytes after each
settlement. The bench pod is the one mocked external service. The signer is
a deterministic TEST keypair.

README.md states the contract. Nothing here imports a module in PATHS at
import time.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]

MODULE = "cascade.audit.lane_d_chain"
MODULE_PATH = REPO / "cascade" / "audit" / "lane_d_chain.py"

HOUR = 3600
DAY = 86_400
HORIZON = 64
HISTORY = 256
MARGIN_H = 6.0
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY
T_X = T0
T_Y = T0 + DAY

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
N_SNAPSHOTS = 5
FLOOR_WINDOWS = 48          # two snapshots of 24 windows each

STATE_SCHEMA = "lane-d-state/1"


def boundary(n: int) -> int:
    """The n-th settlement boundary after arming (n >= 1)."""
    return ARM_BLOCK + n * SETTLEMENT


SOURCES = ("src0", "src1", "src2", "src3")
FEED_LAG_HOURS = 1.0
EMBARGO_HOURS = 48.0        # [lane_d] embargo_hours (rev6.10 (a))
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)
GATE_SUITES = ("gift", "boom", "time", "fev")
BENCH_ITEMS = 40

# ── the import-graph contract (README "Invariant 12") ───────────────────────

# Producer eligibility, verdict, weight and emission code, plus the audit (a
# second independent re-derivation the replay must not lean on). A module is
# forbidden when its dotted name equals an entry or starts with entry + ".".
FORBIDDEN_PACKAGES = (
    "cascade.validator",        # lane_d, lane_d_frozen, lane_d_forward, lane_d_bench,
                                # lane_d_gate, lane_d_report, lane_d_state, weights, ...
    "cascade.shared.emission",  # lane_d_vector, lane_parts (weights)
    "cascade.shared.chain",     # decayed_share_vector (weights)
    "cascade.shared.lane_b",    # lane B weights
    "cascade.shared.lane_c",    # lane C weights
    # rev6.18 B2: this module now lives IN the audit package (the chain rules
    # of check_lane_d); it still never leans on the per-report checker or
    # the rest of the audit (cascade.audit.lane_d imports it, not the reverse)
    "cascade.audit.lane_d", "cascade.audit.checks", "cascade.audit.main",
    "cascade.audit.rederive",
    "cascade.lane_d",           # the lane D intake service
    "cascade.laned",            # lane D operator tooling (vault sweeper, restore, ...)
    "cascade.eval.lane_d_scorer", "cascade.eval.lane_d_ingest",  # eval-host scoring
    "cascade.eval.lane_d_t7", "cascade.eval.lane_d_handshake",
    "cascade.trainer",
    "torch",
)
# The shared pure primitives the replay may import directly (besides its own
# package `cascade.satellite` and the parent packages of these modules).
ALLOWED_PRIMITIVES = (
    "cascade.eval.bootstrap",       # cohort_maxt_lcbs: the generic statistic
    "cascade.eval.crps",            # bootstrap's own dependency
    "cascade.shared.sr25519",       # verify_hotkey: the signature verifier
    "cascade.shared.config.lane_d", # the parsed [lane_d] config type
)
ALLOWED_PACKAGES = ("cascade", "cascade.eval", "cascade.shared", "cascade.shared.config",
                    "cascade.audit")

UNVERIFIABLE_REASONS = ("unknown_schema", "report_missing", "report_unreadable",
                        "signature_unverifiable", "prior_unverified",
                        # rev6.11 (b): a report carrying a confirmation stage
                        "confirmation_stage_present")


def forbidden(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PACKAGES)


# ── canonical bytes ─────────────────────────────────────────────────────────


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def body_sha(report: dict) -> str:
    return sha(canonical({k: v for k, v in report.items() if k != "signature"}))


def rows_sha(rows: list) -> str:
    return sha(canonical(list(rows)))


def genesis_digest(house: dict) -> str:
    """W7c `genesis(house)` digest, re-derived from its documented shape."""
    h = {"digest": str(house["digest"]), "reveal_ts": int(house["reveal_ts"])}
    return sha(canonical({"schema": STATE_SCHEMA, "boundary": 0, "house": dict(h),
                          "king": {**h, "hotkey": None, "house": True},
                          "prior_kings": [], "entries": {}}))


# ── config ──────────────────────────────────────────────────────────────────


# rev6.18 F4 (A3): these fixtures pin the live-only rule: the public half off
# (public_weight = 0, a spec setting) and no per-window ratio clamp in effect
# (a floor far below any fixture ratio), so their pre-F4 numbers stand. F4's
# own tests (tests/accept/w5e/test_w5e_combined.py) pin the combined rule.
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}


def lane_d_cfg(**over):
    """The real W2 parse of an armed `[lane_d]` table. Floors 48 windows / 4
    sources (rev6.11, as in the W7c fixture): two snapshots of 24 windows, so
    an entry waits on the floors at its first settlement and is judged, once,
    at its second."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": FLOOR_WINDOWS, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **LIVE_ONLY, **over}, era_blocks_at=lambda _block: ERA)


# ── pool_d fixtures (trimmed copy of tests/accept/w7c/helpers.py) ───────────


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
    # rev6.10 / rev6.12 (W6b): every listed feed pins its effective embargo
    # (48 h, the decided default; a listed feed without it fails closed).
    feeds = {r["feed"]: {"mirror_lag_hours": FEED_LAG_HOURS, "forecast_product": False,
                         "embargo_hours": EMBARGO_HOURS}
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
    """Snapshot n: 24 fresh windows over 4 sources, eligible for T_X (and for
    T_Y from snapshot 2 on), a day after snapshot n - 1's band."""
    return _group(f"s{n}", 6, _band_start(T_X) + (n - 1) * DAY)


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


QUALITY = {"sharper": (0.001, 0.03, 0.0), "sharp": (0.005, 0.1, 0.0),
           "bad": (0.0, 0.3, 3.0)}


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
    return keypair_class().create_from_uri(f"//w7d-{name}")


def block_hash(tag: str) -> str:
    return "0x" + hashlib.sha256(f"w7d block {tag}".encode()).hexdigest()


# ── the bench host (the one external service: recorded suite outputs) ───────


def _suite_rows(suite: str, tag: str, factor: float) -> list[dict]:
    base = random.Random(f"w7d base {suite}")
    noise = random.Random(f"w7d noise {suite} {tag}")
    out = []
    for i in range(BENCH_ITEMS):
        c, m = math.exp(base.gauss(-0.7, 0.6)), math.exp(base.gauss(0.1, 0.4))
        out.append({"item": f"{suite}/task-{i:03d}",
                    "crps": c * factor * math.exp(noise.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(noise.gauss(0.0, 0.01))})
    return out


class BenchHost:
    """A stand-in for the bench pod: recorded per-item rows (a factor below 1
    is better than the references)."""

    def __init__(self, factors: dict):
        self.table = {d: {s: _suite_rows(s, d, f) for s in GATE_SUITES}
                      for d, f in factors.items()}

    def __call__(self, digests, suites) -> dict:
        return {d: {s: {"status": "ok", "rows": copy.deepcopy(self.table.get(d, {})[s])}
                    for s in suites if s in self.table.get(d, {})} for d in digests}


GATE_KEYWORDS = ("boundary_block_hash", "sentinel", "bench")


def settle_takes_gate_inputs() -> bool:
    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    return all(k in params for k in GATE_KEYWORDS)


# ── the published chain ─────────────────────────────────────────────────────


@dataclass
class Publication:
    """What a satellite is given: the signed report files (W7c layout), the
    receipt records in settlement order, the state snapshot bytes per
    boundary (optional input), the metagraph `{hotkey: uid}` per boundary, the
    config and the pinned signer."""

    reports_dir: Path
    receipts: list[dict]
    states: dict[int, bytes]
    uid_of: dict[int, dict]
    cfg: object
    signer_ss58: str
    boundaries: list[int] = field(default_factory=list)
    settlements: list = field(default_factory=list)

    def reports(self) -> dict[int, dict]:
        """`{boundary: report}` parsed from the published files."""
        return {b: json.loads((self.reports_dir / f"lane_d_report_{b}.json").read_text())
                for b in self.boundaries}

    def report(self, n: int) -> dict:
        return self.reports()[self.boundaries[n - 1]]


class Chain:
    """One honest fixture chain: X (sharp, revealed after arming), Y
    (sharper, revealed after boundary(2)) and the genesis house H0 (bad).
    `step` settles boundary(n) on snapshot n (earlier snapshots retained)."""

    def __init__(self, root: Path, *, cfg=None, run: str = "run"):
        self.root = Path(root)
        self.cfg = cfg if cfg is not None else lane_d_cfg()
        self.pools_dir = self.root / "pools"
        self.records = [_snapshot_records(n) for n in range(1, N_SNAPSHOTS + 1)]
        self.pools = []
        for n, recs in enumerate(self.records, start=1):
            d = self.pools_dir / f"pool{n}"
            if not (d / "tgt.npz").is_file():
                _write_pool(d, recs, f"w7d-pool{n}")
            self.pools.append(d)
        self.h0 = {"digest": digest("w7d house-0"), "reveal_ts": HOUSE_T}
        self.h1 = {"digest": digest("w7d house-1"), "reveal_ts": HOUSE_T + HOUR}
        self.x = self._entry("x", T_X, after=0)
        self.y = self._entry("y", T_Y, after=2)
        every = [r for recs in self.records for r in recs]
        self.forecasts = {self.x["digest"]: _model_forecasts(every, "sharp", "x"),
                          self.y["digest"]: _model_forecasts(every, "sharper", "y"),
                          self.h0["digest"]: _model_forecasts(every, "bad", "house-0"),
                          self.h1["digest"]: _model_forecasts(every, "bad", "house-1")}
        self.uids = {self.x["hotkey"]: 11, self.y["hotkey"]: 12}
        self.signer = fixture_key("signer")
        self.state_path = self.root / run / "state" / "lane_d_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.pub = Publication(reports_dir=self.state_path.parent / "lane_d_state_reports",
                               receipts=[], states={}, uid_of={}, cfg=self.cfg,
                               signer_ss58=self.signer.ss58_address)

    @staticmethod
    def _entry(name: str, reveal_ts: int, *, after: int) -> dict:
        base = ARM_BLOCK + after * SETTLEMENT
        return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w7d {name}"),
                "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
                "ready_block": base + 40, "reveal_block": base + 60, "reveal_ts": int(reveal_ts)}

    def step(self, n: int, *, entries, house=None, gated: bool = True):
        """`settle_lane_d` at boundary(n). `gated` passes the W5e gate inputs
        (a bench on which Y beats X beats both houses); otherwise the legacy
        call (an armed crowning candidate is then `bench_hold`)."""
        from cascade.validator.lane_d import settle_lane_d

        kw = {}
        if n > 1:
            kw["prior_pool_dirs"] = list(self.pools[:n - 1])
        if gated and settle_takes_gate_inputs():
            from cascade.validator.lane_d_bench import SweepSession

            host = BenchHost({self.h0["digest"]: 1.0, self.h1["digest"]: 1.0,
                              self.x["digest"]: 0.9, self.y["digest"]: 0.8})
            kw.update(boundary_block_hash=block_hash(f"b{n}"), sentinel=None,
                      bench=SweepSession(f"w7d-sess-b{n}", host))
        b = boundary(n)
        s = settle_lane_d(**_rev614_ts(settle_lane_d, b), cfg=self.cfg, boundary=b, pool_dir=self.pools[n - 1],
                          entries=[dict(e) for e in entries],
                          house=dict(house if house is not None else self.h0),
                          forecasts=self.forecasts, uid_of=dict(self.uids),
                          state_path=self.state_path, signer=self.signer, **kw)
        self.pub.boundaries.append(b)
        self.pub.settlements.append(s)
        self.pub.receipts.append(copy.deepcopy(getattr(s, "receipt", None)))
        self.pub.states[b] = self.state_path.read_bytes()
        self.pub.uid_of[b] = dict(self.uids)
        return s


# ── the honest chains (README "Fixture chains") ─────────────────────────────


#    rev6.11 (b): there is no confirmation duel. An entry is judged once, at
#    its first verdict, and a clear that passes A10 and the gate is crowned
#    there. The 48-window floors make b1 a floor wait for X (snapshot 1
#    only), so b2 (snapshots 1 + 2) is X's first, crowning verdict.


def chain_gated(root: Path) -> Chain:
    """b1 X waits on the floors (nothing judged); b2 X's first verdict vs
    H0: clear, gate pass, crowned; b3 nothing."""
    c = Chain(root)
    c.step(1, entries=[c.x], gated=False)
    c.step(2, entries=[c.x])
    c.step(3, entries=[c.x])
    return c


def chain_held(root: Path) -> Chain:
    """b1 floor wait; b2 X's first verdict clears ungated (`bench_hold`,
    nothing crowns, X stays `pending` holding its clear); b3 X re-gated
    without a duel, pass, crowned (a gate result, no verdict)."""
    c = Chain(root)
    c.step(1, entries=[c.x], gated=False)
    c.step(2, entries=[c.x], gated=False)
    c.step(3, entries=[c.x])
    return c


def chain_house_bar(root: Path) -> Chain:
    """X crowned at b2 (as `gated`); Y (revealed after b2, eligible from
    snapshot 2 on: 48 windows at b3) is judged against KING_D = X AND the
    house (a second bar) at b3, its first verdict, and crowned there; X
    becomes a prior king, so the court has two paid slots."""
    c = Chain(root)
    c.step(1, entries=[c.x], gated=False)
    c.step(2, entries=[c.x])
    c.step(3, entries=[c.x, c.y])
    return c


def chain_refresh(root: Path) -> Chain:
    """b1 X floor wait vs H0; the house refreshes to H1 at b2 (the throne
    moves in place while the house holds it), and X's first verdict is
    against H1: clear, gate pass, crowned; b3 nothing."""
    c = Chain(root)
    c.step(1, entries=[c.x], gated=False)
    c.step(2, entries=[c.x], house=c.h1)
    c.step(3, entries=[c.x], house=c.h1)
    return c


CHAINS = {"gated": chain_gated, "held": chain_held, "house_bar": chain_house_bar,
          "refresh": chain_refresh}


# ── doctoring (the lie is re-signed by the validator's key) ─────────────────


def resign(report: dict, key) -> dict:
    """`report` re-signed by `key` over its canonical body (sr25519, 128 hex)."""
    out = copy.deepcopy(report)
    out.pop("signature", None)
    out["signer"] = str(key.ss58_address)
    sig = key.sign(canonical(out))
    out["signature"] = bytes(sig).hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return out


def reseal_rows(report: dict) -> None:
    report["rows_sha256"] = rows_sha(report["rows"])


def doctored(pub: Publication, n: int, mutate, *, key=None, reseal: bool = True
             ) -> tuple[dict[int, dict], list[dict]]:
    """`(reports, receipts)` with settlement n's report mutated in place by
    `mutate(report)`, its rows re-sealed (unless `reseal=False`), re-signed by
    `key` (default: the validator's own key) and its receipt re-issued to
    bind the doctored body (so only the doctored class can disagree)."""
    reports = copy.deepcopy(pub.reports())
    receipts = copy.deepcopy(pub.receipts)
    b = pub.boundaries[n - 1]
    rep = reports[b]
    mutate(rep)
    if reseal:
        reseal_rows(rep)
    rep = resign(rep, key if key is not None else fixture_key("signer"))
    reports[b] = rep
    receipts[n - 1]["report_sha256"] = body_sha(rep)
    return reports, receipts


# ── the module under test ───────────────────────────────────────────────────


def replay_module():
    import importlib

    return importlib.import_module(MODULE)


def replay(pub: Publication, *, reports=None, receipts=None, states=None, uid_of=None,
           cfg=None):
    """`replay_chain` over the publication (the reports directory unless
    `reports` is given)."""
    mod = replay_module()
    kw = {}
    if states is not None:
        kw["states"] = states
    if uid_of is not None:
        kw["uid_of"] = uid_of
    return mod.replay_chain(reports if reports is not None else pub.reports_dir,
                            receipts if receipts is not None else pub.receipts,
                            cfg if cfg is not None else pub.cfg,
                            signer_ss58=pub.signer_ss58, **kw)


def has_path(result, path: str) -> bool:
    """`result.mismatches` holds `path` or a path below it."""
    return any(p == path or p.startswith(path + ".") or p.startswith(path + "[")
               for p in result.mismatches)


def describe(results) -> str:
    return "; ".join(f"{r.boundary}:{r.status}:{r.reason}:{list(r.mismatches)[:6]}"
                     for r in results)


CHAIN_TIMEOUT = 600


def build_all(root: Path) -> dict[str, Chain]:
    return {name: build(Path(root) / name) for name, build in CHAINS.items()}


def save_publication(pub: Publication, out: Path) -> Path:
    """Write the publication to `out` for a subprocess: `receipts.json`,
    `states/<b>.bin`, `uid_of.json`, `meta.json` (signer, boundaries, the
    reports directory). The cfg is rebuilt there from `cfg_over`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipts.json").write_text(json.dumps(pub.receipts))
    (out / "uid_of.json").write_text(json.dumps({str(b): u for b, u in pub.uid_of.items()}))
    sd = out / "states"
    sd.mkdir(exist_ok=True)
    for b, raw in pub.states.items():
        (sd / f"{b}.bin").write_bytes(raw)
    cfg_over: dict = {}     # every honest chain runs on the default fixture config
    (out / "meta.json").write_text(json.dumps({"signer": pub.signer_ss58,
                                               "boundaries": pub.boundaries,
                                               "reports_dir": str(pub.reports_dir),
                                               "cfg_over": cfg_over}))
    return out


# Run in a fresh interpreter: replay a saved publication (with states and
# metagraphs) and print the canonical results plus every loaded cascade/torch
# module. `argv[1]` is the saved publication, `argv[2]` "block-sdk" to make the
# bittensor SDK unimportable first.
REPLAY_SCRIPT = r"""
import dataclasses, json, sys
from collections.abc import Mapping
from pathlib import Path


def plain(v):
    if isinstance(v, Mapping):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v



if len(sys.argv) > 2 and sys.argv[2] == "block-sdk":
    for name in ("bittensor", "bittensor.sp_core", "bittensor_wallet", "substrateinterface"):
        sys.modules[name] = None
from tests.accept.w7d.helpers import lane_d_cfg
import cascade.audit.lane_d_chain as rp
pub = Path(sys.argv[1])
meta = json.loads((pub / "meta.json").read_text())
receipts = json.loads((pub / "receipts.json").read_text())
uid_of = {int(b): u for b, u in json.loads((pub / "uid_of.json").read_text()).items()}
states = {int(p.stem): p.read_bytes() for p in sorted((pub / "states").iterdir())}
res = rp.replay_chain(Path(meta["reports_dir"]), receipts, lane_d_cfg(**meta["cfg_over"]),
                      signer_ss58=meta["signer"], uid_of=uid_of, states=states)
mods = sorted(m for m in sys.modules if m.split(".")[0] in ("cascade", "torch"))
rows = [{f.name: plain(getattr(r, f.name)) for f in dataclasses.fields(r)} for r in res]
print(json.dumps({"results": rows, "modules": mods}, sort_keys=True))
"""


# ── the static import graph (README "Invariant 12") ─────────────────────────


def _module_file(name: str) -> tuple[Path | None, bool]:
    p = REPO.joinpath(*name.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py", True
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py"), False
    return None, False


def module_name(path: Path) -> str:
    rel = Path(path).resolve().relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def direct_imports(name: str) -> set[str]:
    """Every module `name`'s source imports, at ANY depth (function-level and
    `TYPE_CHECKING` imports included), as absolute dotted names; for
    `from a import b` both `a` and `a.b`."""
    import ast

    f, is_pkg = _module_file(name)
    if f is None:
        return set()
    tree = ast.parse(f.read_text(encoding="utf-8"))
    pkg = name if is_pkg else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = str(node.module)
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def dynamic_imports(path: Path) -> list[str]:
    """Calls the static graph cannot follow: `importlib.import_module`,
    `__import__`, `importlib.util` loaders, `exec`/`eval` of text."""
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__", "spec_from_file_location",
                        "module_from_spec", "exec", "eval"):
                out.append(f"{path.name}:{node.lineno}: {name}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            mod = getattr(node, "module", None) or ""
            if "importlib" in names or mod.startswith("importlib"):
                out.append(f"{path.name}:{node.lineno}: importlib")
    return out


def _repo_modules(names) -> set[str]:
    """The cascade modules (files in the repo) among `names` and their parents."""
    out = set()
    for m in names:
        if not (m == "cascade" or m.startswith("cascade.")):
            continue
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if _module_file(sub)[0] is not None:
                out.add(sub)
    return out


def static_closure(starts) -> set[str]:
    """The transitive static import closure of `starts` over repo modules
    (a module implies its parent packages), plus every non-repo top-level
    name imported along the way (e.g. `torch`)."""
    seen: set[str] = set()
    external: set[str] = set()
    todo = list(starts)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        imps = direct_imports(m)
        external |= {i.split(".")[0] for i in imps if not i.startswith("cascade")}
        todo.extend(_repo_modules(imps) | _repo_modules([m]))
    return seen | external


def satellite_modules() -> list[Path]:
    """rev6.18 B2: the chain-rules module (the former satellite replay, now
    `cascade/audit/lane_d_chain.py`)."""
    return [MODULE_PATH] if MODULE_PATH.is_file() else []


def court_shares(cfg, court: list[dict]) -> list[tuple[str, int | None, float]]:
    """§1.4 pay, re-derived here: slot i of the court is worth
    `court_weights[i]` of `share` (rev6.23, owner 2026-11-24: 0.80 / 0.15 /
    0.05), normalised over the court's slots; a slot without a uid burns."""
    raw = [float(w) for w in cfg.court_weights][:len(court)]
    total = sum(raw)
    return [(s["digest"], s["uid"], float(cfg.share) * w / total)
            for s, w in zip(court, raw, strict=True)]


def _rev614_ts(settle, boundary) -> dict:
    """rev6.14 (a): the fixture's on-chain boundary timestamp (far after every
    fixture T + eval_delay_hours, 12 s per block), passed only where
    ``settle_lane_d`` takes it."""
    import inspect

    if "boundary_ts" not in inspect.signature(settle).parameters:
        return {}
    return {"boundary_ts": 2_000_000_000 + 12 * int(boundary)}
