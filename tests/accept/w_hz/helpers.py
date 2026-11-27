"""W-HZ fixtures: band ``pool_d/1`` snapshots (a stored target region with
its ``clk.npz`` clocks), per-window forecasts at each window's own horizon,
the settle call, and the OLD (pre-band, deployed) checker run in a
subprocess from a ``git archive`` of a deployed release."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ERA = 3600
ARM_BLOCK = 2700 * ERA
SETTLEMENT = 900
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
HASH_A = "0x" + hashlib.sha256(b"w_hz boundary A").hexdigest()
HASH_B = "0x" + hashlib.sha256(b"w_hz boundary B").hexdigest()
HOUR, DAY, MINUTE = 3600, 86_400, 60
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY
HISTORY = 256          # history points before every series' region
CHALLENGER_UID = 6
FIXED_H = 64
# Deployed releases other validators run (owner 2026-11-27); their lane D
# checker (cascade/audit/lane_d.py) is byte-identical across the three.
OLD_RELEASES = ("09f19baa", "a4600dfe", "00f3139c")

SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)

# GIFT-Eval (SalesforceAIResearch/gift-eval src/gift_eval/data.py @1b520f9):
# PRED_LENGTH_MAP {M:12, W:8, D:30, H:48, T:48, S:60}, M4_PRED_LENGTH_MAP
# {Q:8, A:6} (GIFT's quarterly / yearly series are M4's), Term.multiplier
# short 1 / medium 10 / long 15; patch-aligned up (32), capped at 2048.
GIFT_TABLE = {"S": {"short": 64, "medium": 608, "long": 928},
              "T": {"short": 64, "medium": 480, "long": 736},
              "H": {"short": 64, "medium": 480, "long": 736},
              "D": {"short": 32, "medium": 320, "long": 480},
              "W": {"short": 32, "medium": 96, "long": 128},
              "M": {"short": 32, "medium": 128, "long": 192},
              "Q": {"short": 32, "medium": 96, "long": 128},
              "A": {"short": 32, "medium": 64, "long": 96}}

# The band config the settlement fixtures run (small floors).
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}
BAND_RAW = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
            "remainder": "burn", "horizon": FIXED_H, "min_windows": 9, "min_clusters": 3,
            "max_source_share": 0.5, "bootstrap_B": 200, "max_windows_per_settlement": 60,
            "min_fresh_window_share": 0.0, "min_cell_windows": 3,
            "horizon_bands": ["short", "medium", "long"], "max_horizon": 2048,
            "band_slots": 1}
BAND_KEYS = ("horizon_bands", "max_horizon", "band_slots", "min_cell_windows")


def need(module: str, *attrs: str):
    """The module, asserting it exists and carries ``attrs`` (so a test
    fails on the pre-band code by assertion, not by an import error)."""
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise AssertionError(f"{module} is missing (per-window horizons not built): {e}") \
            from None
    missing = [a for a in attrs if not hasattr(mod, a)]
    assert not missing, f"{module} lacks {missing} (per-window horizons not built)"
    return mod


def takes(fn, *params: str) -> None:
    got = set(inspect.signature(fn).parameters)
    assert set(params) <= got, f"{fn.__qualname__} takes no {sorted(set(params) - got)}"


def band_cfg(**over):
    from cascade.shared.config.lane_d import parse_lane_d

    cfg = parse_lane_d({**BAND_RAW, **LIVE_ONLY, **over}, era_blocks_at=lambda _block: ERA)
    assert tuple(getattr(cfg, "horizon_bands", ())) == tuple(
        over.get("horizon_bands", BAND_RAW["horizon_bands"])), \
        "[lane_d] horizon_bands is not parsed (per-window horizons not built)"
    return cfg


def old_raw(**over) -> dict:
    """The same [lane_d] table as a follower on a pre-band release holds it
    (no band keys: its deployed chain.toml has none)."""
    raw = {**BAND_RAW, **LIVE_ONLY, **over}
    return {k: v for k, v in raw.items() if k not in BAND_KEYS}


# ── band pool_d snapshots ───────────────────────────────────────────────────


def write_npz(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]),
                                      allow_pickle=False)
            zf.writestr(info, buf.getvalue())


def values(series_id: str, n: int, period: int = 60) -> np.ndarray:
    seed = int(hashlib.sha256(series_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return 20.0 + 3.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)


def band_series(series_id: str, source: str, *, freq: str, step: int, region: int,
                start: int, seen_lag: int = 30, feed: str | None = None,
                bound: int) -> dict:
    """One series record + its arrays: ``region`` target-region points from
    ``start`` (data time, ``step`` s apart), first seen ``seen_lag`` s later,
    ``HISTORY`` history points before the region, and 3-hourly frontier
    records (``bound`` is unused: kept for call-site clarity)."""
    ts = [int(start + i * step) for i in range(region)]
    seen = [t + seen_lag for t in ts]
    full = values(series_id, HISTORY + region)
    rec = {"series_id": series_id, "source": source, "feed": feed or source, "freq": freq,
           "seasonal_period": 1, "target_ts": ts[-FIXED_H:], "first_seen_ts": seen[-FIXED_H:],
           "first_target_ts": ts[-FIXED_H], "end_ts": ts[-1],
           "history_len": HISTORY + region - FIXED_H, "target_points": region,
           # earlier 3-hourly snapshots, each holding the series up to a minute
           # before its as_of (S0 is the first one after the bound)
           "frontier": [[t, t - 60] for t in range(start - 48 * 3600, ts[-1], 3 * 3600)]}
    return {"rec": rec, "hist": full[:-FIXED_H].astype(np.float32),
            "tgt": full[-FIXED_H:].astype(np.float32),
            "clk": np.asarray([ts, seen], dtype=np.int64)}


def write_band_pool(directory: Path, series: list[dict], *, feeds: dict, snapshot: str = "hz",
                    as_of: int | None = None, clocks: bool = True) -> Path:
    """A ``pool_d/1`` band snapshot (``clocks=False``: the same arrays with
    no region clocks, i.e. a pre-band snapshot whose region is its target)."""
    directory.mkdir(parents=True, exist_ok=True)
    recs = [dict(s["rec"]) for s in series]
    if not clocks:
        for r in recs:
            r.pop("target_points")
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": FIXED_H,
            "history_length": 16384, "feeds": feeds, "series": recs,
            "as_of_ts": as_of or max(s["rec"]["end_ts"] for s in series) + HOUR,
            "first_seen_roots": [], "embargo_hours_global": 48.0}
    write_npz(directory / "hist.npz", {s["rec"]["series_id"]: s["hist"] for s in series})
    write_npz(directory / "tgt.npz", {s["rec"]["series_id"]: s["tgt"] for s in series})
    if clocks:
        write_npz(directory / "clk.npz", {s["rec"]["series_id"]: s["clk"] for s in series})
        meta["clocks"] = {"file": "clk.npz", "sha256": hashlib.sha256(
            (directory / "clk.npz").read_bytes()).hexdigest()}
        meta["horizon_bands"] = ["short", "medium", "long"]
        meta["max_horizon"] = 2048
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    return directory


def feeds_for(series: list[dict], embargo: float = 0.0) -> dict:
    return {f: {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": embargo}
            for f in sorted({s["rec"]["feed"] for s in series})}


def minute_fleet(*, bound: int, sources: int = 4, per_source: int = 6,
                 freq: str = "min", step: int = MINUTE, region: int = 736) -> list[dict]:
    """Minutely series (bands 64 / 480 / 736) whose whole region follows
    ``bound``."""
    out = []
    for s in range(sources):
        for k in range(per_source):
            out.append(band_series(f"hz-{freq}-s{s}-{k}", f"src{s}", freq=freq, step=step,
                                   region=region, start=bound + HOUR + (s * per_source + k) * 7,
                                   bound=bound))
    return out


# ── forecasts and the settle call ───────────────────────────────────────────


def _quantiles(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def forecasts(pools: list[dict], challengers, house: str, *, house_off: float = 3.0) -> dict:
    """Every pool window's forecast at its OWN ``h`` (``(h, 9)``)."""
    fc = {d: {} for d in [*challengers, house]}
    rng = np.random.default_rng(11)
    for p in pools:
        for w in p["windows"]:
            y = np.asarray(w["target"], dtype=np.float64)
            sd = float(np.std(y)) or 1.0
            for d in challengers:
                fc[d][w["window_id"]] = _quantiles(y + rng.normal(0, 0.05 * sd, y.size), 0.3 * sd)
            fc[house][w["window_id"]] = _quantiles(y + house_off * sd, 0.3 * sd)
    return fc


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair.create_from_uri(f"//w_hz-{name}")


def entry(name: str, *, reveal_ts: int = T0) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w_hz {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": ARM_BLOCK + 40, "reveal_block": ARM_BLOCK + 60,
            "reveal_ts": int(reveal_ts)}


def house() -> dict:
    return {"digest": digest("w_hz house"), "reveal_ts": HOUSE_T}


def settle(cfg, boundary, pool_dir, entries, house_rec, fc, uid_of, state_path, block_hash,
           **kw):
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house_rec),
                         forecasts=fc, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"), boundary_block_hash=block_hash,
                         boundary_ts=2_000_000_000 + 12 * int(boundary), **kw)


# ── the OLD checker (a deployed release), in a subprocess ───────────────────


def old_tree(tmp: Path, release: str = OLD_RELEASES[0]) -> Path:
    """``cascade/`` of ``release`` (``git archive``; CI checks out full
    history) unpacked under ``tmp``."""
    raw = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", release, "cascade"],
                         check=True, capture_output=True).stdout
    dest = tmp / f"old-{release}"
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(dest, filter="data")
    return dest


OLD_CHECK = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import cascade
assert cascade.__file__.startswith(sys.argv[1]), cascade.__file__
from cascade.shared.config.lane_d import parse_lane_d
from cascade.audit.lane_d import lane_d_findings, refusal_class
raw = json.loads(sys.argv[3])
cfg = parse_lane_d(raw, era_blocks_at=lambda _b: int(sys.argv[5]))
report = json.loads(open(sys.argv[2]).read())
f = lane_d_findings(report, cfg, signer_hotkey=sys.argv[4])
print(json.dumps({"burn": f.burn, "alerts": f.alerts, "transient": f.transient,
                  "findings": f.findings, "n_rows": f.n_rows, "n_cohorts": f.n_cohorts}))
"""


def run_old_checker(tree: Path, report: dict, raw: dict, signer: str, tmp: Path) -> dict:
    path = tmp / "report.json"
    path.write_text(json.dumps(report))
    out = subprocess.run([sys.executable, "-I", "-c", OLD_CHECK, str(tree), str(path),
                          json.dumps(raw), signer, str(ERA)],
                         capture_output=True, text=True, cwd=str(tmp))
    assert out.returncode == 0, f"the old checker crashed: {out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])
