"""Shared spec and fixture builders for the W7b report-signing tests.

The fixture report is a real ``lane-d-report/1`` produced by ``settle_lane_d``
on a small synthetic chain (no torch, numpy only): two pool_d snapshots, one
challenger centred on the targets, a house 3 sd off. The builders are a
trimmed copy of tests/accept/w7a/helpers.py so W7b does not depend on another
WP's test files. README.md states the contract.

Nothing here imports a module in PATHS at import time. Every builder is
called from test bodies only, so on main each test fails inside its own body:
at its first import of a new name from ``cascade.validator.lane_d_report``
(an ``ImportError`` for a path in PATHS) or at an ``assert``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCHEMA_FILE = REPO / "cascade" / "validator" / "schemas" / "lane_d_report_1.json"
CHAIN_TOML = REPO / "chain.toml"

# ── the pinned signing contract (README "Contract") ─────────────────────────

SIGNATURE_SCHEME = "sr25519"
SIGNATURE_PATTERN = r"^[0-9a-f]{128}$"

# W7a's closed top-level field set (tests/accept/w7a/README.md).
TOP_LEVEL_REQUIRED = ("schema", "boundary", "signer", "signature", "scoring_method", "params",
                      "seed", "pool", "house", "king_before", "king_after", "crowned",
                      "prior_kings", "entries", "judged", "skipped", "cohorts", "windows",
                      "verdicts", "rows", "rows_sha256", "pay")
SIGNED_TOP_LEVEL = tuple(f for f in TOP_LEVEL_REQUIRED if f != "signature")

# ── the synthetic chain (copied from W7a's fixture, one settlement used) ─────

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
READY_BLOCK = ARM_BLOCK + 40
REVEAL_BLOCK = ARM_BLOCK + 60

HORIZON = 64
CONTEXT = 4096
HISTORY = 256
N_SOURCES = 6
PER_SOURCE = 8
HOUR = 3600
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * 86_400
SNAPSHOT_START = T0 + 12 * HOUR
FIRST_SEEN_LAG = 1800
CHALLENGER_UID = 6
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)

ARMED = {
    "from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a", "remainder": "burn",
    "contexts": [4096, 8192, 16384], "horizon": HORIZON, "forward_margin_hours": 6.0,
    "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200,
}
# W5a fails closed: a feed missing from `feeds` is `unlisted_feed`, so every
# feed the series name is listed (as in tests/accept/w7a/helpers.py).
FEEDS = {"feed-a": {"mirror_lag_hours": 1.5, "forecast_product": False},
         "src1": {"mirror_lag_hours": None, "forecast_product": False},
         "src2": {"mirror_lag_hours": 1.0, "forecast_product": False},
         "src3": {"mirror_lag_hours": None, "forecast_product": False},
         "src4": {"mirror_lag_hours": 1.0, "forecast_product": False},
         "src5": {"mirror_lag_hours": 1.0, "forecast_product": False}}
# W5a fails closed on a missing `frontier` too (`absent_from_s0`), so every
# series records its S0 as one `[as_of_ts, end_ts]` frontier record: S0 was
# taken 3 h before the snapshot's first target (after the bound, T0 + 6 h)
# and held data up to 1 h before it was taken, so no target is a backfill.
FRONTIER_AS_OF = SNAPSHOT_START - 3 * HOUR
FRONTIER_END_LAG = HOUR


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_bytes(obj) -> bytes:
    """README "Canonical body", re-derived here independently."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def body_of(report: dict) -> bytes:
    return canonical_bytes({k: v for k, v in report.items() if k != "signature"})


def rows_digest(rows) -> str:
    return hashlib.sha256(canonical_bytes(list(rows))).hexdigest()


def reseal(report: dict) -> dict:
    """``report`` with ``rows_sha256`` recomputed over its (mutated) rows."""
    out = copy.deepcopy(report)
    out["rows_sha256"] = rows_digest(out["rows"])
    return out


def keypair_class():
    from bittensor.sp_core import Keypair

    return Keypair


def make_key(directory: Path, name: str):
    """A fresh sr25519 TEST keypair whose secret lives only in ``directory``
    (a tmp dir); never a real wallet."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.uri"
    if not path.exists():
        path.write_text(secrets.token_hex(16))
        path.chmod(0o600)
    return keypair_class().create_from_uri(f"//w7b-{name}-" + path.read_text())


def raw_sign(key, message: bytes) -> str:
    sig = key.sign(message)
    return sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)


def raw_verify(ss58: str, message: bytes, signature_hex: str) -> bool:
    """The sr25519 layer alone (no schema, no signer pin)."""
    from cascade.shared.sr25519 import verify_hotkey

    return verify_hotkey(ss58, message, signature_hex)


def lane_d_cfg(**over):
    from cascade.shared.config.lane_d import parse_lane_d

    return parse_lane_d({**ARMED, **over}, era_blocks_at=lambda _block: ERA)


def _quantiles_around(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def _series(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY, dtype=np.float64)
    out = []
    for j in range(N_SOURCES):
        for k in range(PER_SOURCE):
            sid = f"src{j}-s{k}"
            hist = 10.0 + 5.0 * j + k + 2.0 * np.sin(2 * np.pi * t / 24 + k) \
                + rng.normal(0, 0.5, HISTORY)
            start = SNAPSHOT_START + (j * PER_SOURCE + k) * HOUR
            ts = [int(start + i * HOUR) for i in range(HORIZON)]
            rec = {"series_id": sid, "source": f"src{j}", "history": hist,
                   "target_ts": ts, "first_seen_ts": [x + FIRST_SEEN_LAG for x in ts],
                   "frontier": [[FRONTIER_AS_OF, FRONTIER_AS_OF - FRONTIER_END_LAG]]}
            if j == 0:
                rec["feed"] = "feed-a"
            out.append(rec)
    return out


def _write_pool(directory: Path, series: list[dict], targets: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "pool_d/1", "snapshot": "w7b-1", "horizon": HORIZON, "series": [],
            "feeds": copy.deepcopy(FEEDS),
            "first_seen_roots": sorted([text_digest("w7b first-seen 2026-09-21"),
                                        text_digest("w7b first-seen 2026-09-22")])}
    for s in series:
        rec = {"series_id": s["series_id"], "source": s["source"], "freq": "H",
               "seasonal_period": 24, "first_target_ts": s["target_ts"][0],
               "end_ts": s["target_ts"][-1], "target_ts": s["target_ts"],
               "first_seen_ts": s["first_seen_ts"]}
        for k in ("feed", "frontier"):
            if k in s:
                rec[k] = s[k]
        meta["series"].append(rec)
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    np.savez(directory / "hist.npz", **{s["series_id"]: s["history"] for s in series})
    np.savez(directory / "tgt.npz", **{s["series_id"]: np.asarray(targets[s["series_id"]])
                                       for s in series})
    return directory


_CACHE: dict[str, dict] = {}


def signed_report(tmp: Path) -> dict:
    """One settlement of the synthetic chain, signed by a tmp test key.
    Built once per scratch dir; called from test bodies only.

    Returns ``{"report", "signer", "other", "cfg"}``: the signed report, the
    signing keypair (stands in for the pinned main validator hotkey) and a
    second, unrelated test keypair."""
    key = str(tmp)
    if key not in _CACHE:
        from cascade.validator.lane_d import settle_lane_d

        digest = text_digest("w7b entry 0")
        house = text_digest("w7b house")
        rng = np.random.default_rng(7)
        series = _series(101)
        targets, fc = {}, {digest: {}, house: {}}
        for s in series:
            sd = float(np.std(s["history"]))
            target = np.resize(s["history"][-24:], HORIZON) + rng.normal(0, 0.3, HORIZON)
            targets[s["series_id"]] = target
            wid = f"{s['series_id']}:{int(s['target_ts'][0])}:h{HORIZON}"
            fc[digest][wid] = _quantiles_around(
                target + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
            fc[house][wid] = _quantiles_around(target + 3.0 * sd, 0.3 * sd)
        pool = _write_pool(tmp / "pool_d_1", series, targets)
        keys = tmp / "keys"
        signer = make_key(keys, "validator")
        other = make_key(keys, "other")
        hotkey = make_key(keys, "challenger").ss58_address
        coldkey = make_key(keys, "coldkey").ss58_address
        e = {"hotkey": hotkey, "digest": digest, "context": CONTEXT,
             "coldkey_at_ready": coldkey, "ready_block": READY_BLOCK,
             "reveal_block": REVEAL_BLOCK, "reveal_ts": T0}
        state = tmp / "state" / "lane_d_state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        cfg = lane_d_cfg()
        s = settle_lane_d(cfg=cfg, boundary=BOUNDARY_1, pool_dir=pool, entries=[e],
                          house={"digest": house, "reveal_ts": HOUSE_T}, forecasts=fc,
                          uid_of={hotkey: CHALLENGER_UID}, state_path=state, signer=signer)
        _CACHE[key] = {"report": s.report, "signer": signer, "other": other, "cfg": cfg}
    built = _CACHE[key]
    return {**built, "report": copy.deepcopy(built["report"])}


def mutate(value):
    """A different JSON value of a similar shape (any change will do: the
    signature must cover it)."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return text_digest(value)
        return value + "x"
    if value is None:
        return text_digest("w7b mutated null")
    if isinstance(value, list):
        return value[:-1] if value else [text_digest("w7b appended")]
    if isinstance(value, dict):
        out = copy.deepcopy(value)
        if out:
            k = sorted(out)[0]
            out[k] = mutate(out[k])
        else:
            out["w7b"] = 1
        return out
    raise TypeError(type(value).__name__)


def get_path(obj, path: tuple):
    for p in path:
        obj = obj[p]
    return obj


def set_path(obj, path: tuple, value) -> None:
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def reversed_keys(obj):
    """``obj`` with every dict's keys inserted in reverse-sorted order."""
    if isinstance(obj, dict):
        return {k: reversed_keys(obj[k]) for k in sorted(obj, reverse=True)}
    if isinstance(obj, list):
        return [reversed_keys(v) for v in obj]
    return obj


def chain_raw(validator_hotkey: str) -> dict:
    """The repo chain.toml parsed, with ``[manifest] validator_hotkey`` set.

    The repo config runs ``[v2] bench_source = "validator"``, under which the
    loader (``check_bench_source``) requires a non-empty ``[manifest]
    validator_hotkey`` to equal ``[netconfig] owner_hotkey`` (both name the
    main validator). A non-empty pin is therefore mirrored into
    ``owner_hotkey`` so the config stays consistent; an empty pin leaves the
    repo's ``owner_hotkey`` in place (the loader accepts an unset manifest
    pin), so the refusal under test comes from the W7b code, not the loader.
    """
    import tomllib

    raw = tomllib.loads(CHAIN_TOML.read_text(encoding="utf-8"))
    raw["manifest"]["validator_hotkey"] = validator_hotkey
    if validator_hotkey.strip():
        raw.setdefault("netconfig", {})["owner_hotkey"] = validator_hotkey
    return raw
