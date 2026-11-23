"""Fixtures for the W3b3 acceptance tests (sentinel sweep + publish).

Nothing here imports ``cascade.validator.lane_d_sentinel`` (the module under
test): every test imports it in its body, so on main the failure is a
``ModuleNotFoundError`` for a PATHS module inside the test.

The crowning chain is REAL settlements (``settle_lane_d`` on main, no
hand-written state), built from the W5c fixture conventions (copied, not
imported, so this WP stands alone):

- BOUNDARY_1: no entry is listed yet; nothing is judged (H0 holds KING_D);
- BOUNDARY_2: K1 enters, clears at its first verdict and is crowned
  (previous KING_D = H0, the house);
- BOUNDARY_3: Y is not listed yet; nothing is judged (K1 holds KING_D);
- BOUNDARY_4: Y enters, clears at its first verdict against K1 (and the
  house) and is crowned (previous KING_D = K1, a miner).

Plan rev6.11 (b) drops the confirmation duel: a crown is decided at the
first verdict (first clear ∧ A10 ∧ the §1.3b gate). The chain lists each
challenger from the settlement that crowns it, so the crowns stay at
BOUNDARY_2 and BOUNDARY_4.

The only stand-ins are the two external services: the bench pod
(:class:`RecordingHost`, recorded per-item sentinel rows behind a real
``SweepSession``) and the public object store (:class:`MemoryStore`, the
``put_text`` / ``get_text`` duck type of ``cascade.shared.hippius.S3Store``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import NormalDist

import numpy as np

SCHEMA = "lane-d-sentinel/1"
SENTINEL_SUITES = ("gift", "boom", "time", "fev")
SESSION = "valpriv-4090/2026-11-22T03:00:00Z/sentinel"
N_ITEMS = 40

# ── the real crowning chain (W5c conventions) ────────────────────────────────

HOUR = 3600
DAY = 86_400
HORIZON = 64
HISTORY = 256
MARGIN_H = 6.0
T0 = 1_790_000_000
HOUSE_T = T0 - 2 * DAY
ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
BOUNDARY_2 = ARM_BLOCK + 2 * SETTLEMENT
BOUNDARY_3 = ARM_BLOCK + 3 * SETTLEMENT
BOUNDARY_4 = ARM_BLOCK + 4 * SETTLEMENT
ERA_START_1 = ARM_BLOCK          # BOUNDARY_1..3 settle in this era
ERA_START_2 = ARM_BLOCK + ERA    # BOUNDARY_4 starts the next era

SOURCES = ("src0", "src1", "src2", "src3")
FEED_LAG_HOURS = 1.0
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)
QUALITY = {"sharp": (0.005, 0.1, 0.0), "good": (0.02, 0.2, 0.0),
           "fair": (0.05, 0.3, 0.0), "bad": (0.0, 0.3, 3.0)}


def lane_d_cfg(**over):
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": [4096, 8192, 16384], "horizon": HORIZON,
           "forward_margin_hours": MARGIN_H, "max_mirror_lag_hours": 6.0,
           "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def _bound(t: int) -> int:
    return int(t + MARGIN_H * HOUR)


def band_start(t: int) -> int:
    return _bound(t) + 3 * HOUR


def _window_id(series_id: str, target_ts) -> str:
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def _record(series_id: str, source: str, start: int) -> dict:
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
            out.append(_record(f"{tag}-{src}-{k}", src, start + i * 60))
    return out


def _target_values(series_id: str, start: int) -> tuple[np.ndarray, np.ndarray]:
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
        h, y = _target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]], tgt[r["series_id"]] = h, y
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def model_forecasts(records, quality: str, name: str) -> dict:
    noise, spread, offset = QUALITY[quality]
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    out = {}
    for r in records:
        _, y = _target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        med = y + offset * sd + rng.normal(0, noise * sd, HORIZON)
        out[_window_id(r["series_id"], r["target_ts"])] = np.stack(
            [med + spread * sd * z for z in _Z], axis=-1)
    return out


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return _keypair_class().create_from_uri(f"//w3b3-{name}")


def entry(name: str, *, reveal_ts: int, ready_block: int, reveal_block: int) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest(f"w3b3 {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": int(ready_block), "reveal_block": int(reveal_block),
            "reveal_ts": int(reveal_ts)}


H0 = {"digest": digest("w3b3 house-0"), "reveal_ts": HOUSE_T}
K1 = entry("k1", reveal_ts=T0, ready_block=ARM_BLOCK + 40, reveal_block=ARM_BLOCK + 60)
Y = entry("y", reveal_ts=T0 + 2 * DAY, ready_block=ARM_BLOCK + 2 * SETTLEMENT + 40,
          reveal_block=ARM_BLOCK + 2 * SETTLEMENT + 60)


def _uids(*entries) -> dict:
    return {e["hotkey"]: 10 + i for i, e in enumerate(entries)}


def _settle(cfg, boundary, pool_dir, entries, fc, state_path, prior=()):
    """`settle_lane_d`, with the W5e gate inputs when it accepts them
    (`gate_inputs`, below): the chain crowns at BOUNDARY_2 and BOUNDARY_4."""
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    kw.update(gate_inputs(boundary, H0["digest"]))
    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(H0),
                         forecasts=fc, uid_of=_uids(*entries), state_path=state_path,
                         signer=fixture_key("signer"), **kw)


class BrokenChain(dict):
    """A chain that did not crown where the fixture needs it: every read
    fails the reading test with the fixture's ``AssertionError``."""

    def __init__(self, why: str):
        super().__init__()
        self.why = why

    def __getitem__(self, key):
        raise AssertionError(self.why)

    def get(self, key, default=None):
        raise AssertionError(self.why)


def build_chain(root: Path, after=None) -> dict:
    """Run BOUNDARY_1..4 under ``root``. Returns ``{boundary: (report, pay)}``
    and the state file path under key ``"state"``; each report is also saved
    as ``root/report-<boundary>.json`` (canonical JSON) and the state after
    each boundary as ``root/state-<boundary>.json``. ``after(boundary,
    report, pay)``, when given, runs right after each settlement (where the
    validator runs its post-settlement steps)."""
    cfg = lane_d_cfg()
    state = root / "state" / "lane_d_state.json"
    out: dict = {"state": state}
    p1 = group("p1", 6, band_start(K1["reveal_ts"]))
    p2 = group("p2", 6, band_start(K1["reveal_ts"]) + DAY)
    p3 = group("p3", 6, band_start(Y["reveal_ts"]))
    p4 = group("p4", 6, band_start(Y["reveal_ts"]) + DAY)
    # rev6.11 (b): a challenger is crowned at its first verdict, so each is
    # listed from the settlement that crowns it.
    plan = ((BOUNDARY_1, p1, []), (BOUNDARY_2, p2, [K1]),
            (BOUNDARY_3, p3, [K1]), (BOUNDARY_4, p4, [K1, Y]))
    for boundary, recs, entries in plan:
        pool = write_pool(root / f"pool-{boundary}", recs, f"w3b3-{boundary}")
        fc = {H0["digest"]: model_forecasts(recs, "bad", "house-0"),
              K1["digest"]: model_forecasts(recs, "fair", "k1"),
              Y["digest"]: model_forecasts(recs, "sharp", "y")}
        s = _settle(cfg, boundary, pool, entries, fc, state)
        want = {BOUNDARY_2: K1["digest"], BOUNDARY_4: Y["digest"]}.get(boundary)
        if s.report is None or s.report["crowned"] != want:
            # Reported inside each test that reads the chain (an assertion in
            # the test body, not a fixture error).
            return BrokenChain(f"fixture: boundary {boundary} must crown {want} at the "
                               f"first verdict (rev6.11 (b)); crowned "
                               f"{s.report and s.report['crowned']}")
        out[boundary] = (s.report, s.pay)
        if after is not None:
            after(boundary, s.report, s.pay)
        (root / f"report-{boundary}.json").write_text(canonical(s.report))
        shutil.copyfile(state, root / f"state-{boundary}.json")
    return out


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ── era hashes and the drawn suites ──────────────────────────────────────────


def era_hash(tag: str) -> str:
    return "0x" + hashlib.sha256(f"w3b3 era {tag}".encode()).hexdigest()


def hash_drawing(suite: str, suites: Iterable[str] = SENTINEL_SUITES) -> str:
    """The first ``era_hash(i)`` whose §16 draw is ``suite``."""
    from cascade.shared.sentinel import sentinel_for_era

    names = tuple(suites)
    for i in range(10_000):
        h = era_hash(str(i))
        if sentinel_for_era(h, names) == suite:
            return h
    raise AssertionError(f"fixture: no era hash draws {suite!r}")


def gate_suite_avoiding(sentinel: str) -> str:
    """The lane D gate suite W3b1 draws for a boundary hash with ``sentinel``
    held out (never equal to it)."""
    from cascade.validator.lane_d_gate import gate_suite_for

    return gate_suite_for(era_hash("boundary"), gate_suites=SENTINEL_SUITES,
                          sentinel=sentinel)


# ── the bench pod stand-in ───────────────────────────────────────────────────


def rows(suite: str, tag: str, *, factor: float = 1.0, n: int = N_ITEMS) -> list[dict]:
    """Per-item raw CRPS/MASE for one checkpoint on ``suite``: a seeded base
    times ``factor`` (above 1 = worse) times per-item noise seeded by ``tag``."""
    base = random.Random(f"w3b3 base {suite}")
    noise = random.Random(f"w3b3 noise {suite} {tag}")
    out = []
    for i in range(n):
        c, m = math.exp(base.gauss(-0.7, 0.6)), math.exp(base.gauss(0.1, 0.4))
        out.append({"item": f"{suite}/task-{i:03d}",
                    "crps": c * factor * math.exp(noise.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(noise.gauss(0.0, 0.01))})
    return out


def geomean(values: Iterable[float]) -> float:
    v = [float(x) for x in values]
    return math.exp(sum(math.log(x) for x in v) / len(v))


class RecordingHost:
    """The bench pod (an external service): returns recorded per-item rows
    for the digests and suites it is asked for and logs every call as
    ``(digests, suites)``. ``table[digest][suite]`` is a list of rows or a
    full suite entry (``{"status", "rows", ...}``); ``raises`` makes every
    call fail like an unreachable pod."""

    def __init__(self, table: Mapping[str, Mapping[str, object]],
                 raises: BaseException | None = None):
        self.table = copy.deepcopy({d: dict(s) for d, s in table.items()})
        self.raises = raises
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        if self.raises is not None:
            raise self.raises
        out: dict = {}
        for d in digests:
            per = {}
            for s in suites:
                v = self.table.get(d, {}).get(s)
                if v is None:
                    continue
                per[s] = (copy.deepcopy(v) if isinstance(v, dict)
                          else {"status": "ok", "rows": copy.deepcopy(v)})
            out[d] = per
        return out


def host_for(king: str, previous: str, house: str, *, king_factor: float = 0.95,
             previous_factor: float = 1.0, house_factor: float = 1.05,
             suites: Iterable[str] = SENTINEL_SUITES) -> RecordingHost:
    table: dict = {}
    for dg, f in ((previous, previous_factor), (house, house_factor), (king, king_factor)):
        table[dg] = {s: rows(s, dg, factor=f) for s in suites}
    return RecordingHost(table)


def session(host, session_id: str = SESSION, cache=None):
    from cascade.validator.lane_d_bench import SweepSession

    return SweepSession(session_id, host, cache)


# ── the public object store stand-in ─────────────────────────────────────────


class MemoryStore:
    """The public bucket (an external service), with the ``put_text`` /
    ``get_text`` duck type of ``cascade.shared.hippius.S3Store``: an absent
    key raises ``ObjectNotFound``. Every put is logged."""

    def __init__(self, fail_put: BaseException | None = None,
                 fail_get: BaseException | None = None):
        self.objects: dict[str, str] = {}
        self.puts: list[tuple[str, str]] = []
        self.fail_put = fail_put
        self.fail_get = fail_get

    def put_text(self, key: str, text: str, *, content_type: str | None = None,
                 acl: str | None = None) -> None:
        if self.fail_put is not None:
            raise self.fail_put
        self.puts.append((key, text))
        self.objects[key] = text

    def get_text(self, key: str) -> str:
        from cascade.shared.hippius import ObjectNotFound

        if self.fail_get is not None:
            raise self.fail_get
        if key not in self.objects:
            raise ObjectNotFound(f"s3_get_missing: {key}")
        return self.objects[key]


def roles_of(report: dict) -> dict:
    """``{role: digest}`` of a crowning report: the new king, the KING_D it
    dethroned and the house."""
    return {"king": report["crowned"], "previous_king": report["king_before"]["digest"],
            "house": report["house"]["digest"]}


def run(report, *, era_hash_: str, bench, store, era_block: int = ERA_START_2,
        suites: Iterable[str] = SENTINEL_SUITES, gate_suite: str | None = None):
    """Call the module under test (imported here, at call time, so on main a
    test fails inside its body) with the era's draw from ``era_hash_``."""
    from cascade.shared.sentinel import sentinel_for_era
    from cascade.validator import lane_d_sentinel

    names = tuple(suites)
    if gate_suite is None:
        gate_suite = gate_suite_avoiding(sentinel_for_era(era_hash_, names))
    return lane_d_sentinel.sweep_and_publish(
        report, era_block=era_block, era_block_hash=era_hash_, sentinel_suites=names,
        gate_suite=gate_suite, bench=bench, store=store)


def strict_loads(text: str):
    """``json.loads`` that refuses NaN / Infinity."""
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(text, parse_constant=bad)


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
        h = hashlib.sha256(f"w3b3 bench {suite} {i}".encode()).digest()
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

    return "0x" + hashlib.sha256(f"w3b3 block {int(boundary)}".encode()).hexdigest()


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

        kw["bench"] = SweepSession(f"w3b3-sess-{int(boundary)}", HealthyBench(house_digest))
    return kw
