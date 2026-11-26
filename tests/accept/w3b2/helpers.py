"""Recorded suite outputs for the W3b2 acceptance tests (no GPU, no pod).

Nothing here imports ``cascade.validator.lane_d_bench`` (the module under
test): every test imports it in its body, so on main the failure is a
``ModuleNotFoundError`` for a PATHS module inside the test.

A *sweep record* is one checkpoint's recorded output of one host session::

    {"session": "<session id>", "digest": "<64 hex>",
     "suites": {"<suite>": {"status": "ok", "rows": [{"item", "crps", "mase"}, ...]}}}

``suites`` has the shape :func:`cascade.eval.benchmarks.item_rows_from_report`
returns, so a real sidecar report maps onto it unchanged. The values are
deterministic (seeded ``random.Random``), so every check is exact.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Iterable, Mapping

SUITES = ("gift", "boom", "time", "fev")
GATE_SUITE = "boom"
SENTINEL = "gift"
SESSION = "valpriv-4090/2026-11-22T03:00:00Z/1"
OTHER_SESSION = "valpriv-4090/2026-11-21T03:00:00Z/1"
ALPHA = 0.05
TAU = 0.01
B = 2000
SEED = "lane-d/1:9612000:w3b2"
N_ITEMS = 60
REFS = ("king_d", "house")
METRICS = ("crps", "mase")


def digest(name: str) -> str:
    return hashlib.sha256(f"cascade w3b2 checkpoint {name}".encode()).hexdigest()


CAND = digest("candidate")
KING = digest("king_d")
HOUSE = digest("house")
DIGESTS = {"candidate": CAND, "king_d": KING, "house": HOUSE}


def _base(suite: str, n: int) -> list[tuple[str, float, float]]:
    rng = random.Random(f"w3b2 base {suite}")
    return [(f"{suite}/task-{i:03d}", math.exp(rng.gauss(-0.7, 0.6)),
             math.exp(rng.gauss(0.1, 0.4))) for i in range(n)]


def rows(suite: str, tag: str, *, crps: float = 1.0, mase: float = 1.0,
         noise: float = 0.01, n: int = N_ITEMS, scale: float = 1.0) -> list[dict]:
    """Per-item raw CRPS/MASE: the suite's base values times ``crps`` / ``mase``
    (a ratio above 1 = this checkpoint is worse), times ``scale`` (a host-wide
    shift), times per-item noise seeded by ``tag``."""
    rng = random.Random(f"w3b2 noise {suite} {tag}")
    out = []
    for item, c, m in _base(suite, n):
        out.append({"item": item,
                    "crps": c * crps * scale * math.exp(rng.gauss(0.0, noise)),
                    "mase": m * mase * scale * math.exp(rng.gauss(0.0, noise))})
    return out


def record(session: str, dg: str, suites: Mapping[str, list[dict]]) -> dict:
    return {"session": session, "digest": dg,
            "suites": {s: {"status": "ok", "rows": copy.deepcopy(r)} for s, r in suites.items()}}


def triple(*, king: tuple[float, float] = (1.0, 1.0), house: tuple[float, float] = (1.0, 1.0),
           sentinel_king: tuple[float, float] = (1.0, 1.0),
           sentinel_house: tuple[float, float] = (1.0, 1.0),
           noise: float = 0.01, scale: float = 1.0, session: str = SESSION,
           suite: str = GATE_SUITE, sentinel: str | None = SENTINEL,
           n: int = N_ITEMS) -> tuple[dict, dict, dict]:
    """``(candidate, king_d, house)`` records from one session. The factors
    are the REFERENCE's CRPS/MASE relative to the candidate: ``(1.05, 1.05)``
    means the reference is 5% worse (the candidate improves on it),
    ``(0.95, 0.95)`` that the reference is 5% better (the candidate
    regresses against it)."""
    def one(tag: str, gate: tuple[float, float], sen: tuple[float, float], dg: str) -> dict:
        suites = {suite: rows(suite, tag, crps=gate[0], mase=gate[1], noise=noise, n=n,
                              scale=scale)}
        if sentinel is not None:
            suites[sentinel] = rows(sentinel, tag, crps=sen[0], mase=sen[1], noise=noise,
                                    n=n, scale=scale)
        return record(session, dg, suites)

    return (one("candidate", (1.0, 1.0), (1.0, 1.0), CAND),
            one("king_d", king, sentinel_king, KING),
            one("house", house, sentinel_house, HOUSE))


def paired_values(ref: dict, cand: dict, suite: str, metric: str
                  ) -> tuple[list[float], list[float]]:
    """The ref/candidate values of ``metric`` paired by item id, sorted by id."""
    r = {x["item"]: x[metric] for x in ref["suites"][suite]["rows"]}
    c = {x["item"]: x[metric] for x in cand["suites"][suite]["rows"]}
    ids = sorted(set(r) & set(c))
    return [float(r[i]) for i in ids], [float(c[i]) for i in ids]


def lcb_reference(ref: dict, cand: dict, suite: str, metric: str, *, alpha: float = ALPHA,
                  b: int = 40_000) -> float:
    """The repo's paired LCB (``1 - geomean(cand)/geomean(ref)``, lower is
    better) at quantile ``alpha``, computed independently with many bags."""
    from cascade.eval.expert_lane import paired_geomean_lcb

    k, c = paired_values(ref, cand, suite, metric)
    return paired_geomean_lcb(k, c, alpha=alpha, B=b, seed="w3b2 reference")


def gate_kwargs(**over) -> dict:
    kw = {"suite": GATE_SUITE, "alpha": ALPHA, "tau_public": TAU, "B": B, "seed": SEED,
          "sentinel": SENTINEL}
    kw.update(over)
    return kw


class RecordingHost:
    """A stand-in for the bench pod (an external service): returns the
    recorded suite outputs for the digests it is asked for and logs every
    sweep call as ``(digests, suites)``."""

    def __init__(self, table: Mapping[str, Mapping[str, list[dict]]]):
        self.table = {d: dict(s) for d, s in table.items()}
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests: Iterable[str], suites: Iterable[str]) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok", "rows": copy.deepcopy(self.table[d][s])}
                    for s in suites if s in self.table[d]} for d in digests}


def host_table(*digests: str) -> dict:
    return {d: {s: rows(s, d) for s in SUITES} for d in digests}
