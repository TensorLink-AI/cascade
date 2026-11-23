"""Shared fixtures for the W3b6b acceptance tests (README.md has the contract).

From docs/plans/LANE_D_PLAN.md, rev6.13 "W3b6 spec" and the W3b6b row: each
swept gate result binds its three sweep records by sha256 inside the signed
body (``rows``), the records are persisted next to the report and published
content-addressed under ``bench_v/lane_d/<boundary>/``, and W7d's replay
re-derives the four LCBs from those published bytes.

The fixture is W8b's production step (W7a's synthetic chain, W8b's fake bench
pod and bucket): X is gated and crowned at B2. No chain, storage, pod, torch
or GPU. Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tests.accept.w8b import helpers as w8b

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROLES = ("candidate", "king_d", "house")
ROWS_PREFIX = "bench_v/lane_d"
SEED_TAG = "cascade|lane_d|paired_gate|v1"
B1, B2, B3 = w8b.B1, w8b.B2, w8b.B3


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def bench_rows_dir(state_path: Path) -> Path:
    """README: ``<state dir>/<state stem>_bench_rows`` (beside the
    ``<stem>_reports`` dir, snapshotted with the state)."""
    p = Path(state_path)
    return p.parent / f"{p.stem}_bench_rows"


def rows_key(boundary: int, digest: str) -> str:
    return f"{ROWS_PREFIX}/{int(boundary)}/{digest}.json"


def rig(tmp: Path, **kw):
    """W8b's production lane D step over its fakes (X gated and crowned at B2)."""
    cfg = w8b.chain_cfg(Path(tmp) / "v")
    return w8b.Rig(Path(tmp), cfg, w8b.fixture(Path(tmp)), **kw)


def gated(tmp: Path, **kw):
    """A rig stepped through B1 and B2; returns ``(rig, report at B2)``."""
    r = rig(tmp, **kw)
    r.step(B1)
    r.step(B2)
    return r, json.loads(r.report_file(B2).read_text(encoding="ascii"))


def gate_results(report: dict) -> list[dict]:
    g = report.get("gate")
    return list(g["results"]) if isinstance(g, dict) else []


def persisted_rows(r) -> dict[str, bytes]:
    """``{sha256: bytes}`` of every bench record the rig's state persisted."""
    d = bench_rows_dir(r.state_path)
    if not d.is_dir():
        return {}
    return {p.stem: p.read_bytes() for p in sorted(d.glob("*.json"))}


def independent_lcb(ref_vals, cand_vals, *, alpha: float, B: int, seed: str) -> float:
    """The paired LCB of ``1 - geomean(cand)/geomean(ref)``, computed here
    from its definition (the bench gate's statistic) for cross-checks."""
    import numpy as np

    from cascade.eval.bootstrap import _seed_to_int

    k = np.log(np.maximum(np.asarray(ref_vals, dtype=np.float64), 1e-12))
    c = np.log(np.maximum(np.asarray(cand_vals, dtype=np.float64), 1e-12))
    idx = np.random.default_rng(_seed_to_int(seed)).integers(0, k.size, size=(B, k.size))
    d = (c[idx] - k[idx]).mean(axis=1)
    return float(np.quantile(1.0 - np.exp(d), alpha))


def replay_mod():
    import importlib

    return importlib.import_module("cascade.satellite.lane_d_replay")


def check_rows_fn():
    m = replay_mod()
    assert hasattr(m, "check_gate_rows"), \
        "W3b6b: lane_d_replay.check_gate_rows(report, bench_rows, cfg) (README)"
    return m.check_gate_rows


def json_text(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


__all__ = ["ROLES", "ROWS_PREFIX", "w8b"]
