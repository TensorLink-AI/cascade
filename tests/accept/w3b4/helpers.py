"""Recorded suite outputs for the W3b4 acceptance tests (no GPU, no pod).

Nothing here imports a PATHS module (``cascade.eval.suite_identity``,
``cascade.validator.lane_d_bench``): every test imports what it needs in its
body, so on main each failure is an ``ImportError`` / ``ModuleNotFoundError``
for a PATHS module inside the test.

A *sweep record* is W3b2's: one checkpoint's output of one host session::

    {"session": "<id>", "digest": "<64 hex>",
     "suites": {"<suite>": {"status": "ok", "rows": [{"item", "crps", "mase"}, ...],
                            "data_revision": "<rev>"}}}

``suites`` has the shape :func:`cascade.eval.benchmarks.item_rows_from_report`
returns. Values are deterministic (seeded ``random.Random``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random

GATE_SUITE = "boom"
SENTINEL = "gift"
SESSION = "valpriv-4090/2026-11-22T03:00:00Z/1"
OTHER_SESSION = "valpriv-4090/2026-11-21T03:00:00Z/1"
REVISION = "boom@3f1c2a9"
ALPHA = 0.05
TAU = 0.01
B = 1000
SEED = "lane-d/1:9612000:w3b4"
N_ITEMS = 40
ROLES = ("candidate", "king_d", "house")
REFS = ("king_d", "house")
METRICS = ("crps", "mase")
HEX = set("0123456789abcdef")


def digest(name: str) -> str:
    return hashlib.sha256(f"cascade w3b4 checkpoint {name}".encode()).hexdigest()


DIGESTS = {"candidate": digest("candidate"), "king_d": digest("king_d"),
           "house": digest("house")}


def item_ids(suite: str = GATE_SUITE, n: int = N_ITEMS) -> list[str]:
    return [f"{suite}/task-{i:03d}" for i in range(n)]


def rows(suite: str, tag: str, *, n: int = N_ITEMS, factor: float = 1.0) -> list[dict]:
    """Per-item raw CRPS/MASE for ``suite``; ``factor`` > 1 = this checkpoint is
    worse. Small per-item noise seeded by ``tag``."""
    base = random.Random(f"w3b4 base {suite}")
    noise = random.Random(f"w3b4 noise {suite} {tag}")
    out = []
    for item in item_ids(suite, n):
        c, m = math.exp(base.gauss(-0.7, 0.5)), math.exp(base.gauss(0.1, 0.3))
        out.append({"item": item,
                    "crps": c * factor * math.exp(noise.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(noise.gauss(0.0, 0.01))})
    return out


def entry(suite: str, tag: str, *, revision: str | None = REVISION, n: int = N_ITEMS,
          factor: float = 1.0) -> dict:
    e = {"status": "ok", "rows": rows(suite, tag, n=n, factor=factor)}
    if revision is not None:
        e["data_revision"] = revision
    return e


def triple(*, suite: str = GATE_SUITE, sentinel: str | None = SENTINEL,
           revision: str | None = REVISION, n: int = N_ITEMS, session: str = SESSION,
           ref_factor: float = 1.0) -> tuple[dict, dict, dict]:
    """Clean ``(candidate, king_d, house)`` records from one session. With
    ``ref_factor`` 1.0 the candidate passes (it matches both references)."""
    def one(role: str) -> dict:
        factor = 1.0 if role == "candidate" else ref_factor
        suites = {suite: entry(suite, role, revision=revision, n=n, factor=factor)}
        if sentinel is not None:
            suites[sentinel] = entry(sentinel, role, revision=f"{sentinel}@r1", n=n)
        return {"session": session, "digest": DIGESTS[role], "suites": suites}

    return one("candidate"), one("king_d"), one("house")


def expected_items_sha256(ids) -> str:
    """The pinned item-list digest, computed independently of the module."""
    canon = json.dumps(sorted(set(ids)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def expected_identity_digest(suite: str, version, items_sha256: str) -> str:
    canon = json.dumps({"items_sha256": items_sha256, "suite": suite, "version": version},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def pin(suite: str = GATE_SUITE, *, revision: str | None = REVISION,
        n: int = N_ITEMS) -> dict:
    """The pinned identity of a complete ``suite`` sweep, built by hand."""
    sha = expected_items_sha256(item_ids(suite, n))
    return {"suite": suite, "version": revision, "items_sha256": sha, "n_items": n,
            "digest": expected_identity_digest(suite, revision, sha)}


def gate_kwargs(**over) -> dict:
    kw = {"suite": GATE_SUITE, "alpha": ALPHA, "tau_public": TAU, "B": B, "seed": SEED,
          "sentinel": SENTINEL}
    kw.update(over)
    return kw


# ── the row-set patterns (each mutates one record's drawn-suite entry) ────────


def drop_item(rec: dict, suite: str = GATE_SUITE) -> None:
    del rec["suites"][suite]["rows"][5]


def nan_crps(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["rows"][7]["crps"] = float("nan")


def inf_mase(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["rows"][3]["mase"] = float("inf")


def neg_inf_crps(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["rows"][9]["crps"] = float("-inf")


def none_mase(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["rows"][11]["mase"] = None


def missing_crps(rec: dict, suite: str = GATE_SUITE) -> None:
    del rec["suites"][suite]["rows"][0]["crps"]


def partial_flag(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["partial"] = True


def empty_rows(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite]["rows"] = []


def error_status(rec: dict, suite: str = GATE_SUITE) -> None:
    rec["suites"][suite] = {"status": "error", "rows": [],
                            "detail": "no config scored: forward pass raised ValueError"}


def absent_suite(rec: dict, suite: str = GATE_SUITE) -> None:
    del rec["suites"][suite]


# pattern -> the typed sub-reason it must classify as
PATTERNS = {
    "drop_item": (drop_item, "missing_items"),
    "empty_rows": (empty_rows, "missing_items"),
    "nan_crps": (nan_crps, "nonfinite_metric"),
    "inf_mase": (inf_mase, "nonfinite_metric"),
    "neg_inf_crps": (neg_inf_crps, "nonfinite_metric"),
    "none_mase": (none_mase, "null_metric"),
    "missing_crps": (missing_crps, "missing_metric"),
    "partial_flag": (partial_flag, "partial"),
    "error_status": (error_status, "error"),
    "absent_suite": (absent_suite, "absent"),
}


def broken(records: tuple[dict, dict, dict], roles, pattern: str,
           suite: str = GATE_SUITE) -> tuple[dict, dict, dict]:
    """A deep copy of ``records`` with ``pattern`` applied to each of ``roles``."""
    out = copy.deepcopy(list(records))
    fn = PATTERNS[pattern][0]
    for i, role in enumerate(ROLES):
        if role in roles:
            fn(out[i], suite)
    return out[0], out[1], out[2]
