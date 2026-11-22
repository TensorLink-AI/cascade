"""LANE_D_PLAN §1.3b "Rule on the drawn suite" / "Faults" (lane D): the paired
gate sweep of a candidate against KING_D and the house on one bench host.

The candidate, KING_D and the house run on ONE host in ONE sweep session and
are compared per item, never against stored absolute numbers, so a host-wide
shift (driver, SKU) cancels in the paired ratio. On the suite W3b1 draws
(:func:`cascade.validator.lane_d_gate.gate_suite_for`), CRPS **and** MASE must
each satisfy ``LCB_α ≥ −tau_public`` against KING_D **and** against the
house, where the LCB is the bench gate's own paired bootstrap
(:func:`cascade.eval.expert_lane.paired_geomean_lcb`, uncorrected one-sided)
of ``1 − geomean(candidate)/geomean(ref)``. The era's §16 sentinel suite is
swept and recorded but never gates.

Faults on the drawn suite (NaN / inf / ``None`` metric, a missing metric key,
``partial``, an item the other records have but this one lacks) are the
miner's ``checkpoint_fault`` when only the candidate's rows are faulty, and an
infra ``bench_hold`` (the candidate is not judged) when any reference's rows
are faulty.

:class:`SweepSession` wraps the bench host's sweep: every digest a session
has not swept yet goes out in one deduplicated host call, and cached rows are
re-used only within that session.

Pure apart from the injected host sweep: no process-local state (no
``hash()``), so every validator computes the same JSON. Wiring into
``settle_lane_d`` is not this module's job.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any

from cascade.eval.benchmarks import ITEM_SUITE_KEY
from cascade.eval.expert_lane import paired_geomean_lcb

KNOWN_SUITES = tuple(ITEM_SUITE_KEY.values())
REFS = ("king_d", "house")
ROLES = ("candidate",) + REFS
METRICS = ("crps", "mase")
OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
_SEED_TAG = "cascade|lane_d|paired_gate|v1"


class MixedSessionError(ValueError):
    """The three sweep records do not come from one host session."""

    code = "mixed_session"


# ── argument checks ──────────────────────────────────────────────────────────


def _session_of(role: str, rec: object) -> str:
    if not isinstance(rec, Mapping):
        raise TypeError(f"{role} sweep record must be a mapping; got {type(rec).__name__}")
    s = rec.get("session")
    if not isinstance(s, str):
        raise TypeError(f"{role} sweep record session must be a str; got {s!r}")
    if not s:
        raise ValueError(f"{role} sweep record has an empty session")
    return s


def _digest_of(role: str, rec: Mapping) -> str:
    d = rec.get("digest")
    if not isinstance(d, str) or not d:
        raise ValueError(f"{role} sweep record digest must be a non-empty str; got {d!r}")
    return d


def _check_suite(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {value!r}")
    if value not in KNOWN_SUITES:
        raise ValueError(f"{name} {value!r} is not one of {list(KNOWN_SUITES)}")
    return value


def _check_params(alpha: object, tau_public: object, B: object, seed: object) -> None:
    for name, v in (("alpha", alpha), ("tau_public", tau_public)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{name} must be a number; got {v!r}")
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite; got {v!r}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5); got {alpha!r}")
    if tau_public < 0:
        raise ValueError(f"tau_public must be >= 0; got {tau_public!r}")
    if isinstance(B, bool) or not isinstance(B, int):
        raise TypeError(f"B must be an int; got {B!r}")
    if B < 1:
        raise ValueError(f"B must be >= 1; got {B!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise TypeError(f"seed must be an int or str; got {seed!r}")


# ── per-record rows and faults ───────────────────────────────────────────────


def _metric(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _suite_rows(rec: Mapping, suite: str) -> tuple[dict[str, tuple[float, float]], str | None]:
    """``({item: (crps, mase)}, fault)`` for one record's ``suite`` — ``fault``
    is a reason string, or ``None`` when the rows are clean."""
    suites = rec.get("suites")
    entry = suites.get(suite) if isinstance(suites, Mapping) else None
    if not isinstance(entry, Mapping):
        return {}, f"suite {suite!r} was not swept"
    if entry.get("status") != "ok":
        return {}, f"suite {suite!r} status {entry.get('status')!r} is not 'ok'"
    if entry.get("partial") is True:
        return {}, f"suite {suite!r} is partial"
    raw = entry.get("rows")
    if not isinstance(raw, list) or not raw:
        return {}, f"suite {suite!r} has no rows"
    out: dict[str, tuple[float, float]] = {}
    for i, row in enumerate(raw):
        if not isinstance(row, Mapping):
            return {}, f"suite {suite!r} row {i} is not an object"
        item = row.get("item")
        if not isinstance(item, str) or not item:
            return {}, f"suite {suite!r} row {i} has no item id"
        if item in out:
            return {}, f"suite {suite!r} item {item!r} appears twice"
        vals = []
        for m in METRICS:
            if m not in row:
                return {}, f"suite {suite!r} item {item!r} is missing {m!r}"
            v = _metric(row[m])
            if v is None:
                return {}, f"suite {suite!r} item {item!r} has a non-finite {m!r}: {row[m]!r}"
            vals.append(v)
        out[item] = (vals[0], vals[1])
    return out, None


def _suite_faults(records: Mapping[str, Mapping], suite: str
                  ) -> tuple[dict[str, dict[str, tuple[float, float]]], dict[str, str]]:
    """Rows per role on ``suite`` and ``{role: reason}`` for every faulty role.
    An item counts as missing from a role when every OTHER checkpoint (by
    digest, so a house identical to KING_D is one checkpoint) has it."""
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    faults: dict[str, str] = {}
    for role in ROLES:
        r, fault = _suite_rows(records[role], suite)
        rows[role] = r
        if fault is not None:
            faults[role] = fault
    for role in ROLES:
        if role in faults:
            continue
        dg = records[role]["digest"]
        others = [o for o in ROLES if records[o]["digest"] != dg]
        if not others or any(o in faults for o in others):
            continue
        common = set.intersection(*(set(rows[o]) for o in others))
        missing = sorted(common - set(rows[role]))
        if missing:
            faults[role] = (f"suite {suite!r} lacks {len(missing)} item(s) the other "
                            f"checkpoints have (first {missing[0]!r})")
    return rows, faults


def _lcbs(rows: Mapping[str, Mapping[str, tuple[float, float]]], suite: str, *,
          alpha: float, B: int, seed: int | str) -> dict[str, dict[str, float]] | None:
    """``{ref: {metric: LCB}}`` paired by item id, or ``None`` when a reference
    shares no item with the candidate. The bootstrap seed depends on the
    suite and metric but not on the reference, so a house identical to
    KING_D yields identical LCBs."""
    cand = rows["candidate"]
    out: dict[str, dict[str, float]] = {}
    for ref in REFS:
        ids = sorted(set(cand) & set(rows[ref]))
        if not ids:
            return None
        out[ref] = {}
        for j, m in enumerate(METRICS):
            lcb = paired_geomean_lcb([rows[ref][i][j] for i in ids],
                                     [cand[i][j] for i in ids],
                                     alpha=alpha, B=B, seed=f"{_SEED_TAG}|{seed}|{suite}|{m}")
            if not math.isfinite(lcb):
                return None
            out[ref][m] = lcb
    return out


def _sentinel_record(records: Mapping[str, Mapping], sentinel: str, *, alpha: float,
                     B: int, seed: int | str) -> dict:
    rows, faults = _suite_faults(records, sentinel)
    if faults:
        return {"suite": sentinel, "lcb": None, "fault_on": sorted(faults),
                "reason": "; ".join(f"{r}: {faults[r]}" for r in sorted(faults))}
    lcb = _lcbs(rows, sentinel, alpha=alpha, B=B, seed=seed)
    if lcb is None:
        return {"suite": sentinel, "lcb": None, "fault_on": [],
                "reason": f"sentinel {sentinel!r} has no item paired with the candidate"}
    return {"suite": sentinel, "lcb": lcb, "fault_on": [], "reason": None}


# ── the gate ─────────────────────────────────────────────────────────────────


def paired_gate(candidate_rows: Mapping, king_rows: Mapping, house_rows: Mapping, *,
                suite: str, alpha: float, tau_public: float, B: int, seed: int | str,
                sentinel: str | None = None) -> dict:
    """The lane D paired gate on the drawn ``suite`` for three sweep records
    (``{"session", "digest", "suites": {suite: {"status", "rows", …}}}``) from
    one host session. Returns a strict-JSON dict whose ``outcome`` is one of
    :data:`OUTCOMES` (see the module docstring). Raises
    :class:`MixedSessionError` (``code == "mixed_session"``) when the records
    come from different sessions, ``ValueError`` / ``TypeError`` on a missing
    session, an unknown suite, a sentinel equal to the gate suite or bad
    parameters."""
    _check_suite("suite", suite)
    if sentinel is not None:
        _check_suite("sentinel", sentinel)
        if sentinel == suite:
            raise ValueError(f"sentinel {sentinel!r} equals the gate suite: the sentinel "
                             "never gates")
    _check_params(alpha, tau_public, B, seed)
    records = {"candidate": candidate_rows, "king_d": king_rows, "house": house_rows}
    sessions = {role: _session_of(role, rec) for role, rec in records.items()}
    if len(set(sessions.values())) != 1:
        raise MixedSessionError(
            "mixed_session: the candidate, KING_D and the house must come from one host "
            f"session; got {sessions}")
    digests = {role: _digest_of(role, rec) for role, rec in records.items()}
    alpha_f, tau_f = float(alpha), float(tau_public)
    res: dict[str, Any] = {"suite": suite, "session": sessions["candidate"],
                           "alpha": alpha_f, "tau_public": tau_f, "B": B, "seed": seed,
                           "digests": digests}
    res["sentinel"] = (None if sentinel is None else
                       _sentinel_record(records, sentinel, alpha=alpha_f, B=B, seed=seed))

    rows, faults = _suite_faults(records, suite)
    ref_faults = [r for r in sorted(REFS) if r in faults]
    if ref_faults:
        res.update(outcome="bench_hold", passed=None, hold=True, fault_on=ref_faults,
                   reason="; ".join(f"{r}: {faults[r]}" for r in ref_faults))
        return res
    if "candidate" in faults:
        res.update(outcome="checkpoint_fault", passed=False, hold=False,
                   fault_on=["candidate"], reason=f"candidate: {faults['candidate']}")
        return res
    lcb = _lcbs(rows, suite, alpha=alpha_f, B=B, seed=seed)
    if lcb is None:
        res.update(outcome="bench_hold", passed=None, hold=True, fault_on=sorted(REFS),
                   reason=f"suite {suite!r}: a reference shares no item with the candidate")
        return res
    blocked = sorted(r for r in REFS if any(lcb[r][m] < -tau_f for m in METRICS))
    res.update(outcome="gate_regression" if blocked else "pass", passed=not blocked,
               hold=False, lcb=lcb, blocked_by=blocked)
    return res


# ── one host session ─────────────────────────────────────────────────────────


Sweep = Callable[[tuple[str, ...], tuple[str, ...]], Mapping[str, Mapping[str, Mapping]]]


class SweepSession:
    """One bench host session. ``sweep(digests, suites)`` is the host's sweep
    (``{digest: {suite: {"status", "rows", …}}}``). ``cache`` (optional, may be
    shared) is keyed by session id, so rows are re-used only within a
    session: a new session re-sweeps KING_D and the house."""

    def __init__(self, session_id: str, sweep: Sweep,
                 cache: MutableMapping | None = None):
        if not isinstance(session_id, str):
            raise TypeError(f"session_id must be a str; got {session_id!r}")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not callable(sweep):
            raise TypeError("sweep must be callable")
        self.session_id = session_id
        self._sweep = sweep
        store = {} if cache is None else cache
        self._rows: dict[str, dict[str, Any]] = store.setdefault(session_id, {})

    @staticmethod
    def _suites(suites: Iterable[str]) -> tuple[str, ...]:
        if isinstance(suites, (str, bytes)):
            raise TypeError(f"suites must be a sequence of names; got {suites!r}")
        out: list[str] = []
        for s in suites:
            if not isinstance(s, str) or not s:
                raise ValueError(f"suite {s!r} is not a non-empty str")
            if s not in out:
                out.append(s)
        if not out:
            raise ValueError("at least one suite is required")
        return tuple(out)

    def _record(self, dg: str) -> dict:
        return {"session": self.session_id, "digest": dg,
                "suites": copy.deepcopy(self._rows.get(dg, {}))}

    def paired(self, candidate: str, king_d: str, house: str, *,
               suites: Iterable[str]) -> tuple[dict, dict, dict]:
        """The ``(candidate, king_d, house)`` sweep records of this session,
        sweeping every digest not yet swept here for all ``suites`` in one
        deduplicated host call."""
        wanted = self._suites(suites)
        order: list[str] = []
        for name, dg in (("candidate", candidate), ("king_d", king_d), ("house", house)):
            if not isinstance(dg, str) or not dg:
                raise ValueError(f"{name} digest must be a non-empty str; got {dg!r}")
            if dg not in order:
                order.append(dg)
        todo = [dg for dg in order
                if not all(s in self._rows.get(dg, {}) for s in wanted)]
        if todo:
            got = self._sweep(tuple(todo), wanted)
            if not isinstance(got, Mapping):
                raise TypeError(f"host sweep returned {type(got).__name__}, not a mapping")
            for dg in todo:
                per = got.get(dg)
                if not isinstance(per, Mapping):
                    continue
                slot = self._rows.setdefault(dg, {})
                for s, entry in per.items():
                    if isinstance(s, str) and isinstance(entry, Mapping):
                        slot[s] = copy.deepcopy(dict(entry))
        return self._record(candidate), self._record(king_d), self._record(house)


__all__ = ["KNOWN_SUITES", "METRICS", "OUTCOMES", "REFS", "MixedSessionError",
           "SweepSession", "paired_gate", "paired_geomean_lcb"]
