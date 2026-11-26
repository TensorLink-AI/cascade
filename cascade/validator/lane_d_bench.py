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

Faults on the drawn suite are classified by :func:`classify_suite` (W3b4,
§1.3b "Faults" / "Suite identity is verified per drawn suite", §13): each
role's entry gets at most one typed code (:data:`HOLD_CODES`). A different
suite version / item set (``suite_mismatch``) or a ``timeout`` on any role is
an infra ``bench_hold``; so is any defect on a reference. A defect on the
candidate alone (:data:`FAULT_CODES`: NaN / inf / ``None`` metric, a missing
metric key, ``partial``, missing items, an ``error`` status or an absent
suite) is the miner's ``checkpoint_fault``. The suite identity
(:func:`cascade.eval.suite_identity.suite_identity`) of every role is recorded
on every outcome.

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

from cascade.eval.benchmarks import ITEM_SUITE_KEY, suite_incomplete
from cascade.eval.expert_lane import paired_geomean_lcb
from cascade.eval.suite_identity import identity_mismatch, suite_identity

KNOWN_SUITES = tuple(ITEM_SUITE_KEY.values())
REFS = ("king_d", "house")
ROLES = ("candidate",) + REFS
METRICS = ("crps", "mase")
OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
CLASSES = ("ok", "checkpoint_fault", "bench_hold")
# Defects that are the miner's fault when they sit on the candidate alone.
FAULT_CODES = ("absent", "error", "unswept", "partial", "missing_items", "duplicate_item",
               "malformed_rows", "missing_metric", "null_metric", "nonfinite_metric")
# Infra codes: a hold whichever role carries them (first match decides the code).
INFRA_CODES = ("suite_mismatch", "timeout")
# Every code a bench_hold can carry.
HOLD_CODES = FAULT_CODES + INFRA_CODES + ("no_shared_items",)
_SIDECAR_NAME = {key: name for name, key in ITEM_SUITE_KEY.items()}
_COVERAGE_KEYS = ("skipped", "n_expected", "max_series")
_HEX = frozenset("0123456789abcdef")
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


def _check_pin(pinned: object, suite: str) -> dict | None:
    """A validated copy of ``pinned`` (one identity for ``suite``), or
    ``None`` when there is no pin."""
    if pinned is None:
        return None
    if not isinstance(pinned, Mapping):
        raise TypeError(f"pinned_identity must be a mapping; got {type(pinned).__name__}")
    if pinned.get("suite") != suite:
        raise ValueError(f"pinned_identity is for suite {pinned.get('suite')!r}, not the "
                         f"gate suite {suite!r}")
    version = pinned.get("version")
    if version is not None and not isinstance(version, str):
        raise TypeError(f"pinned_identity version must be a str or None; got {version!r}")
    sha = pinned.get("items_sha256")
    if not isinstance(sha, str):
        raise TypeError(f"pinned_identity items_sha256 must be a str; got {sha!r}")
    if len(sha) != 64 or not set(sha) <= _HEX:
        raise ValueError("pinned_identity items_sha256 must be 64 lower-case hex; "
                         f"got {sha!r}")
    out: dict[str, Any] = {"suite": suite, "version": version, "items_sha256": sha}
    if "n_items" in pinned:
        n = pinned["n_items"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"pinned_identity n_items must be an int >= 0; got {n!r}")
        out["n_items"] = n
    if "digest" in pinned:
        d = pinned["digest"]
        if not isinstance(d, str) or len(d) != 64 or not set(d) <= _HEX:
            raise ValueError(f"pinned_identity digest must be 64 lower-case hex; got {d!r}")
        out["digest"] = d
    return out


def _check_records(candidate_rows: object, king_rows: object, house_rows: object
                   ) -> tuple[dict[str, Mapping], str]:
    records = {"candidate": candidate_rows, "king_d": king_rows, "house": house_rows}
    sessions = {role: _session_of(role, rec) for role, rec in records.items()}
    if len(set(sessions.values())) != 1:
        raise MixedSessionError(
            "mixed_session: the candidate, KING_D and the house must come from one host "
            f"session; got {sessions}")
    for role, rec in records.items():
        _digest_of(role, rec)
    return records, sessions["candidate"]


# ── per-role classification ──────────────────────────────────────────────────


def _entry(rec: Mapping, suite: str) -> Mapping | None:
    suites = rec.get("suites")
    entry = suites.get(suite) if isinstance(suites, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def _status_defect(entry: Mapping | None, suite: str) -> tuple[str, str] | None:
    """``(code, reason)`` when the entry was not swept ``"ok"``."""
    if entry is None:
        return "absent", f"suite {suite!r} was not swept"
    status = entry.get("status")
    if status == "ok":
        return None
    detail = entry.get("detail")
    text = detail.lower() if isinstance(detail, str) else ""
    if status == "timeout" or (status == "error" and ("timeout" in text
                                                      or "timed out" in text)):
        return "timeout", f"suite {suite!r} timed out ({status!r}: {detail!r})"
    if status == "error":
        return "error", f"suite {suite!r} errored: {detail!r}"
    return "unswept", f"suite {suite!r} status {status!r} is not 'ok'"


def _is_partial(entry: Mapping, suite: str) -> bool:
    """Coverage recomputed with :func:`suite_incomplete` (so FEV scope skips
    are exempt and the entry's own flag is ignored) when the entry carries
    coverage counts; otherwise its ``partial`` flag."""
    if any(k in entry for k in _COVERAGE_KEYS):
        return suite_incomplete({**entry, "suite": _SIDECAR_NAME[suite]})
    return entry.get("partial") is True


def _metric_defect(v: object) -> str | None:
    if v is None:
        return "null_metric"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "malformed_rows"
    return None if math.isfinite(float(v)) else "nonfinite_metric"


def _row_check(entry: Mapping, suite: str
               ) -> tuple[dict[str, tuple[float, float]], tuple[str, str] | None]:
    """``({item: (crps, mase)}, defect)`` for an ``"ok"`` entry; ``defect`` is
    ``(code, reason)`` or ``None`` when the rows are clean."""
    raw = entry.get("rows")
    if not isinstance(raw, list):
        return {}, ("malformed_rows", f"suite {suite!r} rows are not a list")
    if not raw:
        return {}, ("missing_items", f"suite {suite!r} has no rows")
    out: dict[str, tuple[float, float]] = {}
    for i, row in enumerate(raw):
        if not isinstance(row, Mapping):
            return {}, ("malformed_rows", f"suite {suite!r} row {i} is not an object")
        item = row.get("item")
        if not isinstance(item, str) or not item:
            return {}, ("malformed_rows", f"suite {suite!r} row {i} has no item id")
        if item in out:
            return {}, ("duplicate_item", f"suite {suite!r} item {item!r} appears twice")
        vals = []
        for m in METRICS:
            if m not in row:
                return {}, ("missing_metric", f"suite {suite!r} item {item!r} is missing {m!r}")
            code = _metric_defect(row[m])
            if code is not None:
                return {}, (code, f"suite {suite!r} item {item!r} has a bad {m!r}: {row[m]!r}")
            vals.append(float(row[m]))
        out[item] = (vals[0], vals[1])
    return out, None


def _item_set(entry: Mapping) -> frozenset[str]:
    rows = entry.get("rows")
    if not isinstance(rows, list):
        return frozenset()
    return frozenset(r.get("item") for r in rows if isinstance(r, Mapping))


def _identity_of(entry: Mapping | None, suite: str) -> dict | None:
    if entry is None or entry.get("status") != "ok":
        return None
    try:
        return suite_identity(suite, entry)
    except (TypeError, ValueError):
        return None


def _version_mismatches(identity: Mapping[str, dict | None], open_roles: list[str],
                        pin: Mapping | None) -> dict[str, str]:
    """``{role: reason}`` for every open role whose suite version is wrong:
    against the pin, or (no pin) against the records' majority version."""
    have = [r for r in open_roles if identity[r] is not None]
    if pin is not None:
        return {r: (f"suite version {identity[r]['version']!r} differs from the pinned "
                    f"{pin['version']!r}")
                for r in have if identity[r]["version"] != pin["version"]}
    versions = {r: identity[r]["version"] for r in have}
    distinct = sorted({repr(v) for v in versions.values()})
    if len(distinct) < 2:
        return {}
    counts = {v: sum(1 for x in versions.values() if repr(x) == v) for v in distinct}
    majority = [v for v, c in counts.items() if 2 * c > len(versions)]
    keep = majority[0] if majority else None
    return {r: f"suite versions disagree across the session: {distinct}"
            for r, v in versions.items() if repr(v) != keep}


def _classify(records: Mapping[str, Mapping], suite: str, pin: Mapping | None
              ) -> tuple[dict, dict[str, dict[str, tuple[float, float]]]]:
    codes: dict[str, str] = {}
    reasons: dict[str, str] = {}

    def mark(role: str, code: str, reason: str) -> None:
        codes[role] = code
        reasons[role] = reason

    entries = {role: _entry(records[role], suite) for role in ROLES}
    identity = {role: _identity_of(entries[role], suite) for role in ROLES}
    for role in ROLES:
        defect = _status_defect(entries[role], suite)
        if defect is not None:
            mark(role, *defect)
    open_roles = [r for r in ROLES if r not in codes]
    for role, reason in _version_mismatches(identity, open_roles, pin).items():
        mark(role, "suite_mismatch", reason)
    for role in [r for r in ROLES if r not in codes]:
        if _is_partial(entries[role], suite):
            mark(role, "partial", f"suite {suite!r} is partial")
    if pin is not None:
        matching = [r for r in ROLES if identity[r] is not None
                    and not identity_mismatch(identity[r], pin)]
        for role in [r for r in ROLES if r not in codes and identity[r] is not None]:
            if identity[role]["items_sha256"] == pin["items_sha256"]:
                continue
            own = _item_set(entries[role])
            if any(own < _item_set(entries[m]) for m in matching):
                mark(role, "missing_items", f"suite {suite!r} lacks items of the pinned set "
                     f"({identity[role]['n_items']} item(s) swept)")
            else:
                mark(role, "suite_mismatch", f"suite {suite!r} item set "
                     f"{identity[role]['items_sha256'][:12]}… differs from the pinned "
                     f"{pin['items_sha256'][:12]}…")
    rows: dict[str, dict[str, tuple[float, float]]] = {role: {} for role in ROLES}
    for role in [r for r in ROLES if r not in codes]:
        rows[role], defect = _row_check(entries[role], suite)
        if defect is not None:
            mark(role, *defect)
    if pin is None:
        # W3b2's rule: an item is missing from a role when every OTHER checkpoint
        # (by digest, so a house identical to KING_D is one checkpoint) has it.
        for role in ROLES:
            if role in codes:
                continue
            dg = records[role]["digest"]
            others = [o for o in ROLES if records[o]["digest"] != dg]
            if not others or any(o in codes for o in others):
                continue
            common = set.intersection(*(set(rows[o]) for o in others))
            missing = sorted(common - set(rows[role]))
            if missing:
                mark(role, "missing_items", f"suite {suite!r} lacks {len(missing)} item(s) "
                     f"the other checkpoints have (first {missing[0]!r})")

    ident = {"pinned": None if pin is None else dict(pin), **identity}
    ref_codes = sorted(r for r in REFS if r in codes)
    infra = [c for c in INFRA_CODES if c in codes.values()]
    if infra:
        outcome, code = "bench_hold", infra[0]
    elif ref_codes:
        outcome, code = "bench_hold", codes[ref_codes[0]]
    elif "candidate" in codes:
        outcome, code = "checkpoint_fault", codes["candidate"]
    else:
        outcome, code = "ok", None
    if outcome == "checkpoint_fault":
        fault_on = ["candidate"]
    elif outcome == "bench_hold":
        fault_on = ref_codes
    else:
        fault_on = []
    reason = ("; ".join(f"{r}: {codes[r]}: {reasons[r]}" for r in ROLES if r in codes)
              if codes else None)
    return ({"suite": suite, "outcome": outcome, "code": code,
             "codes": {r: codes[r] for r in ROLES if r in codes}, "fault_on": fault_on,
             "identity": ident, "reason": reason}, rows)


def classify_suite(candidate_rows: Mapping, king_rows: Mapping, house_rows: Mapping, *,
                   suite: str, pinned_identity: Mapping | None = None) -> dict:
    """Classify the drawn ``suite`` of three sweep records from one host
    session: ``{"outcome", "code", "codes", "fault_on", "identity", …}`` with
    ``outcome`` one of :data:`CLASSES` and every code in :data:`HOLD_CODES`
    (see the module docstring). Raises :class:`MixedSessionError` for mixed
    sessions and ``ValueError`` / ``TypeError`` for an unknown suite or a
    malformed / foreign ``pinned_identity``. Never mutates its inputs."""
    _check_suite("suite", suite)
    pin = _check_pin(pinned_identity, suite)
    records, _ = _check_records(candidate_rows, king_rows, house_rows)
    return _classify(records, suite, pin)[0]


# ── the gate ─────────────────────────────────────────────────────────────────


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
    """The sentinel's paired LCBs (recorded, never gating). Not checked
    against the gate suite's pin."""
    cls, rows = _classify(records, sentinel, None)
    if cls["codes"]:
        return {"suite": sentinel, "lcb": None, "fault_on": sorted(cls["codes"]),
                "codes": cls["codes"], "reason": cls["reason"]}
    lcb = _lcbs(rows, sentinel, alpha=alpha, B=B, seed=seed)
    if lcb is None:
        return {"suite": sentinel, "lcb": None, "fault_on": [], "codes": {},
                "reason": f"sentinel {sentinel!r} has no item paired with the candidate"}
    return {"suite": sentinel, "lcb": lcb, "fault_on": [], "codes": {}, "reason": None}


def paired_gate(candidate_rows: Mapping, king_rows: Mapping, house_rows: Mapping, *,
                suite: str, alpha: float, tau_public: float, B: int, seed: int | str,
                sentinel: str | None = None, pinned_identity: Mapping | None = None
                ) -> dict:
    """The lane D paired gate on the drawn ``suite`` for three sweep records
    (``{"session", "digest", "suites": {suite: {"status", "rows", …}}}``) from
    one host session. Returns a strict-JSON dict whose ``outcome`` is one of
    :data:`OUTCOMES`; a fault or hold carries :func:`classify_suite`'s
    ``outcome`` / ``code`` / ``codes`` / ``fault_on``, and every outcome its
    ``identity``. Raises :class:`MixedSessionError` (``code ==
    "mixed_session"``) when the records come from different sessions,
    ``ValueError`` / ``TypeError`` on a missing session, an unknown suite, a
    sentinel equal to the gate suite, a malformed / foreign
    ``pinned_identity`` or bad parameters."""
    _check_suite("suite", suite)
    if sentinel is not None:
        _check_suite("sentinel", sentinel)
        if sentinel == suite:
            raise ValueError(f"sentinel {sentinel!r} equals the gate suite: the sentinel "
                             "never gates")
    _check_params(alpha, tau_public, B, seed)
    pin = _check_pin(pinned_identity, suite)
    records, session = _check_records(candidate_rows, king_rows, house_rows)
    digests = {role: _digest_of(role, rec) for role, rec in records.items()}
    alpha_f, tau_f = float(alpha), float(tau_public)
    res: dict[str, Any] = {"suite": suite, "session": session,
                           "alpha": alpha_f, "tau_public": tau_f, "B": B, "seed": seed,
                           "digests": digests}
    res["sentinel"] = (None if sentinel is None else
                       _sentinel_record(records, sentinel, alpha=alpha_f, B=B, seed=seed))

    cls, rows = _classify(records, suite, pin)
    res["identity"] = cls["identity"]
    if cls["outcome"] != "ok":
        hold = cls["outcome"] == "bench_hold"
        res.update(outcome=cls["outcome"], code=cls["code"], codes=cls["codes"],
                   fault_on=cls["fault_on"], passed=None if hold else False, hold=hold,
                   reason=cls["reason"])
        return res
    lcb = _lcbs(rows, suite, alpha=alpha_f, B=B, seed=seed)
    if lcb is None:
        res.update(outcome="bench_hold", code="no_shared_items",
                   codes={r: "no_shared_items" for r in REFS}, passed=None, hold=True,
                   fault_on=sorted(REFS),
                   reason=f"suite {suite!r}: a reference shares no item with the candidate")
        return res
    blocked = sorted(r for r in REFS if any(lcb[r][m] < -tau_f for m in METRICS))
    res.update(outcome="gate_regression" if blocked else "pass", code=None, codes={},
               passed=not blocked, hold=False, lcb=lcb, blocked_by=blocked)
    return res


# ── the bound record text (W3b6b, plan rev6.13) ──────────────────────────────


def _finite_or_null(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _finite_or_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_null(v) for v in obj]
    return obj


def sweep_record_text(record: Mapping) -> str:
    """One sweep record's canonical text (sorted keys, compact separators,
    ASCII-escaped, every non-finite float written as ``null``): the bytes a
    gate result's ``rows`` digest binds and ``bench_v/lane_d/`` publishes."""
    import json

    return json.dumps(_finite_or_null(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


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


__all__ = ["CLASSES", "FAULT_CODES", "HOLD_CODES", "INFRA_CODES", "KNOWN_SUITES", "METRICS",
           "OUTCOMES", "REFS", "MixedSessionError", "SweepSession", "classify_suite",
           "paired_gate", "paired_geomean_lcb", "suite_identity", "sweep_record_text"]
