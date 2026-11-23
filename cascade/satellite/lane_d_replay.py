"""Independent satellite replay of a lane D receipt chain (LANE_D_PLAN §14.3,
§1.5; W7d).

:func:`replay_chain` takes what a satellite is given (the published signed
``lane-d-report/1`` files, the ``lane-d-receipt/1`` records in settlement
order, the parsed ``[lane_d]`` config in force and the pinned report signer;
optionally the metagraph per boundary and the published state snapshots) and
returns one :class:`SettlementReplay` per receipt: ``MATCH``, ``MISMATCH``
(with the field paths that disagree) or ``UNVERIFIABLE`` (with a typed
reason).

Invariant 12: this module re-implements the settlement rules from the plan
and the signed formats. It imports only the shared pure primitives
(``cascade.eval.bootstrap`` for the max-T statistic, ``cascade.shared.sr25519``
for the signature check), the standard library and numpy; never the
producer's eligibility, verdict, gate, state or weight code, and never the
audit.

Per settlement, in order (the first two can end the settlement early):

1. the report is located and parsed strictly (``report_missing`` /
   ``report_unreadable``), its ``schema`` is ``lane-d-report/1``
   (``unknown_schema``) and the pinned signer's sr25519 signature over the
   canonical body is checked (``signature_unverifiable`` when the verifier
   cannot run, a mismatch at ``signer`` / ``signature`` when it fails);
2. the stated ``params`` equal the config, ``seed`` and ``rows_sha256`` are
   re-derived, the rev6.6 armed context set (``armed_contexts`` +
   ``armed_evidence``, when stated) is verified from its own evidence (never
   re-derived from ``cfg.contexts``), and every cohort is re-judged from the signed rows: the
   forward-only bound on both clocks, the floors, one finite row per
   (window, scored model), the max-T LCB, gain and clear against KING_D and,
   when a miner holds KING_D, against the house as a second bar;
3. the chain rules: ``king_before`` (C2), every cohort a first (and only)
   verdict against KING_D (C4, rev6.11 (b): no confirmation stage, record or
   D20 rule), held candidates (C6), the §1.3b gate
   draw and its results' outcomes from their LCBs, the crown, ``king_after``,
   ``prior_kings`` (C3), the paid court and its decayed shares (C5) and the
   receipt links (C1);
4. with a state snapshot, its digest, canonical bytes and consensus fields.

W5f (§1.4 "The house", §2 items 1-2, T10), re-derived from the chain: a new
house digest while a miner holds KING_D (and is neither KING_D nor a prior
king) opens a house challenger. It is judged alone in its own cohort
(``members == [house]``, ``king_digest`` = KING_D, no ``house_digest``, its T
the report's house record's), crowned at its first clear (A10 and the gate
decide) on its own progress, never a miner entry's of the same digest. A
cleared, passing house retakes KING_D: ``king_after`` is the house record, the miner king the
newest prior king, the slot burns (``house_king``) and ``crowned`` is null. A
lost house is not judged again; a retake or a new pointer ends it. An entry
is ``duplicate_digest`` exactly when it is a copy: a new entry naming the
house or KING_D, an open entry naming a new house, or one already closed so.

Only structured fields are compared (invariant 7). A mismatch is a field
path (``verdicts[0].lcb``, ``pay.court[1].uid``, ``receipt.report_sha256``,
``state.king``); ``extensions`` is never read. Once a settlement is not
``MATCH`` every later one is ``UNVERIFIABLE`` (``prior_unverified``).

The replay is pure: it reads files, writes nothing and never mutates an
input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np

from cascade.eval.bootstrap import _seed_to_int, cohort_maxt_lcbs
from cascade.shared.sr25519 import SignatureVerifierUnavailable, verify_hotkey

MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNVERIFIABLE = "UNVERIFIABLE"
STATUSES = (MATCH, MISMATCH, UNVERIFIABLE)

# ``receipt_malformed``: the receipt has no integer boundary, so no report
# can be located for it. ``confirmation_stage_present`` (rev6.11 (b)): the
# signed report carries a confirmation stage (a cohort or verdict ``stage``
# other than ``first``, a verdict ``confirmation`` block or an entry
# ``confirm_pending``); there is no such stage, so it is refused, never judged.
UNVERIFIABLE_REASONS = ("unknown_schema", "report_missing", "report_unreadable",
                        "signature_unverifiable", "prior_unverified", "receipt_malformed",
                        "confirmation_stage_present")

REPORT_SCHEMA = "lane-d-report/1"
STATE_SCHEMA = "lane-d-state/1"
RECEIPT_SCHEMA = "lane-d-receipt/1"
SCORING_METHOD = "quantile_head"
# The quantile head's nine levels, stated in every report's params.
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RECEIPT_KEYS = ("schema", "boundary", "prior_state_sha256", "posterior_state_sha256",
                "report_sha256")

GATE_OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
GATE_REFS = ("king_d", "house")
GATE_METRICS = ("crps", "mase")
_GATE_KEYS = frozenset({"suite", "block", "block_hash", "sentinel", "eligible", "results"})
_RESULT_KEYS = frozenset({"digest", "outcome", "lcb", "blocked_by", "session"})
# W3b1: the gate draw's own domain tag over the boundary block hash.
_DRAW_TAG = b"cascade|lane_d|gate_suite|v1|"
_DRAW_SPACE = 1 << 256

_ROW_KEYS = frozenset({"window_id", "digest", "cohort", "crps", "mase"})
_WINDOW_KEYS = frozenset({"window_id", "source", "feed", "min_target_ts", "min_first_seen_ts",
                          "mirror_lag_hours", "snapshot_sha256",
                          # rev6.10 (a) / rev6.12 (W7a): the embargo record and the
                          # W5j draw cell
                          "freq", "embargo_hours", "min_release_ts", "rule", "embargo_window",
                          "timeframe", "bin", "kind", "freq_class", "h", "origin"})
_COHORT_KEYS = frozenset({"index", "king_digest", "king_reveal_ts", "stage", "members", "t_max",
                          "window_ids", "set_sha256", "first_seen_roots", "refused",
                          "fresh_windows", "embargo_windows"})
_VERDICT_KEYS = frozenset({"digest", "king_digest", "stage", "cohort", "lcb", "gain", "cleared",
                           "outcome", "lowfreq", "fresh", "embargo"})
# W5j: the draw record (top level, present iff a cohort was drawn).
_DRAW_KEYS = frozenset({"boundary_block_hash", "seed", "max_windows_per_settlement", "cohorts"})
_DRAW_COHORT_KEYS = frozenset({"index", "quota", "capacity", "candidates_sha256"})
_DRAW_SEED_TAG = "lane_d_windows"
_BLOCK_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
# rev6.9 / W7a: the class of a pool freq (exact match), else subdaily.
_FREQ_CLASS = {"D": "daily", "W": "weekly", "M": "monthly", "ME": "monthly", "MS": "monthly",
               "Q": "quarterly", "QE": "quarterly", "QS": "quarterly",
               "A": "yearly", "Y": "yearly", "YE": "yearly"}
_ENTRY_STABLE = ("hotkey", "coldkey", "context", "ready_block", "reveal_block", "reveal_ts")
OPEN_STATUSES = ("pending", "hold")
# The chain's internal key of the house challenger's progress (pending first
# clear, hold, used windows, status): never a miner entry's key, so a miner
# copy of the house never lends it its progress.
_HOUSE_KEY = "house:"

_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(\[[0-9]+\]|\.[a-z0-9_]+)*$")
_KEY_RE = re.compile(r"^[a-z0-9_]+$")
# rev6.6 (§1.1): the armed-context evidence a report carries, and the source
# floor of the readiness gate (stated by the plan, not a config key).
ARMED_MIN_SOURCES = 20
_ARMED_EVIDENCE_KEYS = frozenset({"min_long_series", "min_sources", "readiness_sha256",
                                  "t7_sha256", "variants"})
_ARMED_VARIANT_KEYS = frozenset({"context", "series", "sources", "t7_fits", "armed"})
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SIG_RE = re.compile(r"^[0-9a-f]{128}$")
_BLOCK_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")


@dataclass(frozen=True)
class SettlementReplay:
    """The replay of one receipt. ``mismatches`` is non-empty iff ``status``
    is ``MISMATCH``; ``reason`` is one of :data:`UNVERIFIABLE_REASONS` iff
    ``UNVERIFIABLE``. The derived fields (``crowned`` … ``pay``) are the
    replay's own; ``pay`` is ``(digest, uid or None, share)`` per court slot,
    king first (a ``None`` uid burns)."""

    boundary: int | None
    status: str
    mismatches: tuple[str, ...] = ()
    reason: str | None = None
    crowned: str | None = None
    king_after: str | None = None
    outcomes: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    gate: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    pay: tuple[tuple[str, int | None, float], ...] = ()
    report_sha256: str | None = None
    posterior_state_sha256: str | None = None


# ── canonical bytes and strict parsing ──────────────────────────────────────


def _canonical(obj) -> bytes:
    """sort_keys, compact separators, ASCII-escaped, no NaN, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _body_sha(report: dict) -> str:
    """sha256 of the report's canonical body (every field but ``signature``)."""
    return _sha(_canonical({k: v for k, v in report.items() if k != "signature"}))


def _genesis_sha(house: dict) -> str:
    """The W7c genesis state digest for ``house`` (the house holds KING_D)."""
    h = {"digest": house["digest"], "reveal_ts": house["reveal_ts"]}
    return _sha(_canonical({"schema": STATE_SCHEMA, "boundary": 0, "house": dict(h),
                            "king": {**h, "hotkey": None, "house": True},
                            "prior_kings": [], "entries": {}}))


class _Unreadable(ValueError):
    """Bytes that are not one strict JSON object."""


def _strict_json(raw: bytes):
    """``raw`` as JSON, refusing bad UTF-8, bad JSON, NaN/Infinity (also as
    an overflowing literal) and a key repeated in any object."""

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise _Unreadable(f"key {k!r} repeats")
            out[k] = v
        return out

    def constant(name):
        raise _Unreadable(f"non-finite number {name}")

    def number(text):
        v = float(text)
        if not math.isfinite(v):
            raise _Unreadable(f"non-finite number {text}")
        return v

    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as e:
        raise _Unreadable(f"not UTF-8 ({e})") from e
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                          parse_float=number)
    except _Unreadable:
        raise
    except (ValueError, RecursionError) as e:
        raise _Unreadable(f"not JSON ({e})") from e


def _load_report(reports, boundary: int) -> tuple[dict | None, str | None]:
    """``(report, None)`` or ``(None, reason)``. The report is always a fresh
    plain copy (a mapping value is round-tripped through canonical JSON)."""
    if isinstance(reports, Mapping):
        if boundary not in reports:
            return None, "report_missing"
        try:
            rep = _strict_json(_canonical(reports[boundary]))
        except (TypeError, ValueError):
            return None, "report_unreadable"
    else:
        p = Path(reports) / f"lane_d_report_{int(boundary)}.json"
        if not p.exists():
            return None, "report_missing"
        try:
            rep = _strict_json(p.read_bytes())
        except (OSError, ValueError):
            return None, "report_unreadable"
    if not isinstance(rep, dict):
        return None, "report_unreadable"
    return rep, None


# ── comparison helpers ──────────────────────────────────────────────────────


class _Paths:
    """The mismatch paths of one settlement, in discovery order, no repeats."""

    def __init__(self):
        self.items: list[str] = []

    def add(self, path: str) -> None:
        if not _PATH_RE.fullmatch(path):
            path = re.split(r"[.\[]", path, maxsplit=1)[0] or "report"
        if path not in self.items:
            self.items.append(path)


def _key(path: str, key) -> str:
    """``path.key`` when ``key`` is a path-safe name, else ``path``."""
    return f"{path}.{key}" if isinstance(key, str) and _KEY_RE.fullmatch(key) else path


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x) -> bool:
    return _is_num(x) and math.isfinite(float(x))


def _same(a, b) -> bool:
    """JSON equality: a bool is never a number, ``None`` only equals ``None``."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return type(a) is type(b) and a == b
    if _is_num(a) and _is_num(b):
        return a == b
    return type(a) is type(b) and a == b


def _compare(d: _Paths, path: str, want, got) -> None:
    """Every path under ``path`` where ``got`` differs from ``want``."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            d.add(path)
            return
        for k in sorted(set(want) | set(got), key=str):
            sub = _key(path, k)
            if k not in want or k not in got:
                d.add(sub)
            else:
                _compare(d, sub, want[k], got[k])
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            d.add(path)
            return
        for i, (w, g) in enumerate(zip(want, got, strict=True)):
            _compare(d, f"{path}[{i}]", w, g)
    elif not _same(want, got):
        d.add(path)


def _close(got, want: float) -> bool:
    return _is_num(got) and math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12)


def _subscore_ok(stated, scores, idx: list[int]) -> bool:
    """rev6.12 (b): ``stated`` is ``{n, geomean}`` of the rows at ``idx``
    (sqrt of the CRPS and MASE geometric means, floors 1e-12 / 1e-9;
    ``geomean`` null iff n = 0)."""
    if not isinstance(stated, dict) or set(stated) != {"n", "geomean"} \
            or not _same(stated["n"], len(idx)):
        return False
    if not idx:
        return stated["geomean"] is None
    crps, mase = scores
    lc = float(np.mean(np.log(np.maximum(crps[idx], 1e-12))))
    lm = float(np.mean(np.log(np.maximum(mase[idx], 1e-9))))
    return _close(stated["geomean"], math.sqrt(math.exp(lc) * math.exp(lm)))


def _king(rec) -> dict:
    """A king record ``{digest, hotkey, house, reveal_ts}`` (raises on a malformed one)."""
    if not isinstance(rec, dict):
        raise TypeError("king record is not an object")
    out = {"digest": rec["digest"], "hotkey": rec.get("hotkey"), "house": rec["house"],
           "reveal_ts": rec["reveal_ts"]}
    if not (isinstance(out["digest"], str) and isinstance(out["house"], bool)
            and _is_int(out["reveal_ts"])
            and (out["hotkey"] is None or isinstance(out["hotkey"], str))):
        raise TypeError("king record fields")
    return out


# ── the config, as a report states it ───────────────────────────────────────


def _gate_weight_pairs(cfg) -> list[tuple[str, float]]:
    raw = cfg.gate_weights
    if raw is None:
        return [(str(s), 1.0) for s in cfg.gate_suites]
    items = raw.items() if isinstance(raw, Mapping) else raw
    return [(str(s), float(w)) for s, w in items]


def expected_params(cfg) -> dict:
    """The ``params`` block a report must state for ``cfg`` (§14.3 item 2)."""
    return {"share": float(cfg.share), "takes_from": str(cfg.takes_from),
            "remainder": str(cfg.remainder), "horizon": int(cfg.horizon),
            "horizons": [int(cfg.horizon)], "contexts": [int(c) for c in cfg.contexts],
            "forward_margin_hours": float(cfg.forward_margin_hours),
            "max_mirror_lag_hours": float(cfg.max_mirror_lag_hours),
            "min_windows": int(cfg.min_windows), "min_clusters": int(cfg.min_clusters),
            "d_margin": float(cfg.d_margin), "alpha": float(cfg.alpha),
            "bootstrap_B": int(cfg.bootstrap_B), "king_decay": float(cfg.king_decay),
            "reward_prior_kings": int(cfg.reward_prior_kings),
            "max_entries_per_settlement": int(cfg.max_entries_per_settlement),
            "max_judged_per_coldkey_per_settlement":
                int(cfg.max_judged_per_coldkey_per_settlement),
            "scoring_levels": [float(q) for q in SCORING_LEVELS],
            "gate_suites": [str(s) for s in cfg.gate_suites],
            "tau_public": float(cfg.tau_public),
            "lowfreq_freqs": [str(f) for f in cfg.lowfreq_freqs],
            "gate_weights": dict(_gate_weight_pairs(cfg)),
            "max_source_share": float(cfg.max_source_share),
            "min_long_series": int(cfg.min_long_series),
            # rev6.10 (a) / rev6.12 (b) (W2e): the embargo cap, the W5j
            # settlement budget and the minimum fresh share.
            "embargo_hours": float(cfg.embargo_hours),
            "max_windows_per_settlement": int(cfg.max_windows_per_settlement),
            "min_fresh_window_share": float(cfg.min_fresh_window_share),
            # rev6.14 (a): the judging start j = T + eval_delay_hours.
            "eval_delay_hours": float(cfg.eval_delay_hours)}


def court_shares(cfg, n_slots: int) -> list[float]:
    """§1.4 pay: slot ``i`` of an ``n_slots`` court is worth
    ``share * king_decay**i / sum_j king_decay**j``."""
    raw = [float(cfg.king_decay) ** i for i in range(n_slots)]
    total = sum(raw)
    return [float(cfg.share) * w / total for w in raw] if total > 0 else [0.0] * n_slots


# ── the statistic ───────────────────────────────────────────────────────────


def combined_log_ratio(r_fwd: float, r_bt: float, *, n_fwd: int, n_bt: int, cap: float,
                       floor: float) -> float:
    """rev6.9 (W5i) combination of the forward and backtest ratios, re-derived
    here: ``(1 - c) * log(r_fwd) + c * log(clamp(r_bt, floor, 1 / floor))``
    with ``c = min(cap, n_bt / (n_bt + n_fwd))`` (``c = 0`` without backtest
    windows). With ``c = 0`` it is exactly ``log(r_fwd)``, so memorising the
    backtest gains at most ``1 - floor ** cap``."""
    c = 0.0 if int(n_bt) == 0 else min(float(cap), int(n_bt) / (int(n_bt) + int(n_fwd)))
    if c <= 0.0:
        return math.log(float(r_fwd))
    clamped = min(max(float(r_bt), float(floor)), 1.0 / float(floor))
    return (1.0 - c) * math.log(float(r_fwd)) + c * math.log(clamped)


def _parts(crps: np.ndarray, mase: np.ndarray):
    """Bootstrap inputs whose per-window weighted quantile loss is ``crps``."""
    return ((crps / 2.0)[:, None], np.ones(len(crps), dtype=np.float64), mase)


def _geomean(crps: np.ndarray, mase: np.ndarray) -> float:
    """sqrt(geomean(CRPS) * geomean(MASE)) with the floors 1e-12 / 1e-9."""
    g_crps = math.exp(float(np.mean(np.log(np.maximum(crps, 1e-12)))))
    g_mase = math.exp(float(np.mean(np.log(np.maximum(mase, 1e-9)))))
    return math.sqrt(g_crps * g_mase)


def _judge(cfg, scores: dict, bar: str, members: list[str], clusters: list[str],
           seed: str) -> list[tuple[float, float, bool]]:
    """``(lcb, gain, cleared)`` per member against ``bar``: the max-T LCB
    over the cohort, the full-sample relative gain, and the clear
    ``lcb >= d_margin and gain > 0``."""
    lcbs = cohort_maxt_lcbs(_parts(*scores[bar]), [_parts(*scores[m]) for m in members],
                            alpha=float(cfg.alpha), B=int(cfg.bootstrap_B), seed=seed,
                            clusters=clusters)
    bg = _geomean(*scores[bar])
    denom = bg if abs(bg) >= 1e-9 else 1e-9
    out = []
    for m, lcb in zip(members, lcbs, strict=True):
        gain = (bg - _geomean(*scores[m])) / denom
        out.append((float(lcb), float(gain),
                    bool(float(lcb) >= float(cfg.d_margin) and gain > 0.0)))
    return out


# ── the §1.3b gate ──────────────────────────────────────────────────────────


def gate_draw(cfg, block_hash, sentinel) -> tuple[str, list[str]] | None:
    """W3b1: ``(suite, eligible)`` for ``block_hash`` (``0x`` + 64 lowercase
    hex): the eligible suites are ``gate_suites`` in order minus the
    sentinel, with positive weight; the tagged sha256 of the hash digits,
    as a 256-bit fraction of the total weight, falls in one suite's
    cumulative interval (exact rationals). ``None`` when nothing can be
    drawn."""
    if not isinstance(block_hash, str) or not _BLOCK_HASH_RE.fullmatch(block_hash):
        return None
    if sentinel is not None and (not isinstance(sentinel, str) or not sentinel):
        return None
    weights = {s: Fraction(w) for s, w in _gate_weight_pairs(cfg)}
    eligible = [str(s) for s in cfg.gate_suites
                if str(s) != sentinel and weights.get(str(s), Fraction(0)) > 0]
    if not eligible:
        return None
    total = sum((weights[s] for s in eligible), Fraction(0))
    d = int.from_bytes(hashlib.sha256(_DRAW_TAG + block_hash[2:].encode("ascii")).digest(), "big")
    point = Fraction(d, _DRAW_SPACE) * total
    acc = Fraction(0)
    for s in eligible:
        acc += weights[s]
        if point < acc:
            return s, eligible
    return None


# ── W3b6b: the bound bench records (plan rev6.13) ───────────────────────────

ROWS_ROLES = ("candidate", "king_d", "house")
_GATE_SEED_TAG = "cascade|lane_d|paired_gate|v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _rows_source(bench_rows, digest: str) -> bytes | None:
    """The published bytes of ``digest`` from a ``{sha256: bytes}`` mapping or
    a directory of ``<sha256>.json`` files, or ``None``."""
    if isinstance(bench_rows, Mapping):
        data = bench_rows.get(digest)
        return bytes(data) if isinstance(data, (bytes, bytearray)) else None
    p = Path(bench_rows) / f"{digest}.json"
    try:
        return p.read_bytes()
    except OSError:
        return None


def _paired_lcb(ref, cand, *, alpha: float, B: int, seed: str) -> float:
    """LCB_alpha of ``1 - geomean(cand)/geomean(ref)`` over paired items,
    items resampled with replacement (the bench gate's statistic, re-derived
    here: invariant 12 keeps the producer's gate code out of the replay)."""
    k = np.log(np.maximum(np.asarray(ref, dtype=np.float64), 1e-12))
    c = np.log(np.maximum(np.asarray(cand, dtype=np.float64), 1e-12))
    n = k.size
    if n == 0 or k.shape != c.shape:
        return float("nan")
    idx = np.random.default_rng(_seed_to_int(seed)).integers(0, n, size=(B, n))
    d = (c[idx] - k[idx]).mean(axis=1)
    return float(np.quantile(1.0 - np.exp(d), alpha))


def _suite_items(rec: Mapping, suite: str) -> dict[str, tuple[float, float]] | None:
    entry = (rec.get("suites") or {}).get(suite) if isinstance(rec.get("suites"), Mapping) \
        else None
    rows = entry.get("rows") if isinstance(entry, Mapping) else None
    if not isinstance(rows, list):
        return None
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("item"), str):
            return None
        c, m = row.get("crps"), row.get("mase")
        if not (_finite(c) and _finite(m)) or row["item"] in out:
            return None
        out[row["item"]] = (float(c), float(m))
    return out


def check_gate_rows(report: Mapping, bench_rows, cfg) -> list[str]:
    """The mismatch paths of ``report``'s bound bench records (W3b6b): each
    bound digest resolves to bytes hashing to it, whose record names the
    result's session and its role's digest; a ``pass`` / ``gate_regression``
    result's four LCBs equal the paired LCBs recomputed from the drawn
    suite's rows (seed ``cascade|lane_d|paired_gate|v1|lane-d/1:gate:<b>:
    <hash>|<suite>|<metric>``, within 1e-12); a pass result for the crowned
    digest must carry ``rows``. Pure; ``[]`` when everything agrees."""
    out: list[str] = []
    gate = report.get("gate") if isinstance(report, Mapping) else None
    if not isinstance(gate, Mapping) or not isinstance(gate.get("results"), list):
        return out
    suite = gate.get("suite")
    seed = f"lane-d/1:gate:{report.get('boundary')}:{gate.get('block_hash')}"
    king = (report.get("king_before") or {}).get("digest") \
        if isinstance(report.get("king_before"), Mapping) else None
    house = (report.get("house") or {}).get("digest") \
        if isinstance(report.get("house"), Mapping) else None
    alpha, B = float(cfg.alpha), int(cfg.bootstrap_B)
    for k, res in enumerate(gate["results"]):
        at = f"gate.results[{k}]"
        if not isinstance(res, Mapping):
            continue
        rows = res.get("rows")
        if rows is None:
            if res.get("outcome") == "pass" and res.get("digest") == report.get("crowned"):
                out.append(f"{at}.rows")
            continue
        if not isinstance(rows, Mapping) or set(rows) != set(ROWS_ROLES) or not all(
                isinstance(v, str) and _HEX64.match(v) for v in rows.values()):
            out.append(f"{at}.rows")
            continue
        want = {"candidate": res.get("digest"), "king_d": king, "house": house}
        recs: dict[str, Mapping] = {}
        for role in ROWS_ROLES:
            data = _rows_source(bench_rows, rows[role])
            rec = None
            if data is not None and hashlib.sha256(data).hexdigest() == rows[role]:
                try:
                    rec = json.loads(data.decode("ascii"))
                except (UnicodeDecodeError, ValueError):
                    rec = None
            if not isinstance(rec, Mapping) or rec.get("session") != res.get("session") \
                    or rec.get("digest") != want[role]:
                out.append(f"{at}.rows")
                recs = {}
                break
            recs[role] = rec
        if not recs or res.get("outcome") not in ("pass", "gate_regression"):
            continue
        items = {role: _suite_items(recs[role], str(suite)) for role in ROWS_ROLES}
        lcb = res.get("lcb")
        if any(v is None for v in items.values()) or not isinstance(lcb, Mapping):
            out.append(f"{at}.lcb")
            continue
        cand = items["candidate"]
        bad = False
        for ref in GATE_REFS:
            ids = sorted(set(cand) & set(items[ref]))
            for j, m in enumerate(GATE_METRICS):
                got = _paired_lcb([items[ref][i][j] for i in ids], [cand[i][j] for i in ids],
                                  alpha=alpha, B=B, seed=f"{_GATE_SEED_TAG}|{seed}|{suite}|{m}")
                stated = (lcb.get(ref) or {}).get(m) if isinstance(lcb.get(ref), Mapping) \
                    else None
                if not (_finite(stated) and math.isfinite(got)
                        and abs(got - float(stated)) <= 1e-12):
                    bad = True
        if bad:
            out.append(f"{at}.lcb")
    return out


def _default_bench_rows(reports):
    """``<stem>_bench_rows`` beside a ``<stem>_reports`` directory, else None."""
    if isinstance(reports, (str, Path)):
        p = Path(reports)
        if p.name.endswith("_reports"):
            return p.parent / (p.name[: -len("_reports")] + "_bench_rows")
    return None


def _gate_result(d: _Paths, at: str, res, tau: float) -> tuple[str | None, str | None]:
    """``(digest, re-derived outcome)`` of one gate result (W3b2's rule: with
    four finite LCBs, ``gate_regression`` iff some LCB < ``-tau_public``,
    blocked by exactly those references; ``checkpoint_fault`` /
    ``bench_hold`` carry no LCB and are taken as stated)."""
    if not isinstance(res, dict):
        d.add(at)
        return None, None
    # ``held_from`` is optional here: whether a result must carry it is
    # ``_Settlement._held_from``'s rule.
    for k in sorted((set(res) - {"held_from", "rows"}) ^ _RESULT_KEYS):
        d.add(_key(at, k))
    dg = res.get("digest")
    if not isinstance(dg, str):
        d.add(f"{at}.digest")
        dg = None
    session = res.get("session")
    if not isinstance(session, str) or not session:
        d.add(f"{at}.session")
    blocked = res.get("blocked_by")
    if not isinstance(blocked, list) or not all(isinstance(r, str) for r in blocked) \
            or len(set(blocked)) != len(blocked):
        d.add(f"{at}.blocked_by")
        blocked = None
    outcome = res.get("outcome")
    if outcome not in GATE_OUTCOMES:
        d.add(f"{at}.outcome")
        return dg, None
    lcb = res.get("lcb")
    if outcome in ("checkpoint_fault", "bench_hold"):
        if lcb is not None:
            d.add(f"{at}.lcb")
        if blocked is not None and blocked:
            d.add(f"{at}.blocked_by")
        return dg, outcome
    if not (isinstance(lcb, dict) and set(lcb) == set(GATE_REFS)
            and all(isinstance(lcb[r], dict) and set(lcb[r]) == set(GATE_METRICS)
                    and all(_finite(lcb[r][m]) for m in GATE_METRICS) for r in GATE_REFS)):
        d.add(f"{at}.lcb")
        return dg, None
    below = [r for r in GATE_REFS if any(float(lcb[r][m]) < -tau for m in GATE_METRICS)]
    want = "gate_regression" if below else "pass"
    if outcome != want:
        d.add(f"{at}.outcome")
    if blocked is not None and set(blocked) != set(below):
        d.add(f"{at}.blocked_by")
    return dg, want


# ── the chain carried between settlements (re-derived, never stated) ────────


@dataclass
class _Chain:
    prior_sha: str | None = None          # the next receipt's prior (None: genesis)
    king: dict | None = None              # KING_D after the last settlement
    house: str | None = None              # the last report's house digest
    prior_kings: list = field(default_factory=list)
    # rev6.11 (b): no confirmation stage, so nothing is ever pending a
    # second set; kept empty so a chain copy has one shape.
    pending: dict = field(default_factory=dict)
    # digest -> an undecided cleared first verdict (a bench_hold) {lcb, king_d}
    holds: dict = field(default_factory=dict)
    # digest -> every window id it has been judged on
    used: dict = field(default_factory=dict)
    # digest -> status / reason in the last report, and its stable entry fields
    # (the house challenger's status under its ``_HOUSE_KEY`` key)
    status: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)
    # W5f: the open house challenger {digest, reveal_ts, status}, or None
    house_entry: dict | None = None


def _house_key(digest: str) -> str:
    return _HOUSE_KEY + digest


@dataclass
class _Verdict:
    index: int
    digest: str
    key: str
    stage: str
    cleared: bool
    lcb: float
    set_sha256: str
    king: str
    king_ts: int
    windows: list
    stated_lcb: float = 0.0
    lowfreq_fail: bool = False
    outcome: str = "lost"


class _Settlement:
    """One report's re-derivation against the chain so far."""

    bench_rows = None   # W3b6b: the published bench records (replay_chain)

    def __init__(self, report: dict, receipt: dict, cfg, chain: _Chain, boundary: int,
                 metagraph: Mapping | None):
        self.r = report
        self.rc = receipt
        self.cfg = cfg
        self.chain = chain
        self.b = boundary
        self.metagraph = metagraph
        self.d = _Paths()
        self.rpk = max(0, int(cfg.reward_prior_kings))
        self.house: dict | None = None
        self.king_before: dict | None = None
        self.entries: dict[str, tuple[int, dict]] = {}
        self.windows: dict[str, dict] = {}
        self.rows: dict[tuple[int, str, str], dict] = {}
        self.verdicts: list[_Verdict] = []
        self.held: list[str] = []
        self.dropped_holds: list[str] = []
        self.gate_out: dict[str, str] = {}
        self.gated = False
        # W5f: the house challenger judged this settlement (None when there
        # is none), its chain key, its status after it, and a retake.
        self.house_ch: dict | None = None
        self.hkey: str | None = None
        self.house_status: str | None = None
        self.crown_key: str | None = None
        self.retake = False
        self.crown: str | None = None
        self.king_after: dict | None = None
        self.prior_kings: list = []
        self.pay: list[tuple[str, int | None, float]] = []
        self.body_sha = _body_sha(report)
        # Re-derived by check_top; the stated values until then.
        self.seed = str(report.get("seed"))
        self.roots = None
        self.snapshot_shas: set = set()
        self.cands: list[dict] = []

    # Each step guards itself: a malformed section is a mismatch at its path.
    def _guarded(self, path: str, fn) -> None:
        try:
            fn()
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            self.d.add(path)

    def run(self) -> None:
        self._guarded("boundary", self.check_top)
        self._guarded("params", lambda: _compare(self.d, "params", expected_params(self.cfg),
                                                 self.r.get("params")))
        self._guarded("rows_sha256", self.check_rows_sha)
        self._guarded("armed_evidence", self.check_armed)
        self._guarded("house", self.check_house)
        self._guarded("king_before", self.check_king_before)
        self._guarded("house", self.derive_house_entry)
        self._guarded("entries", self.check_entries)
        self._guarded("entries", self.check_duplicates)
        self._guarded("windows", self.index_windows)
        self._guarded("rows", self.index_rows)
        self._guarded("cohorts", self.check_cohorts)
        self._guarded("draw", self.check_draw)
        self._guarded("skipped", self.check_judging_start_skips)
        self._guarded("gate", self.check_gate)
        self._guarded("verdicts", self.check_outcomes)
        self._guarded("king_after", self.check_crown)
        self._guarded("pay", self.check_pay)
        self._guarded("entries", self.check_entry_progress)

    # ── report-level fields ─────────────────────────────────────────────────

    def check_top(self) -> None:
        r = self.r
        if not _is_int(r.get("boundary")) or r["boundary"] != self.b \
                or not self.cfg.active(self.b):
            self.d.add("boundary")
        if r.get("scoring_method") != SCORING_METHOD:
            self.d.add("scoring_method")
        pool = r.get("pool")
        if not isinstance(pool, dict) or not isinstance(pool.get("sha256"), str):
            self.d.add("pool")
            self.seed = str(r.get("seed"))
            return
        self.seed = f"lane-d/1:{self.b}:{pool['sha256']}"
        if r.get("seed") != self.seed:
            self.d.add("seed")
        self.roots = pool.get("first_seen_roots")
        snaps = pool.get("snapshots")
        self.snapshot_shas = ({s.get("sha256") for s in snaps if isinstance(s, dict)}
                              if isinstance(snaps, list) else set())
        if pool["sha256"] not in self.snapshot_shas:
            self.d.add("pool.snapshots")

    def check_rows_sha(self) -> None:
        rows = self.r.get("rows")
        if not isinstance(rows, list) or _sha(_canonical(rows)) != self.r.get("rows_sha256"):
            self.d.add("rows_sha256")

    def check_armed(self) -> None:
        """rev6.6: the stated armed set is VERIFIED from the evidence the
        report carries, never re-derived from ``cfg.contexts`` (a strict
        subset with supporting evidence matches). Re-implemented from the
        plan (§1.1): a variant is armed iff ``series >= min_long_series``,
        ``sources >= 20`` and its T7 row fits. Neither field: nothing to
        verify (the pre-rev6.6 form)."""
        r = self.r
        has_set, has_ev = "armed_contexts" in r, "armed_evidence" in r
        if not has_set and not has_ev:
            return
        if not has_set:
            self.d.add("armed_contexts")
        ev = r.get("armed_evidence")
        if not isinstance(ev, dict) or set(ev) != _ARMED_EVIDENCE_KEYS:
            self.d.add("armed_evidence")
            return
        floor = ev["min_long_series"]
        if not _is_int(floor) or floor != int(self.cfg.min_long_series):
            self.d.add("armed_evidence.min_long_series")
        if not _is_int(ev["min_sources"]) or ev["min_sources"] != ARMED_MIN_SOURCES:
            self.d.add("armed_evidence.min_sources")
        for k in ("readiness_sha256", "t7_sha256"):
            if ev[k] is not None and not (isinstance(ev[k], str) and _HEX64_RE.fullmatch(ev[k])):
                self.d.add(f"armed_evidence.{k}")
        want = sorted(int(c) for c in self.cfg.contexts)
        variants = ev["variants"]
        if not isinstance(variants, list) or len(variants) != len(want):
            self.d.add("armed_evidence.variants")
            return
        armed = []
        for i, (c, v) in enumerate(zip(want, variants, strict=True)):
            at = f"armed_evidence.variants[{i}]"
            if not isinstance(v, dict) or set(v) != _ARMED_VARIANT_KEYS:
                self.d.add(at)
                continue
            if not _is_int(v["context"]) or v["context"] != c:
                self.d.add(f"{at}.context")
            ok_counts = all(_is_int(v[k]) and v[k] >= 0 for k in ("series", "sources"))
            if not ok_counts or not isinstance(v["t7_fits"], bool) \
                    or not isinstance(v["armed"], bool):
                self.d.add(at)
                continue
            gates = (v["series"] >= int(self.cfg.min_long_series)
                     and v["sources"] >= ARMED_MIN_SOURCES and v["t7_fits"])
            if v["armed"] is not gates:
                self.d.add(f"{at}.armed")
            if gates:
                armed.append(c)
        if has_set:
            stated = r["armed_contexts"]
            if not armed or not isinstance(stated, list) or stated != armed \
                    or not all(_is_int(c) for c in stated):
                self.d.add("armed_contexts")

    def check_house(self) -> None:
        h = self.r.get("house")
        want_shape = (isinstance(h, dict) and set(h) == {"digest", "hotkey", "house", "reveal_ts"}
                      and isinstance(h.get("digest"), str) and h.get("hotkey") is None
                      and h.get("house") is True and _is_int(h.get("reveal_ts")))
        if not want_shape:
            self.d.add("house")
            raise TypeError("house")
        self.house = dict(h)

    def check_king_before(self) -> None:
        """C2: the previous ``king_after``; at genesis, or when the house held
        KING_D and the house digest changed (§1.4 refresh), the report's house."""
        c = self.chain
        if self.house is None:
            raise TypeError("no house")
        if c.king is None or (c.king["house"] and self.house["digest"] != c.house):
            kb = dict(self.house)
        else:
            kb = dict(c.king)
        _compare(self.d, "king_before", kb, self.r.get("king_before"))
        self.king_before = kb

    def derive_house_entry(self) -> None:
        """W5f (§1.4 "It enters at that settlement"): a new house digest ends
        any earlier house challenger; while a miner holds KING_D it opens a
        fresh one, unless its digest is KING_D's or a prior king's. The fresh
        challenger starts with no progress of its own (first clear, hold and
        used windows), whatever a miner entry of that digest had."""
        c = self.chain
        if self.house is None or self.king_before is None:
            raise TypeError("no house")
        hd = self.house["digest"]
        he = dict(c.house_entry) if c.house_entry is not None else None
        if c.king is not None and hd != c.house:
            fresh = copy.deepcopy(c)
            old = [k for k in (*fresh.pending, *fresh.holds, *fresh.used, *fresh.status)
                   if k.startswith(_HOUSE_KEY)]
            for k in old:
                for d in (fresh.pending, fresh.holds, fresh.used, fresh.status, fresh.reasons):
                    d.pop(k, None)
            self.chain = fresh
            known = {self.king_before["digest"], *(p["digest"] for p in c.prior_kings)}
            he = None
            if not self.king_before["house"] and hd not in known:
                he = {"digest": hd, "reveal_ts": int(self.house["reveal_ts"]),
                      "status": "pending"}
        if he is not None and (he["digest"] != hd or self.king_before["house"]):
            he = None
        self.house_ch = he
        self.hkey = _house_key(hd) if he is not None else None

    def _house_open(self) -> bool:
        return self.house_ch is not None and self.house_ch["status"] in OPEN_STATUSES

    def _dg(self, key: str) -> str:
        """The digest a chain key stands for (the house challenger's is the house's)."""
        return self.house_ch["digest"] if key == self.hkey else key

    def check_duplicates(self) -> None:
        """An entry is ``duplicate_digest`` exactly when it is a copy: a new
        entry naming the house or KING_D (refused at registration), an open
        entry naming a new house (W5f), or one already closed so."""
        if self.house is None or self.king_before is None:
            raise TypeError("no house")
        hd = self.house["digest"]
        taken = {hd, self.king_before["digest"]}
        c = self.chain
        for dg, (i, e) in sorted(self.entries.items()):
            if dg not in c.entries:
                want, excused = dg in taken, False
            elif c.reasons.get(dg) == "duplicate_digest":
                want, excused = True, False
            else:
                # An open entry loses its UID before a copy closes (B8).
                want = dg == hd and c.status.get(dg) in OPEN_STATUSES
                excused = e.get("reason") == "deregistered"
            got = e.get("reason") == "duplicate_digest"
            if got != want and not (want and excused):
                self.d.add(f"entries[{i}].reason")
            if got and want and e.get("status") != "closed":
                self.d.add(f"entries[{i}].status")

    def check_entries(self) -> None:
        entries = self.r["entries"]
        if not isinstance(entries, list):
            raise TypeError("entries")
        digests = []
        for i, e in enumerate(entries):
            at = f"entries[{i}]"
            if not isinstance(e, dict) or not isinstance(e.get("digest"), str):
                self.d.add(at)
                continue
            dg = e["digest"]
            digests.append(dg)
            if dg in self.entries:
                self.d.add(f"{at}.digest")
                continue
            self.entries[dg] = (i, e)
            if not _is_int(e.get("reveal_ts")):
                self.d.add(f"{at}.reveal_ts")
            known = self.chain.entries.get(dg)
            if known is not None:
                for k in _ENTRY_STABLE:
                    if not _same(known.get(k), e.get(k)):
                        self.d.add(f"{at}.{k}")
        if digests != sorted(digests):
            self.d.add("entries")
        missing = [dg for dg in self.chain.entries if dg not in self.entries]
        if missing:
            self.d.add("entries")

    def index_windows(self) -> None:
        ws = self.r["windows"]
        if not isinstance(ws, list):
            raise TypeError("windows")
        ids = []
        max_lag = float(self.cfg.max_mirror_lag_hours)
        for k, w in enumerate(ws):
            at = f"windows[{k}]"
            if not isinstance(w, dict) or set(w) != _WINDOW_KEYS \
                    or not isinstance(w.get("window_id"), str):
                self.d.add(at)
                continue
            ids.append(w["window_id"])
            if w["window_id"] in self.windows:
                self.d.add(f"{at}.window_id")
                continue
            if not (_is_int(w["min_target_ts"]) and _is_int(w["min_first_seen_ts"])):
                self.d.add(at)
                continue
            if not isinstance(w["source"], str):
                self.d.add(f"{at}.source")
                continue
            lag = w["mirror_lag_hours"]
            if lag is not None and (not _finite(lag) or float(lag) > max_lag):
                self.d.add(f"{at}.mirror_lag_hours")
            if w["snapshot_sha256"] not in self.snapshot_shas:
                self.d.add(f"{at}.snapshot_sha256")
            self._window_record(at, w)
            self.windows[w["window_id"]] = w
        if ids != sorted(ids):
            self.d.add("windows")

    def _window_record(self, at: str, w: dict) -> None:
        """rev6.10 (a) / rev6.12 / W5j: the window's own embargo record and
        draw cell (eligibility at the cohort's bound is the cohort's check)."""
        d, h = self.d, int(self.cfg.horizon)
        e = w["embargo_hours"]
        if not _is_num(e) or not _finite(e) or float(e) < 0 \
                or float(e) > float(self.cfg.embargo_hours):
            d.add(f"{at}.embargo_hours")
            return
        e = float(e)
        if w["rule"] != ("embargo" if e > 0.0 else "embargo0"):
            d.add(f"{at}.rule")
        if not _same(w["min_release_ts"], int(w["min_first_seen_ts"]) + int(round(3600.0 * e))):
            d.add(f"{at}.min_release_ts")
        if not isinstance(w["embargo_window"], bool) or (e == 0.0 and w["embargo_window"]):
            d.add(f"{at}.embargo_window")
        freq = w["freq"]
        if not isinstance(freq, str) or w["timeframe"] != freq:
            d.add(f"{at}.timeframe")
        elif w["freq_class"] != _FREQ_CLASS.get(freq, "subdaily"):
            d.add(f"{at}.freq_class")
        if w["kind"] != "forward" or w["bin"] != "forward":
            d.add(f"{at}.bin")
        if not _same(w["h"], h) or w["window_id"].rsplit(":", 2)[-1] != f"h{h}":
            d.add(f"{at}.h")
        parts = w["window_id"].rsplit(":", 2)
        if not _is_int(w["origin"]) or len(parts) != 3 or parts[1] != str(w["origin"]) \
                or w["origin"] != w["min_target_ts"]:
            d.add(f"{at}.origin")

    def index_rows(self) -> None:
        rows = self.r["rows"]
        if not isinstance(rows, list):
            raise TypeError("rows")
        for k, row in enumerate(rows):
            at = f"rows[{k}]"
            if not isinstance(row, dict) or set(row) != _ROW_KEYS or not _is_int(row["cohort"]) \
                    or not isinstance(row["window_id"], str) or not isinstance(row["digest"], str):
                self.d.add(at)
                continue
            key = (row["cohort"], row["window_id"], row["digest"])
            if key in self.rows:
                self.d.add(at)
                continue
            self.rows[key] = row

    # ── cohorts and verdicts ────────────────────────────────────────────────

    def _expected_stage(self, m: str) -> tuple[str, str, int]:
        """C4: ``(stage, king digest, king reveal_ts)`` member ``m`` is judged
        at. rev6.11 (b): always its first (and only) verdict against KING_D."""
        kb = self.king_before
        return "first", kb["digest"], int(kb["reveal_ts"])

    def _held_candidates(self) -> None:
        """C6: the entries holding an undecided cleared first verdict. After
        a king change the hold is dropped (no D20 rule; the entry is judged
        afresh)."""
        kb = self.king_before["digest"]
        for dg in sorted(self.chain.holds):
            if dg.startswith(_HOUSE_KEY) and (dg != self.hkey or not self._house_open()):
                # Only an open house challenger's hold is re-gated.
                self.dropped_holds.append(dg)
            elif dg == self.house["digest"]:
                # A held miner entry the new pointer names is a copy of the
                # house: closed ``duplicate_digest``, its hold dropped (W5f).
                self.dropped_holds.append(dg)
            elif self.chain.holds[dg]["king_d"] != kb:
                self.dropped_holds.append(dg)
            else:
                self.held.append(dg)

    def check_cohorts(self) -> None:
        if self.king_before is None or self.house is None:
            raise TypeError("no king")
        self._held_candidates()
        cohorts = self.r["cohorts"]
        verdicts = self.r["verdicts"]
        if not isinstance(cohorts, list) or not isinstance(verdicts, list):
            raise TypeError("cohorts")
        by_digest: dict[str, int] = {}
        for j, v in enumerate(verdicts):
            if not isinstance(v, dict) or not isinstance(v.get("digest"), str):
                self.d.add(f"verdicts[{j}]")
                continue
            if v["digest"] in by_digest:
                self.d.add(f"verdicts[{j}].digest")
                continue
            by_digest[v["digest"]] = j
        if self.r.get("judged") != [v.get("digest") if isinstance(v, dict) else None
                                    for v in verdicts]:
            self.d.add("judged")
        covered: set[str] = set()
        window_union: set[str] = set()
        n_rows = 0
        for i, c in enumerate(cohorts):
            try:
                n_rows += self._cohort(i, c, by_digest, covered, window_union)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                self.d.add(f"cohorts[{i}]")
        for dg, j in by_digest.items():
            if dg not in covered:
                self.d.add(f"verdicts[{j}]")
        if n_rows != len(self.r["rows"]):
            self.d.add("rows")
        if set(self.windows) != window_union:
            self.d.add("windows")

    def check_judging_start_skips(self) -> None:
        """rev6.14 (a): a ``before_eval_delay`` skip names an entry whose
        judging start max(T_E, T_K) + eval_delay_hours is after the signed
        ``boundary_ts``; ``boundary_ts_missing`` appears only without it."""
        r, d = self.r, self.d
        bts = r.get("boundary_ts")
        if "boundary_ts" in r and not _is_int(bts):
            d.add("boundary_ts")
            return
        delay = 3600.0 * float(self.cfg.eval_delay_hours)
        kt = int(self.king_before["reveal_ts"]) if self.king_before is not None else None
        for i, sk in enumerate(r.get("skipped") or []):
            if not isinstance(sk, dict):
                continue
            why = sk.get("reason")
            if why == "boundary_ts_missing" and bts is not None:
                d.add(f"skipped[{i}].reason")
            elif why == "before_eval_delay":
                e = self.entries.get(sk.get("digest"))
                t = int(e[1]["reveal_ts"]) if e is not None else None
                if bts is None or t is None or kt is None or bts >= max(t, kt) + delay:
                    d.add(f"skipped[{i}].reason")

    def check_draw(self) -> None:
        """W5j: the draw record (present iff a cohort was drawn: a cohort and a
        boundary block hash) is bound to its block hash, the budget and the
        report's cohorts; each cohort's per-cell quota is its judged windows
        of that cell, at most the cell's capacity; the judged windows stay
        within max_windows_per_settlement."""
        r, d, cfg = self.r, self.d, self.cfg
        cohorts = r.get("cohorts") or []
        judged = {x for c in cohorts if isinstance(c, dict)
                  for x in (c.get("window_ids") or []) if isinstance(x, str)}
        if len(judged) > int(cfg.max_windows_per_settlement):
            d.add("cohorts")
        gate = r.get("gate")
        if "draw" not in r:
            # A gated settlement had its block hash: a judged cohort was drawn.
            if cohorts and isinstance(gate, dict):
                d.add("draw")
            return
        dr = r["draw"]
        if not isinstance(dr, dict) or set(dr) != _DRAW_KEYS or not cohorts:
            d.add("draw")
            return
        bh = dr["boundary_block_hash"]
        if not isinstance(bh, str) or not _BLOCK_HASH_RE.match(bh) or (
                isinstance(gate, dict) and gate.get("block_hash") != bh):
            d.add("draw.boundary_block_hash")
        if dr["seed"] != f"{bh}:{_DRAW_SEED_TAG}":
            d.add("draw.seed")
        if not _same(dr["max_windows_per_settlement"], int(cfg.max_windows_per_settlement)):
            d.add("draw.max_windows_per_settlement")
        dcs = dr["cohorts"]
        if not isinstance(dcs, list) or len(dcs) != len(cohorts):
            d.add("draw.cohorts")
            return
        for i, (dc, c) in enumerate(zip(dcs, cohorts, strict=True)):
            at = f"draw.cohorts[{i}]"
            if not isinstance(dc, dict) or set(dc) != _DRAW_COHORT_KEYS or not isinstance(c, dict):
                d.add(at)
                continue
            if not _same(dc["index"], i):
                d.add(f"{at}.index")
            quota, cap = dc["quota"], dc["capacity"]
            if not isinstance(quota, dict) or not isinstance(cap, dict):
                d.add(at)
                continue
            count: dict[str, int] = {}
            for x in c.get("window_ids") or []:
                w = self.windows.get(x)
                if w is not None:
                    k = f"{w['timeframe']}|{w['bin']}"
                    count[k] = count.get(k, 0) + 1
            for k in sorted(set(quota) | set(count) | set(cap), key=str):
                q, n = quota.get(k, 0), cap.get(k, 0)
                if not (_is_int(q) and _is_int(n)) or q != count.get(k, 0) or q > n:
                    d.add(f"{at}.quota")
            if not isinstance(dc["candidates_sha256"], str) \
                    or not _HEX64_RE.match(dc["candidates_sha256"]):
                d.add(f"{at}.candidates_sha256")

    def _cohort(self, i: int, c, by_digest: dict, covered: set, window_union: set) -> int:
        """Re-judge cohort ``i``; returns the number of rows it accounts for."""
        at = f"cohorts[{i}]"
        cfg, d = self.cfg, self.d
        if not isinstance(c, dict):
            d.add(at)
            return 0
        bars_house = self.house["digest"]
        for k in sorted(set(c) ^ (_COHORT_KEYS | {"house_digest"})):
            if k != "house_digest":
                d.add(_key(at, k))
        if c.get("index") != i:
            d.add(f"{at}.index")
        members = c["members"]
        if not isinstance(members, list) or not members \
                or not all(isinstance(m, str) for m in members) \
                or len(set(members)) != len(members):
            d.add(f"{at}.members")
            return 0
        # W5f: the house challenges a miner KING_D alone in its own cohort
        # (the house digest is never a miner member: a miner copy is closed).
        is_house = bars_house in members
        if is_house and (members != [bars_house] or not self._house_open()):
            d.add(f"{at}.members")
            return 0
        keys = [self.hkey] if is_house else list(members)
        for m in members:
            if (not is_house and m not in self.entries) or m not in by_digest:
                d.add(f"{at}.members")
                return 0
        for m, key in zip(members, keys, strict=True):
            if key in self.held:
                d.add(f"verdicts[{by_digest[m]}]")
        reveal = ({bars_house: int(self.house_ch["reveal_ts"])} if is_house
                  else {m: int(self.entries[m][1]["reveal_ts"]) for m in members})
        if members != sorted(members, key=lambda m: (reveal[m], m)):
            d.add(f"{at}.members")
        # C4: the stage and king every member is judged at.
        expect = [self._expected_stage(k) for k in keys]
        stage, kd, king_ts = expect[0]
        if any(e != expect[0] for e in expect[1:]):
            d.add(f"{at}.members")
        if c.get("stage") != stage:
            d.add(f"{at}.stage")
        if c.get("king_digest") != kd:
            d.add(f"{at}.king_digest")
        if not _same(c.get("king_reveal_ts"), king_ts):
            d.add(f"{at}.king_reveal_ts")
        t_max = max([king_ts, *reveal.values()])
        if not _same(c.get("t_max"), t_max):
            d.add(f"{at}.t_max")
        # rev6.14 (a): never frozen before the judging start j = t_max +
        # eval_delay_hours, against the signed boundary timestamp.
        bts = self.r.get("boundary_ts")
        if not _is_int(bts):
            d.add("boundary_ts")
        elif bts < t_max + 3600.0 * float(cfg.eval_delay_hours):
            d.add(f"{at}.frozen_before_eval_delay")
        wids = c["window_ids"]
        if not isinstance(wids, list) or not all(isinstance(w, str) for w in wids):
            d.add(f"{at}.window_ids")
            return 0
        window_union.update(wids)
        bound = t_max + 3600.0 * float(cfg.forward_margin_hours)
        ok = len(set(wids)) == len(wids)
        n_emb = 0
        for wid in wids:
            w = self.windows.get(wid)
            if w is None or not _finite(w.get("embargo_hours")):
                ok = False
                continue
            # rev6.10 (a) / rev6.12 (a), every cadence: eligible at the feed
            # bound b - e (released after b, data time after b - e); fresh
            # iff it also passes at e = 0, else an embargo window.
            e = 3600.0 * float(w["embargo_hours"])
            if not (w["min_first_seen_ts"] + e > bound and w["min_target_ts"] > bound - e):
                ok = False
            fresh = w["min_target_ts"] > bound and w["min_first_seen_ts"] > bound
            if w["embargo_window"] is fresh:
                d.add(f"{at}.window_ids")
            n_emb += 0 if fresh else 1
        # rev6.12 (b): the split is signed and the fresh share holds.
        if not _same(c.get("embargo_windows"), n_emb):
            d.add(f"{at}.embargo_windows")
        if not _same(c.get("fresh_windows"), len(wids) - n_emb):
            d.add(f"{at}.fresh_windows")
        if wids and (len(wids) - n_emb) < float(cfg.min_fresh_window_share) * len(wids):
            d.add(f"{at}.embargo_windows")
        for key in keys:
            if self.chain.used.get(key, set()) & set(wids):
                ok = False
        if not ok:
            d.add(f"{at}.window_ids")
        clusters = [self.windows[w]["source"] for w in wids if w in self.windows]
        if len(wids) < int(cfg.min_windows) or len(set(clusters)) < int(cfg.min_clusters):
            d.add(f"{at}.window_ids")
        set_sha = _sha("\n".join(wids).encode())
        if c.get("set_sha256") != set_sha:
            d.add(f"{at}.set_sha256")
        if self.roots is not None and c.get("first_seen_roots") != self.roots:
            d.add(f"{at}.first_seen_roots")
        refused = c.get("refused")
        if not isinstance(refused, dict) or not all(_is_int(n) and n >= 0
                                                    for n in refused.values()):
            d.add(f"{at}.refused")
        # The house is a second bar exactly when a miner holds the cohort's
        # KING_D and the house is not the challenger (never its own bar).
        bars = [kd] if kd == bars_house or is_house else [kd, bars_house]
        if len(bars) == 1:
            if "house_digest" in c:
                d.add(f"{at}.house_digest")
        elif c.get("house_digest") != bars_house:
            d.add(f"{at}.house_digest")
        if len(clusters) != len(wids) or not wids:
            return len(wids) * (len(bars) + len(members))
        scores = {}
        for dg in [*bars, *members]:
            crps, mase = [], []
            for wid in wids:
                row = self.rows.get((i, wid, dg))
                if row is None or not (_finite(row["crps"]) and _finite(row["mase"])
                                       and row["crps"] >= 0 and row["mase"] >= 0):
                    d.add("rows")
                    return len(wids) * (len(bars) + len(members))
                crps.append(float(row["crps"]))
                mase.append(float(row["mase"]))
            scores[dg] = (np.asarray(crps, dtype=np.float64), np.asarray(mase, dtype=np.float64))
        judged = {bar: _judge(cfg, scores, bar, members, clusters, self.seed) for bar in bars}
        for n, (m, key) in enumerate(zip(members, keys, strict=True)):
            j = by_digest[m]
            covered.add(m)
            lcb, gain, cleared = judged[kd][n]
            if len(bars) > 1:
                cleared = cleared and judged[bars_house][n][2]
            v = verdicts_of(self.r)[j]
            # rev6.12 (b): the fresh and embargo sub-scores from the rows.
            for name, emb in (("fresh", False), ("embargo", True)):
                idx = [x for x, wid in enumerate(wids) if self.windows[wid]["embargo_window"] is emb]
                if not _subscore_ok(v.get(name), scores[m], idx):
                    d.add(f"verdicts[{j}].{name}")
            lf_fail = self._verdict(j, v, key, stage, kd, i, (lcb, gain),
                                    judged[bars_house][n] if len(bars) > 1 else None, cleared)
            stated = float(v["lcb"]) if _is_num(v.get("lcb")) else lcb
            self.verdicts.append(_Verdict(index=j, digest=m, key=key, stage=stage,
                                          cleared=cleared,
                                          lcb=lcb, set_sha256=set_sha, king=kd,
                                          king_ts=king_ts, windows=list(wids),
                                          stated_lcb=stated, lowfreq_fail=lf_fail))
        return len(wids) * (len(bars) + len(members))

    def _verdict(self, j: int, v: dict, m: str, stage: str, kd: str, cohort: int,
                 vs_king: tuple[float, float], vs_house, cleared: bool) -> bool:
        """Check verdict ``j``; returns whether its A10 slice re-derives
        ``fail`` (a final loss that never reaches the gate)."""
        at = f"verdicts[{j}]"
        d = self.d
        keys = _VERDICT_KEYS | ({"house"} if vs_house is not None else set())
        for k in sorted(set(v) ^ keys):
            d.add(_key(at, k))
        if v.get("king_digest") != kd:
            d.add(f"{at}.king_digest")
        if v.get("stage") != stage:
            d.add(f"{at}.stage")
        if not _same(v.get("cohort"), cohort):
            d.add(f"{at}.cohort")
        if not _close(v.get("lcb"), vs_king[0]):
            d.add(f"{at}.lcb")
        if not _close(v.get("gain"), vs_king[1]):
            d.add(f"{at}.gain")
        if v.get("cleared") is not cleared:
            d.add(f"{at}.cleared")
        if vs_house is not None:
            hv = v.get("house")
            if not isinstance(hv, dict):
                d.add(f"{at}.house")
            else:
                for k in sorted(set(hv) ^ {"digest", "lcb", "gain", "cleared"}):
                    d.add(_key(f"{at}.house", k))
                if hv.get("digest") != self.house["digest"]:
                    d.add(f"{at}.house.digest")
                if not _close(hv.get("lcb"), vs_house[0]):
                    d.add(f"{at}.house.lcb")
                if not _close(hv.get("gain"), vs_house[1]):
                    d.add(f"{at}.house.gain")
                if hv.get("cleared") is not vs_house[2]:
                    d.add(f"{at}.house.cleared")
        # rev6.11 (b): a cleared first verdict is the crowning candidate.
        return self._lowfreq(at, v.get("lowfreq"), cleared)

    def _lowfreq(self, at: str, lf, candidate: bool) -> bool:
        """W3b5 (A10): a crowning candidate's low-frequency slice is
        ``no_windows`` (empty, no LCB) or ``pass``/``fail`` with four LCBs,
        ``fail`` iff some LCB < ``-tau_public``; every other verdict is
        ``not_candidate`` with an empty slice. Returns the re-derived ``fail``."""
        d, cfg = self.d, self.cfg
        tau = float(cfg.tau_public)
        base = {"result", "freqs", "n_windows", "tau_public"}
        if not isinstance(lf, dict) or not base <= set(lf) <= base | {"lcb"} \
                or lf.get("freqs") != [str(f) for f in cfg.lowfreq_freqs] \
                or not _same(lf.get("tau_public"), tau) or not _is_int(lf.get("n_windows")):
            d.add(f"{at}.lowfreq")
            return False
        result, n = lf.get("result"), lf["n_windows"]
        if not candidate:
            if result != "not_candidate" or n != 0 or "lcb" in lf:
                d.add(f"{at}.lowfreq")
            return False
        if result == "no_windows":
            if n != 0 or "lcb" in lf:
                d.add(f"{at}.lowfreq")
            return False
        lcb = lf.get("lcb")
        if result not in ("pass", "fail") or n <= 0 or not (
                isinstance(lcb, dict) and set(lcb) == set(GATE_REFS)
                and all(isinstance(lcb[r], dict) and set(lcb[r]) == set(GATE_METRICS)
                        and all(_finite(lcb[r][m]) for m in GATE_METRICS) for r in GATE_REFS)):
            d.add(f"{at}.lowfreq")
            return result == "fail"
        fail = any(float(lcb[r][m]) < -tau for r in GATE_REFS for m in GATE_METRICS)
        if result != ("fail" if fail else "pass"):
            d.add(f"{at}.lowfreq.result")
        return fail

    # ── the gate, outcomes and the crown ────────────────────────────────────

    def _candidates(self) -> list[dict]:
        out = [{"digest": v.digest, "key": v.key, "lcb": v.lcb, "held": False}
               for v in self.verdicts
               if v.cleared and not v.lowfreq_fail]
        out += [{"digest": self._dg(k), "key": k, "lcb": float(self.chain.holds[k]["lcb"]),
                 "held": True} for k in self.held]
        for c in out:
            if c["key"] == self.hkey:
                c["reveal_ts"] = int(self.house_ch["reveal_ts"])
            elif c["digest"] in self.entries:
                c["reveal_ts"] = int(self.entries[c["digest"]][1]["reveal_ts"])
            else:
                c["reveal_ts"] = 0
                self.d.add("entries")
        return out

    def check_gate(self) -> None:
        cands = self._candidates()
        self.cands = cands
        if "gate" not in self.r:
            return
        self.gated = True
        g = self.r["gate"]
        if not isinstance(g, dict):
            raise TypeError("gate")
        if not cands:
            # The gate runs only over crowning candidates.
            self.d.add("gate")
        for k in sorted(set(g) ^ _GATE_KEYS):
            self.d.add(_key("gate", k))
        if not _is_int(g.get("block")) or g["block"] != self.b:
            self.d.add("gate.block")
        drawn = gate_draw(self.cfg, g.get("block_hash"), g.get("sentinel"))
        if drawn is None:
            self.d.add("gate.block_hash")
        else:
            if g.get("suite") != drawn[0]:
                self.d.add("gate.suite")
            if g.get("eligible") != drawn[1]:
                self.d.add("gate.eligible")
        results = g.get("results")
        if not isinstance(results, list) or not results:
            self.d.add("gate.results")
            return
        tau = float(self.cfg.tau_public)
        names = {c["digest"]: c for c in cands}
        for k, res in enumerate(results):
            at = f"gate.results[{k}]"
            dg, outcome = _gate_result(self.d, at, res, tau)
            if dg is None:
                continue
            if dg in self.gate_out:
                self.d.add(f"{at}.digest")
                continue
            # C6: a result is a crowning candidate's: a cleared first verdict
            # of this settlement, or a held candidate whose entry was still
            # on hold.
            c = names.get(dg)
            if c is None or (c["held"] and self.chain.status.get(c["key"]) != "hold"):
                self.d.add(f"{at}.digest")
                continue
            self._held_from(at, res, c)
            self.gate_out[dg] = outcome if outcome is not None else "gate_regression"
        for c in cands:
            if c["digest"] not in self.gate_out:
                self.d.add("gate.results")
        # W3b6b: the bound bench records, re-derived from the published bytes.
        if self.bench_rows is not None:
            for p in check_gate_rows(self.r, self.bench_rows, self.cfg):
                self.d.add(p)

    def _held_from(self, at: str, res: dict, cand: dict) -> None:
        """RV7 B2: a held candidate's result names its hold, and only a held
        candidate's does. The hold is re-derived from this chain: the earlier
        verified report (by its body digest) in which the digest's first
        verdict cleared as ``cleared`` with the gate absent or
        ``bench_hold``, that cohort's ``set_sha256``, the verdict's KING_D
        ``lcb`` and the KING_D it was taken under. An invented or altered
        ``held_from`` differs from it."""
        if not cand["held"]:
            if "held_from" in res:
                self.d.add(f"{at}.held_from")
            return
        hold = self.chain.holds[cand["key"]]
        want = {k: hold[k] for k in ("boundary", "report_sha256", "set_sha256", "lcb", "king_d")}
        _compare(self.d, f"{at}.held_from", want, res.get("held_from"))

    def _gate_outcome(self, dg: str) -> str:
        """The candidate's gate outcome; without a gate record every
        candidate is ``bench_hold`` (fail closed)."""
        if not self.gated:
            return "bench_hold"
        return self.gate_out.get(dg, "bench_hold")

    def check_outcomes(self) -> None:
        """The crown (the highest-LCB passing candidate, then the earlier
        reveal, then the digest) and every verdict's outcome."""
        cands = self.cands
        passing = [c for c in cands if self._gate_outcome(c["digest"]) == "pass"]
        passing.sort(key=lambda c: (-c["lcb"], c["reveal_ts"], c["digest"]))
        self.crown_key = passing[0]["key"] if passing else None
        # W5f: a passing house challenger retakes KING_D; ``crowned`` names
        # a miner crowning only, so it stays null.
        self.retake = self.crown_key is not None and self.crown_key == self.hkey
        self.crown = None if self.retake or not passing else passing[0]["digest"]
        verdicts = verdicts_of(self.r)
        for v in self.verdicts:
            if not v.cleared or v.lowfreq_fail:
                v.outcome = "lost"
            else:
                o = self._gate_outcome(v.digest)
                if o in ("gate_regression", "checkpoint_fault"):
                    v.outcome = "lost"
                elif o == "pass" and v.key == self.crown_key:
                    v.outcome = "crowned"
                else:
                    v.outcome = "cleared"
            if verdicts[v.index].get("outcome") != v.outcome:
                self.d.add(f"verdicts[{v.index}].outcome")

    def check_crown(self) -> None:
        if self.r.get("crowned") != self.crown or "crowned" not in self.r:
            self.d.add("crowned")
        kb = self.king_before
        if kb is None:
            raise TypeError("no king")
        if self.retake:
            self.king_after = {"digest": self.house["digest"], "hotkey": None, "house": True,
                               "reveal_ts": self.house["reveal_ts"]}
        elif self.crown is not None:
            e = self.entries[self.crown][1]
            self.king_after = {"digest": self.crown, "hotkey": e["hotkey"], "house": False,
                               "reveal_ts": e["reveal_ts"]}
        else:
            self.king_after = dict(kb)
        _compare(self.d, "king_after", self.king_after, self.r.get("king_after"))
        # C3: a crown (or a house retake) over a miner king pushes it onto
        # the prior kings.
        prior = [dict(k) for k in self.chain.prior_kings]
        if self.crown_key is not None and not kb["house"]:
            prior = [dict(kb), *prior][:self.rpk]
        self.prior_kings = prior
        _compare(self.d, "prior_kings", prior, self.r.get("prior_kings"))

    def check_pay(self) -> None:
        """C5: the court is KING_D then the prior kings by recency; a slot's
        uid comes from the metagraph when given (else the stated one, taken
        as signed), and a slot without a uid burns with its reason."""
        pay = self.r["pay"]
        if not isinstance(pay, dict) or set(pay) != {"share", "court"}:
            self.d.add("pay")
        if not isinstance(pay, dict):
            raise TypeError("pay")
        if not _is_num(pay.get("share")) or float(pay["share"]) != float(self.cfg.share):
            self.d.add("pay.share")
        if self.king_after is None:
            raise TypeError("no king")
        stated = pay.get("court")
        stated = stated if isinstance(stated, list) else []
        want = []
        for i, k in enumerate([self.king_after, *self.prior_kings[:self.rpk]]):
            hk = None if k["house"] else k["hotkey"]
            if hk is None:
                uid = None
            elif self.metagraph is not None:
                got = self.metagraph.get(hk)
                uid = got if _is_int(got) else None
            else:
                slot = stated[i] if i < len(stated) and isinstance(stated[i], dict) else {}
                uid = slot.get("uid") if _is_int(slot.get("uid")) else None
            burn = None if uid is not None else ("house_king" if hk is None else "deregistered")
            want.append({"digest": k["digest"], "hotkey": hk, "uid": uid, "burn_reason": burn})
        _compare(self.d, "pay.court", want, pay.get("court"))
        shares = court_shares(self.cfg, len(want))
        self.pay = [(s["digest"], s["uid"], sh) for s, sh in zip(want, shares, strict=True)]

    def check_entry_progress(self) -> None:
        """Each decided entry's status and reason as its verdict or gate
        result leaves it."""
        want: dict[str, tuple[str, str]] = {}
        for v in self.verdicts:
            if v.outcome == "crowned":
                want[v.key] = ("crowned", "crowned")
            elif v.outcome == "lost":
                o = self._gate_outcome(v.digest) if v.cleared else None
                if v.cleared and v.lowfreq_fail:
                    want[v.key] = ("lost", "lowfreq_regression")
                elif o == "checkpoint_fault":
                    want[v.key] = ("closed", "checkpoint_fault")
                elif o == "gate_regression":
                    want[v.key] = ("lost", "gate_regression")
                else:
                    want[v.key] = ("lost", "lost_duel")
            elif self._gate_outcome(v.digest) == "bench_hold":
                want[v.key] = ("hold", "bench_hold")
        for key in self.held:
            o = self._gate_outcome(self._dg(key))
            if key == self.crown_key:
                want[key] = ("crowned", "crowned")
            elif o == "bench_hold":
                want[key] = ("hold", "bench_hold")
            elif o == "gate_regression":
                want[key] = ("lost", "gate_regression")
            elif o == "checkpoint_fault":
                want[key] = ("closed", "checkpoint_fault")
        # W5f: the house challenger is not a listed entry; its status is
        # carried in the chain (and bound by the state snapshot).
        if self.hkey is not None:
            if self.hkey in self.dropped_holds:
                self.house_status = "pending"
            if self.hkey in want:
                self.house_status = want.pop(self.hkey)[0]
        for dg in sorted(want):
            if dg not in self.entries:
                self.d.add("entries")
                continue
            i, e = self.entries[dg]
            if e.get("status") != want[dg][0]:
                self.d.add(f"entries[{i}].status")
            if e.get("reason") != want[dg][1]:
                self.d.add(f"entries[{i}].reason")

    # ── receipt and state ───────────────────────────────────────────────────

    def check_receipt(self) -> None:
        """C1: the receipt binds this report's boundary and body and links
        from the previous posterior (the W7c genesis digest at first)."""
        rc, d = self.rc, self.d
        for k in sorted(set(rc) ^ set(RECEIPT_KEYS), key=str):
            d.add(_key("receipt", k))
        if rc.get("schema") != RECEIPT_SCHEMA:
            d.add("receipt.schema")
        if not _same(rc.get("boundary"), self.r.get("boundary")):
            d.add("receipt.boundary")
        for k in ("prior_state_sha256", "posterior_state_sha256", "report_sha256"):
            if not isinstance(rc.get(k), str) or not _HEX64_RE.fullmatch(rc[k]):
                d.add(f"receipt.{k}")
        if rc.get("report_sha256") != self.body_sha:
            d.add("receipt.report_sha256")
        prior = self.chain.prior_sha
        if prior is None:
            prior = _genesis_sha(self.house) if self.house is not None else None
        if prior is None or rc.get("prior_state_sha256") != prior:
            d.add("receipt.prior_state_sha256")

    def check_state(self, raw) -> None:
        """A published snapshot binds the posterior: its sha256 is the
        receipt's, its bytes are canonical, and its consensus projection is
        the one this report and receipt re-derive."""
        d = self.d
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            d.add("state")
            return
        raw = bytes(raw)
        if _sha(raw) != self.rc.get("posterior_state_sha256"):
            d.add("receipt.posterior_state_sha256")
        try:
            state = _strict_json(raw)
        except ValueError:
            d.add("state")
            return
        if not isinstance(state, dict) or _canonical(state) != raw:
            d.add("state")
            if not isinstance(state, dict):
                return
        if state.get("schema") != STATE_SCHEMA:
            d.add("state.schema")
        if not _same(state.get("boundary"), self.b):
            d.add("state.boundary")
        if self.house is not None:
            _compare(d, "state.house", {"digest": self.house["digest"],
                                        "reveal_ts": self.house["reveal_ts"]},
                     state.get("house"))
        king = state.get("king")
        if self.king_after is not None:
            got = ({k: king.get(k) for k in self.king_after} if isinstance(king, dict)
                   and set(self.king_after) <= set(king) else king)
            _compare(d, "state.king", self.king_after, got)
        pk = state.get("prior_kings")
        got_pk = ([{k: p.get(k) for k in ("digest", "hotkey", "house", "reveal_ts")}
                   if isinstance(p, dict) else p for p in pk] if isinstance(pk, list) else pk)
        _compare(d, "state.prior_kings", self.prior_kings, got_pk)
        ents = state.get("entries")
        if not isinstance(ents, dict) or set(ents) != set(self.entries):
            d.add("state.entries")
        else:
            for dg in sorted(ents):
                rec = ents[dg]
                at = _key("state.entries", dg)
                e = self.entries[dg][1]
                if not isinstance(rec, dict):
                    d.add(at)
                    continue
                for k in ("status", "reason"):
                    if not _same(rec.get(k), e.get(k)):
                        d.add(f"{at}.{k}" if at != "state.entries" else at)
        # W5f: the house challenger's record is present exactly while there
        # is one (its digest and T the house's; its status when this
        # settlement decided it).
        he = self.house_after()
        got_he = state.get("house_entry")
        if he is None:
            if "house_entry" in state:
                d.add("state.house_entry")
        elif not isinstance(got_he, dict):
            d.add("state.house_entry")
        else:
            keys = ("digest", "reveal_ts", "status") if self.house_status is not None \
                else ("digest", "reveal_ts")
            _compare(d, "state.house_entry", {k: he[k] for k in keys},
                     {k: got_he.get(k) for k in keys})
        _compare(d, "state.last_report",
                 {"boundary": self.b, "rows_sha256": self.r.get("rows_sha256"),
                  "report_sha256": self.body_sha,
                  "prior_state_sha256": self.rc.get("prior_state_sha256")},
                 state.get("last_report"))

    # ── the chain after this settlement ─────────────────────────────────────

    def next_chain(self) -> _Chain:
        c = copy.deepcopy(self.chain)
        c.prior_sha = self.rc.get("posterior_state_sha256")
        c.king = dict(self.king_after) if self.king_after is not None else c.king
        c.house = self.house["digest"] if self.house is not None else c.house
        c.prior_kings = [dict(k) for k in self.prior_kings]
        for dg in self.dropped_holds:
            c.holds.pop(dg, None)
            c.pending.pop(dg, None)
        for v in self.verdicts:
            c.used.setdefault(v.key, set()).update(v.windows)
        # A hold keeps its first record: the settlement (and its verified
        # report body) whose first verdict cleared and the gate first held.
        decided = [(v.key, {"boundary": self.b, "report_sha256": self.body_sha,
                            "set_sha256": v.set_sha256, "lcb": v.stated_lcb,
                            "king_d": self.king_before["digest"]})
                   for v in self.verdicts
                   if v.cleared and not v.lowfreq_fail]
        decided += [(k, c.holds[k]) for k in self.held if k in c.holds]
        for key, record in decided:
            o = self._gate_outcome(self._dg(key))
            if o == "bench_hold":
                c.holds[key] = c.holds.get(key) or dict(record)
                continue
            c.holds.pop(key, None)
            if o != "pass" or key == self.crown_key:
                c.pending.pop(key, None)
        for v in self.verdicts:
            if v.outcome == "lost":
                c.pending.pop(v.key, None)
                c.holds.pop(v.key, None)
        c.status = {dg: e.get("status") for dg, (_, e) in sorted(self.entries.items())}
        c.reasons = {dg: e.get("reason") for dg, (_, e) in sorted(self.entries.items())}
        c.entries = {dg: {k: e.get(k) for k in _ENTRY_STABLE}
                     for dg, (_, e) in sorted(self.entries.items())}
        # W5f: the house challenger carries on, or ends with a retake.
        c.house_entry = self.house_after()
        for d in (c.pending, c.holds, c.used):
            for k in [k for k in d if k.startswith(_HOUSE_KEY)]:
                if c.house_entry is None or k != _house_key(c.house_entry["digest"]):
                    d.pop(k, None)
        if c.house_entry is not None:
            c.status[_house_key(c.house_entry["digest"])] = c.house_entry["status"]
        return c

    def house_after(self) -> dict | None:
        """The house challenger after this settlement (None after a retake)."""
        if self.house_ch is None or self.retake:
            return None
        he = dict(self.house_ch)
        if self.house_status is not None:
            he["status"] = self.house_status
        return he


def verdicts_of(report: dict) -> list:
    v = report.get("verdicts")
    return v if isinstance(v, list) else []


def _signature_paths(report: dict, signer_ss58: str) -> list[str]:
    """The pinned signer's sr25519 signature over the canonical body.
    Raises ``SignatureVerifierUnavailable`` when it cannot be checked."""
    out = []
    if report.get("signer") != signer_ss58:
        out.append("signer")
    sig = report.get("signature")
    body = _canonical({k: v for k, v in report.items() if k != "signature"})
    if not isinstance(sig, str) or not _SIG_RE.fullmatch(sig) \
            or not verify_hotkey(str(signer_ss58), body, sig):
        out.append("signature")
    return out


def _confirmation_stage_present(report: dict) -> bool:
    """rev6.11 (b): does the signed report carry a confirmation stage?"""
    def items(key):
        v = report.get(key)
        return [x for x in v if isinstance(x, Mapping)] if isinstance(v, list) else []

    for c in items("cohorts"):
        if "stage" in c and c["stage"] != "first":
            return True
    for v in items("verdicts"):
        if ("stage" in v and v["stage"] != "first") or "confirmation" in v:
            return True
    return any("confirm_pending" in (e.get("status"), e.get("reason")) for e in items("entries"))


def _unverifiable(boundary, reason: str, **kw) -> SettlementReplay:
    return SettlementReplay(boundary=boundary, status=UNVERIFIABLE, reason=reason, **kw)


def _replay_one(reports, receipt, cfg, chain: _Chain, signer_ss58: str, uid_of, states,
                bench_rows=None) -> tuple[SettlementReplay, _Chain]:
    b = receipt.get("boundary") if isinstance(receipt, Mapping) else None
    if not _is_int(b):
        return _unverifiable(None, "receipt_malformed"), chain
    report, why = _load_report(reports, b)
    if why is not None:
        return _unverifiable(b, why), chain
    if report.get("schema") != REPORT_SCHEMA:
        return _unverifiable(b, "unknown_schema"), chain
    body_sha = _body_sha(report)
    try:
        sig_paths = _signature_paths(report, signer_ss58)
    except SignatureVerifierUnavailable:
        return _unverifiable(b, "signature_unverifiable", report_sha256=body_sha), chain
    if not sig_paths and _confirmation_stage_present(report):
        return _unverifiable(b, "confirmation_stage_present", report_sha256=body_sha), chain
    metagraph = None
    if uid_of is not None and b in uid_of and isinstance(uid_of[b], Mapping):
        metagraph = uid_of[b]
    s = _Settlement(report, dict(receipt), cfg, chain, b, metagraph)
    s.bench_rows = bench_rows
    for p in sig_paths:
        s.d.add(p)
    s.run()
    s.check_receipt()
    if states is not None and b in states:
        s.check_state(states[b])
    status = MISMATCH if s.d.items else MATCH
    result = SettlementReplay(
        boundary=b, status=status, mismatches=tuple(s.d.items), reason=None,
        crowned=s.crown,
        king_after=s.king_after["digest"] if s.king_after is not None else None,
        outcomes=MappingProxyType({v.digest: v.outcome for v in s.verdicts}),
        gate=MappingProxyType(dict(sorted(s.gate_out.items()))),
        pay=tuple(s.pay), report_sha256=body_sha,
        posterior_state_sha256=(receipt.get("posterior_state_sha256")
                                if status == MATCH else None))
    return result, s.next_chain()


def replay_chain(reports, receipts, cfg, *, signer_ss58: str,
                 uid_of: Mapping[int, Mapping[str, int]] | None = None,
                 states: Mapping[int, bytes] | None = None,
                 bench_rows=None) -> tuple[SettlementReplay, ...]:
    """Replay the lane D receipt chain (see the module docstring).

    ``reports`` is the published reports directory (``lane_d_report_<b>.json``
    files) or a mapping ``{boundary: report}``; ``receipts`` the receipt
    records in settlement order (one result each, same order); ``cfg`` the
    parsed ``[lane_d]`` config in force; ``signer_ss58`` the pinned signer
    (the report's own ``signer`` is never trusted); ``uid_of`` the metagraph
    ``{hotkey: uid}`` per boundary; ``states`` the published state snapshot
    bytes per boundary; bench_rows (W3b6b) the published bench records,
    a {sha256: bytes} mapping or a directory of <sha256>.json files
    (default: the <stem>_bench_rows directory beside a <stem>_reports
    reports directory; with a mapping of reports and no bench_rows the
    record checks are skipped, as before W3b6b). Pure: nothing is written
    and no input is mutated."""
    if isinstance(receipts, (str, bytes, Mapping)):
        raise TypeError("receipts must be a sequence of receipt records")
    chain = _Chain()
    if bench_rows is None:
        bench_rows = _default_bench_rows(reports)
    out: list[SettlementReplay] = []
    trusted = True
    for receipt in [copy.deepcopy(r) for r in receipts]:
        if not trusted:
            b = receipt.get("boundary") if isinstance(receipt, Mapping) else None
            out.append(_unverifiable(b if _is_int(b) else None, "prior_unverified"))
            continue
        res, chain = _replay_one(reports, receipt, cfg, chain, signer_ss58, uid_of, states,
                                 bench_rows)
        out.append(res)
        trusted = res.status == MATCH
    return tuple(out)


__all__ = ["MATCH", "MISMATCH", "RECEIPT_SCHEMA", "REPORT_SCHEMA", "STATE_SCHEMA", "STATUSES",
           "UNVERIFIABLE", "UNVERIFIABLE_REASONS", "SettlementReplay", "check_gate_rows",
           "combined_log_ratio", "court_shares", "expected_params", "gate_draw", "replay_chain"]
