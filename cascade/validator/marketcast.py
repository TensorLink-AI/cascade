"""MarketCast judging on the validator (DEC-CA-0064).

An ``mkt:<suite>`` lane-C entry is judged here instead of on the lane-C
pool, in its OWN pipeline that never holds lane A, lane B or non-MarketCast
lane C (review fixes 2026-10-19):

1. **Registration** (every validator, at the settlement that first sees the
   entry). Pre-checks (:func:`mkt_prechecks`, pure): the target is an
   enabled ``[lane_c]`` target of an enabled suite (``target``); the entry's
   trainer-signed ``marketcast`` record names the PINNED base (``base_ref``
   + ``base_sha256`` — anything else is ``stale``) and carries the leg's
   dispatch time ``train_time``; the hotkey's commitment as of
   ``train_block`` is ``metro-v1:ft:mkt:<suite>:hippius:<ref>``
   (``binding``). Exact-identity dedup (DEC-CA-0008 spirit): a fine-tune
   whose weights' sha256 equals an EARLIER-committed entry's (any target;
   seniority = the commitment block, never a UID) is a ``duplicate``. The
   survivors — this settlement's new entries plus every MarketCast entry
   left pending / deferred — become ONE MarketCast round (``mkt_backlog``
   in :class:`~cascade.shared.lane_c.LaneCState`).
2. **Rows** (the DEC-CA-0063 pattern). The MAIN validator (``[netconfig]
   owner_hotkey``) computes each round's rows once, on its GPU: the signed
   snapshot active at the round's block, the entry's DRAWN windows
   (DEC-CA-0074, :func:`entry_draw`: per (asset, series, horizon) the
   ``backtest_windows_per_cell`` best-ranked by the round block's hash, lying
   wholly in the ``embargo_eval_days`` before the TRAINER-stamped dispatch —
   bars no leg reads — in the judge or confirm half of a fixed public
   partition; plus the ``forward_horizons``' closed forward windows after
   ``train_time + start_margin_seconds`` and, on a confirm, after the
   judge-phase snapshot's end), the pinned base and the fine-tune forecast
   on them, losses computed
   from the float32 quantiles it publishes as a sidecar. A compute failure
   is retried :data:`MKT_COMPUTE_ATTEMPTS` times, then the entry is marked
   ``infra_deferred`` (never judged, never dropped for it). It publishes the
   signed report and never overwrites one.
3. **Availability is decided by the trainer-signed lane-C manifest**, not a
   clock: a later manifest records ``mkt_reports[<round>]`` = the sha256 of
   the report text the trainer read, or ``"absent"`` after ``[marketcast]
   report_timeout_settlements`` settlements. Every validator applies the
   backlog STRICTLY in round order, each round when (and as) a manifest
   records it: a recorded sha256 ⇒ those exact bytes (an ordinary storage
   read failure there holds, since the bytes provably exist; bytes that do
   not match ⇒ treated as absent); ``"absent"`` ⇒ every candidate of the
   round is an ``infra`` deferral. A round no manifest ever records expires
   to ``"absent"`` after ``2 × report_timeout_settlements`` MarketCast
   settlements (a deterministic count of folded settlements — no clock).
4. **Verdict** (:func:`judge_marketcast_entry`, pure, every validator, the
   main one included): :func:`~cascade.shared.marketcast_report.
   verdict_from_rows` on the signed rows — ``gain LCB ≥ gain_margin`` at
   ``alpha / k`` (``k`` = the target's judge-phase candidates of the round;
   ``alpha`` on a confirm), positive point gain, ``min_windows`` over
   ``min_clusters`` assets and, on ``mkt:returns``, the coverage gate. An
   entry inconclusive for lack of windows / clusters WAITS while within
   ``verdict_wait_blocks`` (capped at 13 h) of its origin (measured on the ROUND's block, so
   a late report never eats the wait); an ``infra`` deferral waits without
   consuming it, up to ``max_infra_deferrals``, then drops with the
   requeue-eligible operator error class.

Slots compare MarketCast wins against the PINNED base
(``[marketcast] base_pointer``), so a king change never displaces a holder
and judging never needs a king. Pending → confirm → slot and the payouts are
lane C's (``resolve_outcomes`` / ``LaneCState``).
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..shared.config.lane_c import MKT_KIND, is_mkt_target, kind_of
from ..shared.lane_c import PendingEntry, competition_past_end
from ..shared.lane_c_manifest import LaneCEntry, commit_block_as_of, finetune_commit_as_of
from ..shared.marketcast_report import (
    INFRA_DEFERRED,
    REPORT_ABSENT,
    MktEntryRows,
    MktReport,
    MktWindowRow,
    base_cov_gain,
    baseline_scores,
    behaviour_distance,
    encode_quantiles,
    guard_scores,
    holder_gain,
    kind_scores,
    load_mkt_report,
    mkt_quantiles_key,
    mkt_ref_quantiles_key,
    mkt_report_key,
    mkt_text_digest,
    score_windows,
    sign_mkt_report,
    verdict_from_rows,
    verify_mkt_report,
)

log = logging.getLogger("cascade.validator")


# ── W-CMP2 (DEC-CA-0075): the competition this runner is judging ────────────
#
# The MarketCast runner is MULTI-INSTANCE: ``mkt`` (its legacy instance —
# ``[marketcast]``, ``mkt_reports``, ``mkt_backlog`` / ``mkt_seen``, report key
# ``marketcast/round-<R>.json``, all byte-identical) and every spec instance
# ``<id>`` (its compiled table, ``reports[<id>]``, ``comp[<id>]`` state, report
# key ``marketcast/<id>/round-<R>.json``). Every mixin method reads the
# instance from :func:`current_ctx` (thread-local; default ``mkt``).

import threading as _threading  # noqa: E402

_CTX = _threading.local()


@dataclass(frozen=True)
class MktCtx:
    """One instance of the MarketCast runner."""
    kind: str
    mc: object

    @property
    def legacy(self) -> bool:
        return self.kind == MKT_KIND

    def is_target(self, target: object) -> bool:
        return kind_of(str(target or "")) == self.kind

    def report_key(self, rid: str) -> str:
        from ..shared.marketcast_report import instance_report_key

        return mkt_report_key(rid) if self.legacy else instance_report_key(self.kind, rid)

    def arbiters(self, cm) -> dict:
        if cm is None:
            return {}
        if self.legacy:
            return dict(getattr(cm, "mkt_reports", None) or {})
        return dict((getattr(cm, "reports", None) or {}).get(self.kind) or {})

    def backlog(self, st) -> list:
        if self.legacy:
            return list(st.mkt_backlog)
        return list(dict(getattr(st, "comp", {}) or {}).get(self.kind, {}).get("backlog", []))

    def seen(self, st) -> dict:
        if self.legacy:
            return dict(st.mkt_seen)
        return dict(dict(getattr(st, "comp", {}) or {}).get(self.kind, {}).get("seen", {}))

    def store(self, st, backlog: list, seen: dict) -> None:
        if self.legacy:
            st.mkt_backlog, st.mkt_seen = list(backlog), dict(seen)
            return
        comp = dict(st.comp or {})
        sec = {k: v for k, v in (("backlog", list(backlog)), ("seen", dict(seen))) if v}
        if sec:
            comp[self.kind] = sec
        else:
            comp.pop(self.kind, None)
        st.comp = comp


def current_ctx(cfg=None) -> MktCtx:
    """The instance being judged on this thread (default: ``mkt`` on
    ``cfg``'s ``[marketcast]``)."""
    c = getattr(_CTX, "ctx", None)
    if c is not None:
        return c
    return MktCtx(MKT_KIND, getattr(cfg, "marketcast", None))


class mkt_context:  # noqa: N801 — a context manager used like a function
    """``with mkt_context(ctx):`` — judge / publish as instance ``ctx``."""

    def __init__(self, ctx: MktCtx) -> None:
        self.ctx = ctx

    def __enter__(self) -> MktCtx:
        self.saved = getattr(_CTX, "ctx", None)
        _CTX.ctx = self.ctx
        return self.ctx

    def __exit__(self, *exc) -> None:
        _CTX.ctx = self.saved

# base quantiles kept in memory per (snapshot, window) before the cache resets
MKT_BASE_CACHE_MAX = 200_000
# a compute failure on the main validator is retried this often within a round
MKT_COMPUTE_ATTEMPTS = 2
# the exact-identity registry keeps this many checkpoint digests
MKT_SEEN_KEEP = 4096


class MktFault(Exception):
    """A deterministic failure of a fine-tune (unloadable, not the base's
    architecture, non-finite weights, a malformed ``marketcast_meta.json``,
    covariates it cannot be given): the entry loses (stage ``fault``)."""


class MktForwardError(Exception):
    """Red-team F7: an exception raised INSIDE the fine-tune's forward pass
    that is not infrastructure-shaped (:func:`infra_like`). Raised once it
    stays an infra retry; raised identically on every
    :data:`MKT_COMPUTE_ATTEMPTS` attempt it is the miner's fault."""


_INFRA_RE = re.compile(r"cuda|cudnn|cublas|nccl|out of memory|device-side|illegal memory|"
                       r"\bhip\b|driver|nvml|\becc\b", re.IGNORECASE)


def infra_like(e: BaseException) -> bool:
    """Whether an exception from a forward pass is the HOST's (memory, I/O,
    network, the GPU stack) rather than the model's — conservative: anything
    naming CUDA / OOM / the driver is infra."""
    if isinstance(e, (MemoryError, OSError, ConnectionError, TimeoutError)):
        return True
    if "OutOfMemory" in type(e).__name__ or "AcceleratorError" in type(e).__name__:
        return True
    return bool(_INFRA_RE.search(f"{type(e).__name__}: {e}"))


def suite_of(target: str) -> str:
    return str(target).split(":", 1)[1]


# ── pure pieces (every validator + the audit) ───────────────────────────────


def mkt_train_time(entry: LaneCEntry) -> float | None:
    """The leg's dispatch time (unix) from the trainer-signed record."""
    v = dict(entry.marketcast or {}).get("train_time")
    try:
        t = float(v)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) and t > 0 else None


# DEC-CA-0074: the trainer's embargo contamination receipt results an entry
# may carry (anything else, or none, and it is not judged on the embargo)
EMBARGO_SCAN_OK = ("clean", "no_generator")
EMBARGO_EMBEDDED = "embargo_series_embedded"


def embargo_scan_problem(entry: LaneCEntry) -> tuple[str, str] | None:
    """DEC-CA-0074 guard (b) on the validator: an entry is judged on
    embargo windows only if its trainer-signed ``marketcast`` record carries
    the trainer's pre-rent embargo scan (``embargo_scan``) with a passing
    result. ``embedded`` is the miner's ``fault``; a missing / unknown
    receipt is ``stale`` (trained without the guard)."""
    rec = dict(dict(entry.marketcast or {}).get("embargo_scan") or {})
    res = str(rec.get("result", "") or "")
    if res in EMBARGO_SCAN_OK:
        return None
    if res == "embedded":
        return "fault", (f"{EMBARGO_EMBEDDED}: the generator embeds embargoed bars "
                         f"({rec.get('window_id')})")
    return "stale", ("windows are drawn from the embargo and the trainer-signed record "
                     f"carries no passing embargo scan (result {res or 'absent'!r})")


def mkt_prechecks(entry: LaneCEntry, *, lc, mc, commitments) -> tuple[str, str] | None:
    """``(stage, reason)`` of the first failed pre-check, or None (the last
    is the DEC-CA-0074 embargo-scan receipt, :func:`embargo_scan_problem`)."""
    if entry.target not in lc.targets or suite_of(entry.target) not in mc.target_suites:
        return "target", f"{entry.target!r} is not an enabled MarketCast target"
    m = dict(entry.marketcast or {})
    if (str(m.get("base_ref", "")) != mc.base_ref
            or str(m.get("base_sha256", "")).lower() != mc.base_sha256):
        return "stale", (f"fine-tuned from {m.get('base_ref')!r} sha256 "
                         f"{m.get('base_sha256')!r}; the pinned base is {mc.base_ref} "
                         f"{mc.base_sha256}")
    if mkt_train_time(entry) is None:
        return "binding", "the trainer-signed marketcast record carries no dispatch train_time"
    if commitments is None:
        return "binding", "no commitment history to bind the recipe against"
    bound = finetune_commit_as_of(commitments, entry.miner_hotkey, int(entry.train_block),
                                  mkt_suites=tuple(mc.target_suites))
    if bound != (entry.target, entry.recipe_ref):
        return "binding", (f"commitment as of block {entry.train_block} is {bound!r}, not "
                           f"({entry.target!r}, {entry.recipe_ref!r})")
    return embargo_scan_problem(entry)


def entry_after(entry: LaneCEntry, mc, *, fresh_after: float = 0.0) -> float:
    """The instant an entry's windows must start strictly after: the
    TRAINER-stamped dispatch time ``train_time`` + ``start_margin_seconds``
    (and, on a confirm, the judge-phase snapshot's end). Separation of
    duties: the main validator's snapshot anchor never moves it."""
    t = mkt_train_time(entry)
    if t is None:
        raise ValueError("no train_time on the entry's marketcast record")
    return max(t + float(mc.start_margin_seconds), float(fresh_after or 0.0))


# DEC-CA-0075 (item 2): the embargo in force at the leg's DISPATCH, stamped
# by the trainer into the signed ``marketcast`` record; the draw / scan /
# verdict read these, never the verdict-time config, so an overlay that
# changes ``[marketcast] embargo_days`` / ``embargo_eval_days`` never scores
# a leg on windows inside its own training data.
EMBARGO_RECORD_KEYS = ("embargo_days", "embargo_eval_days")


def entry_embargo_mc(entry, mc):
    """``mc`` with the entry's RECORDED dispatch-time embargo (legacy entry
    without the record: ``mc`` unchanged — the value then was the config's)."""
    from dataclasses import replace

    rec = dict(getattr(entry, "marketcast", None) or {})
    got = {k: int(rec[k]) for k in EMBARGO_RECORD_KEYS
           if type(rec.get(k)) is int and int(rec[k]) > 0}
    return replace(mc, **got) if got else mc


def entry_draw(entry: LaneCEntry, mc, *, phase: str, block: int, block_hash: str,
               fresh_after: float = 0.0) -> dict:
    """DEC-CA-0074: the entry's signed draw record at round ``block``
    (:func:`cascade.pool.marketcast.embargo_draw`): backtests from the
    embargo before its trainer-stamped dispatch, forwards after
    :func:`entry_after`. DEC-CA-0075: on the embargo RECORDED at the leg's
    dispatch (:func:`entry_embargo_mc`)."""
    from ..pool.marketcast import embargo_draw

    mc = entry_embargo_mc(entry, mc)
    t = mkt_train_time(entry)
    if t is None:
        raise ValueError("no train_time on the entry's marketcast record")
    return embargo_draw(mc, train_time=t, phase=phase, block=int(block), block_hash=block_hash,
                        after=entry_after(entry, mc, fresh_after=fresh_after))


def _draw_mismatch(entry: LaneCEntry, rows: MktEntryRows, mc, *, phase: str, block: int,
                   fresh_after: float) -> str:
    """Why the rows' signed draw is not the DEC-CA-0074 rule's for ``entry``
    at round ``block`` (``""`` = it is). Pure; the block hash is taken from
    the rows (the main validator read it; ``cascade-audit`` checks it on
    chain)."""
    from ..pool.marketcast import draw_problem

    d = dict(rows.draw or {})
    if not d:
        return "no draw record"
    try:
        want = entry_draw(entry, mc, phase=phase, block=block,
                          block_hash=str(d.get("block_hash", "")), fresh_after=fresh_after)
    except ValueError as e:
        return str(e)
    return draw_problem(d, want)


def anchor_time(train_block: int, snapshot_meta: dict) -> float:
    """``train_block``'s time on a snapshot's block → time anchor (the
    audit's consistency check against ``train_time``)."""
    return float(snapshot_meta["anchor_time"]) + (int(train_block) - int(
        snapshot_meta["anchor_block"])) * float(snapshot_meta["block_seconds"])


def _finite_tree(x):
    """A diagnostics tree with every non-finite float as None (JSON-safe)."""
    if isinstance(x, dict):
        return {str(k): _finite_tree(v) for k, v in x.items()}
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def _inconclusive(reason: str) -> str:
    if reason.startswith("inconclusive:"):
        if "time blocks" in reason:
            return "time_blocks"
        return "windows" if "windows" in reason else "clusters"
    return ""


def near_duplicate_of(rows: MktEntryRows, mc) -> dict | None:
    """Red-team F4: the signed ``near_dup`` record when it makes the entry a
    duplicate (``mean_rel < near_dup_tol`` over at least one shared window),
    else None."""
    nd = dict(rows.near_dup or {})
    try:
        d = float(nd.get("mean_rel", float("nan")))
    except (TypeError, ValueError):
        return None
    if int(nd.get("n_windows", 0) or 0) < 1 or not math.isfinite(d):
        return None
    return nd if d < float(mc.near_dup_tol) else None


def mkt_owner(entry: LaneCEntry) -> str:
    """Red-team F6: the entry's OWNER for the Bonferroni family — the
    coldkey the trainer stamped into the signed ``marketcast`` record
    (resolved from the commitment it dispatched on), else the hotkey."""
    ck = str(dict(entry.marketcast or {}).get("coldkey", "") or "")
    return f"ck:{ck}" if ck else f"hk:{entry.miner_hotkey}"


def mkt_family_sizes(cands) -> dict[tuple[str, str, str], int]:
    """Red-team F6: per judge-phase candidate ``(target, hotkey, pointer)``
    its effective family size ``k = C × m`` — ``C`` the target's distinct
    OWNERS (:func:`mkt_owner`) this round, ``m`` the owner's own entries on
    the target. A weighted Bonferroni: the levels ``α / (C·m)`` of all the
    target's entries sum to ``α``, so sybil hotkeys of one coldkey share ONE
    owner's ``α / C`` (they never raise the bar for the others, and never buy
    extra looks for their owner)."""
    per: dict[str, dict[str, int]] = {}
    for e in cands:
        o = per.setdefault(e.target, {})
        o[mkt_owner(e)] = o.get(mkt_owner(e), 0) + 1
    return {(e.target, e.miner_hotkey, e.trained_pointer):
            len(per[e.target]) * per[e.target][mkt_owner(e)] for e in cands}


def judge_marketcast_entry(entry: LaneCEntry, rows: MktEntryRows | None, *, mc, k: int,
                           phase: str, seed: str, round_block: int, origin_block: int,
                           base_pointer: str = "", report: dict | None = None,
                           why_infra: str = "", infra_deferrals: int = 0,
                           lc=None, era_blocks: int = 0):
    """The judgement of one MarketCast entry from its signed rows (a
    :class:`~cascade.validator.lane_c.LaneCJudgement`). ``rows`` None, or
    rows the main validator marked ``infra_deferred``, is an ``infra``
    judgement (``why_infra``): never a loss — it waits (``infra_deferrals``
    + 1 ≤ ``max_infra_deferrals``) or drops with the operator error class
    (requeue-eligible, uncharged). ``within_wait`` is measured on the
    ROUND's block. A passing verdict carries its gain-scaled full-pay
    duration (DEC-CA-0073) on ``[lane_c]`` (``lc``) terms at ``era_blocks``
    (the era length at the confirming settlement); ``lc`` None = no record."""
    from ..funding.queue import OPERATOR_FAULT_CLASS
    from .lane_c import LaneCJudgement

    alpha = float(mc.alpha) / max(1, int(k))
    within = int(round_block) - int(origin_block) <= int(mc.wait_blocks)   # <= 13 h
    rec = {**(report or {}), "origin_block": int(origin_block), "within_wait": bool(within)}

    def _j(passed, stage, reason, **kw):
        return LaneCJudgement(entry, bool(passed), stage, reason, phase=phase, k=int(k),
                              alpha=alpha, base_pointer=base_pointer, mkt=rec, **kw)

    if rows is None or rows.status == INFRA_DEFERRED:
        n = int(infra_deferrals) + 1
        rec["inconclusive"] = ""
        rec["infra_deferrals"] = n
        why = why_infra or (rows.infra_reason if rows is not None else "") or "no rows"
        if n > int(mc.max_infra_deferrals):
            rec["error_class"] = OPERATOR_FAULT_CLASS
            return _j(False, "infra", f"infra: {why} — {n} > max_infra_deferrals="
                                      f"{mc.max_infra_deferrals}: dropped, requeue-eligible")
        return _j(False, "infra", f"infra: {why} (deferral {n}/{mc.max_infra_deferrals})")
    rec["after"] = float(rows.after)
    if infra_deferrals:
        rec["infra_deferrals"] = int(infra_deferrals)
    if rows.fault:
        rec["inconclusive"] = ""
        return _j(False, "fault", rows.fault)
    nd = near_duplicate_of(rows, mc)
    if nd is not None:
        # red-team F4: behaviourally the SENIOR entry's forecasts — the
        # earliest commit holds (never a UID); signed, replayed by the audit
        rec["inconclusive"] = ""
        rec["near_dup"] = dict(rows.near_dup)
        return _j(False, "duplicate", (
            f"near duplicate: its forecasts sit {float(nd['mean_rel']):.3g} (mean |Δq| / base "
            f"5–95 % width; max {float(nd['max_rel']):.3g}) from "
            f"{str(nd.get('hotkey', ''))[:12]}'s fine-tune committed at block "
            f"{nd.get('commit_block')} — below near_dup_tol={mc.near_dup_tol}"))
    if rows.near_dup:
        rec["near_dup"] = dict(rows.near_dup)
    v = verdict_from_rows(rows, mc, suite=suite_of(entry.target), k=k, seed=seed)
    if v.diagnostics.get("suites"):
        # the composite (owner 2026-11-10): every suite's gain / LCB / windows
        rec["suites"] = {s: {k_: (float(x) if isinstance(x, float) and math.isfinite(x)
                                  else (None if isinstance(x, float) else x))
                             for k_, x in d.items()}
                         for s, d in v.diagnostics["suites"].items()}
    for key in ("vs_baseline", "base_vs_baseline"):
        # owner 2026-11-18: skill vs the strong simple baselines (reported; the
        # composite's baseline_gate is inside the verdict)
        if v.diagnostics.get(key):
            rec[key] = _finite_tree(v.diagnostics[key])
    vs = holder_gain(rows, mc, k=k, seed=seed)
    if vs is not None:
        rec["vs_holder"] = vs        # owner 2026-11-18: the paired slot test
    if getattr(rows, "draw", None):
        # DEC-CA-0074 (c): which rule drew the windows, and the forward /
        # backtest sub-scores (reported; the verdict is the combined set's)
        rec["draw"] = {k_: rows.draw.get(k_) for k_ in ("rule", "block", "block_hash", "phase",
                                                         "lo", "hi", "n")}
        rec["kinds"] = kind_scores(rows, mc, suite=suite_of(entry.target), k=k, seed=seed)
    inc = _inconclusive(v.reason)
    rec["inconclusive"] = inc
    excluded = sum(1 for w in rows.windows if not math.isfinite(float(w.base)))
    if excluded:
        rec["base_excluded"] = int(excluded)      # review M11: recorded, never silent
    g2 = base_cov_gain(rows)
    if g2 is not None:
        rec["gain_vs_base_cov"] = g2              # review M13: reported, never gating
    rec["masked_cells"] = int(v.diagnostics.get("masked_cells", 0))   # F3: recorded
    if v.diagnostics.get("masked"):
        rec["masked"] = list(v.diagnostics["masked"])
    if v.coverage is not None:
        rec["coverage"] = {r.key: r.empirical for r in v.coverage.rows}
        rec["coverage_passed"] = bool(v.coverage.passed)
    gain = float(v.gain) if np.isfinite(v.gain) else None
    lcb = float(v.gain_lcb) if np.isfinite(v.gain_lcb) else None
    # DEC-CA-0073: an mkt: win's full-pay duration scales with its gain LCB
    # from [marketcast] gain_margin (5 days) to [lane_c] full_pay_max_gain (20)
    from ..shared.lane_c import target_full_pay_record

    fp = (target_full_pay_record(entry.target, lcb, lc, mc, era_blocks=int(era_blocks))
          if v.passed and lc is not None else None)
    return _j(v.passed, "verdict", v.reason, gain_lcb=lcb, gain=gain, n=int(v.n),
              clusters=int(v.n_clusters), inconclusive=inc, full_pay=fp)


# ── rows (the main validator's GPU path; forecasters injected) ──────────────


def _covariate_rows(windows, snap, mc, covs) -> tuple[list, list]:
    """Per window the fine-tune's ``(past, known)`` covariate rows; raises
    ``ValueError`` / ``KeyError`` / ``TypeError`` when the checkpoint's own
    list is outside the allow-list or the windows cannot be given it."""
    from ..pool.marketcast import window_covariates
    from ..shared.marketcast_recipe import covariates_problem

    why = covariates_problem(list(covs), mc)
    if why:
        raise ValueError(why)
    past, known = [], []
    for w in windows:
        p, kn = window_covariates(snap, w, tuple(covs),
                                  context_length=int(mc.eval_context_length),
                                  event_times=tuple(getattr(mc, "event_times", ())))
        past.append(p)
        known.append(kn)
    return past, known


def _ft_forward(fc, windows, idx, steps, *, knots, batch, past, known):
    """The fine-tune's forecasts on ``windows[idx]`` (one step count).
    ``ValueError`` (valid input rejected) propagates as-is; any other
    exception that is not :func:`infra_like` is re-raised as
    :class:`MktForwardError` (red-team F7); infra propagates unchanged."""
    from ..eval.timestral_adapter import batched_forecasts

    pp = [past[i] for i in idx]
    kk = [known[i] for i in idx]
    try:
        return batched_forecasts(fc, [windows[i].context for i in idx], steps,
                                 batch=batch, levels=knots,
                                 past=pp if any(p is not None for p in pp) else None,
                                 known=kk if any(k is not None for k in kk) else None)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — classified: the model's or the host's
        if infra_like(e):
            raise
        raise MktForwardError(f"{type(e).__name__}: {e}") from e


def compute_entry_rows(entry: LaneCEntry, *, phase: str, after: float, snap, mc,
                       base_fc, load_ft: Callable, base_cache: dict, round_id: str = "",
                       batch: int = 64, ft_cache: dict | None = None,
                       draw: dict | None = None
                       ) -> tuple[MktEntryRows, bytes | None]:
    """Forecast the base and the fine-tune on EVERY eligible window of the
    entry and return ``(rows, quantile sidecar bytes)``. ``load_ft(pointer)
    -> (forecaster, covariate names)`` raises :class:`MktFault` for a
    deterministic failure; a covariate list outside the allow-list, or one
    the snapshot's windows cannot be given, is a fault too (recorded as the
    entry's ``fault``, sidecar None); so is a ``ValueError`` from the
    fine-tune's forward. A non-infra exception inside the fine-tune's
    forward raises :class:`MktForwardError` (the caller makes it a fault when
    every attempt raises it identically — red-team F7); anything else
    propagates (the caller retries, then marks the entry ``infra_deferred``).
    Base quantiles are cached in ``base_cache`` per ``(snapshot digest,
    window id)``, the fine-tune's in ``ft_cache`` per ``(pointer, snapshot
    digest, window id)`` (the F4 near-duplicate test reuses them). Losses
    come from the float32 quantiles the sidecar publishes
    (:func:`~cascade.shared.marketcast_report.score_windows`). ``draw``
    (DEC-CA-0074, :func:`entry_draw`) is the windows' signed draw; it rides
    the rows (None = the plain forward-only cut after ``after``: tests)."""
    from ..eval.marketcast import COVERAGE_GATE_SUITES
    from ..eval.timestral_adapter import batched_forecasts
    from ..pool.marketcast import target_windows
    from ..shared.config.marketcast import base_suite

    knots = tuple(mc.quantile_knots)
    windows = target_windows(snap, mc, suite_of(entry.target), after=after, draw=draw or None)
    base_rows = MktEntryRows(entry.miner_hotkey, entry.trained_pointer, entry.target, phase,
                             float(after), draw=dict(draw or {}))
    try:
        ft_fc, covs = load_ft(entry.trained_pointer)
    except MktFault as e:
        return replace(base_rows, fault=str(e)[:300]), None
    try:
        # the checkpoint's own covariate list (marketcast_meta.json): checked
        # again HERE, inside the fault path — a bad name is the miner's fault
        past, known = _covariate_rows(windows, snap, mc, covs)
    except (ValueError, KeyError, TypeError) as e:
        return replace(base_rows, fault=f"fine-tune covariates unusable: {e}"[:300]), None
    by_steps: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        by_steps.setdefault(int(w.steps), []).append(i)
    qb: list = [None] * len(windows)
    qf: list = [None] * len(windows)
    for steps, idx in sorted(by_steps.items()):
        todo = [i for i in idx if (snap.digest, windows[i].wid) not in base_cache]
        if todo:
            q = batched_forecasts(base_fc, [windows[i].context for i in todo], steps,
                                  batch=batch, levels=knots)
            for i, x in zip(todo, q, strict=True):
                base_cache[(snap.digest, windows[i].wid)] = np.asarray(x, dtype=np.float32)
        for i in idx:
            qb[i] = base_cache[(snap.digest, windows[i].wid)]
        try:
            q = _ft_forward(ft_fc, windows, idx, steps, knots=knots, batch=batch, past=past,
                            known=known)
        except ValueError as e:      # the fine-tune rejects valid input: deterministic
            return replace(base_rows, fault=f"fine-tune forecast failed: {e}"[:300]), None
        for i, x in zip(idx, q, strict=True):
            qf[i] = np.asarray(x, dtype=np.float32)
            if not np.isfinite(qf[i]).all():
                # review M11: never masked — a fine-tune that cannot forecast a
                # valid eligible window loses (deterministic, signed, auditable)
                return replace(base_rows, fault=f"fine-tune produced non-finite quantiles on "
                                                f"{windows[i].wid}"[:300]), None
    if ft_cache is not None:
        for w, x in zip(windows, qf, strict=True):
            ft_cache[(entry.trained_pointer, snap.digest, w.wid)] = x
    bc = [float("nan")] * len(windows)
    if covs:
        # review M13: a SECOND, reported (never gating) baseline — the base
        # given the fine-tune's own covariates
        for steps, idx in sorted(by_steps.items()):
            pp = [past[i] for i in idx]
            kk = [known[i] for i in idx]
            q = batched_forecasts(base_fc, [windows[i].context for i in idx], steps,
                                  batch=batch, levels=knots,
                                  past=pp if any(p is not None for p in pp) else None,
                                  known=kk if any(k is not None for k in kk) else None)
            sc = score_windows([windows[i].target for i in idx], list(q), list(q), knots=knots,
                               coverage_levels=(), gated=False,
                               scales=[windows[i].scale for i in idx])
            for i, row in zip(idx, sc, strict=True):
                bc[i] = row[0]
    scored = score_windows([w.target for w in windows], qb, qf, knots=knots,
                           coverage_levels=tuple(mc.coverage_levels),
                           gated=[base_suite(w.suite) in COVERAGE_GATE_SUITES for w in windows],
                           tail_levels=tuple(mc.tail_levels),
                           scales=[w.scale for w in windows])
    # a non-finite BASE forecast leaves base = null: the window is excluded for
    # everyone (verdict_from_rows) and the row records it
    gs = guard_scores([w.target for w in windows], qb, qf, knots=knots,
                      guards=[mc.guards_of(w.suite) for w in windows],
                      scales=[w.scale for w in windows])
    bls = baseline_scores(windows, knots=knots)
    out = tuple(MktWindowRow(w.wid, b, f, ci, cn, bc[i], tl, **gs[i], bl=bls[i])
                for i, (w, (b, f, ci, cn, tl)) in enumerate(zip(windows, scored, strict=True)))
    data = encode_quantiles(qb, qf)
    import hashlib

    qkey = mkt_quantiles_key(round_id, entry.miner_hotkey, entry.trained_pointer, phase)
    return replace(base_rows, windows=out, quantiles_key=qkey,
                   quantiles_sha256=hashlib.sha256(data).hexdigest()), data


def reference_quantiles(pointer: str, windows, *, snap, mc, load_ft: Callable,
                        ft_cache: dict, batch: int = 64) -> list[np.ndarray]:
    """Red-team F4: a SENIOR entry's float32 forecasts on ``windows`` (a
    candidate's), from ``ft_cache`` where it already forecast them, else by
    loading and running it (same covariate rules as the candidate). Raises
    on any failure (the caller then skips that reference)."""
    knots = tuple(mc.quantile_knots)
    todo = [i for i, w in enumerate(windows) if (pointer, snap.digest, w.wid) not in ft_cache]
    if todo:
        fc, covs = load_ft(pointer)
        past, known = _covariate_rows(windows, snap, mc, covs)
        by_steps: dict[int, list[int]] = {}
        for i in todo:
            by_steps.setdefault(int(windows[i].steps), []).append(i)
        for steps, idx in sorted(by_steps.items()):
            q = _ft_forward(fc, windows, idx, steps, knots=knots, batch=batch, past=past,
                            known=known)
            for i, x in zip(idx, q, strict=True):
                ft_cache[(pointer, snap.digest, windows[i].wid)] = np.asarray(x, dtype=np.float32)
    return [ft_cache[(pointer, snap.digest, w.wid)] for w in windows]


def near_dup_record(rows: MktEntryRows, data: bytes, refs, *, snap, mc, load_ft: Callable,
                    ft_cache: dict, round_id: str) -> tuple[dict, bytes | None]:
    """Red-team F4: measure a computed entry against every SENIOR reference
    ``refs`` (``[(hotkey, pointer, commit_block), ...]``: earlier-committed
    candidates of the same target this round and the target's slot holder)
    on the entry's own windows, and return ``(near_dup record, reference
    sidecar bytes)`` for the CLOSEST one (smallest mean distance,
    :func:`~cascade.shared.marketcast_report.behaviour_distance`); ``({},
    None)`` when no reference could be measured. The reference sidecar is
    ``encode_quantiles(base, reference)`` so the audit recomputes the
    distance from published bytes."""
    import hashlib

    from ..pool.marketcast import target_windows
    from ..shared.marketcast_report import decode_quantiles

    if not refs or not rows.windows:
        return {}, None
    windows = target_windows(snap, mc, suite_of(rows.target), after=float(rows.after),
                             draw=rows.draw or None)
    qs = decode_quantiles(data, [w.steps for w in windows], len(mc.quantile_knots))
    qb = [q[0] for q in qs]
    qf = [q[1] for q in qs]
    best = None
    for hk, ptr, cb in refs:
        try:
            qr = reference_quantiles(ptr, windows, snap=snap, mc=mc, load_ft=load_ft,
                                     ft_cache=ft_cache)
        except Exception as e:  # noqa: BLE001 — an unmeasurable reference is skipped
            log.warning("marketcast near-dup: reference %s unusable for %s (%s)", ptr[:40],
                        rows.trained_pointer[:40], e)
            continue
        qr = [np.asarray(x, dtype=np.float32).astype(np.float64) for x in qr]
        mean, mx = behaviour_distance(qb, qf, qr)
        if not math.isfinite(mean):
            continue
        if best is None or (mean, int(cb)) < (best[0], int(best[3])):
            best = (mean, mx, (hk, ptr), int(cb), qr)
    if best is None:
        return {}, None
    blob = encode_quantiles(qb, best[4])
    key = mkt_ref_quantiles_key(round_id, rows.hotkey, rows.trained_pointer, rows.phase)
    return ({"hotkey": best[2][0], "pointer": best[2][1], "commit_block": int(best[3]),
             "mean_rel": float(best[0]), "max_rel": float(best[1]),
             "n_windows": len(windows), "quantiles_key": key,
             "quantiles_sha256": hashlib.sha256(blob).hexdigest()}, blob)


def holder_record(rows: MktEntryRows, data: bytes, holder, *, snap, mc, load_ft: Callable,
                  ft_cache: dict, round_id: str) -> tuple[MktEntryRows, bytes | None]:
    """Owner 2026-11-18: the PAIRED challenger-vs-holder evidence — the
    target's slot holder (``holder`` = ``(hotkey, pointer, commit_block)``)
    forecast on the entry's own windows, its per-window losses scored by the
    one loss rule (:func:`~cascade.shared.marketcast_report.score_windows`)
    into the rows' ``hold``, its quantiles published as ``encode_quantiles(
    base, holder)`` so the audit recomputes them. ``(rows, None)`` unchanged
    when there is no holder, it is the entry itself, or it cannot be
    forecast (the slot rule then falls back to the recorded-LCB comparison)."""
    import hashlib

    from ..pool.marketcast import target_windows
    from ..shared.marketcast_report import decode_quantiles, mkt_holder_quantiles_key

    if holder is None or not rows.windows or holder[1] == rows.trained_pointer:
        return rows, None
    windows = target_windows(snap, mc, suite_of(rows.target), after=float(rows.after),
                             draw=rows.draw or None)
    qb = [q[0] for q in decode_quantiles(data, [w.steps for w in windows],
                                         len(mc.quantile_knots))]
    try:
        qh = reference_quantiles(holder[1], windows, snap=snap, mc=mc, load_ft=load_ft,
                                 ft_cache=ft_cache)
    except Exception as e:  # noqa: BLE001 — an unmeasurable holder is skipped
        log.warning("marketcast holder pairing: %s unusable for %s (%s)", holder[1][:40],
                    rows.trained_pointer[:40], e)
        return rows, None
    sc = score_windows([w.target for w in windows], qb, qh, knots=tuple(mc.quantile_knots),
                       coverage_levels=(), gated=False, scales=[w.scale for w in windows])
    blob = encode_quantiles(qb, qh)
    key = mkt_holder_quantiles_key(round_id, rows.hotkey, rows.trained_pointer, rows.phase)
    out = tuple(replace(w, hold=float(s[1])) for w, s in zip(rows.windows, sc, strict=True))
    return replace(rows, windows=out, holder={
        "hotkey": str(holder[0]), "pointer": str(holder[1]), "quantiles_key": key,
        "quantiles_sha256": hashlib.sha256(blob).hexdigest()}), blob


def snapshot_record(snap, key: str, sha256: str) -> dict:
    m = snap.meta
    return {"key": str(key), "sha256": str(sha256), "digest": str(snap.digest),
            "t0": int(m["t0"]), "end": int(m["end"]), "anchor_block": int(m["anchor_block"]),
            "anchor_time": float(m["anchor_time"]), "block_seconds": float(m["block_seconds"])}


# ── the round state (pure: every validator folds it identically) ─────────────


@dataclass
class MktPlan:
    """What one settlement does to the MarketCast state, computed at judge
    time on a copy and replayed by the fold: the wins offered per applied
    round (in order), the slot evictions of superseded copies, and the
    resulting backlog + identity registry."""
    offered: list = field(default_factory=list)       # TargetWin, in fold order
    evict: list = field(default_factory=list)         # (target, hotkey, pointer), BEFORE offers
    backlog: list = field(default_factory=list)
    seen: dict = field(default_factory=dict)
    # Slot holders superseded by an earlier commit of the same weights
    # registered THIS settlement — judge time deletes them AFTER the applied
    # rounds' wins are recorded (a win confirmed and superseded in one
    # settlement ends with no holder), so the fold must too. Transient (a
    # plan is folded in the poll that computed it, never persisted).
    evict_after: list = field(default_factory=list)   # (target, hotkey, pointer), AFTER offers
    # Owner 2026-11-24: every slot holder's registration consulted this
    # settlement — ``{target, hotkey, uid}`` (uid None = deregistered) as of
    # the boundary − 1; the receipt records them (``lane_c_registrations``)
    # so satellites / the audit / late validators never need chain history.
    registrations: list = field(default_factory=list)


def _candidate(entry: LaneCEntry, phase: str, p: PendingEntry | None,
               commit_block: int | None = None) -> dict:
    from ..shared.lane_c import _pending_dict

    e = asdict(entry)
    return {"entry": e, "phase": str(phase),
            "pending": None if p is None else _pending_dict(p),
            # red-team F4: seniority for the near-duplicate test (a block, never a UID)
            "commit_block": int(entry.train_block if commit_block is None else commit_block)}


def _seniority(c: dict) -> tuple[int, int, str]:
    e = c["entry"]
    return (int(c.get("commit_block", e["train_block"])), int(e["train_block"]),
            str(e["miner_hotkey"]))


def _unpack(c: dict) -> tuple[LaneCEntry, str, PendingEntry | None]:
    p = c.get("pending")
    return (LaneCEntry(**c["entry"]), str(c["phase"]),
            None if p is None else PendingEntry(**p))


def _superseded(seen: dict, e: LaneCEntry) -> dict | None:
    """The registry holder that makes ``e`` a duplicate (None = it is not)."""
    sha = str(dict(e.marketcast or {}).get("checkpoint_sha256", "") or "")
    h = seen.get(sha) if sha else None
    if h is None or (h.get("hotkey"), h.get("pointer")) == (e.miner_hotkey, e.trained_pointer):
        return None
    return h


def _slot_holders(work, ctx: MktCtx | None = None) -> dict[str, tuple[str, str, int]]:
    """Red-team F4: ``{mkt target: (hotkey, pointer, commit block)}`` of the
    current slot holders — references for the near-duplicate test (a
    holder whose commit block the identity registry lacks is senior: -1)."""
    cbs = {(v.get("hotkey"), v.get("pointer")): int(v.get("commit_block", -1))
           for v in work.mkt_seen.values()}
    return {t: (w.hotkey, w.trained_pointer, cbs.get((w.hotkey, w.trained_pointer), -1))
            for t, w in work.slots.items()
            if (ctx.is_target(t) if ctx is not None else is_mkt_target(t))}


def _dup_reason(h: dict) -> str:
    return (f"exact duplicate: its weights' sha256 is {h.get('hotkey', '')[:12]}'s fine-tune "
            f"({h.get('target')}, committed at block {h.get('commit_block')}) — the earliest "
            f"commit holds it (DEC-CA-0008)")


# ── the mixin (live validator) ───────────────────────────────────────────────


class MarketCastMixin:
    """MarketCast rows: computed + published on the main validator, read
    everywhere else; the backlog pipeline every validator folds.
    ``mkt_forecaster_fn(pointer_or_"base") -> (forecaster, covariates)``
    (tests) replaces the checkpoint fetch + library load;
    ``mkt_snapshot_fn(block) -> (snapshot, key, sha256) | None`` the
    bucket."""

    mkt_forecaster_fn: Callable | None = None
    mkt_snapshot_fn: Callable | None = None
    # DEC-CA-0074: ``block -> 0x hash`` of a round block (tests; None = chain)
    mkt_block_hash_fn: Callable | None = None
    # GPU offload (cascade.validator.mkt_offload): the hosts file the
    # provisioner's [provisioner.mkt] stage publishes (None = forecast on
    # --device); how long one publish run waits for the pod; injectable
    # lease / session factories (tests)
    mkt_eval_hosts: Path | None = None
    mkt_host_wait_s: float = 5400.0       # = provision.policy.MKT_BOOT_WORST_S
    mkt_lease_fn: Callable | None = None
    mkt_session_fn: Callable | None = None

    def _mkt_cfg(self):
        """The MarketCast-shaped table of the instance being judged."""
        return current_ctx(getattr(self, "cfg", None)).mc

    def _mkt_main_hotkey(self) -> str:
        return str(getattr(getattr(self.cfg, "netconfig", None), "owner_hotkey", "") or "")

    def _mkt_is_main(self) -> bool:
        signer = getattr(self, "bench_signer", None)
        hk = getattr(signer, "hotkey", signer)
        me = str(getattr(hk, "ss58_address", "") or "")
        main = self._mkt_main_hotkey()
        return bool(main) and me == main

    def _mkt_store(self):
        st = getattr(self, "_mkt_store_obj", None)
        if st is not None:
            return st
        return self._bench_store()

    def _mkt_snapshot(self, block: int):
        fn = self.mkt_snapshot_fn
        if fn is not None:
            return fn(block)
        mc = self._mkt_cfg()
        # DEC-CA-0075: the competition spec names the pool (prefix + signer +
        # schema + series filter); a spec change takes a new source, and each
        # instance keeps its own (W-CMP2)
        signer = str(mc.pool_signer or "") or self._mkt_main_hotkey()
        key = (mc.pool_prefix, signer, mc.pool_schema, tuple(mc.pool_select),
               tuple(getattr(mc, "pool_channels", ())))
        srcs = self.__dict__.setdefault("_mkt_sources", {})
        src = srcs.get(current_ctx(getattr(self, "cfg", None)).kind)
        if src is None or srcs.get(("key", current_ctx(getattr(self, "cfg", None)).kind)) != key:
            from ..pool.marketcast import MktSnapshotSource
            from ..shared.hippius import pool_s3_store

            src = MktSnapshotSource(
                pool_s3_store(self.cfg.storage), prefix=mc.pool_prefix,
                signer_hotkey=signer,
                cache_dir=Path(getattr(self, "cache_dir", None) or "./_eval_pool"),
                verify_signature=bool(self.verify_signatures),
                schema=mc.pool_schema, select=dict(mc.pool_select),
                channels=dict(getattr(mc, "pool_channels", ())))
            srcs[current_ctx(getattr(self, "cfg", None)).kind] = src
            srcs[("key", current_ctx(getattr(self, "cfg", None)).kind)] = key
        return src.snapshot_for(block)

    def _mkt_block_hash(self, block: int) -> str:
        """The hash of ``block`` (DEC-CA-0074 draw seed): ``mkt_block_hash_fn``
        (tests) or the loop's chain client."""
        fn = self.mkt_block_hash_fn
        if fn is not None:
            return str(fn(int(block)))
        chain = getattr(self, "_lane_b_chain", None)
        if chain is None:
            raise RuntimeError("no chain client wired to read the draw's block hash")
        return str(chain.block_hash(int(block)))

    def _mkt_base(self):
        fn = self.mkt_forecaster_fn
        if fn is not None:
            return fn("base")[0]
        sess = self.__dict__.get("_mkt_sess")
        if sess is not None:
            return sess.forecaster(self._mkt_base_dir_local(), key="base")
        hit = getattr(self, "_mkt_base_fc", None)
        if hit is None:  # pragma: no cover — GPU + the timestral extra
            from ..eval.timestral_adapter import load_pinned_base

            mc = self._mkt_cfg()
            hit, _d = load_pinned_base(mc.base_ref, mc.base_sha256,
                                       cache_dir=getattr(self, "cache_dir", None),
                                       device=str(getattr(self, "device", "cpu")))
            self._mkt_base_fc = hit
            self._mkt_base_dir = _d
        return hit

    def _mkt_base_dir_local(self) -> Path:  # pragma: no cover — box-side, no timestral import
        d = getattr(self, "_mkt_base_dir", None)
        if d is None:
            d = self._mkt_base_dir = resolve_pinned_base_dir(
                self._mkt_cfg(), cache_dir=getattr(self, "cache_dir", None))
        return Path(d)

    def _mkt_load_ft(self, pointer: str):
        fn = self.mkt_forecaster_fn
        if fn is not None:
            return fn(pointer)
        return self._mkt_load_ft_hub(pointer)  # pragma: no cover

    def _mkt_load_ft_hub(self, pointer: str):  # pragma: no cover — GPU + the timestral extra
        import json

        from ..eval.timestral_adapter import (
            META_FILE,
            WEIGHTS_FILE,
            TimestralQuantileForecaster,
            safetensors_all_finite,
            same_structure,
        )
        from ..shared.marketcast_recipe import meta_problem

        base_dir = self._mkt_base_dir_local()
        dest = self._fetch_checkpoint_dir(SimpleNamespace(trained_pointer=pointer))
        try:
            diff = same_structure(base_dir / WEIGHTS_FILE, dest / WEIGHTS_FILE)
            if diff:
                raise MktFault(f"fine-tune is not the base architecture: {diff[:3]}")
            if not safetensors_all_finite(dest / WEIGHTS_FILE):
                raise MktFault("fine-tune weights are not finite")
            raw = (dest / META_FILE).read_bytes()
            meta = json.loads(raw)
            why = meta_problem(meta, self._mkt_cfg(), size=len(raw))
            if why:
                raise MktFault(why)
            sess = self.__dict__.get("_mkt_sess")
            fc = (sess.forecaster(dest) if sess is not None
                  else TimestralQuantileForecaster.from_dir(dest, device=str(self.device)))
        except MktFault:
            raise
        except (ValueError, KeyError, OSError) as e:
            raise MktFault(f"fine-tune unloadable: {e}") from e
        return fc, tuple(meta.get("covariates", ()) or ())

    # ── reading a recorded report ──

    def _mkt_read_recorded(self, rid: str, arbiter: str) -> tuple[MktReport | None, str, str]:
        """``(report | None, sha256 of the bytes, why unusable)`` of the round
        ``rid`` report a lane-C manifest recorded as ``arbiter``. A recorded
        sha256 means the bytes provably existed: a READ failure raises
        :class:`~cascade.validator.lane_c.LaneCTransient` (holds, as any
        bound read); bytes that are not the recorded ones, a malformed or
        unsigned report are unusable (every candidate an ``infra``
        deferral)."""
        from .lane_c import LaneCTransient

        if arbiter == REPORT_ABSENT:
            return None, "", "the main validator's report never appeared (recorded absent)"
        key = current_ctx(getattr(self, "cfg", None)).report_key(rid)
        try:
            text = self._mkt_store().get_text(key)
        except Exception as e:  # noqa: BLE001 — the recorded bytes exist: retried
            raise LaneCTransient(f"marketcast report {key} (recorded sha256 {arbiter[:12]}) "
                                 f"unreadable: {type(e).__name__}: {e}") from e
        sha = mkt_text_digest(text)
        if sha != arbiter:
            return None, sha, "the report's bytes are not the ones the trainer recorded"
        try:
            rep = load_mkt_report(text)
        except (ValueError, KeyError, TypeError) as e:
            return None, sha, f"the main validator's report is unparseable ({e})"
        if str(rep.round_id) != str(rid):
            return None, sha, f"the report names round {rep.round_id}, not {rid}"
        if self.verify_signatures and not verify_mkt_report(rep, self._mkt_main_hotkey()):
            return None, sha, "the report fails the main validator's signature"
        return rep, sha, ""

    # ── the settlement pipeline ──

    def _mkt_settle(self, manifest, cm, *, rid: str, block: int, era_index: int, st,
                    commitments_fn: Callable) -> tuple[MktPlan | None, list]:
        """This settlement's MarketCast part — ``(plan, judgements)``,
        computed on copies of ``st``'s MarketCast state (``plan`` None ⇒
        nothing to fold). Apply every backlog round the lane-C manifest
        ``cm`` (or the fallback expiry) arbitrates, in round order; register
        this settlement's round; and, on the main validator, publish the
        rows of every unarbitrated round still lacking a report (never
        raises for that). Raises :class:`~cascade.validator.lane_c.
        LaneCTransient` only for a READ of recorded bytes or the commitment
        history."""
        from ..shared.lane_c import LaneCState
        from .lane_c import LaneCJudgement

        ctx = current_ctx(getattr(self, "cfg", None))
        mc = ctx.mc
        new = [e for e in (cm.entries if cm is not None else ()) if ctx.is_target(e.target)]
        legacy = [p for p in st.pending if ctx.is_target(p.target)]
        if mc is None or not mc.active(block):
            js = [LaneCJudgement(e, False, "target", "MarketCast is not active at this "
                                 "settlement", phase="judge", k=1,
                                 alpha=float(getattr(mc, "alpha", 0.05)),
                                 mkt={"inconclusive": ""}) for e in new]
            return None, js
        quiet = not (new or legacy or ctx.backlog(st))
        comp = self.cfg.lane_c.competition(ctx.kind)
        if comp is not None and competition_past_end(comp, block):
            if quiet:
                return None, []
            return self._mkt_ended(comp, new, legacy, st, mc=mc, rid=rid)
        registration = getattr(self, "registration_at_block_fn", None)
        if registration is None and self.__dict__.get("_mkt_no_registration_warned") != rid:
            # the client lacks ``uid_for_hotkey_at_block``: a deregistered
            # holder keeps its slot (expiry still evicts) — loud, once per
            # settlement, the verdict unchanged
            self.__dict__["_mkt_no_registration_warned"] = rid
            log.error("marketcast round=%s: the chain client has no uid_for_hotkey_at_block "
                      "hook — deregistration eviction of slot holders is DISABLED on this "
                      "validator (an archive RPC endpoint provides it)", rid)
        if quiet and not st.slots:
            return None, []
        bp = mc.base_pointer
        work = LaneCState(slots=dict(st.slots), mkt_seen={k: dict(v) for k, v in
                                                          ctx.seen(st).items()})
        import copy

        backlog = copy.deepcopy(ctx.backlog(st))
        arbiters = ctx.arbiters(cm)
        expiry = 2 * int(mc.report_timeout_settlements)
        for item in backlog:
            if not item.get("arbiter") and str(item["round_id"]) in arbiters:
                item["arbiter"] = str(arbiters[str(item["round_id"])])
            item["age"] = int(item.get("age", 0)) + 1
            if not item.get("arbiter") and item["age"] > expiry:
                item["arbiter"], item["expired"] = REPORT_ABSENT, True
        plan = MktPlan()
        from ..shared.era import era_length_blocks
        from ..shared.lane_c import win_expired

        for target, holder in list(work.slots.items()):
            if not ctx.is_target(target):
                continue                   # another competition's slot (W-CMP2)
            # DEC-CA-0073: the holder's full-pay duration ran out — the slot ends
            expired = win_expired(work.eras_since(holder, block,
                                                 era_length_blocks(self.cfg.round, block)),
                                  holder, self.cfg.lane_c)
            unregistered = registration is not None and self._mkt_holder_uid(
                registration, target, holder.hotkey, max(0, block - 1), rid=rid,
                plan=plan) is None
            if expired or unregistered:
                plan.evict.append((target, holder.hotkey, holder.trained_pointer))
                del work.slots[target]
        if quiet:
            # no round to apply or register: only the eviction of an expired /
            # deregistered holder (no rows; payout already excluded it)
            if not plan.evict:
                return None, []
            plan.backlog, plan.seen = backlog, work.mkt_seen
            return plan, []
        judgements: list = []
        carried: list[PendingEntry] = list(legacy)
        while backlog and backlog[0].get("arbiter"):
            item = backlog.pop(0)
            js, offered, pend = self._mkt_apply_item(item, work, settle_rid=rid,
                                                     settle_block=block, era_index=era_index)
            judgements += js
            plan.offered += offered
            carried += pend
        # registration of this settlement's round
        commitments = commitments_fn() if (new or carried) else None
        lc = self.cfg.lane_c
        reg: list = []
        cands: list = []

        def _reg_row(e, stage, reason, p, **mk):
            j = LaneCJudgement(e, False, stage, reason, phase="judge" if p is None or
                               p.kind == "deferred" else "confirm", k=1, alpha=float(mc.alpha),
                               base_pointer=bp,
                               mkt={"inconclusive": "", "round": rid, "at": "register", **mk})
            j.outcome = "" if p is None else "dropped"
            if p is not None:
                j.carried = p.kind == "deferred"
                j.origin_round = str(p.round_id)
            reg.append(j)

        for p in sorted(carried, key=lambda p: (p.kind != "confirm", p.sort_key())):
            e = LaneCEntry(**p.entry)
            bad = mkt_prechecks(e, lc=lc, mc=mc, commitments=commitments)
            h = None if bad else _superseded(work.mkt_seen, e)
            if bad:
                _reg_row(e, bad[0], bad[1], p)
            elif h is not None:
                _reg_row(e, "duplicate", _dup_reason(h), p)
            else:
                cb0 = commit_block_as_of(commitments or [], e.miner_hotkey, int(e.train_block))
                cands.append(_candidate(e, "confirm" if p.kind == "confirm" else "judge", p,
                                        commit_block=cb0))
        firsts = []
        for e in new:
            bad = mkt_prechecks(e, lc=lc, mc=mc, commitments=commitments)
            if bad:
                _reg_row(e, bad[0], bad[1], None)
                continue
            cb = commit_block_as_of(commitments or [], e.miner_hotkey, int(e.train_block))
            firsts.append((int(cb if cb is not None else e.train_block), int(e.train_block),
                           e.miner_hotkey, e))
        new_ok = []
        new_cb: dict = {}
        for cb, tb, hk, e in sorted(firsts, key=lambda x: x[:3]):
            sha = str(dict(e.marketcast or {}).get("checkpoint_sha256", "") or "")
            h = _superseded(work.mkt_seen, e)
            if h is not None and (int(h["commit_block"]), int(h["train_block"]),
                                  str(h["hotkey"])) <= (cb, tb, hk):
                _reg_row(e, "duplicate", _dup_reason(h), None)
                continue
            if sha:
                if h is not None:
                    # the earlier commit takes the identity: a later copy holding
                    # a slot loses it (judged candidates lose at their stage)
                    for t, w in list(work.slots.items()):
                        if (w.hotkey, w.trained_pointer) == (h["hotkey"], h["pointer"]):
                            plan.evict_after.append((t, w.hotkey, w.trained_pointer))
                            del work.slots[t]
                            _reg_row(LaneCEntry(**{**e.__dict__, "miner_hotkey": w.hotkey,
                                                   "miner_uid": int(w.uid), "target": t,
                                                   "trained_pointer": w.trained_pointer,
                                                   "recipe_ref": w.recipe_ref}),
                                     "duplicate", f"slot holder evicted: {e.miner_hotkey[:12]} "
                                     f"committed the same weights earlier (block {cb})",
                                     None, evicted="slot")
                work.mkt_seen[sha] = {"hotkey": e.miner_hotkey, "pointer": e.trained_pointer,
                                      "target": e.target, "commit_block": cb, "train_block": tb}
            new_ok.append(e)
            new_cb[(e.miner_hotkey, e.trained_pointer)] = cb
        # candidate order: confirms, this settlement's new entries, deferred carries
        ordered = ([c for c in cands if c["phase"] == "confirm"]
                   + [_candidate(e, "judge", None,
                                 commit_block=new_cb[(e.miner_hotkey, e.trained_pointer)])
                      for e in new_ok]
                   + [c for c in cands if c["phase"] != "confirm"])
        if ordered:
            backlog.append({"round_id": rid, "block": int(block),
                            "created_block": int(getattr(manifest, "created_block", block)),
                            "candidates": ordered, "arbiter": "", "age": 0})
        if len(work.mkt_seen) > MKT_SEEN_KEEP:
            keep = sorted(work.mkt_seen.items(), key=lambda kv: (-int(kv[1]["commit_block"]),
                                                                 kv[0]))[:MKT_SEEN_KEEP]
            work.mkt_seen = dict(keep)
        plan.backlog, plan.seen = backlog, work.mkt_seen
        if self._mkt_is_main():
            holders = _slot_holders(work, ctx)
            todo = [item for item in backlog if not item.get("arbiter")]
            if self.mkt_eval_hosts is not None:
                # offloaded: off the settlement thread (the pod may take
                # minutes to come up; the trainer's arbiter waits for it)
                self._mkt_publish_async(copy.deepcopy(todo), dict(holders), ctx=ctx)
            else:
                for item in todo:
                    self._mkt_publish_round(item, holders=holders)
        from .lane_c import OUTCOME_BELOW_HOLDER, OUTCOME_CONFIRMED, _holder_dict

        for j in judgements:
            if j.phase == "confirm" and j.passed:
                holder = work.slots.get(j.entry.target)
                holds = holder is not None and (holder.hotkey, holder.trained_pointer) == (
                    j.entry.miner_hotkey, j.entry.trained_pointer)
                j.outcome = OUTCOME_CONFIRMED if holds else OUTCOME_BELOW_HOLDER
                if not holds:
                    j.holder = _holder_dict(holder)
        return plan, judgements + reg

    def _mkt_holder_uid(self, registration: Callable, target: str, hotkey: str, block: int,
                        *, rid: str, plan: MktPlan):
        """A slot holder's uid AS OF ``block`` (the boundary − 1), recorded on
        ``plan``. Sources, in order (owner 2026-11-24 — no chain history
        needed): this node's boundary snapshot (read live at the boundary,
        :meth:`_lane_c_registration_tick`); the MAIN validator's signed
        receipt record for this round (a late node catching up); else the
        chain read (whose client falls back to an archive endpoint for a
        pruned block)."""
        found, uid = self._mkt_known_registration(hotkey, block, rid=rid)
        if not found:
            uid = self._mkt_registration(registration, hotkey, block, rid=rid)
        plan.registrations.append({"target": str(target), "hotkey": str(hotkey),
                                   "uid": None if uid is None else int(uid)})
        return uid

    def _lane_c_reg_snapshots(self):
        from ..shared.lane_c_registrations import RegistrationSnapshots

        snaps = self.__dict__.get("_lane_c_reg_snaps")
        if snaps is None:
            snaps = RegistrationSnapshots(Path(
                str(self.cfg.validator.state_db_path) + ".lane_c_registrations.json"))
            self.__dict__["_lane_c_reg_snaps"] = snaps
        return snaps

    def _mkt_known_registration(self, hotkey: str, block: int, *, rid: str
                                ) -> tuple[bool, int | None]:
        """``(found, uid)`` from a recorded source (never the chain)."""
        from ..shared.lane_c_registrations import recorded_uid

        try:
            snap = self._lane_c_reg_snapshots().get(int(block))
        except Exception:  # noqa: BLE001 — a cache fault falls through
            snap = None
        if snap is not None:
            uid = snap.get(str(hotkey))
            return True, (None if uid is None else int(uid))
        return recorded_uid(self._mkt_main_registrations(rid), hotkey, int(block))

    def _mkt_main_registrations(self, rid: str) -> dict | None:
        """The MAIN validator's signed ``lane_c_registrations`` record for
        round ``rid`` (verified against ``[netconfig] owner_hotkey``), or
        None — absent, unreadable or unsigned (cached per round)."""
        cache = self.__dict__.setdefault("_mkt_main_reg_cache", {})
        if rid in cache:
            return cache[rid]
        rec = None
        main = self._mkt_main_hotkey()
        if main:
            from ..shared.hippius import receipt_round_key
            from ..shared.receipt import load_receipt, verify_receipt_signature

            try:
                receipt = load_receipt(self._mkt_store().get_text(
                    receipt_round_key(str(rid), main)))
                if (str(receipt.round_id) == str(rid)
                        and verify_receipt_signature(receipt, main)):
                    rec = getattr(receipt, "lane_c_registrations", None)
            except Exception:  # noqa: BLE001 — not published yet / unreadable
                rec = None
        if rec is not None:
            cache[rid] = rec
        return rec

    def _lane_c_registration_tick(self, client: object) -> None:
        """Snapshot ``hotkey → uid`` of the netuid AS OF the current
        settlement boundary − 1 while that block is fresh on any node (lane C
        active only; one metagraph read per boundary). Never raises."""
        from ..shared.lane_c_registrations import SNAPSHOT_FRESH_BLOCKS

        lc = getattr(self.cfg, "lane_c", None)
        fn = getattr(client, "hotkey_uids_at_block", None)
        if lc is None or not lc.from_block or fn is None:
            return
        try:
            cur = int(client.current_block())  # type: ignore[attr-defined]
            grid = int(self._lane_b_epoch_blocks(cur))
            b = cur // grid * grid
            if b <= 0 or not lc.active(b):
                return
            snaps = self._lane_c_reg_snapshots()
            key = b - 1
            if snaps.get(key) is not None or cur - key > SNAPSHOT_FRESH_BLOCKS:
                return
            snaps.put(key, dict(fn(key)))
        except Exception as e:  # noqa: BLE001 — the settlement falls back
            log.warning("lane C: boundary registration snapshot failed (%s); the settlement "
                        "falls back to the main validator's record / the chain", e)

    @staticmethod
    def _mkt_registration(registration: Callable, hotkey: str, block: int, *, rid: str):
        """``registration(hotkey, block)`` — the hotkey's uid in the metagraph
        AS OF ``block`` (the settlement boundary − 1; historical state, an
        archive RPC endpoint) or None when deregistered. A failed read
        (``ChainError``, a timeout, any RPC fault) is a
        :class:`~cascade.validator.lane_c.LaneCTransient`: the settlement
        HOLDS with the lane-C read grace — never a verdict, never an
        unclassified exception out of the poll."""
        from .lane_c import LaneCTransient

        try:
            return registration(hotkey, int(block))
        except Exception as e:  # noqa: BLE001 — a chain read failure is a transient
            raise LaneCTransient(f"marketcast round={rid}: registration of slot holder "
                                 f"{hotkey[:12]} as of block {block} unreadable: "
                                 f"{type(e).__name__}: {e}") from e

    def _mkt_ended(self, comp, new: list, legacy: list, st, *, mc, rid: str
                   ) -> tuple[MktPlan, list]:
        """Concurrent competitions (owner 2026-10-20): MarketCast is past its
        ``until_block`` — no new verdicts. Every new entry, carried pending
        entry and backlog candidate gets an ``ended`` row (dropped, never
        judged; the miner is not at fault) and the backlog empties; slots
        and the identity registry stay (the wind-down pays from them)."""
        from .lane_c import LaneCJudgement

        why = (f"competition {comp.kind!r} ended at block {comp.until_block} — "
               "no new verdicts")
        js: list = []

        def _row(e: LaneCEntry, p: PendingEntry | None, phase: str) -> None:
            j = LaneCJudgement(e, False, "ended", why, phase=phase, k=1, alpha=float(mc.alpha),
                               base_pointer=mc.base_pointer,
                               mkt={"inconclusive": "", "round": rid, "at": "register"})
            if p is not None:
                j.outcome = "dropped"
                j.carried = p.kind == "deferred"
                j.origin_round = str(p.round_id)
            elif phase == "confirm":
                j.outcome = "dropped"
            js.append(j)

        for e in new:
            _row(e, None, "judge")
        for p in sorted(legacy, key=PendingEntry.sort_key):
            _row(LaneCEntry(**p.entry), p, "confirm" if p.kind == "confirm" else "judge")
        ctx = current_ctx(getattr(self, "cfg", None))
        for item in ctx.backlog(st):
            for c in item.get("candidates") or ():
                e, phase, p = _unpack(c)
                _row(e, p, phase)
        return MktPlan(backlog=[], seen=ctx.seen(st)), js

    def _mkt_apply_item(self, item: dict, work, *, settle_rid: str, settle_block: int,
                        era_index: int) -> tuple[list, list, list]:
        """Judge one arbitrated backlog round against ``work`` (mutated: its
        wins are recorded) → ``(judgements, offered wins, pending entries)``."""
        from ..shared.lane_c import TargetWin
        from .lane_c import (
            OUTCOME_DEFERRED,
            OUTCOME_PENDING,
            _best_key,
            _confirmed_win,
            _pending_of,
            resolve_outcomes,
        )

        mc = self._mkt_cfg()
        R, B = str(item["round_id"]), int(item["block"])
        arbiter = str(item["arbiter"])
        rep, sha, why = self._mkt_read_recorded(R, arbiter)
        cands = [_unpack(c) for c in item["candidates"]]
        # red-team F6: the family counts OWNERS (coldkeys), weighted Bonferroni
        ks = mkt_family_sizes([e for e, ph, _p in cands if ph == "judge"])
        # DEC-CA-0075: the competition spec's test family (bonferroni = today)
        from ..shared.rules_menu import family_size

        ks = {key: family_size(mc.test_family, v) for key, v in ks.items()}
        info = {"report_key": current_ctx(getattr(self, "cfg", None)).report_key(R), "report_sha256": sha,
                "round": R,
                "round_block": B}
        if item.get("expired"):
            info["report_expired"] = True
        if rep is not None:
            info["snapshot_sha256"] = str(rep.snapshot.get("sha256", ""))
            info["snapshot_end"] = float(rep.snapshot.get("end", 0) or 0)
        from ..shared.era import era_length_blocks

        settle_eb = int(era_length_blocks(self.cfg.round, int(settle_block)))
        js = []
        for e, phase, p in cands:
            base = e.marketcast or {}
            if base.get("base_ref") != mc.base_ref or base.get("base_sha256") != mc.base_sha256:
                from .lane_c import OUTCOME_DROPPED, LaneCJudgement

                js.append(LaneCJudgement(e, False, "stale", "pinned MarketCast base changed",
                                          phase=phase, outcome=OUTCOME_DROPPED, base_pointer=mc.base_pointer))
                continue
            k = 1 if phase == "confirm" else ks.get(
                (e.target, e.miner_hotkey, e.trained_pointer), 1)
            pm = dict(p.mkt or {}) if p is not None else {}
            origin = B
            if p is not None:
                origin = int(pm.get("origin_block", p.block))
                if pm.get("infra_at") is not None:
                    origin += B - int(pm["infra_at"])      # infra time never eats the wait
            fresh = float(pm.get("fresh_after", 0.0) or 0.0) if phase == "confirm" else 0.0
            rec = {**info, "train_time": mkt_train_time(e), "owner": mkt_owner(e),
                   # the round whose trainer-signed manifest carried the entry
                   "entry_round": str(pm.get("entry_round", "") or R) if p is not None else R}
            if phase == "confirm":
                rec["fresh_after"] = fresh
            h = _superseded(work.mkt_seen, e)
            rows, why_e = None, why
            if h is not None:
                from .lane_c import LaneCJudgement

                j = LaneCJudgement(e, False, "duplicate", _dup_reason(h), phase=phase, k=int(k),
                                   alpha=float(mc.alpha) / max(1, int(k)),
                                   base_pointer=mc.base_pointer,
                                   mkt={**rec, "inconclusive": "", "origin_block": origin,
                                        "within_wait": False})
            else:
                if rep is not None:
                    rows = rep.entry_for(e.miner_hotkey, e.trained_pointer, phase)
                    if rows is None:
                        why_e = "the main validator's report lacks this entry"
                    elif not rows.deferred and not math.isclose(
                            float(rows.after), entry_after(e, mc, fresh_after=fresh),
                            rel_tol=0.0, abs_tol=1e-6):
                        rows, why_e = None, (f"the report cuts windows after {rows.after}, the "
                                             f"trainer's dispatch time gives "
                                             f"{entry_after(e, mc, fresh_after=fresh)}")
                    elif not rows.deferred:
                        # DEC-CA-0074: the signed draw must be the rule's for
                        # this round, entry and phase (the hash is the audit's)
                        why_d = _draw_mismatch(e, rows, mc, phase=phase, block=B,
                                               fresh_after=fresh)
                        if why_d:
                            rows, why_e = None, f"the report's window draw: {why_d}"
                j = judge_marketcast_entry(
                    e, rows, mc=mc, k=k, phase=phase, seed=f"{R}|{e.miner_hotkey}|{e.target}",
                    round_block=B, origin_block=origin, base_pointer=mc.base_pointer, report=rec,
                    why_infra=why_e, infra_deferrals=int(pm.get("infra_deferrals", 0) or 0),
                    # the duration is on the era grid at the CONFIRMING settlement
                    lc=self.cfg.lane_c, era_blocks=settle_eb)
            if p is not None:
                j.carried = phase == "judge"
                j.origin_round = str(p.round_id)
            js.append(j)
        resolve_outcomes(work, js, round_id=settle_rid, settle_block=settle_block,
                         era_index=era_index)
        offered: list[TargetWin] = []
        for j in sorted((j for j in js if j.phase == "confirm" and j.passed),
                        key=lambda j: (j.entry.target, _best_key(j))):
            w = _confirmed_win(j, round_id=settle_rid, settle_block=settle_block,
                               era_index=era_index)
            work.record(w)
            offered.append(w)
        pend = [_pending_of(j, round_id=R, settle_block=B) for j in js
                if j.outcome in (OUTCOME_PENDING, OUTCOME_DEFERRED)]
        return js, offered, pend

    def _mkt_near_dups(self, entries: list, datas: dict, holders: dict, *, snap, ft_cache: dict,
                       round_id: str, sidecars: dict) -> list:
        """Attach each computed entry's ``near_dup`` record (red-team F4)
        and stage the reference sidecars in ``sidecars``."""
        from .mkt_offload import MktOffloadError

        mc = self._mkt_cfg()
        out = []
        same_target: dict = {}
        for (hk, ptr, ph), v in datas.items():
            tgt = next((r.target for r in entries if (r.hotkey, r.trained_pointer, r.phase)
                        == (hk, ptr, ph)), "")
            same_target.setdefault(tgt, {})[(hk, ptr, ph)] = v
        for r in entries:
            got = datas.get((r.hotkey, r.trained_pointer, r.phase))
            if got is None or r.fault or r.deferred:
                out.append(r)
                continue
            refs = _near_dup_refs(r, got[1], same_target.get(r.target, {}), holders)
            try:
                nd, blob = near_dup_record(r, got[0], refs, snap=snap, mc=mc,
                                           load_ft=self._mkt_load_ft, ft_cache=ft_cache,
                                           round_id=round_id)
            except MktOffloadError:
                raise                       # the pod, not the entry: the round waits
            except Exception as e:  # noqa: BLE001 — never blocks the report
                log.warning("marketcast round=%s: near-dup test failed for %s (%s)", round_id,
                            r.hotkey[:12], e)
                nd, blob = {}, None
            if nd and blob is not None:
                sidecars[nd["quantiles_key"]] = blob
                r = replace(r, near_dup=nd)
            try:
                r, hblob = holder_record(r, got[0], holders.get(r.target), snap=snap, mc=mc,
                                         load_ft=self._mkt_load_ft, ft_cache=ft_cache,
                                         round_id=round_id)
            except MktOffloadError:
                raise
            except Exception as e:  # noqa: BLE001 — never blocks the report
                log.warning("marketcast round=%s: holder pairing failed for %s (%s)", round_id,
                            r.hotkey[:12], e)
                hblob = None
            if hblob is not None:
                sidecars[r.holder["quantiles_key"]] = hblob
            out.append(r)
        return out

    # ── the main validator's rows ──

    def _mkt_publish_async(self, todo: list, holders: dict, *, ctx: MktCtx | None = None) -> bool:
        """Offloaded publishing: one background run over ``todo`` (False =
        a run is still going; the next settlement retries what it leaves)."""
        t = self.__dict__.get("_mkt_pub_thread")
        if t is not None and t.is_alive():
            log.info("marketcast: an offloaded publish run is still going; %d round(s) "
                     "retried at the next settlement", len(todo))
            return False
        if not todo:
            return False
        import threading

        ctx = ctx if ctx is not None else current_ctx(getattr(self, "cfg", None))

        def _run():
            # the instance travels with the run (W-CMP2)
            with mkt_context(ctx):
                self._mkt_publish_remote(todo, holders)

        t = threading.Thread(target=_run, name=f"mkt-publish-{ctx.kind}", daemon=True)
        self.__dict__["_mkt_pub_thread"] = t
        t.start()
        return True

    def _mkt_publish_remote(self, todo: list, holders: dict) -> int:
        """Lease the MarketCast pod, publish every round of ``todo`` still
        lacking a report on it, release. Never raises; returns the number
        published (0 when no pod came up within ``mkt_host_wait_s``)."""
        from ..shared.hippius import ObjectNotFound
        from .mkt_offload import MktLease, MktSession

        store = self._mkt_store()
        need = []
        for item in todo:
            try:
                store.get_text(current_ctx(getattr(self, "cfg", None)).report_key(str(item["round_id"])))
            except ObjectNotFound:
                need.append(item)
            except Exception as e:  # noqa: BLE001 — unknown: retried next settlement
                log.warning("marketcast round=%s: report existence unreadable (%s)",
                            item["round_id"], e)
        # the pod-free prechecks BEFORE the lease: a round whose snapshot or
        # block hash is unavailable would only fail inside the session, so a
        # missing pool_mkt snapshot rented (and paid for) a pod every settlement
        need = [item for item in need if self._mkt_round_ready(item)]
        if not need:
            return 0
        rid0 = str(need[0]["round_id"])
        lease = (self.mkt_lease_fn or MktLease)(self.mkt_eval_hosts)
        published = 0
        try:
            host = lease.acquire(rid0, float(self.mkt_host_wait_s))
            if host is None:
                log.error("marketcast: no offload pod published to %s within %.0fs; %d "
                          "round(s) retried at the next settlement", self.mkt_eval_hosts,
                          self.mkt_host_wait_s, len(need))
                return 0
            sess = (self.mkt_session_fn or MktSession)(host)
            try:
                sess.start()
                self.__dict__["_mkt_sess"] = sess
                beat = getattr(lease, "heartbeat", None)
                with (beat(rid0) if callable(beat) else contextlib.nullcontext()):
                    for item in need:
                        lease.touch(str(item["round_id"]))
                        published += bool(self._mkt_publish_round(item, holders=holders))
            finally:
                self.__dict__.pop("_mkt_sess", None)
                sess.close()
        except Exception as e:  # noqa: BLE001 — never kills the validator
            log.error("marketcast: offloaded publish run failed (%s); retried at the next "
                      "settlement", e)
        finally:
            lease.release()
        return published

    def _mkt_round_ready(self, item: dict) -> bool:
        """Whether round ``item`` can be computed at all — its snapshot is
        published and its block hash readable — checked WITHOUT a pod (the
        same reads :meth:`_mkt_publish_round` repeats inside the session).
        False ⇒ logged and retried at the next settlement; never raises."""
        rid = str(item.get("round_id", ""))
        try:
            got = self._mkt_snapshot(int(item["block"]))
        except Exception as e:  # noqa: BLE001
            log.error("marketcast round=%s: snapshot unavailable (%s); no offload pod "
                      "requested — retried at the next settlement", rid, e)
            return False
        if got is None:
            log.error("marketcast round=%s: no pool_mkt snapshot for block %s; no offload "
                      "pod requested — retried at the next settlement", rid, item["block"])
            return False
        try:
            self._mkt_block_hash(int(item["block"]))
        except Exception as e:  # noqa: BLE001 — a chain read: retried next settlement
            log.error("marketcast round=%s: block %s hash unreadable (%s); no offload pod "
                      "requested — retried at the next settlement", rid, item["block"], e)
            return False
        return True

    def _mkt_publish_round(self, item: dict, *, holders: dict | None = None) -> bool:
        """Main validator: compute + sign + publish round ``item``'s report
        unless one is already there (never overwritten — a trainer may have
        recorded its sha256). Per entry a compute failure is retried
        :data:`MKT_COMPUTE_ATTEMPTS` times, then marked ``infra_deferred`` —
        unless every attempt raised the SAME :class:`MktForwardError` (a
        non-infra exception inside the fine-tune's forward: red-team F7),
        which is the entry's ``fault``. Every computed entry is then measured
        against its SENIOR references (earlier-committed candidates of its
        target this round + the target's slot holder, ``holders``) and
        carries the closest one's ``near_dup`` record (red-team F4). Never
        raises: a missing snapshot / a failed publish is retried at the next
        settlement (the trainer records ``"absent"`` if it never lands)."""
        from ..shared.hippius import ObjectNotFound, StorageError

        rid = str(item["round_id"])
        key = current_ctx(getattr(self, "cfg", None)).report_key(rid)
        store = self._mkt_store()
        try:
            store.get_text(key)
            return False                                  # already published
        except ObjectNotFound:
            pass
        except Exception as e:  # noqa: BLE001 — existence unknown: never overwrite blind
            log.warning("marketcast round=%s: report existence unreadable (%s); retried at the "
                        "next settlement", rid, e)
            return False
        mc = self._mkt_cfg()
        sess = self.__dict__.get("_mkt_sess")
        if self.mkt_eval_hosts is not None and sess is None:
            log.warning("marketcast round=%s: offload configured but no pod session; "
                        "retried at the next settlement", rid)
            return False
        try:
            got = self._mkt_snapshot(int(item["block"]))
        except Exception as e:  # noqa: BLE001
            log.error("marketcast round=%s: snapshot unavailable (%s); retried at the next "
                      "settlement", rid, e)
            return False
        if got is None:
            log.error("marketcast round=%s: no snapshot under %s/ for block %s; retried at the "
                      "next settlement", rid, mc.pool_prefix, item["block"])
            return False
        snap, skey, ssha = got
        try:
            # DEC-CA-0074: the round block's hash ranks the embargo draw (it
            # did not exist while the legs trained)
            bhash = self._mkt_block_hash(int(item["block"]))
        except Exception as e:  # noqa: BLE001 — a chain read: retried next settlement
            log.error("marketcast round=%s: block %s hash unreadable (%s); retried at the "
                      "next settlement", rid, item["block"], e)
            return False
        cache = getattr(self, "_mkt_base_cache", None)
        if cache is None or len(cache) > MKT_BASE_CACHE_MAX:
            cache = self._mkt_base_cache = {}
        from .mkt_offload import MktOffloadError

        base_fc, base_err = None, ""
        for _ in range(MKT_COMPUTE_ATTEMPTS):
            try:
                base_fc = self._mkt_base()
                break
            except MktOffloadError as e:
                log.error("marketcast round=%s: offload pod failed loading the base (%s); "
                          "retried at the next settlement", rid, e)
                return False
            except Exception as e:  # noqa: BLE001 — the base must load: infra
                base_err = f"{type(e).__name__}: {e}"
        entries, sidecars = [], {}
        ft_cache: dict = {}
        datas: dict = {}
        for c in item["candidates"]:
            e, phase, p = _unpack(c)
            fresh = float((p.mkt or {}).get("fresh_after", 0.0) or 0.0) \
                if (p is not None and phase == "confirm") else 0.0
            after = entry_after(e, mc, fresh_after=fresh)
            stub = MktEntryRows(e.miner_hotkey, e.trained_pointer, e.target, phase, after)
            draw = entry_draw(e, mc, phase=phase, block=int(item["block"]),
                              block_hash=bhash, fresh_after=fresh)
            n = int((p.mkt or {}).get("infra_deferrals", 0) or 0) + 1 if p is not None else 1
            # the trainer requeues (uncharged) an entry whose deferral is its last
            cap = (f" — past max_infra_deferrals ({n} > {mc.max_infra_deferrals})"
                   if n > int(mc.max_infra_deferrals) else "")
            if base_fc is None:
                entries.append(replace(stub, status=INFRA_DEFERRED,
                                       infra_reason=f"pinned base failed to load: "
                                                    f"{base_err}"[:240] + cap))
                continue
            err = ""
            fwd: list[str] = []
            for _ in range(MKT_COMPUTE_ATTEMPTS):
                try:
                    rows, data = compute_entry_rows(
                        e, phase=phase, after=after, snap=snap, mc=mc, base_fc=base_fc,
                        load_ft=self._mkt_load_ft, base_cache=cache, round_id=rid,
                        ft_cache=ft_cache, draw=draw)
                    break
                except MktOffloadError as x:
                    # the POD failed, not this entry: nothing is deferred or
                    # charged — the whole round waits for the next settlement
                    log.error("marketcast round=%s: offload pod failed (%s); retried at "
                              "the next settlement", rid, x)
                    return False
                except Exception as x:  # noqa: BLE001 — OOM / fetch / library: retried
                    err = f"{type(x).__name__}: {x}"
                    fwd.append(str(x) if isinstance(x, MktForwardError) else "")
                    rows = data = None
            if rows is None and len(fwd) == MKT_COMPUTE_ATTEMPTS and fwd[0] and len(set(fwd)) == 1:
                # red-team F7: the fine-tune's own forward raised the same
                # non-infra exception on every attempt — deterministic, its fault
                entries.append(replace(stub, fault=f"fine-tune forward failed identically "
                                                   f"{MKT_COMPUTE_ATTEMPTS}x: {fwd[0]}"[:300]))
                continue
            if rows is None:
                entries.append(replace(stub, status=INFRA_DEFERRED,
                                       infra_reason=f"compute failed {MKT_COMPUTE_ATTEMPTS}x: "
                                                    f"{err}"[:240] + cap))
                continue
            entries.append(rows)
            if data is not None:
                sidecars[rows.quantiles_key] = data
                datas[(rows.hotkey, rows.trained_pointer, rows.phase)] = (data, _seniority(c))
        try:
            entries = self._mkt_near_dups(entries, datas, holders or {}, snap=snap,
                                          ft_cache=ft_cache, round_id=rid, sidecars=sidecars)
        except MktOffloadError as x:
            log.error("marketcast round=%s: offload pod failed in the reference tests (%s); "
                      "retried at the next settlement", rid, x)
            return False
        entries.sort(key=lambda r: (r.target, r.hotkey, r.trained_pointer, r.phase))
        report = MktReport(round_id=rid, created_block=int(item.get("created_block", 0)),
                           snapshot=snapshot_record(snap, skey, ssha), entries=tuple(entries),
                           coverage_levels=tuple(mc.coverage_levels),
                           tail_levels=tuple(mc.tail_levels))
        signer = getattr(self, "bench_signer", None)
        if signer is not None:
            report = sign_mkt_report(report, signer)
        else:
            log.warning("marketcast round=%s: no wallet — publishing UNSIGNED rows (every "
                        "validator will defer them as infra)", rid)
        text = report.to_json()
        try:
            for qk, data in sorted(sidecars.items()):     # the sidecars first: rows bind them
                try:
                    store.put_bytes(qk, data, content_type="application/octet-stream",
                                    acl="public-read")
                except StorageError:
                    store.put_bytes(qk, data, content_type="application/octet-stream")
            try:
                store.put_text(key, text, content_type="application/json", acl="public-read")
            except StorageError:
                store.put_text(key, text, content_type="application/json")
        except Exception as e:  # noqa: BLE001 — retried next settlement
            log.error("marketcast round=%s: publish failed (%s); retried at the next "
                      "settlement", rid, e)
            return False
        log.info("marketcast round=%s: published %d entr%s → %s", rid, len(entries),
                 "y" if len(entries) == 1 else "ies", key)
        return True


def resolve_pinned_base_dir(mc, *, cache_dir=None) -> Path:  # box-side: no timestral import (torch)
    """The pinned base's verified local directory (nothing loaded)."""
    from ..eval.timestral_adapter import resolve_pinned_base

    return resolve_pinned_base(mc.base_ref, mc.base_sha256, cache_dir=cache_dir)


def _near_dup_refs(rows: MktEntryRows, sen: tuple, datas: dict, holders: dict) -> list:
    """The SENIOR references of one computed entry: the round's other
    computed entries on the same target committed strictly earlier
    (``(commit block, train block, hotkey)``, never a UID) and the target's
    slot holder when it is senior; never the entry's own hotkey."""
    refs = []
    for (hk, ptr, _ph), (_d, sen2) in datas.items():
        if hk != rows.hotkey and sen2 < sen and (hk, ptr) not in {(r[0], r[1]) for r in refs}:
            refs.append((hk, ptr, int(sen2[0])))
    h = holders.get(rows.target)
    if h is not None and h[0] != rows.hotkey and (int(h[2]), -1, "") < sen and (
            h[0], h[1]) not in {(r[0], r[1]) for r in refs}:
        refs.append((h[0], h[1], int(h[2])))
    return refs


def fold_mkt_plan(st, plan: MktPlan | None, ctx: MktCtx | None = None) -> list:
    """Replay a :class:`MktPlan` onto the live state (the fold) in the exact
    order judge time applied it to its scratch copy (``_mkt_settle``): evict
    expired / deregistered holders, offer the wins in order, evict the
    holders superseded by an earlier identical commit, take the backlog and
    the registry. Returns the wins that took a slot."""
    took: list = []
    if plan is None:
        return took

    def _evict(rows):
        for t, hk, ptr in rows:
            cur = st.slots.get(t)
            if cur is not None and (cur.hotkey, cur.trained_pointer) == (hk, ptr):
                del st.slots[t]

    _evict(plan.evict)
    for w in plan.offered:
        if st.record(w):
            took.append(w)
    _evict(getattr(plan, "evict_after", ()))
    ctx = ctx if ctx is not None else MktCtx(MKT_KIND, None)
    ctx.store(st, list(plan.backlog), dict(plan.seen))
    st.pending = [p for p in st.pending if not ctx.is_target(p.target)]
    return took


__all__ = [
    "MKT_COMPUTE_ATTEMPTS", "MarketCastMixin", "MktCtx", "current_ctx", "mkt_context", "MktFault", "MktPlan", "anchor_time",
    "compute_entry_rows", "holder_record", "entry_after", "entry_draw", "entry_embargo_mc",
    "EMBARGO_RECORD_KEYS", "embargo_scan_problem", "fold_mkt_plan", "judge_marketcast_entry",
    "mkt_prechecks", "mkt_train_time", "snapshot_record", "suite_of",
]
