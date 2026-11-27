"""Box-side lane D settlement (LANE_D_PLAN §1.3–§1.5; W5a–h and W7a–c
deepen it).

:func:`settle_lane_d` runs once per settlement boundary on the main
validator. Unarmed (``not cfg.active(boundary)``) it returns
``report=None, pay=None`` and touches nothing. Armed it:

1. registers newly revealed entries in ``lane_d_state.json`` (``ready``
   TTL, duplicate digests, deregistered hotkeys → closed with a typed reason);
2. selects this settlement's judged entries (W5h,
   :mod:`~cascade.validator.lane_d_queue`): entries pending past
   ``max_reveal_delay_days`` close, the coldkey re-check at reveal refuses a
   changed coldkey and holds (bounded) an unreadable one, then FIFO by reveal
   time under the per-settlement and per-coldkey caps (the rest wait);
3. advances each judged entry's ``frozen_set`` (W5b,
   :mod:`~cascade.validator.lane_d_frozen`): the union over the newest
   snapshot ``pool_dir`` and the retained ``prior_pool_dirs`` of the windows
   the §1.3 verdicts of
   :func:`~cascade.validator.lane_d_forward.eligible_windows` admit for
   ``max(T_E, T_K)`` (both clocks after the bound, mirror lag, forecast
   products, present in S0, no backfill), excluding windows the entry was
   already judged on, capped at
   ``max_source_share`` with a drop seeded by the boundary, frozen at the
   floors (``min_windows``, ``min_clusters``). An unfrozen entry stays
   ``pending_floor``; nothing resets at an era boundary (T18). A cohort is
   the frozen members, judged on the intersection of their frozen sets that
   is eligible for ``max(T over cohort ∪ king)`` and has forecasts; the
   latest-T members are deferred until that intersection meets the floors.
   Each judged window is scored, and its report row taken, from ONE
   snapshot: the earliest loaded one admitting it for that bound (RV4); the
   report names it (``windows[].snapshot_sha256``, ``pool.snapshots``).
   At ``until_block`` nothing is judged and every open entry closes
   (``lane_closed``);
4. scores every window for the king and each member from the eval host's
   ``(horizon, 9)`` quantile forecasts (``score_window``) into report rows;
5. decides with the rev6.18 A3 combined rule (F4, :func:`judge_cohort`):
   per window the ratio ``s_bar / s_member`` of the lane D per-window
   geomeans, clamped to ``[window_ratio_floor, 1 / window_ratio_floor]``,
   its log averaged over the cohort's live (forward: fresh + embargo)
   windows into ``L_live`` and over the settlement's public windows
   (GiftEvalPretrain + RMISC, the same for every cohort) into ``L_pub``;
   ``L = (1 − w) L_live + w L_pub`` with ``w = public_weight``. One max-T
   two-part cluster bootstrap over the cohort (live clusters = source,
   public clusters = ``<source>|<family>``, resampled within each half,
   recombined per replicate) gives each member's LCB on the gain scale
   ``1 − exp(−L)``; a member clears iff ``LCB ≥ d_margin`` AND its live
   point gain > 0 AND (``w > 0``) its public point gain > 0. Each verdict
   carries its per-half, per-source and per-domain point gains and window
   counts (A8). When a miner holds KING_D, every member is also judged, the
   same way and on the same set, against the current house, and clears only
   if it clears both (Proposed change 3: the house ranks first on ties, so a
   copy of the house never clears); the house's rows sit next to the king's.
   rev6.11 (b): there is no confirmation duel. A first clear is the crowning
   candidate and is crowned at that settlement when it passes the §1.3b
   gate (``confirm_on_king_change`` is a legacy key, parsed and ignored;
   there is no D20 king-change rule). rev6.18 A7: the A10 low-frequency
   check is removed; D24: there are no backtest windows;
5b. gates the crowning candidates (W5e, §1.3b): every verdict that cleared,
   plus every entry still holding a first clear the
   gate could not decide (``bench_hold``, status ``hold``), is swept with
   KING_D and the house in the settlement's ``bench`` session on the ONE
   suite drawn from ``boundary_block_hash``
   (:func:`~cascade.validator.lane_d_gate.gate_record`; the era's
   ``sentinel`` is swept and recorded, never gates) and decided by
   :func:`~cascade.validator.lane_d_bench.paired_gate`: ``pass`` is
   eligible to crown (the best LCB among the passing candidates is crowned),
   ``gate_regression`` is ``lost``, ``checkpoint_fault`` is ``closed`` and
   ``bench_hold`` stays open, its clear kept in ``gate_hold`` and re-gated
   at a later settlement without a new duel. W5g: holds are bounded; the
   ``gate_hold`` record counts the consecutive holds of one held clear
   (``holds``) and the hold after ``max_bench_holds`` re-queues the entry
   unjudged (``pending``/``bench_hold``, its clear dropped, never charged;
   a re-gate is then a typed ``bench_hold`` skip). FAIL CLOSED: with no ``bench``
   or no ``boundary_block_hash`` (the legacy call) every candidate is
   ``bench_hold``: never crowned, lost or charged. The report carries the
   ``gate`` record whenever the gate ran; a crowned report always has one;
6. writes the signed ``lane-d-report/1`` (W7a: the schema file
   ``schemas/lane_d_report_1.json``; every entry considered and not judged is
   a typed skip, court slots that burn name a ``burn_reason``; ``pool.snapshots``
   binds every snapshot a judged window's version came from, the newest
   always included, and each window names its ``snapshot_sha256``; rev6.6:
   given the W5d ``armed`` record, ``armed_contexts`` / ``armed_evidence``,
   checked by :func:`check_armed`). The
   signed report is checked by ``validate_report``: a report outside the
   schema raises and nothing is returned or saved (fail closed). Then it
   saves the new state and returns the pay
   record ``{"share", "court_uids"}`` — king first (``None`` while the house
   holds KING_D: that slot burns, D21), then up to ``reward_prior_kings``
   prior MINER kings (an unregistered one is ``None`` and burns).

W5f (§1.4 "The house", §2 items 1-2, T10) is the house's entry and refresh:

- ``house`` is the verified record ``{"digest", "reveal_ts"}`` a signed
  pointer enters (:mod:`~cascade.validator.lane_d_house`); armed, anything
  else is a typed ``LaneDHouseError`` before any state, report or pool is
  read or written;
- a new house while the house holds KING_D moves the throne in place (no
  duel; the slot keeps burning, ``prior_kings`` untouched);
- a new house while a miner holds KING_D is the second bar of every miner
  cohort from that boundary AND enters as a challenger
  (``lane_d_state.HOUSE_ENTRY_KEY``): judged forward-only against KING_D in
  its own cohort (``members == [house]``, no ``house_digest``), first clear,
  A10 and the §1.3b gate (swept with KING_D as its only anchor: the house
  reference is itself). A cleared, passing house retakes KING_D: the dethroned miner king becomes
  the newest prior king, the slot burns (``house_king``) and ``crowned``
  stays null (a retake is not a miner crowning). A weaker house loses and
  is never in the court or a prior king. An open miner entry whose digest
  the new pointer names is a copy of the house (``duplicate_digest``).

W7c (§1.5 "State", invariant 5, T11) makes the settlement restart-safe and
binds it into a receipt record (:mod:`~cascade.validator.lane_d_state`):

- the state is read through the strict ``load_prior_state`` (a defective file
  is refused with its typed code before anything is written; once a report is
  persisted, a missing state is ``state_missing`` and a rolled-back one is
  ``state_digest_mismatch``, never genesis; ``expect_prior_sha256`` pins the
  prior to the last published posterior; ``restore_state`` is the way back);
- the signed report is written atomically to ``report_path`` BEFORE the state
  file, and the state's ``last_report`` binds the report's canonical body and
  the prior state digest, never the randomized signature;
- ``receipt`` = ``{schema, boundary, prior_state_sha256,
  posterior_state_sha256, report_sha256}`` (``None`` unarmed);
- every armed report binds the prior state digest inside its signed body
  (``extensions.prior_state_sha256``);
- a crash after the report landed is recovered by ADOPTING the persisted
  report (RV7 B1), never by re-sweeping the bench: it must verify against the
  settling validator's key (``report_unverified``), name this boundary and
  bind the loaded state's digest (``report_conflict``); the settlement is then
  derived with the report's recorded gate outcomes (a ``pass`` crowns, a
  ``bench_hold`` holds) and no bench call, its body must equal the persisted
  one (else ``report_conflict``), the file is kept byte-for-byte and the state
  is written; every refusal writes nothing; a re-call at the settled boundary
  returns the persisted report, pay and receipt with nothing re-scored,
  re-gated or written; an older boundary is ``stale_boundary``.

W14 (the lane D-only fast kill, Proposed design change 4): a settlement with
no lane D report goes through :func:`absent_settlement` instead. It reads
the prior state under the same fail-closed rules, writes nothing, refuses a
boundary already settled or whose report is persisted, and returns the
absent receipt record with ``pay`` = lane D's share and an empty court: the
caller pays it with ``lane_parts(..., d_absent=True)``, so only lane D's
share burns. KING_D, the court and every pending entry are untouched, and
the next real settlement judges and pays them.

Light imports only (numpy + the scorer's numpy half): this module is on the
validator package's import path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.bootstrap import clamped_log_ratios, log_gain, parts_maxt_lcbs
from ..eval.lane_d_channels import apply_pick, channel_plan, channel_record, forecast_shape
from ..eval.lane_d_scorer import (
    GATE_MIN_SOURCES,
    SCORING_LEVELS,
    SCORING_METHOD,
    effective_context,
    score_window,
)
from ..shared.lane_d_horizons import bands_on, cell_class
from .lane_d_bench import paired_gate, sweep_record_text
from .lane_d_draw import draw_band_windows, draw_windows, freq_class
from .lane_d_forward import eligible_windows, window_horizon
from .lane_d_frozen import advance_frozen_set
from .lane_d_gate import gate_record
from .lane_d_house import check_house
from .lane_d_queue import select_queue
from .lane_d_report import (
    REPORT_SCHEMA,
    SKIP_REASONS,
    rows_sha256,
    sign_report,
    validate_report,
)
from .lane_d_state import (
    ABSENT_REASONS,
    CONFIRM_MIGRATION_EXTENSION,
    CONFIRM_MIGRATION_TAG,
    HOUSE_ENTRY_KEY,
    KING_TENURE_KEY,
    PRIOR_STATE_EXTENSION,
    LaneDReceiptError,
    LaneDStateError,
    check_persisted_report,
    load_prior_state,
    load_report_file,
    make_absent_receipt,
    make_receipt,
    migrate_confirm_pending,
    refresh_house,
    report_digest,
    report_path,
    save_report_file,
    save_state,
    state_digest,
)

POOL_SCHEMA = "pool_d/1"
CLOCKS_FILE = "clk.npz"
# W-IMG (plan rev6.24 (e)): the eval host code identity a settlement ran on
EVAL_IMAGE_EXTENSION = "eval_image"
OPEN_STATUSES = ("pending", "hold")


@dataclass(frozen=True)
class LaneDSettlement:
    """``receipt`` is the W7c lane D receipt record (``None`` unarmed)."""

    report: dict | None
    pay: dict | None
    receipt: dict | None = None


# ── pool_d ──────────────────────────────────────────────────────────────────


def window_id(series_id: str, target_ts, horizon: int) -> str:
    """§1.3 stable window identity ``<series_id>:<target_start_ts>:h<h>``."""
    return f"{series_id}:{int(target_ts[0])}:h{int(horizon)}"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_feeds(raw, where: Path) -> dict:
    """Top-level ``feeds``: ``{feed: {"mirror_lag_hours": float | None,
    "forecast_product": bool}}``. An absent block parses to ``{}``: no feed is
    listed, so every window fails closed as ``unlisted_feed``
    (``lane_d_forward``)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: pool 'feeds' must be an object; got {type(raw).__name__}")
    out = {}
    for name, f in raw.items():
        if not isinstance(f, dict):
            raise ValueError(f"{where}: feed {name!r} must be an object")
        lag = f.get("mirror_lag_hours")
        if lag is not None and (isinstance(lag, bool) or not isinstance(lag, int | float)
                                or not math.isfinite(float(lag))):
            raise ValueError(f"{where}: feed {name!r} mirror_lag_hours {lag!r} is not a number")
        fp = f.get("forecast_product", False)
        if not isinstance(fp, bool):
            raise ValueError(f"{where}: feed {name!r} forecast_product {fp!r} is not a bool")
        out[str(name)] = {"mirror_lag_hours": None if lag is None else float(lag),
                          "forecast_product": fp}
        # W5a (rev6.10 (a)): the pinned per-feed embargo, carried unchanged
        # (absent stays absent): a missing or non-numeric value is the
        # window's refusal in lane_d_forward (unlisted_feed), never a pool error.
        if "embargo_hours" in f:
            out[str(name)]["embargo_hours"] = f["embargo_hours"]
    return out


def _parse_frontier(raw, sid: str, where: Path) -> list[tuple[int, int]] | None:
    """Per-series ``frontier``: ``[[as_of_ts, end_ts], ...]`` for each earlier
    snapshot holding the series; ``None`` when the key is absent or ``null``.
    ``lane_d_forward`` fails closed on ``None``: the series is
    ``absent_from_s0``, exactly like an empty list."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{where}: series {sid!r} frontier must be a list")
    out = []
    for rec in raw:
        if not isinstance(rec, list | tuple) or len(rec) != 2:
            raise ValueError(f"{where}: series {sid!r} frontier record {rec!r} "
                             "is not [as_of_ts, end_ts]")
        out.append((int(rec[0]), int(rec[1])))
    return out


_HEX64 = frozenset("0123456789abcdef")


def _parse_roots(raw, where: Path) -> list[str]:
    """Top-level ``first_seen_roots`` (B12): the daily first-seen Merkle roots
    the snapshot relies on, as sorted 64-hex strings. Absent (a W1.5-format
    snapshot) = ``[]``; W6fs makes the key mandatory."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: pool 'first_seen_roots' must be a list")
    for root in raw:
        if not (isinstance(root, str) and len(root) == 64 and set(root) <= _HEX64):
            raise ValueError(f"{where}: first-seen root {root!r} is not a sha256 hex")
    return sorted(set(raw))


def _band_clocks(d: Path, meta: dict):
    """The ``clk.npz`` of a band snapshot (owner 2026-11-27), checked against
    its metadata sha256; ``None`` for a pre-band snapshot (its region is the
    stored 64-point target)."""
    rec = meta.get("clocks")
    if rec is None:
        return None
    if not isinstance(rec, dict) or rec.get("file") != CLOCKS_FILE:
        raise ValueError(f"{d}: pool clocks record {rec!r} is not {{file: {CLOCKS_FILE!r}}}")
    if _file_sha(d / CLOCKS_FILE) != rec.get("sha256"):
        raise ValueError(f"{d}: {CLOCKS_FILE} does not hash to the metadata clocks.sha256")
    return np.load(d / CLOCKS_FILE)


def _base_window(s: dict, sid: str, d: Path) -> dict:
    source = str(s.get("source") or sid)
    return {"series_id": sid, "source": source, "feed": str(s.get("feed") or source),
            "freq": str(s.get("freq") or ""),
            "frontier": _parse_frontier(s.get("frontier"), sid, d),
            "seasonal_period": int(s.get("seasonal_period") or 1)}


def _band_windows(s: dict, sid: str, d: Path, hist, tgt, clk, bands, *, max_horizon: int,
                  slots: int, history_length: int) -> list[dict]:
    """Owner 2026-11-27: one series' forward candidates at their band
    horizons. The region is the series' newest ``target_points`` points (the
    tail of its history, then its target); each candidate's target is a
    region slice (:func:`~cascade.shared.lane_d_horizons.band_slots`) and its
    history every point before the cut (embargo-held points included: only
    the target must be unseen). Arrays are views of one per-series array."""
    from ..shared.lane_d_horizons import band_slots

    hist_a = np.asarray(hist[sid], dtype=np.float64)
    tgt_a = np.asarray(tgt[sid], dtype=np.float64)
    if clk is not None and sid in clk.files:
        c = np.asarray(clk[sid], dtype=np.int64)
        ts, seen = c[0], c[1]
    else:
        ts = np.asarray([int(t) for t in s.get("target_ts") or []], dtype=np.int64)
        seen = np.asarray([int(t) for t in s.get("first_seen_ts") or []], dtype=np.int64)
    if not ts.size or ts.size != seen.size or hist_a.shape[:-1] != tgt_a.shape[:-1]:
        return []
    full = np.concatenate([hist_a, tgt_a], axis=-1)
    if ts.size > full.shape[-1]:
        return []
    g = full.shape[-1] - ts.size
    base = _base_window(s, sid, d)
    out = []
    for band, h, start in band_slots(ts, base["freq"], bands, slots=slots,
                                     max_horizon=max_horizon):
        cut = g + start
        t_ts, t_seen = ts[start:start + h], seen[start:start + h]
        out.append({**base, "window_id": window_id(sid, t_ts, h), "h": int(h), "band": band,
                    "target_ts": t_ts, "first_seen_ts": t_seen, "end_ts": int(t_ts[-1]),
                    "history": full[..., max(0, cut - int(history_length)):cut],
                    "target": full[..., cut:cut + h]})
    return out


def load_pool(pool_dir, horizon: int, bands=None, *, max_horizon: int = 2048,
              slots: int = 1) -> dict:
    """One ``pool_d/1`` snapshot: ``{"snapshot", "sha256", "feeds",
    "first_seen_roots", "windows": [...]}`` with each window's history,
    target, both clocks, cluster, feed, frequency and frontier (``None`` when
    the series' ``frontier`` is absent or ``null``; eligibility then fails
    closed as ``absent_from_s0``).

    ``bands`` (owner 2026-11-27; ``None`` = the fixed ``horizon`` rule, byte
    for byte): each series yields its per-band candidates
    (:func:`_band_windows`), every window carrying its ``h`` and ``band``.
    A pre-band snapshot loads too: its region is its stored target."""
    d = Path(pool_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema") != POOL_SCHEMA:
        raise ValueError(f"{d}: pool schema {meta.get('schema')!r} is not {POOL_SCHEMA}")
    if int(meta.get("horizon", -1)) != int(horizon):
        raise ValueError(f"{d}: pool horizon {meta.get('horizon')} != [lane_d] horizon {horizon}")
    sha = hashlib.sha256("".join(_file_sha(d / n) for n in
                                 ("metadata.json", "hist.npz", "tgt.npz")).encode()).hexdigest()
    windows = []
    clk = _band_clocks(d, meta) if bands else None
    try:
        with np.load(d / "hist.npz") as hist, np.load(d / "tgt.npz") as tgt:
            for s in meta.get("series", []):
                sid = str(s["series_id"])
                if sid not in hist.files or sid not in tgt.files:
                    continue
                if bands:
                    windows += _band_windows(
                        s, sid, d, hist, tgt, clk, tuple(bands), max_horizon=int(max_horizon),
                        slots=int(slots),
                        history_length=int(meta.get("history_length") or 16384))
                    continue
                ts = [int(t) for t in s.get("target_ts") or []]
                seen = [int(t) for t in s.get("first_seen_ts") or []]
                target = np.asarray(tgt[sid], dtype=np.float64)
                windows.append({
                    "window_id": window_id(sid, ts, horizon) if ts else "",
                    **_base_window(s, sid, d),
                    "target_ts": ts, "first_seen_ts": seen,
                    "end_ts": s.get("end_ts"),
                    "history": np.asarray(hist[sid], dtype=np.float64), "target": target,
                })
    finally:
        if clk is not None:
            clk.close()
    return {"snapshot": str(meta.get("snapshot", "")), "sha256": sha,
            "feeds": _parse_feeds(meta.get("feeds"), d),
            "first_seen_roots": _parse_roots(meta.get("first_seen_roots"), d),
            "windows": windows}


def load_pool_for(pool_dir, cfg) -> dict:
    """:func:`load_pool` under ``[lane_d]`` ``cfg``: the band rule when
    ``horizon_bands`` is non-empty, else the fixed ``horizon``."""
    from ..shared.lane_d_horizons import bands_on

    if not bands_on(cfg):
        return load_pool(pool_dir, int(cfg.horizon))
    return load_pool(pool_dir, int(cfg.horizon), tuple(cfg.horizon_bands),
                     max_horizon=int(cfg.max_horizon), slots=int(cfg.band_slots))


# ── statistic ───────────────────────────────────────────────────────────────


LIVE_SOURCE = "live"


def _breakdown(x: np.ndarray, labels: list[str]) -> dict:
    """``{label: {"n", "gain"}}`` of one member's per-window log-ratios
    grouped by ``labels`` (sorted keys; gain = ``1 − exp(−mean)``)."""
    out = {}
    lab = np.asarray(labels, dtype=object)
    for key in sorted(set(labels)):
        sel = lab == key
        out[str(key)] = {"n": int(sel.sum()), "gain": log_gain(float(np.mean(x[sel])))}
    return out


def judge_cohort(king: tuple, members: list[tuple], *, clusters: list, cfg, seed: str,
                 public: dict | None = None, bar: float | None = None,
                 cells: list | None = None) -> list[dict]:
    """rev6.18 A3 (F4): ``[{"lcb", "gain", "cleared", "halves", "sources",
    "domains"}]`` per member against the bar ``king`` (module docstring,
    item 5).

    ``king`` / ``members`` are ``(crps, mase)`` per-window arrays on the
    cohort's live windows, ``clusters`` their sources. ``public`` (required
    iff ``public_weight > 0``) is ``{"anchor": (crps, mase), "members":
    [(crps, mase)], "clusters": [...], "sources": [...]}`` on the settlement's
    public windows (clusters ``<source>|<family>``). ``lcb`` / ``gain`` are
    the combined statistic's on the gain scale ``1 − exp(−L)``; ``halves``
    the live and public point gains, ``sources`` per source (``live``,
    ``gift_eval_pretrain``, ``rmisc``) and ``domains`` per cluster
    (``live|<source>``, ``<source>|<family>``), each ``{"n", "gain"}``.

    ``bar`` is the settlement's crown bar (rev6.23,
    :meth:`LaneDConfig.crown_bar`), ``None`` = ``cfg.d_margin`` (the bar's
    floor); a member clears iff ``lcb >= bar`` and every half's point gain
    is positive.

    ``cells`` (owner 2026-11-27, per-window horizons): each live window's
    ``"<frequency class>|<band>"`` label; ``public["cells"]`` the public
    windows'. Given, each result also carries ``cells``, the point gain per
    ``"<half>|<class>|<band>"`` (an optional report extra). The statistic
    itself is unchanged: the draws balance the cells to equal counts, so the
    unweighted mean already weights every cell equally."""
    bar = float(cfg.d_margin) if bar is None else float(bar)
    floor = float(cfg.window_ratio_floor)
    w = float(cfg.public_weight)
    xl = [clamped_log_ratios(*king, *m, floor=floor) for m in members]
    parts = [(xl, list(clusters), 1.0 - w)]
    xp = None
    if w > 0.0:
        if public is None or not len(public["clusters"]):
            raise ValueError("public_weight > 0 needs the public half's scores")
        xp = [clamped_log_ratios(*public["anchor"], *pm, floor=floor) for pm in public["members"]]
        parts.append((xp, list(public["clusters"]), w))
    lcbs = parts_maxt_lcbs(parts, alpha=float(cfg.alpha), B=int(cfg.bootstrap_B), seed=seed)
    live_labels = [f"{LIVE_SOURCE}|{c}" for c in clusters]
    out = []
    for j, lcb in enumerate(lcbs):
        l_live = float(np.mean(xl[j]))
        halves = {"live": {"n": int(xl[j].size), "gain": log_gain(l_live)}}
        sources = {LIVE_SOURCE: dict(halves["live"])}
        domains = _breakdown(xl[j], live_labels)
        big_l = (1.0 - w) * l_live
        ok = halves["live"]["gain"] > 0.0
        if xp is not None:
            l_pub = float(np.mean(xp[j]))
            halves["public"] = {"n": int(xp[j].size), "gain": log_gain(l_pub)}
            sources.update(_breakdown(xp[j], list(public["sources"])))
            domains.update(_breakdown(xp[j], list(public["clusters"])))
            big_l += w * l_pub
            ok = ok and halves["public"]["gain"] > 0.0
        res = {"lcb": float(lcb), "gain": log_gain(big_l),
               "cleared": bool(float(lcb) >= bar and ok),
               "halves": halves, "sources": dict(sorted(sources.items())),
               "domains": domains}
        if cells is not None:
            per = _breakdown(xl[j], [f"{LIVE_SOURCE}|{c}" for c in cells])
            if xp is not None and public.get("cells") is not None:
                per.update(_breakdown(xp[j], [f"public|{c}" for c in public["cells"]]))
            res["cells"] = dict(sorted(per.items()))
        out.append(res)
    return out


# ── settlement ──────────────────────────────────────────────────────────────


def _register(state: dict, entries: list[dict], cfg, uid_of: dict) -> list[dict]:
    """Register newly revealed entries; returns the entries voided this
    settlement as ``{"digest", "reason"}`` skips (a new entry refused at
    registration, or an open entry whose hotkey lost its UID, B8)."""
    known = state["entries"]
    voided: list[dict] = []
    taken = {state["house"]["digest"], state["king"]["digest"]}
    for e in sorted(entries, key=lambda x: (int(x["reveal_ts"]), str(x["digest"]))):
        dg = str(e["digest"])
        if dg in known:
            continue
        rec = {"hotkey": str(e["hotkey"]), "coldkey_at_ready": str(e.get("coldkey_at_ready") or ""),
               "context": int(e["context"]), "ready_block": int(e["ready_block"]),
               "reveal_block": int(e["reveal_block"]), "reveal_ts": int(e["reveal_ts"]),
               "status": "pending", "reason": "pending_floor", "used_windows": [],
               "confirm": None, "frozen_set": None}
        open_same_hotkey = any(v["hotkey"] == rec["hotkey"] and v["status"] in OPEN_STATUSES
                               for v in known.values())
        if dg in taken:
            rec.update(status="closed", reason="duplicate_digest")
        elif open_same_hotkey:
            rec.update(status="closed", reason="hotkey_has_open_entry")
        elif not rec["coldkey_at_ready"]:
            rec.update(status="closed", reason="coldkey_unresolved")
        elif rec["context"] not in tuple(cfg.contexts):
            rec.update(status="closed", reason="context_not_armed")
        elif rec["reveal_block"] - rec["ready_block"] > int(cfg.ready_ttl_blocks):
            rec.update(status="closed", reason="ready_expired")
        known[dg] = rec
        if rec["status"] == "closed":
            voided.append({"digest": dg, "reason": rec["reason"]})
    for dg, rec in sorted(known.items()):
        if rec["status"] in OPEN_STATUSES and rec["hotkey"] not in uid_of:
            rec.update(status="closed", reason="deregistered")
            voided.append({"digest": dg, "reason": "deregistered"})
    return voided


def _close_house_copies(state: dict, skipped: list[dict]) -> None:
    """W5f: an open miner entry whose digest a new house pointer now names is
    a copy of the house (``duplicate_digest``, closed and never judged), as a
    copy registered after the swap is at :func:`_register`."""
    rec = state["entries"].get(str(state["house"]["digest"]))
    if rec is not None and rec["status"] in OPEN_STATUSES:
        rec.pop("gate_hold", None)
        rec.update(status="closed", reason="duplicate_digest")
        skipped.append({"digest": str(state["house"]["digest"]), "reason": "duplicate_digest"})


def _judging_start_skip(boundary_ts, t_entry, t_king, delay_s: float) -> str | None:
    """rev6.14 (a): ``None`` when an entry revealed at ``t_entry`` may be
    drawn, frozen or judged at a boundary stamped ``boundary_ts`` (>= j =
    max(T_E, T_K) + delay), else its typed skip reason."""
    if boundary_ts is None:
        return "boundary_ts_missing"
    t = max(int(t_entry), int(t_king)) if t_king is not None else int(t_entry)
    return None if int(boundary_ts) >= t + delay_s else "before_eval_delay"


def _king_reveal_ts(state: dict, digest: str) -> int | None:
    if state["king"]["digest"] == digest:
        return int(state["king"]["reveal_ts"])
    if state["house"]["digest"] == digest:
        return int(state["house"]["reveal_ts"])
    for k in state["prior_kings"]:
        if k["digest"] == digest:
            return int(k["reveal_ts"])
    return None


# ── the §1.3b gate (W5e) ────────────────────────────────────────────────────


def _held_candidates(state: dict) -> list[str]:
    """The open entries holding a first clear the gate could not decide
    (``gate_hold``, a ``bench_hold``). They are re-gated without a new duel.
    A clear held against a KING_D that has since changed is dropped (no D20
    rule, rev6.11 (b)): the entry is judged afresh against the new king (its
    windows stay used)."""
    return [dg for dg, rec in sorted(state["entries"].items()) if _keeps_hold(rec, state)]


def _keeps_hold(rec: dict, state: dict) -> bool:
    """Is ``rec`` (a miner entry or the house entry) an open holder of a
    ``gate_hold`` that is re-gated this settlement (see
    :func:`_held_candidates`)? A hold taken under another KING_D is dropped
    here."""
    hold = rec.get("gate_hold")
    if rec["status"] not in OPEN_STATUSES or not hold:
        return False
    if hold["king_d"] != state["king"]["digest"]:
        rec.pop("gate_hold")
        rec.update(status="pending", reason="pending_floor", confirm=None, frozen_set=None)
        return False
    return True


def _ranked(cands: list[dict]) -> list[dict]:
    """Max-T order: highest LCB first, then the earlier reveal, then digest."""
    return sorted(cands, key=lambda c: (-c["lcb"], c["reveal_ts"], c["digest"]))


def _gate_result(res: dict, digest: str) -> dict:
    """A :func:`paired_gate` result projected onto one ``gate.results[]``
    item (no free text, §14.3 item 4)."""
    lcb = res.get("lcb") if res["outcome"] in ("pass", "gate_regression") else None
    return {"digest": digest, "outcome": res["outcome"],
            "lcb": None if lcb is None else {r: {m: float(lcb[r][m]) for m in ("crps", "mase")}
                                             for r in ("king_d", "house")},
            "blocked_by": [str(r) for r in res.get("blocked_by") or []]
            if res["outcome"] == "gate_regression" else [],
            "session": str(res["session"])}


BENCH_ROWS_ROLES = ("candidate", "king_d", "house")


def bench_rows_dir(state_path) -> Path:
    """W3b6b: where a settlement's bound bench records are persisted,
    ``<state dir>/<state stem>_bench_rows/<sha256>.json`` (beside
    ``<stem>_reports``, snapshotted with the state)."""
    p = Path(state_path)
    return p.parent / f"{p.stem}_bench_rows"


def _save_bench_rows(state_path, texts: dict[str, str]) -> None:
    """Write each bound record's exact bytes (atomic per file; an existing
    file is content-addressed, so it is already these bytes)."""
    if not texts:
        return
    d = bench_rows_dir(state_path)
    d.mkdir(parents=True, exist_ok=True)
    for digest, text in sorted(texts.items()):
        path = d / f"{digest}.json"
        if path.is_file():
            continue
        tmp = d / f".{digest}.tmp"
        tmp.write_bytes(text.encode("ascii"))
        os.replace(tmp, path)


def _run_gate(cands: list[dict], state: dict, cfg, boundary: int,
              block_hash: str | None, sentinel: str | None, bench,
              texts: dict[str, str] | None = None
              ) -> tuple[dict | None, dict[str, str]]:
    """``(gate record, {digest: outcome})`` for this settlement's crowning
    candidates. No candidate: ``(None, {})`` and no host call. No ``bench``
    or no ``block_hash``: the §1.3b gate cannot run, so every candidate is
    ``bench_hold`` and no record is written (fail closed). Otherwise one
    draw (:func:`gate_record`) and, per candidate in max-T order, one paired
    sweep with KING_D and the house in ``bench``'s session (the drawn suite,
    plus the sentinel suite when one is given) decided by
    :func:`paired_gate`. A host sweep that raises is that candidate's
    ``bench_hold`` (infrastructure, never the miner's fault)."""
    if not cands:
        return None, {}
    if bench is None or block_hash is None:
        return None, {c["digest"]: "bench_hold" for c in cands}
    record = gate_record(cfg, int(boundary), block_hash, sentinel=sentinel)
    suite = record["suite"]
    suites = [suite] if sentinel is None else [suite, str(sentinel)]
    king_d, house_d = str(state["king"]["digest"]), str(state["house"]["digest"])
    seed = f"lane-d/1:gate:{int(boundary)}:{record['block_hash']}"
    results, outcomes = [], {}
    for c in _ranked(cands):
        dg = c["digest"]
        bound = None
        try:
            cand_rec, king_rec, house_rec = bench.paired(dg, king_d, house_d, suites=suites)
        except Exception:  # any host failure is infrastructure: hold, never crown
            res = {"outcome": "bench_hold", "session": bench.session_id}
        else:
            res = paired_gate(cand_rec, king_rec, house_rec, suite=suite,
                              alpha=float(cfg.alpha), tau_public=float(cfg.tau_public),
                              B=int(cfg.bootstrap_B), seed=seed, sentinel=sentinel)
            # W3b6b: bind the three records the gate decided on (their
            # canonical text's sha256, inside the signed body).
            bound = {}
            for role, rec in zip(BENCH_ROWS_ROLES, (cand_rec, king_rec, house_rec),
                                 strict=True):
                text = sweep_record_text(rec)
                digest = hashlib.sha256(text.encode("ascii")).hexdigest()
                bound[role] = digest
                if texts is not None:
                    texts[digest] = text
        outcomes[dg] = str(res["outcome"])
        result = _gate_result(res, dg)
        if bound is not None:
            result["rows"] = bound
        if "held_from" in c:
            # RV7 B2: a held candidate (no verdict here) names its hold.
            result["held_from"] = dict(c["held_from"])
        results.append(result)
    return {**record, "results": results}, outcomes


def _adopted_gate(cands: list[dict], persisted: dict, where) -> tuple[dict | None, dict[str, str]]:
    """RV7 B1 recovery: ``(gate record, {digest: outcome})`` as the persisted
    (verified) report recorded them, with no bench call. A crashed settlement
    is never re-gated: a restarted validator's bench session differs and its
    bench may answer differently, so the signed record is the outcome. No
    candidate: ``(None, {})``. A report without a ``gate`` record held every
    candidate (the gate could not run). Otherwise the record's results must
    name exactly this settlement's candidates, else ``report_conflict``."""
    if not cands:
        return None, {}
    gate = persisted.get("gate")
    if gate is None:
        return None, {c["digest"]: "bench_hold" for c in cands}
    results = gate.get("results") if isinstance(gate, dict) else None
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        raise LaneDStateError("report_conflict", f"{where}: the persisted gate record has no "
                              "results list")
    outcomes = {str(r.get("digest")): str(r.get("outcome")) for r in results}
    if len(outcomes) != len(results) or set(outcomes) != {c["digest"] for c in cands}:
        raise LaneDStateError("report_conflict", f"{where}: the persisted gate results do not "
                              "name this settlement's crowning candidates")
    return json.loads(json.dumps(gate)), outcomes


def _load_pools(pool_dir, prior_pool_dirs, cfg) -> list[dict]:
    """The retained snapshots, oldest first and the newest (``pool_dir``)
    last; a directory listed twice (or equal to ``pool_dir``) loads once."""
    newest = Path(pool_dir).resolve()
    seen, dirs = {newest}, []
    for d in prior_pool_dirs or ():
        p = Path(d).resolve()
        if p not in seen:
            seen.add(p)
            dirs.append(p)
    return [load_pool_for(d, cfg) for d in dirs] + [load_pool_for(newest, cfg)]


def _pick_channels(pools: list[dict], block_hash: str | None, max_channels: int | None
                   ) -> dict:
    """rev6.18 A4 (F2): reduce every multichannel pool window to the channels
    it is scored on, in place, and return ``{window_id: ChannelPick}``.

    The pick of an id is decided by its first version in pin order (oldest
    first, as the box's eval-host request is built,
    :meth:`~cascade.validator.lane_d_inputs.LaneDInputsSource._windows`):
    all-empty channels drop, then at most ``max_channels`` (``[generator]
    max_channels``) are kept, picked by the boundary block hash for a wider
    series (:mod:`cascade.eval.lane_d_channels`). A version that cannot
    carry its id's pick, or a pick with no channel (no non-empty channel,
    an over-cap series without a block hash, or no ``max_channels`` given:
    the legacy call), leaves the pool: no model is scored on it. Univariate
    windows are untouched (the same dicts), so a univariate settlement is
    unchanged."""
    windows = [w for p in pools for w in p["windows"]]
    if max_channels is None:
        picks = channel_plan(windows, None, 1)
        picks = {wid: type(p)(p.series_id, p.n_raw, p.empty, ()) for wid, p in picks.items()}
    else:
        picks = channel_plan(windows, block_hash, int(max_channels))
    if not picks and not any(np.ndim(w["history"]) == 2 or np.ndim(w["target"]) == 2
                             for w in windows):
        return picks
    for p in pools:
        kept = []
        for w in p["windows"]:
            v = apply_pick(w, picks.get(w["window_id"]))
            if v is not None:
                kept.append(v)
        p["windows"] = kept
    return picks


def _target_usable(w: dict, horizon: int) -> bool:
    """The stored target is a finite, non-zero ``(horizon,)`` vector."""
    t = w["target"]
    return (t.shape == (horizon,) and bool(np.isfinite(t).all())
            and float(np.abs(t).sum()) > 0.0)


def _forecast_ok(fc: dict | None, wid: str, horizon: int) -> bool | None:
    """True = a usable forecast, None = absent, False = malformed (a fault)."""
    if fc is None or wid not in fc:
        return None
    q = np.asarray(fc[wid], dtype=np.float64)
    return bool(q.shape == (horizon, len(SCORING_LEVELS)) and np.isfinite(q).all())


def _fc_ok(fc: dict | None, wid: str, h: int, picks: dict) -> bool | None:
    """:func:`_forecast_ok` with the
    window's scored shape: ``(h, 9)``, or ``(k, h, 9)`` for a window scored
    on k ≥ 2 channels (F2)."""
    if wid not in picks:
        return _forecast_ok(fc, wid, h)
    if fc is None or wid not in fc:
        return None
    q = np.asarray(fc[wid], dtype=np.float64)
    return bool(q.shape == forecast_shape(picks[wid], h, len(SCORING_LEVELS))
                and np.isfinite(q).all())


def _usable(w: dict, h: int) -> bool:
    """:func:`_target_usable`, and for a
    ``(k, h)`` target (F2): finite, with at least one channel of scale."""
    t = w["target"]
    if np.ndim(t) != 2:
        return _target_usable(w, h)
    return (t.shape[1] == h and bool(np.isfinite(t).all())
            and float(np.abs(t).sum()) > 0.0)


def _pool_window_ids(pools: list[dict]) -> set[str]:
    """Every window id any loaded snapshot holds."""
    return {w["window_id"] for p in pools for w in p["windows"] if w["window_id"]}


def _admitted_versions(pools: list[dict], t_entry: int, t_king: int, cfg
                       ) -> tuple[dict[str, tuple[dict, dict, object]], dict[str, int]]:
    """RV4 one version per window: ``{window_id: (window, pool, verdict)}`` taken from
    the EARLIEST loaded snapshot (oldest first, ``pool_dir`` last) whose §1.3
    verdict admits it for ``max(t_entry, t_king)``. A judged window is scored,
    and its report row taken, from that one snapshot only: never admission
    from one snapshot with clocks, values or lag from another. A window no
    snapshot admits is absent.

    Also ``{reason: count}`` of the windows the §1.3 rule refused (a window id
    counts once, from the newest snapshot holding it, and only when no
    snapshot admits it)."""
    verdicts = [list(eligible_windows(p, t_entry=int(t_entry), t_king=int(t_king), cfg=cfg))
                for p in pools]
    admitted: dict[str, tuple[dict, dict, object]] = {}
    for pool, vs in zip(pools, verdicts, strict=True):
        for w, v in zip(pool["windows"], vs, strict=True):
            wid = w["window_id"]
            if wid and v.eligible and wid not in admitted:
                admitted[wid] = (w, pool, v)
    refused: dict[str, int] = {}
    seen: set[str] = set()
    for pool, vs in reversed(list(zip(pools, verdicts, strict=True))):
        for w, v in zip(pool["windows"], vs, strict=True):
            wid = w["window_id"]
            if wid and (wid in seen or wid in admitted):
                continue
            if v.reason is not None:
                refused[str(v.reason)] = refused.get(str(v.reason), 0) + 1
        seen |= {w["window_id"] for w in pool["windows"] if w["window_id"]}
    return admitted, dict(sorted(refused.items()))


def _cohort_set(pools: list[dict], members: list[str], bars: list[str], king_ts: int,
                recs: dict[str, dict], forecasts: dict, cfg, pinned: dict[str, str],
                picks: dict | None = None
                ) -> tuple[list[tuple[dict, dict, object]], dict[str, int]]:
    """The cohort's eligible window set as ``(window, its snapshot, its §1.3
    verdict)`` triples (the W5j draw picks the judged windows from it): the
    intersection of the members' frozen sets that is eligible for
    ``max(T over cohort ∪ king)``, each window in the version of its earliest
    admitting snapshot (:func:`_admitted_versions`), with a usable target, a
    usable forecast from every bar the cohort is judged against (KING_D, plus
    the house when a miner holds KING_D) and from every member, and no window
    a member was already judged on; plus the §1.3 refusal counts.

    ``pinned`` maps a window an earlier cohort of this settlement already
    judged to its snapshot's sha256. The report carries one row per window,
    so a window whose admitting snapshot for this cohort's bound differs is
    left out of this cohort (fail closed) rather than scored on a second
    version."""
    t_entry = max(int(recs[m]["reveal_ts"]) for m in members)
    admitted, refused = _admitted_versions(pools, t_entry, int(king_ts), cfg)
    common = set.intersection(*(set(recs[m]["frozen_set"]["window_ids"])
                                for m in members))
    out = []
    for wid in sorted(common):
        if wid not in admitted:
            continue
        w, pool, v = admitted[wid]
        if wid in pinned and pinned[wid] != pool["sha256"]:
            continue
        picks = picks or {}
        h = window_horizon(w, cfg)
        if not _usable(w, h) or any(
                _fc_ok(forecasts.get(b), wid, h, picks) is not True for b in bars):
            continue
        if any(wid in recs[m]["used_windows"]
               or _fc_ok(forecasts.get(m), wid, h, picks) is not True for m in members):
            continue
        out.append((w, pool, v))
    return out, refused


def _draw_cells(picked: list[tuple[dict, dict, object]], cfg) -> list[dict]:
    """The W5j draw input for a cohort's eligible set: one forward cell per
    pool ``freq`` (timeframe, bin ``forward``, h = ``horizon``), one draw
    series per pool ``series_id`` holding its eligible windows' target points
    in time order. A window is atomic (its id, target and forecasts exist at
    origin 0 only), so an ineligible separator point follows each window: no
    candidate straddles two windows, and every candidate is a pool window.
    ``e0`` is all-false on an embargo window (it fails the e = 0 rule)."""
    h = int(cfg.horizon)
    by_cell: dict[str, dict[str, list]] = {}
    for w, _pool, v in picked:
        by_cell.setdefault(str(w.get("freq") or ""), {}).setdefault(
            str(w["series_id"]), []).append((w, bool(v.embargo_window)))
    cells = []
    for freq in sorted(by_cell):
        members = []
        for sid in sorted(by_cell[freq]):
            ts, el, e0, src = [], [], [], None
            for w, emb in sorted(by_cell[freq][sid], key=lambda x: int(x[0]["target_ts"][0])):
                src = str(w["source"]) if src is None else src
                ts += [int(t) for t in w["target_ts"][:h]]
                el += [True] * h
                e0 += [not emb] * h
                ts.append(int(w["target_ts"][h - 1]))
                el.append(False)
                e0.append(False)
            members.append({"series_id": sid, "source": src, "ts": ts, "eligible": el,
                            "e0": e0})
        cells.append({"timeframe": freq, "bin": "forward", "kind": "forward",
                      "h_lo": h, "h_hi": h, "series": members})
    return cells


def _drawn(picked: list[tuple[dict, dict, object]], cfg, block_hash: str | None,
           budget: int) -> tuple[list[tuple[dict, dict, object]], dict | None, bool]:
    """W5j: ``(judged windows, draw result, meets floors)`` of one cohort.
    With ``block_hash`` the judged windows are the seeded per-bin draw over
    the eligible set within ``budget``; without it (the legacy call, which
    can never crown: W5e fails it closed) the whole eligible set is judged
    up to ``budget``, as before W5j."""
    if block_hash is None:
        ws = picked[:max(0, int(budget))]
        return ws, None, _meets_floors([w for w, _, _ in ws], cfg)
    if bands_on(cfg):
        d = draw_band_windows([_band_cand(w, v) for w, _, v in picked],
                              boundary_block_hash=str(block_hash), cfg=cfg, budget=int(budget))
        by_id = {w["window_id"]: (w, pool, v) for w, pool, v in picked}
        return [by_id[x["window_id"]] for x in d["windows"]], d, bool(d["meets_floors"])
    d = draw_windows(_draw_cells(picked, cfg), boundary_block_hash=str(block_hash), cfg=cfg,
                     budget=int(budget))
    by_id = {w["window_id"]: (w, pool, v) for w, pool, v in picked}
    ws = [by_id[x["window_id"]] for x in d["windows"]]
    return ws, d, bool(d["meets_floors"])


def _band_cand(w: dict, v) -> dict:
    """A band draw candidate (:func:`~cascade.validator.lane_d_draw.draw_band_windows`)."""
    return {"window_id": w["window_id"], "series_id": str(w["series_id"]),
            "source": str(w["source"]), "freq": str(w.get("freq") or ""),
            "band": str(w["band"]), "h": int(w["h"]), "t0": int(w["target_ts"][0]),
            "t1": int(w["target_ts"][-1]), "embargo_window": bool(v.embargo_window)}


def _window_embargo_fields(w: dict, v, h: int) -> dict:
    """The signed rev6.10-rev6.12 fields of one judged window (W7a): ``freq``,
    the effective ``embargo_hours`` e, ``min_release_ts`` = min first-seen +
    e, ``rule`` (``embargo`` when e > 0, else ``embargo0``), ``embargo_window``
    (rev6.12: False = fresh), and the W5j draw cell (``timeframe`` = freq,
    ``bin`` / ``kind`` forward, ``freq_class``, ``h``, ``origin`` = the target
    start)."""
    freq = str(w.get("freq") or "")
    e = float(v.embargo_hours)
    # owner 2026-11-27: a band window names its band (an optional extra; its
    # ``h`` is the band horizon of its frequency)
    band = {"band": str(w["band"])} if w.get("band") is not None else {}
    return {**band, "freq": freq, "embargo_hours": e,
            "min_release_ts": int(min(w["first_seen_ts"])) + int(round(3600.0 * e)),
            "rule": "embargo" if e > 0.0 else "embargo0",
            "embargo_window": bool(v.embargo_window),
            "timeframe": freq, "bin": "forward", "kind": "forward",
            "freq_class": freq_class(freq), "h": int(h), "origin": int(w["target_ts"][0])}


def _subscore(scores, idx: list[int]) -> dict:
    """rev6.12 (b): ``{n, geomean}`` of one digest's rows at positions ``idx``
    (the lane D geomean of its CRPS and MASE, floors 1e-12 / 1e-9); null
    geomean when n = 0."""
    if not idx:
        return {"n": 0, "geomean": None}
    crps, mase = scores
    lc = float(np.mean(np.log(np.maximum(np.asarray(crps)[idx], 1e-12))))
    lm = float(np.mean(np.log(np.maximum(np.asarray(mase)[idx], 1e-9))))
    return {"n": len(idx), "geomean": math.sqrt(math.exp(lc) * math.exp(lm))}


def _advance_member(rec: dict, pools: list[dict], king_ts: int, cfg, seed: str) -> dict:
    """Advance one entry's ``frozen_set`` against a king revealed at
    ``king_ts``. The record carries the bound it was built for
    (``t_max = max(T_E, T_K)``): a not-yet-judged set built for another bound
    (the comparison king changed) restarts, since its windows' §1.3 verdicts
    no longer hold."""
    t_max = max(int(rec["reveal_ts"]), int(king_ts))
    prog = rec.get("frozen_set")
    if prog is not None and int(prog.get("t_max", -1)) != t_max:
        prog = None
    prog = advance_frozen_set(prog, pools, t_entry=int(rec["reveal_ts"]), t_king=int(king_ts),
                              cfg=cfg, seed=seed, exclude=rec["used_windows"])
    prog["t_max"] = t_max
    return prog


def _meets_floors(ws: list[dict], cfg) -> bool:
    return (len(ws) >= int(cfg.min_windows)
            and len({w["source"] for w in ws}) >= int(cfg.min_clusters))


def _court(state: dict, cfg, uid_of: dict) -> list[dict]:
    k = state["king"]
    slots = [{"digest": k["digest"], "hotkey": None if k.get("house") else k["hotkey"]}]
    slots += [{"digest": p["digest"], "hotkey": p["hotkey"]}
              for p in state["prior_kings"][:int(cfg.reward_prior_kings)]]
    for s in slots:
        uid = uid_of.get(s["hotkey"]) if s["hotkey"] is not None else None
        s["uid"] = None if uid is None else int(uid)
        # Invariant 11: a burning slot names why. The house holding KING_D
        # burns (D21); a court member without a UID burns (B8).
        if s["uid"] is not None:
            s["burn_reason"] = None
        else:
            s["burn_reason"] = "house_king" if s["hotkey"] is None else "deregistered"
    return slots


def _king_record(k: dict) -> dict:
    return {"digest": k["digest"], "hotkey": k.get("hotkey"), "house": bool(k.get("house")),
            "reveal_ts": int(k["reveal_ts"])}


def _params(cfg) -> dict:
    return {"share": float(cfg.share), "takes_from": str(cfg.takes_from),
            "remainder": str(cfg.remainder), "horizon": int(cfg.horizon),
            "horizons": [int(cfg.horizon)],
            "contexts": [int(c) for c in cfg.contexts],
            "forward_margin_hours": float(cfg.forward_margin_hours),
            "max_mirror_lag_hours": float(cfg.max_mirror_lag_hours),
            "min_windows": int(cfg.min_windows), "min_clusters": int(cfg.min_clusters),
            "d_margin": float(cfg.d_margin), "alpha": float(cfg.alpha),
            "bootstrap_B": int(cfg.bootstrap_B),
            # rev6.23: the tenure-decaying crown bar (d_margin is its end) and
            # the explicit court weights (they replace king_decay).
            "d_margin_start": float(cfg.d_margin_start),
            "d_margin_warmup_settlements": int(cfg.d_margin_warmup_settlements),
            "court_weights": [float(w) for w in cfg.court_weights],
            "reward_prior_kings": int(cfg.reward_prior_kings),
            "max_entries_per_settlement": int(cfg.max_entries_per_settlement),
            "max_judged_per_coldkey_per_settlement":
                int(cfg.max_judged_per_coldkey_per_settlement),
            "scoring_levels": [float(q) for q in SCORING_LEVELS],
            "gate_suites": [str(s) for s in cfg.gate_suites],
            "tau_public": float(cfg.tau_public),
            # rev6.18 A3 (F4): the public half's weight and the ratio clamp.
            "public_weight": float(cfg.public_weight),
            "window_ratio_floor": float(cfg.window_ratio_floor),
            # §1.3b draw weights as parsed (a suite without a weight is absent),
            # the T18e/B14 source cap and the §1.7 context floor.
            "gate_weights": {str(s): float(w) for s, w in cfg.gate_weights},
            "max_source_share": float(cfg.max_source_share),
            "min_long_series": int(cfg.min_long_series),
            # rev6.10 (a) / rev6.12 (b) (W2e): the embargo cap, the W5j
            # settlement budget and the minimum fresh share.
            "embargo_hours": float(cfg.embargo_hours),
            "max_windows_per_settlement": int(cfg.max_windows_per_settlement),
            "min_fresh_window_share": float(cfg.min_fresh_window_share),
            # rev6.14 (a): the judging start j = T + eval_delay_hours.
            "eval_delay_hours": float(cfg.eval_delay_hours)}


def _pool_record(pools: list[dict], windows_out: dict) -> dict:
    """``pool`` of the report: the newest snapshot's ``snapshot`` /
    ``sha256``, and ``snapshots``, every loaded snapshot a judged window's
    version came from plus the newest, oldest first, each once (two loaded
    snapshots with equal bytes are one, listed as the newer).
    ``first_seen_roots`` is the sorted union of the listed snapshots' roots."""
    newest = pools[-1]
    used = {w["snapshot_sha256"] for w in windows_out.values()} | {newest["sha256"]}
    snaps, seen = [], set()
    for p in reversed(pools):
        if p["sha256"] in used and p["sha256"] not in seen:
            seen.add(p["sha256"])
            snaps.append({"snapshot": p["snapshot"], "sha256": p["sha256"],
                          "first_seen_roots": sorted(p["first_seen_roots"])})
    snaps.reverse()
    roots = sorted({r for s in snaps for r in s["first_seen_roots"]})
    return {"snapshot": newest["snapshot"], "sha256": newest["sha256"],
            "first_seen_roots": roots, "snapshots": snaps}


def _closing(cfg, boundary: int) -> bool:
    """T18: the settlement at ``until_block`` (the last active one) judges
    nothing and closes every open entry."""
    until = int(cfg.until_block)
    return until > 0 and int(boundary) == until


class LaneDPublicError(ValueError):
    """rev6.18 A3 (F4): the public half a ``public_weight > 0`` settlement
    needs is missing or does not match its draw; ``code`` is one of
    :data:`PUBLIC_ERROR_CODES`. Raised before anything is read beyond the
    state or written (infrastructure, never a miner's fault)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in PUBLIC_ERROR_CODES:
            raise ValueError(f"LaneDPublicError: unknown code {code!r}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


PUBLIC_ERROR_CODES = (
    "public_missing",      # w > 0 and no indexes, scores or block hash
    "public_sources",      # the indexes are not exactly the pinned public sources
    "public_mismatch",     # the scores / dropped list do not match the draw
    "public_empty",        # a source keeps no window after the read-time drops
)


def _public_half(cfg, public_indexes, public_scores, block_hash, horizon: int,
                 digests) -> dict | None:
    """The settlement's public half (rev6.18 A1/A3, F4), or ``None`` at
    ``public_weight = 0`` (not drawn). Draws every pinned public source from
    its index and ``block_hash`` (:func:`~cascade.validator.lane_d_public.draw_public`)
    and binds the eval host's per-window scores to it:

    ``public_scores = {"scores": {digest: {source: [{window_id, family, crps,
    mase}]}}, "dropped": {source: [{window_id, reason}]}}`` (the inputs
    source's; every digest of ``digests`` — the settlement's forecasts — has
    a score set). The kept windows are the drawn ones minus the dropped
    ones, in ``window_id`` order; a digest's scores must name exactly them
    (else ``public_mismatch``, the eval host's fault). Its values are checked
    per digest by :func:`_public_arrays` (a non-finite score is the
    checkpoint's)."""
    if float(cfg.public_weight) <= 0.0:
        return None
    if not public_indexes or not isinstance(public_scores, dict) or block_hash is None:
        raise LaneDPublicError("public_missing", "public_weight > 0 needs the public indexes, "
                               "the eval host's public scores and the boundary block hash")
    from ..eval.lane_d_public_index import PUBLIC_SOURCES
    from .lane_d_public import DROP_REASONS, balance_public, public_record

    if sorted(public_indexes) != sorted(PUBLIC_SOURCES):
        raise LaneDPublicError("public_sources", f"indexes for {sorted(public_indexes)}, not "
                               f"{sorted(PUBLIC_SOURCES)}")
    draws = public_draws(cfg, public_indexes, str(block_hash))
    dropped_in = public_scores.get("dropped") or {}
    scores_in = public_scores.get("scores")
    if not isinstance(dropped_in, dict) or not isinstance(scores_in, dict):
        raise LaneDPublicError("public_mismatch", "public scores are not {scores, dropped}")
    windows, dropped = [], {}
    for d in draws:
        src = d["source"]
        drawn = {w["window_id"]: w for w in d["windows"]}
        drops = list(dropped_in.get(src) or [])
        if any(not isinstance(x, dict) or x.get("window_id") not in drawn
               or x.get("reason") not in DROP_REASONS for x in drops) \
                or len({x["window_id"] for x in drops}) != len(drops):
            raise LaneDPublicError("public_mismatch", f"{src}: a dropped window is not a "
                                   f"drawn one with a typed reason")
        gone = {x["window_id"] for x in drops}
        dropped[src] = [{"window_id": str(x["window_id"]), "reason": str(x["reason"])}
                        for x in drops]
        kept = [drawn[k] for k in sorted(drawn) if k not in gone]
        if not kept:
            raise LaneDPublicError("public_empty", f"{src}: no public window is kept")
        windows += [{"window_id": w["window_id"], "source": src, "family": str(w["family"]),
                     "freq": str(w["freq"]), **({"band": w["band"]} if "band" in w else {})}
                    for w in kept]
    windows.sort(key=lambda w: w["window_id"])
    balanced: set[str] = set()
    cells = None
    if bands_on(cfg):
        # owner 2026-11-27: equal counts per (frequency class, band) cell; the
        # windows left out are recorded as drops (the deployed checker's kept
        # set = drawn minus dropped, so it scores exactly these windows)
        windows, gone = balance_public(windows, boundary_block_hash=str(block_hash),
                                       min_cell=int(cfg.min_cell_windows))
        if not windows:
            raise LaneDPublicError("public_empty", "no public (frequency class, band) cell "
                                   f"holds min_cell_windows={cfg.min_cell_windows}")
        balanced = {x["window_id"] for x in gone}
        src_of = {x["window_id"]: x["window_id"].split(":", 1)[0] for x in gone}
        for x in gone:
            dropped[src_of[x["window_id"]]].append(dict(x))
        for src in dropped:
            dropped[src].sort(key=lambda x: x["window_id"])
        cells = [f"{cell_class(w['freq'])}|{w['band']}" for w in windows]
    missing = sorted(dg for dg in digests if dg not in scores_in)
    if missing:
        raise LaneDPublicError("public_mismatch", f"no public scores for {missing[:3]}")
    return {"draws": draws, "dropped": dropped, "windows": windows, "scores": scores_in,
            "record": public_record, "balanced": balanced, "cells": cells,
            "_pos": {w["window_id"]: i for i, w in enumerate(windows)}}


def public_draws(cfg, public_indexes, block_hash: str) -> list[dict]:
    """Every public source's draw at ``block_hash`` under ``cfg`` (the
    inputs source and the settlement call this one function): the band rule
    when ``horizon_bands`` is set, else every window at ``horizon``."""
    from .lane_d_public import draw_public

    extra = ({"bands": tuple(cfg.horizon_bands), "max_horizon": int(cfg.max_horizon)}
             if bands_on(cfg) else {})
    return [draw_public(public_indexes[src], source=src, boundary_block_hash=str(block_hash),
                        horizon=int(cfg.horizon), **extra) for src in sorted(public_indexes)]


def _pub_ready(x) -> bool:
    """A digest's public half is usable (``True`` at ``public_weight = 0``)."""
    return x is True or isinstance(x, tuple)


def _public_arrays(pub: dict, dg: str) -> tuple[np.ndarray, np.ndarray] | None | bool:
    """One digest's ``(crps, mase)`` on the kept public windows, ``None``
    when it has no score set, ``False`` when a score is not a finite
    non-negative number (the checkpoint's fault). A score set that names
    other windows or families is the eval host's (``public_mismatch``)."""
    per = pub["scores"].get(dg)
    if per is None:
        return None
    if not isinstance(per, dict):
        raise LaneDPublicError("public_mismatch", f"{dg[:16]}: public scores are not per source")
    got = {}
    skip = pub.get("balanced") or set()
    for src, rows in per.items():
        for r in rows or []:
            if not isinstance(r, dict) or "window_id" not in r:
                raise LaneDPublicError("public_mismatch", f"{dg[:16]}: a malformed public row")
            if str(r["window_id"]) in skip:
                continue
            got[str(r["window_id"])] = (str(src), r)
    want = pub["windows"]
    if sorted(got) != [w["window_id"] for w in want]:
        raise LaneDPublicError("public_mismatch", f"{dg[:16]}: public scores do not cover "
                               f"exactly the kept public windows")
    crps, mase = [], []
    for w in want:
        src, r = got[w["window_id"]]
        if src != w["source"] or str(r.get("family")) != w["family"]:
            raise LaneDPublicError("public_mismatch", f"{dg[:16]}: {w['window_id']} names "
                                   f"another source or family")
        try:
            c, m = float(r["crps"]), float(r["mase"])
        except (KeyError, TypeError, ValueError):
            return False
        if not (math.isfinite(c) and math.isfinite(m) and c >= 0.0 and m >= 0.0):
            return False
        crps.append(c)
        mase.append(m)
    return np.asarray(crps), np.asarray(mase)


class LaneDArmedError(ValueError):
    """A settlement's armed record (rev6.6) is refused: ``code`` is one of
    :data:`ARMED_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


ARMED_CODES = ("no_armed_context", "armed_invalid")
_ARMED_KEYS = frozenset({"armed_contexts", "evidence"})
_EVIDENCE_KEYS = frozenset({"min_long_series", "min_sources", "readiness_sha256", "t7_sha256",
                            "variants"})
_VARIANT_KEYS = frozenset({"context", "series", "sources", "t7_fits", "armed"})


def _nonneg_int(x) -> bool:
    return type(x) is int and x >= 0


def check_armed(armed, cfg) -> dict:
    """The report fields ``{"armed_contexts", "armed_evidence"}`` of a valid
    W5d armed record for ``cfg``; ``LaneDArmedError`` otherwise. The record
    must be exactly what :func:`~cascade.eval.lane_d_scorer.armed_record`
    gives for ``cfg.contexts`` and ``cfg.min_long_series``: every variant's
    ``armed`` flag is recomputed from its gates and the armed set from the
    flags. Pure; never mutates ``armed``."""
    def bad(why: str):
        raise LaneDArmedError("armed_invalid", why)

    if not isinstance(armed, dict) or set(armed) != _ARMED_KEYS:
        bad(f"the record must be an object with exactly {sorted(_ARMED_KEYS)}")
    ev, stated = armed["evidence"], armed["armed_contexts"]
    if not isinstance(ev, dict) or set(ev) != _EVIDENCE_KEYS:
        bad(f"evidence must be an object with exactly {sorted(_EVIDENCE_KEYS)}")
    if ev["min_long_series"] != int(cfg.min_long_series) or type(ev["min_long_series"]) is not int:
        bad(f"min_long_series {ev['min_long_series']!r} != config {int(cfg.min_long_series)}")
    if type(ev["min_sources"]) is not int or ev["min_sources"] != GATE_MIN_SOURCES:
        bad(f"min_sources {ev['min_sources']!r} != {GATE_MIN_SOURCES}")
    for k in ("readiness_sha256", "t7_sha256"):
        if ev[k] is not None and not _is_hex64(ev[k]):
            bad(f"{k} must be 64 lower-case hex or null")
    variants = ev["variants"]
    want = [int(c) for c in sorted(cfg.contexts)]
    if not isinstance(variants, list) or len(variants) != len(want):
        bad(f"variants must cover the contexts {want}")
    flags = []
    for c, v in zip(want, variants, strict=True):
        if not isinstance(v, dict) or set(v) != _VARIANT_KEYS or v["context"] != c \
                or type(v["context"]) is not int:
            bad(f"variant for {c} malformed")
        if not (_nonneg_int(v["series"]) and _nonneg_int(v["sources"])
                and type(v["t7_fits"]) is bool and type(v["armed"]) is bool):
            bad(f"variant {c}: counts are non-negative ints, flags bools")
        gates = (v["series"] >= ev["min_long_series"] and v["sources"] >= ev["min_sources"]
                 and v["t7_fits"])
        if v["armed"] is not gates:
            bad(f"variant {c}: armed {v['armed']} disagrees with its gates")
        if gates:
            flags.append(c)
    if not isinstance(stated, list) or any(type(c) is not int for c in stated) \
            or stated != flags:
        bad(f"armed_contexts {stated!r} != the armed variants {flags}")
    if not flags:
        raise LaneDArmedError("no_armed_context", "no lane D variant is armed; nothing can be scored")
    return {"armed_contexts": list(flags),
            "armed_evidence": json.loads(json.dumps(ev, allow_nan=False))}


def settle_lane_d(*, cfg, boundary: int, pool_dir, entries: list[dict], house: dict,
                  forecasts: dict, uid_of: dict, state_path, signer,
                  prior_pool_dirs=(), expect_prior_sha256: str | None = None,
                  boundary_block_hash: str | None = None, sentinel: str | None = None,
                  bench=None, armed: dict | None = None,
                  boundary_ts: int | None = None,
                  max_channels: int | None = None,
                  public_indexes: dict | None = None,
                  public_scores: dict | None = None,
                  host: dict | None = None,
                  eval_identity: dict | None = None) -> LaneDSettlement:
    """Settle lane D at ``boundary`` (see the module docstring).

    ``boundary_ts`` (rev6.14 (a)) is the boundary block's on-chain
    ``Timestamp.Now`` (seconds), signed in the report. An entry (and the house
    challenger) is drawn, frozen or judged only when ``boundary_ts`` >= its
    judging start ``j = max(T_E, T_K) + 3600 * eval_delay_hours``, else it is
    skipped ``before_eval_delay``. Without it (the legacy call) nothing is
    frozen or judged: every open entry is skipped ``boundary_ts_missing``
    (fail closed).

    ``public_indexes`` (rev6.18 A1/A2/A6, F3) is ``{source: index dir}`` of the
    public sources (W6p indexes) and ``public_scores`` (F4) the eval host's
    per-window scores of every forecast digest on their drawn windows
    (:func:`_public_half`). At ``public_weight > 0`` both are required with
    ``boundary_block_hash`` (else :class:`LaneDPublicError`
    ``public_missing``, before anything is written): each source is drawn
    (:func:`~cascade.validator.lane_d_public.draw_public`, 5000 windows, the
    same for KING_D, the house and every challenger), every judged digest is
    scored on the kept windows (the public half of :func:`judge_cohort`) and
    the report carries the ``public`` block (revision, index and file-list
    hashes, window ids, the read-time drops, the public ``rows`` and their
    ``rows_sha256``). A member whose public scores are not finite is
    ``checkpoint_fault``. At ``public_weight = 0`` nothing is drawn or
    recorded.

    ``max_channels`` (rev6.18 A4, F2) is ``[generator] max_channels``: the
    channel cap of a multichannel ``(C, L)`` pool window, whose pick
    (empty channels dropped, a wider series cut to ``max_channels`` by
    ``boundary_block_hash``) is :mod:`cascade.eval.lane_d_channels`'. Each
    such window is ONE per-window score per model (channels averaged,
    :func:`~cascade.eval.lane_d_scorer.score_window`), one resampling unit
    in its source's cluster, and its report window carries the pick
    (``series_id``, ``n_channels_raw``, ``empty_channels``, ``n_channels``,
    ``channels_sha256``). ``None`` (the legacy call) scores no
    multichannel window. Univariate windows are unaffected.

    ``pool_dir`` is the newest ``pool_d`` snapshot; ``prior_pool_dirs`` are
    the earlier retained ones. They feed every open entry's union and supply
    the history and targets of windows frozen from them.

    The §1.3b gate inputs (W5e): ``boundary_block_hash`` is the settlement
    boundary's block hash (``0x`` + 64 hex, the draw's input), ``sentinel``
    the era's §16 sentinel suite (``None`` = W-S1 unarmed) and ``bench`` the
    bench host session (a :class:`~cascade.validator.lane_d_bench.SweepSession`).
    Only a settlement with a crowning candidate uses them; without ``bench``
    or ``boundary_block_hash`` every crowning candidate is ``bench_hold``
    (fail closed: nothing crowns ungated).

    ``expect_prior_sha256`` (the last published receipt's
    ``posterior_state_sha256``), when given, must equal the digest of the
    state as loaded (``state_digest_mismatch``). A lost state after a
    persisted settlement is ``state_missing`` and a rolled-back one is
    ``state_digest_mismatch`` (:func:`~cascade.validator.lane_d_state.load_prior_state`);
    every such refusal comes before any pool or forecast is read and writes
    nothing.

    ``armed`` (rev6.6) is the settlement's W5d
    :func:`~cascade.eval.lane_d_scorer.armed_record`. Given, it is checked by
    :func:`check_armed` (a ``LaneDArmedError`` before anything is read or
    written) and recorded as the report's ``armed_contexts`` /
    ``armed_evidence``; the forecasts are the eval host's at each entry's
    ``effective_context``. ``None`` (the legacy call) records neither.

    ``host`` (rev6.18 B7) is the eval host record the W4 handshake verified
    (:func:`~cascade.eval.lane_d_handshake.check_worker`: environment, GPU
    model, driver, torch build, fingerprint and the B6 host identity). Every
    forecast in ``forecasts`` came from that one host (B6). Given, it must be
    well formed (a ``ValueError`` before anything is read or written) and is
    signed as the report's ``host``; ``None`` records no host.

    ``eval_identity`` (W-IMG, plan rev6.24 (e)): the eval host's code
    identity the forecasts ran on, recorded under ``extensions`` as
    :data:`EVAL_IMAGE_EXTENSION` (logged, never consensus). ``None`` (every
    settlement before ``[v2] image_epoch_from_block``) records nothing."""
    if not cfg.active(boundary):
        return LaneDSettlement(report=None, pay=None, receipt=None)
    if boundary_ts is not None and (isinstance(boundary_ts, bool) or not isinstance(boundary_ts, int)
                                    or boundary_ts < 0):
        raise ValueError(f"boundary_ts must be a non-negative int timestamp; got {boundary_ts!r}")
    armed_fields = None if armed is None else check_armed(armed, cfg)
    if host is not None:
        from ..shared.config.lane_d import host_record_problem

        why = host_record_problem(host)
        if why:
            raise ValueError(f"lane D host record refused: {why}")
        host = {k: str(v) for k, v in host.items()}
    # W5f: the house is the verified record a signed pointer enters
    # (lane_d_house). Anything else is refused before any state, report or
    # pool is read or written (fail closed).
    house = check_house(house)
    boundary = int(boundary)
    state = load_prior_state(state_path, house, boundary,
                             expect_prior_sha256=expect_prior_sha256)
    settled = int(state["boundary"])
    rp = report_path(state_path, boundary)
    if settled == boundary and "last_report" in state:
        return _already_settled(state, rp, boundary)
    if settled >= boundary:
        raise LaneDStateError("stale_boundary", f"lane D already settled through {settled}; "
                              f"got {boundary}")
    # rev6.18 A3 (F4): the public half, drawn and bound to the eval host's
    # scores before anything else is read or written (fail closed).
    pub = _public_half(cfg, public_indexes, public_scores, boundary_block_hash,
                       int(cfg.horizon), sorted(forecasts))
    pub_arr: dict[str, object] = {}

    def pub_of(dg: str):
        if pub is None:
            return True
        if dg not in pub_arr:
            pub_arr[dg] = _public_arrays(pub, dg)
        return pub_arr[dg]

    # A report file without its state: a crash between the two writes. RV7
    # B1: recovery ADOPTS it, never re-sweeps the bench. It is parsed
    # strictly and checked (the settling validator's signature, the boundary,
    # the prior state digest bound in its signed body) before anything else
    # is read; a refusal calls no bench and writes nothing.
    persisted = load_report_file(rp) if rp.exists() else None
    prior_sha = state_digest(state)
    if persisted is not None:
        check_persisted_report(persisted,
                               signer_ss58=str(getattr(signer, "hotkey", signer).ss58_address),
                               boundary=boundary, prior_state_sha256=prior_sha, where=rp)
    # W7c (rev6.11 (b); orchestrator decision 2026-11-23): a testnet state
    # written by the old code may hold `confirm_pending` entries. Migrated
    # after the prior digest is taken, so the receipt binds legacy -> migrated.
    migrated = migrate_confirm_pending(state)
    refresh_house(state, house)
    king_before = _king_record(state["king"])
    # rev6.23 (owner 2026-11-24): the crown bar decays with KING_D's tenure,
    # the lane D settlements it has held the crown before this one (a house
    # KING_D counts; a house refresh in place does not reset it). Absent in
    # a state written before rev6.23: 0.
    tenure = int(state.get(KING_TENURE_KEY, 0))
    crown_bar = cfg.crown_bar(tenure)
    pools = _load_pools(pool_dir, prior_pool_dirs, cfg)
    # rev6.18 A4 (F2): multichannel windows reduced to their scored channels
    # (a no-op on a univariate pool).
    picks = _pick_channels(pools, boundary_block_hash, max_channels)
    pool = pools[-1]
    pool_ids = _pool_window_ids(pools)
    # owner 2026-11-27: each pool window's own horizon (the fixed one off bands)
    pool_h = {w["window_id"]: window_horizon(w, cfg) for p in pools for w in p["windows"]
              if w["window_id"]}
    seed = f"lane-d/1:{boundary}:{pool['sha256']}"
    # T18e: the max_source_share drop is seeded by the settlement alone.
    cap_seed = f"lane-d/1:cap:{boundary}"
    # Every entry considered and not judged is a typed skip, including the
    # entries voided this settlement (invariant 11).
    skipped = _register(state, entries, cfg, uid_of)
    h = int(cfg.horizon)
    # W5f: the house entry (§1.4 "It enters at that settlement"), opened by
    # ``refresh_house`` when a new house meets a miner KING_D. It is judged
    # like an entry but lives outside ``entries`` (it has no hotkey and is
    # never paid); each judging job carries the records it judges.
    # DEC-CA-0076 (owner 2026-11-24): after the first miner crown the house
    # never challenges; an entry left by older code is dropped.
    if not state["king"].get("house"):
        state.pop(HOUSE_ENTRY_KEY, None)
    house_rec = state.get(HOUSE_ENTRY_KEY)
    hd = str(state["house"]["digest"])
    _close_house_copies(state, skipped)

    def skip(dg: str, reason: str, rec: dict | None = None) -> None:
        (state["entries"][dg] if rec is None else rec)["reason"] = reason
        if rec is not None and rec is house_rec:
            # The house entry is not a listed entry: its reason is kept in
            # state, never a typed ``skipped`` record (which names entries).
            return
        skipped.append({"digest": dg, "reason": reason})

    held: list[str] = []
    house_held = False
    house_open = house_rec is not None and house_rec["status"] in OPEN_STATUSES
    if _closing(cfg, boundary):
        for dg, rec in sorted(state["entries"].items()):
            if rec["status"] in OPEN_STATUSES:
                rec["status"] = "closed"
                skip(dg, "lane_closed")
        if house_open:
            house_rec["status"] = "closed"
            skip(hd, "lane_closed", house_rec)
        chosen = []
        house_open = False
    else:
        # A held gate candidate is re-gated, never re-duelled (W5e).
        held = _held_candidates(state)
        # W5h (lane_d_queue): the reveal-delay close, the coldkey re-check
        # at reveal, then FIFO by T under the caps.
        chosen, queued = select_queue(state, cfg, boundary, entries, frozenset(held))
        skipped += queued
        house_held = house_open and _keeps_hold(house_rec, state)

    # A member whose forecasts are malformed on any pool window is the miner's fault
    # (the house's own: its entry closes, fail closed).
    for dg in list(chosen):
        fc = forecasts.get(dg)
        if (fc is not None and any(_fc_ok(fc, wid, pool_h[wid], picks) is False
                                   for wid in sorted(pool_ids))) or pub_of(dg) is False:
            state["entries"][dg]["status"] = "closed"
            skip(dg, "checkpoint_fault")
            chosen.remove(dg)
    # rev6.15 (W5d / W7a): an entry is scored at the highest armed variant
    # <= its declared context, never above it; with none armed at or below
    # it, it is not judged this settlement (`pending` / `context_below_armed`).
    # Only on a settlement handed the armed record; the house is exempt.
    if armed_fields is not None:
        armed_set = [int(c) for c in armed_fields["armed_contexts"]]
        for dg in list(chosen):
            rec = state["entries"][dg]
            if effective_context(int(rec["context"]), armed_set) is None:
                skip(dg, "context_below_armed")
                chosen.remove(dg)
    # rev6.14 (a): the judging start j = max(T_E, T_K) + eval_delay_hours,
    # against the signed boundary timestamp; fail closed without it.
    delay_s = 3600.0 * float(cfg.eval_delay_hours)
    king_t = _king_reveal_ts(state, str(state["king"]["digest"]))
    for dg in list(chosen):
        rec = state["entries"][dg]
        why = _judging_start_skip(boundary_ts, rec["reveal_ts"], king_t, delay_s)
        if why is not None:
            skip(dg, why)
            chosen.remove(dg)
    if house_open and not house_held:
        why = _judging_start_skip(boundary_ts, house_rec["reveal_ts"], king_t, delay_s)
        if why is not None:
            skip(hd, why, house_rec)
            house_open = False
    if house_open and not house_held:
        fc = forecasts.get(hd)
        if (fc is not None and any(_fc_ok(fc, wid, pool_h[wid], picks) is False
                                   for wid in sorted(pool_ids))) or pub_of(hd) is False:
            house_rec["status"] = "closed"
            skip(hd, "checkpoint_fault", house_rec)
            house_open = False

    # rev6.11 (b): every member is judged at its first (and only) verdict
    # against the current KING_D: one miner cohort, then the house challenger
    # in its own cohort against KING_D (§2 item 2): the house is never its own
    # second bar.
    kd0 = str(state["king"]["digest"])
    jobs = []
    if chosen:
        jobs.append((kd0, "first", list(chosen),
                     {m: state["entries"][m] for m in chosen}, False))
    if house_open and not house_held and not state["king"].get("house"):
        jobs.append((kd0, "first", [hd], {hd: house_rec}, True))

    # RV4: each judged window is scored from ONE snapshot's version; the
    # report names that snapshot (``windows[].snapshot_sha256``) and lists
    # every such snapshot in ``pool.snapshots``. ``pinned`` holds the
    # snapshot each already-judged window was scored from.
    pinned: dict[str, str] = {}
    rows, windows_out, cohorts, verdicts = [], {}, [], []
    draws: list[dict] = []
    # The crowning candidates: first verdicts that clear (rev6.11 (b)).
    candidates: list[dict] = []
    # rev6.18 A3 (F4): every digest scored on the public half (report rows).
    pub_scored: set[str] = set()
    for kd, stage, members, recs, is_house in jobs:
        king_ts = _king_reveal_ts(state, kd)
        # DEC-CA-0076 (owner 2026-11-24): KING_D is the only bar. At genesis
        # it is the house; after the first miner crown the house is no
        # second bar (W5c change 3 retired).
        bars = [kd]
        if king_ts is None or any(forecasts.get(b) is None or not _pub_ready(pub_of(b))
                                  for b in bars):
            for m in members:
                skip(m, "king_unscored", recs[m])
            continue
        # A member without a public score set (never asked of the eval host)
        # cannot be judged on the combined rule: it waits (fail closed).
        for m in [m for m in members if not _pub_ready(pub_of(m))]:
            skip(m, "king_unscored", recs[m])
            members.remove(m)
        if not members:
            continue
        members = sorted(members, key=lambda m: (recs[m]["reveal_ts"], m))
        frozen = []
        for m in members:
            rec = recs[m]
            rec["frozen_set"] = _advance_member(rec, pools, int(king_ts), cfg, cap_seed)
            if rec["frozen_set"]["frozen"]:
                frozen.append(m)
            else:
                skip(m, "pending_floor", rec)
        members = frozen
        if not members:
            continue
        # W5j: the judged windows are the cohort's seeded per-bin draw over
        # its eligible set, within what is left of the settlement's
        # max_windows_per_settlement; a draw below the floors (or short of
        # the minimum fresh share, or out of budget) leaves the latest-T
        # member pending.
        budget = int(cfg.max_windows_per_settlement) - len(windows_out)
        eligible, refused = _cohort_set(pools, members, bars, king_ts, recs, forecasts, cfg,
                                        pinned, picks)
        picked, drawn, ok = _drawn(eligible, cfg, boundary_block_hash, budget)
        while members and not ok:
            m = members.pop()
            skip(m, "pending_floor", recs[m])
            if members:
                eligible, refused = _cohort_set(pools, members, bars, king_ts, recs,
                                                forecasts, cfg, pinned, picks)
                picked, drawn, ok = _drawn(eligible, cfg, boundary_block_hash, budget)
        if not members:
            continue
        ws = [w for w, _, _ in picked]
        scores = {}
        scored = [*bars, *members]
        for dg in scored:
            pairs = [score_window(forecasts[dg][w["window_id"]], w["target"], w["history"],
                                  w["seasonal_period"]) for w in ws]
            scores[dg] = (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
        for i, (w, src_pool, v) in enumerate(picked):
            lag = (src_pool["feeds"].get(w["feed"]) or {}).get("mirror_lag_hours")
            pinned[w["window_id"]] = str(src_pool["sha256"])
            windows_out[w["window_id"]] = {
                "window_id": w["window_id"], "source": w["source"], "feed": w["feed"],
                "min_target_ts": int(min(w["target_ts"])),
                "min_first_seen_ts": int(min(w["first_seen_ts"])),
                "mirror_lag_hours": None if lag is None else float(lag),
                # The snapshot whose version of the window was scored.
                "snapshot_sha256": str(src_pool["sha256"]),
                # rev6.10 (a) / rev6.12 (W7a): the effective embargo, the
                # release time and the rule it was admitted under, the
                # fresh / embargo class, and the W5j draw cell.
                **_window_embargo_fields(w, v, window_horizon(w, cfg))}
            if w["window_id"] in picks:
                # rev6.18 A4 (F2, W7a): a multichannel window's channel
                # pick (absent on a univariate window).
                windows_out[w["window_id"]].update(channel_record(picks[w["window_id"]]))
            for dg in scored:
                rows.append({"window_id": w["window_id"], "digest": dg, "cohort": len(cohorts),
                             "crps": float(scores[dg][0][i]), "mase": float(scores[dg][1][i])})
        clusters = [w["source"] for w in ws]
        pub_m = None if pub is None else [pub_of(m) for m in members]
        # owner 2026-11-27: the (frequency class, band) cell of each window
        live_cells = ([f"{cell_class(w.get('freq'))}|{w['band']}" for w in ws]
                      if bands_on(cfg) else None)

        def pub_in(bar: str, pub_m=pub_m) -> dict | None:
            if pub is None:
                return None
            return {"anchor": pub_of(bar), "members": pub_m,
                    "clusters": [f"{w['source']}|{w['family']}" for w in pub["windows"]],
                    "sources": [w["source"] for w in pub["windows"]],
                    "cells": pub.get("cells")}

        if pub is not None:
            pub_scored.update(scored)
        res = judge_cohort(scores[kd], [scores[m] for m in members],
                           clusters=clusters, cfg=cfg, seed=seed, public=pub_in(kd),
                           bar=crown_bar, cells=live_cells)
        if len(bars) > 1:
            # The same max-T judgment with the house as the bar: a member
            # clears only if it clears both. ``judge_cohort`` needs a
            # positive gain, so an exact tie with the house never clears
            # (the house ranks first on ties). RV5: the signed verdict keeps
            # the KING_D ``lcb`` / ``gain`` and records the house judgement
            # as ``house``, so a satellite re-derives both from the rows.
            vs_house = judge_cohort(scores[hd], [scores[m] for m in members],
                                    clusters=clusters, cfg=cfg, seed=seed, public=pub_in(hd),
                                    bar=crown_bar, cells=live_cells)
            res = [{**r, "cleared": bool(r["cleared"] and rh["cleared"]),
                    "house": {"digest": hd, "lcb": rh["lcb"], "gain": rh["gain"],
                              "cleared": bool(rh["cleared"])}}
                   for r, rh in zip(res, vs_house, strict=True)]
        set_sha = hashlib.sha256("\n".join(w["window_id"] for w in ws).encode()).hexdigest()
        t_max = max([king_ts] + [recs[m]["reveal_ts"] for m in members])
        cohort = {"index": len(cohorts), "king_digest": kd, "king_reveal_ts": int(king_ts),
                  "stage": stage, "members": list(members), "t_max": int(t_max),
                  "window_ids": [w["window_id"] for w in ws], "set_sha256": set_sha,
                  "first_seen_roots": [], "refused": refused,
                  # rev6.12 (b): the fresh / embargo split of the judged set.
                  "fresh_windows": sum(1 for _, _, v in picked if not v.embargo_window),
                  "embargo_windows": sum(1 for _, _, v in picked if v.embargo_window)}
        if len(bars) > 1:
            # Present only with a house bar (absent, not null, when the
            # house holds KING_D: one bar, today's shape).
            cohort["house_digest"] = hd
        if drawn is not None and "cells" in drawn:
            # owner 2026-11-27 (optional extras): the judged windows per
            # (frequency class, band) cell, equal by construction, and the
            # cells left out below min_cell_windows.
            cohort["cells"] = dict(drawn["cells"])
            cohort["cells_dropped"] = dict(drawn["cells_dropped"])
        cohorts.append(cohort)
        if drawn is not None:
            # quota: the windows the draw kept per "<timeframe>|<bin>" cell
            # (after the source cap and the fresh share), so a satellite
            # binds each cell's count to the judged set.
            capacity = dict(drawn["capacity"])
            if "cells" in drawn:
                # band draw: the signed per-"<timeframe>|<bin>" record stays the
                # pre-band shape (every band of a freq is its "forward" bin), so
                # the deployed checker binds it unchanged; the per-cell counts
                # are the cohort's ``cells``.
                capacity = {}
                for w, _, _ in eligible:
                    k = f"{str(w.get('freq') or '')}|forward"
                    capacity[k] = capacity.get(k, 0) + 1
            kept: dict[str, int] = {k: 0 for k in capacity}
            for w, _, _ in picked:
                k = f"{str(w.get('freq') or '')}|forward"
                kept[k] = kept.get(k, 0) + 1
            draws.append({"index": cohort["index"], "quota": kept,
                          "capacity": capacity,
                          # the candidate set the draw ran over
                          "candidates_sha256": hashlib.sha256("\n".join(sorted(
                              w["window_id"] for w, _, _ in eligible)).encode()).hexdigest()})
        fresh_idx = [i for i, (_, _, v) in enumerate(picked) if not v.embargo_window]
        emb_idx = [i for i, (_, _, v) in enumerate(picked) if v.embargo_window]
        for m, r in zip(members, res, strict=True):
            rec = recs[m]
            rec["used_windows"] = sorted(set(rec["used_windows"]) | {w["window_id"] for w in ws})
            if not r["cleared"]:
                outcome = "lost"
                rec.update(status="lost", reason="lost_duel")
            else:
                # A crowning candidate: crowned only through the §1.3b gate
                # (``crowned`` below), else it stays ``cleared`` (held, or a
                # passing non-best). rev6.18 A7: no A10 check.
                outcome = "cleared"
                candidates.append({"digest": m, "lcb": float(r["lcb"]),
                                   "reveal_ts": int(rec["reveal_ts"]), "set_sha256": set_sha,
                                   "house": is_house})
            verdicts.append({"digest": m, "king_digest": kd, "stage": stage,
                             "cohort": len(cohorts) - 1, **r, "outcome": outcome,
                             # rev6.12 (b): the fresh and embargo sub-scores.
                             "fresh": _subscore(scores[m], fresh_idx),
                             "embargo": _subscore(scores[m], emb_idx)})

    # Held candidates (an earlier bench_hold) are re-gated with their first
    # clear, without a duel: no rows, no verdict this settlement.
    held_recs = [(dg, state["entries"][dg], False) for dg in held]
    if house_held:
        held_recs.append((hd, house_rec, True))
    for dg, rec, is_house in held_recs:
        hold = rec["gate_hold"]
        candidates.append({"digest": dg, "lcb": float(hold["lcb"]),
                           "reveal_ts": int(rec["reveal_ts"]),
                           "set_sha256": str(hold["set_sha256"]), "held": True,
                           "house": is_house,
                           # RV7 B2: the signed provenance of the hold (its
                           # first record; report_sha256 is that settlement's
                           # report body digest).
                           "held_from": {"boundary": int(hold["boundary"]),
                                         "report_sha256": str(hold["report_sha256"]),
                                         "set_sha256": str(hold["set_sha256"]),
                                         "lcb": float(hold["lcb"]),
                                         "king_d": str(hold["king_d"])}})
    bench_texts: dict[str, str] = {}
    if persisted is None:
        gate, gate_outcomes = _run_gate(candidates, state, cfg, boundary, boundary_block_hash,
                                        sentinel, bench, bench_texts)
    else:
        # Recovery: the outcomes the persisted, verified report recorded.
        gate, gate_outcomes = _adopted_gate(candidates, persisted, rp)
    by_verdict = {v["digest"]: v for v in verdicts}
    passing = []
    for c in candidates:
        dg, outcome = c["digest"], gate_outcomes[c["digest"]]
        rec = house_rec if c["house"] else state["entries"][dg]
        if outcome == "pass":
            rec.pop("gate_hold", None)
            passing.append(c)
            continue
        if outcome == "bench_hold":
            # Infrastructure, never the miner's fault: never crowned, lost or
            # charged. W5g (§10.4): the consecutive holds of one held clear
            # are counted in its ``gate_hold`` record (state, restart-safe).
            # Holds 1..max_bench_holds keep the clear (an existing hold keeps
            # its first record) with status ``hold``; the next one re-queues
            # the entry unjudged: ``pending`` / ``bench_hold``, the clear
            # dropped, judged afresh on the next fresh windows.
            prev = rec.get("gate_hold")
            holds = (int(prev.get("holds", 1)) if prev else 0) + 1
            if holds > int(cfg.max_bench_holds):
                rec.pop("gate_hold", None)
                rec.update(status="pending", confirm=None, frozen_set=None)
            else:
                rec["gate_hold"] = {**(prev or {
                    "boundary": boundary, "king_d": str(state["king"]["digest"]),
                    "lcb": c["lcb"], "set_sha256": c["set_sha256"]}), "holds": holds}
                rec["status"] = "hold"
            reason = "bench_hold"
        else:
            rec.pop("gate_hold", None)
            # gate_regression is a judged loss; checkpoint_fault the miner's fault.
            rec["status"] = "lost" if outcome == "gate_regression" else "closed"
            reason = outcome
            if dg in by_verdict:
                by_verdict[dg]["outcome"] = "lost"
        if c.get("held") and reason in SKIP_REASONS:
            # Considered and not judged this settlement: a typed skip
            # (gate_regression is a judged loss, an entry reason only).
            skip(dg, reason, rec)
        else:
            rec["reason"] = reason

    new_king = None
    if passing:
        best = _ranked(passing)[0]
        old = state["king"]
        if not old.get("house"):
            state["prior_kings"] = [_king_record(old)] + state["prior_kings"]
            state["prior_kings"] = state["prior_kings"][:max(0, int(cfg.reward_prior_kings))]
        if best["house"]:
            # §2 item 2 / T10: the house retakes KING_D. The dethroned miner
            # king is the newest prior king; the slot burns (D21). The house
            # entry ends: the house holds KING_D and is never in the court.
            # ``crowned`` names a miner crowning only, so it stays null.
            state["king"] = {"digest": hd, "hotkey": None, "house": True,
                             "reveal_ts": int(state["house"]["reveal_ts"])}
            state.pop(HOUSE_ENTRY_KEY, None)
            house_rec = None
        else:
            new_king = best["digest"]
            rec = state["entries"][new_king]
            rec.update(status="crowned", reason="crowned")
            state["king"] = {"digest": new_king, "hotkey": rec["hotkey"], "house": False,
                             "reveal_ts": rec["reveal_ts"], "since_boundary": boundary}
        for v in verdicts:
            if v["digest"] == best["digest"]:
                v["outcome"] = "crowned"
    # rev6.23: a crown (a miner, or the house retaking) restarts the tenure;
    # the next settlement is the new KING_D's first after its crowning.
    state[KING_TENURE_KEY] = 1 if passing else tenure + 1
    court = _court(state, cfg, uid_of)
    pay = {"share": float(cfg.share), "court_uids": [s["uid"] for s in court]}
    pool_rec = _pool_record(pools, windows_out)
    # The first-seen roots the frozen sets rely on: the union over the listed
    # snapshots, known once every cohort is judged, so each cohort takes it
    # here and the report's two root lists agree.
    for c in cohorts:
        c["first_seen_roots"] = list(pool_rec["first_seen_roots"])
    report = {
        "schema": REPORT_SCHEMA, "boundary": boundary, "scoring_method": SCORING_METHOD,
        "params": _params(cfg), "seed": seed,
        "pool": {"snapshot": pool_rec["snapshot"], "sha256": pool_rec["sha256"], "horizon": h,
                 "first_seen_roots": pool_rec["first_seen_roots"],
                 "snapshots": pool_rec["snapshots"]},
        "house": _king_record({**state["house"], "hotkey": None, "house": True}),
        "king_before": king_before, "king_after": _king_record(state["king"]),
        "crowned": new_king,
        "prior_kings": [_king_record(p) for p in state["prior_kings"]],
        "entries": [{"digest": dg, "hotkey": r["hotkey"], "coldkey": r["coldkey_at_ready"] or None,
                     **{k: r[k] for k in ("context", "ready_block", "reveal_block", "reveal_ts",
                                          "status", "reason")}}
                    for dg, r in sorted(state["entries"].items())],
        "judged": [v["digest"] for v in verdicts],
        "skipped": skipped, "cohorts": cohorts,
        "windows": [windows_out[k] for k in sorted(windows_out)],
        "verdicts": verdicts, "rows": rows, "rows_sha256": rows_sha256(rows),
        "pay": {"share": pay["share"], "court": court},
        # rev6.23: the tenure the bar was taken at and the bar applied.
        "crown_bar": {"tenure": tenure, "margin": crown_bar},
        # RV7 B1: the state this settlement ran from, under the signature, so
        # recovery can check a persisted report against the state as loaded.
        "extensions": {PRIOR_STATE_EXTENSION: prior_sha,
                       # W7c: the one settlement that migrated a legacy
                       # confirm_pending state (never consensus).
                       **({CONFIRM_MIGRATION_EXTENSION: {"tag": CONFIRM_MIGRATION_TAG,
                                                         "entries": migrated}}
                          if migrated else {}),
                       # W-IMG: the eval image the forecasts ran on (logged)
                       **({EVAL_IMAGE_EXTENSION: {k: str(v) for k, v in
                                                  sorted(dict(eval_identity).items())}}
                          if eval_identity else {})},
    }
    if bands_on(cfg):
        # owner 2026-11-27: the per-window horizon rule, signed (an optional
        # extra: params, and so the deployed checker's config echo, unchanged)
        from ..shared.lane_d_horizons import horizons_record

        report["horizon_bands"] = horizons_record(cfg)
    if draws:
        # W5j: the settlement's draw record, present iff a cohort was drawn
        # (absent, never null, otherwise; the legacy call has no block hash).
        report["draw"] = {"boundary_block_hash": str(boundary_block_hash),
                          "seed": f"{boundary_block_hash}:lane_d_windows",
                          "max_windows_per_settlement": int(cfg.max_windows_per_settlement),
                          "cohorts": draws}
    if gate is not None:
        # Present iff the gate ran (absent, never null, otherwise).
        report["gate"] = gate
    if pub is not None:
        # rev6.18 F3 (W5k) / F4: one public draw per source, the read-time
        # drops, and every judged digest's public rows (A3), bound by their
        # own sha256 (the same canonical form as ``rows_sha256``).
        prows = []
        for w in pub["windows"]:
            for dg in sorted(pub_scored):
                c, m_ = pub_of(dg)
                i = pub["_pos"][w["window_id"]]
                prows.append({"window_id": w["window_id"], "digest": dg,
                              "crps": float(c[i]), "mase": float(m_[i])})
        report["public"] = {**pub["record"](pub["draws"], pub["dropped"]),
                            "rows": prows, "rows_sha256": rows_sha256(prows)}

    if boundary_ts is not None:
        # rev6.14 (a): the boundary block's on-chain timestamp the judging
        # start was applied against (absent, never null, on the legacy call).
        report["boundary_ts"] = int(boundary_ts)
    if armed_fields is not None:
        # rev6.6: the settlement's armed set and the evidence it was armed
        # on, signed, so satellites verify it (never re-derive it from
        # cfg.contexts). Absent on the legacy call.
        report.update(armed_fields)
    if host is not None:
        # rev6.18 B7: the scoring host, signed; the audit checks its env is
        # qualified at this boundary.
        report["host"] = host
    report = sign_report(report, signer)
    # Fail closed: a report outside lane-d-report/1 is never returned, and
    # neither the report file nor the state is written (ReportSchemaError
    # propagates). Checked on the signed form, since ``signer`` and
    # ``signature`` are required fields.
    validate_report(report)
    body_sha = report_digest(report)
    if persisted is None:
        # W3b6b: the bound records' bytes land before the report that binds them.
        _save_bench_rows(state_path, bench_texts)
        save_report_file(rp, report)
    elif report_digest(persisted) != body_sha:
        # The posterior derived from the persisted outcomes must be the one
        # that report records; anything else is refused, never overwritten.
        raise LaneDStateError("report_conflict", f"{rp}: the persisted report's body is not "
                              f"the settlement derived from its outcomes (sha256 {body_sha})")
    else:
        # Recovery: the persisted report (its signature too) is the record.
        report = persisted
    # RV7 B2: a hold first taken at this settlement records the report that
    # holds it (its body digest is known only now), so a later re-gate can
    # sign its provenance (``gate.results[].held_from``).
    for rec in [*state["entries"].values(), *([house_rec] if house_rec is not None else [])]:
        hold = rec.get("gate_hold")
        if hold and hold["boundary"] == boundary and "report_sha256" not in hold:
            hold["report_sha256"] = body_sha
    state["boundary"] = boundary
    state["last_report"] = {"boundary": boundary, "rows_sha256": report["rows_sha256"],
                            "report_sha256": body_sha, "prior_state_sha256": prior_sha}
    save_state(state_path, state)
    receipt = make_receipt(boundary=boundary, prior_state_sha256=prior_sha,
                           posterior_state_sha256=state_digest(state), report_sha256=body_sha)
    return LaneDSettlement(report=report, pay=pay, receipt=receipt)


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _persisted_pay(report: dict, where) -> dict:
    """The pay record ``{"share", "court_uids"}`` a persisted report carries
    (``report["pay"]``); a malformed one is ``report_conflict``."""
    pay = report.get("pay")
    court = pay.get("court") if isinstance(pay, dict) else None
    share = pay.get("share") if isinstance(pay, dict) else None
    if (not isinstance(court, list) or isinstance(share, bool)
            or not isinstance(share, int | float)
            or not all(isinstance(s, dict) and "uid" in s for s in court)):
        raise LaneDStateError("report_conflict", f"{where}: the persisted report has no "
                              "well-formed pay record")
    uids = [s["uid"] for s in court]
    if not all(u is None or (isinstance(u, int) and not isinstance(u, bool)) for u in uids):
        raise LaneDStateError("report_conflict", f"{where}: a court uid is not an int or null")
    return {"share": float(share), "court_uids": uids}


def _already_settled(state: dict, rp: Path, boundary: int) -> LaneDSettlement:
    """T11 re-call at a settled boundary: the persisted report, its pay and
    the receipt, with nothing re-scored, re-gated or written."""
    last = state.get("last_report")
    if (not isinstance(last, dict) or last.get("boundary") != boundary
            or not _is_hex64(last.get("report_sha256"))
            or not _is_hex64(last.get("prior_state_sha256"))):
        raise LaneDStateError("report_conflict", f"lane D state at {boundary} has no "
                              "well-formed last_report")
    if not rp.is_file():
        raise LaneDStateError("report_conflict", f"{rp}: the settled boundary's report file "
                              "is missing")
    report = load_report_file(rp)
    if report.get("boundary") != boundary or report_digest(report) != last["report_sha256"]:
        raise LaneDStateError("report_conflict", f"{rp}: the persisted report is not the one "
                              f"the state bound (report_sha256 {last['report_sha256']})")
    receipt = make_receipt(boundary=boundary, prior_state_sha256=last["prior_state_sha256"],
                           posterior_state_sha256=state_digest(state),
                           report_sha256=last["report_sha256"])
    return LaneDSettlement(report=report, pay=_persisted_pay(report, rp), receipt=receipt)


def absent_settlement(*, cfg, boundary: int, state_path, house: dict, reason: str,
                      expect_prior_sha256: str | None = None) -> LaneDSettlement:
    """W14: settle lane D at ``boundary`` with NO lane D report (the operator
    stopped reports, ``reason="report_absent"``, or the settlement could not
    run, ``"infra_deferred"``; :data:`~cascade.validator.lane_d_state.ABSENT_REASONS`).

    Unarmed (``not cfg.active(boundary)``): ``LaneDSettlement(None, None,
    None)``. Armed: ``report`` is ``None``, ``receipt`` the absent record
    binding the loaded state's digest as prior and posterior, and ``pay``
    ``{"share": cfg.share, "court_uids": []}`` (the caller burns lane D's
    whole share with ``lane_parts(..., d_absent=True)``).

    The prior state is read with :func:`~cascade.validator.lane_d_state.load_prior_state`
    (``state_missing`` / ``state_digest_mismatch``, and ``expect_prior_sha256``
    as in :func:`settle_lane_d`). It never replaces a judgement: a boundary
    the state already settled is ``report_conflict`` (that boundary) or
    ``stale_boundary`` (an older one), and a boundary whose report is
    persisted (a crash W7c recovery must adopt) is ``report_conflict``. An
    unknown ``reason`` is ``LaneDReceiptError`` ``bad_value``. Nothing is
    written, on success or refusal, so a re-call returns the same settlement."""
    if not cfg.active(boundary):
        return LaneDSettlement(report=None, pay=None, receipt=None)
    if not isinstance(reason, str) or reason not in ABSENT_REASONS:
        raise LaneDReceiptError("bad_value", f"absent reason {reason!r} is not one of "
                                f"{list(ABSENT_REASONS)}")
    house = check_house(house)
    boundary = int(boundary)
    state = load_prior_state(state_path, house, boundary,
                             expect_prior_sha256=expect_prior_sha256)
    settled = int(state["boundary"])
    if settled == boundary:
        raise LaneDStateError("report_conflict", f"lane D already settled boundary "
                              f"{boundary}; it cannot be recorded absent")
    if settled > boundary:
        raise LaneDStateError("stale_boundary", f"lane D already settled through {settled}; "
                              f"got {boundary}")
    rp = report_path(state_path, boundary)
    if os.path.lexists(rp):
        raise LaneDStateError("report_conflict", f"{rp}: a report is persisted for boundary "
                              f"{boundary}; recovery must adopt it (settle_lane_d), never "
                              "record it absent")
    receipt = make_absent_receipt(boundary=boundary, state_sha256=state_digest(state),
                                  reason=reason)
    return LaneDSettlement(report=None, pay={"share": float(cfg.share), "court_uids": []},
                           receipt=receipt)


__all__ = ["LaneDSettlement", "absent_settlement", "judge_cohort", "load_pool",
           "settle_lane_d", "window_id"]
