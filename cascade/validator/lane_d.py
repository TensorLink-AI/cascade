"""Box-side lane D settlement (LANE_D_PLAN §1.3–§1.5; W5a–h and W7a–c
deepen it).

:func:`settle_lane_d` runs once per settlement boundary on the main
validator. Unarmed (``not cfg.active(boundary)``) it returns
``report=None, pay=None`` and touches nothing. Armed it:

1. registers newly revealed entries in ``lane_d_state.json`` (``ready``
   TTL, duplicate digests, deregistered hotkeys → closed with a typed reason);
2. selects this settlement's judged entries FIFO by reveal time under the
   per-settlement and per-coldkey caps (the rest wait, ``coldkey_cap_settlement``);
3. advances each judged entry's ``frozen_set`` (W5b,
   :mod:`~cascade.validator.lane_d_frozen`): the union over the newest
   snapshot ``pool_dir`` and the retained ``prior_pool_dirs`` of the windows
   the §1.3 verdicts of
   :func:`~cascade.validator.lane_d_forward.eligible_windows` admit for
   ``max(T_E, T_K)`` (both clocks after the bound, mirror lag, forecast
   products, present in S0, no backfill), excluding windows the entry was
   already judged on (the confirmation set is disjoint), capped at
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
5. decides with the paired cluster bootstrap (clusters = source), max-T over
   the cohort: a member clears iff ``LCB ≥ d_margin`` and its point gain is
   positive. When a miner holds KING_D, every member is also judged, the
   same way and on the same set, against the current house, and clears only
   if it clears both (Proposed change 3: the house ranks first on ties, so a
   copy of the house never clears); the house's rows sit next to the king's.
   A first clear goes ``confirm_pending`` and must clear again on a
   disjoint set (D20 ``confirm_on_king_change``) before it is crowned;
   each verdict reports the §1.3b A10 low-frequency slice (report-only here:
   W3b5 computes ``pass``/``fail`` and its ``lowfreq_regression`` effect);
6. writes the signed ``lane-d-report/1`` (W7a: the schema file
   ``schemas/lane_d_report_1.json``; every entry considered and not judged is
   a typed skip, court slots that burn name a ``burn_reason``; ``pool.snapshots``
   binds every snapshot a judged window's version came from, the newest
   always included, and each window names its ``snapshot_sha256``). The
   signed report is checked by ``validate_report``: a report outside the
   schema raises and nothing is returned or saved (fail closed). Then it
   saves the new state and returns the pay
   record ``{"share", "court_uids"}`` — king first (``None`` while the house
   holds KING_D: that slot burns, D21), then up to ``reward_prior_kings``
   prior MINER kings (an unregistered one is ``None`` and burns).

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
- a crash after the report landed is recovered by recomputing: the body must
  equal the persisted file's (else ``report_conflict``), the file is kept
  byte-for-byte and the state is written; a re-call at the settled boundary
  returns the persisted report, pay and receipt with nothing re-scored,
  re-gated or written; an older boundary is ``stale_boundary``.

Light imports only (numpy + the scorer's numpy half): this module is on the
validator package's import path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.bootstrap import cohort_maxt_lcbs
from ..eval.lane_d_scorer import SCORING_LEVELS, SCORING_METHOD, score_window
from .lane_d_forward import eligible_windows
from .lane_d_frozen import advance_frozen_set
from .lane_d_report import REPORT_SCHEMA, rows_sha256, sign_report, validate_report
from .lane_d_state import (
    LaneDStateError,
    load_prior_state,
    load_report_file,
    make_receipt,
    refresh_house,
    report_digest,
    report_path,
    save_report_file,
    save_state,
    state_digest,
)

POOL_SCHEMA = "pool_d/1"
OPEN_STATUSES = ("pending", "confirm_pending")


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


def load_pool(pool_dir, horizon: int) -> dict:
    """One ``pool_d/1`` snapshot: ``{"snapshot", "sha256", "feeds",
    "first_seen_roots", "windows": [...]}`` with each window's history,
    target, both clocks, cluster, feed, frequency and frontier (``None`` when
    the series' ``frontier`` is absent or ``null``; eligibility then fails
    closed as ``absent_from_s0``)."""
    d = Path(pool_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema") != POOL_SCHEMA:
        raise ValueError(f"{d}: pool schema {meta.get('schema')!r} is not {POOL_SCHEMA}")
    if int(meta.get("horizon", -1)) != int(horizon):
        raise ValueError(f"{d}: pool horizon {meta.get('horizon')} != [lane_d] horizon {horizon}")
    sha = hashlib.sha256("".join(_file_sha(d / n) for n in
                                 ("metadata.json", "hist.npz", "tgt.npz")).encode()).hexdigest()
    windows = []
    with np.load(d / "hist.npz") as hist, np.load(d / "tgt.npz") as tgt:
        for s in meta.get("series", []):
            sid = str(s["series_id"])
            ts = [int(t) for t in s.get("target_ts") or []]
            seen = [int(t) for t in s.get("first_seen_ts") or []]
            if sid not in hist.files or sid not in tgt.files:
                continue
            target = np.asarray(tgt[sid], dtype=np.float64)
            source = str(s.get("source") or sid)
            windows.append({
                "window_id": window_id(sid, ts, horizon) if ts else "",
                "series_id": sid, "source": source,
                "feed": str(s.get("feed") or source),
                "freq": str(s.get("freq") or ""),
                "frontier": _parse_frontier(s.get("frontier"), sid, d),
                "seasonal_period": int(s.get("seasonal_period") or 1),
                "target_ts": ts, "first_seen_ts": seen,
                "end_ts": s.get("end_ts"),
                "history": np.asarray(hist[sid], dtype=np.float64), "target": target,
            })
    return {"snapshot": str(meta.get("snapshot", "")), "sha256": sha,
            "feeds": _parse_feeds(meta.get("feeds"), d),
            "first_seen_roots": _parse_roots(meta.get("first_seen_roots"), d),
            "windows": windows}


def _target_usable(w: dict, horizon: int) -> bool:
    """The stored target is a finite, non-zero ``(horizon,)`` vector."""
    t = w["target"]
    return (t.shape == (horizon,) and bool(np.isfinite(t).all())
            and float(np.abs(t).sum()) > 0.0)


# ── statistic ───────────────────────────────────────────────────────────────


def _geo(crps: np.ndarray, mase: np.ndarray) -> float:
    return math.sqrt(math.exp(float(np.mean(np.log(np.maximum(crps, 1e-12)))))
                     * math.exp(float(np.mean(np.log(np.maximum(mase, 1e-9))))))


def _components(crps: np.ndarray, mase: np.ndarray):
    """Bootstrap inputs whose per-window WQL is exactly ``crps``."""
    return ((np.asarray(crps, dtype=np.float64) / 2.0)[:, None],
            np.ones(len(crps), dtype=np.float64), np.asarray(mase, dtype=np.float64))


def judge_cohort(king: tuple, members: list[tuple], *, clusters: list, cfg, seed: str
                 ) -> list[dict]:
    """``[{"lcb", "gain", "cleared"}]`` per member: max-T LCB over the cohort
    (k = 1 is the plain percentile LCB) and the full-sample relative gain."""
    lcbs = cohort_maxt_lcbs(_components(*king), [_components(*m) for m in members],
                            alpha=float(cfg.alpha), B=int(cfg.bootstrap_B), seed=seed,
                            clusters=clusters)
    kg = _geo(*king)
    out = []
    for (c, m), lcb in zip(members, lcbs, strict=True):
        gain = (kg - _geo(c, m)) / (kg if abs(kg) >= 1e-9 else 1e-9)
        out.append({"lcb": float(lcb), "gain": float(gain),
                    "cleared": bool(float(lcb) >= float(cfg.d_margin) and gain > 0.0)})
    return out


# ── settlement ──────────────────────────────────────────────────────────────


def _forecast_ok(fc: dict | None, wid: str, horizon: int) -> bool | None:
    """True = a usable forecast, None = absent, False = malformed (a fault)."""
    if fc is None or wid not in fc:
        return None
    q = np.asarray(fc[wid], dtype=np.float64)
    return bool(q.shape == (horizon, len(SCORING_LEVELS)) and np.isfinite(q).all())


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


def _select(state: dict, cfg) -> tuple[list[str], list[dict]]:
    """FIFO by reveal time under the per-settlement / per-coldkey caps."""
    open_ = sorted((dg for dg, r in state["entries"].items() if r["status"] in OPEN_STATUSES),
                   key=lambda dg: (state["entries"][dg]["reveal_ts"], dg))
    chosen, skipped, per_cold = [], [], {}
    for dg in open_:
        cold = state["entries"][dg]["coldkey_at_ready"]
        if (len(chosen) >= int(cfg.max_entries_per_settlement)
                or per_cold.get(cold, 0) >= int(cfg.max_judged_per_coldkey_per_settlement)):
            state["entries"][dg]["reason"] = "coldkey_cap_settlement"
            skipped.append({"digest": dg, "reason": "coldkey_cap_settlement"})
            continue
        per_cold[cold] = per_cold.get(cold, 0) + 1
        chosen.append(dg)
    return chosen, skipped


def _comparison_king(rec: dict, state: dict, cfg) -> tuple[str, str]:
    """(stage, king digest) a member is judged against (D20)."""
    conf = rec.get("confirm")
    if rec["status"] == "confirm_pending" and conf:
        if conf["king_digest"] == state["king"]["digest"]:
            return "confirm", state["king"]["digest"]
        if cfg.confirm_on_king_change == "original":
            return "confirm", conf["king_digest"]
    return "first", state["king"]["digest"]


def _king_reveal_ts(state: dict, digest: str) -> int | None:
    if state["king"]["digest"] == digest:
        return int(state["king"]["reveal_ts"])
    if state["house"]["digest"] == digest:
        return int(state["house"]["reveal_ts"])
    for k in state["prior_kings"]:
        if k["digest"] == digest:
            return int(k["reveal_ts"])
    return None


def _load_pools(pool_dir, prior_pool_dirs, horizon: int) -> list[dict]:
    """The retained snapshots, oldest first and the newest (``pool_dir``)
    last; a directory listed twice (or equal to ``pool_dir``) loads once."""
    newest = Path(pool_dir).resolve()
    seen, dirs = {newest}, []
    for d in prior_pool_dirs or ():
        p = Path(d).resolve()
        if p not in seen:
            seen.add(p)
            dirs.append(p)
    return [load_pool(d, horizon) for d in dirs] + [load_pool(newest, horizon)]


def _pool_window_ids(pools: list[dict]) -> set[str]:
    """Every window id any loaded snapshot holds."""
    return {w["window_id"] for p in pools for w in p["windows"] if w["window_id"]}


def _admitted_versions(pools: list[dict], t_entry: int, t_king: int, cfg
                       ) -> tuple[dict[str, tuple[dict, dict]], dict[str, int]]:
    """RV4 one version per window: ``{window_id: (window, pool)}`` taken from
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
    admitted: dict[str, tuple[dict, dict]] = {}
    for pool, vs in zip(pools, verdicts, strict=True):
        for w, v in zip(pool["windows"], vs, strict=True):
            wid = w["window_id"]
            if wid and v.eligible and wid not in admitted:
                admitted[wid] = (w, pool)
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
                state: dict, forecasts: dict, cfg, pinned: dict[str, str]
                ) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    """The cohort's judged window set as ``(window, its snapshot)`` pairs: the
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
    t_entry = max(int(state["entries"][m]["reveal_ts"]) for m in members)
    h = int(cfg.horizon)
    admitted, refused = _admitted_versions(pools, t_entry, int(king_ts), cfg)
    common = set.intersection(*(set(state["entries"][m]["frozen_set"]["window_ids"])
                                for m in members))
    out = []
    for wid in sorted(common):
        if wid not in admitted:
            continue
        w, pool = admitted[wid]
        if wid in pinned and pinned[wid] != pool["sha256"]:
            continue
        if not _target_usable(w, h) or any(
                _forecast_ok(forecasts.get(b), wid, h) is not True for b in bars):
            continue
        if any(wid in state["entries"][m]["used_windows"]
               or _forecast_ok(forecasts.get(m), wid, h) is not True for m in members):
            continue
        out.append((w, pool))
    return out, refused


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
            "bootstrap_B": int(cfg.bootstrap_B), "king_decay": float(cfg.king_decay),
            "reward_prior_kings": int(cfg.reward_prior_kings),
            "confirm_on_king_change": str(cfg.confirm_on_king_change),
            "max_entries_per_settlement": int(cfg.max_entries_per_settlement),
            "max_judged_per_coldkey_per_settlement":
                int(cfg.max_judged_per_coldkey_per_settlement),
            "scoring_levels": [float(q) for q in SCORING_LEVELS],
            "gate_suites": [str(s) for s in cfg.gate_suites],
            "tau_public": float(cfg.tau_public),
            "lowfreq_freqs": [str(f) for f in cfg.lowfreq_freqs],
            # §1.3b draw weights as parsed (a suite without a weight is absent),
            # the T18e/B14 source cap and the §1.7 context floor.
            "gate_weights": {str(s): float(w) for s, w in cfg.gate_weights},
            "max_source_share": float(cfg.max_source_share),
            "min_long_series": int(cfg.min_long_series)}


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


def _lowfreq_candidate(ws: list[dict], cfg) -> dict:
    """The report's §1.3b A10 ``lowfreq`` block for a crowning candidate (a
    verdict that cleared at stage ``confirm``): ``n_windows`` counts the
    frozen set's windows whose frequency is in ``lowfreq_freqs``.

    W7a reports the slice only and never changes a verdict or the crowning.
    W3b5 owns the non-inferiority test itself (CRPS and MASE
    ``LCB_α >= -tau_public`` against KING_D and the house) and its
    ``pass``/``fail`` result (``fail`` = ``lowfreq_regression``, not crowned).
    Until W3b5 lands, a candidate with no slice window is ``no_windows`` and a
    candidate with slice windows is ``not_candidate`` (not evaluated) with the
    true count."""
    freqs = {str(f) for f in cfg.lowfreq_freqs}
    n = sum(1 for w in ws if w["freq"] in freqs)
    return {"result": "no_windows" if n == 0 else "not_candidate",
            "freqs": [str(f) for f in cfg.lowfreq_freqs], "n_windows": n,
            "tau_public": float(cfg.tau_public)}


def _not_candidate(cfg) -> dict:
    return {"result": "not_candidate", "freqs": [str(f) for f in cfg.lowfreq_freqs],
            "n_windows": 0, "tau_public": float(cfg.tau_public)}


def _closing(cfg, boundary: int) -> bool:
    """T18: the settlement at ``until_block`` (the last active one) judges
    nothing and closes every open entry."""
    until = int(cfg.until_block)
    return until > 0 and int(boundary) == until


def settle_lane_d(*, cfg, boundary: int, pool_dir, entries: list[dict], house: dict,
                  forecasts: dict, uid_of: dict, state_path, signer,
                  prior_pool_dirs=(), expect_prior_sha256: str | None = None
                  ) -> LaneDSettlement:
    """Settle lane D at ``boundary`` (see the module docstring).

    ``pool_dir`` is the newest ``pool_d`` snapshot; ``prior_pool_dirs`` are
    the earlier retained ones. They feed every open entry's union and supply
    the history and targets of windows frozen from them.

    ``expect_prior_sha256`` (the last published receipt's
    ``posterior_state_sha256``), when given, must equal the digest of the
    state as loaded (``state_digest_mismatch``). A lost state after a
    persisted settlement is ``state_missing`` and a rolled-back one is
    ``state_digest_mismatch`` (:func:`~cascade.validator.lane_d_state.load_prior_state`);
    every such refusal comes before any pool or forecast is read and writes
    nothing."""
    if not cfg.active(boundary):
        return LaneDSettlement(report=None, pay=None, receipt=None)
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
    # A report file without its state: a crash between the two writes. It is
    # parsed strictly now, before any scoring, and must match the recompute.
    persisted = load_report_file(rp) if rp.exists() else None
    prior_sha = state_digest(state)
    refresh_house(state, house)
    king_before = _king_record(state["king"])
    pools = _load_pools(pool_dir, prior_pool_dirs, int(cfg.horizon))
    pool = pools[-1]
    pool_ids = _pool_window_ids(pools)
    seed = f"lane-d/1:{boundary}:{pool['sha256']}"
    # T18e: the max_source_share drop is seeded by the settlement alone.
    cap_seed = f"lane-d/1:cap:{boundary}"
    # Every entry considered and not judged is a typed skip, including the
    # entries voided this settlement (invariant 11).
    skipped = _register(state, entries, cfg, uid_of)
    h = int(cfg.horizon)

    def skip(dg: str, reason: str) -> None:
        state["entries"][dg]["reason"] = reason
        skipped.append({"digest": dg, "reason": reason})

    if _closing(cfg, boundary):
        for dg, rec in sorted(state["entries"].items()):
            if rec["status"] in OPEN_STATUSES:
                rec["status"] = "closed"
                skip(dg, "lane_closed")
        chosen = []
    else:
        chosen, capped = _select(state, cfg)
        skipped += capped

    # A member whose forecasts are malformed on any pool window is the miner's fault.
    for dg in list(chosen):
        fc = forecasts.get(dg)
        if fc is not None and any(_forecast_ok(fc, wid, h) is False for wid in sorted(pool_ids)):
            state["entries"][dg]["status"] = "closed"
            skip(dg, "checkpoint_fault")
            chosen.remove(dg)

    groups: dict[tuple[str, str], list[str]] = {}
    for dg in chosen:
        stage, kd = _comparison_king(state["entries"][dg], state, cfg)
        if stage == "first" and state["entries"][dg]["status"] == "confirm_pending":
            state["entries"][dg].update(status="pending", reason="pending_floor", confirm=None,
                                        frozen_set=None)
        groups.setdefault((kd, stage), []).append(dg)

    # RV4: each judged window is scored from ONE snapshot's version; the
    # report names that snapshot (``windows[].snapshot_sha256``) and lists
    # every such snapshot in ``pool.snapshots``. ``pinned`` holds the
    # snapshot each already-judged window was scored from.
    pinned: dict[str, str] = {}
    rows, windows_out, cohorts, verdicts = [], {}, [], []
    crowned: list[dict] = []
    for (kd, stage), members in sorted(groups.items()):
        king_ts = _king_reveal_ts(state, kd)
        # Change 3: when a miner holds KING_D, the current house is a second
        # bar on the same set (a copy of a fresh house must not win). Its
        # forecasts are required: a house that cannot be scored fails closed.
        hd = str(state["house"]["digest"])
        bars = [kd] if kd == hd else [kd, hd]
        if king_ts is None or any(forecasts.get(b) is None for b in bars):
            for m in members:
                skip(m, "king_unscored")
            continue
        members = sorted(members, key=lambda m: (state["entries"][m]["reveal_ts"], m))
        frozen = []
        for m in members:
            rec = state["entries"][m]
            rec["frozen_set"] = _advance_member(rec, pools, int(king_ts), cfg, cap_seed)
            if rec["frozen_set"]["frozen"]:
                frozen.append(m)
            else:
                skip(m, "pending_floor")
        members = frozen
        if not members:
            continue
        picked, refused = _cohort_set(pools, members, bars, king_ts, state, forecasts, cfg,
                                      pinned)
        while members and not _meets_floors([w for w, _ in picked], cfg):
            skip(members.pop(), "pending_floor")
            if members:
                picked, refused = _cohort_set(pools, members, bars, king_ts, state, forecasts,
                                              cfg, pinned)
        if not members:
            continue
        ws = [w for w, _ in picked]
        scores = {}
        scored = [*bars, *members]
        for dg in scored:
            pairs = [score_window(forecasts[dg][w["window_id"]], w["target"], w["history"],
                                  w["seasonal_period"]) for w in ws]
            scores[dg] = (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
        for i, (w, src_pool) in enumerate(picked):
            lag = (src_pool["feeds"].get(w["feed"]) or {}).get("mirror_lag_hours")
            pinned[w["window_id"]] = str(src_pool["sha256"])
            windows_out[w["window_id"]] = {
                "window_id": w["window_id"], "source": w["source"], "feed": w["feed"],
                "min_target_ts": int(min(w["target_ts"])),
                "min_first_seen_ts": int(min(w["first_seen_ts"])),
                "mirror_lag_hours": None if lag is None else float(lag),
                # The snapshot whose version of the window was scored.
                "snapshot_sha256": str(src_pool["sha256"])}
            for dg in scored:
                rows.append({"window_id": w["window_id"], "digest": dg, "cohort": len(cohorts),
                             "crps": float(scores[dg][0][i]), "mase": float(scores[dg][1][i])})
        clusters = [w["source"] for w in ws]
        res = judge_cohort(scores[kd], [scores[m] for m in members],
                           clusters=clusters, cfg=cfg, seed=seed)
        if len(bars) > 1:
            # The same max-T judgment with the house as the bar: a member
            # clears only if it clears both. ``judge_cohort`` needs a
            # positive gain, so an exact tie with the house never clears
            # (the house ranks first on ties). RV5: the signed verdict keeps
            # the KING_D ``lcb`` / ``gain`` and records the house judgement
            # as ``house``, so a satellite re-derives both from the rows.
            vs_house = judge_cohort(scores[hd], [scores[m] for m in members],
                                    clusters=clusters, cfg=cfg, seed=seed)
            res = [{**r, "cleared": bool(r["cleared"] and rh["cleared"]),
                    "house": {"digest": hd, "lcb": rh["lcb"], "gain": rh["gain"],
                              "cleared": bool(rh["cleared"])}}
                   for r, rh in zip(res, vs_house, strict=True)]
        set_sha = hashlib.sha256("\n".join(w["window_id"] for w in ws).encode()).hexdigest()
        t_max = max([king_ts] + [state["entries"][m]["reveal_ts"] for m in members])
        cohort = {"index": len(cohorts), "king_digest": kd, "king_reveal_ts": int(king_ts),
                  "stage": stage, "members": list(members), "t_max": int(t_max),
                  "window_ids": [w["window_id"] for w in ws], "set_sha256": set_sha,
                  "first_seen_roots": [], "refused": refused}
        if len(bars) > 1:
            # Present only with a house bar (absent, not null, when the
            # house holds KING_D: one bar, today's shape).
            cohort["house_digest"] = hd
        cohorts.append(cohort)
        for m, r in zip(members, res, strict=True):
            rec = state["entries"][m]
            rec["used_windows"] = sorted(set(rec["used_windows"]) | {w["window_id"] for w in ws})
            first = rec.get("confirm") if stage == "confirm" else None
            confirmation = {"stage": stage, "rule": str(cfg.confirm_on_king_change),
                            "first_set_sha256": first["first_set_sha256"] if first else None,
                            "first_king_digest": first["king_digest"] if first else None}
            # Report-only (W3b5 decides pass/fail): never changes the outcome.
            lowfreq = (_lowfreq_candidate(ws, cfg) if r["cleared"] and stage == "confirm"
                       else _not_candidate(cfg))
            if not r["cleared"]:
                outcome = "lost"
                rec.update(status="lost",
                           reason="lost_confirmation" if stage == "confirm" else "lost_duel")
            elif stage == "first":
                outcome = "confirm_pending"
                # The confirmation accumulates a fresh set, disjoint from
                # ``used_windows`` (D20).
                rec.update(status="confirm_pending", reason="confirm_pending",
                           confirm={"first_set_sha256": set_sha, "king_digest": kd,
                                    "rule": str(cfg.confirm_on_king_change),
                                    "boundary": boundary},
                           frozen_set=None)
            else:
                outcome = "confirmed"
                crowned.append({"digest": m, "lcb": r["lcb"], "reveal_ts": rec["reveal_ts"]})
            verdicts.append({"digest": m, "king_digest": kd, "stage": stage,
                             "cohort": len(cohorts) - 1, **r, "outcome": outcome,
                             "confirmation": confirmation, "lowfreq": lowfreq})

    new_king = None
    if crowned:
        best = sorted(crowned, key=lambda c: (-c["lcb"], c["reveal_ts"], c["digest"]))[0]
        new_king = best["digest"]
        old = state["king"]
        if not old.get("house"):
            state["prior_kings"] = [_king_record(old)] + state["prior_kings"]
            state["prior_kings"] = state["prior_kings"][:max(0, int(cfg.reward_prior_kings))]
        rec = state["entries"][new_king]
        rec.update(status="crowned", reason="crowned")
        state["king"] = {"digest": new_king, "hotkey": rec["hotkey"], "house": False,
                         "reveal_ts": rec["reveal_ts"], "since_boundary": boundary}
        for v in verdicts:
            if v["digest"] == new_king:
                v["outcome"] = "crowned"
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
    }
    report = sign_report(report, signer)
    # Fail closed: a report outside lane-d-report/1 is never returned, and
    # neither the report file nor the state is written (ReportSchemaError
    # propagates). Checked on the signed form, since ``signer`` and
    # ``signature`` are required fields.
    validate_report(report)
    body_sha = report_digest(report)
    if persisted is None:
        save_report_file(rp, report)
    elif persisted.get("boundary") != boundary or report_digest(persisted) != body_sha:
        raise LaneDStateError("report_conflict", f"{rp}: the persisted report's body is not "
                              f"the recomputed settlement's (sha256 {body_sha})")
    else:
        # Recovery: the persisted report (its signature too) is the record.
        report = persisted
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


__all__ = ["LaneDSettlement", "judge_cohort", "load_pool", "settle_lane_d",
           "window_id"]
