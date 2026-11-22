"""Box-side lane D settlement (LANE_D_PLAN §1.3–§1.5; W5a–h and W7a–c
deepen it).

:func:`settle_lane_d` runs once per settlement boundary on the main
validator. Unarmed (``not cfg.active(boundary)``) it returns
``report=None, pay=None`` and touches nothing. Armed it:

1. registers newly revealed entries in ``lane_d_state.json`` (``ready``
   TTL, duplicate digests, deregistered hotkeys → closed with a typed reason);
2. selects this settlement's judged entries FIFO by reveal time under the
   per-settlement and per-coldkey caps (the rest wait, ``coldkey_cap_settlement``);
3. builds each cohort's window set from the §1.3 verdicts of
   :func:`~cascade.validator.lane_d_forward.eligible_windows` for
   ``max(T over cohort ∪ king)`` (both clocks after the bound, mirror lag,
   forecast products, present in S0, no backfill), excluding windows a
   member was already judged on (the confirmation set is disjoint), and
   defers the latest-T members until the floors (``min_windows``,
   ``min_clusters``) are met;
4. scores every window for the king and each member from the eval host's
   ``(horizon, 9)`` quantile forecasts (``score_window``) into report rows;
5. decides with the paired cluster bootstrap (clusters = source), max-T over
   the cohort: a member clears iff ``LCB ≥ d_margin`` and its point gain is
   positive; a first clear goes ``confirm_pending`` and must clear again on a
   disjoint set (D20 ``confirm_on_king_change``) before it is crowned;
6. writes the signed ``lane-d-report/1``, the new state, and returns the pay
   record ``{"share", "court_uids"}`` — king first (``None`` while the house
   holds KING_D: that slot burns, D21), then up to ``reward_prior_kings``
   prior MINER kings (an unregistered one is ``None`` and burns).

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
from .lane_d_report import REPORT_SCHEMA, rows_sha256, sign_report
from .lane_d_state import load_state, refresh_house, save_state

POOL_SCHEMA = "pool_d/1"
OPEN_STATUSES = ("pending", "confirm_pending")


@dataclass(frozen=True)
class LaneDSettlement:
    report: dict | None
    pay: dict | None


# ── pool_d ──────────────────────────────────────────────────────────────────


def window_id(series_id: str, target_ts, horizon: int) -> str:
    """§1.3 stable window identity ``<series_id>:<target_start_ts>:h<h>``."""
    return f"{series_id}:{int(target_ts[0])}:h{int(horizon)}"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_feeds(raw, where: Path) -> dict:
    """Top-level ``feeds``: ``{feed: {"mirror_lag_hours": float | None,
    "forecast_product": bool}}`` (optional; absent = every lag unmeasured)."""
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
    snapshot holding the series; ``None`` when the key is absent (a pre-W6b
    snapshot: the S0 rules are skipped, see ``lane_d_forward``)."""
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


def load_pool(pool_dir, horizon: int) -> dict:
    """One ``pool_d/1`` snapshot: ``{"snapshot", "sha256", "feeds", "windows":
    [...]}`` with each window's history, target, both clocks, cluster, feed
    and frontier (``None`` when the series records no ``frontier`` key)."""
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
                "frontier": _parse_frontier(s.get("frontier"), sid, d),
                "seasonal_period": int(s.get("seasonal_period") or 1),
                "target_ts": ts, "first_seen_ts": seen,
                "end_ts": s.get("end_ts"),
                "history": np.asarray(hist[sid], dtype=np.float64), "target": target,
            })
    return {"snapshot": str(meta.get("snapshot", "")), "sha256": sha,
            "feeds": _parse_feeds(meta.get("feeds"), d), "windows": windows}


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


def _register(state: dict, entries: list[dict], cfg, uid_of: dict) -> None:
    known = state["entries"]
    taken = {state["house"]["digest"], state["king"]["digest"]}
    for e in sorted(entries, key=lambda x: (int(x["reveal_ts"]), str(x["digest"]))):
        dg = str(e["digest"])
        if dg in known:
            continue
        rec = {"hotkey": str(e["hotkey"]), "coldkey_at_ready": str(e.get("coldkey_at_ready") or ""),
               "context": int(e["context"]), "ready_block": int(e["ready_block"]),
               "reveal_block": int(e["reveal_block"]), "reveal_ts": int(e["reveal_ts"]),
               "status": "pending", "reason": "pending_floor", "used_windows": [],
               "confirm": None}
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
    for rec in known.values():
        if rec["status"] in OPEN_STATUSES and rec["hotkey"] not in uid_of:
            rec.update(status="closed", reason="deregistered")


def _select(state: dict, cfg) -> tuple[list[str], list[dict]]:
    """FIFO by reveal time under the per-settlement / per-coldkey caps."""
    open_ = sorted((dg for dg, r in state["entries"].items() if r["status"] in OPEN_STATUSES),
                   key=lambda dg: (state["entries"][dg]["reveal_ts"], dg))
    chosen, skipped, per_cold = [], [], {}
    for dg in open_:
        cold = state["entries"][dg]["coldkey_at_ready"]
        if (len(chosen) >= int(cfg.max_entries_per_settlement)
                or per_cold.get(cold, 0) >= int(cfg.max_judged_per_coldkey_per_settlement)):
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


def _window_set(pool: dict, members: list[str], king_digest: str, king_ts: int, state: dict,
                forecasts: dict, cfg) -> list[dict]:
    t_entry = max(int(state["entries"][m]["reveal_ts"]) for m in members)
    h = int(cfg.horizon)
    verdicts = eligible_windows(pool, t_entry=t_entry, t_king=int(king_ts), cfg=cfg)
    out = []
    for w, v in zip(pool["windows"], verdicts, strict=True):
        wid = w["window_id"]
        if not wid or not v.eligible or not _target_usable(w, h):
            continue
        if _forecast_ok(forecasts.get(king_digest), wid, h) is not True:
            continue
        if any(wid in state["entries"][m]["used_windows"]
               or _forecast_ok(forecasts.get(m), wid, h) is not True for m in members):
            continue
        out.append(w)
    return sorted(out, key=lambda w: w["window_id"])


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
    return slots


def _king_record(k: dict) -> dict:
    return {"digest": k["digest"], "hotkey": k.get("hotkey"), "house": bool(k.get("house")),
            "reveal_ts": int(k["reveal_ts"])}


def _params(cfg) -> dict:
    return {"share": float(cfg.share), "takes_from": str(cfg.takes_from),
            "remainder": str(cfg.remainder), "horizon": int(cfg.horizon),
            "contexts": [int(c) for c in cfg.contexts],
            "forward_margin_hours": float(cfg.forward_margin_hours),
            "min_windows": int(cfg.min_windows), "min_clusters": int(cfg.min_clusters),
            "d_margin": float(cfg.d_margin), "alpha": float(cfg.alpha),
            "bootstrap_B": int(cfg.bootstrap_B), "king_decay": float(cfg.king_decay),
            "reward_prior_kings": int(cfg.reward_prior_kings),
            "confirm_on_king_change": str(cfg.confirm_on_king_change),
            "max_entries_per_settlement": int(cfg.max_entries_per_settlement),
            "max_judged_per_coldkey_per_settlement":
                int(cfg.max_judged_per_coldkey_per_settlement),
            "scoring_levels": [float(q) for q in SCORING_LEVELS]}


def settle_lane_d(*, cfg, boundary: int, pool_dir, entries: list[dict], house: dict,
                  forecasts: dict, uid_of: dict, state_path, signer) -> LaneDSettlement:
    """Settle lane D at ``boundary`` (see the module docstring)."""
    if not cfg.active(boundary):
        return LaneDSettlement(report=None, pay=None)
    boundary = int(boundary)
    state = load_state(state_path, house)
    if int(state["boundary"]) >= boundary:
        raise ValueError(f"lane D already settled through {state['boundary']}; got {boundary}")
    refresh_house(state, house)
    king_before = _king_record(state["king"])
    pool = load_pool(pool_dir, int(cfg.horizon))
    seed = f"lane-d/1:{boundary}:{pool['sha256']}"
    _register(state, entries, cfg, uid_of)
    chosen, skipped = _select(state, cfg)
    h = int(cfg.horizon)

    # A member whose forecasts are malformed on any pool window is the miner's fault.
    for dg in list(chosen):
        fc = forecasts.get(dg)
        if fc is not None and any(_forecast_ok(fc, w["window_id"], h) is False
                                  for w in pool["windows"] if w["window_id"]):
            state["entries"][dg].update(status="closed", reason="checkpoint_fault")
            chosen.remove(dg)

    groups: dict[tuple[str, str], list[str]] = {}
    for dg in chosen:
        stage, kd = _comparison_king(state["entries"][dg], state, cfg)
        if stage == "first" and state["entries"][dg]["status"] == "confirm_pending":
            state["entries"][dg].update(status="pending", reason="pending_floor", confirm=None)
        groups.setdefault((kd, stage), []).append(dg)

    rows, windows_out, cohorts, verdicts = [], {}, [], []
    crowned: list[dict] = []
    for (kd, stage), members in sorted(groups.items()):
        king_ts = _king_reveal_ts(state, kd)
        if king_ts is None or forecasts.get(kd) is None:
            skipped += [{"digest": m, "reason": "king_unscored"} for m in members]
            continue
        members = sorted(members, key=lambda m: (state["entries"][m]["reveal_ts"], m))
        ws = _window_set(pool, members, kd, king_ts, state, forecasts, cfg)
        while members and not _meets_floors(ws, cfg):
            late = members.pop()
            skipped.append({"digest": late, "reason": "pending_floor"})
            ws = _window_set(pool, members, kd, king_ts, state, forecasts, cfg) if members else []
        if not members:
            continue
        scores = {}
        for dg in [kd, *members]:
            pairs = [score_window(forecasts[dg][w["window_id"]], w["target"], w["history"],
                                  w["seasonal_period"]) for w in ws]
            scores[dg] = (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
        for i, w in enumerate(ws):
            windows_out[w["window_id"]] = {
                "window_id": w["window_id"], "source": w["source"],
                "min_target_ts": int(min(w["target_ts"])),
                "min_first_seen_ts": int(min(w["first_seen_ts"]))}
            for dg in [kd, *members]:
                rows.append({"window_id": w["window_id"], "digest": dg, "cohort": len(cohorts),
                             "crps": float(scores[dg][0][i]), "mase": float(scores[dg][1][i])})
        res = judge_cohort(scores[kd], [scores[m] for m in members],
                           clusters=[w["source"] for w in ws], cfg=cfg, seed=seed)
        set_sha = hashlib.sha256("\n".join(w["window_id"] for w in ws).encode()).hexdigest()
        t_max = max([king_ts] + [state["entries"][m]["reveal_ts"] for m in members])
        cohorts.append({"index": len(cohorts), "king_digest": kd, "king_reveal_ts": int(king_ts),
                        "stage": stage, "members": list(members), "t_max": int(t_max),
                        "window_ids": [w["window_id"] for w in ws], "set_sha256": set_sha})
        for m, r in zip(members, res, strict=True):
            rec = state["entries"][m]
            rec["used_windows"] = sorted(set(rec["used_windows"]) | {w["window_id"] for w in ws})
            if not r["cleared"]:
                outcome = "lost"
                rec.update(status="lost",
                           reason="lost_confirmation" if stage == "confirm" else "lost_duel")
            elif stage == "first":
                outcome = "confirm_pending"
                rec.update(status="confirm_pending", reason="confirm_pending",
                           confirm={"first_set_sha256": set_sha, "king_digest": kd,
                                    "rule": str(cfg.confirm_on_king_change),
                                    "boundary": boundary})
            else:
                outcome = "confirmed"
                crowned.append({"digest": m, "lcb": r["lcb"], "reveal_ts": rec["reveal_ts"]})
            verdicts.append({"digest": m, "king_digest": kd, "stage": stage,
                             "cohort": len(cohorts) - 1, **r, "outcome": outcome})

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
    report = {
        "schema": REPORT_SCHEMA, "boundary": boundary, "scoring_method": SCORING_METHOD,
        "params": _params(cfg), "seed": seed,
        "pool": {"snapshot": pool["snapshot"], "sha256": pool["sha256"], "horizon": h},
        "house": _king_record({**state["house"], "hotkey": None, "house": True}),
        "king_before": king_before, "king_after": _king_record(state["king"]),
        "crowned": new_king,
        "prior_kings": [_king_record(p) for p in state["prior_kings"]],
        "entries": [{"digest": dg, **{k: r[k] for k in ("hotkey", "coldkey_at_ready", "context",
                                                        "ready_block", "reveal_block",
                                                        "reveal_ts", "status", "reason")}}
                    for dg, r in sorted(state["entries"].items())],
        "skipped": skipped, "cohorts": cohorts,
        "windows": [windows_out[k] for k in sorted(windows_out)],
        "verdicts": verdicts, "rows": rows, "rows_sha256": rows_sha256(rows),
        "pay": {"share": pay["share"], "court": court},
    }
    report = sign_report(report, signer)
    state["boundary"] = boundary
    state["last_report"] = {"boundary": boundary, "rows_sha256": report["rows_sha256"],
                            "signature": report["signature"]}
    save_state(state_path, state)
    return LaneDSettlement(report=report, pay=pay)


__all__ = ["LaneDSettlement", "judge_cohort", "load_pool", "settle_lane_d",
           "window_id"]
