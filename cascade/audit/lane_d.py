"""``check_lane_d`` — the audit of one signed lane D report (LANE_D_PLAN
§1.5, §14.3; W9 deepens it).

Invariant 12: this module re-derives everything from the report and the
config. It never imports the producer's eligibility, verdict or weight code
(``cascade.validator.lane_d*``, the emission's lane D producer); it shares
only the generic statistic (``cascade.eval.bootstrap``) and the sr25519
verifier. Checks, in order (the first failure is the verdict):

1. schema ``lane-d-report/1`` and the settlement is one lane D judges;
2. the signature (the signer's sr25519 over the canonical body);
3. the stated parameters equal ``cfg`` (§14.3 rule 2);
4. ``rows_sha256`` over the rows;
5. per cohort: forward-only on both clocks against ``max(T) + margin``, the
   floors, and exactly one finite row per (window, model);
6. per cohort: the max-T LCBs, gains, clears and outcomes re-derived from
   the rows; the crowning and the paid court follow from them.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ..eval.bootstrap import cohort_maxt_lcbs
from .checks import CheckResult, _fail, _ok, _warn

NAME = "lane_d"
SCHEMA = "lane-d-report/1"
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _expected_params(cfg) -> dict:
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
            "scoring_levels": list(LEVELS),
            # §14.3 fields W7a added to lane-d-report/1, re-derived from cfg.
            "horizons": [int(cfg.horizon)],
            "max_mirror_lag_hours": float(cfg.max_mirror_lag_hours),
            "gate_suites": [str(s) for s in cfg.gate_suites],
            "tau_public": float(cfg.tau_public),
            "lowfreq_freqs": [str(f) for f in cfg.lowfreq_freqs],
            # RV4: gate params W7a added to lane-d-report/1 params.
            "gate_weights": {str(s): float(w) for s, w in cfg.gate_weights},
            "max_source_share": float(cfg.max_source_share),
            "min_long_series": int(cfg.min_long_series)}


def _signature(report: dict, signer_hotkey: str) -> str | None:
    """``None`` when the signature verifies, else why not; raises
    ``SignatureVerifierUnavailable`` when it cannot be checked."""
    from ..shared.sr25519 import verify_hotkey

    sig = report.get("signature")
    if not sig:
        return "unsigned"
    if report.get("signer") != signer_hotkey:
        return f"signer {report.get('signer')!r} is not the pinned {signer_hotkey!r}"
    body = _canon({k: v for k, v in report.items() if k != "signature"})
    return None if verify_hotkey(signer_hotkey, body, str(sig)) else "signature does not verify"


def _geo(crps: np.ndarray, mase: np.ndarray) -> float:
    return math.sqrt(math.exp(float(np.mean(np.log(np.maximum(crps, 1e-12)))))
                     * math.exp(float(np.mean(np.log(np.maximum(mase, 1e-9))))))


def _comp(crps: np.ndarray, mase: np.ndarray):
    return (crps[:, None] / 2.0, np.ones(len(crps)), mase)


def _cohort_problems(report: dict, cfg, cohort: dict, rows_by: dict, windows: dict,
                     entries: dict) -> list[str]:
    out = []
    idx = int(cohort["index"])
    kd, members, wids = cohort["king_digest"], list(cohort["members"]), list(cohort["window_ids"])
    ts = [int(cohort["king_reveal_ts"])]
    for m in members:
        if m not in entries:
            return [f"cohort {idx}: member {m[:12]} is not a listed entry"]
        ts.append(int(entries[m]["reveal_ts"]))
    t_max = max(ts)
    if int(cohort["t_max"]) != t_max:
        out.append(f"cohort {idx}: t_max {cohort['t_max']} != re-derived {t_max}")
    bound = t_max + 3600.0 * float(cfg.forward_margin_hours)
    if len(set(wids)) != len(wids):
        out.append(f"cohort {idx}: duplicate window ids")
    for wid in wids:
        w = windows.get(wid)
        if w is None:
            out.append(f"cohort {idx}: window {wid} has no clock record")
            continue
        if not (int(w["min_target_ts"]) > bound and int(w["min_first_seen_ts"]) > bound):
            out.append(f"cohort {idx}: window {wid} is not forward-only for max(T)={t_max}")
    if len(wids) < int(cfg.min_windows):
        out.append(f"cohort {idx}: {len(wids)} windows < min_windows {cfg.min_windows}")
    clusters = [windows[w]["source"] for w in wids if w in windows]
    if len(set(clusters)) < int(cfg.min_clusters):
        out.append(f"cohort {idx}: {len(set(clusters))} clusters < min_clusters {cfg.min_clusters}")
    if out:
        return out
    scores = {}
    for dg in [kd, *members]:
        crps, mase = [], []
        for wid in wids:
            r = rows_by.get((idx, wid, dg))
            if r is None:
                return [f"cohort {idx}: no row for {dg[:12]} on {wid}"]
            c, m = float(r["crps"]), float(r["mase"])
            if not (math.isfinite(c) and math.isfinite(m) and c >= 0.0 and m >= 0.0):
                return [f"cohort {idx}: non-finite or negative row for {dg[:12]} on {wid}"]
            crps.append(c)
            mase.append(m)
        scores[dg] = (np.asarray(crps), np.asarray(mase))
    lcbs = cohort_maxt_lcbs(_comp(*scores[kd]), [_comp(*scores[m]) for m in members],
                            alpha=float(cfg.alpha), B=int(cfg.bootstrap_B),
                            seed=str(report["seed"]), clusters=clusters)
    kg = _geo(*scores[kd])
    verdicts = {v["digest"]: v for v in report.get("verdicts", []) if int(v["cohort"]) == idx}
    if set(verdicts) != set(members):
        return [f"cohort {idx}: verdicts {sorted(verdicts)} do not cover members {members}"]
    for m, lcb in zip(members, lcbs, strict=True):
        v = verdicts[m]
        gain = (kg - _geo(*scores[m])) / (kg if abs(kg) >= 1e-9 else 1e-9)
        cleared = bool(lcb >= float(cfg.d_margin) and gain > 0.0)
        if not (math.isclose(float(v["lcb"]), lcb, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(float(v["gain"]), gain, rel_tol=1e-9, abs_tol=1e-12)):
            out.append(f"cohort {idx}: {m[:12]} lcb/gain {v['lcb']}/{v['gain']} != "
                       f"re-derived {lcb}/{gain}")
        if bool(v["cleared"]) != cleared:
            out.append(f"cohort {idx}: {m[:12]} cleared={v['cleared']} != re-derived {cleared}")
        stage = cohort["stage"]
        want = ("lost",) if not cleared else (
            ("confirm_pending",) if stage == "first" else ("confirmed", "crowned"))
        if v["outcome"] not in want or v["stage"] != stage or v["king_digest"] != kd:
            out.append(f"cohort {idx}: {m[:12]} outcome {v['outcome']} not in {want}")
    return out


def check_lane_d(report: dict, cfg, *, signer_hotkey: str) -> CheckResult:
    """PASS on a genuine report; FAIL when the signature, ``rows_sha256``,
    a stated parameter or the verdict re-derived from the rows disagrees."""
    from ..shared.sr25519 import SignatureVerifierUnavailable

    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        got = report.get("schema") if isinstance(report, dict) else type(report).__name__
        return _fail(NAME, f"unknown report schema {got!r}")
    try:
        boundary = int(report["boundary"])
        if not cfg.active(boundary):
            return _fail(NAME, f"lane D does not judge the settlement at {boundary}")
        try:
            why = _signature(report, signer_hotkey)
        except SignatureVerifierUnavailable as e:
            return _warn(NAME, f"cannot verify the signature: {e}")
        if why:
            return _fail(NAME, why)
        if report.get("scoring_method") != "quantile_head":
            return _fail(NAME, f"scoring_method {report.get('scoring_method')!r}")
        want = _expected_params(cfg)
        if report.get("params") != want:
            diff = sorted(k for k in set(want) | set(report.get("params") or {})
                          if (report.get("params") or {}).get(k) != want.get(k))
            return _fail(NAME, f"stated params differ from config on {diff}")
        rows = list(report.get("rows") or [])
        if hashlib.sha256(_canon(rows)).hexdigest() != report.get("rows_sha256"):
            return _fail(NAME, "rows_sha256 does not match the rows")
        rows_by = {}
        for r in rows:
            key = (int(r["cohort"]), str(r["window_id"]), str(r["digest"]))
            if key in rows_by:
                return _fail(NAME, f"duplicate row {key}")
            rows_by[key] = r
        windows = {w["window_id"]: w for w in report.get("windows") or []}
        entries = {e["digest"]: e for e in report.get("entries") or []}
        cohorts = list(report.get("cohorts") or [])
        expected_rows = sum(len(c["window_ids"]) * (1 + len(c["members"])) for c in cohorts)
        if expected_rows != len(rows):
            return _fail(NAME, f"{len(rows)} rows; the cohorts account for {expected_rows}")
        problems = []
        for c in cohorts:
            problems += _cohort_problems(report, cfg, c, rows_by, windows, entries)
        if problems:
            return _fail(NAME, "; ".join(problems[:5]))
        # The crowning and the paid court follow from the verdicts.
        confirmed = [v for v in report.get("verdicts", []) if v["outcome"] in ("confirmed", "crowned")]
        before, after = report["king_before"], report["king_after"]
        if confirmed:
            best = sorted(confirmed, key=lambda v: (-float(v["lcb"]),
                                                    int(entries[v["digest"]]["reveal_ts"]),
                                                    v["digest"]))[0]["digest"]
            if report.get("crowned") != best or after["digest"] != best or after["house"]:
                return _fail(NAME, f"crowned {report.get('crowned')!r}; re-derived {best!r}")
        elif report.get("crowned") is not None or after["digest"] != before["digest"]:
            return _fail(NAME, "the king changed without a confirmed verdict")
        court = (report.get("pay") or {}).get("court") or []
        if float((report.get("pay") or {}).get("share", -1.0)) != float(cfg.share):
            return _fail(NAME, "pay share differs from config")
        if not court or court[0]["digest"] != after["digest"] \
                or (court[0]["hotkey"] is None) != bool(after["house"]):
            return _fail(NAME, "the paid court does not start with KING_D (house slot unpaid)")
        if len(court) > 1 + int(cfg.reward_prior_kings):
            return _fail(NAME, f"court has {len(court)} slots; at most "
                               f"{1 + int(cfg.reward_prior_kings)}")
        if any(s["digest"] == report["house"]["digest"] for s in court[1:]):
            return _fail(NAME, "the house sits in the court as a prior king")
    except (KeyError, TypeError, ValueError) as e:
        return _fail(NAME, f"malformed report: {type(e).__name__}: {e}")
    n_rows = len(report.get("rows") or [])
    return _ok(NAME, f"boundary {boundary}: {len(cohorts)} cohort(s), {n_rows} rows re-derived")


__all__ = ["check_lane_d"]
