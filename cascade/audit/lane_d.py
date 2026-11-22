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
   the rows; when a miner holds KING_D the house is a second bar, and its
   judgement (``cohorts[].house_digest``, ``verdicts[].house``) is
   re-derived the same way and must be present exactly then;
7. the §1.3b gate (W5e): the signed ``gate`` record's draw (``block`` is
   the boundary; ``suite`` / ``eligible`` re-drawn from ``block_hash``, the
   config's ``gate_suites`` / ``gate_weights`` and the stated sentinel),
   every result's outcome and ``blocked_by`` from its four LCBs against
   ``-tau_public``, and each crowning candidate's verdict from its gate
   outcome. Without a gate record nothing crowns (fail closed). The crown
   is the highest-LCB passing confirmation or a held candidate re-gated
   ``pass``; the paid court follows from it.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

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


def _close(got, want: float) -> bool:
    return (isinstance(got, (int, float)) and not isinstance(got, bool)
            and math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-12))


def _judge(report: dict, cfg, scores: dict, bar: str, members: list[str],
           clusters: list) -> list[tuple[float, float, bool]]:
    """``(lcb, gain, cleared)`` per member: the max-T judgement of the
    cohort against ``bar`` on the signed rows and ``report["seed"]``."""
    lcbs = cohort_maxt_lcbs(_comp(*scores[bar]), [_comp(*scores[m]) for m in members],
                            alpha=float(cfg.alpha), B=int(cfg.bootstrap_B),
                            seed=str(report["seed"]), clusters=clusters)
    bg = _geo(*scores[bar])
    out = []
    for m, lcb in zip(members, lcbs, strict=True):
        gain = (bg - _geo(*scores[m])) / (bg if abs(bg) >= 1e-9 else 1e-9)
        out.append((float(lcb), float(gain), bool(lcb >= float(cfg.d_margin) and gain > 0.0)))
    return out


def _house_problems(idx: int, m: str, hv, hd: str, want: tuple[float, float, bool]
                    ) -> list[str]:
    """The verdict's ``house`` judgement against the one re-derived from the rows."""
    if not isinstance(hv, dict):
        return [f"cohort {idx}: {m[:12]} has no house judgement (house {hd[:12]} is a bar)"]
    if set(hv) != {"digest", "lcb", "gain", "cleared"}:
        return [f"cohort {idx}: {m[:12]} house judgement keys {sorted(hv)}"]
    out = []
    if hv["digest"] != hd:
        out.append(f"cohort {idx}: {m[:12]} house judgement names {str(hv['digest'])[:12]}, "
                   f"not the house {hd[:12]}")
    lcb, gain, cleared = want
    if not (_close(hv["lcb"], lcb) and _close(hv["gain"], gain)):
        out.append(f"cohort {idx}: {m[:12]} house lcb/gain {hv['lcb']}/{hv['gain']} != "
                   f"re-derived {lcb}/{gain}")
    if hv["cleared"] is not cleared:
        out.append(f"cohort {idx}: {m[:12]} house cleared={hv['cleared']} != re-derived {cleared}")
    return out


def _cohort_problems(report: dict, cfg, cohort: dict, rows_by: dict, windows: dict,
                     entries: dict, confirms: dict) -> list[str]:
    """The cohort's problems. Every member whose re-derived judgement clears
    stage ``confirm`` (a crowning candidate) is added to ``confirms``; its
    outcome is the gate's to decide (:func:`_crown_problems`)."""
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
    # W5c change 3 (RV5): with a miner holding KING_D, the report's house is
    # a second bar. Whether it applies is re-derived, never read from the
    # house keys, so an omitted judgement cannot hide the bar.
    hd = str(report["house"]["digest"])
    bars = [kd] if kd == hd else [kd, hd]
    if len(bars) == 1:
        if "house_digest" in cohort:
            return [f"cohort {idx}: house_digest where the house holds KING_D"]
    elif cohort.get("house_digest") != hd:
        return [f"cohort {idx}: house_digest {cohort.get('house_digest')!r} is not the "
                f"report's house {hd[:12]} (a miner holds KING_D)"]
    scores = {}
    for dg in [*bars, *members]:
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
    judged = {b: _judge(report, cfg, scores, b, members, clusters) for b in bars}
    verdicts = {v["digest"]: v for v in report.get("verdicts", []) if int(v["cohort"]) == idx}
    if set(verdicts) != set(members):
        return [f"cohort {idx}: verdicts {sorted(verdicts)} do not cover members {members}"]
    for i, m in enumerate(members):
        v = verdicts[m]
        lcb, gain, cleared = judged[kd][i]
        if not (_close(v["lcb"], lcb) and _close(v["gain"], gain)):
            out.append(f"cohort {idx}: {m[:12]} lcb/gain {v['lcb']}/{v['gain']} != "
                       f"re-derived {lcb}/{gain}")
        if len(bars) == 1:
            if "house" in v:
                out.append(f"cohort {idx}: {m[:12]} carries a house judgement where the "
                           f"house holds KING_D")
        else:
            out += _house_problems(idx, m, v.get("house"), hd, judged[hd][i])
            cleared = bool(cleared and judged[hd][i][2])
        if bool(v["cleared"]) != cleared:
            out.append(f"cohort {idx}: {m[:12]} cleared={v['cleared']} != re-derived {cleared}")
        stage = cohort["stage"]
        if v["stage"] != stage or v["king_digest"] != kd:
            out.append(f"cohort {idx}: {m[:12]} stage/king {v['stage']}/"
                       f"{str(v['king_digest'])[:12]} != cohort {stage}/{kd[:12]}")
            continue
        if cleared and stage == "confirm":
            confirms[m] = v
            continue
        want = "lost" if not cleared else "confirm_pending"
        if v["outcome"] != want:
            out.append(f"cohort {idx}: {m[:12]} outcome {v['outcome']} != re-derived {want}")
    return out


# ── the §1.3b gate (W5e), re-derived ────────────────────────────────────────
#
# The draw is W3b1's consensus function, re-implemented here from its spec
# (invariant 12: the audit never imports ``cascade.validator.lane_d*``): one
# suite from the boundary's block hash over ``gate_suites`` minus the era's
# sentinel, weighted by ``gate_weights`` (absent = 0), by its own tagged
# sha256 mapped onto the cumulative weights in exact rational arithmetic.

_DRAW_TAG = b"cascade|lane_d|gate_suite|v1|"
_DRAW_SPACE = 1 << 256
_HEX = frozenset("0123456789abcdef")
GATE_OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
GATE_REFS = ("king_d", "house")
_GATE_METRICS = ("crps", "mase")
_GATE_KEYS = {"suite", "block", "block_hash", "sentinel", "eligible", "results"}
_RESULT_KEYS = {"digest", "outcome", "lcb", "blocked_by", "session"}


def _gate_weights(cfg, suites: list[str]) -> dict[str, Fraction]:
    raw = cfg.gate_weights
    if raw is None:
        return {s: Fraction(1) for s in suites}
    pairs = list(raw.items()) if isinstance(raw, dict) else [tuple(p) for p in raw]
    return {str(s): Fraction(float(w)) for s, w in pairs if str(s) in suites}


def _gate_draw(cfg, block_hash, sentinel) -> tuple[str, list[str]] | str:
    """``(suite, eligible)`` W3b1 draws for ``block_hash`` over ``cfg``, or
    why the stated hash / sentinel cannot be drawn from."""
    if not isinstance(block_hash, str) or not block_hash.startswith("0x") \
            or len(block_hash) != 66 or not set(block_hash[2:]) <= _HEX:
        return f"gate block_hash {block_hash!r} is not 0x + 64 lowercase hex digits"
    if sentinel is not None and (not isinstance(sentinel, str) or not sentinel):
        return f"gate sentinel {sentinel!r} is neither null nor a suite name"
    suites = [str(s) for s in cfg.gate_suites]
    weights = _gate_weights(cfg, suites)
    eligible = [s for s in suites if s != sentinel and weights.get(s, Fraction(0)) > 0]
    if not eligible:
        return f"no eligible gate suite once the sentinel {sentinel!r} is removed"
    total = sum((weights[s] for s in eligible), Fraction(0))
    d = int.from_bytes(hashlib.sha256(_DRAW_TAG + block_hash[2:].encode("ascii")).digest(),
                       "big")
    point = Fraction(d, _DRAW_SPACE) * total
    acc = Fraction(0)
    for s in eligible:
        acc += weights[s]
        if point < acc:
            return s, eligible
    return "the gate draw fell past the cumulative total"


def _finite(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def _result_problem(i: int, res, tau: float) -> str | None:
    """Why ``gate.results[i]`` disagrees with W3b2's ``paired_gate`` rule
    re-derived from its own LCBs, or ``None``."""
    if not isinstance(res, dict) or set(res) != _RESULT_KEYS:
        keys = sorted(res) if isinstance(res, dict) else type(res).__name__
        return f"gate result {i}: keys {keys}"
    dg, outcome = str(res["digest"])[:12], res["outcome"]
    if outcome not in GATE_OUTCOMES:
        return f"gate result {i} ({dg}): outcome {outcome!r} not in {GATE_OUTCOMES}"
    if not isinstance(res["session"], str) or not res["session"]:
        return f"gate result {i} ({dg}): no bench session"
    blocked = res["blocked_by"]
    if not isinstance(blocked, list) or len(set(map(str, blocked))) != len(blocked):
        return f"gate result {i} ({dg}): blocked_by {blocked!r} is not a list of refs"
    lcb = res["lcb"]
    if outcome in ("checkpoint_fault", "bench_hold"):
        if lcb is not None or blocked:
            return f"gate result {i} ({dg}): {outcome} carries LCBs or blocked_by"
        return None
    if not isinstance(lcb, dict) or set(lcb) != set(GATE_REFS) or any(
            not isinstance(lcb[r], dict) or set(lcb[r]) != set(_GATE_METRICS)
            or not all(_finite(lcb[r][m]) for m in _GATE_METRICS) for r in GATE_REFS):
        return f"gate result {i} ({dg}): {outcome} without four finite LCBs"
    below = {r for r in GATE_REFS if any(float(lcb[r][m]) < -tau for m in _GATE_METRICS)}
    want = "gate_regression" if below else "pass"
    if outcome != want:
        return (f"gate result {i} ({dg}): outcome {outcome}; its LCBs re-derive {want} "
                f"(-tau_public={-tau})")
    if set(map(str, blocked)) != below:
        return (f"gate result {i} ({dg}): blocked_by {sorted(map(str, blocked))} != "
                f"re-derived {sorted(below)}")
    return None


def _gate_problems(report: dict, cfg, boundary: int) -> tuple[dict, list[str]]:
    """``({digest: outcome}, problems)`` for the signed ``gate`` record: the
    draw and every result's outcome re-derived from the config and its LCBs.
    No ``gate`` key gives ``({}, [])`` (the gate did not run)."""
    if "gate" not in report:
        return {}, []
    gate = report["gate"]
    if not isinstance(gate, dict):
        return {}, [f"gate is {type(gate).__name__}, not a record"]
    if set(gate) != _GATE_KEYS:
        return {}, [f"gate keys {sorted(gate)}"]
    block = gate["block"]
    if isinstance(block, bool) or not isinstance(block, int) or block != boundary:
        return {}, [f"gate block {block!r} is not the boundary {boundary}"]
    drawn = _gate_draw(cfg, gate["block_hash"], gate["sentinel"])
    if isinstance(drawn, str):
        return {}, [drawn]
    suite, eligible = drawn
    out = []
    if gate["eligible"] != eligible:
        out.append(f"gate eligible {gate['eligible']!r} != re-derived {eligible}")
    if gate["suite"] != suite:
        out.append(f"gate suite {gate['suite']!r} is not the draw {suite!r} for "
                   f"block_hash {gate['block_hash']} (sentinel {gate['sentinel']!r})")
    results = gate["results"]
    if not isinstance(results, list) or not results:
        return {}, [*out, "gate results are empty: the gate ran over no candidate"]
    outcomes: dict = {}
    tau = float(cfg.tau_public)
    for i, res in enumerate(results):
        why = _result_problem(i, res, tau)
        if why:
            out.append(why)
            continue
        if res["digest"] in outcomes:
            out.append(f"gate result {i}: digest {str(res['digest'])[:12]} listed twice")
        outcomes[res["digest"]] = res["outcome"]
    return outcomes, out


def _crown_problems(report: dict, cfg, boundary: int, confirms: dict, entries: dict
                    ) -> tuple[str | None, list[str]]:
    """``(re-derived crown, problems)``. ``confirms`` maps every verdict
    whose re-derived judgement clears stage ``confirm`` (a crowning
    candidate) to that verdict. W5e: the gate decides each candidate
    (``pass`` crowns or stays ``confirmed``, ``gate_regression`` /
    ``checkpoint_fault`` is ``lost``, ``bench_hold`` stays ``confirmed``);
    without a gate record every candidate is held and nothing crowns; the
    crown is the highest-LCB passing confirmation, or a held candidate
    re-gated ``pass`` (it has no verdict this settlement)."""
    outcomes, out = _gate_problems(report, cfg, boundary)
    if out:
        return None, out
    gated = "gate" in report
    verdicts = {v["digest"]: v for v in report.get("verdicts") or []}
    for dg in confirms:
        if gated and dg not in outcomes:
            out.append(f"cleared confirmation {dg[:12]} has no gate result")
    held = []
    for dg in outcomes:
        if dg in confirms:
            continue
        if dg in verdicts:
            out.append(f"gate result for {str(dg)[:12]}, whose verdict is not a cleared "
                       f"confirmation")
        elif dg not in entries:
            out.append(f"gate result for {str(dg)[:12]}, neither a verdict nor a listed entry")
        else:
            held.append(dg)
    if out:
        return None, out
    passing = sorted((v for dg, v in confirms.items() if outcomes.get(dg) == "pass"),
                     key=lambda v: (-float(v["lcb"]), int(entries[v["digest"]]["reveal_ts"]),
                                    v["digest"]))
    # A held candidate's clear (its LCB) is not in this report, so when one
    # passes the crown is the best passing confirmation or that candidate.
    allowed = [v["digest"] for v in passing[:1]] + [dg for dg in held if outcomes[dg] == "pass"]
    stated = report.get("crowned")
    crown = stated if stated in allowed else (allowed[0] if allowed else None)
    for dg, v in confirms.items():
        o = outcomes.get(dg)
        if o in (None, "bench_hold"):
            want = "confirmed"
        elif o in ("gate_regression", "checkpoint_fault"):
            want = "lost"
        else:
            want = "crowned" if dg == crown else "confirmed"
        if v["outcome"] != want:
            out.append(f"{dg[:12]}: outcome {v['outcome']} != re-derived {want} "
                       f"(gate {o or 'absent'})")
    return crown, out


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
        # One row per (window, model): KING_D, the house when it is a second
        # bar (a miner holds KING_D), and every member.
        house = report["house"]["digest"]
        expected_rows = sum(len(c["window_ids"])
                            * ((1 if c["king_digest"] == house else 2) + len(c["members"]))
                            for c in cohorts)
        if expected_rows != len(rows):
            return _fail(NAME, f"{len(rows)} rows; the cohorts account for {expected_rows}")
        problems, confirms = [], {}
        for c in cohorts:
            problems += _cohort_problems(report, cfg, c, rows_by, windows, entries, confirms)
        if problems:
            return _fail(NAME, "; ".join(problems[:5]))
        # The §1.3b gate decides every crowning candidate; the crowning and
        # the paid court follow from it.
        crown, problems = _crown_problems(report, cfg, boundary, confirms, entries)
        if problems:
            return _fail(NAME, "; ".join(problems[:5]))
        before, after = report["king_before"], report["king_after"]
        if crown is not None:
            if report.get("crowned") != crown or after["digest"] != crown or after["house"]:
                return _fail(NAME, f"crowned {report.get('crowned')!r}; re-derived {crown!r}")
        elif report.get("crowned") is not None or after["digest"] != before["digest"]:
            why = ("no gate record (the gate did not run)" if "gate" not in report
                   else "no gate result passes")
            return _fail(NAME, f"crowned {report.get('crowned')!r} / king changed, but {why}: "
                               f"nothing may crown")
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
