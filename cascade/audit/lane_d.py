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
   re-derived the same way and must be present exactly then. W5f: a new
   house challenges a miner KING_D alone in its own cohort (its T is the
   report's house record's) with KING_D as its only bar, and a cleared,
   passing house retakes KING_D (``king_after`` the house record,
   ``crowned`` null);
7. the §1.3b gate (W5e): the signed ``gate`` record's draw (``block`` is
   the boundary; ``suite`` / ``eligible`` re-drawn from ``block_hash``, the
   config's ``gate_suites`` / ``gate_weights`` and the stated sentinel),
   every result's outcome and ``blocked_by`` from its four LCBs against
   ``-tau_public``, and each crowning candidate's verdict from its gate
   outcome. Without a gate record nothing crowns (fail closed). The crown
   is the highest-LCB passing first clear or a held candidate re-gated
   ``pass``; the paid court follows from it. rev6.11 (b): there is no
   confirmation stage; a crown is first clear ∧ A10 ∧ gate, and a cohort or
   verdict of any stage but ``first`` is a problem.

RV7 B2 adds the crown's provenance (README "Correction (RV7 B2)"):

- ``seed`` is ``lane-d/1:{boundary}:{pool.sha256}``; every cohort is
  stage ``first`` and judged against ``king_before``;
- a cleared verdict whose A10 ``lowfreq`` block reads ``pass`` /
  ``fail`` has that result re-derived from its LCBs; a ``fail`` is ``lost``
  and never gated;
- a held candidate's gate result carries ``held_from`` (its hold: an
  earlier boundary, an LCB that cleared ``d_margin``, the same KING_D: a
  king change drops a hold); a verdict-backed result carries none;
- chain mode (``prior_reports``, what W9 runs): each ``held_from`` is shown
  by an earlier report signed by the
  same key, matched by boundary and body digest (recomputed here from the
  canonical body, never through the producer);
- ``era_sentinel``: ``gate.sentinel`` is the era's W-S1 suite.

W9 re-derives what main took on trust (tests/accept/w9/README.md):

- the judged set and every selection skip: ``judged`` is the verdicts,
  ``skipped`` matches ``entries[].reason``, no open entry is silently
  omitted, and the caps walked FIFO by ``(reveal_ts, digest)`` over the
  selection candidates name exactly the ``coldkey_cap_settlement`` skips;
- the registration rules a signed entry record shows (a resolved coldkey,
  an armed context, ``reveal_block - ready_block <= ready_ttl_blocks``);
- each judged window's mirror lag and the king's T (``king_reveal_ts`` of a
  cohort judged against ``king_before`` or the house is that record's);
- the canonical row order under ``rows_sha256``;
- chain mode: ``king_before`` is where the latest verified prior report left
  KING_D (or, after a house holder, this report's refreshed house), the
  house record carries over, and an entry's recorded inputs never change;
- with ``pool_dirs``: every judged snapshot's digest, its ``frontier`` /
  ``feeds`` completeness and each judged window's record and §1.3
  eligibility (:func:`window_refusal`, re-implemented here from the spec).

rev6.18 B3 (owner 2026-11-23): every check has a refusal code
(:data:`REFUSAL_CODES`, the plan's table), ``burn`` or ``alert``. Only the
report signature, the config echo, the scores reproducing the named winner
and the weights matching the paid court burn (lane D's share only, B1:
``cascade.audit.checks.lane_d_pay``); every other failure is an alert that
is logged (cascade-ops tag ``lane_d_check_alert``) while the signed court
stays paid. :func:`lane_d_findings` returns every finding, never stopping
at an alert; :func:`check_lane_d` keeps its one-result form, ``<code>:
<detail>`` of the first. rev6.18 B2: this is the ONE lane D checker for
outside validators; the former satellite replay's rules run inside it
(per report: the court and ``king_after`` records; chain mode, ``chain=``:
:mod:`cascade.audit.lane_d_chain`; ``bench_rows=``: W3b6b).

W14 (the lane D-only fast kill): :func:`check_lane_d_absent` re-derives a
settlement with no lane D report from its absent receipt record (the
structure rules re-implemented here, never imported from the producer). A
satellite burns lane D's share, and only lane D's, on its ``PASS``.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..eval.bootstrap import cohort_maxt_lcbs
from .checks import CheckResult, _fail, _ok, _warn

NAME = "lane_d"
SCHEMA = "lane-d-report/1"
# W14: the absent lane D receipt record (README w14 "The absent record"),
# re-stated from the spec: status is MarketCast's REPORT_ABSENT ("absent") and
# the typed reasons include its INFRA_DEFERRED ("infra_deferred"). Literals,
# not imports: this module's import allowlist is fixed (invariant 12, W9).
ABSENT_NAME = "lane_d_absent"
RECEIPT_SCHEMA = "lane-d-receipt/1"
ABSENT_STATUS = "absent"
ABSENT_REASONS = ("report_absent", "infra_deferred")
ABSENT_KEYS = ("schema", "status", "boundary", "reason", "prior_state_sha256",
               "posterior_state_sha256")
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
            "min_long_series": int(cfg.min_long_series),
            # rev6.10 (a) / rev6.12 (b) (W2e): the embargo cap, the W5j
            # settlement budget and the minimum fresh share.
            "embargo_hours": float(cfg.embargo_hours),
            "max_windows_per_settlement": int(cfg.max_windows_per_settlement),
            "min_fresh_window_share": float(cfg.min_fresh_window_share)}


def _body_digest(report: dict) -> str:
    """sha256 hex of the report's canonical body (every field but
    ``signature``), the digest ``held_from.report_sha256`` names."""
    return hashlib.sha256(_canon({k: v for k, v in report.items() if k != "signature"})
                          ).hexdigest()


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


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
        out.append((float(lcb), float(gain), bool(lcb >= _report_bar(report, cfg)
                                                  and gain > 0.0)))
    return out


def crown_bar_of(cfg, tenure: int) -> float:
    """rev6.23, re-derived from the parsed ``[lane_d]`` keys (never the
    producer's): ``d_margin_start`` at tenure 0, falling linearly to
    ``d_margin`` at ``d_margin_warmup_settlements`` and flat after."""
    w = int(cfg.d_margin_warmup_settlements)
    t = max(0, int(tenure))
    start, end = float(cfg.d_margin_start), float(cfg.d_margin)
    if t >= w:
        return end
    return start - (start - end) * t / w


def _crown_bar_problem(report: dict, cfg) -> str | None:
    """rev6.23: ``crown_bar`` is ``{tenure, margin}`` with ``margin`` the bar
    re-derived from the config at ``tenure`` (the tenure itself is a chain
    rule: :func:`_continuity_problem`)."""
    cb = report.get("crown_bar")
    if not isinstance(cb, dict) or set(cb) != {"tenure", "margin"}:
        return f"crown_bar {cb!r} is not {{tenure, margin}}"
    t = cb["tenure"]
    if not _int_value(t) or t < 0:
        return f"crown_bar.tenure {t!r} is not an int >= 0"
    want = crown_bar_of(cfg, t)
    if not _close(cb["margin"], want):
        return f"crown_bar.margin {cb['margin']!r} != {want!r} re-derived at tenure {t}"
    return None


def _report_bar(report: dict, cfg) -> float:
    """The bar a verdict clears against: re-derived at the report's stated
    tenure (checked by :func:`_crown_bar_problem` before any cohort)."""
    return crown_bar_of(cfg, int(report["crown_bar"]["tenure"]))


def next_tenure(prior: dict) -> int | None:
    """rev6.23: the tenure the settlement after ``prior`` must state: 1
    after a crown (a miner crowned, or the house retaking from a miner), else
    the prior tenure + 1; ``None`` when ``prior`` carries no ``crown_bar``."""
    cb = prior.get("crown_bar")
    if not isinstance(cb, dict) or not _int_value(cb.get("tenure")):
        return None
    kb, ka = prior.get("king_before") or {}, prior.get("king_after") or {}
    retake = bool(ka.get("house")) and not bool(kb.get("house"))
    return 1 if (prior.get("crowned") is not None or retake) else int(cb["tenure"]) + 1


def _bars(kd: str, hd: str, members) -> list[str]:
    """The bars a cohort is judged against: KING_D, plus the house as a
    second bar when a miner holds KING_D and the house is not the
    challenger (W5f: the house is never its own second bar)."""
    return [kd] if kd == hd or hd in members else [kd, hd]


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


def _stage_problem(report: dict, cfg, cohort: dict, v: dict) -> str | None:
    """Why the cohort's stage cannot hold for verdict ``v``, or ``None``.
    rev6.11 (b): every judgement is the first (and only) verdict, against
    ``king_before``; there is no confirmation stage and no confirmation
    block."""
    stage, kd = cohort["stage"], cohort["king_digest"]
    kb = report["king_before"]["digest"]
    if stage != "first" or v.get("stage") != "first":
        return f"stage {stage!r}/{v.get('stage')!r}: only first verdicts exist (rev6.11 (b))"
    if "confirmation" in v:
        return "a verdict carries a confirmation block (rev6.11 (b): none exists)"
    if kd != kb:
        return f"stage first judged against {str(kd)[:12]}, not king_before {str(kb)[:12]}"
    return None


def _lowfreq_problem(cfg, v: dict) -> tuple[str | None, bool]:
    """W3b5 (A10) for a cleared first verdict: ``(problem, is_fail)``. A
    ``pass`` / ``fail`` block carries four finite LCBs and its result is
    re-derived from them (``fail`` iff one is below ``-tau_public``); a
    ``fail`` is ``lost``. ``no_windows`` (and an unevaluated block) leaves
    the candidate to the gate."""
    lf = v.get("lowfreq")
    if not isinstance(lf, dict) or lf.get("result") not in ("pass", "fail"):
        return None, False
    lcb = lf.get("lcb")
    if not isinstance(lcb, dict) or set(lcb) != set(GATE_REFS) or any(
            not isinstance(lcb[r], dict) or set(lcb[r]) != set(_GATE_METRICS)
            or not all(_finite(lcb[r][q]) for q in _GATE_METRICS) for r in GATE_REFS):
        return f"lowfreq {lf['result']} without four finite LCBs", False
    worst = min(float(lcb[r][q]) for r in GATE_REFS for q in _GATE_METRICS)
    want = "fail" if worst < -float(cfg.tau_public) else "pass"
    if lf["result"] != want:
        return (f"lowfreq {lf['result']}; its LCBs re-derive {want} "
                f"(worst {worst}, -tau_public={-float(cfg.tau_public)})"), False
    if want == "fail" and v["outcome"] != "lost":
        return f"lowfreq fail with outcome {v['outcome']}, not lost", True
    return None, want == "fail"


def _cohort_problems(report: dict, cfg, cohort: dict, rows_by: dict, windows: dict,
                     entries: dict, confirms: dict) -> list[str]:
    """The cohort's problems. Every member whose re-derived judgement clears
    (rev6.11 (b): a first clear is the crowning candidate) and does not fail
    A10 is added to ``confirms``; its outcome is the gate's to decide
    (:func:`_crown_problems`)."""
    out = []
    idx = int(cohort["index"])
    kd, members, wids = cohort["king_digest"], list(cohort["members"]), list(cohort["window_ids"])
    hd = str(report["house"]["digest"])
    # W5f: the house challenges a miner KING_D in its own cohort, with its T
    # from the report's house record (it is not a listed entry).
    house_challenger = hd in members
    if house_challenger and (members != [hd] or kd == hd):
        return [f"cohort {idx}: the house {hd[:12]} is a member, but not alone against a "
                f"miner KING_D"]
    ts = [int(cohort["king_reveal_ts"])]
    for m in members:
        if house_challenger:
            ts.append(int(report["house"]["reveal_ts"]))
            continue
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
        # rev6.10 (a) / rev6.12 (a), every cadence: eligible at the feed bound
        # b - e with the window's signed effective embargo e.
        e = 3600.0 * float(w["embargo_hours"]) if _finite(w.get("embargo_hours")) else None
        if e is None or not (int(w["min_target_ts"]) > bound - e
                             and int(w["min_first_seen_ts"]) + e > bound):
            out.append(f"cohort {idx}: window {wid} is not forward-only for max(T)={t_max} "
                       f"under its embargo")
    if len(wids) < int(cfg.min_windows):
        out.append(f"cohort {idx}: {len(wids)} windows < min_windows {cfg.min_windows}")
    clusters = [windows[w]["source"] for w in wids if w in windows]
    if len(set(clusters)) < int(cfg.min_clusters):
        out.append(f"cohort {idx}: {len(set(clusters))} clusters < min_clusters {cfg.min_clusters}")
    if out:
        return out
    # W5c change 3 (RV5): with a miner holding KING_D, the report's house is
    # a second bar. Whether it applies is re-derived, never read from the
    # house keys, so an omitted judgement cannot hide the bar. The house is
    # never its own second bar (W5f).
    bars = _bars(kd, hd, members)
    if len(bars) == 1:
        if "house_digest" in cohort:
            return [f"cohort {idx}: house_digest where the house holds KING_D or is the "
                    f"challenger"]
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
                           f"house holds KING_D or is the challenger")
        else:
            out += _house_problems(idx, m, v.get("house"), hd, judged[hd][i])
            cleared = bool(cleared and judged[hd][i][2])
        if bool(v["cleared"]) != cleared:
            out.append(f"cohort {idx}: {m[:12]} cleared={v['cleared']} != re-derived {cleared}")
        stage = cohort["stage"]
        why = _stage_problem(report, cfg, cohort, v)
        if why:
            out.append(f"cohort {idx}: {m[:12]} {why}")
            continue
        if v["stage"] != stage or v["king_digest"] != kd:
            out.append(f"cohort {idx}: {m[:12]} stage/king {v['stage']}/"
                       f"{str(v['king_digest'])[:12]} != cohort {stage}/{kd[:12]}")
            continue
        if cleared:
            why, lowfreq_fail = _lowfreq_problem(cfg, v)
            if why:
                out.append(f"cohort {idx}: {m[:12]} {why}")
            elif not lowfreq_fail:
                confirms[m] = v
            continue
        want = "lost"
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
# RV7 B2: a held candidate's result also names its hold (optional key).
_HELD_KEYS = {"boundary", "report_sha256", "set_sha256", "lcb", "king_d"}
# W3b6b: the optional bound bench records of a gate result.
_ROWS_ROLES = {"candidate", "king_d", "house"}


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
    if not isinstance(res, dict) or set(res) - {"held_from", "rows"} != _RESULT_KEYS:
        keys = sorted(res) if isinstance(res, dict) else type(res).__name__
        return f"gate result {i}: keys {keys}"
    if "rows" in res:
        # W3b6b: the bound bench records, three sha256 (verified against the published
        # bytes by check_gate_rows when bench_rows is given).
        rows = res["rows"]
        if not isinstance(rows, dict) or set(rows) != _ROWS_ROLES or not all(
                isinstance(v, str) and len(v) == 64 and set(v) <= _HEX for v in rows.values()):
            return f"gate result {i}: rows is not three sha256 digests"
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


_MISSING = object()


def _held_from_problem(report: dict, cfg, boundary: int, hf) -> str | None:
    """RV7 B2 (a), single report: why a held candidate's ``held_from`` cannot
    name its hold, or ``None``. Exactly the five keys; the hold settled at an
    earlier boundary on a KING_D LCB that cleared ``d_margin``; its KING_D
    is still ``king_before`` (no D20 rule: a king change drops a hold).
    Whether that hold exists is chain mode's to show."""
    if hf is _MISSING:
        return "no held_from: nothing binds it to a cleared, held first verdict"
    if not isinstance(hf, dict) or set(hf) != _HELD_KEYS:
        keys = sorted(hf) if isinstance(hf, dict) else type(hf).__name__
        return f"held_from keys {keys}"
    b = hf["boundary"]
    if isinstance(b, bool) or not isinstance(b, int) or b >= boundary:
        return f"held_from boundary {b!r} is not before {boundary}"
    for k in ("report_sha256", "set_sha256", "king_d"):
        if not _is_hex64(hf[k]):
            return f"held_from {k} {hf[k]!r} is not 64 lowercase hex digits"
    if not (_finite(hf["lcb"]) and float(hf["lcb"]) >= float(cfg.d_margin)):
        return f"held_from lcb {hf['lcb']!r} did not clear d_margin {cfg.d_margin}"
    if hf["king_d"] != report["king_before"]["digest"]:
        return "held_from king_d is not king_before (a king change drops a hold)"
    return None


def _crown_problems(report: dict, cfg, boundary: int, confirms: dict, entries: dict
                    ) -> tuple[str | None, list[str]]:
    """``(re-derived crown, problems)``. ``confirms`` maps every verdict
    whose re-derived judgement clears (a first clear that did not fail A10:
    a crowning candidate) to that verdict. W5e: the gate decides each
    candidate (``pass`` crowns or stays ``cleared``, ``gate_regression`` /
    ``checkpoint_fault`` is ``lost``, ``bench_hold`` stays ``cleared``);
    without a gate record every candidate is held and nothing crowns; the
    crown is the highest-LCB passing first clear, or a held candidate
    re-gated ``pass`` (it has no verdict this settlement)."""
    outcomes, out = _gate_problems(report, cfg, boundary)
    if out:
        return None, out
    gated = "gate" in report
    verdicts = {v["digest"]: v for v in report.get("verdicts") or []}
    house = report["house"]

    def reveal_ts(dg: str) -> int:
        # W5f: the house challenger's T is the report's house record's.
        return int(house["reveal_ts"]) if dg == house["digest"] else int(entries[dg]["reveal_ts"])

    for dg in confirms:
        if gated and dg not in outcomes:
            out.append(f"cleared candidate {dg[:12]} has no gate result")
    held = []
    for dg in outcomes:
        if dg in confirms:
            continue
        if dg in verdicts:
            out.append(f"gate result for {str(dg)[:12]}, whose verdict is not a cleared "
                       f"candidate")
        elif dg not in entries and dg != house["digest"]:
            out.append(f"gate result for {str(dg)[:12]}, neither a verdict, a listed entry "
                       f"nor the house")
        else:
            held.append(dg)
    results = {r["digest"]: r for r in report["gate"]["results"]} if gated else {}
    for dg in confirms:
        if "held_from" in results.get(dg, {}):
            out.append(f"{dg[:12]}: held_from on a result judged this settlement")
    for dg in held:
        why = _held_from_problem(report, cfg, boundary, results[dg].get("held_from", _MISSING))
        if why:
            out.append(f"held result {str(dg)[:12]}: {why}")
    if out:
        return None, out
    passing = sorted((v for dg, v in confirms.items() if outcomes.get(dg) == "pass"),
                     key=lambda v: (-float(v["lcb"]), reveal_ts(v["digest"]), v["digest"]))
    # A held candidate's clear (its LCB) is not in this report, so when one
    # passes the crown is the best passing first clear or that candidate.
    allowed = [v["digest"] for v in passing[:1]] + [dg for dg in held if outcomes[dg] == "pass"]
    stated = report.get("crowned")
    if stated is None and house["digest"] in allowed and report["king_after"].get("house") \
            and report["king_after"]["digest"] == house["digest"]:
        # W5f: a house retake names no miner crown (``crowned`` is null).
        crown = house["digest"]
    else:
        crown = stated if stated in allowed else (allowed[0] if allowed else None)
    for dg, v in confirms.items():
        o = outcomes.get(dg)
        if o in (None, "bench_hold"):
            want = "cleared"
        elif o in ("gate_regression", "checkpoint_fault"):
            want = "lost"
        else:
            want = "crowned" if dg == crown else "cleared"
        if v["outcome"] != want:
            out.append(f"{dg[:12]}: outcome {v['outcome']} != re-derived {want} "
                       f"(gate {o or 'absent'})")
    return crown, out


# ── chain mode (RV7 B2): the provenance shown by earlier signed reports ──────


def _verified_priors(prior_reports, signer_hotkey: str) -> list[tuple[int, str, dict]]:
    """``(boundary, body digest, report)`` for each prior report whose
    signature verifies under ``signer_hotkey``; anything else is not
    evidence. Raises ``SignatureVerifierUnavailable`` like :func:`_signature`."""
    out = []
    for p in prior_reports:
        if not isinstance(p, dict) or _signature(p, signer_hotkey):
            continue
        b = p.get("boundary")
        if isinstance(b, bool) or not isinstance(b, int):
            continue
        try:
            dg = _body_digest(p)
        except (TypeError, ValueError):
            continue
        out.append((b, dg, p))
    return out


def _hold_problem(priors, res: dict) -> str | None:
    """Why no verified earlier report shows ``res["held_from"]``: the report
    at that boundary with that body digest, where the digest's first verdict
    cleared (outcome ``cleared``) with that LCB, KING_D and set, and its gate
    record is absent or holds it (``bench_hold``)."""
    hf, dg = res["held_from"], res["digest"]
    p = next((p for b, d, p in priors
              if b == hf["boundary"] and d == hf["report_sha256"]), None)
    if p is None:
        return (f"{str(dg)[:12]}: no verified report at boundary {hf['boundary']} with body "
                f"digest {hf['report_sha256'][:12]} (the hold is not in the chain)")
    for pv in p.get("verdicts") or []:
        if not isinstance(pv, dict) or pv.get("digest") != dg:
            continue
        cohorts = p.get("cohorts") or []
        ci = pv.get("cohort")
        c = cohorts[ci] if isinstance(ci, int) and 0 <= ci < len(cohorts) else None
        if pv.get("stage") == "first" and pv.get("cleared") is True \
                and pv.get("outcome") == "cleared" and pv.get("lcb") == hf["lcb"] \
                and pv.get("king_digest") == hf["king_d"] and isinstance(c, dict) \
                and c.get("stage") == "first" and c.get("set_sha256") == hf["set_sha256"]:
            break
    else:
        return f"{str(dg)[:12]}: held_from does not match a cleared, held first verdict"
    pg = p.get("gate")
    if pg is not None and not any(isinstance(r, dict) and r.get("digest") == dg
                                  and r.get("outcome") == "bench_hold"
                                  for r in (pg.get("results") if isinstance(pg, dict)
                                            else None) or []):
        return f"{str(dg)[:12]}: held_from's report gated it without a bench_hold"
    return None


def _chain_problems(report: dict, prior_reports, signer_hotkey: str,
                    entries: dict) -> str | None:
    """Chain mode: every ``held_from`` is shown by a verified earlier
    report, and the report continues the chain
    (:func:`_continuity_problem`), else why not."""
    priors = _verified_priors(list(prior_reports), signer_hotkey)
    why = _continuity_problem(report, priors, entries)
    if why:
        return why
    for res in (report.get("gate") or {}).get("results") or []:
        if "held_from" in res:
            why = _hold_problem(priors, res)
            if why:
                return why
    return None


def _sentinel_problem(report: dict, era_sentinel) -> str | None:
    """``gate.sentinel`` against the era's W-S1 record (``None``: unarmed)."""
    if era_sentinel is not None and (not isinstance(era_sentinel, dict)
                                     or not isinstance(era_sentinel.get("suite"), str)):
        return f"era_sentinel {era_sentinel!r} is neither None nor a W-S1 record"
    gate = report.get("gate")
    if not isinstance(gate, dict):
        return None
    want = None if era_sentinel is None else era_sentinel["suite"]
    if gate.get("sentinel") != want:
        return f"gate.sentinel {gate.get('sentinel')!r} is not the era's W-S1 suite {want!r}"
    return None


# ── W9: the judged set, the caps and the registration rules ─────────────────

OPEN_STATUSES = ("pending", "hold")
_CAP_SKIP = "coldkey_cap_settlement"
# An entry selected and then dropped (the floors, an unscorable king, a
# malformed forecast) counts against the caps like a judged one, as in the
# producer's selection.
_SELECTED_SKIPS = ("pending_floor", "king_unscored", "checkpoint_fault")
# The signed inputs of an entry record: fixed once registered.
_INPUT_KEYS = ("hotkey", "coldkey", "context", "ready_block", "reveal_block", "reveal_ts")


def _int_value(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _registration_problem(cfg, dg: str, e: dict) -> str | None:
    """The registration rules a signed entry record shows, or ``None``."""
    ck = e.get("coldkey")
    if not isinstance(ck, str) or not ck:
        return f"{dg[:12]}: coldkey {ck!r} is unresolved (coldkey_unresolved), never selected"
    ctx = e.get("context")
    if not _int_value(ctx) or ctx not in [int(c) for c in cfg.contexts]:
        return f"{dg[:12]}: context {ctx!r} is not armed (context_not_armed)"
    ready, reveal = e.get("ready_block"), e.get("reveal_block")
    if not (_int_value(ready) and _int_value(reveal)):
        return f"{dg[:12]}: ready_block / reveal_block {ready!r} / {reveal!r} are not blocks"
    if reveal - ready > int(cfg.ready_ttl_blocks):
        return (f"{dg[:12]}: revealed {reveal - ready} blocks after ready > ready_ttl_blocks "
                f"{int(cfg.ready_ttl_blocks)} (ready_expired)")
    if not _int_value(e.get("reveal_ts")):
        return f"{dg[:12]}: reveal_ts {e.get('reveal_ts')!r} is not a unix time"
    return None


def _judged_set_problems(report: dict, cfg, entries: dict) -> list[str]:
    """README "Judged set" rules 1-5: ``judged`` is the verdicts, every skip
    is a listed entry's typed reason, no open entry is silently omitted, and
    the caps re-derived FIFO by ``(reveal_ts, digest)`` over the selection
    candidates name exactly the ``coldkey_cap_settlement`` skips."""
    verdicts = list(report.get("verdicts") or [])
    vdigests = [v["digest"] for v in verdicts]
    judged = report.get("judged")
    if not isinstance(judged, list) or judged != vdigests:
        return [f"judged {judged!r} is not the verdicts' digests {vdigests}"]
    if len(set(judged)) != len(judged):
        return ["judged lists a digest twice"]
    members = [m for c in report.get("cohorts") or [] for m in c["members"]]
    if sorted(members) != sorted(judged):
        return [f"the verdicts {sorted(judged)} are not exactly the cohort members "
                f"{sorted(members)}"]
    judged_set = set(judged)
    house = report["house"]["digest"]
    skipped = report.get("skipped")
    if not isinstance(skipped, list):
        return [f"skipped is {type(skipped).__name__}, not a list"]
    skip_of: dict[str, str] = {}
    for i, s in enumerate(skipped):
        if not isinstance(s, dict) or set(s) != {"digest", "reason"}:
            return [f"skipped[{i}] is not a {{digest, reason}} record"]
        dg = s["digest"]
        if dg not in entries:
            return [f"skipped {str(dg)[:12]} is not a listed entry"]
        if dg in skip_of:
            return [f"skipped {str(dg)[:12]} appears twice"]
        if dg in judged_set and not (dg == house and s["reason"] == "duplicate_digest"):
            # W5f: a miner copy of the house is refused at dedup while the
            # house itself (same digest, not an entry) is judged.
            return [f"{str(dg)[:12]} is both judged and skipped"]
        if entries[dg].get("reason") != s["reason"]:
            return [f"skipped {str(dg)[:12]} reason {s['reason']!r} != its entry's reason "
                    f"{entries[dg].get('reason')!r}"]
        skip_of[dg] = s["reason"]
    gated = {r["digest"] for r in ((report.get("gate") or {}).get("results") or [])
             if isinstance(r, dict) and "digest" in r}
    out = []
    for dg in sorted(entries):
        if entries[dg].get("status") in OPEN_STATUSES and dg not in judged_set \
                and dg not in skip_of and dg not in gated:
            out.append(f"open entry {dg[:12]} ({entries[dg].get('status')}) is neither judged, "
                       f"skipped nor gated: silently omitted")
    if out:
        return out
    held = gated - judged_set
    # W5f: the house challenger is not a listed entry and is never selected
    # under the caps (it has no coldkey); only miner entries are candidates.
    cands = [dg for dg in judged if dg not in held and dg != house]
    cands += [dg for dg, why in skip_of.items()
              if why in (_CAP_SKIP, *_SELECTED_SKIPS) and dg not in held]
    for dg in cands:
        why = _registration_problem(cfg, dg, entries[dg])
        if why:
            out.append(why)
    if out:
        return out
    cap_all = int(cfg.max_entries_per_settlement)
    cap_cold = int(cfg.max_judged_per_coldkey_per_settlement)
    n, per_cold = 0, {}
    for dg in sorted(cands, key=lambda d: (int(entries[d]["reveal_ts"]), d)):
        ck = entries[dg]["coldkey"]
        over = n >= cap_all or per_cold.get(ck, 0) >= cap_cold
        capped = skip_of.get(dg) == _CAP_SKIP
        if over and not capped:
            out.append(f"{dg[:12]} is over the caps (FIFO by T: {n} selected, {per_cold.get(ck, 0)} "
                       f"on its coldkey) but not skipped {_CAP_SKIP}")
        elif capped and not over:
            out.append(f"{dg[:12]} is skipped {_CAP_SKIP} but its coldkey and the settlement "
                       f"had room (FIFO by T)")
        if not over:
            n += 1
            per_cold[ck] = per_cold.get(ck, 0) + 1
    return out


def _clock_problems(report: dict, cfg, windows: dict) -> list[str]:
    """README rules 6-7: each judged window's mirror lag is within the limit
    (null = unmeasured), and a cohort judged against ``king_before`` or the
    house states that record's T."""
    out = []
    lim = float(cfg.max_mirror_lag_hours)
    named = [report["king_before"], report["house"]]
    for c in report.get("cohorts") or []:
        idx = c["index"]
        for wid in c["window_ids"]:
            lag = windows[wid].get("mirror_lag_hours")
            if lag is not None and not (_finite(lag) and float(lag) <= lim):
                out.append(f"cohort {idx}: window {wid} mirror_lag_hours {lag!r} > "
                           f"max_mirror_lag_hours {lim}")
        for rec in named:
            if c["king_digest"] == rec["digest"] and c["king_reveal_ts"] != rec["reveal_ts"]:
                out.append(f"cohort {idx}: king_reveal_ts {c['king_reveal_ts']!r} is not the "
                           f"named king {str(rec['digest'])[:12]}'s reveal_ts {rec['reveal_ts']!r}")
    return out


CONFIRMATION_STAGE_PRESENT = "confirmation_stage_present"
_DRAW_SEED_TAG = "lane_d_windows"
_FREQ_CLASS = {"D": "daily", "W": "weekly", "M": "monthly", "ME": "monthly", "MS": "monthly",
               "Q": "quarterly", "QE": "quarterly", "QS": "quarterly",
               "A": "yearly", "Y": "yearly", "YE": "yearly"}


def combined_log_ratio(r_fwd: float, r_bt: float, *, n_fwd: int, n_bt: int, cap: float,
                       floor: float) -> float:
    """rev6.9 (W5i) combination, re-derived for the audit: ``(1 - c) log r_fwd
    + c log clamp(r_bt, floor, 1/floor)``, ``c = min(cap, n_bt / (n_bt +
    n_fwd))`` (0 without backtest windows: exactly ``log r_fwd``)."""
    c = 0.0 if int(n_bt) == 0 else min(float(cap), int(n_bt) / (int(n_bt) + int(n_fwd)))
    if c <= 0.0:
        return math.log(float(r_fwd))
    clamped = min(max(float(r_bt), float(floor)), 1.0 / float(floor))
    return (1.0 - c) * math.log(float(r_fwd)) + c * math.log(clamped)


def _confirmation_problem(report: dict) -> str | None:
    """rev6.11 (b): there is no confirmation stage; a report carrying one (a
    cohort or verdict stage other than ``first``, a verdict ``confirmation``
    block, an entry ``confirm_pending``) is refused."""
    for key in ("cohorts", "verdicts"):
        for i, x in enumerate(report.get(key) or []):
            if isinstance(x, dict) and (("stage" in x and x["stage"] != "first")
                                        or (key == "verdicts" and "confirmation" in x)):
                return f"{CONFIRMATION_STAGE_PRESENT}: {key}[{i}] carries a confirmation stage"
    for i, e in enumerate(report.get("entries") or []):
        if isinstance(e, dict) and "confirm_pending" in (e.get("status"), e.get("reason")):
            return f"{CONFIRMATION_STAGE_PRESENT}: entries[{i}] is confirm_pending"
    return None


def _geomean_of(rows: list) -> float | None:
    if not rows:
        return None
    lc = sum(math.log(max(float(r["crps"]), 1e-12)) for r in rows) / len(rows)
    lm = sum(math.log(max(float(r["mase"]), 1e-9)) for r in rows) / len(rows)
    return math.sqrt(math.exp(lc) * math.exp(lm))


def _embargo_problems(report: dict, cfg, windows: dict, rows_by: dict) -> list[str]:
    """rev6.10 (a) / rev6.12 (a)/(b) / W5j, re-derived: each judged window's
    embargo record and class at its cohort's bound, each cohort's fresh /
    embargo split and the minimum fresh share, each verdict's sub-scores
    from the rows, the settlement budget (the draw record: :func:`_draw_problems`)."""
    out: list[str] = []
    cap, h = float(cfg.embargo_hours), int(cfg.horizon)
    margin = 3600.0 * float(cfg.forward_margin_hours)
    for wid, w in windows.items():
        e = w.get("embargo_hours")
        if not _finite(e) or isinstance(e, bool) or float(e) < 0 or float(e) > cap:
            out.append(f"window {wid}: embargo_hours {e!r} is not in [0, {cap}]")
            continue
        e = float(e)
        if w.get("rule") != ("embargo" if e > 0 else "embargo0"):
            out.append(f"window {wid}: rule {w.get('rule')!r} for e = {e}")
        if not isinstance(w.get("min_release_ts"), int) or isinstance(w.get("min_release_ts"), bool):
            out.append(f"window {wid}: min_release_ts {w.get('min_release_ts')!r} is not a timestamp")
        freq = w.get("freq")
        if w.get("timeframe") != freq or w.get("freq_class") != _FREQ_CLASS.get(freq, "subdaily") \
                or w.get("kind") != "forward" or w.get("bin") != "forward" or w.get("h") != h \
                or wid.rsplit(":", 2)[1:] != [str(w.get("origin")), f"h{h}"]:
            out.append(f"window {wid}: the draw cell (timeframe/freq_class/kind/bin/h/origin) "
                       f"does not match the window")
    share = float(cfg.min_fresh_window_share)
    judged: set[str] = set()
    for c in report.get("cohorts") or []:
        idx, wids = c["index"], [x for x in c["window_ids"] if x in windows]
        judged.update(wids)
        bound = float(int(c["t_max"])) + margin
        n_emb = 0
        flag = {}
        for wid in wids:
            w = windows[wid]
            fresh = int(w["min_target_ts"]) > bound and int(w["min_first_seen_ts"]) > bound
            flag[wid] = not fresh
            if w.get("embargo_window") is fresh:
                out.append(f"cohort {idx}: window {wid} embargo_window "
                           f"{w.get('embargo_window')!r}; re-derived {not fresh}")
            n_emb += 0 if fresh else 1
        if c.get("embargo_windows") != n_emb or c.get("fresh_windows") != len(wids) - n_emb:
            out.append(f"cohort {idx}: fresh/embargo {c.get('fresh_windows')!r}/"
                       f"{c.get('embargo_windows')!r}; re-derived {len(wids) - n_emb}/{n_emb}")
        if wids and (len(wids) - n_emb) < share * len(wids):
            out.append(f"cohort {idx}: fresh_share_short: {len(wids) - n_emb} fresh of "
                       f"{len(wids)} < {share}")
        for v in report.get("verdicts") or []:
            if int(v["cohort"]) != int(idx):
                continue
            for name, emb in (("fresh", False), ("embargo", True)):
                rs = [rows_by[(int(idx), x, v["digest"])] for x in wids
                      if flag[x] is emb and (int(idx), x, v["digest"]) in rows_by]
                st = v.get(name)
                want = _geomean_of(rs)
                if not isinstance(st, dict) or st.get("n") != len(rs) or (
                        (want is None) is not (st.get("geomean") is None)) or (
                        want is not None and not math.isclose(float(st["geomean"]), want,
                                                              rel_tol=1e-9)):
                    out.append(f"verdict {str(v['digest'])[:12]}: {name} sub-score is not its "
                               f"rows'")
    if len(judged) > int(cfg.max_windows_per_settlement):
        out.append(f"{len(judged)} judged windows > max_windows_per_settlement "
                   f"{cfg.max_windows_per_settlement}")
    return out


def _draw_problems(report: dict, cfg, windows: dict) -> list[str]:
    """W5j: the draw record is bound to its block hash (the gate's, when the
    gate ran), the budget and the cohorts; per cell the quota is the cohort's
    judged windows of that cell, at most its capacity."""
    cohorts = report.get("cohorts") or []
    gate = report.get("gate")
    if "draw" not in report:
        return ["a gated settlement with judged cohorts has no draw record"] \
            if cohorts and isinstance(gate, dict) else []
    d = report["draw"]
    out = []
    if not isinstance(d, dict) or not cohorts:
        return ["draw record without a drawn cohort"]
    bh = d.get("boundary_block_hash")
    if isinstance(gate, dict) and gate.get("block_hash") != bh:
        out.append(f"draw boundary_block_hash {bh!r} is not the gate's block hash")
    if d.get("seed") != f"{bh}:{_DRAW_SEED_TAG}":
        out.append(f"draw seed {d.get('seed')!r} is not <boundary_block_hash>:{_DRAW_SEED_TAG}")
    if d.get("max_windows_per_settlement") != int(cfg.max_windows_per_settlement):
        out.append("draw max_windows_per_settlement is not the config's")
    dcs = d.get("cohorts")
    if not isinstance(dcs, list) or len(dcs) != len(cohorts):
        return out + ["draw cohorts do not match the report's cohorts"]
    for dc, c in zip(dcs, cohorts, strict=True):
        if not isinstance(dc, dict) or dc.get("index") != c["index"]:
            out.append(f"draw cohort {c['index']}: index mismatch")
            continue
        count: dict[str, int] = {}
        for x in c["window_ids"]:
            w = windows.get(x)
            if w is not None:
                k = f"{w.get('timeframe')}|{w.get('bin')}"
                count[k] = count.get(k, 0) + 1
        quota, capacity = dc.get("quota") or {}, dc.get("capacity") or {}
        for k in sorted(set(quota) | set(count) | set(capacity)):
            if quota.get(k, 0) != count.get(k, 0) or quota.get(k, 0) > capacity.get(k, 0):
                out.append(f"draw cohort {c['index']}: quota {k} {quota.get(k)!r} is not its "
                           f"{count.get(k, 0)} judged windows (capacity {capacity.get(k)!r})")
    return out


def _row_order_problem(report: dict, rows: list) -> str | None:
    """README rule 8: the rows in the canonical order (cohorts by index, each
    cohort's ``window_ids`` in order, per window ``[king, house?, *members]``)."""
    want = []
    for c in sorted(report.get("cohorts") or [], key=lambda c: int(c["index"])):
        models = [c["king_digest"], *([c["house_digest"]] if "house_digest" in c else []),
                  *c["members"]]
        want += [(int(c["index"]), str(w), str(m)) for w in c["window_ids"] for m in models]
    got = [(int(r["cohort"]), str(r["window_id"]), str(r["digest"])) for r in rows]
    if got != want:
        at = next((i for i, (g, w) in enumerate(zip(got, want, strict=False)) if g != w),
                  min(len(got), len(want)))
        return f"rows are not in the canonical order (first difference at row {at})"
    return None


def _continuity_problem(report: dict, priors, entries: dict) -> str | None:
    """README chain rules 9-11 over the verified prior reports below this
    boundary: KING_D continues from the latest one (or, after a house holder,
    from this report's refreshed house), a house digest keeps its record,
    and an entry's recorded inputs never change."""
    boundary = int(report["boundary"])
    earlier = [(b, p) for b, _dg, p in priors if b < boundary]
    if not earlier:
        return None
    latest = max(b for b, _ in earlier)
    kb, house = report["king_before"], report["house"]
    for b, p in earlier:
        if b != latest:
            continue
        ka = p.get("king_after")
        if kb == ka:
            continue
        if isinstance(ka, dict) and ka.get("house") is True and kb == house:
            continue
        return (f"king_before {str(kb.get('digest'))[:12]} is not where the report at {b} "
                f"left KING_D ({str((ka or {}).get('digest'))[:12]})")
    # rev6.23: KING_D's tenure continues from the latest earlier report.
    want_t = next_tenure(next(p for b, p in earlier if b == latest))
    got_t = (report.get("crown_bar") or {}).get("tenure")
    if want_t is not None and got_t != want_t:
        return (f"crown_bar.tenure {got_t!r} does not continue the report at {latest} "
                f"(want {want_t})")
    for b, p in sorted(earlier, key=lambda x: x[0]):
        ph = p.get("house")
        if isinstance(ph, dict) and ph.get("digest") == house["digest"] and ph != house:
            return (f"the house {str(house['digest'])[:12]} differs from its record in the "
                    f"report at {b}")
        for e in p.get("entries") or []:
            if not isinstance(e, dict) or e.get("digest") not in entries:
                continue
            mine = entries[e["digest"]]
            for k in _INPUT_KEYS:
                if e.get(k) != mine.get(k):
                    return (f"{str(e['digest'])[:12]}: {k} {mine.get(k)!r} differs from the "
                            f"report at {b} ({e.get(k)!r})")
    return None


# ── W9 (rev6.1): window eligibility from the judged snapshot itself ─────────
#
# §1.3 re-implemented from W5a's README (invariant 12: never through
# ``cascade.validator.lane_d_forward``), with its labels and precedence.

_SNAPSHOT_FILES = ("metadata.json", "hist.npz", "tgt.npz")


def _feed_key(series: dict):
    return series.get("feed") or series.get("source")


def _feed_embargo(feed: dict, cfg) -> float | None:
    """rev6.10 (a): the effective embargo of a listed feed, ``min(its pinned
    embargo_hours, [lane_d] embargo_hours)``; None (fail closed) when the
    value is absent, not a number (a bool is not one), negative or not
    finite."""
    v = feed.get("embargo_hours")
    if isinstance(v, bool) or not _finite(v) or float(v) < 0:
        return None
    return min(float(v), float(cfg.embargo_hours))


def window_refusal(series: dict, feeds: dict, bound: float, cfg) -> str | None:
    """The first failing §1.3 rule's label for one ``pool_d/1`` series
    record at ``bound`` (``t_max + 3600 * forward_margin_hours``), or
    ``None`` when the window is eligible. Missing metadata fails closed: an
    unlisted feed (or no ``feeds`` block) is ``unlisted_feed``; an absent,
    ``null``, empty or malformed ``frontier`` is ``absent_from_s0``."""
    h = int(cfg.horizon)
    ts, seen, end = series.get("target_ts"), series.get("first_seen_ts"), series.get("end_ts")
    if not (isinstance(ts, list) and isinstance(seen, list) and h > 0
            and len(ts) == h and len(seen) == h
            and all(_finite(t) for t in ts) and all(_finite(t) for t in seen)
            and _finite(end) and float(end) == float(ts[-1])):
        return "missing_clock"
    key = _feed_key(series)
    feed = feeds.get(key) if isinstance(feeds, dict) and isinstance(key, str) else None
    if not isinstance(feed, dict):
        return "unlisted_feed"
    # rev6.10 (a) / rev6.12 (a), every cadence: e = min(the snapshot's pinned
    # feed embargo_hours, [lane_d] embargo_hours); the clock, S0 and backfill
    # rules are judged at the feed bound b - e. A listed feed without a valid
    # embargo fails closed (W5a row).
    e = _feed_embargo(feed, cfg)
    if e is None:
        return "unlisted_feed"
    bound = float(bound) - 3600.0 * e
    if bool(feed.get("forecast_product", False)):
        return "forecast_product"
    lag = feed.get("mirror_lag_hours")
    if lag is not None and not (_finite(lag) and float(lag) <= float(cfg.max_mirror_lag_hours)):
        return "over_lag"
    data_ok = min(float(t) for t in ts) > bound
    seen_ok = min(float(t) for t in seen) > bound
    if not data_ok and not seen_ok:
        return "before_margin"
    if not seen_ok:
        return "publish_ahead"
    if not data_ok:
        return "late_mirror"
    frontier = series.get("frontier")
    after = []
    for rec in frontier if isinstance(frontier, list) else ():
        if not (isinstance(rec, (list, tuple)) and len(rec) == 2
                and _finite(rec[0]) and _finite(rec[1])):
            return "absent_from_s0"
        if float(rec[0]) > bound:
            after.append((float(rec[0]), float(rec[1])))
    if not after:
        return "absent_from_s0"
    if min(float(t) for t in ts) <= min(after)[1]:
        return "backfill"
    return None


# rev6.18 A4 (F2): the multichannel window's channel pick, re-stated from the
# spec (plan rev6.18 A4; tests/accept/w9/README.md "Multichannel windows"),
# never imported from the producer (invariant 12).
CHANNEL_SEED_TAG = "lane_d_channels"
CHANNEL_FIELDS = ("series_id", "n_channels_raw", "empty_channels", "n_channels", "channels",
                  "channels_sha256")


def channel_pick(block_hash, series_id: str, candidates, max_channels: int) -> list[int]:
    """The scored channels: every candidate when at most ``max_channels``;
    else the ``max_channels`` with the smallest ``sha256(seed ‖ ":" ‖ c)``,
    ``seed = sha256(block_hash ‖ series_id ‖ "lane_d_channels")``, ascending;
    ``[]`` for an over-cap set without a block hash."""
    cand = sorted({int(c) for c in candidates})
    if len(cand) <= int(max_channels):
        return cand
    if block_hash is None:
        return []
    seed = hashlib.sha256(f"{block_hash}{series_id}{CHANNEL_SEED_TAG}".encode()).hexdigest()
    ranked = sorted(cand, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    return sorted(ranked[:int(max_channels)])


def _channel_problems(report: dict, windows: dict, max_channels) -> list[str]:
    """F2: every judged multichannel window's recorded pick is the one the
    rule gives for its raw channel count, its dropped channels, the
    settlement's block hash (the draw's) and ``[generator] max_channels``."""
    draw = report.get("draw")
    bh = draw.get("boundary_block_hash") if isinstance(draw, dict) else None
    out = []
    for c in report.get("cohorts") or []:
        for wid in c["window_ids"]:
            w = windows.get(wid)
            if w is None or not any(k in w for k in CHANNEL_FIELDS):
                continue
            if any(k not in w for k in CHANNEL_FIELDS):
                out.append(f"window {wid}: a multichannel window lacks a channel field")
                continue
            if max_channels is None:
                out.append(f"window {wid}: multichannel, but no [generator] max_channels to "
                           f"re-derive its pick")
                continue
            raw, empty = int(w["n_channels_raw"]), {int(x) for x in w["empty_channels"]}
            want = channel_pick(bh, str(w["series_id"]),
                                [i for i in range(raw) if i not in empty], int(max_channels))
            if not want:
                out.append(f"window {wid}: no channel can be scored, yet it was judged")
                continue
            body = json.dumps(want, separators=(",", ":")).encode()
            if ([int(x) for x in w["channels"]] != want or int(w["n_channels"]) != len(want)
                    or w["channels_sha256"] != hashlib.sha256(body).hexdigest()):
                out.append(f"window {wid}: channels {w['channels']!r} are not the re-derived "
                           f"pick {want!r}")
    return out


def _snapshot_channels_problem(z, s: dict, w: dict) -> str | None:
    """F2 with the snapshot (z its open hist.npz): the window's
    multichannel record matches the stored series (its id and channel
    count), every scored channel holds a finite value, and a window recorded
    univariate is not a C ≥ 2 series."""
    sid = str(s["series_id"])
    try:
        hist = np.asarray(z[sid], dtype=np.float64)
    except (OSError, KeyError, ValueError) as e:
        return f"window {w['window_id']}: history unreadable ({type(e).__name__})"
    multi = hist.ndim == 2 and hist.shape[0] > 1
    if not any(k in w for k in CHANNEL_FIELDS):
        return (f"window {w['window_id']}: a {hist.shape[0]}-channel series recorded as "
                f"univariate") if multi else None
    if not multi or w.get("series_id") != sid or int(w["n_channels_raw"]) != hist.shape[0]:
        return (f"window {w['window_id']}: channel record (series {w.get('series_id')!r}, "
                f"{w.get('n_channels_raw')!r} channels) is not the snapshot's {sid!r} "
                f"{tuple(hist.shape)}")
    chans = [int(x) for x in w.get("channels") or []]
    if any(not 0 <= x < hist.shape[0] or not np.isfinite(hist[x]).any() for x in chans):
        return f"window {w['window_id']}: a scored channel is empty in the snapshot"
    return None


def _snapshot_windows_channels(d: Path, judged: list, by_id: dict) -> str | None:
    """:func:`_snapshot_channels_problem` over one snapshot's judged windows
    (its ``hist.npz`` opened once)."""
    try:
        with np.load(d / "hist.npz") as z:
            for w, _bound in judged:
                s = by_id.get(w["window_id"])
                if s is None:
                    continue
                why = _snapshot_channels_problem(z, s, w)
                if why is not None:
                    return why
    except (OSError, ValueError) as e:
        return f"snapshot {d.name}: hist.npz unreadable ({type(e).__name__})"
    return None


def _snapshot_digest(d: Path) -> str:
    parts = "".join(hashlib.sha256((d / n).read_bytes()).hexdigest() for n in _SNAPSHOT_FILES)
    return hashlib.sha256(parts.encode()).hexdigest()


def _snapshot_problem(report: dict, cfg, pool_dirs, windows: dict) -> str | None:
    """README rules 12-14 against the provided snapshot directories."""
    metas: dict[str, object] = {}
    dirs: dict[str, Path] = {}
    for d in pool_dirs:
        p = Path(d)
        try:
            dg = _snapshot_digest(p)
            meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metas.setdefault(dg, meta)
        dirs.setdefault(dg, p)
    judged: dict[str, list] = {}
    margin = 3600.0 * float(cfg.forward_margin_hours)
    for c in report.get("cohorts") or []:
        bound = float(int(c["t_max"])) + margin
        for wid in c["window_ids"]:
            w = windows[wid]
            sha = w.get("snapshot_sha256")
            if sha not in metas:
                return (f"window {wid} names snapshot {str(sha)[:12]}, which no provided "
                        f"pool directory holds")
            judged.setdefault(sha, []).append((w, bound))
    h = int(cfg.horizon)
    for sha in sorted(judged):
        meta = metas[sha]
        series = meta.get("series") if isinstance(meta, dict) else None
        if not isinstance(series, list):
            return f"snapshot {sha[:12]} has no series list"
        feeds = meta.get("feeds")
        feeds = feeds if isinstance(feeds, dict) else {}
        by_id = {}
        for s in series:
            if not isinstance(s, dict):
                return f"snapshot {sha[:12]}: a series record is not an object"
            sid = s.get("series_id")
            if not isinstance(s.get("frontier"), list):
                return f"snapshot {sha[:12]}: series {sid!r} has no frontier list"
            if _feed_key(s) not in feeds:
                return f"snapshot {sha[:12]}: series {sid!r} feed {_feed_key(s)!r} is not in feeds"
            ts = s.get("target_ts")
            if isinstance(sid, str) and isinstance(ts, list) and ts and _finite(ts[0]):
                by_id[f"{sid}:{int(ts[0])}:h{h}"] = s
        why = _snapshot_windows_channels(dirs[sha], judged[sha], by_id)
        if why is not None:
            return why
        for w, bound in judged[sha]:
            wid = w["window_id"]
            s = by_id.get(wid)
            if s is None:
                return f"window {wid} is not a series of snapshot {sha[:12]}"
            why = window_refusal(s, feeds, bound, cfg)
            if why is not None:
                return f"window {wid} is refused on its snapshot at bound {bound}: {why}"
            # rev6.10 (a): the snapshot's pinned e is at most the config's, a
            # publish-ahead series (a point seen before its data time) is
            # only protected at e = 0, and the window states that e.
            pinned = feeds[_feed_key(s)].get("embargo_hours")
            if float(pinned) > float(cfg.embargo_hours):
                return (f"window {wid}: its feed pins embargo_hours {pinned!r} above "
                        f"[lane_d] embargo_hours {cfg.embargo_hours}")
            e = _feed_embargo(feeds[_feed_key(s)], cfg)
            if e > 0.0 and any(float(t) > float(f) for t, f in zip(s["target_ts"],
                                                                  s["first_seen_ts"],
                                                                  strict=False)):
                return (f"window {wid}: a publish-ahead series (a point first seen before its "
                        f"data time) with a non-zero embargo {e}")
            if w.get("min_release_ts") != int(min(s["first_seen_ts"])) + int(round(3600.0 * e)):
                return (f"window {wid}: min_release_ts {w.get('min_release_ts')!r} is not the "
                        f"snapshot's first-seen + 3600 * {e}")
            if w.get("origin") != int(s["target_ts"][0]):
                return f"window {wid}: origin {w.get('origin')!r} is not the snapshot's target start"
            if not _finite(w.get("embargo_hours")) or float(w["embargo_hours"]) != e:
                return (f"window {wid}: embargo_hours {w.get('embargo_hours')!r} is not the "
                        f"snapshot's effective {e}")
            source = s.get("source") or s["series_id"]
            feed = s.get("feed") or source
            lag = feeds[_feed_key(s)].get("mirror_lag_hours")
            want = {"source": source, "feed": feed,
                    "min_target_ts": int(min(s["target_ts"])),
                    "min_first_seen_ts": int(min(s["first_seen_ts"])),
                    "mirror_lag_hours": None if lag is None else float(lag)}
            for k, v in want.items():
                if w.get(k) != v:
                    return f"window {wid}: {k} {w.get(k)!r} is not the snapshot's {v!r}"
    return None


# ── rev6.18 A1/A2/A6 (F3): the public-source draw, re-derived ───────────────
#
# Re-stated from the spec (tests/accept/w5k/README.md, w6p/README.md), never
# imported from the producer (invariant 12): the pinned sources, the draw
# constants, the index format and the draw itself. A mismatch is alert class:
# ``ld_public_index`` for the pinned revision and the index / file-list
# hashes, ``ld_draw`` for the drawn windows.
PUBLIC_SCHEMA = "lane-d-public/1"
PUBLIC_INDEX_SCHEMA = "lane-d-public-index/1"
PUBLIC_SEED_TAG = "lane_d_public"
PUBLIC_PINS = {
    "gift_eval_pretrain": ("Salesforce/GiftEvalPretrain",
                           "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874", False),
    "rmisc": ("nju-zhangsq/RMISC", "d0cdd935300ca9b3b2d6f23e6085db24cb7560c2", True),
}
PUBLIC_N_WINDOWS = 5000
PUBLIC_MIN_HISTORY = 128


def _pub_canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _pub_quota(caps: dict[str, int], total: int) -> dict[str, int]:
    """The even split: smallest capacity first (ties by key), scarce cells
    whole, the surplus re-split."""
    quota = {k: 0 for k in caps}
    remaining = dict(sorted(caps.items()))
    want = int(total)
    while remaining and want > 0:
        share = max(1, want // len(remaining))
        key = min(remaining, key=lambda k: (remaining[k], k))
        cap = remaining.pop(key)
        take = min(cap, share if remaining else want)
        quota[key] = take
        want -= take
    return quota


def _pub_hint(seed: str, kind: str, o: int, r: int) -> int:
    return int(hashlib.sha256(f"{seed}:{kind}:{o}:{r}".encode()).hexdigest(), 16)


def _pub_cut(seed: str, o: int, r: int, length: int, h: int, m: int, multi: bool,
             used: list[int]) -> int:
    """One window per series: ``c = m + H(cut, o, 1) mod (L - h - m + 1)``.
    Several: slots ``m + off + j*h`` (``j < k``), ``off = H(offset, o, 0) mod
    (L - m - k*h + 1)``; round ``r`` takes the unused slot at ``H(slot, o, r)
    mod #unused`` (ascending)."""
    if not multi:
        return m + _pub_hint(seed, "cut", o, 1) % (length - h - m + 1)
    k = max(0, (length - m) // h)
    off = _pub_hint(seed, "offset", o, 0) % (length - m - k * h + 1)
    free = [m + off + j * h for j in range(k) if m + off + j * h not in used]
    return free[_pub_hint(seed, "slot", o, r) % len(free)]


def _pub_rows(index_dir) -> tuple[dict, list[dict]]:
    """The manifest and rows of a public index, every hash checked (a
    ``ValueError`` names the first defect)."""
    import gzip

    d = Path(index_dir)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if m.get("schema") != PUBLIC_INDEX_SCHEMA:
        raise ValueError(f"manifest schema {m.get('schema')!r}")
    files = sorted(({"path": str(f["path"]), "size": int(f["size"]), "sha256": str(f["sha256"])}
                    for f in m["files"]), key=lambda f: f["path"])
    if hashlib.sha256(_pub_canon(files)).hexdigest() != m.get("files_sha256"):
        raise ValueError("files_sha256 is not the manifest's file list")
    h, rows = hashlib.sha256(), []
    with gzip.open(d / "index.jsonl.gz", "rb") as gz:
        for line in gz:
            h.update(line)
            row = json.loads(line)
            if row.get("ordinal") != len(rows) or _pub_canon(row) + b"\n" != line:
                raise ValueError(f"index line {len(rows)} is not canonical / in ordinal order")
            rows.append(row)
    if h.hexdigest() != m.get("index_sha256") or len(rows) != m.get("n_series"):
        raise ValueError("the index bytes do not hash to index_sha256")
    return m, rows


def public_redraw(rows: list[dict], *, source: str, block_hash: str, horizon: int,
                  n_windows: int = PUBLIC_N_WINDOWS, min_history: int = PUBLIC_MIN_HISTORY,
                  multi_window: bool | None = None) -> dict:
    """The public draw re-derived from index rows: ``{capacity, quota,
    windows}`` (windows ``{window_id, ordinal, family, freq, cut, h,
    length}`` sorted by id)."""
    multi = PUBLIC_PINS[source][2] if multi_window is None else bool(multi_window)
    h, m = int(horizon), int(min_history)
    seed = f"{block_hash}:{PUBLIC_SEED_TAG}:{source}"

    def slots(n: int) -> int:
        return max(0, (int(n) - m) // h)

    cells: dict[str, list[dict]] = {}
    for r in rows:
        if slots(r["length"]) >= 1:
            cells.setdefault(f"{r['family']}|{r['freq']}", []).append(r)
    caps = {k: (sum(slots(r["length"]) for r in v) if multi else len(v)) for k, v in cells.items()}
    quota = _pub_quota(caps, n_windows)
    taken: dict[int, list[int]] = {}
    for k in sorted(cells):
        left, rnd = quota.get(k, 0), 1
        members = sorted(cells[k], key=lambda r: r["ordinal"])
        while left > 0:
            pool = [r for r in members if len(taken.get(r["ordinal"], ())) == rnd - 1
                    and slots(r["length"]) >= rnd]
            if not pool:
                break
            pool.sort(key=lambda r: (_pub_hint(seed, "pick", r["ordinal"], rnd), r["ordinal"]))
            for r in pool[:left]:
                o = r["ordinal"]
                c = _pub_cut(seed, o, rnd, int(r["length"]), h, m, multi, taken.get(o, []))
                taken.setdefault(o, []).append(c)
                left -= 1
            if not multi:
                break
            rnd += 1
    by_o = {r["ordinal"]: r for r in rows}
    windows = sorted(({"window_id": f"{source}:{by_o[o]['series_id']}:{c}:h{h}", "ordinal": o,
                       "family": by_o[o]["family"], "freq": by_o[o]["freq"], "cut": c, "h": h,
                       "length": int(by_o[o]["length"])}
                      for o, cuts in taken.items() for c in cuts),
                     key=lambda w: w["window_id"])
    return {"capacity": dict(sorted(caps.items())), "quota": dict(sorted(quota.items())),
            "windows": windows}


def _public_block_hash(report: dict):
    for k, f in (("draw", "boundary_block_hash"), ("gate", "block_hash")):
        v = report.get(k)
        if isinstance(v, dict) and isinstance(v.get(f), str):
            return v[f]
    return None


def _public_index_problems(report: dict, public_indexes) -> list[str]:
    """``ld_public_index``: each source is a pinned one at its pinned repo and
    revision; with ``public_indexes`` ({source: index dir}) the auditor's
    index verifies and its revision, ``index_sha256`` and ``files_sha256``
    are the report's."""
    if "public" not in report:
        return []
    pub = report["public"]
    if not isinstance(pub, dict) or pub.get("schema") != PUBLIC_SCHEMA:
        return [f"public schema {pub.get('schema') if isinstance(pub, dict) else pub!r}"]
    out = []
    for s in pub["sources"]:
        src = s["source"]
        pin = PUBLIC_PINS.get(src)
        if pin is None:
            out.append(f"public source {src!r} is not a pinned source")
            continue
        if (s["repo"], s["revision"]) != pin[:2]:
            out.append(f"public {src}: {s['repo']}@{s['revision']} is not the pinned "
                       f"{pin[0]}@{pin[1]}")
        if public_indexes is None or src not in public_indexes:
            continue
        try:
            m, _rows = _pub_rows(public_indexes[src])
        except (OSError, ValueError, KeyError, TypeError) as e:
            out.append(f"public {src}: the auditor's index does not verify: {e}")
            continue
        for k in ("index_sha256", "files_sha256", "revision", "repo"):
            if m.get(k) != s[k]:
                out.append(f"public {src}: {k} {s[k]!r} is not the index's {m.get(k)!r}")
        if m.get("source") != src:
            out.append(f"public {src}: the auditor's index is of {m.get('source')!r}")
    return out


def _public_draw_problems(report: dict, cfg, public_indexes) -> list[str]:
    """``ld_draw`` for the public half: the seed is the settlement's block
    hash, the draw constants are the spec's (``n_windows`` 5000,
    ``min_history`` 128, ``horizon`` = ``[lane_d] horizon``, ``multi_window``
    per source); with ``public_indexes`` the capacity, quota and every drawn
    window are re-derived from the index and must match."""
    if "public" not in report:
        return []
    bh = _public_block_hash(report)
    out = []
    for s in report["public"]["sources"]:
        src = s["source"]
        if src not in PUBLIC_PINS:
            continue
        if bh is None or s["seed"] != f"{bh}:{PUBLIC_SEED_TAG}:{src}":
            out.append(f"public {src}: seed {s['seed']!r} is not "
                       f"<boundary_block_hash>:{PUBLIC_SEED_TAG}:{src}")
            continue
        want = {"n_windows": PUBLIC_N_WINDOWS, "min_history": PUBLIC_MIN_HISTORY,
                "horizon": int(cfg.horizon), "multi_window": PUBLIC_PINS[src][2]}
        bad = sorted(k for k, v in want.items() if s[k] != v)
        if bad:
            out.append(f"public {src}: draw constants {bad} are not the spec's")
            continue
        if public_indexes is None or src not in public_indexes:
            continue
        try:
            m, rows = _pub_rows(public_indexes[src])
        except (OSError, ValueError, KeyError, TypeError):
            continue          # ld_public_index names it
        if m.get("index_sha256") != s["index_sha256"]:
            continue          # ld_public_index names it
        got = public_redraw(rows, source=src, block_hash=bh, horizon=int(cfg.horizon))
        for k in ("capacity", "quota"):
            if got[k] != s[k]:
                out.append(f"public {src}: {k} is not the re-derived draw's")
        if got["windows"] != s["windows"]:
            mine = {w["window_id"] for w in got["windows"]}
            theirs = {w["window_id"] for w in s["windows"]}
            diff = sorted(mine ^ theirs)[:3] or ["(same ids, other fields)"]
            out.append(f"public {src}: the drawn windows are not the re-derived draw "
                       f"(first differences {diff})")
    return out


_UNSET = object()

# ── rev6.18 B3: per-check refusal codes (LANE_D_PLAN rev6.18 B3 table) ──────
#
# Only four checks stop pay: the report signature, the config echo
# (= ``config_at(boundary)``), the scores reproducing the named winner
# (``rows_sha256`` and the bootstrap) and the weights matching the paid
# court. Their codes are ``burn``: lane D's share, and only lane D's, goes to
# ``burn_uid`` (B1). Every other failed check is ``alert``: it is logged and
# alerted (cascade-ops tag ``lane_d_check_alert``) and the signed court keeps
# being paid. A verifier that cannot run is neither (``TRANSIENT_CODE``: the
# receipt is retried).
BURN = "burn"
ALERT = "alert"
REFUSAL_CODES: dict[str, str] = {
    "lane_d_record_missing": BURN,
    "lane_d_record_malformed": BURN,
    "lane_d_boundary_mismatch": BURN,
    "lane_d_report_mismatch": BURN,
    "ld_schema": BURN,
    "ld_signature": BURN,
    "ld_not_armed": BURN,
    "lane_d_not_armed": BURN,
    "ld_params": BURN,
    "ld_pay_share": BURN,
    "ld_seed": BURN,
    "ld_rows_sha256": BURN,
    "ld_rows_count": BURN,
    "ld_rows_duplicate": BURN,
    "ld_cohort": BURN,
    "ld_crown": BURN,
    "ld_court": BURN,
    "lane_d_carve_unsupported": BURN,
    "ld_malformed": BURN,
    "ld_scoring_method": ALERT,
    "ld_sentinel": ALERT,
    "ld_row_order": ALERT,
    "ld_clock": ALERT,
    "ld_embargo": ALERT,
    "ld_draw": ALERT,
    "ld_judged_set": ALERT,
    "ld_chain": ALERT,
    "ld_snapshot": ALERT,
    "ld_public_index": ALERT,
    "ld_bench": ALERT,
}
TRANSIENT_CODE = "ld_verifier_unavailable"
# The cascade-ops follower tag of an alert-class finding (W11).
ALERT_TAG = "lane_d_check_alert"


def refusal_class(code: str) -> str:
    """``burn`` / ``alert`` for a :data:`REFUSAL_CODES` code; an unknown code
    is ``burn`` (fail closed)."""
    return REFUSAL_CODES.get(str(code), BURN)


class LaneDFindings:
    """Every failed check of one report, in check order, as ``(code,
    detail)``. ``transient``: a verifier could not run (retry; neither burn
    nor pay). Checks stop at the first ``burn`` finding; ``alert`` findings
    never stop them, so a burn check behind an alert still runs."""

    def __init__(self) -> None:
        self.findings: list[tuple[str, str]] = []
        self.transient = ""
        self.boundary: int | None = None
        self.n_cohorts = 0
        self.n_rows = 0

    @property
    def burn(self) -> tuple[str, str] | None:
        return next((f for f in self.findings if refusal_class(f[0]) == BURN), None)

    @property
    def alerts(self) -> list[tuple[str, str]]:
        return [f for f in self.findings if refusal_class(f[0]) == ALERT]

    def add(self, code: str, detail: str) -> bool:
        """Record a finding; True when it is a burn (the checks stop)."""
        if code not in REFUSAL_CODES:
            raise ValueError(f"lane D refusal code {code!r} is not in REFUSAL_CODES")
        self.findings.append((code, str(detail)))
        return refusal_class(code) == BURN


def lane_d_findings(report: dict, cfg, *, signer_hotkey: str, prior_reports=None,
                    era_sentinel=_UNSET, pool_dirs=None, bench_rows=None, uid_of=None,
                    chain=None, max_channels=None, public_indexes=None) -> LaneDFindings:
    """Run every check of :func:`check_lane_d` and return its findings
    (rev6.18 B3). Same arguments as :func:`check_lane_d`.

    rev6.18 B2 (the W7d satellite replay's rules, ported so no verification
    is lost): ``bench_rows`` (W3b6b): the bound bench records, a
    ``{sha256: bytes}`` mapping or a directory of ``<sha256>.json`` files;
    ``None`` skips :func:`~cascade.audit.lane_d_chain.check_gate_rows`.
    ``uid_of``: the metagraph ``{hotkey: uid}`` at the boundary, pinning each
    court slot's uid (``None``: the stated uid, taken as signed). ``chain``:
    ``{"receipts": [...], "states"?: {...}, "uid_of"?: {...},
    "bench_rows"?: ...}`` runs the chain rules of
    :mod:`cascade.audit.lane_d_chain` over ``prior_reports`` + this report
    (receipt links, ``king_before``, ``prior_kings``, holds, used windows,
    the house challenger, entry progress, state snapshots); their mismatch
    paths become alert-class codes. Every check that can stop pay is
    re-derived from this report alone."""
    from ..shared.sr25519 import SignatureVerifierUnavailable
    from .lane_d_chain import check_gate_rows

    f = LaneDFindings()
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        got = report.get("schema") if isinstance(report, dict) else type(report).__name__
        f.add("ld_schema", f"unknown report schema {got!r}")
        return f

    def many(code: str, problems) -> bool:
        return bool(problems) and f.add(code, "; ".join(list(problems)[:5]))

    def one(code: str, why) -> bool:
        return bool(why) and f.add(code, why)

    def alert(code: str, fn) -> None:
        """An alert-class check: a malformed field fails that check only."""
        try:
            got = fn()
        except (KeyError, TypeError, ValueError) as e:
            f.add(code, f"malformed report: {type(e).__name__}: {e}")
            return
        if isinstance(got, list):
            many(code, got)
        else:
            one(code, got)

    try:
        boundary = int(report["boundary"])
        f.boundary = boundary
        if not cfg.active(boundary):
            f.add("ld_not_armed", f"lane D does not judge the settlement at {boundary}")
            return f
        try:
            why = _signature(report, signer_hotkey)
        except SignatureVerifierUnavailable as e:
            f.transient = f"cannot verify the signature: {e}"
            return f
        if one("ld_signature", why):
            return f
        if one("ld_cohort", _confirmation_problem(report)):
            return f
        alert("ld_scoring_method",
              lambda: None if report.get("scoring_method") == "quantile_head"
              else f"scoring_method {report.get('scoring_method')!r}")
        want = _expected_params(cfg)
        if report.get("params") != want:
            diff = sorted(k for k in set(want) | set(report.get("params") or {})
                          if (report.get("params") or {}).get(k) != want.get(k))
            f.add("ld_params", f"stated params differ from config on {diff}")
            return f
        # RV7 B2 (c): the bootstrap seed is the boundary's and the pool's,
        # never a free choice the producer could grind.
        want_seed = f"lane-d/1:{boundary}:{report['pool']['sha256']}"
        if report.get("seed") != want_seed:
            f.add("ld_seed", f"seed {report.get('seed')!r} != {want_seed!r}")
            return f
        if era_sentinel is not _UNSET:
            alert("ld_sentinel", lambda: _sentinel_problem(report, era_sentinel))
        rows = list(report.get("rows") or [])
        f.n_rows = len(rows)
        if hashlib.sha256(_canon(rows)).hexdigest() != report.get("rows_sha256"):
            f.add("ld_rows_sha256", "rows_sha256 does not match the rows")
            return f
        rows_by = {}
        for r in rows:
            key = (int(r["cohort"]), str(r["window_id"]), str(r["digest"]))
            if key in rows_by:
                f.add("ld_rows_duplicate", f"duplicate row {key}")
                return f
            rows_by[key] = r
        windows = {w["window_id"]: w for w in report.get("windows") or []}
        entries = {e["digest"]: e for e in report.get("entries") or []}
        cohorts = list(report.get("cohorts") or [])
        f.n_cohorts = len(cohorts)
        # One row per (window, model): KING_D, the house when it is a second
        # bar (a miner holds KING_D), and every member.
        house = report["house"]["digest"]
        expected_rows = sum(len(c["window_ids"])
                            * (len(_bars(c["king_digest"], house, c["members"]))
                               + len(c["members"]))
                            for c in cohorts)
        if expected_rows != len(rows):
            f.add("ld_rows_count", f"{len(rows)} rows; the cohorts account for {expected_rows}")
            return f
        # rev6.23: the crown bar applied is the config's at the stated
        # tenure; every verdict below clears against it.
        if one("ld_params", _crown_bar_problem(report, cfg)):
            return f
        problems, confirms = [], {}
        for c in cohorts:
            problems += _cohort_problems(report, cfg, c, rows_by, windows, entries, confirms)
        if many("ld_cohort", problems):
            return f
        alert("ld_row_order", lambda: _row_order_problem(report, rows))
        alert("ld_clock", lambda: _clock_problems(report, cfg, windows))
        alert("ld_embargo", lambda: _embargo_problems(report, cfg, windows, rows_by))
        alert("ld_draw", lambda: _draw_problems(report, cfg, windows))
        # rev6.18 A4 (F2): each multichannel window's channel pick is part of
        # the seeded draw (the boundary block hash): re-derived, alert class.
        alert("ld_draw", lambda: _channel_problems(report, windows, max_channels))
        # rev6.18 A1/A2/A6 (F3): the public draw, re-derived from the auditor's
        # index (``public_indexes``) and the block hash; the pinned revision
        # and the index / file-list hashes. Both alert class.
        alert("ld_public_index", lambda: _public_index_problems(report, public_indexes))
        alert("ld_draw", lambda: _public_draw_problems(report, cfg, public_indexes))
        # The §1.3b gate decides every crowning candidate; the crowning and
        # the paid court follow from it.
        crown, problems = _crown_problems(report, cfg, boundary, confirms, entries)
        if many("ld_crown", problems):
            return f
        alert("ld_judged_set", lambda: _judged_set_problems(report, cfg, entries))
        before, after = report["king_before"], report["king_after"]
        if crown is not None and crown == report["house"]["digest"]:
            # W5f (§2 item 2, T10): the house retakes KING_D from a miner.
            if (report.get("crowned") is not None or after["digest"] != crown
                    or not after["house"] or after.get("hotkey") is not None
                    or before["house"]):
                f.add("ld_crown", f"house retake {crown[:12]}: crowned "
                                  f"{report.get('crowned')!r}, king_after "
                                  f"{str(after['digest'])[:12]} (house={after['house']}), "
                                  f"king_before house={before['house']}")
                return f
        elif crown is not None:
            if report.get("crowned") != crown or after["digest"] != crown or after["house"]:
                f.add("ld_crown", f"crowned {report.get('crowned')!r}; re-derived {crown!r}")
                return f
        elif report.get("crowned") is not None or after["digest"] != before["digest"]:
            why = ("no gate record (the gate did not run)" if "gate" not in report
                   else "no gate result passes")
            f.add("ld_crown", f"crowned {report.get('crowned')!r} / king changed, but {why}: "
                              f"nothing may crown")
            return f
        if prior_reports is not None:
            try:
                alert("ld_chain", lambda: (lambda w: f"chain: {w}" if w else None)(
                    _chain_problems(report, prior_reports, signer_hotkey, entries)))
            except SignatureVerifierUnavailable as e:
                f.transient = f"cannot verify the prior reports' signatures: {e}"
                return f
        if pool_dirs is not None:
            alert("ld_snapshot", lambda: (lambda w: f"snapshot: {w}" if w else None)(
                _snapshot_problem(report, cfg, list(pool_dirs), windows)))
        if bench_rows is not None:
            alert("ld_bench", lambda: [f"mismatch at {p}" for p in
                                       check_gate_rows(report, bench_rows, cfg)])
        if one("ld_crown", _king_after_problem(report, entries, crown)):
            return f
        pay = report.get("pay")
        court = (pay or {}).get("court") or []
        if float((pay or {}).get("share", -1.0)) != float(cfg.share):
            f.add("ld_pay_share", "pay share differs from config")
            return f
        if not court or court[0]["digest"] != after["digest"] \
                or (court[0]["hotkey"] is None) != bool(after["house"]):
            f.add("ld_court", "the paid court does not start with KING_D (house slot unpaid)")
            return f
        if len(court) > 1 + int(cfg.reward_prior_kings):
            f.add("ld_court", f"court has {len(court)} slots; at most "
                              f"{1 + int(cfg.reward_prior_kings)}")
            return f
        if any(s["digest"] == report["house"]["digest"] for s in court[1:]):
            f.add("ld_court", "the house sits in the court as a prior king")
            return f
        if one("ld_court", _court_problem(report, cfg, uid_of)):
            return f
        if chain is not None:
            r = _chain_findings(report, cfg, signer_hotkey, prior_reports, chain)
            if r == "transient":
                f.transient = "cannot verify a chain report's signature"
                return f
            for code, detail in r:
                f.add(code, detail)
    except (KeyError, TypeError, ValueError) as e:
        f.add("ld_malformed", f"malformed report: {type(e).__name__}: {e}")
    return f


def _king_after_problem(report: dict, entries: dict, crown) -> str | None:
    """Ported from the W7d replay (G8): the whole ``king_after`` record. A
    miner crown is ``{digest, hotkey, house: False, reveal_ts}`` of the
    crowned entry; a house retake the house record; no crown leaves
    ``king_before``."""
    before, after, house = report["king_before"], report["king_after"], report["house"]
    if crown is not None and crown == house["digest"]:
        want = {"digest": house["digest"], "hotkey": None, "house": True,
                "reveal_ts": house["reveal_ts"]}
    elif crown is not None:
        e = entries[crown]
        want = {"digest": crown, "hotkey": e["hotkey"], "house": False,
                "reveal_ts": e["reveal_ts"]}
    else:
        want = dict(before)
    if after != want:
        diff = sorted(k for k in set(want) | set(after) if after.get(k) != want.get(k))
        return f"king_after differs from the re-derived record on {diff}"
    return None


def _court_problem(report: dict, cfg, uid_of) -> str | None:
    """Ported from the W7d replay (C5, G5): ``pay`` is exactly ``{share,
    court}`` and the court is KING_D then ``prior_kings`` by recency, each
    slot ``{digest, hotkey, uid, burn_reason}``: a house slot has no hotkey
    and no uid (``house_king``), a slot without a uid burns
    (``deregistered``). ``uid_of`` (``{hotkey: uid}``, the metagraph at the
    boundary) pins each uid; without it the stated uid is taken as signed."""
    pay = report["pay"]
    if not isinstance(pay, dict) or set(pay) != {"share", "court"}:
        return f"pay keys {sorted(pay) if isinstance(pay, dict) else pay!r} are not [court, share]"
    stated = pay["court"]
    rpk = max(0, int(cfg.reward_prior_kings))
    kings = [report["king_after"], *list(report.get("prior_kings") or [])[:rpk]]
    want = []
    for i, k in enumerate(kings):
        hk = None if k["house"] else k["hotkey"]
        if hk is None:
            uid = None
        elif uid_of is not None:
            got = uid_of.get(hk)
            uid = got if isinstance(got, int) and not isinstance(got, bool) else None
        else:
            slot = stated[i] if i < len(stated) and isinstance(stated[i], dict) else {}
            got = slot.get("uid")
            uid = got if isinstance(got, int) and not isinstance(got, bool) else None
        burn = None if uid is not None else ("house_king" if hk is None else "deregistered")
        want.append({"digest": k["digest"], "hotkey": hk, "uid": uid, "burn_reason": burn})
    if stated != want:
        at = next((i for i, (g, w) in enumerate(zip(stated, want, strict=False)) if g != w),
                  min(len(stated), len(want)))
        return f"pay.court[{at}] is not the re-derived slot (KING_D, then prior_kings)"
    return None


# Chain mode (rev6.18 B2): the ported W7d chain rules' mismatch paths, by
# leading field, to their B3 code. All are alert-class: the four burn checks
# are re-derived per report above.
_CHAIN_PATH_CODES = (
    ("gate.results", "ld_bench"), ("windows", "ld_draw"), ("draw", "ld_draw"),
    ("entries", "ld_judged_set"), ("armed_", "ld_judged_set"), ("pool", "ld_snapshot"),
    ("scoring_method", "ld_scoring_method"),
)


def _chain_code(path: str) -> str:
    return next((c for p, c in _CHAIN_PATH_CODES if path.startswith(p)), "ld_chain")


def _chain_findings(report: dict, cfg, signer_hotkey: str, prior_reports, chain):
    """``chain`` = ``{"receipts": [lane-d-receipt/1 records, settlement order,
    ending with this report's], "states"?: {boundary: state bytes},
    "uid_of"?: {boundary: {hotkey: uid}}, "bench_rows"?: ...}``. The W7d
    rules (:mod:`cascade.audit.lane_d_chain`) over the reports
    ``prior_reports`` + this one: a list of ``(code, detail)`` for this
    report's boundary, or ``"transient"``."""
    from .lane_d_chain import MISMATCH, UNVERIFIABLE, replay_chain

    if not isinstance(chain, dict) or not isinstance(chain.get("receipts"), (list, tuple)):
        return [("ld_chain", "chain: no receipt records given")]
    b = int(report["boundary"])
    reports = {}
    for p in list(prior_reports or []):
        if isinstance(p, dict) and isinstance(p.get("boundary"), int):
            reports[p["boundary"]] = p
    reports[b] = report
    receipts = [r for r in chain["receipts"]
                if isinstance(r, dict) and isinstance(r.get("boundary"), int)
                and r["boundary"] <= b]
    if not receipts or receipts[-1]["boundary"] != b:
        return [("ld_chain", f"chain: the receipt records do not end at boundary {b}")]
    res = replay_chain(reports, receipts, cfg, signer_ss58=signer_hotkey,
                       uid_of=chain.get("uid_of"), states=chain.get("states"),
                       bench_rows=chain.get("bench_rows"))
    mine = res[-1]
    if mine.status == UNVERIFIABLE:
        if mine.reason == "signature_unverifiable":
            return "transient"
        return [("ld_chain", f"chain: unverifiable ({mine.reason})")]
    if mine.status != MISMATCH:
        return []
    by_code: dict[str, list[str]] = {}
    for p in mine.mismatches:
        by_code.setdefault(_chain_code(p), []).append(p)
    return [(code, "chain: mismatch at " + ", ".join(paths[:8]))
            for code, paths in by_code.items()]


def check_lane_d(report: dict, cfg, *, signer_hotkey: str, prior_reports=None,
                 era_sentinel=_UNSET, pool_dirs=None, bench_rows=None, uid_of=None,
                 chain=None, max_channels=None, public_indexes=None) -> CheckResult:
    """PASS on a genuine report; FAIL when the signature, ``rows_sha256``,
    a stated parameter or the verdict re-derived from the rows disagrees,
    or the crown's provenance does not hold (RV7 B2). rev6.18 B3: the FAIL
    detail is ``<code>: <detail>`` for the first failed check (codes in
    :data:`REFUSAL_CODES`; :func:`lane_d_findings` has every finding and its
    burn / alert class); a verifier that cannot run is a WARN
    ``ld_verifier_unavailable: ...``.

    ``prior_reports`` (chain mode, what W9 runs): the earlier signed reports
    the auditor holds, in any order; each
    each ``held_from`` must be shown by one that verifies under
    ``signer_hotkey``. ``era_sentinel``: the era's W-S1 receipt record
    (``{"suite", "block", "block_hash"}``) or ``None`` when W-S1 is unarmed;
    left unset, ``gate.sentinel`` is not pinned. ``pool_dirs``: snapshot
    directories the auditor holds; every snapshot a judged window names must
    be among them, valid, and give that window's record and eligibility
    (README rules 12-14); ``None`` skips those rules. ``bench_rows``: the
    W3b6b bound bench records (:func:`check_gate_rows`). ``max_channels``
    (rev6.18 A4, F2): ``[generator] max_channels`` at the boundary; every
    judged multichannel window's channel pick is re-derived with it
    (``ld_draw``; with ``pool_dirs`` also checked against the stored series,
    ``ld_snapshot``); ``None`` fails any multichannel window.
    ``public_indexes`` (rev6.18 A6, F3): ``{source: index dir}``, the auditor's
    copies of the public indexes; each named source's index is verified
    against the report's revision / ``index_sha256`` / ``files_sha256``
    (``ld_public_index``) and its draw re-derived from the index and the
    block hash (``ld_draw``); ``None`` checks the report's public block
    alone (pins, seed, constants)."""
    f = lane_d_findings(report, cfg, signer_hotkey=signer_hotkey, prior_reports=prior_reports,
                        era_sentinel=era_sentinel, pool_dirs=pool_dirs, bench_rows=bench_rows,
                        uid_of=uid_of, chain=chain, max_channels=max_channels,
                        public_indexes=public_indexes)
    if f.findings:
        code, detail = f.findings[0]
        return _fail(NAME, f"{code}: {detail}")
    if f.transient:
        return _warn(NAME, f"{TRANSIENT_CODE}: {f.transient}")
    return _ok(NAME, f"boundary {f.boundary}: {f.n_cohorts} cohort(s), {f.n_rows} rows "
                     f"re-derived")




def _absent_problem(record) -> str | None:
    """The first structure defect of an absent receipt record, or None."""
    if not isinstance(record, dict):
        return f"not an object ({type(record).__name__})"
    missing = [k for k in ABSENT_KEYS if k not in record]
    if missing:
        return f"missing {missing}"
    extra = sorted(str(k) for k in record if k not in ABSENT_KEYS)
    if extra:
        return f"unknown fields {extra}"
    if record["schema"] != RECEIPT_SCHEMA:
        return f"receipt schema {record['schema']!r} is not {RECEIPT_SCHEMA}"
    if record["status"] != ABSENT_STATUS:
        return f"status {record['status']!r} is not {ABSENT_STATUS!r}"
    reason = record["reason"]
    if not isinstance(reason, str) or reason not in ABSENT_REASONS:
        return f"reason {reason!r} is not one of {list(ABSENT_REASONS)}"
    b = record["boundary"]
    if isinstance(b, bool) or not isinstance(b, int) or b < 0:
        return f"boundary {b!r} is not a non-negative int"
    for k in ("prior_state_sha256", "posterior_state_sha256"):
        if not _is_hex64(record[k]):
            return f"{k} {record[k]!r} is not 64 lower-case hex"
    if record["prior_state_sha256"] != record["posterior_state_sha256"]:
        return "an absent settlement moves no state, but its prior != its posterior"
    return None


def check_lane_d_absent(record, cfg, *, boundary: int, previous_posterior_sha256: str,
                        report=None) -> CheckResult:
    """PASS exactly when ``record`` is a well-formed absent lane D receipt
    record, lane D judges the settlement at ``boundary`` (``cfg.active``),
    the record names that boundary, its prior is
    ``previous_posterior_sha256`` (the previous lane D receipt's posterior,
    or the genesis digest before the first settlement) and no ``report`` is
    given for the boundary (a published report contradicts the record).
    Otherwise FAIL. Pure and deterministic."""
    why = _absent_problem(record)
    if why:
        return _fail(ABSENT_NAME, f"absent record: {why}")
    at = record["boundary"]
    if isinstance(boundary, bool) or not isinstance(boundary, int) or at != boundary:
        return _fail(ABSENT_NAME, f"the absent record names boundary {at}, not {boundary!r}")
    if not cfg.active(boundary):
        return _fail(ABSENT_NAME, f"lane D does not judge the settlement at {boundary}")
    if record["prior_state_sha256"] != previous_posterior_sha256:
        return _fail(ABSENT_NAME, f"the absent record's state {record['prior_state_sha256']} "
                                  f"is not the previous posterior {previous_posterior_sha256!r}")
    if report is not None:
        return _fail(ABSENT_NAME, f"a lane D report is given for boundary {boundary}, which "
                                  "the record calls absent")
    return _ok(ABSENT_NAME, f"boundary {boundary}: lane D absent ({record['reason']}); "
                            "lane D's share burns")


__all__ = ["ALERT", "ALERT_TAG", "BURN", "CONFIRMATION_STAGE_PRESENT", "REFUSAL_CODES",
           "TRANSIENT_CODE", "LaneDFindings", "check_lane_d", "check_lane_d_absent",
           "combined_log_ratio", "lane_d_findings", "refusal_class", "window_refusal"]
