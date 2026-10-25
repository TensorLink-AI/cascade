"""Public-benchmark no-regression gate over GIFT-Eval, BOOM and TIME
(DEC-CA-0054), and its four-suite successor (DEC-CA-0061 rule v2,
"avg_noninf_winrate", below) — a second AND-ed condition on a dethrone, like
:mod:`.gift_gate` but over all the suites and on signed bench rows every
validator reads identically (the trainer's bench report, or the main
validator's ``bench_v`` report under DEC-CA-0063 ``[v2] bench_source =
"validator"``).

The private pool decides *whether the challenger is better*. This gate checks
two things, and can only BLOCK a dethrone the private LCB already granted:

1. **No suite significantly worse.** Per suite, the paired items (configs /
   tasks) shared by king and challenger are each scored ``geomean(CRPS, MASE)``
   and bootstrapped like the main eval (:func:`.expert_lane.
   paired_geomean_lcb`, items resampled with replacement, seeded by the
   round's block hash). A suite fails when even the upper bound on the
   challenger's relative improvement is below zero — significantly worse at
   ``alpha / 3`` (Bonferroni over the three suites, so the false-block rate
   stays at ``alpha`` when the two checkpoints are equal).
2. **The six-number mean is better.** The Cascade score (geomean of the six
   suite-level CRPS/MASE numbers, :func:`cascade.validator.cascade.
   cascade_score`) must be strictly lower for the challenger. No threshold.

DEC-CA-0061 rule v2 (``rule = RULE_V2`` = "avg_noninf_winrate", from ``[v2]
bench_gate_v2_from_block``; owner 2026-10-20: "improvement on average and no
statistically significant regression"). The unit is one benchmark TASK — one
signed item row (the sidecar's ``full`` key: a GIFT-Eval config, a BOOM
dataset, a TIME task, an FEV task); series inside a task are never resampled
separately. Over GIFT-Eval, BOOM, TIME and FEV, on the tasks both sides scored
with usable (finite, > 0) CRPS AND MASE (others are masked and counted):

* per task ``r_crps = chal / king``, ``r_mase = chal / king`` and the task
  score ``s = sqrt(r_crps * r_mase)``;
* per suite the ratio ``R_s`` = geomean of every ``r_crps`` and ``r_mase``
  (= geomean of the task scores);

1. **Average.** ``geomean_s(R_s) < 1`` — each SUITE weighs the same (GIFT-Eval's
   ~97 configs must not swamp BOOM/TIME/FEV). A point estimate: the private
   pool already supplies the significance.
2. **No significant regression.** Per suite, the paired bootstrap over tasks
   (:func:`.expert_lane.paired_geomean_lcb`, the DEC-CA-0054 machinery and
   seed ``{seed}|bench|{suite}``) at ``alpha / 4``; the one-sided lower bound
   of ``R_s`` (``1 - ucb``) must not exceed ``1 + tolerance``.
3. **Win rate.** Per suite the share of tasks with ``s < 1`` (an exact tie
   counts half); the suite-equal mean of the four shares must be
   ``>= min_win_rate``.

It replaced (in place, before any settlement was judged by it) the
2026-10-19 rule of eight strict suite-level comparisons, which seed noise
alone fails; :func:`strict_eight` keeps that comparison for the
report-only calibration script (``scripts/bench_seed_noise.py``).

Pure numpy over the rows; no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .expert_lane import paired_geomean_lcb
from .public_guards import METRICS, MissingPublicGuard, _index, _usable, pair_public_rows

SUITES = ("gift", "boom", "time")
_SIX = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase")

# The effective rule, stamped as ``KothParams.bench_gate_rule`` (CONSENSUS;
# :func:`cascade.shared.era.bench_gate_rule_at`). RULE_V1 is the default and
# is dropped from a receipt's params, so DEC-CA-0054 receipts are unchanged.
RULE_V1 = "dec-ca-0054"
RULE_V2 = "avg_noninf_winrate"
RULES = (RULE_V1, RULE_V2)
SUITES_V2 = (*SUITES, "fev")
# DEC-CA-0061 rule v2 defaults ([scoring] bench_gate_tolerance /
# bench_gate_min_win_rate; CONSENSUS, drop-when-default in receipt params).
DEFAULT_TOLERANCE = 0.01
DEFAULT_MIN_WIN_RATE = 0.5
# Item-row suite key → the prefix of its two suite-level numbers in BenchScores.
SCORE_PREFIX = {"gift": "gifteval", "boom": "boom", "time": "time", "fev": "fev"}


@dataclass(frozen=True)
class BenchGateResult:
    """``computed`` False ⇒ uncomputable (the caller holds the round, never a
    silent pass); ``passed`` is then ``None``. ``suite_ucb`` is each suite's
    upper bound on the relative improvement (negative ⇒ significantly worse)."""

    computed: bool
    passed: bool | None
    reason: str = ""
    suite_ucb: dict[str, float] = field(default_factory=dict)
    suite_n: dict[str, int] = field(default_factory=dict)
    king_score: float = float("nan")
    chal_score: float = float("nan")
    # DEC-CA-0061 only: the rule that judged, the three headline numbers
    # (suite-equal geomean ratio, the worst suite's ratio LCB, the
    # suite-equal win rate), the thresholds, and per suite {"n", "masked",
    # "ratio", "ratio_lcb", "win_rate"}. Empty under DEC-CA-0054.
    rule: str = RULE_V1
    suites: dict[str, dict[str, float]] = field(default_factory=dict)
    avg_ratio: float = float("nan")
    worst_lcb: float = float("nan")
    win_rate: float = float("nan")
    tolerance: float = DEFAULT_TOLERANCE
    min_win_rate: float = DEFAULT_MIN_WIN_RATE

    def detail(self) -> dict | None:
        """The receipt record of a DEC-CA-0061 judgement (``None`` under
        DEC-CA-0054, so those receipts keep their bytes): the rule, the three
        numbers, the thresholds, each suite's task count / masked count /
        ratio / ratio LCB / win rate, and the reason. Non-finite numbers are
        left out (strict JSON)."""
        if self.rule != RULE_V2:
            return None
        suites: dict[str, dict] = {}
        for s in SUITES_V2:
            row = {k: (int(v) if k in ("n", "masked") else float(v))
                   for k, v in sorted((self.suites.get(s) or {}).items())
                   if isinstance(v, (int, float)) and math.isfinite(v)}
            if row:
                suites[s] = row
        out: dict = {"rule": self.rule, "computed": bool(self.computed),
                     "passed": self.passed, "reason": self.reason,
                     "tolerance": float(self.tolerance),
                     "min_win_rate": float(self.min_win_rate), "suites": suites}
        for k in ("avg_ratio", "worst_lcb", "win_rate"):
            v = getattr(self, k)
            if math.isfinite(v):
                out[k] = float(v)
        return out


def uncomputable_bench_gate(reason: str, *, rule: str = RULE_V1,
                            tolerance: float = DEFAULT_TOLERANCE,
                            min_win_rate: float = DEFAULT_MIN_WIN_RATE) -> BenchGateResult:
    return BenchGateResult(computed=False, passed=None, reason=reason, rule=rule,
                           tolerance=float(tolerance), min_win_rate=float(min_win_rate))


def checkpoint_fault_bench_gate(reason: str, *, rule: str = RULE_V1,
                                tolerance: float = DEFAULT_TOLERANCE,
                                min_win_rate: float = DEFAULT_MIN_WIN_RATE) -> BenchGateResult:
    """DEC-CA-0063 item 10: the challenger's sweep reproduced a CHECKPOINT
    fault (the signed report's ``checkpoint_fault``) — the gate is computed
    and FAILS (a loss), with no suite comparisons."""
    return BenchGateResult(computed=True, passed=False,
                           reason=f"checkpoint fault: {reason}", rule=rule,
                           tolerance=float(tolerance), min_win_rate=float(min_win_rate))


def _six_score(scores: Mapping[str, float] | None) -> float | None:
    if not scores:
        return None
    vals = [scores.get(k) for k in _SIX]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in vals):
        return None
    return math.exp(sum(math.log(float(v)) for v in vals) / len(vals))


def _item_scores(paired, suite: str) -> tuple[list[float], list[float]]:
    """Per-item geomean(CRPS, MASE) over items present in both metrics."""
    kc, cc = paired.public[f"{suite}_crps"]
    km, cm = paired.public[f"{suite}_mase"]
    crps = dict(zip(paired.items[f"{suite}_crps"], zip(kc, cc, strict=True), strict=True))
    mase = dict(zip(paired.items[f"{suite}_mase"], zip(km, cm, strict=True), strict=True))
    king, chal = [], []
    for item in sorted(set(crps) & set(mase)):
        (k1, c1), (k2, c2) = crps[item], mase[item]
        king.append(math.sqrt(k1 * k2))
        chal.append(math.sqrt(c1 * c2))
    return king, chal


def evaluate_bench_gate(
    king_rows: Mapping[str, Mapping] | None,
    chal_rows: Mapping[str, Mapping] | None,
    king_scores: Mapping[str, float] | None,
    chal_scores: Mapping[str, float] | None,
    *,
    alpha: float,
    B: int,
    seed: int | str,
    min_items: int = 10,
    rule: str = RULE_V1,
    king_revisions: Mapping[str, str] | None = None,
    chal_revisions: Mapping[str, str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
) -> BenchGateResult:
    """Judge one challenger against the king under ``rule``.

    ``*_rows`` are :func:`cascade.eval.benchmarks.item_rows_from_report`
    results; ``*_scores`` the suite-level numbers
    (:func:`~cascade.eval.benchmarks.extract_bench_scores`: six, plus
    ``fev_crps`` / ``fev_mase``; the DEC-CA-0061 rule reads the rows only
    and records the six-number scores for the log). ``tolerance`` /
    ``min_win_rate`` are DEC-CA-0061 rule v2's (``[scoring]
    bench_gate_tolerance`` / ``bench_gate_min_win_rate``; ignored by
    DEC-CA-0054). ``*_revisions`` (DEC-CA-0063
    validator reports: the sidecar's pinned data revision per suite) — when
    both are given, a suite both name on different revisions makes the gate
    uncomputable (king and challenger must be judged on identical public
    data, the GIFT gate's rule). ``None`` / empty = unchecked (every
    trainer-report caller, byte-identical)."""
    if king_revisions and chal_revisions:
        diff = sorted(s for s in set(king_revisions) & set(chal_revisions)
                      if str(king_revisions[s]) != str(chal_revisions[s]))
        if diff:
            return uncomputable_bench_gate(
                "data-revision mismatch on " + ", ".join(
                    f"{s} (king {king_revisions[s]} != chal {chal_revisions[s]})" for s in diff),
                rule=rule, tolerance=tolerance, min_win_rate=min_win_rate)
    if rule == RULE_V2:
        return _evaluate_v2(king_rows, chal_rows, king_scores, chal_scores,
                            alpha=alpha, B=B, seed=seed, min_items=min_items,
                            tolerance=float(tolerance), min_win_rate=float(min_win_rate))
    if rule != RULE_V1:
        raise ValueError(f"unknown bench gate rule {rule!r}; expected one of {RULES}")
    ks, cs = _six_score(king_scores), _six_score(chal_scores)
    if ks is None or cs is None:
        return uncomputable_bench_gate("six-number bench scores missing")
    try:
        paired = pair_public_rows(king_rows, chal_rows, benchmarks=SUITES, require=SUITES)
    except MissingPublicGuard as e:
        return BenchGateResult(computed=False, passed=None, reason=str(e),
                               king_score=ks, chal_score=cs)
    ucb: dict[str, float] = {}
    ns: dict[str, int] = {}
    for suite in SUITES:
        k, c = _item_scores(paired, suite)
        ns[suite] = len(k)
        if len(k) < min_items:
            return BenchGateResult(computed=False, passed=None,
                                   reason=f"{suite}: {len(k)} paired items < {min_items}",
                                   suite_n=ns, king_score=ks, chal_score=cs)
        ucb[suite] = paired_geomean_lcb(k, c, alpha=1.0 - alpha / len(SUITES), B=B,
                                        seed=f"{seed}|bench|{suite}")
    worse = [s for s in SUITES if ucb[s] < 0.0]
    if worse:
        reason = f"significantly worse on {', '.join(worse)}"
    elif not cs < ks:
        reason = f"six-number geomean not better ({cs:.5f} >= {ks:.5f})"
    else:
        reason = ""
    return BenchGateResult(computed=True, passed=not reason, reason=reason,
                           suite_ucb=ucb, suite_n=ns, king_score=ks, chal_score=cs)


def _suite_numbers(scores: Mapping[str, float] | None, suite: str) -> tuple[float, float] | None:
    """``(crps, mase)`` suite-level for ``suite``, or ``None`` when either is
    absent, non-finite or <= 0."""
    if not scores:
        return None
    out = []
    for m in ("crps", "mase"):
        v = scores.get(f"{SCORE_PREFIX[suite]}_{m}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            return None
        out.append(float(v))
    return out[0], out[1]


def strict_eight(king_scores: Mapping[str, float] | None,
                 chal_scores: Mapping[str, float] | None) -> dict[str, bool] | None:
    """REPORT-ONLY (no consensus path reads it): the dropped 2026-10-19
    DEC-CA-0061 comparison — per suite-level number of GIFT-Eval, BOOM, TIME
    and FEV, is the challenger's strictly below the king's? ``{"gifteval_crps":
    True, …}`` (eight keys), or ``None`` when a number is missing on either
    side. ``scripts/bench_seed_noise.py`` counts how often seed noise flips
    one."""
    out: dict[str, bool] = {}
    for suite in SUITES_V2:
        k, c = _suite_numbers(king_scores, suite), _suite_numbers(chal_scores, suite)
        if k is None or c is None:
            return None
        out[f"{SCORE_PREFIX[suite]}_crps"] = bool(c[0] < k[0])
        out[f"{SCORE_PREFIX[suite]}_mase"] = bool(c[1] < k[1])
    return out


def task_log_ratios(paired, suite: str) -> list[float]:
    """Per TASK (one item row, in the pairing's sorted id order) of ``suite``:
    ``log s = (log r_crps + log r_mase) / 2`` with ``r = chal / king`` — over
    the tasks whose CRPS AND MASE are usable on both sides (the pairing
    already masked zero / non-finite values; the DEC-CA-0009 rule)."""
    kc, cc = paired.public[f"{suite}_crps"]
    km, cm = paired.public[f"{suite}_mase"]
    crps = dict(zip(paired.items[f"{suite}_crps"], zip(kc, cc, strict=True), strict=True))
    mase = dict(zip(paired.items[f"{suite}_mase"], zip(km, cm, strict=True), strict=True))
    out = []
    for item in sorted(set(crps) & set(mase)):
        (k1, c1), (k2, c2) = crps[item], mase[item]
        out.append(0.5 * ((math.log(c1) - math.log(k1)) + (math.log(c2) - math.log(k2))))
    return out


def suite_win_rate(log_ratios: list[float]) -> float:
    """Share of tasks the challenger wins (``log s < 0``); an exact tie
    (``log s == 0``) counts half. NaN when empty."""
    if not log_ratios:
        return float("nan")
    wins = sum(1.0 if x < 0.0 else 0.5 if x == 0.0 else 0.0 for x in log_ratios)
    return wins / len(log_ratios)


def combine_suites(suites: Mapping[str, Mapping[str, float]], *, tolerance: float,
                   min_win_rate: float) -> tuple[float, float, float, str]:
    """CONSENSUS: the three headline numbers of DEC-CA-0061 rule v2 from the
    per-suite ``{"ratio", "ratio_lcb", "win_rate"}`` and the failure reason
    (``""`` = pass). Suite-equal: ``avg_ratio = geomean_s(ratio)``,
    ``win_rate = mean_s(win_rate)``; ``worst_lcb = max_s(ratio_lcb)``. Shared
    by the gate, the king check and the audit's consistency replay."""
    names = [s for s in SUITES_V2 if s in suites]
    avg = math.exp(math.fsum(math.log(float(suites[s]["ratio"])) for s in names) / len(names))
    worst = max(float(suites[s]["ratio_lcb"]) for s in names)
    win = math.fsum(float(suites[s]["win_rate"]) for s in names) / len(names)
    reasons = []
    if not avg < 1.0:
        reasons.append(f"average ratio {avg:.5f} not below 1")
    regress = [s for s in names if float(suites[s]["ratio_lcb"]) > 1.0 + tolerance]
    if regress:
        reasons.append("significant regression on " + ", ".join(
            f"{s} (ratio LCB {float(suites[s]['ratio_lcb']):.5f} > {1.0 + tolerance:.5f})"
            for s in regress))
    if not win >= min_win_rate:
        reasons.append(f"win rate {win:.4f} < {min_win_rate:.4f}")
    return avg, worst, win, "; ".join(reasons)


def detail_verdict(detail: Mapping) -> bool | None:
    """The DEC-CA-0061 outcome a recorded ``detail`` itself implies: the
    three numbers re-derived from the recorded per-suite numbers must equal
    the recorded ones, and the thresholds decide. ``None`` when the detail
    says uncomputable; ``False`` when it is malformed."""
    if not detail.get("computed"):
        return None
    suites = detail.get("suites") or {}
    try:
        if any(not {"ratio", "ratio_lcb", "win_rate"} <= set(suites.get(s) or {})
               for s in SUITES_V2):
            return False
        tol, mwr = float(detail["tolerance"]), float(detail["min_win_rate"])
        avg, worst, win, reason = combine_suites(suites, tolerance=tol, min_win_rate=mwr)
        for k, v in (("avg_ratio", avg), ("worst_lcb", worst), ("win_rate", win)):
            if not math.isclose(float(detail[k]), v, rel_tol=1e-9, abs_tol=1e-12):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return not reason


def _evaluate_v2(king_rows, chal_rows, king_scores, chal_scores, *, alpha: float, B: int,
                 seed: int | str, min_items: int, tolerance: float,
                 min_win_rate: float) -> BenchGateResult:
    """DEC-CA-0061 rule v2 (module docstring): four suites, unit = one task,
    the suite-equal average ratio below 1, no suite's ratio LCB above
    ``1 + tolerance`` at ``alpha / 4``, the suite-equal win rate at least
    ``min_win_rate``. Missing rows (FEV included) or fewer than
    ``min_items`` usable tasks in a suite are uncomputable, exactly like a
    missing suite under DEC-CA-0054. Only tasks both reports scored count
    (the inner join), so a king swept on another day or a partial suite is
    compared like with like; masked tasks are recorded per suite."""
    ks, cs = _six_score(king_scores), _six_score(chal_scores)
    ks = float("nan") if ks is None else ks
    cs = float("nan") if cs is None else cs
    common = dict(rule=RULE_V2, king_score=ks, chal_score=cs, tolerance=tolerance,
                  min_win_rate=min_win_rate)
    for suite in SUITES_V2:
        kr, cr = (king_rows or {}).get(suite) or {}, (chal_rows or {}).get(suite) or {}
        if kr.get("partial") or cr.get("partial"):
            return BenchGateResult(computed=False, passed=None,
                                   reason=f"{suite}: partial benchmark report", **common)
        if kr.get("status") != "ok" or cr.get("status") != "ok":
            continue  # whole absent/unavailable sweep remains an infrastructure hold
        king_items, king_dup = _index(kr.get("rows") or [])
        chal_items, chal_dup = _index(cr.get("rows") or [])
        for item, row in king_items.items():
            if item in king_dup or not all(_usable(row.get(m)) for m in METRICS):
                continue
            candidate = chal_items.get(item) or {}
            if item in chal_dup or not all(_usable(candidate.get(m)) for m in METRICS):
                # A candidate cannot improve its result by omitting its worst tasks.
                return BenchGateResult(computed=True, passed=False,
                                       reason=f"{suite}: challenger failed required task {item}", **common)
    try:
        paired = pair_public_rows(king_rows, chal_rows, benchmarks=SUITES_V2, require=SUITES_V2)
    except MissingPublicGuard as e:
        return BenchGateResult(computed=False, passed=None, reason=str(e), **common)
    suites: dict[str, dict[str, float]] = {}
    ucb: dict[str, float] = {}
    ns: dict[str, int] = {}
    for suite in SUITES_V2:
        logs = task_log_ratios(paired, suite)
        k, c = _item_scores(paired, suite)
        # every task either report lists (the pairing's union) minus the
        # usable ones: missing on one side, duplicated, zero / non-finite
        union = len(paired.items[f"{suite}_crps"]) + int(paired.dropped[f"{suite}_crps"])
        masked = union - len(logs)
        ns[suite] = len(logs)
        row: dict[str, float] = {"n": len(logs), "masked": masked}
        suites[suite] = row
        if union and masked / union > 0.1:
            return BenchGateResult(computed=False, passed=None, suites=suites, suite_n=ns,
                                   reason=f"{suite}: more than 10% of tasks masked", **common)
        if len(logs) < min_items:
            return BenchGateResult(computed=False, passed=None, suites=suites, suite_n=ns,
                                   reason=f"{suite}: {len(logs)} usable tasks < {min_items}"
                                          f" ({masked} masked)", **common)
        ucb[suite] = paired_geomean_lcb(k, c, alpha=1.0 - alpha / len(SUITES_V2), B=B,
                                        seed=f"{seed}|bench|{suite}")
        row.update(ratio=math.exp(math.fsum(logs) / len(logs)),
                   ratio_lcb=1.0 - ucb[suite], win_rate=suite_win_rate(logs))
    avg, worst, win, reason = combine_suites(suites, tolerance=tolerance,
                                             min_win_rate=min_win_rate)
    return BenchGateResult(computed=True, passed=not reason, reason=reason, suite_ucb=ucb,
                           suite_n=ns, suites=suites, avg_ratio=avg, worst_lcb=worst,
                           win_rate=win, **common)
