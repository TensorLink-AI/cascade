"""Fault attribution as data: ONE table of who a leg failure is billed to.

Retro 2026-11-28 (class C6, plan item P7): the fault class of a failed leg was
decided at each failure site by string matching, and the unclear default was
the MINER (an "infra" burn of their attempt, or "generator"). Alemaj's lane C
leg was told a NaN of OUR adapter was their recipe; a lemon pod and our own
throughput watchdog were charged to the miner's attempts.

The policy this module encodes:

* every worker-exit verdict comes from :data:`RULES`, an ordered table — the
  first matching row decides; nothing else string-matches a worker exit;
* a row is ``miner`` only on POSITIVE evidence it names (the worker's own
  rejection exit, a fund-time fit check that vouched for the recipe, ...);
* anything no row recognises is ``unknown`` and resolves to the OPERATOR:
  requeued, never burned, make-good eligible (:data:`UNKNOWN_RULE`). The
  entry's operator-fault cap (``[round] operator_fault_cap``) still bounds
  the requeues, so an unknown fault cannot loop forever;
* a miner verdict must cite a rule. :func:`enforce_miner_citation` is the
  gate the verdict recorders (``_record_funded_failure``, ``LegFailure``) run:
  a ``miner_fault=True`` verdict whose error class no miner row cites is
  re-attributed to the operator and logged at ERROR.

Stdlib only; the matchers are pure functions of :class:`FailureFacts`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("cascade.trainer")

MINER, OPERATOR, UNKNOWN = "miner", "operator", "unknown"
PARTIES = frozenset({MINER, OPERATOR, UNKNOWN})

# Fund-time fit evidence an entry carries (``FundedEntry.extra["fit_evidence"]``
# for lane C; derived for lane B): "fit" = the intake's memory estimate said
# the recipe fits the GPU it rents; "own_risk" = the miner chose a GPU the
# check could not vouch for and was warned an OOM ends the entry; "" = no
# check vouched (an older entry, an unreadable recipe, no estimate).
FIT_OK, FIT_OWN_RISK = "fit", "own_risk"
FIT_EVIDENCE = frozenset({FIT_OK, FIT_OWN_RISK})


@dataclass(frozen=True)
class FailureFacts:
    """What a classifier may look at for one failed worker exit."""

    rc: int | None = None
    text: str = ""
    final_exception: str = ""     # the dispatcher-parsed final traceback line
    role: str = ""
    stalled_before: bool = False
    shards_leg: bool = False
    fit_evidence: str = ""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    party: str                    # MINER | OPERATOR | UNKNOWN (resolves to operator)
    error_class: str              # the queue's last_error_class (public_reason keys on it)
    burn: bool                    # consumes one of the entry's infra attempts
    evidence: str                 # the positive evidence the row requires
    reason: str                   # one-line, operator-side description
    match: Callable[[FailureFacts], bool] | None = None   # None = class citation only


@dataclass(frozen=True)
class Attribution:
    """One failure's verdict, with the rule that produced it."""

    party: str
    error_class: str
    burn: bool
    rule_id: str
    evidence: str

    @property
    def miner_fault(self) -> bool:
        return self.party == MINER

    @property
    def make_good_eligible(self) -> bool:
        """Lost on OUR side (or unknown, which is ours by policy)."""
        return self.party != MINER

    @property
    def unknown(self) -> bool:
        return self.party == UNKNOWN

    def verdict(self) -> tuple[bool, str, bool]:
        """The legacy ``(miner_fault, error_class, burn)`` triple."""
        return self.miner_fault, self.error_class, self.burn

    def tag(self) -> str:
        """The marker appended to the recorded failure text (the trainer log
        and ``last_error``; never interpolated into a public reason)."""
        return f"[rule {self.rule_id}]"


# ── matchers (pure) ─────────────────────────────────────────────────────────

def _faults():
    from . import faults
    return faults


def _nonfinite(f: FailureFacts) -> bool:
    return _faults().nonfinite_training(f.text)


def _stall(f: FailureFacts) -> bool:
    return f.rc == 3 and _faults()._STALL_MARKER in (f.text or "")


_UNCUTTABLE_RE = re.compile(r"could not cut an? \S+ training sample")


def _uncuttable(f: FailureFacts) -> bool:
    return f.rc == 3 and bool(_UNCUTTABLE_RE.search(f.text or ""))


def _own_oom(f: FailureFacts) -> bool:
    fl = _faults()
    return f.role in fl.MINER_SIZED_ROLES and fl.own_run_cuda_oom(f.final_exception)


def _shard_fetch(f: FailureFacts) -> bool:
    return f.shards_leg and f.rc != 3 and _faults().shard_fetch_failure(f.final_exception)


# Pod / provider / our-watchdog failures the trainer itself reports: a pod that
# never became ready (a lemon), a short pod, our throughput watchdog, a
# transport loss. Host-side or ours: never the miner's recipe.
_HOST_MARKERS = (
    "THROUGHPUT ABORT", "not ready within", "never became ready", "short pod",
    "pod unreachable", "Connection refused", "detached launch failed",
    "first-step deadline", "NCCL preflight",
)


def _host(f: FailureFacts) -> bool:
    text = f.text or ""
    return f.rc != 3 and any(m in text for m in _HOST_MARKERS)


def _stall_first(f: FailureFacts) -> bool:
    return _stall(f) and not f.stalled_before


def _stall_again(f: FailureFacts) -> bool:
    return _stall(f) and f.stalled_before


def _rejected(f: FailureFacts) -> bool:
    return f.rc == 3


def _oom_fit(f: FailureFacts) -> bool:
    return _own_oom(f) and f.fit_evidence in FIT_EVIDENCE


def _oom_unvouched(f: FailureFacts) -> bool:
    return _own_oom(f) and f.fit_evidence not in FIT_EVIDENCE


# ── the table ───────────────────────────────────────────────────────────────
# Order matters: the first matching row decides. Classes are the queue's
# (cascade.funding.queue / public_reason templates).

_OP = "operator"          # funding.queue.OPERATOR_FAULT_CLASS (imported lazily below)

RULES: tuple[Rule, ...] = (
    Rule("O-NONFINITE", OPERATOR, _OP, False,
         "the MarketCast worker's own 'non-finite training loss/gradient at step N' line",
         "recipe numbers are bounded at parse time; a NaN is our adapter/pipeline "
         "(2026-11-28 flat-context sqrt(0))", _nonfinite),
    Rule("O-STALL-FIRST", OPERATOR, "stall", False,
         "rc 3 generator_stalled, first time for this hotkey",
         "a stall also looks like a stale worker image: unburned the first time",
         _stall_first),
    Rule("M-STALL-REPEAT", MINER, "generator", False,
         "rc 3 generator_stalled AND the same hotkey's previous leg stalled",
         "the generator stalls on two different pods", _stall_again),
    Rule("M-GEOMETRY", MINER, "recipe", False,
         "rc 3 'could not cut a <suite>/<horizon> training sample' (no sample of "
         "context+horizon bars exists in the recipe's data range)",
         "the recipe's data range cannot produce its training samples", _uncuttable),
    Rule("M-REJECTED", MINER, "generator", False,
         "the worker's rc 3 'miner submission rejected' exit (CorpusError / MktLegError)",
         "the worker rejected the miner's submission", _rejected),
    Rule("M-OOM-FIT", MINER, "gpu_oom", False,
         "a CUDA OOM of the leg's OWN run (>= half the device) on a miner-sized role, "
         "AND fund-time fit evidence ('fit' or an own-risk warning)",
         "the recipe outgrew a GPU it was checked (or warned) against", _oom_fit),
    Rule("O-OOM-UNVOUCHED", OPERATOR, _OP, False,
         "a CUDA OOM of the leg's own run with NO fund-time fit evidence",
         "nothing vouched the recipe would fit the GPU we rented: ours", _oom_unvouched),
    Rule("O-SHARD-FETCH", OPERATOR, _OP, False,
         "a shards leg's final exception is our shard download / sha256 failure",
         "our storage boundary (the operator tier wrote the corpus)", _shard_fetch),
    Rule("O-HOST", OPERATOR, _OP, False,
         "the trainer's own pod / watchdog / transport text (lemon, short pod, "
         "THROUGHPUT ABORT, unreachable)",
         "a provider host or our watchdog: never the miner's recipe", _host),
)

UNKNOWN_RULE = Rule(
    "U-DEFAULT", UNKNOWN, _OP, False, "none (no row matched)",
    "unclassified: OPERATOR by policy — requeued, unburned, make-good eligible")

# Miner verdicts raised at explicit sites (not worker exits). Each cites the
# evidence that site holds; a miner_fault=True verdict of any other class is
# re-attributed to the operator (:func:`enforce_miner_citation`).
MINER_CITATIONS: tuple[Rule, ...] = (
    Rule("M-TAMPER", MINER, "tamper", False,
         "the pod failed its identity pin (host key / container / checkpoint listing)",
         "the miner's pod changed under us"),
    Rule("M-AUTH", MINER, "auth", False,
         "the payer's provider key refused (401/402/403) or absent from the vault",
         "the miner's credential"),
    Rule("M-PAYER-ACCOUNT", MINER, "payer_account", False,
         "the payer's account lists nothing the operator's listing shows (2 controls)",
         "the miner's provider account"),
    Rule("M-CKPT-UNVERIFIABLE", MINER, "checkpoint_unverifiable", False,
         "the checkpoint differs from the pod's own listing after a re-harvest; "
         "terminal, unburned, the owner decides any burn",
         "evidence kept for the owner"),
    Rule("M-GENERATOR", MINER, "generator", False,
         "the worker rejected the generator (its rc 3 exit) or the harvested "
         "checkpoint's weights are NaN/inf on the miner's corpus (CheckpointDiverged)",
         "the miner's generator / corpus"),
    Rule("M-OWN-RISK-GPU", MINER, "own_risk_gpu_failed", False,
         "the miner chose an unmeasured GPU at their own risk (warned at the fund)",
         "own-risk GPU"),
    Rule("M-RECIPE", MINER, "recipe", True,
         "the recipe failed its parse / embargo / geometry check (RecipeError)",
         "the miner's recipe"),
    Rule("M-INFEASIBLE", MINER, "infeasible", False,
         "the pre-rent feasibility check (cascade.funding.feasibility, #1007) refused a "
         "miner-side choice — GPU arch vs the pinned torch, recipe memory, data "
         "geometry — BEFORE anything was rented (entry released, nothing charged)",
         "the miner's own choice cannot run as funded"),
    Rule("M-TARGET", MINER, "target", True,
         "the target is not open for this lane", "the miner's target choice"),
    Rule("M-ENTRY-CLOSED", MINER, "competition_ended", False,
         "the competition closed before the leg could train (ends the entry, "
         "nothing charged)", "entry closed, not billed"),
    Rule("M-LANE-CLOSED", MINER, "lane_ended", False,
         "the lane closed before the leg could train (ends the entry, nothing charged)",
         "entry closed, not billed"),
)

_CITE_BY_CLASS: dict[str, str] = {}
for _r in (*MINER_CITATIONS, *RULES):     # a class's generic citation first
    if _r.party == MINER:
        _CITE_BY_CLASS.setdefault(_r.error_class, _r.rule_id)


def miner_rule_for_class(error_class: str) -> str:
    """The rule id that may cite a miner verdict of ``error_class`` ("" = none)."""
    return _CITE_BY_CLASS.get(str(error_class or ""), "")


def operator_class() -> str:
    from ..funding.queue import OPERATOR_FAULT_CLASS
    return OPERATOR_FAULT_CLASS


def attribute(facts: FailureFacts) -> Attribution:
    """The verdict of one failed worker exit: the first matching row of
    :data:`RULES`, else :data:`UNKNOWN_RULE` (operator by policy)."""
    for rule in RULES:
        if rule.match is not None and rule.match(facts):
            return _attr(rule)
    return _attr(UNKNOWN_RULE)


def _attr(rule: Rule) -> Attribution:
    cls = operator_class() if rule.error_class == _OP else rule.error_class
    return Attribution(rule.party, cls, rule.burn, rule.rule_id, rule.evidence)


def enforce_miner_citation(msg: str, miner_fault: bool, error_class: str,
                           burn: bool) -> tuple[str, bool, str, bool]:
    """The gate every verdict recorder runs: a miner verdict keeps its class
    only when a rule cites it (its tag is appended to ``msg`` once); one no
    rule cites is re-attributed to the OPERATOR (unburned) and logged."""
    if not miner_fault:
        return msg, miner_fault, error_class, burn
    rid = miner_rule_for_class(error_class)
    if not rid:
        log.error("fault attribution: miner verdict of class %r cites no rule — "
                  "attributed to the OPERATOR (unburned, make-good eligible): %s",
                  error_class, str(msg)[-200:])
        return (f"{msg} [rule {UNKNOWN_RULE.rule_id}: uncited miner class "
                f"{error_class!r}]", False, operator_class(), False)
    if "[rule " not in str(msg):
        msg = f"{msg} [rule {rid}]"
    return msg, miner_fault, error_class, burn


def fit_evidence_of(entry, role: str) -> str:
    """Fund-time fit evidence of a queue ``entry`` for a ``role`` leg.

    Lane C (``mkt_finetune`` / ``finetune``): what the intake recorded
    (``extra["fit_evidence"]``). Lane B (``expert``): an own-risk GPU
    (``gpu_measured`` False) is :data:`FIT_OWN_RISK`; a miner-set
    ``context_length`` lever on a measured GPU was fit-checked against its
    measured memory (``context_length_unfit``) — :data:`FIT_OK`; the default
    recipe on a measured GPU is OUR sizing ("")."""
    if entry is None:
        return ""
    if role == "expert":
        if not bool(getattr(entry, "gpu_measured", True)):
            return FIT_OWN_RISK
        levers = getattr(entry, "levers", None) or {}
        return FIT_OK if isinstance(levers, dict) and levers.get("context_length") else ""
    extra = getattr(entry, "extra", None) or {}
    ev = str(extra.get("fit_evidence", "") or "") if isinstance(extra, dict) else ""
    return ev if ev in FIT_EVIDENCE else ""


def table_rows() -> list[dict]:
    """The table as plain rows (docs, audits, tests)."""
    return [{"rule_id": r.rule_id, "party": r.party, "error_class": r.error_class,
             "burn": r.burn, "evidence": r.evidence, "reason": r.reason,
             "kind": "worker_exit" if r.match is not None else "citation"}
            for r in (*RULES, UNKNOWN_RULE, *MINER_CITATIONS)]


__all__ = ["Attribution", "FIT_EVIDENCE", "FIT_OK", "FIT_OWN_RISK", "FailureFacts",
           "MINER", "MINER_CITATIONS", "OPERATOR", "RULES", "UNKNOWN", "UNKNOWN_RULE",
           "attribute", "enforce_miner_citation", "fit_evidence_of", "miner_rule_for_class",
           "table_rows"]
