"""Public, sanitized one-line reasons and verdicts for the lane queues.

Owner 2026-11-28: a miner reading ``GET /v1/lane-b/queue`` (or the
``status/lane_*.json`` docs, the CLI, the website) should see WHY an entry
waits or stopped, and what a settled entry's verdict was, in plain English
with the actionable hint, never just ``no_capacity``.

Safety rule (the reason of this module): the trainer's ``last_error`` is
operator-side prose: it can carry pod names, IPs, provider replies, the
operator's balance, other miners' prices. NOTHING of it is passed through.
:func:`public_reason` maps the entry's ``last_error_class`` (plus a few fixed
substrings of the trainer's own message shapes) onto a WHITELIST of templates;
the only values a template interpolates are tokens that match strict
patterns (a GPU spec like ``4xA100``, an ISO date, a small integer, a
``suite/horizon`` slug). An unknown class falls back to a generic sentence.

:func:`public_verdict` likewise rebuilds a verdict dict from typed,
whitelisted fields only (outcome slug, round id digits, per-target finite
numbers). Display only: none of this is signed, digested or read by any
scoring path; the signed manifests and receipts are unchanged.

Stdlib only (the intake imports it).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

__all__ = ["MAKE_GOOD_WAIT_CODE", "MAKE_GOOD_WAIT_TEXT", "OUTCOME_TEXT", "judgement_margins",
           "public_reason", "public_verdict", "verdict_line"]

_GPU_RE = re.compile(r"^[1-9][0-9]?x[A-Za-z0-9]{2,20}$")
_DATE_RE = re.compile(r"\b(20[0-9]{2}-[01][0-9]-[0-3][0-9])\b")
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")
_OUTCOME_RE = re.compile(r"^[a-z_]{1,48}$")
_ROUND_RE = re.compile(r"^[0-9]{1,24}$")
_REF_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,256}$")

# Phases where the leg is actively progressing: a stale last_error from an
# EARLIER wait is no longer the reason (the phase is).
_ACTIVE_PHASE_TEXT = {
    "booting": "starting: GPU allocated, starting and checking the worker",
    "training": "training: your leg is running",
    "harvesting": "harvesting: checking and uploading the checkpoint",
    "evaluating": "evaluating: benchmark evaluation in progress",
    "awaiting_settlement": "trained: waiting for the next settlement to judge it",
}
_PHASE_REASON_TEXT = {
    "availability_check": "waiting: checking GPU availability for your choice",
    "market_capacity": "waiting: no GPU fits your type/price choice right now; retrying automatically",
    "king_priority": "waiting: king work has provisioning priority; your leg follows",
    "operator_recovery": "waiting: an operator make-good lane is being prepared (no cost to you)",
}
_WAIT_PHASE_TEXT = {
    "queued": "queued: waiting for a lane in reveal order",
    "admitted": "admitted: preparing your leg",
    "waiting_admission": "waiting: shared compute admission (one leg per coldkey at a time)",
    "preparing_data": "preparing: building your training data",
    "waiting_gpu": "waiting: GPU provisioning",
    "waiting_launch": "waiting: the lane has not activated yet; nothing has started",
    "waiting_operator": "waiting: operator GPU capacity for a make-good leg (no cost to you)",
}
_STATUS_TEXT = {
    "awaiting_operator_approval": "parked: the operator must approve paying for this leg; "
                                  "nothing was charged",
    "paused": "paused after repeated operator-side faults; nothing was burned, keep it funded",
    "paused_spend_cap": "paused: too much of your funding was lost to failed attempts; "
                        "your submission is kept (fund again to resume)",
    "paused_budget": "paused: your --max-total-spend cannot cover the next leg; "
                     "raise it and fund again",
    "withdrawn": "withdrawn: nothing further runs",
}

# A settled entry's outcome → plain English.
OUTCOME_TEXT = {
    "qualified": "qualified: took a target slot; merge acceptance and pay are separate",
    "qualification_not_selected": "passed the target checks, but another entry won the slot(s)",
    "qualification_rejected": "did not clear the qualification bar on any declared target",
    "merge_refused": "the expert merge was refused at settlement (byte check)",
    "leg_record_missing": "the leg's training record was missing at settlement (our side)",
    "stale": "dropped unjudged: the leg's base king was too old; requeued at no cost",
    "routing_stale": "dropped unjudged: the king re-routed the domain; requeued at no cost",
    "accepted": "accepted into the warm start",
    "acceptance_rejected": "warm-start acceptance rejected the contribution",
    "published": "published for forward-only judging; the win is decided by later settlements",
    "confirmed": "confirmed: took the target slot",
    "not_passed": "the forward evaluation did not pass",
    "rejected": "rejected by forward evaluation/confirmation",
    "pending": "passed initially; a later settlement must confirm it",
    "deferred": "forward evaluation deferred (missing evidence is not a loss)",
    "dropped": "dropped from forward judging",
    "not_selected": "passed initial judging; another entry advanced",
}


def _gpu(value) -> str:
    text = str(value or "").strip()
    return text if _GPU_RE.match(text) else ""


def _int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 0 <= n < 10**12 else None


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return round(v, 6) if math.isfinite(v) else None


def _get(entry, name, default=""):
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _error_reason(cls: str, err: str, entry) -> tuple[str, str]:
    """``(code, text)`` for a classified failure — templates only."""
    low = (err or "").lower()
    gpu = _gpu(_get(entry, "gpu"))
    if not gpu:
        m = re.search(r"\bno ([1-9][0-9]?x[A-Za-z0-9]{2,20}) capacity\b", err or "")
        gpu = _gpu(m.group(1)) if m else ""
    on = f" on {gpu}" if gpu else ""
    if cls == "no_capacity":
        if "no fallback" in low or ("miner-chosen" in low and not _get(entry, "gpu_allow_fallback")):
            what = gpu or "the GPU type you chose"
            return ("gpu_unavailable_no_fallback",
                    f"waiting: no {what} available on your GPU provider and your fund allows no "
                    "fallback GPU: re-fund with --allow-fallback or another --gpu "
                    "(nothing rented, no attempt charged)")
        if "coldkey compute slot" in low or "hotkey busy" in low:
            return ("coldkey_busy", "waiting: another leg of your coldkey is still running; this "
                                    "one starts when it finishes (nothing rented)")
        if "coldkey identity" in low:
            return ("chain_metadata", "waiting: fresh chain metadata for your coldkey; retrying "
                                      "automatically")
        if "operator lane" in low:
            return ("operator_lane_busy", "waiting: no operator lane free before the latest "
                                          "safe start; retrying next window (no cost to you)")
        if "era ended" in low or "after its era ended" in low or "unjudged at settlement" in low:
            return ("missed_settlement", "requeued: the leg missed its settlement window; it "
                                         "retrains on the current king at no attempt cost")
        if "measured" in low and "step rate" in low:
            return ("no_measured_gpu", "waiting: no lane GPU type has a measured step rate on "
                                       "the pinned image yet (operator side; nothing rented)")
        if "gpu_mismatch" in low:
            return ("gpu_mismatch", "requeued: the leg trained on a different GPU type than "
                                    "the king; it retrains at no attempt cost")
        return ("no_capacity", "waiting: no fitting GPU capacity right now; retrying "
                               "automatically (nothing rented, no attempt charged)")
    if cls == "rate_limited":
        return ("rate_limited", "waiting: your GPU provider key is rate-limited; retrying "
                                "with backoff")
    if cls == "auth":
        if "vaulted key" in low or "expired" in low:
            return ("key_expired", "your GPU provider key expired or is missing here: fund "
                                   "again with a fresh key (no attempt charged)")
        return ("key_refused", "your GPU provider key was refused (auth/permission): fix the "
                               "key and fund again")
    if cls == "payer_balance":
        return ("payer_balance", "top up your GPU provider balance: the next leg does not fit "
                                 "it (nothing rented, no attempt charged)")
    if cls == "payer_account":
        return ("payer_account", "your GPU provider account lists no executor of your chosen "
                                 "type under your price cap: raise --max-price or choose "
                                 "another --gpu")
    if cls in ("payer_pod_lost", "payer_interference"):
        return (cls, "your pod ended mid-leg on your provider account (stopped, cancelled or "
                     "out of balance): requeued on your key")
    if cls == "gpu_oom":
        return ("gpu_oom", f"out of GPU memory{on or ' on the rented GPU'}: reduce batch/"
                           "context/LoRA rank or choose a larger GPU (the entry ended; it is "
                           "not re-rented)")
    if cls == "own_risk_gpu_failed":
        return ("own_risk_gpu_failed", f"the leg failed on your unmeasured (own-risk) GPU{on}: "
                                       "no refund; choose a measured --gpu")
    if cls in ("operator", "operator_fault", "operator_transient", "operator_disk", "infra_operator",
               "stall", "owner_makegood", "owner_unburn", "operator_rerun"):
        if "non-finite training" in low:
            return ("operator_nonfinite", "our fault (non-finite training on our side): "
                                          "requeued at no cost")
        return ("operator_fault", "our fault: requeued at no cost")
    if cls == "operator_blocked":
        return ("operator_blocked", "waiting: operator-paid capacity is on hold; your leg "
                                    "resumes when it frees (no cost to you)")
    if cls == "operator_denied":
        return ("operator_denied", "the operator declined to pay for this make-good leg; it "
                                   "retries on your own key (no attempt charged)")
    if cls == "awaiting_operator_approval":
        return (cls, _STATUS_TEXT[cls])
    if cls == "infra":
        n = _int(_get(entry, "attempts", 0)) or 0
        return ("infra", f"a pod or transport failure ({n} retry attempt(s) used); retrying "
                         "automatically")
    if cls == "generator":
        if "diverged" in low or "non-finite" in low:
            return ("generator_diverged", "your generator's corpus made training diverge "
                                          "(non-finite values): this submission is spent")
        if "stalled" in low:
            return ("generator_stalled", "your generator stalled (no series for too long): "
                                         "this submission is spent")
        if "string constant" in low:
            return ("generator_static_guard", "static guard: your generator ships Python "
                                              "inside a string constant; every module must be "
                                              "a .py file")
        return ("generator", "your generator was rejected by the trainer: this submission "
                             "is spent")
    if cls == "recipe":
        if "embargo" in low:
            return ("recipe_embargo", "your recipe tree embeds embargoed market bars: remove "
                                      "the embargoed history and fund a new ref")
        if "could not cut" in low or "sample geometry" in low:
            m = _DATE_RE.search(err or "") if "data.start to" in low else None
            fix = (f"earliest data.start that fits: {m.group(1)}" if m else
                   "move data.start later (`cascade lane-c data` shows the recorded history)")
            return ("recipe_geometry", "your recipe's data range cannot cut its training "
                                       f"samples: {fix}")
        return ("recipe", "your recipe was rejected: fix it and fund a new ref")
    if cls == "target":
        return ("target", "your target is not open for this lane: choose another target")
    if cls in ("competition_ended", "lane_ended"):
        return (cls, "the competition/lane closed before this entry could train; nothing "
                     "was charged")
    if cls == "blocked":
        return ("blocked", "this hotkey is on the operator's admission denylist")
    if cls == "burned":
        return ("burned", "this hotkey already used its one lifetime submission: register a "
                          "new hotkey")
    if cls == "duplicate":
        return ("duplicate", "your generator duplicates an earlier commit (the earliest "
                             "commit keeps the seat)")
    if cls == "funding_expired":
        return ("funding_expired", "funding expired before the leg started: fund again")
    if cls == "tamper":
        return ("tamper", "a pod identity / integrity check failed: the entry ended")
    if cls == "checkpoint_unverifiable":
        return (cls, "your pod's checkpoint could not be verified against the pod's own "
                     "listing: the entry ended unburned; the operator reviews the evidence")
    if cls in ("spend_cap", "budget"):
        return (cls, _STATUS_TEXT["paused_spend_cap" if cls == "spend_cap" else "paused_budget"])
    if cls == "price_cap_unavailable":
        return (cls, "a provider cannot enforce your price cap; nothing was rented through it")
    if cls == "commit_before_open":
        return (cls, "your reveal predates the competition opening: commit again and fund it")
    if cls == "superseded":
        return (cls, "a newer commitment replaced this submission")
    return ("unclassified", "the last attempt failed with an unclassified error; ask the "
                            "operator (no detail is published)")


# Owner 2026-11-28 ("add make goods", lane C 5Cu2sxndK8Dw): an entry parked for
# the owner's approval of an operator make-good — a leg lost on OUR side,
# re-run on the operator's account — says so, never "the operator must
# approve paying" (which reads like the miner's problem).
MAKE_GOOD_WAIT_CODE = "operator_rerun_pending"
MAKE_GOOD_WAIT_TEXT = "our fault: waiting for an operator re-run, no cost to you"


def _make_good_wait(entry, err: str) -> bool:
    """An approval wait that is an operator make-good: the entry is already on
    the operator's bill (``operator_billed``: lost on our side) or the trainer's
    own park message names a make-good (a fixed shape, never echoed)."""
    if bool(_get(entry, "operator_billed", False)):
        return True
    low = (err or "").lower()
    return "awaiting operator approval" in low and "make-good" in low


def public_reason(entry) -> tuple[str, str]:
    """``(reason_code, reason)`` for one queue entry (a ``FundedEntry`` or a
    row mapping with its fields). Never raises; never returns error text."""
    try:
        status = str(_get(entry, "status") or "")
        phase = str(_get(entry, "phase") or "")
        cls = str(_get(entry, "last_error_class") or "")
        err = str(_get(entry, "last_error") or "")
        if status == "done":
            v = _get(entry, "verdict", None)
            line = verdict_line(v) if isinstance(v, Mapping) else ""
            return ("settled", f"settled: {line}" if line else "settled: queue work completed; "
                    "the verdict is reported separately")
        if status == "awaiting_operator_approval" and _make_good_wait(entry, err):
            return (MAKE_GOOD_WAIT_CODE, MAKE_GOOD_WAIT_TEXT)
        if status in _STATUS_TEXT:
            return (status, _STATUS_TEXT[status])
        if status == "failed":
            return _error_reason(cls, err, entry) if cls else ("failed", "stopped")
        if status in ("in_flight", "in_round") and phase in _ACTIVE_PHASE_TEXT:
            return (phase, _ACTIVE_PHASE_TEXT[phase])
        if cls and cls != "denied":
            return _error_reason(cls, err, entry)
        pr = str(_get(entry, "phase_reason") or "")
        if pr in _PHASE_REASON_TEXT:
            return (pr, _PHASE_REASON_TEXT[pr])
        key = phase if phase in _WAIT_PHASE_TEXT else status
        if key in _WAIT_PHASE_TEXT:
            return (key, _WAIT_PHASE_TEXT[key])
        return ("", "")
    except Exception:  # noqa: BLE001 — a display field never breaks the queue view
        return ("", "")


# ── verdicts ────────────────────────────────────────────────────────────────


def judgement_margins(judgement, *, cap: int = 16) -> dict:
    """``{"stage", "targets": [...]}`` from one ``ExpertJudgement`` (duck-typed,
    no eval import): per declared (or passed) target the LCB, the bar, the
    point gain and the margin (LCB − bar). Display telemetry for the driver
    state; never signed."""
    rows = []
    try:
        verdicts = judgement.target_verdicts() or {}
    except Exception:  # noqa: BLE001
        verdicts = {}
    for t, v in verdicts.items():
        declared = bool(getattr(v, "declared", True))
        passed = bool(getattr(v, "passed", False))
        if not (declared or passed):
            continue
        reason = str(getattr(v, "reason", "") or "")
        inc = str(getattr(v, "inconclusive", "") or "")
        why = ("cleared" if passed else f"inconclusive_{inc}" if inc else
               "no_gain" if reason.startswith("no gain") else "guard" if reason else "")
        try:
            bar = _num(v.bar)                # the applied gain bar (published as gain_bar)
        except AttributeError:
            bar = None
        lcb = _num(getattr(v, "target_lcb", None))
        row = {"target": str(t), "declared": declared, "passed": passed, "why": why,
               "lcb": lcb, "gain_bar": bar, "gain": _num(getattr(v, "target_gain", None)),
               "metric": str(getattr(v, "gain_metric", "") or "")}
        row = {k: val for k, val in row.items() if val not in (None, "")}
        rows.append(row)
        if len(rows) >= cap:
            break
    return {"stage": str(getattr(judgement, "stage", "") or ""), "targets": rows}


def _target_row(raw) -> dict | None:
    if not isinstance(raw, Mapping):
        return None
    t = str(raw.get("target") or "")
    if not _SLUG_RE.match(t):
        return None
    row: dict = {"target": t}
    for k in ("declared", "passed"):
        if isinstance(raw.get(k), bool):
            row[k] = raw[k]
    for k in ("lcb", "gain_bar", "gain"):
        v = _num(raw.get(k))
        if v is not None:
            row[k] = v
    if "lcb" in row and "gain_bar" in row:
        row["margin"] = round(row["lcb"] - row["gain_bar"], 6)
    for k in ("why", "metric"):
        v = str(raw.get(k) or "")
        if v and _OUTCOME_RE.match(v):
            row[k] = v
    return row


def public_verdict(raw, *, lane: str = "") -> dict | None:
    """A verdict dict rebuilt from whitelisted, typed fields only (None when
    ``raw`` carries no usable outcome)."""
    if not isinstance(raw, Mapping):
        return None
    outcome = str(raw.get("outcome") or raw.get("stage") or "")
    if not _OUTCOME_RE.match(outcome):
        return None
    out: dict = {"outcome": outcome}
    lane = str(raw.get("lane") or lane or "")
    if lane in ("a", "b", "c", "d"):
        out["lane"] = lane
    rid = str(raw.get("round_id") or "")
    if _ROUND_RE.match(rid):
        out["round_id"] = rid
    rrid = str(raw.get("receipt_round_id") or "")
    if _ROUND_RE.match(rrid):
        out["receipt_round_id"] = rrid
    ref = str(raw.get("ref") or raw.get("gen_ref") or "")
    if _REF_RE.match(ref):
        out["ref"] = ref
    stage = str(raw.get("judge_stage") or "")
    if _OUTCOME_RE.match(stage):
        out["judge_stage"] = stage
    rows = [r for r in (_target_row(x) for x in (raw.get("margins") or [])[:16]) if r]
    if rows:
        out["margins"] = rows
    out["summary"] = OUTCOME_TEXT.get(outcome, outcome.replace("_", " "))
    return out


def verdict_line(verdict) -> str:
    """One plain line: ``<summary> (round R)``."""
    if not isinstance(verdict, Mapping):
        return ""
    outcome = str(verdict.get("outcome") or "")
    if not outcome:
        return ""
    text = OUTCOME_TEXT.get(outcome, outcome.replace("_", " "))
    rid = verdict.get("round_id") or verdict.get("receipt_round_id")
    return f"{outcome}: {text}" + (f" (round {rid})" if rid else "")
