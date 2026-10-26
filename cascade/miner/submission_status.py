"""Explain public submission observations without guessing capacity or verdicts."""

from __future__ import annotations

import json
import math
import re
import shlex
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from ..funding.queue import OPERATOR_FAULT_RETRIES
from .live_status import freshness

QUEUE_PATHS = {"a": "/v1/queue", "b": "/v1/lane-b/queue", "c": "/v1/lane-c/queue"}
PHASE_TEXT = {
    "unknown": "Unrecognized state; ask the operator to check this feed",
    "queued": "Queued for admission",
    "admitted": "Admitted; preparation has not yet been reported",
    "waiting_admission": "Waiting for shared compute admission",
    "waiting_launch": "Queued for v2 activation; training has not started",
    "preparing_data": "Preparing training data (includes CPU capacity waits)",
    "waiting_gpu": "Waiting for GPU provisioning",
    "booting": "GPU allocated; starting and checking the worker",
    "training": "Training worker dispatched; completion not yet reported",
    "harvesting": "Checking and uploading the checkpoint",
    "evaluating": "Benchmark evaluation in progress",
    "awaiting_settlement": "Trained; awaiting settlement/judging",
    "waiting_operator": "Waiting for configured operator GPU capacity",
    "in_flight": "Admitted; this deployment does not report the current stage",
    "in_round": "Selected for a round; current stage unreported",
    "done": "Queue work completed; inspect the verdict separately",
    "failed": "Entry stopped; action may be required",
    "withdrawn": "Entry withdrawn",
    "queued_behind": "New reference parked behind the current leg",
}
ERROR_TEXT = {
    "commit_before_open": "The reveal predates the competition opening. Make a fresh commitment and fund it; no training attempt was consumed.",
    "price_cap_unavailable": "A listed provider cannot enforce the requested price ceiling; nothing was rented through it.",
    "no_capacity": "No fitting capacity or admission window was available at the last retry.",
    "rate_limited": "The provider rate-limited the request.",
    "operator": "An operator/provider fault deferred the work.",
    "infra": "An infrastructure failure caused a retry.",
    "auth": "The payer credential is missing, expired or refused; review funding with the operator.",
    "funding_expired": "Funding expired while idle; review a new funding request.",
    "generator": "The submitted generator failed; inspect its verification and training report.",
    "tamper": "The checkpoint or worker identity failed an integrity check.",
    "superseded": "A different commitment replaced this submission.",
    "spent": "This commitment has already consumed its judged submission.",
    "duplicate": "Admission rejected the entry as a duplicate.",
    "competition_ended": "The competition closed before this entry could train.",
}

EVALUATION_TEXT = {
    "awaiting_qualification": "Trained; waiting for target qualification and regression guards.",
    "qualified": "Qualification passed; this alone does not establish merge acceptance or payment.",
    "qualified_waiting_acceptance": "Qualified; waiting for warm-start acceptance. Overlapping rows or occupied target slots may defer it to a later init. No reward yet.",
    "qualification_rejected": "Qualification did not pass; this is not an accepted expert contribution.",
    "qualification_not_selected": "Target checks passed, but another qualifying entry won the target slot(s). This entry has no pending merge or reward from this qualification.",
    "acceptance_rejected": "Warm-start acceptance rejected this contribution; it does not merge or earn a reward.",
    "merge_refused": "The expert merge was refused; inspect the signed report before proposing further work.",
    "accepted": "The trainer reports committed warm-start acceptance. Check the validator's signed acceptance and payout record for payment.",
    "awaiting_settlement": "Trained; waiting for publication in a compatible settlement.",
    "awaiting_forward_judgement": "Published for forward-only judging. Training completion does not establish a win.",
    "pending": "Forward evaluation passed initially; a later settlement must confirm it. No paid slot is established yet.",
    "deferred": "Forward evaluation deferred; missing scoring evidence is not a scientific loss. Monitor later settlements.",
    "confirmed": "Confirmation reported: this entry took the target slot at the named settlement. This is historical evidence, not proof of current payment.",
    "below holder LCB": "Confirmation passed but did not beat the existing holder's confidence bound; this entry did not take the paid slot.",
    "not_selected": "This entry passed initial judging; another entry advanced to confirmation. No reward was earned.",
    "dropped": "The entry was dropped from forward judging/confirmation; no new paid slot is established. Inspect the signed reason.",
    "rejected": "Forward evaluation/confirmation rejected the entry; no new paid slot is established.",
    "not_passed": "The reported forward evaluation did not pass. Inspect the signed reason and later carry/recovery state; no paid slot is established by this result.",
}


def evaluation_view(snapshot, *, lane, row, now, max_age):
    """Keep scientific progress separate from execution; match the exact attempt."""
    result = {"stage": "unknown", "reward_status": "not_verified",
              "message": "Per-entry scientific progress unavailable; inspect the signed lane reports."}
    if lane == "a":
        result["message"] = "Use cascade duel --hotkey HOTKEY --history --json for settled Lane A verdicts; queue completion alone is not a win."
        return result
    doc = (snapshot.get("queues") or {}).get(lane)
    clock = freshness(doc, now=now, max_age=max_age)
    result["freshness"] = clock
    if not clock["fresh"]:
        return result
    records = doc.get("submission_progress")
    if not isinstance(records, list):
        return result
    def target_set(value):
        return set(value) if isinstance(value, list) and all(isinstance(v, str) for v in value) else set()

    wanted = target_set(row.get("targets") or ([row["target"]] if row.get("target") else []))
    matches = [r for r in records if isinstance(r, dict) and r.get("hotkey") == row.get("hotkey")
               and r.get("ref") == row.get("ref") and wanted and target_set(r.get("targets")) == wanted
               and row.get("attempt_id") and r.get("attempt_id") == row["attempt_id"]]
    if not matches:
        return result
    report = dict(matches[-1])
    # The latest receipt can advance C only when it names this trained artifact.
    receipt = snapshot.get("submission_receipt")
    comp_clock = (snapshot.get("freshness") or {}).get("competition") or {}
    boundary = receipt.get("boundary") if isinstance(receipt, dict) else None
    previous_boundary = report.get("receipt_boundary")
    if (lane == "c" and isinstance(receipt, dict) and comp_clock.get("fresh")
            and receipt.get("status") == "scored" and type(boundary) is int
            and (type(previous_boundary) is not int or boundary >= previous_boundary)):
        judged = [r for r in receipt.get("lane_c") or [] if isinstance(r, dict)
                  and r.get("hotkey") == row.get("hotkey")
                  and isinstance(r.get("target"), str) and r["target"] in wanted and report.get("trained_pointer")
                  and r.get("trained_pointer") == report["trained_pointer"]
                  and report.get("train_block") is not None
                  and r.get("train_block") == report["train_block"]]
        if judged:
            report.update(stage=judged[-1].get("outcome") or (
                              "not_passed" if judged[-1].get("passed") is False else "unknown"),
                          receipt_round_id=receipt.get("round_id"),
                          receipt_boundary=receipt.get("boundary"),
                          holder=judged[-1].get("holder"))
    stage = report.get("stage")
    result.update(reported=report, stage=stage if stage in EVALUATION_TEXT else "unknown",
                  message=EVALUATION_TEXT.get(stage, result["message"]),
                  source_kind="public_telemetry_not_verified_receipt")
    return result


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def safe_text(value):
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "?", str(value or ""))[:512]


def read_queue(base, lane, *, timeout):
    from .cli import _intake_transport_ok

    if not _intake_transport_ok(base):
        raise ValueError("Intake must use HTTPS or loopback HTTP.")
    request = Request(base.rstrip("/") + QUEUE_PATHS[lane],
                      headers={"Accept": "application/json", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Queue response exceeds size limit.")
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or not isinstance(doc.get("entries"), list)
            or any(not isinstance(row, dict) for row in doc["entries"])):
        raise ValueError("Malformed queue response.")
    return doc


def queue_document(snapshot, lane):
    doc = (snapshot.get("queues") or {}).get(lane)
    if not isinstance(doc, dict):
        return None
    if lane == "b" and isinstance(doc.get("queue"), dict):
        return {**doc["queue"], "as_of": doc.get("as_of")}
    if lane == "c" and isinstance(doc.get("funded_queue"), dict):
        return {**doc["funded_queue"], "as_of": doc.get("as_of")}
    return doc if isinstance(doc.get("entries"), list) else None


def submission_view(snapshot, *, hotkey, lane, ref=None, queue=None, source=None,
                    now=None, max_age=180, targets=None):
    now = now or datetime.now(UTC)
    doc = queue if queue is not None else queue_document(snapshot, lane)
    if (not isinstance(doc, dict) or not isinstance(doc.get("entries"), list)
            or any(not isinstance(r, dict) for r in doc["entries"])):
        doc = None
    clock = freshness(doc, now=now, max_age=max_age)
    rows = [r for r in (doc or {}).get("entries", []) if isinstance(r, dict)]
    candidates = [r for r in rows if r.get("hotkey") == hotkey]
    def normalized(values):
        return {"horizon:" + str(v)[1:] if re.fullmatch(r"h\d+", str(v)) else str(v)
                for v in values if v}

    wanted = normalized(targets.split(",")) if isinstance(targets, str) else set()

    def target_match(row, *, parked=False):
        if not wanted:
            return True
        prefix = "queued_" if parked else ""
        return normalized(row.get(prefix + "targets") or [row.get(prefix + "target")]) == wanted

    row = next((r for r in candidates if (not ref or r.get("ref") == ref) and target_match(r)), None)
    parked = next((r for r in candidates if ref and r.get("queued_ref") == ref
                   and target_match(r, parked=True)), None)
    result = {"lane": lane, "hotkey": hotkey, "ref": ref,
              "source": source, "freshness": clock, "status": "unknown",
              "observed": None, "auto_retry": None, "action_required": None,
              "eta_seconds": None, "verdict": "not_checked",
              "message": "Queue unavailable; this is not evidence that your entry is missing.",
              "next_action": "Refresh the feed or ask the operator to restore queue telemetry."}
    if doc is None:
        return result
    if not row and parked:
        row = {**parked, "ref": ref, "status": "queued_behind", "phase": "queued_behind",
               "phase_at": None, "phase_history": [], "phase_reason": "", "funded_at": None,
               "attempts": 0, "requeues": 0, "last_error_class": "", "operator_billed": False,
               "target_boundary": None, "era_index": None, "started_block": None}
        row["targets"] = parked.get("queued_targets") or [parked.get("queued_target")]
    if row is None:
        result.update(status="not_found" if clock["fresh"] else "stale",
                      message="No matching public queue entry. Pending reveals are hidden; absence does not prove rejection.",
                      next_action=f"Check cascade reveal-status {hotkey} --json, the funding receipt and the exact reference; do not fund again blindly.")
        if candidates and ref:
            result["message"] = "The queue holds a different reference for this hotkey; its progress is not this submission's progress."
        return result
    status = row.get("status")
    phase = row.get("phase") if status == "in_flight" else status
    if status == "queued" and row.get("phase") == "waiting_launch":
        phase = "waiting_launch"
    phase = phase if isinstance(phase, str) and phase in PHASE_TEXT else (
        status if isinstance(status, str) and status in PHASE_TEXT else "unknown")
    ref = row.get("ref")
    error = row.get("last_error_class") or ""
    error = error if isinstance(error, str) else "unknown"
    phase_at, funded = number(row.get("phase_at")), number(row.get("funded_at"))
    retry = number(row.get("not_before"))
    raw_history = row.get("phase_history")
    history = [dict(r) for r in raw_history if isinstance(r, dict)][-16:] if isinstance(raw_history, list) else []
    queued = [r for r in rows if r.get("status") == "queued"]
    # This is reveal ordering, not a GPU reservation: type/cap and gates matter.
    queued.sort(key=lambda r: (number(r.get("reveal_block")) or 0, str(r.get("hotkey", ""))))
    position = next((i for i, r in enumerate(queued, 1) if r is row), None)
    observed = {"status": status, "phase": phase, "phase_at": phase_at,
                "reported_phase": row.get("phase"),
                "attempt_id": row.get("attempt_id"),
                "targets": row.get("targets") or ([row["target"]] if row.get("target") else []),
                "phase_reason": row.get("phase_reason") or "", "last_error_class": error,
                "funded_at": funded, "requeues": row.get("requeues", 0),
                "attempts": row.get("attempts", 0), "target_boundary": row.get("target_boundary"),
                "era_index": row.get("era_index"), "started_block": row.get("started_block"),
              "gpu": row.get("gpu") or "default", "operator_billed": bool(row.get("operator_billed")),
                "operator_faults": int(number(row.get("operator_faults")) or 0),
                "gpu_max_price_per_hour": number(row.get("gpu_max_price_per_hour")),
                "gpu_allow_fallback": bool(row.get("gpu_allow_fallback")),
                "phase_history": history}
    result.update(ref=ref, observed=observed,
                  worker_liveness="not_measured",
                  status=phase if clock["fresh"] else "stale",
                  message=PHASE_TEXT[phase] if clock["fresh"] else "Queue observation is stale; the current stage is unknown.",
                  elapsed_since_funding_seconds=max(0, round(now.timestamp() - funded)) if funded is not None else None,
                  phase_age_seconds=max(0, round(now.timestamp() - phase_at)) if phase_at else None,
                  reveal_order_position=position,
                  retry_at=retry or None,
                  retry_in_seconds=max(0, round(retry - now.timestamp())) if retry else None,
                  last_retry_explanation=ERROR_TEXT.get(error, "No classified failure reported." if not error else "Review the operator's failure report."))
    if not clock["fresh"]:
        result["next_action"] = "Refresh before acting; do not infer a failure or resubmit from stale telemetry."
        return result
    result["evaluation"] = evaluation_view(snapshot, lane=lane, row=row, now=now, max_age=max_age)
    if status in ("queued", "in_flight", "in_round", "queued_behind"):
        result.update(auto_retry=True, action_required=False,
                      next_action="Keep this entry funded/revealed and monitor; do not resubmit for each round.",
                      participation="Automatically remains eligible for later admission/settlement while funding, commitment and lane rules remain valid; a particular round is not guaranteed.")
        faults = observed["operator_faults"]
        if error == "price_cap_unavailable" and status == "queued":
            result["action_required"] = True
            result["next_action"] = ("Use a provider that supports your price ceiling, or explicitly change "
                                     "the GPU/price authorization. Do not resubmit the generator.")
        elif row.get("operator_billed") and faults >= OPERATOR_FAULT_RETRIES:
            result["payer_retry_allowed"] = False
            result["next_action"] = (f"Payer-funded retries are paused after {faults} operator-side failures. "
                                     "Recovery needs configured operator capacity, or the operator to fix "
                                     "the cause and reset the loss counter. Nothing was burned; do not resubmit.")
        elif row.get("operator_billed") or faults:
            result["next_action"] = (f"Work was lost on the operator's side ({faults} of {OPERATOR_FAULT_RETRIES} "
                                     "losses). Unless operator recovery capacity is configured, the leg retries "
                                     "on your provider account with a new pod; keep the entry funded and do not resubmit.")
        elif phase == "waiting_launch":
            result["next_action"] = "Wait for v2 activation. Keep this entry; no training attempt has started and no resubmission is needed."
        elif phase == "waiting_gpu":
            result["next_action"] = "Keep the existing entry; no resubmission is needed for later rounds. For a prolonged wait, ask the operator to check your GPU type/price cap, market capacity and backlog."
        elif retry and retry > now.timestamp():
            result["message"] = "Queued in retry backoff"
    elif status in ("failed", "withdrawn"):
        result.update(auto_retry=False, action_required=status == "failed",
                      next_action=ERROR_TEXT.get(error, "Review the terminal outcome with the operator before proposing a new submission."))
        if status == "withdrawn":
            result["next_action"] = "Entry withdrawn; no automatic retry. Re-enter only if the human decides to submit/fund again."
    elif status == "done":
        result.update(auto_retry=False, action_required=False,
                      next_action=f"Read cascade duel --hotkey {hotkey} --history --json for Lane A; inspect the lane's signed verdict/acceptance report for B/C. Queue completion is not a win or reward.")
        if lane == "b":
            result["message"] = "Qualification work consumed; warm-start acceptance/rewards are separate and may still be pending."
        elif lane == "c":
            result["message"] = "Training entry consumed; forward judging/confirmation and rewards are separate."
    chain = snapshot.get("chain") or {}
    block, target = number(chain.get("current_block")), number(row.get("target_boundary"))
    if block is not None and target:
        result["target_passed"] = block >= target
        result["target_note"] = "Target is an admission estimate, not a guaranteed settlement; completed checks and compatible era/init determine inclusion."
    return result


def render_submission(out, view):
    out.info(f"Lane {view['lane'].upper()} — {safe_text(view['hotkey'])}")
    if view.get("ref"):
        out.info(f"  Reference: {safe_text(view['ref'])}")
    out.info(f"  Status: {safe_text(view['message'])}")
    out.info(f"  Source as of: {safe_text(view['freshness'].get('as_of')) or 'unknown'}; "
             f"{'fresh' if view['freshness']['fresh'] else 'stale/unavailable'}")
    obs = view.get("observed") or {}
    if obs:
        funded_age = view.get("elapsed_since_funding_seconds")
        if funded_age is not None:
            out.info(f"  Time since funding: {funded_age // 3600}h {(funded_age % 3600) // 60}m")
        if obs.get("targets"):
            out.info(f"  Targets: {', '.join(safe_text(t) for t in obs['targets'] if t)}")
        if view["status"] == "stale":
            out.info(f"  Last reported stage: {PHASE_TEXT.get(obs.get('phase'), 'unknown')}")
        age = view.get("phase_age_seconds")
        if age is not None:
            out.info(f"  Stage last changed {age // 3600}h {(age % 3600) // 60}m ago; this is not a worker heartbeat.")
        out.info(f"  Requested GPU: {safe_text(obs['gpu'])}; retries: {obs['requeues']}; consumed attempts: {obs['attempts']}")
        if obs["gpu"] != "default":
            cap = obs.get("gpu_max_price_per_hour")
            out.info(f"  Requested hourly cap: {f'${cap:.2f}/h' if cap else 'no miner-specific cap'}; "
                     f"GPU fallback: {'allowed' if obs['gpu_allow_fallback'] else 'disabled'}; actual device unreported here.")
        reason = {"market_capacity": "No fitting GPU was available under the current choices/caps.",
                  "king_priority": "King work has provisioning priority.",
                  "availability_check": "Provisioning requested; no capacity outcome reported yet."}.get(obs.get("phase_reason"))
        if reason:
            out.info(f"  Waiting reason: {reason}")
        if view.get("reveal_order_position"):
            out.info(f"  Reveal-order position: {view['reveal_order_position']} (not a GPU ETA)")
        if obs.get("last_error_class"):
            out.info(f"  Last retry: {safe_text(obs['last_error_class'])} — {view['last_retry_explanation']}")
        if view.get("retry_in_seconds"):
            out.info(f"  Retry allowed in {view['retry_in_seconds']}s; capacity is not reserved.")
        if obs.get("target_boundary"):
            out.info(f"  Target block: {obs['target_boundary']}; era: {obs.get('era_index')}; target is not a guarantee.")
            if view.get("target_passed"):
                out.info("  This target block has passed; inclusion in a later settlement is not confirmed.")
        moves = [e for e in obs.get("phase_history", []) if e.get("phase") in ("admitted", "queued")]
        for event in moves[-4:]:
            when = number(event.get("at"))
            try:
                stamp = datetime.fromtimestamp(when, UTC).isoformat(timespec="seconds") if when else "unknown time"
            except (OverflowError, OSError, ValueError):
                stamp = "unknown time"
            why = safe_text(event.get("error_class"))
            assignment = (f"era={event.get('era_index')}; target={event.get('target_boundary')}"
                          if event.get("phase") == "admitted" else "no era assigned")
            out.info(f"  {stamp}: {safe_text(event.get('phase'))}; {assignment}" + (f"; reason={why}" if why else ""))
    if view.get("participation"):
        out.info(f"  {view['participation']}")
    evaluation = view.get("evaluation")
    if evaluation:
        out.info(f"  Evaluation: {safe_text(evaluation['message'])}")
        reported = evaluation.get("reported") or {}
        if reported.get("round_id"):
            out.info(f"  Evaluation origin round: {safe_text(reported['round_id'])}")
        if reported.get("receipt_round_id"):
            out.info(f"  Reported receipt: {safe_text(reported['receipt_round_id'])}; boundary={safe_text(reported.get('receipt_boundary'))}")
    out.info(f"  Next: {view['next_action']}")


def tracking_result(out, args, *, hotkey, ref, lane="a", targets=()):
    argv = ["cascade", "status", "--hotkey", hotkey, "--lane", lane, "--ref", ref,
            "--intake", args.intake_url]
    if getattr(args, "chain_toml", None):
        argv += ["--chain-toml", str(args.chain_toml)]
    if targets:
        argv += ["--targets", ",".join(targets)]
    out.result(tracking={"argv": [*argv, "--json"], "poll_seconds": 30,
                         "round_guaranteed": False})
    out.info("Track this exact submission: " + shlex.join([*argv, "--watch"]))


def lane_queue_status(args, lane):
    from .cli import _intake_transport_ok
    from .output import NETWORK, OK, USAGE, reporter

    out = reporter(args)
    if not _intake_transport_ok(args.intake_url):
        return out.fail("insecure_intake", "Intake must use HTTPS or loopback HTTP.", USAGE)
    try:
        doc = read_queue(args.intake_url, lane, timeout=20)
    except (OSError, ValueError):
        return out.fail("intake_unreachable", "Submission queue unavailable; current status is unknown.", NETWORK)
    rows = doc["entries"]
    selected = [r for r in rows if not args.hotkey or r.get("hotkey") == args.hotkey]
    out.result(queued_depth=doc.get("queued_depth"), entries=selected, as_of=doc.get("as_of"))
    clock = freshness(doc, now=datetime.now(UTC), max_age=180)
    views = [submission_view({}, hotkey=row["hotkey"], lane=lane, queue=doc,
                             source=args.intake_url.rstrip("/") + QUEUE_PATHS[lane])
             for row in selected if isinstance(row.get("hotkey"), str)]
    if args.hotkey and not views:
        views = [submission_view({}, hotkey=args.hotkey, lane=lane, queue=doc)]
    out.result(submissions=views, freshness=clock)
    for view in views:
        render_submission(out, view)
    if not clock["fresh"]:
        return out.fail("submission_status_unavailable", "Queue timestamp missing or stale; update the intake/publisher before relying on it.", NETWORK)
    return OK
