"""Public, identity-bound scientific progress; never a payment calculation."""

import json


def b_identity(sub, funding=None):
    return {"hotkey": sub.get("hotkey"), "ref": sub.get("gen_ref"),
            "targets": list(sub.get("targets") or [sub.get("domain")]),
            "commit_block": sub.get("commit_block"),
            "started_block": (funding or {}).get("started_block"),
            "attempt_id": (funding or {}).get("attempt_id")}


def lane_b_progress(state):
    rows = []
    for item in list(getattr(state, "submission_results", [])) + state.outcomes:
        if item.get("outcome") not in ("qualified", "qualification_rejected", "qualification_not_selected", "accepted",
                                        "acceptance_rejected", "merge_refused"):
            continue
        # Older outcomes have no exact generator/attempt binding. Do not infer it.
        if item.get("gen_ref"):
            prior = next((r for r in reversed(rows)
                          if r.get("trained_pointer") == item.get("leg_pointer")
                          and r.get("hotkey") == item.get("hotkey")
                          and r.get("round_id") == item.get("round_id")
                          and r.get("ref") == item["gen_ref"]
                          and r.get("commit_block") == item.get("commit_block")), {})
            rows.append({"hotkey": item.get("hotkey"), "ref": item["gen_ref"],
                         "targets": list(item.get("targets") or [item.get("domain")]),
                         "commit_block": item.get("commit_block"),
                         "started_block": item.get("started_block") or prior.get("started_block"),
                         "attempt_id": item.get("attempt_id") or prior.get("attempt_id"),
                         "trained_pointer": item.get("leg_pointer"),
                         "round_id": item.get("round_id"), "stage": item["outcome"]})
    for merge in state.pending_merges:
        wins = merge.get("qualified_wins") or []
        if wins:
            w = wins[0]
            prior = next((r for r in reversed(rows)
                          if r.get("trained_pointer") == merge.get("rebased_pointer")
                          and r.get("round_id") == merge.get("round_id")
                          and r.get("hotkey") == w.get("hotkey")
                          and r.get("commit_block") == w.get("commit_block")
                          and r.get("ref") == w.get("gen_ref")), {})
            rows.append({**prior, **b_identity(w),
                         "started_block": prior.get("started_block"),
                         "attempt_id": prior.get("attempt_id"),
                         "trained_pointer": merge.get("rebased_pointer"),
                         "round_id": merge.get("round_id"),
                         "stage": "qualified_waiting_acceptance"})
    for f in state.finished:
        rows.append({**b_identity(f["leg"]["sub"], f["leg"].get("funding")),
                     "trained_pointer": f["leg_pointer"], "stage": "awaiting_qualification"})
    return rows[-256:]


def c_identity(leg):
    return {"hotkey": leg.get("hotkey"), "ref": leg.get("ref"),
            "targets": [leg.get("target")], "commit_block": leg.get("commit_block"),
            "started_block": leg.get("started_block"), "attempt_id": leg.get("attempt_id")}


def lane_c_progress(state):
    rows = []
    for rid, rec in state.settled.items():
        identities = {(r.get("hotkey"), tuple(r.get("targets") or []), r.get("trained_pointer")): r
                      for r in rec.get("submission_identities", [])}
        try:
            entries = json.loads(rec.get("text") or "{}").get("entries", [])
        except (ValueError, TypeError, AttributeError):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            pointer = e.get("trained_pointer")
            rows.append({"hotkey": e.get("miner_hotkey"), "ref": e.get("recipe_ref"),
                         "targets": [e.get("target")], "trained_pointer": pointer,
                         "train_block": e.get("train_block"),
                         **identities.get((e.get("miner_hotkey"), (e.get("target"),), pointer), {}),
                         "round_id": rid, "stage": "awaiting_forward_judgement"})
    for f in state.finished:
        rows.append({**c_identity(f["leg"]), "trained_pointer": f["trained_pointer"],
                     "train_block": f.get("train_block"),
                     "stage": "awaiting_settlement"})
    rows.extend(getattr(state, "submission_verdicts", []))
    return rows[-256:]


def receipt_progress(receipt):
    """Projection of the latest *reported* receipt, including its original boundary.

    Publication time does not turn an old receipt into a current payout. No
    signature verification or new consensus fields are introduced here.
    """
    if not isinstance(receipt, dict):
        return None
    keys = ("hotkey", "target", "trained_pointer", "train_block", "phase", "outcome", "stage", "passed",
            "holder", "origin_round")
    return {"round_id": receipt.get("round_id"),
            "boundary": receipt.get("epoch_start_block"),
            "status": receipt.get("status"),
            "lane_c": [{k: r[k] for k in keys if k in r}
                       for r in receipt.get("lane_c") or [] if isinstance(r, dict)]}
