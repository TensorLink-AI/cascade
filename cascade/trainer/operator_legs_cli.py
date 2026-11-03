"""Owner commands for operator-billed legs (2026-11-02).

    cascade-trainer list-operator-legs   [--work-root _train_work] [--json]
    cascade-trainer approve-operator-leg <hotkey> [--attempt-id ID] [--max-cost USD] [--queue NAME]
    cascade-trainer deny-operator-leg    <hotkey> [--attempt-id ID] [--queue NAME]
    cascade-trainer resume-operator-leg  <hotkey> [--queue NAME]
    cascade-trainer rent-status          [--json]
    cascade-trainer resume-rents         --reason TEXT

``list`` shows legs awaiting approval (hotkey, uid, SKU, projected cost, why)
and entries paused after operator-side faults. ``approve`` records a
approval of that hotkey's pending request (and, with ``--max-cost``, a cost
ceiling) and returns the entry to the queue. With NO pending request it
PRE-approves the hotkey's next operator-billed leg (``--kind`` narrows it).
The approval lasts for one leg: it is used when that leg's pod is actually
rented (or its lane dispatched); a no-capacity / rent error keeps it, and a
retry after the leg failed needs a new approval. ``deny`` records a single-use
denial: the leg returns to the queue on the miner's own key, or fails
unburned when it has none ("re-fund needed"). ``resume`` returns a paused
entry to the queue with its operator-fault count reset — run it only once the
cause is fixed.

Pure file operations under the trainer's work root (flock'd, like the
queue); no chain, no provider, no key material. Run them on the trainer box
from the deployment directory (the work root and queue paths are relative).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMANDS = ("list-operator-legs", "approve-operator-leg", "deny-operator-leg",
            "resume-operator-leg", "rent-status", "resume-rents")
KINDS = ("", "expired_key", "make_good", "operator_lane", "paid_fallback", "retrain_after_loss")


def _queues(work_root: Path) -> list:
    """Every funding-lane queue file under ``work_root`` (A, B, C)."""
    from ..funding.queue import (
        LANE_B_QUEUE_FILE,
        LANE_C_QUEUE_FILE,
        FundedQueue,
        LaneBQueue,
        LaneCQueue,
    )

    out = []
    for name, cls in (("funded_queue.json", FundedQueue), (LANE_B_QUEUE_FILE, LaneBQueue),
                      (LANE_C_QUEUE_FILE, LaneCQueue)):
        p = work_root / name
        if p.is_file():
            out.append(cls(p))
    return out


def _queue_for(work_root: Path, rec: dict | None, queue_name: str | None):
    from ..funding.queue import FundedQueue

    path = (rec or {}).get("queue_path") or ""
    if path and Path(path).is_file():
        return FundedQueue(path)
    for q in _queues(work_root):
        if queue_name is None or q.path.name == queue_name:
            return q
    return None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-trainer", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in COMMANDS:
        s = sub.add_parser(cmd)
        s.add_argument("--work-root", type=Path, default=Path("./_train_work"))
        if cmd not in ("list-operator-legs", "rent-status", "resume-rents"):
            s.add_argument("hotkey")
            s.add_argument("--queue", default=None,
                           help="queue file name when the hotkey has entries in several lanes")
        if cmd in ("approve-operator-leg", "deny-operator-leg"):
            s.add_argument("--attempt-id", default=None,
                           help="bind to this pending request (as listed)")
        if cmd == "approve-operator-leg":
            s.add_argument("--max-cost", type=float, default=None,
                           help="refuse the dispatch if its projected cost exceeds this (USD)")
            s.add_argument("--kind", default="", choices=KINDS,
                           help="a pre-approval (no pending request) for this kind only")
        if cmd in ("list-operator-legs", "rent-status"):
            s.add_argument("--json", action="store_true")
        if cmd == "resume-operator-leg":
            s.add_argument("--reset-spend", action="store_true",
                           help="also reset the miner waste count (a waste-cap pause always does)")
        if cmd == "resume-rents":
            s.add_argument("--reason", required=True,
                           help="why it is safe to rent again (logged)")
    return p


def main(argv: list[str]) -> int:
    from ..funding.operator_approvals import OperatorApprovals
    from ..funding.queue import (
        AWAITING_APPROVAL_STATUS,
        PAUSED_BUDGET_STATUS,
        PAUSED_SPEND_STATUS,
        PAUSED_STATUS,
    )

    args = _parser().parse_args(argv)
    wr = Path(args.work_root)
    store = OperatorApprovals.for_work_root(wr)
    if args.cmd in ("rent-status", "resume-rents"):
        return _rents(args, wr)
    if args.cmd == "list-operator-legs":
        pending = store.pending()
        paused = [{"queue": q.path.name, "hotkey": e.hotkey, "ref": e.ref, "status": e.status,
                   "operator_requeues": e.operator_requeues, "last_error": e.last_error}
                  for q in _queues(wr) for e in q.entries()
                  if e.status in (PAUSED_STATUS, PAUSED_SPEND_STATUS, PAUSED_BUDGET_STATUS)]
        halt = _halt(wr)
        spend = _spend(wr)
        if args.json:
            print(json.dumps({"awaiting_approval": pending, "paused": paused,
                              "rents_halted": halt, "operator_spend_today": spend}, indent=2))
            return 0
        print(f"operator spend today (UTC): booked ${spend['booked_usd']:.2f} "
              f"(projected ${spend['projected_usd']:.2f}) over {len(spend['legs'])} leg(s)")
        for row in spend["legs"]:
            print(f"  {str(row.get('hotkey'))[:16]} {row.get('kind')}: booked "
                  f"${float(row.get('usd') or 0):.2f}, projected "
                  f"${float(row.get('projected_usd') or row.get('usd') or 0):.2f}")
        if halt:
            print(f"RENTS HALTED since {halt.get('since')}: {halt.get('reason')} "
                  "(`cascade-trainer rent-status`)")
        if not pending and not paused:
            print("no operator-billed legs awaiting approval; no paused entries")
        for r in pending:
            print(f"AWAITING  {r['hotkey']}  uid={r.get('uid')}  queue={r['queue']}  "
                  f"attempt={r.get('request_attempt_id')}  sku={r.get('sku') or '?'}  "
                  f"~${float(r.get('projected_cost_usd') or 0):.2f} "
                  f"(${float(r.get('rate_per_hour') or 0):.2f}/h x "
                  f"{float(r.get('expected_hours') or 0):.1f} h)  why: {r.get('reason')}")
        for r in paused:
            print(f"PAUSED    {r['hotkey']}  queue={r['queue']}  operator faults="
                  f"{r['operator_requeues']}  last: {r['last_error'][:160]}")
        return 0
    hk = args.hotkey
    if args.cmd == "resume-operator-leg":
        for q in _queues(wr):
            if args.queue and q.path.name != args.queue:
                continue
            if q.unpark(hk, expect_status=PAUSED_STATUS,
                        reason="resumed by the operator after a fault pause"):
                if getattr(args, "reset_spend", False):
                    _reset_spend(q, hk)
                print(f"resumed {hk} in {q.path.name} (operator-fault count reset)")
                return 0
            if q.unpark(hk, expect_status=PAUSED_BUDGET_STATUS,
                        reason="resumed by the operator after the miner budget pause"):
                if getattr(args, "reset_spend", False):
                    _reset_spend(q, hk)
                print(f"resumed {hk} in {q.path.name} (budget pause)")
                return 0
            if q.unpark(hk, expect_status=PAUSED_SPEND_STATUS,
                        reason="resumed by the operator after the waste cap"):
                _reset_spend(q, hk)                # a waste-cap resume IS a fresh count
                print(f"resumed {hk} in {q.path.name} (miner waste count reset)")
                return 0
        print(f"no paused entry for {hk}", file=sys.stderr)
        return 1
    approve = args.cmd == "approve-operator-leg"
    try:
        rec = store.decide(hk, approve=approve, attempt_id=args.attempt_id,
                           max_cost_usd=getattr(args, "max_cost", None), queue_name=args.queue,
                           preapprove=_live_entry(wr, hk, args.queue,
                                                  kind=getattr(args, "kind", "")))
    except (LookupError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    q = _queue_for(wr, rec, args.queue)
    if rec.get("origin") == "spend_cap_make_good" or (
            not approve and rec.get("kind") == "spend_cap_make_good"):
        # Approve: the paused leg finishes OPERATOR-billed (never the miner's
        # key again). Deny: it stays paused for a miner re-fund — never
        # rejected.
        moved = bool(approve and q is not None and q.unpark(
            hk, expect_status=PAUSED_SPEND_STATUS, operator_billed=True,
            reason="the operator finishes this leg on its own account (make-good)"))
    else:
        moved = bool(q is not None and q.unpark(
            hk, expect_status=AWAITING_APPROVAL_STATUS, deny=not approve,
            reason=("approved by the operator (single use)" if approve
                    else "the operator declined to pay for this leg")))
    verb = "APPROVED (single use)" if approve else "DENIED"
    cost = rec.get("projected_cost_usd")
    print(f"{verb}: {hk} uid={rec.get('uid')} {rec.get('kind')} ~${float(cost or 0):.2f}"
          + (f" ceiling ${rec['max_cost_usd']:.2f}" if rec.get("max_cost_usd") else "")
          + ("; entry returned to the queue" if moved else
             "; entry was not parked (it is picked up at its next dispatch)"))
    return 0


def _reset_spend(queue, hotkey: str) -> None:
    from ..funding.miner_spend import MinerSpend

    MinerSpend.beside(queue.path).reset(hotkey)


def _live_entry(work_root: Path, hotkey: str, queue_name: str | None, *, kind: str) -> dict | None:
    """The hotkey's live queue entry as a pre-approval binding, or None."""
    for q in _queues(work_root):
        if queue_name and q.path.name != queue_name:
            continue
        e = q.get(hotkey)
        if e is not None and e.status in ("queued", "in_flight", "awaiting_operator_approval",
                                          "paused", "paused_spend_cap", "paused_budget"):
            return {"queue": q.path.name, "queue_path": str(q.path), "ref": e.ref,
                    "funded_at": e.funded_at, "kind": kind}
    return None


def _spend(work_root: Path) -> dict:
    """Today's operator spend: booked (landed $/h) vs projected, per leg."""
    from .money_guard import OperatorSpend

    try:
        rows = OperatorSpend(work_root / "operator_spend.json").rows()
    except Exception:  # noqa: BLE001 — unreadable: show none
        rows = []
    return {"booked_usd": round(sum(float(r.get("usd") or 0) for r in rows), 2),
            "projected_usd": round(sum(float(r.get("projected_usd") or r.get("usd") or 0)
                                       for r in rows), 2),
            "legs": rows}


def _halt(work_root: Path) -> dict | None:
    from .box_health import BoxHealth

    try:
        return BoxHealth(work_root / "box_health.json").halt()
    except Exception as e:  # noqa: BLE001
        return {"reason": f"box health unreadable: {e}"}


def _rents(args, work_root: Path) -> int:
    from .box_health import BoxHealth

    bh = BoxHealth(work_root / "box_health.json")
    if args.cmd == "resume-rents":
        if bh.clear(f"owner: {args.reason}"):
            print(f"rents RESUMED ({args.reason})")
            return 0
        print("rents were not halted")
        return 0
    st = bh.state()
    if getattr(args, "json", False):
        print(json.dumps({"rents_halted": st.get("halt"), "recent_faults": st.get("events"),
                          "legs": st.get("legs"), "last_clear": st.get("last_clear")},
                         indent=2, default=str))
        return 0
    h = st.get("halt")
    print(f"RENTS HALTED since {h.get('since')}: {h.get('reason')}" if h else "rents: OK")
    for e in st.get("events") or []:
        print(f"  fault {e.get('cls')} on {str(e.get('hotkey'))[:12]} at {e.get('at')}: "
              f"{str(e.get('detail'))[:120]}")
    for hk, rec in (st.get("legs") or {}).items():
        print(f"  leg {hk[:12]}: {rec.get('cls')} trained={rec.get('trained')} "
              f"kept={rec.get('kept')} pod={rec.get('pod_id')}")
    return 0


__all__ = ["COMMANDS", "main"]
