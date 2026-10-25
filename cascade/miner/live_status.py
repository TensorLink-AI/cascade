"""Fresh, read-only competition discovery for humans and mining agents.

Public trainer/validator telemetry is discovery, not signed consensus evidence.
The command deliberately preserves unknown policy fields so new competitions
and supported policy changes do not require a CLI release to be visible.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .output import NETWORK, OK, USAGE, reporter

SOURCES = {
    "chain": "status/chain.json",
    "competition": "status/v2.json",
    "lane_a_queue": "funded/latest.json",
    "lane_b_queue": "status/lane_b.json",
    "lane_c_queue": "status/lane_c.json",
}


class LiveStateUnavailable(RuntimeError):
    """A current training init cannot be confirmed from public sources."""


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def add_status(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("status", help="Fresh competition rules, warm-start init and queues; no wallet.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Select network/storage config.")
    p.add_argument("--max-age", type=positive_seconds, default=180.0,
                   help="Maximum age of required live feeds in seconds (default: 180).")
    p.add_argument("--timeout", type=positive_seconds, default=10.0,
                   help="Timeout per parallel public request (default: 10 seconds).")
    p.add_argument("--hotkey", default=None, help="Track this miner's submission without a wallet.")
    p.add_argument("--lane", choices=("a", "b", "c"), default="a")
    p.add_argument("--ref", default=None, help="Track this exact immutable ref, not a newer entry.")
    p.add_argument("--targets", "--target", default=None, help="Match exact B/C target(s), comma-separated.")
    p.add_argument("--intake", default=None, help="Read the current intake queue as well as public discovery.")
    p.add_argument("--watch", action="store_true", help="Refresh a human submission view until Ctrl-C.")
    p.add_argument("--refresh", type=positive_seconds, default=30.0,
                   help="Watch interval, at least 5 seconds (default: 30).")
    p.set_defaults(func=cmd_status)


def freshness(doc: dict | None, *, now: datetime, max_age: float) -> dict:
    doc = doc if isinstance(doc, dict) else None
    stamp = doc.get("as_of") if doc else None
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        age = (now - when).total_seconds()
    except (ValueError, TypeError, OverflowError):
        return {"available": doc is not None, "as_of": stamp, "age_seconds": None, "fresh": False}
    return {"available": True, "as_of": stamp, "age_seconds": round(age, 1),
            "fresh": -30 <= age <= max_age}


def status_snapshot(docs: dict, *, now: datetime, max_age: float) -> dict:
    docs = {name: value if isinstance(value, dict) else None for name, value in docs.items()}
    chain = docs.get("chain") or {}
    comp = docs.get("competition") or {}
    if isinstance(comp.get("lane_a_queue"), dict):
        docs["lane_a_queue"] = comp["lane_a_queue"]
    clocks = {name: freshness(docs.get(name), now=now, max_age=max_age) for name in SOURCES}
    fresh = all(clocks[name]["fresh"] for name in ("chain", "competition"))
    warm_doc = comp.get("warm_start")
    warm = dict(warm_doc) if isinstance(warm_doc, dict) else {}
    block, era_blocks = chain.get("current_block"), chain.get("era_blocks")
    current_era_start = None
    if type(block) is int and block >= 0 and type(era_blocks) is int and era_blocks > 0:
        current_era_start = block // era_blocks * era_blocks
    from ..shared.manifest import parse_trained_pointer

    pointer = warm.get("pointer", "")
    valid_init = (warm.get("mode") == "random_init" and pointer == "") or (
        warm.get("mode") == "checkpoint" and isinstance(pointer, str)
        and parse_trained_pointer(pointer) is not None)
    warm["current"] = bool(fresh and valid_init and current_era_start is not None
                           and warm.get("era_start_block") == current_era_start)
    return {
        "fetched_at": now.isoformat(timespec="seconds"),
        "fresh": fresh, "freshness": clocks,
        "ready_for_training": warm["current"] and (comp.get("launch_preflight") or {}).get("state") != "blocked",
        "source_kind": "public_telemetry_not_consensus",
        "chain": chain, "warm_start": warm,
        "golive": comp.get("golive"), "model": comp.get("model"),
        "training_contract": comp.get("training_contract"),
        "compute": comp.get("compute"),
        "operations": comp.get("operations"),
        "launch_preflight": comp.get("launch_preflight"),
        "agent_protocol": comp.get("agent_protocol"),
        "submission_receipt": comp.get("submission_receipt"),
        "king": comp.get("king"),
        "admission": comp.get("admission"),
        "policy": comp.get("policy"),
        "lanes": {"a": comp.get("lane_a"), "b": comp.get("lane_b"), "c": comp.get("lane_c")},
        "queues": {lane: docs.get(f"lane_{lane}_queue") for lane in ("a", "b", "c")},
    }


def fetch_snapshot(cfg, *, timeout=10.0, max_age=180.0):
    """One anonymous snapshot, shared by status, preflight and release checks."""
    from .dashboard import fetch_public_json

    def fetch(item):
        name, path = item
        return name, fetch_public_json(cfg.storage, path, timeout=timeout)

    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        docs = dict(pool.map(fetch, SOURCES.items()))
    snapshot = status_snapshot(docs, now=datetime.now(UTC), max_age=max_age)
    base = f"{cfg.storage.s3_endpoint.rstrip('/')}/{cfg.storage.manifest_bucket}"
    snapshot["sources"] = {name: f"{base}/{path}" for name, path in SOURCES.items()}
    if isinstance((docs.get("competition") or {}).get("lane_a_queue"), dict):
        snapshot["sources"]["lane_a_queue"] = f"{base}/status/v2.json"
    snapshot["network"] = {"netuid": cfg.subnet.netuid, "manifest_bucket": cfg.storage.manifest_bucket}
    chain_netuid = (snapshot.get("chain") or {}).get("netuid")
    snapshot["network"]["chain_netuid"] = chain_netuid
    if chain_netuid is not None and chain_netuid != cfg.subnet.netuid:
        snapshot["network"]["matches"] = False
        snapshot["fresh"] = snapshot["ready_for_training"] = False
    return snapshot


def cmd_status(args: argparse.Namespace) -> int:
    import time

    from ..shared.config import load_chain_config
    from .submission_status import read_queue, render_submission, submission_view

    out = reporter(args)
    hotkey = getattr(args, "hotkey", None)
    watch = getattr(args, "watch", False)
    if (watch and (not hotkey or out.json_mode)) or getattr(args, "refresh", 30) < 5:
        return out.fail("bad_watch_options", "--watch needs --hotkey and human output; --refresh must be at least 5 seconds. Agents should poll --json.", USAGE)
    if (getattr(args, "ref", None) or getattr(args, "intake", None)
            or getattr(args, "targets", None)) and not hotkey:
        return out.fail("hotkey_required", "--ref, --targets and --intake require --hotkey.", USAGE)
    if getattr(args, "targets", None) and args.lane == "a":
        return out.fail("bad_targets", "--targets applies to Lane B/C submission tracking.", USAGE)
    if getattr(args, "intake", None):
        from .cli import _intake_transport_ok

        if not _intake_transport_ok(args.intake):
            return out.fail("insecure_intake", "Intake must use HTTPS or loopback HTTP.", USAGE)
    cfg = load_chain_config(args.chain_toml)
    snapshot, queue_ok = {}, True
    try:
        while True:
            snapshot = fetch_snapshot(cfg, timeout=args.timeout, max_age=args.max_age)
            if (snapshot.get("network") or {}).get("matches") is False:
                out.result(**snapshot)
                return out.fail("network_mismatch", "Discovery feed belongs to a different netuid; check the network configuration.", USAGE)
            queue_ok = True
            views = []
            if hotkey:
                lanes = (args.lane,)
                for lane in lanes:
                    queue, source, error = None, (snapshot.get("sources") or {}).get(f"lane_{lane}_queue"), None
                    if getattr(args, "intake", None):
                        from .submission_status import QUEUE_PATHS

                        source = args.intake.rstrip("/") + QUEUE_PATHS[lane]
                        try:
                            queue = read_queue(args.intake, lane, timeout=args.timeout)
                        except (OSError, ValueError) as e:
                            error = type(e).__name__
                    view = submission_view(snapshot, hotkey=hotkey, lane=lane,
                                           ref=getattr(args, "ref", None), queue=queue,
                                           source=source, max_age=args.max_age,
                                           targets=getattr(args, "targets", None))
                    if error:
                        # Failed intake reads cannot silently masquerade as a
                        # successful current queue from a different source.
                        view = submission_view(snapshot, hotkey=hotkey, lane=lane,
                                               ref=getattr(args, "ref", None), queue={"entries": []}, source=source)
                        view.update(status="unknown", message="Intake queue unavailable; current submission state is unknown.",
                                    auto_retry=None, action_required=None, source_error=error)
                        view["freshness"]["fresh"] = False
                    queue_ok = queue_ok and view["freshness"]["fresh"] and view["status"] != "unknown"
                    views.append(view)
                    render_submission(out, view)
                snapshot["submissions"] = views
                snapshot["requested_submission_fresh"] = queue_ok
            out.result(**snapshot)
            out.info(f"Competition status: {'fresh' if snapshot['fresh'] else 'stale or unavailable'}")
            if not hotkey:
                out.info(f"Warm start: {snapshot['warm_start'].get('pointer') or snapshot['warm_start'].get('mode', 'unavailable')}")
            if not watch:
                break
            time.sleep(args.refresh if snapshot["fresh"] and queue_ok else min(120, args.refresh * 2))
    except KeyboardInterrupt:
        out.info("Stopped watching; no submission was changed.")
        return OK
    if not snapshot["fresh"]:
        return out.fail("live_status_unavailable", "Required live feeds are missing or stale; retry before acting.", NETWORK)
    if not queue_ok:
        return out.fail("submission_status_unavailable", "Requested submission queue is missing or stale; do not infer failure or resubmit.", NETWORK)
    if not hotkey and not snapshot["ready_for_training"]:
        out.warn("Current era warm-start init is not confirmed by the live feed; refresh before training.")
    return OK
