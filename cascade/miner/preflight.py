"""Read-only preparation report. Unknown evidence never grants admission."""
from __future__ import annotations

import argparse
from pathlib import Path

from .live_status import LiveStateUnavailable, fetch_snapshot, positive_seconds
from .output import NETWORK, OK, REJECTED, USAGE, reporter


def add_preflight(sub):
    p = sub.add_parser("preflight", help="Check live rules, local entry, identity and cost limits without writing or loading keys.")
    p.add_argument("repo_dir", type=Path)
    p.add_argument("--lane", choices=("a", "b", "c"), default="a")
    p.add_argument("--targets", default="", help="Comma-separated lane-B targets or one lane-C target.")
    p.add_argument("--hotkey", default=None, help="Public ss58 address; enables read-only registration/reveal checks.")
    p.add_argument("--ref", default=None, help="Expected immutable ref, for comparing an existing reveal.")
    p.add_argument("--network", default="finney")
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--timeout", type=positive_seconds, default=10.0)
    p.add_argument("--max-price", type=positive_seconds, default=None, help="Proposed USD per GPU-hour ceiling; no quote is fetched.")
    p.add_argument("--max-spend", type=positive_seconds, default=None, help="Proposed total USD budget for the human to review.")
    p.set_defaults(func=cmd_preflight)


def identity_check(cfg, hotkey, network, ref):
    from ..interface.validation import parse_commit
    from ..shared.chain import ChainClient
    from ..shared.expert_manifest import parse_expert_targets
    from ..shared.lane_c_manifest import parse_finetune_commit
    from .cli import _pending_timelock

    client = ChainClient.from_config(cfg, network=network)
    commitments = client.poll_commitments()
    match = next((c for c in commitments if c.hotkey == hotkey), None)
    # Registration is separate from having a revealed commitment.
    registered = hotkey in client.registered_hotkeys()
    pending = _pending_timelock(client, hotkey)
    payload = str(match.payload) if match else ""
    a = parse_commit(payload)
    b = parse_expert_targets(payload, cfg.lane_b.targets, max_targets=cfg.lane_b.max_domains_per_leg)
    c = parse_finetune_commit(payload, mkt_suites=tuple(cfg.marketcast.suites))
    revealed_ref = a.ref if a else b[1] if b else c[1] if c else None
    lane = "a" if a else "b" if b else "c" if c else None
    targets = b[0] if b else (c[0],) if c else ()
    return {"registered": registered, "pending_commit_block": pending[0] if pending else None,
            "reveal_block": int(match.commit_block) if match else None,
            "revealed_payload": payload or None, "revealed_ref": revealed_ref,
            "lane": lane, "targets": targets,
            "ref_matches": (revealed_ref == ref) if match and ref else None}


def assess(snapshot, *, lane, targets, hotkey, max_price, max_spend):
    """Pure policy assessment, preserving missing/stale evidence as unknown."""
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "state": "unknown" if passed is None else "pass" if passed else "fail",
                       "detail": detail})

    fresh = snapshot.get("fresh", False)
    check("fresh_rules", fresh, "Current chain and competition feeds are required.")
    policy = snapshot.get("policy") or {}
    view = (snapshot.get("lanes") or {}).get(lane)
    # Lane A is open in the legacy competition too; B/C require explicit activation.
    active = (True if lane == "a" and snapshot.get("ready_for_training") else
              view.get("active") if isinstance(view, dict) else None)
    check("lane_active", active if fresh else None, f"Lane {lane.upper()} activation.")
    if lane in ("a", "b"):
        check("current_init", snapshot.get("ready_for_training"), "Current-era init must be confirmed.")
    if lane == "b":
        lb = policy.get("lane_b") or {}
        configured = (view or {}).get("targets") or lb.get("targets") or lb.get("domains")
        check("targets", bool(targets) and len(set(targets)) == len(targets)
              and len(targets) <= lb.get("max_domains_per_leg", 0)
              and all(t in configured for t in targets) if configured else None,
              "Targets must match the live list and per-leg limit.")
        gate = (view or {}).get("gate") or {}
        check("routing_gate", gate.get("open") if isinstance(gate, dict) else None,
              "The lineage/routing gate must be open.")
    if lane == "c":
        configured = (view or {}).get("targets")
        check("targets", len(targets) == 1 and targets[0] in configured if configured else None,
              "Choose exactly one active MarketCast target.")
        kind = targets[0].partition(":")[0] if len(targets) == 1 else ""
        comp = next((c for c in (view or {}).get("competitions", [])
                     if c.get("kind") == kind), None)
        block = (snapshot.get("chain") or {}).get("current_block")
        accepting = None
        if fresh and comp is not None and type(block) is int:
            end = comp.get("until_block", 0)
            if type(end) is int:
                accepting = comp.get("state") == "active" and (not end or block < end)
        check("competition_active", accepting,
              "The selected competition must accept new work; pending, ended and wind-down slots cannot.")
    if lane == "a":
        spent = (snapshot.get("admission") or {}).get("lane_a_spent_hotkeys")
        check("submission_unspent", hotkey not in spent if hotkey and isinstance(spent, list) and fresh else None,
              "Trainer admission telemetry; intake enforces the final lifetime-submission check.")
    queue_key = f"lane_{lane}_queue"
    qfresh = (snapshot.get("freshness") or {}).get(queue_key, {}).get("fresh", False)
    check("queue_available", True if qfresh else None, "Missing/stale queue data cannot establish capacity or an idle hotkey.")
    if hotkey:
        def matching(value):
            if isinstance(value, dict):
                if value.get("hotkey") == hotkey and value.get("status") in ("in_flight", "queued", "pending_reveal"):
                    return True
                return any(matching(v) for v in value.values())
            return isinstance(value, list) and any(matching(v) for v in value)

        queues = snapshot.get("queues") or {}
        busy = any(matching(q) for q in queues.values() if q is not None)
        required = [other for other in ("a", "b", "c") if other == "a" or
                    ((snapshot.get("lanes") or {}).get(other) or {}).get("active")]
        all_fresh = all((snapshot.get("freshness") or {}).get(f"lane_{other}_queue", {}).get("fresh") for other in required)
        check("hotkey_idle", False if busy else True if all_fresh else None,
              "One live leg per hotkey across all lanes; an existing queued/in-flight entry must be tracked first.")
    check("price_ceiling", max_price is not None, "Set an explicit per-GPU-hour ceiling for the proposed funding command.")
    check("total_budget", max_spend is not None, "Record a total budget for review; standard lane funding currently enforces an hourly cap only.")
    return checks


def cmd_preflight(args: argparse.Namespace):
    from ..shared.config import load_chain_config
    from .training_context import contract_from_snapshot
    from .verify import verify_repo

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    targets = tuple(f"horizon:{t[1:]}" if t.startswith("h") and t[1:].isdigit() else t for t in targets)
    if args.lane == "a" and targets:
        return out.fail("bad_targets", "Lane A has no declared targets.", USAGE)
    snapshot = fetch_snapshot(cfg, timeout=args.timeout)
    checks = assess(snapshot, lane=args.lane, targets=targets, hotkey=args.hotkey,
                    max_price=args.max_price, max_spend=args.max_spend)
    contract = None
    try:
        from .training_context import published_policy
        cfg = published_policy(cfg, snapshot)
        if args.lane != "c":
            contract = contract_from_snapshot(snapshot)
            checks.append({"check": "live_contract", "state": "pass", "detail": snapshot["training_contract"]["digest"]})
        else:
            mc = (snapshot.get("policy") or {}).get("marketcast") or {}
            checks.append({"check": "pinned_base", "state": "pass" if mc.get("base_ref") and mc.get("base_sha256") else "unknown",
                           "detail": {"base_ref": mc.get("base_ref"), "base_sha256": mc.get("base_sha256")}})
    except LiveStateUnavailable as exc:
        checks.append({"check": "live_contract", "state": "unknown", "detail": str(exc)})
    if args.lane == "c":
        from .lane_c import has_local_generator, recipe_error
        error = recipe_error(cfg, args.repo_dir, targets[0] if len(targets) == 1 else None)
        local = {"ok": error is None, "scope": "recipe_structure", "error": error}
        if error is None and has_local_generator(args.repo_dir):
            generator_report = verify_repo(args.repo_dir, cfg, quick=True).to_dict()
            local.update(ok=generator_report["ok"], scope="recipe_and_generator",
                         generator=generator_report)
    else:
        if contract is not None:
            from .training_context import with_contract
            cfg = with_contract(cfg, contract)
        local = verify_repo(args.repo_dir, cfg, quick=True).to_dict()
    checks.append({"check": "local_entry", "state": "pass" if local["ok"] else "fail", "detail": local})
    identity = None
    if args.hotkey:
        try:
            identity = identity_check(cfg, args.hotkey, args.network, args.ref)
        except Exception:  # noqa: BLE001 — an unavailable chain is unknown, never admission
            identity = {"registered": None, "error": "chain_identity_unavailable"}
    checks.append({"check": "registered_hotkey", "state": "pass" if identity and identity.get("registered") else
                   "fail" if identity and identity.get("registered") is False else "unknown", "detail": identity})
    if args.ref:
        matches = identity.get("ref_matches") if identity else None
        checks.append({"check": "revealed_ref", "state": "unknown" if matches is None else "pass" if matches else "fail",
                       "detail": "Before funding, the intended ref must have revealed; inspect targets in revealed_payload too."})
        lane_matches = (identity.get("lane") == args.lane and set(identity.get("targets", ())) == set(targets)) if identity else None
        pending = (identity.get("pending_commit_block") or 0) if identity else 0
        reveal = (identity.get("reveal_block") or 0) if identity else 0
        checks.append({"check": "revealed_lane_and_targets", "state": "unknown" if lane_matches is None else
                       "pass" if lane_matches and pending <= reveal else "fail",
                       "detail": "The revealed lane/targets must match, with no newer unrevealed commitment."})
    chain = snapshot.get("chain") or {}
    block, era = chain.get("current_block"), chain.get("era_blocks")
    timing = None
    if isinstance(block, int) and isinstance(era, int) and era > 0:
        from ..shared.chain import blocks_until_boundary_reveal

        epoch = chain.get("epoch_blocks", era)
        margin = (snapshot.get("policy") or {}).get("round", {}).get("reveal_margin_blocks")
        timing = {"current_block": block, "next_era_block": (block // era + 1) * era,
                  "next_settlement_block": (block // epoch + 1) * epoch,
                  "reveal_margin_blocks": margin,
                  "estimated_reveal_block": block + blocks_until_boundary_reveal(block, epoch, margin)
                                            if isinstance(margin, int) and 0 <= margin < epoch else None,
                  "note": "Timed reveal is calculated again by the commit command against the current chain head."}
    ready = all(c["state"] == "pass" for c in checks)
    out.result(lane=args.lane, targets=targets, checks=checks, identity=identity, timing=timing,
               network=snapshot.get("network"),
               ready_to_propose=ready, authorized_to_spend=False, submitted=False,
               spending={"max_price_per_gpu_hour": args.max_price, "proposed_total_budget": args.max_spend,
                         "quote_confirmed": False, "total_cap_enforced_by_standard_fund": False},
               refresh_before_funding=True, source_kind="public_telemetry_not_consensus")
    if not snapshot.get("fresh"):
        return out.fail("preflight_unavailable", "Live evidence is missing or stale; review the individual checks.", NETWORK)
    if not ready:
        return out.fail("preflight_not_ready", "Some checks failed or remain unknown; nothing was sent.", REJECTED)
    out.info("Preflight passed for human review; refresh and authorize before submission or funding.")
    return OK
