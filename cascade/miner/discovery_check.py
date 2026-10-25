"""Check a deployment's agent discovery contract without credentials or writes."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from .live_status import fetch_snapshot, positive_seconds
from .output import NETWORK, OK, reporter

CAPABILITIES = ("status", "preflight", "verify_quick", "lane_b_test", "lane_c_test", "submission_status")


def add_discovery_check(sub):
    p = sub.add_parser("discovery-check", help="Read-only release check for live feeds, current contract and llms.txt.")
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--site-url", default=None, help="Public dashboard root (default: configured manifest bucket).")
    p.add_argument("--timeout", type=positive_seconds, default=10.0)
    p.set_defaults(func=cmd_discovery_check)


def read_guide(url, timeout):
    from urllib.request import Request, urlopen

    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout) as response:
        text = response.read(512_001)
        if len(text) > 512_000:
            raise ValueError("Guide exceeds size limit")
        return text.decode("utf-8")


def check_snapshot(snapshot):
    from .training_context import contract_from_snapshot

    checks = {"fresh_feeds": bool(snapshot.get("fresh")),
              "current_init": bool(snapshot.get("ready_for_training"))}
    try:
        contract_from_snapshot(snapshot)
        checks["effective_contract"] = True
    except Exception:  # noqa: BLE001 — report incompatible data as an incomplete release
        checks["effective_contract"] = False
    protocol = snapshot.get("agent_protocol") or {}
    checks["agent_protocol"] = protocol.get("schema") == 1 and set(CAPABILITIES) <= set(protocol.get("capabilities") or ())
    checks["submission_tracking"] = ("submission_status" in (protocol.get("capabilities") or ())
                                      and (protocol.get("submission_status") or {}).get("schema") == 1)
    for lane in ("a", "b", "c"):
        active = lane == "a" or ((snapshot.get("lanes") or {}).get(lane) or {}).get("active")
        if active:
            checks[f"lane_{lane}_queue"] = bool((snapshot.get("freshness") or {}).get(f"lane_{lane}_queue", {}).get("fresh"))
    return checks


def cmd_discovery_check(args: argparse.Namespace):
    from ..shared.config import load_chain_config

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    snapshot = fetch_snapshot(cfg, timeout=args.timeout)
    checks = check_snapshot(snapshot)
    base = args.site_url or f"{cfg.storage.s3_endpoint.rstrip('/')}/{cfg.storage.manifest_bucket}"
    if urlsplit(base).scheme not in ("https", "http"):
        return out.fail("bad_site_url", "Site URL must use HTTP or HTTPS.", 2)
    guide_url = base.rstrip("/") + "/llms.txt"
    try:
        guide = read_guide(guide_url, args.timeout)
        checks["agent_guide"] = all(token in guide for token in
            ("cascade status --json", "cascade preflight", "cascade verify", "Lane B", "Lane C"))
    except (OSError, ValueError):
        checks["agent_guide"] = False
    ready = all(checks.values())
    out.result(ready=ready, checks=checks, guide_url=guide_url, network=snapshot.get("network"),
               installed_capabilities=CAPABILITIES, writes_performed=False,
               remaining=[name for name, passed in checks.items() if not passed],
               release_steps=["Release the tested CLI build.",
                              "Release trainer/validator publishers with the effective contract and era clock fields.",
                              "Publish matching dashboard assets and llms.txt.",
                              "Re-run this command; every required check must pass."])
    if not ready:
        return out.fail("discovery_release_incomplete", "Deployment is missing required agent discovery features; see remaining checks.", NETWORK)
    out.info("Agent discovery is ready on this deployment.")
    return OK
