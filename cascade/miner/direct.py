"""``cascade direct`` — miner commands for the direct-MoE lane (DEC-CA-0052).

* ``status`` — read-only: fetch the owner-signed network config and the
  trainer's signed journal, verify and replay them exactly as a validator does,
  and summarise the round, the checkpoint, the king (and whether it is stale),
  open cohort terms and your jobs and payouts. No wallet.
* ``deposit-key`` — file the marketplace key (Lium, Vast or RunPod) your jobs
  are rented with, signed by your hotkey. The key is read from the
  environment only and lands in that marketplace's vault; nothing is spent.
* ``fund`` — admit a job into the current cohort: the terms are built from the
  published journal and owner config, you choose the gang, provider and spend
  cap, and the signed ``fund-job`` envelope is posted to the intake. Needs
  ``--max-spend`` and ``--yes``; ``--dry-run`` prints the envelope instead.

The lane is shipped inert: until the operator activates it, the intake answers
``direct_protocol_disabled`` and no journal is published.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .output import NEEDS_CONFIRM, NETWORK, OK, REJECTED, USAGE, reporter

OWNER_ENV = "CASCADE_DIRECT_OWNER_HOTKEY"
BUCKET_ENV = "CASCADE_DIRECT_CONFIG_BUCKET"
PROVIDERS = ("lium", "vast", "runpod")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--owner-hotkey", default=os.environ.get(OWNER_ENV),
                   help=f"Owner hotkey that signs the network config (default ${OWNER_ENV}).")
    p.add_argument("--config-bucket", default=os.environ.get(BUCKET_ENV),
                   help=f"Hippius bucket holding the network config (default ${BUCKET_ENV}).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--offline", action="store_true",
                   help="Skip the chain: no on-chain config-commitment check, no round clock.")


def _wallet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")


def add_direct(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("direct", help="Direct-MoE lane: status, deposit a compute key, fund a job.")
    dsub = p.add_subparsers(dest="direct_cmd", required=True)

    s = dsub.add_parser("status", help="Verified round/king/cohort summary and your jobs (read-only).")
    _common(s)
    s.add_argument("--hotkey", default=None, help="Your hotkey (ss58): adds your jobs and payouts.")
    s.set_defaults(func=cmd_status, command="direct status")

    k = dsub.add_parser("deposit-key", help="File your Lium/Vast/RunPod key for direct jobs.")
    k.add_argument("intake_url", help="Operator intake base URL (https://…).")
    k.add_argument("--provider", required=True, choices=PROVIDERS)
    k.add_argument("--ref", required=True, help="Your revealed generator ref (repo@digest).")
    k.add_argument("--key-env", default=None,
                   help="Env var holding the key (default <PROVIDER>_API_KEY, e.g. VAST_API_KEY). "
                        "The key is never accepted on the command line.")
    _wallet_args(k)
    k.set_defaults(func=cmd_deposit_key, command="direct deposit-key")

    f = dsub.add_parser("fund", help="Admit a funded job into the current cohort (paid).")
    f.add_argument("intake_url", help="Operator intake base URL (https://…).")
    f.add_argument("--ref", required=True, help="Your revealed generator ref (repo@digest).")
    f.add_argument("--provider", action="append", required=True, choices=PROVIDERS,
                   help="Marketplace(s) the job may rent on (repeatable); a key must be on file.")
    f.add_argument("--profile", action="append", required=True,
                   help="Qualified hardware profile(s) for the gang (repeatable).")
    f.add_argument("--world-size", type=int, required=True, help="GPUs in the gang.")
    f.add_argument("--spend-cap", type=int, required=True,
                   help="Most this job may spend, in the ledger's smallest money unit.")
    f.add_argument("--max-spend", type=int, default=None,
                   help="Your own ceiling: refuse if --spend-cap exceeds it. Required to send.")
    f.add_argument("--role", default="challenger", choices=("challenger", "expert", "incumbent"))
    f.add_argument("--reward", type=int, default=0, help="Incumbent completion reward units.")
    f.add_argument("--cohort", default=None, help="Cohort key (default: the open cohort).")
    f.add_argument("--guard", default=None,
                   help="Guard contract digest (default: the only one the owner config lists).")
    f.add_argument("--reference", default=None, help="Reference checkpoint (default: the anchor).")
    f.add_argument("--job", default=None,
                   help="Job id (default: derived from payer, generator and cohort, so a retry "
                        "is idempotent).")
    f.add_argument("--expires-in", type=int, default=600, help="Authorization lifetime (s).")
    f.add_argument("--dry-run", action="store_true", help="Print the signed envelope; send nothing.")
    f.add_argument("--yes", action="store_true", help="Confirm: the job rents GPUs on your key.")
    _common(f)
    _wallet_args(f)
    f.set_defaults(func=cmd_fund, command="direct fund")


def load_direct(args, out):
    """``(exit_code, network, signed_journal, state, now_block)``, verified like a validator."""
    from ..direct.journal import fetch_journal
    from ..direct.network import fetch_network_config
    from ..shared.config import load_chain_config
    from ..validator.direct_moe import verify_under_network

    if not (args.owner_hotkey and args.config_bucket):
        return out.fail("missing_direct_config",
                        f"need --owner-hotkey and --config-bucket (or ${OWNER_ENV} / "
                        f"${BUCKET_ENV})", USAGE), None, None, None, None
    cfg = load_chain_config(args.chain_toml)
    storage = {"endpoint": cfg.storage.s3_endpoint, "region": cfg.storage.s3_region}
    committed = now_block = None
    try:
        if not args.offline:
            from ..shared.chain import ChainClient

            client = ChainClient.from_config(cfg, network=args.network)
            committed = client.plain_commitment(args.owner_hotkey)
            if committed is None:
                return out.fail("no_config_commitment",
                                "the owner hotkey has no on-chain network-config commitment "
                                "(lane not active, or wrong --owner-hotkey)", REJECTED), \
                    None, None, None, None
            now_block = client.current_block()
        network = fetch_network_config({**storage, "bucket": args.config_bucket},
                                       owner=args.owner_hotkey, committed=committed)
        signed = fetch_journal({**storage, **network["journal"]})
        state = verify_under_network(signed, network, now_block=now_block)
    except ValueError as e:
        return out.fail("direct_verification_failed", f"refusing: {e}", REJECTED), \
            None, None, None, None
    except Exception as e:  # noqa: BLE001 — storage/chain outage
        return out.fail("direct_unreachable", f"could not load the direct lane: "
                        f"{type(e).__name__}: {e}", NETWORK), None, None, None, None
    return OK, network, signed, state, now_block


def open_cohort(state):
    """``(key, contract)`` of the newest cohort on the current anchor, or ``(None, None)``."""
    live = [(k, c) for k, c in state["cohorts"].items()
            if c["contract"]["anchor"] == state["anchor"] and k not in state["advanced"]]
    return live[-1] if live else (None, None)


def round_clock(state, now_block):
    if "rounds" not in state or now_block is None:
        return None
    p = state["policy"]
    current = (now_block - p["round_origin"]) // p["round_blocks"]
    return {"round": current, "closes_at_block": p["round_origin"] + (current + 1) * p["round_blocks"],
            "blocks_left": p["round_origin"] + (current + 1) * p["round_blocks"] - now_block,
            "round_blocks": p["round_blocks"], "era_blocks": p["era_blocks"],
            "cohort_opened": state["rounds"]["opened"].get(str(current))}


def king_status(state, clock):
    king = state["king"]
    rows = [r for r in state["king_history"] if r["payer"] == king]
    last_round = max((r["round"] for r in rows if "round" in r), default=None)
    out = {"hotkey": king, "promotions": len(rows), "last_promotion_round": last_round}
    if clock is not None and king is not None:
        # Same rule the reducer applies at a payout boundary (DEC-CA-0052 §11).
        out["stale"] = last_round is None or last_round < clock["round"] - 1
    return out


def summarise(network, state, now_block, hotkey=None):
    key, cohort = open_cohort(state)
    clock = round_clock(state, now_block)
    jobs = state["jobs"]
    counts = {}
    for j in jobs.values():
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    summary = {
        "trainer": network["trainer"], "config_seq": network["seq"], "burn_all": network["burn_all"],
        "anchor": state["anchor"], "phase": state["phase"], "clock": clock,
        "king": king_status(state, clock),
        "open_cohort": None if key is None else {
            "key": key, "anchor": cohort["contract"]["anchor"], "budget": cohort["contract"]["budget"],
            "winner": cohort["winner"], "round": cohort.get("round")},
        "guards": network["guards"], "jobs": counts,
        "spend": {"spent": state["spent"], "reserved": state["reserved_spend"],
                  "pool": state["policy"]["spend_pool"]},
        "payout_boundaries": len(state["payout_boundaries"]),
    }
    if hotkey:
        mine = {jid: {"state": j["state"], "role": j["terms"]["role"], "cohort": j["terms"]["cohort"],
                      "spent": j["spent"], "cap": j["cap"], "reason": j["reason"],
                      "paid": state["paid"].get(jid, 0)}
                for jid, j in jobs.items() if j["terms"]["payer"] == hotkey}
        summary["you"] = {"hotkey": hotkey, "is_king": state["king"] == hotkey, "jobs": mine,
                          "paid_total": sum(v["paid"] for v in mine.values())}
    return summary


def cmd_status(args) -> int:
    out = reporter(args)
    rc, network, _signed, state, now_block = load_direct(args, out)
    if rc != OK:
        return rc
    s = summarise(network, state, now_block, args.hotkey)
    out.result(**s)
    out.info(f"direct-MoE lane — trainer {s['trainer'][:12]}…, owner config seq {s['config_seq']}"
             + ("  [BURN-ALL: no miner emissions]" if s["burn_all"] else ""))
    clock = s["clock"]
    if clock is not None:
        out.info(f"round {clock['round']}: closes at block {clock['closes_at_block']} "
                 f"({clock['blocks_left']} blocks left); cohort "
                 + ("opened" if clock["cohort_opened"] else "not opened yet"))
    out.info(f"checkpoint (anchor): {s['anchor']}   phase: {s['phase']}")
    k = s["king"]
    out.info(f"king: {k['hotkey'] or 'none'}  own promotions: {k['promotions']}"
             + (f", last in round {k['last_promotion_round']}" if k["last_promotion_round"] is not None else "")
             + ("  STALE: throne share burned until its next promotion" if k.get("stale") else ""))
    c = s["open_cohort"]
    out.info("open cohort: none" if c is None else
             f"open cohort: {c['key'][:16]}…  budget {c['budget']}  winner {c['winner'] or '-'}")
    out.info("jobs: " + (", ".join(f"{n} {st}" for st, n in sorted(s["jobs"].items())) or "none")
             + f"   spend {s['spend']['spent']}+{s['spend']['reserved']} reserved / {s['spend']['pool']}")
    if "you" in s:
        you = s["you"]
        out.info(f"you: {len(you['jobs'])} job(s), paid {you['paid_total']} units"
                 + ("  (you are king)" if you["is_king"] else ""))
        for jid, j in you["jobs"].items():
            out.info(f"  {jid[:16]}  {j['role']}  {j['state']}  spent {j['spent']}/{j['cap']}"
                     + (f"  ({j['reason']})" if j["reason"] else ""))
    return OK


def _key_from_env(var, out):
    key = (os.environ.get(var) or "").strip()
    if not key:
        out.fail("missing_api_key", f"no key in ${var} — export it first (never pass the key "
                                    "itself as an argument)", USAGE)
    return key


def _open_wallet(args, out):
    try:
        from ..shared.chain import open_wallet

        return open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path).hotkey
    except Exception as e:  # noqa: BLE001 — wallet errors are usage errors here
        out.fail("wallet_unavailable", f"could not load wallet for signing: {e}", USAGE)
        return None


def cmd_deposit_key(args) -> int:
    import urllib.request

    from ..funding.intake import compute_key_digest
    from .cli import _intake_transport_ok, build_fund_headers, post_intake

    out = reporter(args)
    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        return out.fail("insecure_intake", "refusing to send a compute key over plain http to a "
                                           "non-local intake; use https", USAGE)
    var = args.key_env or f"{args.provider.upper()}_API_KEY"
    key = _key_from_env(var, out)
    if not key:
        return USAGE
    hotkey = _open_wallet(args, out)
    if hotkey is None:
        return USAGE
    headers = build_fund_headers("credentials", hotkey.ss58_address, args.ref, "", hotkey.sign,
                                 key_digest=compute_key_digest(args.provider, key))
    headers.update({"X-Compute-Provider": args.provider, "X-Compute-Api-Key": key})
    req = urllib.request.Request(f"{base}/v2/direct/credentials", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    if 200 <= status < 300:
        out.result(provider=args.provider, status=body.get("status", "stored"),
                   hotkey=hotkey.ss58_address)
        out.info(f"{args.provider} key stored for {hotkey.ss58_address}; it is used only when a "
                 "job you fund names this provider.")
        return OK
    return out.fail(str(body.get("code") or "deposit_rejected"),
                    f"deposit rejected ({status}): {body.get('code', '?')} — "
                    f"{body.get('message', '')}", http_status=status)


def build_terms(args, network, state, payer):
    """The admission terms for this job, from the published journal and config."""
    from ..shared.direct_moe import digest

    key, cohort = (args.cohort, state["cohorts"].get(args.cohort)) if args.cohort else open_cohort(state)
    if cohort is None:
        raise ValueError("no open cohort to fund into" if not args.cohort
                         else f"unknown cohort {args.cohort}")
    guard = args.guard
    if guard is None:
        if len(network["guards"]) != 1:
            raise ValueError(f"the owner config lists {len(network['guards'])} guards; pick one "
                             "with --guard")
        guard = network["guards"][0]
    if guard not in network["guards"]:
        raise ValueError("--guard is not listed by the owner config")
    if args.role == "incumbent" and payer != state["king"]:
        raise ValueError("only the king funds the incumbent job")
    if args.role != "incumbent" and args.reward:
        raise ValueError("--reward applies only to the incumbent job")
    if args.world_size < 1 or args.spend_cap < 1:
        raise ValueError("--world-size and --spend-cap must be positive")
    contract = cohort["contract"]
    job = args.job or digest({"payer": payer, "generator": args.ref, "cohort": key})
    providers = sorted(set(args.provider))
    return job, {
        "cohort": key, "role": args.role, "payer": payer, "generator": args.ref,
        "funding_ref": f"{'+'.join(providers)}:{payer}:{job}",
        "anchor": contract["anchor"], "budget": contract["budget"],
        "spend_cap": args.spend_cap, "reward": args.reward, "slot": key,
        "profiles": sorted(set(args.profile)), "providers": providers,
        "world_size": args.world_size, "reference": args.reference or contract["anchor"],
        "guard_contract": guard,
    }


def cmd_fund(args) -> int:
    import json
    import urllib.request

    from ..direct.signing import sign
    from .cli import _intake_transport_ok, post_intake

    out = reporter(args)
    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        return out.fail("insecure_intake", "refusing plain http to a non-local intake; use https",
                        USAGE)
    if not args.dry_run:
        if args.max_spend is None:
            return out.fail("max_spend_required", "refusing: pass --max-spend (your own ceiling "
                            "for --spend-cap). Nothing was sent.", NEEDS_CONFIRM)
        if args.spend_cap > args.max_spend:
            return out.fail("over_max_spend", f"refusing: --spend-cap {args.spend_cap} exceeds "
                            f"--max-spend {args.max_spend}. Nothing was sent.", NEEDS_CONFIRM)
        if not args.yes:
            return out.fail("confirmation_required",
                            f"refusing: this job may spend up to {args.spend_cap} units on your "
                            f"{'/'.join(args.provider)} key. Nothing was sent; re-run with --yes "
                            "(or --dry-run to inspect).", NEEDS_CONFIRM)
    rc, network, _signed, state, _now = load_direct(args, out)
    if rc != OK:
        return rc
    if network["burn_all"]:
        out.warn("warning: the owner config has burn_all set — no miner emissions are paid now")
    hotkey = _open_wallet(args, out)
    if hotkey is None:
        return USAGE
    try:
        job, terms = build_terms(args, network, state, hotkey.ss58_address)
    except ValueError as e:
        return out.fail("bad_terms", f"refusing: {e}", USAGE)
    envelope = sign({"job": job, "nonce": job, "expires_at": int(time.time()) + args.expires_in,
                     "terms": terms}, hotkey, scope="fund-job")
    out.result(job=job, terms=terms, dry_run=args.dry_run)
    if args.dry_run:
        out.result(envelope=envelope)
        out.info(json.dumps(envelope, indent=2, sort_keys=True))
        out.info("dry run: nothing was sent")
        return OK
    req = urllib.request.Request(f"{base}/v2/direct/fund", method="POST",
                                 data=json.dumps(envelope).encode(),
                                 headers={"Content-Type": "application/json"})
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    if 200 <= status < 300:
        out.result(status=body.get("status", "admitted"))
        out.info(f"job {job[:16]}… {body.get('status', 'admitted')} into cohort {terms['cohort'][:16]}… "
                 f"(cap {terms['spend_cap']}); follow it with `cascade direct status --hotkey "
                 f"{hotkey.ss58_address}`")
        return OK
    return out.fail(str(body.get("code") or "fund_rejected"),
                    f"fund rejected ({status}): {body.get('code', '?')} — "
                    f"{body.get('message', '')}", http_status=status)
