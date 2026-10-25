"""``cascade-validator`` console-script entry point.

Runs the round loop indefinitely against the live chain. ``--offline`` builds
the runner and prints the loaded champion state without any chain or HF I/O — a
plumbing smoke check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .loop import build_runner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-validator", description="cascade validator loop.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--cache-dir", type=Path, default=None, help="Local cache for fetched pool/ckpts.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--private-eval-hosts", type=Path,
                   help="explicitly send private histories (never targets or keys) to this "
                   "operator-owned GPU host; unavailable host holds the round")
    p.add_argument("--private-eval-workers", type=int, default=1,
                   help="independent private-window workers; each loads its own model. "
                   "Default 1 preserves sequential execution; qualify on the chosen hardware")
    p.add_argument("--private-eval-timeout", type=int, default=3600,
                   help="total private GPU offload deadline in seconds")
    p.add_argument("--eval-hosts", type=Path, default=None,
                   help="hosts.toml with a GPU pod to offload the public-benchmark "
                        "evals to — the GIFT-Eval gate and, on the main validator, "
                        "the DEC-CA-0063 lane-A bench sweeps ([v2] bench_source = "
                        "\"validator\") (the first 'final'/'any'-stage host is used). Re-read at "
                        "EACH offloaded eval, so an elastic provisioner-rented pod "
                        "is picked up/dropped without a restart; a missing/empty "
                        "file just runs that eval locally on --device. Wallet + all "
                        "consensus decisions stay on this box; only the heavy eval "
                        "runs on the pod. Omit to always run locally.")
    p.add_argument("--force-burn", action="store_true",
                   help="Operator kill-switch: burn to [scoring] burn_uid on every "
                        "weight-set instead of voting the champion, while staying "
                        "Yuma-active. Same as [validator] force_burn = true. Champion "
                        "state is untouched; drop the flag and restart to resume voting.")
    p.add_argument("--offline", action="store_true", help="No chain/Hippius; print state and exit.")
    p.add_argument("--satellite", action="store_true",
                   help="Run as a DEC-CA-0058 satellite: replay the main validator's "
                        "([netconfig] owner_hotkey) signed receipts and set the vector the "
                        "replay derives (no model, no pool, no torch). Same as the "
                        "cascade-satellite entry point; --cache-dir is its state dir.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--direct-trainer-hotkey", default=None)
    p.add_argument("--direct-receipt", type=Path, default=None,
                   help="Replay an offline direct-MoE journal without chain access.")
    p.add_argument("--direct-journal-bucket", default=None,
                   help="Hippius S3 bucket holding the trainer's signed direct-MoE journal: "
                        "verify the signature, replay the trainer's scores (no evaluation) "
                        "and set weights from the latest settled payout boundary.")
    p.add_argument("--direct-poll-seconds", type=float, default=600.0)
    p.add_argument("--direct-owner-hotkey", default=None,
                   help="Owner (main validator) hotkey that signs the direct-MoE network "
                        "config. With --direct-config-bucket, the trainer hotkey, journal "
                        "bucket, burn uid, poll interval and allowed scoring rules all come "
                        "from that signed config instead of local flags/chain.toml.")
    p.add_argument("--direct-config-bucket", default=None,
                   help="Hippius S3 bucket holding the owner-signed direct-MoE network config.")
    p.add_argument("--publish-direct-config", type=Path, default=None,
                   help="Owner only: sign RULES.json with this wallet's hotkey as the next "
                        "network config version (seq/prev filled in), publish it to "
                        "--direct-config-bucket and commit its digest on chain (use a "
                        "dedicated registered hotkey: one plain commitment per hotkey).")
    # DEC-CA-0057 owner-signed network config (main validator hotkey only)
    p.add_argument("--netconfig-policy", action="store_true",
                   help="Print this release's supported overlay keys as JSON and exit; "
                        "no wallet, chain or storage access.")
    p.add_argument("--netconfig-overlay", type=Path, default=None,
                   help="Owner only: build the next DEC-CA-0057 network-config overlay from "
                        "this TOML file of overrides ([scoring], [eval], [round] schedule, "
                        "[v2] runtime policy, [lane_b], [lane_c], numeric [marketcast] "
                        "controls; see --netconfig-policy). DRY RUN by default: "
                        "prints the merged diff and every validator check; nothing is "
                        "signed or written without --netconfig-publish.")
    p.add_argument("--netconfig-effective-block", type=int, default=None,
                   help="The era start the overlay takes effect at (required with "
                        "--netconfig-overlay).")
    p.add_argument("--netconfig-comment", default="", help="Free text carried in the body.")
    p.add_argument("--netconfig-publish", action="store_true",
                   help="Sign with --wallet-name/--wallet-hotkey (must be [netconfig] "
                        "owner_hotkey) and publish to config/network/ in the manifest bucket.")
    return p


def _netconfig_overlay(args) -> int:
    """Owner: plan (dry run) or sign + publish the next network-config overlay."""
    import tomllib

    from ..shared.config import load_chain_config
    from ..shared.hippius import open_manifest_store
    from ..shared.netconfig import (
        NetConfigError,
        fetch_overlays,
        plan_overlay,
        publish_overlay,
        sign_overlay,
    )

    if args.netconfig_effective_block is None:
        print("--netconfig-effective-block is required", file=sys.stderr)
        return 2
    cfg = load_chain_config(args.chain_toml)
    owner = cfg.netconfig.owner_hotkey
    with args.netconfig_overlay.open("rb") as fh:
        overrides = tomllib.load(fh)
    previous = None
    try:
        store = open_manifest_store(cfg.storage)
        if owner:
            envs = fetch_overlays(store, owner=owner)
            previous = envs[-1]["body"] if envs else None
    except NetConfigError as e:
        print(f"the published overlay chain does not verify: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        if args.netconfig_publish:
            print(f"cannot read the published overlays ({e}); refusing to publish",
                  file=sys.stderr)
            return 2
        print(f"note: published overlays unreadable ({e}); planning as seq 1", file=sys.stderr)
        store = None
    client = None
    head = None
    if args.netconfig_publish:
        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network,
                                         wallet_name=args.wallet_name,
                                         wallet_hotkey=args.wallet_hotkey,
                                         wallet_path=args.wallet_path)
        head = int(client.current_block())
    plan = plan_overlay(cfg, overrides, args.netconfig_effective_block, previous,
                        comment=args.netconfig_comment, head_block=head)
    print(plan.render())
    if not plan.ok:
        return 1
    if not args.netconfig_publish:
        print("dry run: nothing signed or published (add --netconfig-publish)")
        return 0
    if not owner:
        print("[netconfig] owner_hotkey is empty in chain.toml: no validator would adopt it",
              file=sys.stderr)
        return 2
    keypair = client.wallet().hotkey
    if keypair.ss58_address != owner:
        print(f"wallet hotkey {keypair.ss58_address} is not [netconfig] owner_hotkey {owner}",
              file=sys.stderr)
        return 2
    key = publish_overlay(store, sign_overlay(plan.body, keypair))
    print(f"published overlay seq {plan.body['seq']} to {key} (+ config/network/latest.json)")
    return 0


def _publish_direct_config(args) -> int:
    """Owner: sign the next network config version, publish it, commit it on chain."""
    import json

    from ..direct.network import (
        COMMITMENT,
        check_round_clock,
        commitment_payload,
        latest_network_config,
        next_network_config,
        publish_network_config,
    )
    from ..direct.signing import sign
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config

    if not args.direct_config_bucket:
        print("--direct-config-bucket is required", file=sys.stderr)
        return 2
    cfg = load_chain_config(args.chain_toml)
    location = {"bucket": args.direct_config_bucket,
                "endpoint": cfg.storage.s3_endpoint, "region": cfg.storage.s3_region}
    client = ChainClient.from_config(cfg, network=args.network, wallet_name=args.wallet_name,
                                     wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path)
    keypair = client.wallet().hotkey
    current = client.plain_commitment(keypair.ss58_address)
    if current is not None and not current[0].startswith(COMMITMENT):
        # One plain commitment per hotkey: never clobber a DEC-CA-0045 readiness note.
        print(f"hotkey already holds a non-direct commitment ({current[0][:40]!r}); "
              "use a dedicated owner hotkey", file=sys.stderr)
        return 2
    previous = latest_network_config(location, owner=keypair.ss58_address)
    body = next_network_config(json.loads(args.publish_direct_config.read_text()), previous)
    check_round_clock(body["ledger_policy"], cfg.round)
    known = {e["panel"]: e["from_block"] for e in (previous or {}).get("private_panels", [])}
    # Margin for the commitment to land: validators refuse a new panel that
    # starts before the block the chain records the commitment at.
    floor = client.current_block() + 10
    early = [e["panel"][:12] for e in body["private_panels"]
             if known.get(e["panel"]) != e["from_block"] and e["from_block"] < floor]
    if early:
        print(f"new/changed panels must start at block >= {floor}: {', '.join(early)}",
              file=sys.stderr)
        return 2
    key = publish_network_config(location, sign(body, keypair, scope="network-config"))
    client.set_plain_commitment(commitment_payload(body))
    print(f"published direct network config seq {body['seq']} to {key} and committed it on chain")
    return 0


def _run_direct(args) -> int:
    """Direct-MoE validator: checks the trainer's signed scores, never evaluates.

    With an owner config (``--direct-owner-hotkey`` + ``--direct-config-bucket``)
    every rule is re-read from the owner's signed config each poll.
    """
    import json
    import time

    from ..direct.journal import fetch_journal
    from ..direct.network import fetch_network_config
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config
    from .direct_moe import (
        burn_vector,
        direct_weights,
        public_probes,
        verify_direct_receipt,
        verify_under_network,
    )

    log = logging.getLogger("cascade.validator.direct")
    owned = bool(args.direct_owner_hotkey and args.direct_config_bucket)
    if not owned and not (args.direct_trainer_hotkey and args.direct_journal_bucket):
        print("need --direct-owner-hotkey + --direct-config-bucket, or "
              "--direct-trainer-hotkey + --direct-journal-bucket", file=sys.stderr)
        return 2
    cfg = load_chain_config(args.chain_toml)
    storage = {"endpoint": cfg.storage.s3_endpoint, "region": cfg.storage.s3_region}
    client = None if args.offline else ChainClient.from_config(
        cfg, network=args.network, wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path)
    network, poll = None, args.direct_poll_seconds
    # The last accepted config survives restarts, so a panel the owner adds
    # later is always checked against its on-chain commitment block.
    saved = args.cache_dir / "direct-network.json" if args.cache_dir else None
    if owned and saved is not None and saved.exists():
        network = json.loads(saved.read_text())
    while True:
        try:
            if owned:
                committed = None
                if client is not None:
                    committed = client.plain_commitment(args.direct_owner_hotkey)
                    if committed is None:
                        raise ValueError("owner hotkey has no on-chain network config commitment")
                network = fetch_network_config({**storage, "bucket": args.direct_config_bucket},
                                               owner=args.direct_owner_hotkey, last=network,
                                               committed=committed)
                if saved is not None:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_text(json.dumps(network))
                poll = network["poll_seconds"]
            if network is not None and network["burn_all"] and client is not None:
                client.set_weight_vector(burn_vector(client.n_uids(), network["burn_uid"]))
                log.info("owner config burn_all: weights burned")
            else:
                journal_cfg = ({**storage, **network["journal"]} if network is not None
                               else {**storage, "bucket": args.direct_journal_bucket})
                signed = fetch_journal(journal_cfg)
                if client is None:
                    if network is None:
                        verify_direct_receipt(signed, expected_trainer=args.direct_trainer_hotkey)
                    else:
                        verify_under_network(signed, network)
                    print("direct-MoE journal verified and replayed (offline: no weights set)")
                    return 0
                available, revealed = (public_probes(storage, network) if network is not None
                                       else (None, None))
                weights = direct_weights(signed, expected_trainer=args.direct_trainer_hotkey,
                                         uid_of=client.uid_for_hotkey, n_uids=client.n_uids(),
                                         burn_uid=cfg.scoring.burn_uid, network=network,
                                         available=available, revealed=revealed,
                                         now_block=client.current_block())
                if weights is None:
                    log.info("no settled direct payout boundary yet; weights unchanged")
                else:
                    # Re-set every poll so the validator stays active between boundaries.
                    client.set_weight_vector(weights)
                    log.info("direct weights set from verified journal")
        except Exception:  # noqa: BLE001 — a bad fetch/replay keeps the old weights
            if client is None:
                raise
            log.exception("direct journal check failed; weights unchanged")
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    args = _build_parser().parse_args(argv)
    if args.netconfig_policy:
        import json

        from ..shared.netconfig import overlay_policy

        print(json.dumps(overlay_policy(), sort_keys=True))
        return 0
    # A validator holds its wallet and the private pool windows: miner
    # generator code runs here only in a hardened sandbox child (none does).
    from ..shared.sandbox_policy import mark_holds_secrets

    mark_holds_secrets("a validator wallet and the private eval pool")
    if args.direct_receipt is not None:
        import json

        from .direct_moe import verify_direct_receipt
        verify_direct_receipt(json.loads(args.direct_receipt.read_text()), expected_trainer=args.direct_trainer_hotkey)
        print("direct-MoE deterministic replay passed; verify transport authenticity separately")
        return 0
    if args.publish_direct_config is not None:
        return _publish_direct_config(args)
    if args.netconfig_overlay is not None:
        return _netconfig_overlay(args)
    if args.satellite:
        logging.basicConfig(level=args.log_level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        from .satellite import run_from_args

        return run_from_args(args)
    if args.direct_journal_bucket is not None or args.direct_config_bucket is not None:
        logging.basicConfig(level=args.log_level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        return _run_direct(args)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    eval_host_fn = None
    private_eval_host_fn = None
    if args.private_eval_workers < 1 or args.private_eval_timeout < 1:
        print("private evaluation workers and timeout must be positive", file=sys.stderr)
        return 2
    if args.private_eval_workers > 1 and args.private_eval_hosts is None:
        import torch

        torch.set_num_threads(1)  # one independent inference stream per worker
    if args.private_eval_hosts is not None:
        from .eval_offload import make_eval_host_fn

        private_eval_host_fn = make_eval_host_fn(args.private_eval_hosts, local_fallback=False)
    if args.eval_hosts is not None:
        # Lazy on purpose: the eval pod is elastic (provisioner-rented per round
        # manifest, torn down on the receipt), so the file is re-read at each
        # offloaded eval rather than pinned at startup — a missing/empty file at
        # eval time just means that eval runs locally on --device.
        from .eval_offload import make_eval_host_fn
        eval_host_fn = make_eval_host_fn(args.eval_hosts)
        logging.getLogger("cascade.validator").info(
            "eval offload configured from %s (re-resolved per eval; local fallback "
            "on --device=%s); wallet + consensus stay local", args.eval_hosts, args.device)

    state_lock = None
    if not args.offline:
        from ..shared.atomic import acquire_state_lock
        from ..shared.config import load_chain_config

        state_path = load_chain_config(args.chain_toml).validator.state_db_path
        try:
            state_lock = acquire_state_lock(str(state_path) + ".lock")
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2

    runner = build_runner(
        chain_toml=args.chain_toml,
        cache_dir=args.cache_dir,
        device=args.device,
        eval_host_fn=eval_host_fn,
        private_eval_host_fn=private_eval_host_fn,
        private_eval_workers=args.private_eval_workers,
        private_eval_timeout_s=args.private_eval_timeout,
    )

    if args.force_burn and not runner.cfg.validator.force_burn:
        from dataclasses import replace
        runner.cfg = replace(
            runner.cfg, validator=replace(runner.cfg.validator, force_burn=True))
    if runner.cfg.validator.force_burn:
        logging.getLogger("cascade.validator").warning(
            "FORCE-BURN mode: every weight-set burns to uid %d until this flag is "
            "removed; champion state still updates normally",
            runner.cfg.scoring.burn_uid)

    if args.offline:
        print(f"netuid:   {runner.cfg.netuid}")
        print(f"king:     {runner.state.king_hotkey}")
        print(f"tenure:   {runner.state.tenure_rounds}")
        print(f"dethrone_cp: {runner.cfg.scoring.dethrone_cp}")
        print(f"cascade: {'on' if runner.cfg.scoring.cascade_enabled else 'off'} "
              f"(reign_rounds={runner.cfg.scoring.cascade_reign_days})")
        if runner.cascade is not None:
            cs = runner.cascade.state
            print(f"cascade king: {cs.king_hotkey} "
                  f"(reign checkpoints logged: {len(cs.checkpoints)})")
        print(f"manifest_bucket: {runner.cfg.storage.manifest_bucket}")
        pool_bucket = runner.cfg.storage.pool_bucket
        if pool_bucket:
            print(f"eval pool: bucket={pool_bucket} (daily snapshots)")
        else:
            print(f"eval pool: static window_pool={runner.cfg.eval.window_pool!r}")
        print("offline validator smoke complete")
        return 0

    if args.wallet_name is None or args.wallet_hotkey is None:
        print("--wallet-name and --wallet-hotkey are required unless --offline", file=sys.stderr)
        return 2

    from ..shared.config import LaunchConfigError, assert_launch_ready

    try:
        assert_launch_ready(runner.cfg, role="validator")
    except LaunchConfigError as e:
        print(e, file=sys.stderr)
        return 2

    from ..shared.chain import ChainClient

    client = ChainClient.from_config(
        runner.cfg, network=args.network,
        wallet_name=args.wallet_name, wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path,
    )
    log = logging.getLogger("cascade.validator")
    # DEC-CA-0063: the wallet the validator bench report is signed with; its
    # hotkey == [netconfig] owner_hotkey makes this the MAIN validator (the
    # only one that sweeps lane-A benchmarks on its GPU).
    try:
        runner.bench_signer = client.wallet()
    except Exception as e:  # noqa: BLE001 — no wallet ⇒ never the main validator
        log.warning("bench signer unavailable (%s); this validator never sweeps", e)
    # A MAIN validator (hotkey == [netconfig] owner_hotkey) whose wallet cannot
    # sign refuses to start: it would otherwise publish no usable report.
    from .validator_bench import check_main_bench_signer

    try:
        check_main_bench_signer(runner.cfg, runner.bench_signer)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    if runner.cfg.storage.pool_bucket:
        from .pool import load_bucket_pool

        log.info("loading daily eval pool from bucket %s …", runner.cfg.storage.pool_bucket)
        window_source = load_bucket_pool(runner.cfg, cache_dir=args.cache_dir)
    else:
        from .pool import load_pool

        log.info("loading static eval pool %s …", runner.cfg.eval.window_pool)
        window_source = load_pool(runner.cfg, cache_dir=args.cache_dir)
    log.info("validator up: netuid=%s manifest_bucket=%s — polling for rounds",
             runner.cfg.netuid, runner.cfg.storage.manifest_bucket)
    # bittensor's logging machine silences all other loggers on import; restore
    # cascade.* levels so the run-loop's progress logs stay visible.
    from ..shared.logging_util import restore_cascade_logging

    restore_cascade_logging(args.log_level)
    runner.run_forever(client, window_source=window_source)
    if state_lock is not None:
        state_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
