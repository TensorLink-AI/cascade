"""Operator entrypoint for the durable direct runtime (activation remains gated)."""

import argparse
import json
import time
from pathlib import Path

from ..eval.direct_guard import GuardContract
from ..funding.direct_ledger import DirectLedger
from ..trainer.direct_moe import DirectController
from .evaluation import PrivatePolicy, PublicSidecar
from .evaluator import CheckpointEvaluator
from .runtime import DirectRuntime, LocalCPUProvider, LocalTorchDispatcher
from .storage import open_objects


def build(config, *, allow_paid=False):
    root = Path(config["state_dir"])
    root.mkdir(parents=True, exist_ok=True)
    ledger = DirectLedger(root / "ledger.sqlite", config["ledger_policy"])
    objects = open_objects(config["storage"])
    if config["provider"]["kind"] == "local-cpu":
        providers = {"local": LocalCPUProvider()}
        dispatcher = LocalTorchDispatcher(root / "workers")
    elif config["provider"]["kind"] == "pods":
        from ..funding.vault import compute_vaults
        from .compute import PayerPodProvider, PodProfile
        from .pods import PodDispatcher
        from .profiles import check_operator_profiles

        cfg = config["provider"]
        vaults = compute_vaults(config["vault_dir"], ttl_seconds=config["vault_ttl_hours"] * 3600.0)
        profiles = {name: PodProfile(**value) for name, value in cfg["profiles"].items()}
        check_operator_profiles(profiles)
        # Each marketplace is its own provider; a job's admitted terms name the
        # one its payer chose, and its key comes from that marketplace's vault.
        providers = {
            compute: PayerPodProvider(
                ledger, vaults[compute], compute, profiles, root / "providers",
                ssh_pubkey=Path(cfg["ssh_pubkey_file"]).read_text().strip(),
                netuid=cfg["netuid"], allow_paid=allow_paid)
            for compute in cfg["computes"]
        }
        dispatcher = PodDispatcher(
            objects, providers, profiles, staging_dir=root / "staging", **config["dispatch"])
    else:
        raise ValueError("unsupported provider")
    ev = config["evaluation"]
    block_source = _block_source(config.get("chain"))
    sidecar = PublicSidecar(**ev["sidecar"]) if "sidecar" in ev else None
    evaluator = CheckpointEvaluator(
        objects,
        root / "evaluation",
        guard=GuardContract.from_dict(ev["guard"]),
        private_policy=PrivatePolicy(**ev["private_policy"]),
        private_panel=ev["private_panel"],
        public_panels=ev.get("fixture_panels"),
        sidecar=sidecar,
        initial_artifacts=ev["initial_artifacts"],
        samples=ev["samples"],
        seed=ev["seed"],
        device=ev["device"],
        block_source=block_source,
    )
    trainer_key = None
    if config.get("trainer_wallet"):
        from bittensor_wallet import Wallet

        trainer_key = Wallet(**config["trainer_wallet"]).hotkey
    return DirectRuntime(
        DirectController(ledger, providers, objects, clock=time.time, block=block_source),
        dispatcher, evaluator, trainer_key=trainer_key
    )


def _block_source(chain):
    """``{"network", "chain_toml"?}`` → the live chain block, read at scoring time."""
    if not chain:
        return None
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config

    client = ChainClient.from_config(load_chain_config(chain.get("chain_toml")),
                                     network=chain.get("network", "finney"))
    return client.current_block


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--allow-paid",
        action="store_true",
        help="explicit operator authorization to rent payer-funded pods",
    )
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--tick", action="store_true")
    action.add_argument("--serve", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--export", type=Path)
    action.add_argument("--bind", type=Path)
    action.add_argument("--cohort", type=Path)
    action.add_argument("--settle")
    action.add_argument("--advance")
    action.add_argument("--resume")
    action.add_argument("--payout", type=int, metavar="EMISSION_UNITS",
                        help="record and settle a payout boundary now (weights, not transfers)")
    p.add_argument("--paused", action="store_true")
    action.add_argument("--weights", action="store_true",
                        help="print the latest settled boundary's reward units per hotkey")
    action.add_argument("--publish", action="store_true",
                        help="publish the signed journal to Hippius for validators")
    action.add_argument("--reveal-panel", type=Path, metavar="PANEL_JSON",
                        help="publish a retired private panel so audits can reproduce its scores")
    action.add_argument("--merge", nargs="*")
    p.add_argument("--optimizer-policy", choices=("reset_edited_rows",))
    p.add_argument("--evidence")
    p.add_argument("--routing-stable", action="store_true")
    args = p.parse_args(argv)
    config = json.loads(args.config.read_text())
    runtime = build(config, allow_paid=args.allow_paid)
    ctl = runtime.controller
    try:
        if args.status:
            print(
                json.dumps(
                    {
                        job: {
                            k: j[k]
                            for k in (
                                "state",
                                "consumed",
                                "attempt",
                                "reason",
                                "next_retry",
                                "fence",
                            )
                        }
                        for job, j in ctl.ledger.state["jobs"].items()
                    },
                    sort_keys=True,
                )
            )
        elif args.export:
            args.export.write_text(
                json.dumps(
                    runtime.signed_journal() if runtime.trainer_key else ctl.ledger.export(),
                    sort_keys=True,
                )
            )
        elif args.cohort:
            from ..shared.direct_moe import BatchSchedule, Cohort

            raw = json.loads(args.cohort.read_text())
            schedule = raw["schedule"]
            schedule["thresholds"] = tuple(schedule["thresholds"])
            schedule["batches"] = tuple(schedule["batches"])
            ctl.cohort(Cohort(**{**raw, "schedule": BatchSchedule(**schedule)}), now=time.time())
        elif args.bind:
            spec = json.loads(args.bind.read_text())
            ctl.apply(
                f"bind:{spec['job']}", "bind_training", time.time(), job=spec["job"], spec=spec
            )
        elif args.merge is not None:
            if not args.optimizer_policy:
                raise ValueError("explicit edited optimizer policy required")
            runtime.merge(args.merge, optimizer_policy=args.optimizer_policy, now=time.time())
        elif args.payout is not None:
            boundary = int(time.time())
            # The boundary IS the recorded protocol time, so bypass the clock override.
            stamp = {}
            if "rounds" in ctl.ledger.state:
                if ctl.block is None:
                    raise ValueError("round journal payouts need a \"chain\" block source")
                stamp = {"block": int(ctl.block())}
            ctl.ledger.apply(f"payout:{boundary}", "payout_boundary", now=boundary,
                             boundary=boundary, emission=args.payout, paused=args.paused, **stamp)
            ctl.ledger.apply(f"settle-boundary:{boundary}", "settle_boundary", now=boundary,
                             boundary=boundary)
            print(json.dumps({"boundary": boundary}, sort_keys=True))
        elif args.weights:
            from .weights import boundary_units, latest_settled_boundary

            boundary = latest_settled_boundary(ctl.ledger.state)
            print(json.dumps(None if boundary is None else
                             {"boundary": boundary, **boundary_units(ctl.ledger.state, boundary)},
                             sort_keys=True))
        elif args.publish:
            from .journal import publish_journal

            print(publish_journal(config["journal"], runtime.signed_journal()))
        elif args.reveal_panel:
            from ..shared.direct_moe import digest
            from .journal import reveal_panel

            panel = json.loads(args.reveal_panel.read_text())
            # Revealing burns the panel: only one already rotated out of scoring.
            if digest(panel) == digest(config["evaluation"]["private_panel"]):
                raise ValueError("rotate the private panel before revealing it")
            body = runtime.evaluator.objects.get(panel["object"]["uri"])
            print(reveal_panel(config["journal"], panel, body))
        elif args.resume:
            if not args.evidence:
                raise ValueError("recovery evidence required")
            from ..shared.direct_moe import digest

            ctl.apply(
                f"resume:{args.resume}:{digest(args.evidence)}",
                "resume_queue",
                time.time(),
                job=args.resume,
                evidence=args.evidence,
            )
        elif args.settle:
            runtime.settle(args.settle, now=time.time())
        elif args.advance:
            ctl.advance(args.advance, routing_stable=args.routing_stable, now=time.time())
        else:
            while True:
                print(json.dumps(runtime.tick(time.time()), sort_keys=True), flush=True)
                for provider in ctl.providers.values():
                    if hasattr(provider, "reap_expired"):
                        provider.reap_expired(time.time())
                if args.tick:
                    break
                time.sleep(config["poll_seconds"])
    finally:
        runtime.evaluator.close()
        for provider in ctl.providers.values():
            if hasattr(provider, "close"):
                provider.close()
        ctl.ledger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
