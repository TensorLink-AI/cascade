"""``metronome-trainer`` console-script — the owner's training service.

The base-model training backend is owner-supplied (the GPU boundary), passed as
``--trainer module:Class`` and instantiated with no args. For real runs the
trainer also needs a wallet (to read the metagraph for the reigning king and to
sign the manifest) and an HF token (to push checkpoints).

``--offline`` skips the chain and prints the round's contract digest and derived
seeds — a config + plumbing smoke check that needs neither GPU nor network.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from ..shared.config import load_chain_config
from ..shared.manifest import contract_digest
from .contract import RoundSeeds


def _load_trainer(spec: str):
    """Instantiate ``module:Class`` (no-arg constructor) as the BaseTrainer."""
    mod_name, _, cls_name = spec.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"--trainer must be 'module:Class'; got {spec!r}")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metronome-trainer", description="metronome trainer service.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--trainer", default=None, help="BaseTrainer as 'module:Class'.")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"))
    p.add_argument("--network", default="finney")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--max-challengers", type=int, default=1)
    p.add_argument(
        "--remote-hosts", type=Path, default=None,
        help="Trainer-local TOML of SSH GPU pods ([[host]] tables). When set, king "
             "and challenger train in parallel on separate pods (see trainer/remote.py).",
    )
    p.add_argument("--base-seed", type=int, default=0, help="Override round base seed (offline).")
    p.add_argument("--offline", action="store_true", help="No chain/GPU; print contract + seeds.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_chain_config(args.chain_toml)

    if args.offline:
        from .contract import compute_base_arch_digest

        seeds = RoundSeeds.derive(args.base_seed, cfg.training)
        computed = compute_base_arch_digest(cfg.training)
        print(f"base_arch:           {cfg.training.base_arch} ({cfg.training.arch_preset})")
        print(f"base_arch_digest:    {cfg.training.base_arch_digest}  (in chain.toml)")
        print(f"computed_arch_digest: {computed}")
        if cfg.training.base_arch_digest != computed:
            print("  ^ MISMATCH — pin [training] base_arch_digest to the computed value above")
        print(
            f"budget:              {cfg.training.target_train_hours:g}h on ref GPU "
            f"≈ {cfg.training.train_tokens:,} point-passes (from scratch)"
        )
        print(f"contract_digest:     {contract_digest(cfg.training)}")
        print(f"generation_seed:     {seeds.generation_seed}")
        print(f"training_seed:       {seeds.training_seed}")
        print("offline trainer smoke complete")
        return 0

    if not args.trainer:
        print("--trainer module:Class is required for a live run", flush=True)
        return 2
    if args.wallet_name is None or args.wallet_hotkey is None:
        print("--wallet-name and --wallet-hotkey are required for a live run", flush=True)
        return 2

    from ..shared.config import LaunchConfigError, assert_launch_ready

    try:
        assert_launch_ready(cfg, role="trainer")
    except LaunchConfigError as e:
        print(e, flush=True)
        return 2

    from ..shared.chain import ChainClient
    from .loop import TrainerRunner

    base_trainer = _load_trainer(args.trainer)
    client = ChainClient.from_config(
        cfg, network=args.network,
        wallet_name=args.wallet_name, wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
    )

    remote_hosts = None
    if args.remote_hosts is not None:
        from .remote import load_hosts

        remote_hosts = load_hosts(args.remote_hosts)
        logging.getLogger("metronome.trainer").info(
            "remote training across %d pod(s): %s",
            len(remote_hosts), ", ".join(h.name for h in remote_hosts),
        )

    runner = TrainerRunner(
        cfg=cfg,
        base_trainer=base_trainer,
        work_root=args.work_root,
        wallet=client.wallet(),
        remote_hosts=remote_hosts,
        trainer_spec=args.trainer,
    )
    logging.getLogger("metronome.trainer").info(
        "trainer up: netuid=%s manifest_bucket=%s registry=%s mode=%s",
        cfg.netuid, cfg.storage.manifest_bucket, cfg.storage.ipfs_api_url,
        "remote" if remote_hosts else "local",
    )
    runner.run_forever(client, max_challengers=args.max_challengers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
