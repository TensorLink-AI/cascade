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
    p.add_argument("--eval-hosts", type=Path, default=None,
                   help="hosts.toml with a GPU pod to offload the GIFT-Eval gate to "
                        "(the first 'final'/'any'-stage host is used). Re-read at "
                        "EACH offloaded eval, so an elastic provisioner-rented pod "
                        "is picked up/dropped without a restart; a missing/empty "
                        "file just runs that eval locally on --device. Wallet + all "
                        "consensus decisions stay on this box; only the heavy eval "
                        "runs on the pod. Omit to always run locally.")
    p.add_argument("--offline", action="store_true", help="No chain/Hippius; print state and exit.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    eval_host_fn = None
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

    runner = build_runner(
        chain_toml=args.chain_toml,
        cache_dir=args.cache_dir,
        device=args.device,
        eval_host_fn=eval_host_fn,
    )

    if args.offline:
        print(f"netuid:   {runner.cfg.netuid}")
        print(f"king:     {runner.state.king_hotkey}")
        print(f"tenure:   {runner.state.tenure_rounds}")
        print(f"dethrone_cp: {runner.cfg.scoring.dethrone_cp}")
        print(f"cascade: {'on' if runner.cfg.scoring.cascade_enabled else 'off'} "
              f"(reign_days={runner.cfg.scoring.cascade_reign_days})")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
