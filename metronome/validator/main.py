"""``metronome-validator`` console-script entry point.

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
    p = argparse.ArgumentParser(prog="metronome-validator", description="metronome validator loop.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--cache-dir", type=Path, default=None, help="Local cache for fetched pool/ckpts.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--offline", action="store_true", help="No chain/Hippius; print state and exit.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runner = build_runner(
        chain_toml=args.chain_toml,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    if args.offline:
        print(f"netuid:   {runner.cfg.netuid}")
        print(f"king:     {runner.state.king_hotkey}")
        print(f"tenure:   {runner.state.tenure_rounds}")
        print(f"dethrone_cp: {runner.cfg.scoring.dethrone_cp}")
        print(f"manifest_bucket: {runner.cfg.storage.manifest_bucket}")
        print(f"window_pool: {runner.cfg.eval.window_pool}")
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
    from .pool import load_pool

    client = ChainClient.from_config(
        runner.cfg, network=args.network,
        wallet_name=args.wallet_name, wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path,
    )
    log = logging.getLogger("metronome.validator")
    log.info("loading private eval pool %s …", runner.cfg.eval.window_pool)
    window_source = load_pool(runner.cfg, cache_dir=args.cache_dir)
    log.info("validator up: netuid=%s manifest_bucket=%s — polling for rounds",
             runner.cfg.netuid, runner.cfg.storage.manifest_bucket)
    runner.run_forever(client, window_source=window_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
