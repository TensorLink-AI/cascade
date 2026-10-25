"""Public-data preparation for local MarketCast experiments; never rents compute."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from .output import NETWORK, OK, USAGE, reporter


def add_data_commands(sub):
    p = sub.add_parser("snapshot", help="Fetch public market bars into a digest-checked local training or evaluation snapshot.")
    p.add_argument("repo_dir", type=Path, help="Recipe directory.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--evaluation", action="store_true", help="Fetch a recent held-out snapshot instead of the recipe's training range.")
    p.add_argument("--days", type=int, default=7, help="Days of held-out market data (1..30, default 7).")
    p.set_defaults(func=cmd_snapshot, command="lane-c snapshot")
    b = sub.add_parser("fetch-base", help="Download and digest-check the configured pinned Timestral base; CPU/file work only.")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--chain-toml", type=Path, default=None)
    b.set_defaults(func=cmd_fetch_base, command="lane-c fetch-base")


def cmd_fetch_base(args):
    from ..eval.timestral_adapter import verify_checkpoint
    from ..shared.config import load_chain_config
    from ..shared.config.marketcast import split_base_ref

    out = reporter(args)
    if args.out.exists():
        return out.fail("destination_exists", "Choose a new directory for the base.", USAGE)
    mc = load_chain_config(args.chain_toml).marketcast
    repo, revision = split_base_ref(mc.base_ref)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=str(args.out),
                          allow_patterns=["config.json", "model.safetensors"])
        verify_checkpoint(args.out / "model.safetensors", mc.base_sha256)
    except Exception:  # noqa: BLE001 — downloader exceptions can include auth headers
        return out.fail("base_download_failed", "Pinned base download or digest verification failed; no model was loaded.", NETWORK)
    out.result(path=str(args.out), base_ref=mc.base_ref, base_sha256=mc.base_sha256, digest_verified=True)
    return OK


def cmd_snapshot(args):
    from ..pool.marketcast import build_snapshot
    from ..pool.marketcast_feeds import http_get_json, http_post_json
    from ..shared.config import load_chain_config
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from ..trainer.marketcast_leg import build_training_snapshot
    from ..trainer.marketcast_runner import has_generator_tree

    out = reporter(args)
    if args.out.exists() or not 1 <= args.days <= 30:
        return out.fail("bad_snapshot_options", "Use a new output directory and --days between 1 and 30.", USAGE)
    cfg = load_chain_config(args.chain_toml)
    now = time.time()
    try:
        recipe = parse_marketcast_recipe(json.loads((args.repo_dir / "marketcast.json").read_text()),
                                        cfg.marketcast, submission_time=now,
                                        has_generator=has_generator_tree(args.repo_dir))
    except (ValueError, OSError) as exc:
        return out.fail("bad_recipe", str(exc), USAGE)
    # Fetch only the recipe universe, rather than every asset configured live.
    mc = replace(cfg.marketcast, assets=recipe.assets, suites=(recipe.suite,))
    try:
        if args.evaluation:
            meta = build_snapshot(args.out, mc, end_time=now, days=args.days, anchor_block=0,
                                  anchor_time=now, post=http_post_json, get=http_get_json)
        else:
            meta = build_training_snapshot(args.out, mc, recipe, post=http_post_json, get=http_get_json, not_after=now)
    except Exception:  # noqa: BLE001 — public upstream errors should remain retryable
        return out.fail("market_data_unavailable", "Public market data preparation failed; retry or choose a supported date range.", NETWORK)
    out.result(path=str(args.out), kind="evaluation" if args.evaluation else "training", metadata=meta,
               validator_signed=False, gpu_rented=False)
    out.info("Local public-data snapshot prepared; it is not validator evidence.")
    return OK
