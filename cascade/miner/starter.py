"""Create source-only starter entries without network access or credentials."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

from .output import OK, USAGE, reporter


def add_init(sub):
    p = sub.add_parser("init", help="Create an offline lane A/B generator or lane-C MarketCast recipe; never overwrites a directory.")
    p.add_argument("repo_dir", type=Path)
    p.add_argument("--lane", choices=("a", "b", "c"), default="a")
    p.add_argument("--target", default=None)
    p.add_argument("--chain-toml", type=Path, default=None)
    p.set_defaults(func=cmd_init)


def marketcast_starter(mc, target):
    from ..shared.config.marketcast import SUITE_CLASSES

    suite = target.removeprefix("mkt:")
    if target != f"mkt:{suite}" or suite not in mc.suites:
        raise ValueError("Choose a configured mkt:<suite> target.")
    cls, assets = next((c, syms) for c, syms in mc.assets_map.items() if syms and c in SUITE_CLASSES[suite])
    end = datetime.now(UTC).date() - timedelta(days=int(mc.embargo_days) + 1)
    return {"target": target,
            "data": {"assets": {cls: list(assets[:2])}, "class_weights": {cls: 1.0},
                     "frequencies": list(mc.frequencies_map[suite]),
                     "start": str(end - timedelta(days=60)), "end": str(end)},
            "context_length": min(4096, mc.max_context_length),
            "horizon_mix": {h: 1.0 for h in mc.horizons_map[suite]},
            "covariates": [], "augmentations": [],
            "objective": {"suite_weights": {suite: 1.0}, "coverage_penalty": 0.5},
            "optimizer": {"lr": min(1e-4, mc.max_lr), "schedule": "cosine", "warmup_fraction": 0.05,
                          "steps": min(100, mc.max_steps), "batch": min(8, mc.max_batch),
                          "weight_decay": 0.01, "clip": 1.0, "ema_decay": 0.999},
            "trainable": {"scope": "lora", "lora_rank": min(8, mc.max_lora_rank)}, "synthetic_fraction": 0.0}


def cmd_init(args):
    from ..shared.config import load_chain_config

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    if args.repo_dir.exists():
        return out.fail("destination_exists", "Choose a new directory; existing files are never overwritten.", USAGE)
    if args.lane == "a" and args.target:
        return out.fail("bad_target", "Lane A has no target.", USAGE)
    if args.lane in ("b", "c") and not args.target:
        return out.fail("target_required", "Pass --target for lane B or C.", USAGE)
    try:
        recipe = None
        if args.lane == "c":
            from ..shared.marketcast_recipe import parse_marketcast_recipe
            recipe = marketcast_starter(cfg.marketcast, args.target)
            parse_marketcast_recipe(recipe, cfg.marketcast, target=args.target)
        elif args.lane == "b":
            from .cli import expert_targets
            _, error = expert_targets(cfg, args.target)
            if error:
                raise ValueError(error)
    except (ValueError, StopIteration) as exc:
        return out.fail("bad_target", str(exc), USAGE)
    args.repo_dir.mkdir(parents=True)
    try:
        if recipe is not None:
            (args.repo_dir / "marketcast.json").write_text(json.dumps(recipe, indent=2) + "\n")
        else:
            template = resources.files("cascade.miner").joinpath("templates", "generator")
            for name in ("generator.py", "config.json", "requirements.txt"):
                (args.repo_dir / name).write_bytes(template.joinpath(name).read_bytes())
        (args.repo_dir / "MINING.md").write_text(
            f"# Lane {args.lane.upper()} starter\n\n"
            f"Declared target: {args.target or 'none'}.\n\n"
            "Read the deployed llms.txt and run cascade status --json before choosing live rules.\n"
            "See docs/AGENT_MINING.md for install, held-out pools, local tests, upload, reveal, funding and outcome tracking.\n"
            "This starter does not establish that a lane is open or that an entry will earn.\n"
            "Do not submit, commit or fund without the human's authorization.\n")
    except Exception:
        shutil.rmtree(args.repo_dir)
        raise
    out.result(path=str(args.repo_dir), lane=args.lane, target=args.target, network_access=False,
               next_command=f"cascade {'lane-c check' if args.lane == 'c' else 'verify'} {args.repo_dir} "
                            + (f"--target {args.target}" if args.lane == "c" else "--quick") + " --json")
    return OK
