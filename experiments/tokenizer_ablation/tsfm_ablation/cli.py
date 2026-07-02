"""Command-line entry point: ``tsfm-ablation <command>``.

Commands
--------
  corpus      build the sharded synthetic corpus (numpy only, no GPU)
  train       train one arm/seed
  matrix      run the full arms x seeds matrix (train + GIFT + probe + push)
  smoke       tiny end-to-end pipeline check (fast; --gift to include GIFT)
  lr-sweep    3-point LR calibration on the T0 control arm
  eval        GIFT-Eval a finished run's final checkpoint
  aggregate   print the cross-run aggregate table + tripwires
  quick-test  GIFT-Eval wiring pre-flight (seasonal-naive on one dataset)

Storage is resolved from --ckpt-dir/--gift-dir or the TSFM_CKPT_DIR/TSFM_GIFT_DIR
env vars (see paths.resolve_storage).
"""
from __future__ import annotations

import argparse

from .config import ARMS, SEEDS, make_arm
from .paths import corpus_dir as _corpus_dir
from .paths import resolve_storage


def _add_storage_args(p):
    p.add_argument("--ckpt-dir", default=None,
                   help="checkpoints/corpus/results root (default: $TSFM_CKPT_DIR or ./tsfm_ckpts)")
    p.add_argument("--gift-dir", default=None,
                   help="GIFT-Eval dataset cache (default: $TSFM_GIFT_DIR or ./gifteval_data)")


def _load_run(arm, seed, ckpt_dir):
    """Load a finished run's final checkpoint. Returns (model, cfg)."""
    import os

    import torch

    from .device import DEVICE
    from .model import MiniTSFM2
    from .train import run_dir
    cfg = make_arm(arm, seed)
    latest = os.path.join(run_dir(cfg, ckpt_dir), "latest.pt")
    if not os.path.exists(latest):
        raise SystemExit(f"no checkpoint at {latest} — train this arm/seed first")
    model = MiniTSFM2(cfg).to(DEVICE)
    model.load_state_dict(torch.load(latest, map_location=DEVICE,
                                     weights_only=False)["model"])
    return model, cfg


def _apply_overrides(cfg, args):
    for attr in ("steps", "batch", "n_variates"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def build_parser():
    ap = argparse.ArgumentParser(prog="tsfm-ablation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -- corpus ------------------------------------------------------------
    p = sub.add_parser("corpus", help="build the sharded synthetic corpus")
    _add_storage_args(p)
    p.add_argument("--pools", nargs="+", default=["short", "long"],
                   choices=["short", "long"])

    # -- train -------------------------------------------------------------
    p = sub.add_parser("train", help="train one arm/seed")
    _add_storage_args(p)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--n-variates", dest="n_variates", type=int, default=None)

    # -- matrix ------------------------------------------------------------
    p = sub.add_parser("matrix", help="run the full arms x seeds matrix")
    _add_storage_args(p)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--no-gift", dest="gift", action="store_false",
                   help="skip GIFT-Eval (train + probe only)")
    p.add_argument("--hf-push", action="store_true", help="upload artifacts to HF Hub")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")

    # -- smoke -------------------------------------------------------------
    p = sub.add_parser("smoke", help="tiny end-to-end pipeline check")
    _add_storage_args(p)
    p.add_argument("--gift", action="store_true", help="include GIFT-Eval in the smoke")

    # -- lr-sweep ----------------------------------------------------------
    p = sub.add_parser("lr-sweep", help="3-point LR calibration on T0")
    _add_storage_args(p)
    p.add_argument("--steps", type=int, default=2_000)

    # -- eval --------------------------------------------------------------
    p = sub.add_parser("eval", help="GIFT-Eval a finished run's final checkpoint")
    _add_storage_args(p)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full", action="store_true",
                   help="run the full 97-config leaderboard (default: 14-task dev subset)")

    # -- time --------------------------------------------------------------
    p = sub.add_parser("time", help="TIME-benchmark a finished run's final checkpoint")
    _add_storage_args(p)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-dir", default=None,
                   help="Real-TSF/TIME dataset path (default: $TSFM_TIME_DIR)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="restrict to these name or name/term specs (default: all)")
    p.add_argument("--max-tasks", type=int, default=None, help="cap tasks for a smoke run")

    # -- aggregate ---------------------------------------------------------
    p = sub.add_parser("aggregate", help="print the cross-run aggregate table")
    _add_storage_args(p)

    # -- quick-test --------------------------------------------------------
    p = sub.add_parser("quick-test", help="GIFT-Eval wiring pre-flight")
    _add_storage_args(p)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    ckpt_dir, gift_dir = resolve_storage(getattr(args, "ckpt_dir", None),
                                         getattr(args, "gift_dir", None))
    cdir = _corpus_dir(ckpt_dir)

    if args.cmd == "corpus":
        from .corpus import build_corpus
        build_corpus(cdir, pools=tuple(args.pools))
        return

    if args.cmd == "train":
        from .train import train_one
        cfg = _apply_overrides(make_arm(args.arm, args.seed), args)
        train_one(cfg, ckpt_dir, cdir)
        return

    if args.cmd == "matrix":
        from .config import default_runs
        from .runner import run_matrix
        runs = default_runs(arms=tuple(args.arms), seeds=tuple(args.seeds))
        run_matrix(runs, ckpt_dir, cdir, gift_dir, do_gift=args.gift,
                   hf_push=args.hf_push, hf_repo=args.hf_repo, wandb_enable=args.wandb)
        return

    if args.cmd == "smoke":
        from .runner import make_smoke_runs, run_matrix
        run_matrix(make_smoke_runs(), ckpt_dir, cdir, gift_dir,
                   smoke=True, do_gift=args.gift)
        return

    if args.cmd == "lr-sweep":
        from .runner import lr_sweep
        lr_sweep(ckpt_dir, cdir, steps=args.steps)
        return

    if args.cmd == "eval":
        from .gift_eval import eval_gift_dev, eval_gift_full
        model, cfg = _load_run(args.arm, args.seed, ckpt_dir)
        run = eval_gift_full if args.full else eval_gift_dev
        run(model, gift_dir, tag=f"{cfg.run_name}@final")
        return

    if args.cmd == "time":
        from .time_eval import eval_time
        model, cfg = _load_run(args.arm, args.seed, ckpt_dir)
        eval_time(model, time_dir=args.time_dir, datasets=args.datasets,
                  max_tasks=args.max_tasks)
        return

    if args.cmd == "aggregate":
        from .probes import aggregate_table, load_results
        aggregate_table(load_results(ckpt_dir))
        return

    if args.cmd == "quick-test":
        from .gift_eval import quick_wiring_test
        quick_wiring_test(gift_dir)
        return


if __name__ == "__main__":
    main()
