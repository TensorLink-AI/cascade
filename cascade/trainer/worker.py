"""``cascade-train-worker`` — train **one** generator for one round on this box.

This is what the orchestrator runs over SSH on each GPU pod (see
:mod:`cascade.trainer.remote`). It does exactly one role's work — fetch the
generator from the Hippius Hub registry by ref, build the corpus in the sandbox,
train a fresh model under the fixed contract, upload the checkpoint to the
registry, and print a :class:`TrainedEntry` receipt — and **never touches the
wallet or the manifest**. The orchestrator collects receipts from king and
challenger pods and signs + publishes the manifest itself.

It reuses :meth:`cascade.trainer.loop.TrainerRunner.train_one`, so a remote run
and a local run are byte-for-byte the same code path; only *where* it runs
differs. Seeds are derived from ``--base-seed`` exactly as the orchestrator
derives them, so the run is reproducible.

The receipt is printed to **stdout** prefixed with
:data:`cascade.trainer.remote.RECEIPT_SENTINEL`; all logs go to **stderr**.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from .contract import RoundSeeds
from .loop import ResolvedGenerator, TrainerRunner
from .main import _load_trainer
from .remote import RECEIPT_SENTINEL


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-train-worker",
                                description="Train one generator for one round (remote worker).")
    p.add_argument("--gen-ref", required=True, help="Generator's Hippius Hub ref (repo@digest).")
    p.add_argument("--uid", type=int, required=True, help="Miner UID.")
    p.add_argument("--hotkey", required=True, help="Miner hotkey.")
    p.add_argument("--role", required=True, choices=["king", "challenger"])
    p.add_argument("--base-seed", type=int, required=True, help="Round base seed (block-hash int).")
    p.add_argument("--block", type=int, required=True, help="Round block height.")
    p.add_argument("--trainer", required=True, help="BaseTrainer as 'module:Class'.")
    p.add_argument("--arch-preset", default=None,
                   help="Size to train (primary if omitted, or a [[training.sizes]] arch_preset).")
    p.add_argument("--train-hours", type=float, default=None,
                   help="Override the training budget in hours (e.g. a cheap heat screen at "
                        "[round] heat_train_hours). Omitted ⇒ the size's full [training] budget.")
    p.add_argument("--repo-suffix", default="",
                   help="Suffix on the checkpoint repo (ckpt-r<seed>-<role>-<size><suffix>) so "
                        "parallel runs at one size — heat challengers, finalists>1 — don't collide.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Logs to stderr so stdout carries only the receipt.
    logging.basicConfig(
        level=args.log_level, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("cascade.trainer.worker")

    cfg = load_chain_config(args.chain_toml)
    base_trainer = _load_trainer(args.trainer)
    # No wallet on a rented box: this worker never signs or publishes.
    runner = TrainerRunner(cfg=cfg, base_trainer=base_trainer, work_root=args.work_root)

    seeds = RoundSeeds.derive(args.base_seed, cfg.training)
    gen = ResolvedGenerator(hotkey=args.hotkey, uid=args.uid, ref=args.gen_ref)

    # Resolve the size this worker trains: the primary contract, or the matching
    # [[training.sizes]] entry. Same per-size contract the local trainer uses, so
    # remote and local runs are the identical code path at identical compute.
    contract = cfg.training.primary_size
    if args.arch_preset and args.arch_preset != contract.arch_preset:
        match = next((s for s in cfg.training.extra_sizes if s.arch_preset == args.arch_preset), None)
        if match is None:
            log.error("unknown --arch-preset %r (not in [[training.sizes]])", args.arch_preset)
            return 2
        contract = cfg.training.for_size(match)

    token_budget = (
        contract.tokens_for_hours(args.train_hours)
        if args.train_hours is not None else contract.train_tokens
    )
    try:
        entry = runner.train_one(gen, args.role, seeds, args.block,
                                 contract=contract, token_budget=token_budget,
                                 repo_suffix=args.repo_suffix)
    except Exception as e:  # noqa: BLE001 — report failure on stderr, nonzero exit
        log.exception("worker training failed: %s", e)
        return 1

    log.info("worker done role=%s trained=%s", args.role, entry.trained_pointer)
    # The receipt: the one line the orchestrator parses.
    print(RECEIPT_SENTINEL + json.dumps(dataclasses.asdict(entry), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
