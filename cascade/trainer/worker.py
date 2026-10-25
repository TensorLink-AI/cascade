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
import os
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from .contract import RoundSeeds
from .corpus import CorpusError
from .loop import ResolvedGenerator, TrainerRunner, expert_kw
from .main import _load_trainer
from .remote import RECEIPT_SENTINEL


def _pod_fault(e: CorpusError) -> bool:
    """A pod that cannot isolate the generator is ours, not the miner's (as the CPU tier)."""
    if "sandbox_isolation_unavailable" not in str(e):
        return False
    logging.getLogger("cascade.trainer.worker").error(
        "pod cannot isolate the generator (infra, not a miner rejection): %s", e)
    return True


def anneal_recipe(contract):
    """The bench-anneal recipe (DEC-CA-0030): the same contract with the LR
    schedule swapped to a pure cosine decay — ``warmup_cosine`` with
    ``warmup_fraction = 0`` decays base_lr → 0 across the (reduced) budget,
    which is DEC-CA-0018's deferred decay shape applied post-hoc. Only these
    two recipe fields move; budget, arch, masking, and objective stay the
    leg's own. Never digest-relevant: this contract exists only inside the
    telemetry leg's process.

    Refuses an EMA-armed contract (DEC-CA-0033): with ``ema_decay`` set the
    scored artifact is already finished form, and an anneal leg would bench
    an EMA-of-a-decay-branch — a recipe nobody calibrated (the DEC-CA-0030
    "never both" rule extends to EMA, same as fork-anneal)."""
    if float(getattr(contract, "ema_decay", 0.0) or 0.0):
        raise ValueError(
            "bench-anneal leg refused: [training] ema_decay is armed — EMA is "
            "the finished-form mechanism; never arm bench_anneal_fraction "
            "with it (DEC-CA-0030 / DEC-CA-0033)"
        )
    if float(getattr(contract, "anneal_fraction", 0.0) or 0.0):
        # Fail fast and clearly: the replaced warmup_cosine contract would
        # otherwise survive to Toto2Trainer.train and die mid-leg with the
        # (accurate but misleading) "anneal_fraction requires wsd" error
        # AFTER the fetch + corpus build were paid.
        raise ValueError(
            "bench-anneal leg refused: [training] anneal_fraction is armed — "
            "fork-anneal already cuts finished-form checkpoints; never arm "
            "bench_anneal_fraction with it (DEC-CA-0030 / DEC-CA-0029)"
        )
    return dataclasses.replace(
        contract, lr_schedule="warmup_cosine", warmup_fraction=0.0
    )


def expert_target_label(domain: str | None) -> str:
    """The W&B ``target`` of a lane-B leg from its ``--domain``: the leg label
    (:func:`cascade.shared.expert_manifest.leg_label` — ``energy`` for one
    domain, ``energy.h64`` for a DEC-CA-0059 multi-target leg) as the
    comma-joined target names (``energy,horizon:64``)."""
    import re

    from ..shared.config.v2 import HORIZON_TARGET_PREFIX

    codes = [c for c in str(domain or "").split(".") if c]
    return ",".join(f"{HORIZON_TARGET_PREFIX}{int(c[1:])}" if re.fullmatch(r"h[0-9]+", c)
                    else c for c in codes)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-train-worker",
                                description="Train one generator for one round (remote worker).")
    p.add_argument("--gen-ref", required=True, help="Generator's Hippius Hub ref (repo@digest).")
    p.add_argument("--uid", type=int, required=True, help="Miner UID.")
    p.add_argument("--hotkey", required=True, help="Miner hotkey.")
    p.add_argument("--role", required=True,
                   choices=["king", "challenger", "expert", "finetune", "mkt_finetune"],
                   help="expert = a lane-B domain leg (DEC-CA-0050): needs --domain, "
                        "--selection-json and --warm-start-ref (the king it trains against); "
                        "finetune = a generic lane-C recipe leg (DEC-CA-0056): needs "
                        "--recipe-json and --warm-start-ref (the era king it fine-tunes) — "
                        "not dispatched today (lane C is MarketCast-only, its legs run as "
                        "mkt_finetune); "
                        "mkt_finetune = a MarketCast leg (DEC-CA-0064): --recipe-json "
                        '{"marketcast": <marketcast.json>, "has_generator": bool}, '
                        "--warm-start-ref = [marketcast] base_ref, --local-only")
    p.add_argument("--recipe-json", default=None,
                   help="Role finetune: the lane-C recipe as JSON {target, lr_scale, "
                        "steps_fraction, warmup_fraction, schedule}, re-validated here "
                        "against [lane_c] bounds. Role mkt_finetune: "
                        '{"marketcast": <marketcast.json>, "has_generator": bool, '
                        '"train_time": <dispatch unix time>}.')
    p.add_argument("--domain", default=None, help="Lane-B target domain (role expert).")
    p.add_argument("--selection-json", default=None,
                   help='Lane-B expert rows as JSON {"<moe layer>": [expert ids]} (role expert).')
    p.add_argument("--base-seed", type=int, required=True, help="Round base seed (block-hash int).")
    p.add_argument("--block", type=int, required=True, help="Round block height.")
    p.add_argument("--trainer", required=True, help="BaseTrainer as 'module:Class'.")
    p.add_argument("--arch-preset", default=None,
                   help="Size to train (primary if omitted, or a [[training.sizes]] arch_preset).")
    p.add_argument("--train-hours", type=float, default=None,
                   help="Override the training budget in hours (a reduced-budget leg, e.g. "
                        "--anneal). Omitted ⇒ the size's full [training] budget.")
    p.add_argument("--repo-suffix", default="",
                   help="Suffix on the checkpoint repo (ckpt-r<seed>-<role>-<size><suffix>) so "
                        "parallel runs at one size (one per challenger uid) don't collide.")
    p.add_argument("--warm-start-ref", default=None,
                   help="Cascade warm-start init: the promoted checkpoint's trained_pointer. "
                        "Fetched from the registry (digest-verified) and loaded as the run's "
                        "init weights; a fetch/load failure fails the run — it never falls "
                        "back to random init. Omitted ⇒ random init.")
    p.add_argument("--anneal", action="store_true",
                   help="Bench-anneal leg (telemetry, DEC-CA-0030): resume --warm-start-ref "
                        "(weights + optimizer state) under a pure cosine decay base_lr -> 0 "
                        "across the --train-hours budget, no warmup. A trainer-LOCAL recipe "
                        "override — the [training] contract and every canonical checkpoint "
                        "are untouched; requires --warm-start-ref and --train-hours.")
    p.add_argument("--local-only", action="store_true",
                   help="Credential-free pod mode (DEC-CA-0036): train and leave the "
                        "checkpoint on disk WITHOUT uploading (this box holds no Hub "
                        "credential); the receipt carries local_checkpoint_dir instead of "
                        "a trained_pointer and the orchestrator harvests, verifies, and "
                        "uploads the checkpoint itself.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--contract-block", type=int, default=None,
                   help="Chain block the leg's contract is effective at (the era start; "
                        "DEC-CA-0047 scheduled switch). Omitted = the base contract.")
    p.add_argument("--router-slow", action="store_true",
                   help="DEC-CA-0050 slow router: train the MoE router at [training] "
                        "router_lr_scale and the routing-bias step at "
                        "route_bias_rate_scale (the lane-B routing gate has opened). "
                        "Emitted by the orchestrator only when true.")
    p.add_argument("--corpus-dir", type=Path, default=None,
                   help="DEC-CA-0053 shards leg: the materialised corpus directory "
                        "(corpus.json + parquet shards) the orchestrator relayed to this "
                        "pod. Required when the leg's contract has corpus_mode = 'shards'; "
                        "the pod then reads shards and runs no generator code.")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"))
    p.add_argument("--continuation-root", default=None, help="Qualified optimizer-boundary recovery directory.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    args = _build_parser().parse_args(argv)
    # Logs to stderr so stdout carries only the receipt.
    logging.basicConfig(
        level=args.log_level, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("cascade.trainer.worker")

    cfg = load_chain_config(args.chain_toml)
    if args.role == "mkt_finetune":
        return run_mkt_worker(args, cfg, log)
    base_trainer = _load_trainer(args.trainer)
    # No wallet on a rented box: this worker never signs or publishes.
    runner = TrainerRunner(cfg=cfg, base_trainer=base_trainer, work_root=args.work_root)

    runner.__dict__["_continuation_args"] = args
    seeds = RoundSeeds.derive(args.base_seed, cfg.training)
    gen = ResolvedGenerator(hotkey=args.hotkey, uid=args.uid, ref=args.gen_ref)

    # Resolve the size this worker trains: the primary contract, or the matching
    # [[training.sizes]] entry. Same per-size contract the local trainer uses, so
    # remote and local runs are the identical code path at identical compute.
    training = cfg.training.at_block(args.contract_block)
    contract = training.primary_size
    if args.arch_preset and args.arch_preset != contract.arch_preset:
        match = next((s for s in training.extra_sizes if s.arch_preset == args.arch_preset), None)
        if match is None:
            log.error("unknown --arch-preset %r (not in [[training.sizes]])", args.arch_preset)
            return 2
        contract = training.for_size(match)
    if args.contract_block is not None and contract.budget_denomination != cfg.training.budget_denomination:
        log.info("contract at block %d: budget_denomination=%s (scheduled switch)",
                 args.contract_block, contract.budget_denomination)

    # A budget override (--train-hours) marks a REDUCED-budget leg (today: the
    # bench-anneal leg); full legs never carry it. One flag drives both the
    # wall-clock scaling and the telemetry label, so they never drift.
    # An EXPERT leg (lane B) is never a heat leg: its reduced budget is the
    # [lane_b] leg_hours contract term, resolved here, so its telemetry and
    # checkpoint keys stay the full-leg ones (an explicit --train-hours on an
    # expert leg is honoured as its budget, still not a heat tag).
    is_heat = args.train_hours is not None and args.role not in ("expert", "finetune")
    # W&B lane label (observability only — the receipt role stays "challenger"
    # for both lane-B and lane-C legs, see remote.receipt_role)
    lane_info: dict | None = None
    expert_hours: float | None = None
    expert_selection = None
    role = args.role
    if role == "expert":
        expert_hours = float(args.train_hours if args.train_hours is not None
                             else cfg.lane_b.leg_hours)
        if not (args.domain and args.selection_json and args.warm_start_ref):
            log.error("--role expert requires --domain, --selection-json and --warm-start-ref")
            return 2
        expert_selection = {int(k): [int(x) for x in v]
                            for k, v in json.loads(args.selection_json).items()}
        # An expert leg uploads like a challenger (its own suffixed repo); it
        # never appears in a lane-A manifest -- the expert manifest carries it.
        role = "challenger"
        lane_info = {"lane": "B", "target": expert_target_label(args.domain)}
        if not args.repo_suffix.startswith("-x"):
            args.repo_suffix = "-x" + args.domain + args.repo_suffix
    finetune = False
    if role == "finetune":
        # DEC-CA-0056 lane-C leg: the era king's checkpoint fine-tuned under the
        # miner's recipe for [lane_c] leg_hours (budget × steps_fraction), fresh
        # optimizer, no finished-form branch; uploads like a challenger under
        # its own ``-c<target>-u<uid>`` repo. Never in a lane-A manifest.
        from .finetune import RecipeError, leg_contract, leg_suffix, recipe_from_json

        if not (args.recipe_json and args.warm_start_ref):
            log.error("--role finetune requires --recipe-json and --warm-start-ref")
            return 2
        try:
            recipe = recipe_from_json(args.recipe_json, cfg)
            contract = leg_contract(cfg, contract, recipe)
        except (RecipeError, ValueError) as e:
            log.error("miner submission rejected: lane-C recipe: %s", e)
            return 3
        role, finetune = "challenger", True
        lane_info = {"lane": "C", "target": recipe.target, "recipe": recipe}
        if not args.repo_suffix.startswith("-c"):
            args.repo_suffix = leg_suffix(recipe.target, args.uid) + args.repo_suffix
        log.info("lane-C fine-tune leg: target=%s lr=%.3g (x%s) schedule=%s warmup=%s "
                 "budget=%s point-passes wall<=%ss", recipe.target, contract.base_lr,
                 recipe.lr_scale, contract.lr_schedule, recipe.warmup_fraction,
                 f"{contract.train_tokens:,}", contract.max_train_seconds)
    if args.anneal and not (args.warm_start_ref and is_heat):
        log.error("--anneal requires --warm-start-ref (the checkpoint to finish) "
                  "and --train-hours (the decay budget)")
        return 2
    if is_heat:
        # Reduced-budget leg: scale the token budget AND the hard wall-clock
        # cap, so a stalling generator costs this pod minutes, never a full
        # leg's max_train_seconds (rented hours are billed). The guard knobs
        # travel via the pod's chain.toml [round].
        contract = contract.for_hours(
            args.train_hours,
            guard_factor=cfg.round.heat_guard_factor,
            guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
        )
    elif expert_hours is not None:
        # the lane-B leg budget scales the token budget AND the wall guard
        # exactly like a reduced leg, without the heat label
        contract = contract.for_hours(
            expert_hours,
            guard_factor=cfg.round.heat_guard_factor,
            guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
        )
    if args.anneal:
        contract = anneal_recipe(contract)
    if args.router_slow:
        from dataclasses import replace as _replace

        contract = _replace(contract, router_slow=True)
        log.info("slow router: router LR × %s, routing-bias step × %s",
                 contract.router_lr_scale, contract.route_bias_rate_scale)
    if contract.corpus_mode == "shards":
        if args.corpus_dir is None:
            log.error("corpus_mode='shards' requires --corpus-dir (DEC-CA-0053)")
            return 2
        from .stream import CORPUS_DIR_ENV

        os.environ[CORPUS_DIR_ENV] = str(args.corpus_dir)
        log.info("shards leg: corpus %s (no generator runs on this pod)", args.corpus_dir)
    if args.contract_block is not None and contract.context_length != cfg.training.context_length:
        log.info("contract at block %d: context_length=%d (scheduled switch)",
                 args.contract_block, contract.context_length)
    token_budget = contract.train_tokens
    if not is_heat:
        # Full-budget leg: this pod is the runtime, so the
        # contract's train_image_digest pin (if set) must match the image digest
        # injected at pod launch. Refuse loudly rather than train off-contract.
        from .contract import TrainImageMismatch, assert_train_image

        try:
            assert_train_image(training)
        except TrainImageMismatch as e:
            log.error("%s", e)
            return 2
    try:
        # heat=is_heat ⇒ a reduced-budget leg's S3/wandb telemetry lands at
        # heat-<hotkey>, off the full leg's <role>-<size> key.
        if args.local_only:
            receipt = runner.train_one_local(
                gen, role, seeds, args.block,
                contract=contract, token_budget=token_budget,
                repo_suffix=args.repo_suffix, heat=is_heat,
                warm_start_ref=args.warm_start_ref, **expert_kw(expert_selection),
                **({"finetune": True} if finetune else {}),
                **({"lane_info": lane_info} if lane_info else {}))
            log.info("worker done role=%s local checkpoint=%s (no upload — "
                     "credential-free pod)", args.role, receipt["local_checkpoint_dir"])
            print(RECEIPT_SENTINEL + json.dumps(receipt, sort_keys=True), flush=True)
            return 0
        entry = runner.train_one(gen, role, seeds, args.block,
                                 contract=contract, token_budget=token_budget,
                                 repo_suffix=args.repo_suffix, heat=is_heat,
                                 warm_start_ref=args.warm_start_ref,
                                 **expert_kw(expert_selection),
                                 **({"finetune": True} if finetune else {}),
                                 **({"lane_info": lane_info} if lane_info else {}))
    except CorpusError as e:
        if _pod_fault(e):
            return 1
        # Miner fault (bad submission, private/missing artifact): one line and a
        # distinct exit code, so the orchestrator logs it as a rejection rather
        # than an infra traceback.
        log.error("miner submission rejected: %s", e)
        return 3
    except Exception as e:  # noqa: BLE001 — report failure on stderr, nonzero exit
        log.exception("worker training failed: %s", e)
        return 1

    log.info("worker done role=%s trained=%s", args.role, entry.trained_pointer)
    # The receipt: the one line the orchestrator parses.
    print(RECEIPT_SENTINEL + json.dumps(dataclasses.asdict(entry), sort_keys=True), flush=True)
    return 0


def run_mkt_worker(args, cfg, log, *, post=None, get=None, synthetic_fn=None,
                   base_loader=None) -> int:
    """``--role mkt_finetune`` (DEC-CA-0064): fine-tune the PINNED Timestral
    base from the miner's MarketCast recipe on this credential-free pod and
    print a ``--local-only`` receipt (the orchestrator harvests). rc 2 =
    bad invocation / not armed, 3 = the miner's recipe or data (miner
    fault), 1 = anything else. ``post`` / ``get`` / ``synthetic_fn`` /
    ``base_loader`` are injectable (tests)."""
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from .marketcast_leg import MktLegError, build_training_snapshot, run_mkt_leg

    mc = cfg.marketcast
    if not args.local_only or not args.recipe_json or not args.warm_start_ref:
        log.error("--role mkt_finetune requires --local-only, --recipe-json and "
                  "--warm-start-ref")
        return 2
    if not mc.from_block:
        log.error("--role mkt_finetune: [marketcast] is not armed (from_block = 0)")
        return 2
    if args.warm_start_ref != mc.base_ref:
        log.error("--warm-start-ref %s is not the pinned [marketcast] base_ref %s",
                  args.warm_start_ref, mc.base_ref)
        return 2
    try:
        wrap = json.loads(args.recipe_json)
        # review M12: the trainer-signed dispatch time — the embargo re-checked
        # against it here, every fetched bar clipped to before it below
        train_time = float(wrap.get("train_time", 0.0) or 0.0) or None
        recipe = parse_marketcast_recipe(wrap["marketcast"], mc,
                                         has_generator=bool(wrap.get("has_generator")),
                                         submission_time=train_time)
    except (ValueError, KeyError, TypeError) as e:
        log.error("miner submission rejected: marketcast recipe: %s", e)
        return 3
    work = Path(args.work_root)
    work.mkdir(parents=True, exist_ok=True)
    try:
        if base_loader is None:
            import torch

            from ..eval.timestral_adapter import load_pinned_base

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            base, base_dir = load_pinned_base(mc.base_ref, mc.base_sha256,
                                              cache_dir=work / "_hf", device=dev)
        else:
            base, base_dir = base_loader()
        from ..pool.marketcast import load_snapshot
        from ..pool.marketcast_feeds import http_get_json, http_post_json
        from .continuation import PROTOCOL, restore_marketcast_data, worker_context
        if getattr(args, "continuation_root", None) and mc.continuation_protocol != PROTOCOL:
            raise ValueError("continuation requires an explicitly activated MarketCast contract")
        max_s = float(cfg.lane_c.leg_hours) * 3600.0
        binding = {"backend": "marketcast", "base": mc.base_ref,
            "base_sha256": mc.base_sha256, "recipe": {k: v for k, v in wrap.items() if k != "train_time"},
            "continuation_protocol": mc.continuation_protocol,
            "seed": int(args.base_seed), "gen_ref": args.gen_ref, "hotkey": args.hotkey,
            "max_seconds": max_s}
        if not restore_marketcast_data(args, cfg, binding, work / "_mkt_train"):
            build_training_snapshot(work / "_mkt_train", mc, recipe, post=post or http_post_json,
                                    get=get or http_get_json, not_after=train_time)
        snap = load_snapshot(work / "_mkt_train", verify_signature=False)
        out_dir = work / f"mkt-{args.hotkey[:12]}-{int(args.block)}"
        continuation = worker_context(args, cfg, {**binding, "data": snap.digest})

        def _run(synthetic):
            return run_mkt_leg(recipe=recipe, mc=mc, base_dir=base_dir,
                               base_fc_inner=getattr(base, "inner", base), train_snap=snap,
                               out_dir=out_dir, seed=int(args.base_seed), max_seconds=max_s,
                               synthetic=synthetic,
                               meta_extra={"hotkey": args.hotkey, "gen_ref": args.gen_ref},
                               **({"continuation": continuation} if continuation is not None else {}))

        if recipe.synthetic_fraction > 0:
            if synthetic_fn is not None:
                with synthetic_fn() as it:
                    res = _run(it)
            else:  # pragma: no cover — sandboxed miner code on a pod
                from ..shared.hippius import HubConfig, fetch_from_hub
                from .sandbox import stream_series

                gen_dir = work / "generator"
                fetch_from_hub(args.gen_ref, gen_dir, HubConfig.from_storage(cfg.storage))
                budget = int(recipe.steps) * int(recipe.batch) * int(recipe.context_length)
                with stream_series(gen_dir, int(args.base_seed), cfg.generator, budget,
                                   max_wall_seconds=int(max_s)) as it:
                    res = _run(it)
        else:
            res = _run(None)
    except MktLegError as e:
        log.error("miner submission rejected: marketcast leg: %s", e)
        return 3
    except CorpusError as e:
        if _pod_fault(e):
            return 1
        log.error("miner submission rejected: %s", e)
        return 3
    except Exception as e:  # noqa: BLE001
        log.exception("marketcast leg failed: %s", e)
        return 1
    mkt = {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
           "checkpoint_sha256": res["checkpoint_sha256"], "recipe_digest": res["recipe_digest"],
           "data_digest": res["data_digest"]}
    receipt = {"miner_hotkey": args.hotkey, "miner_uid": int(args.uid), "role": "challenger",
               "gen_ref": args.gen_ref, "corpus_digest": res["data_digest"],
               "train_block": int(args.block), "local_checkpoint_dir": str(out_dir.resolve()),
               "gpu_name": "", "size": "",
               "tensor_digests": {"model.safetensors": res["checkpoint_sha256"]},
               "marketcast": mkt}
    log.info("marketcast leg done: %d step(s), loss %.4g, checkpoint %s", res["steps"],
             res["final_loss"], res["checkpoint_sha256"][:16])
    print(RECEIPT_SENTINEL + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
