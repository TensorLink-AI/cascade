"""``cascade-trainer`` console-script — the owner's training service.

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
    p = argparse.ArgumentParser(prog="cascade-trainer", description="cascade trainer service.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--trainer", default=None, help="BaseTrainer as 'module:Class'.")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"))
    p.add_argument("--network", default="finney")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument(
        "--remote-hosts", type=Path, default=None,
        help="Trainer-local TOML of SSH GPU pods ([[host]] tables). When set, king "
             "and challenger train in parallel on separate pods (see trainer/remote.py). "
             "RE-READ at the start of every round, so an elastic provisioner can "
             "rewrite it (rent pods per round, tear down after) without a restart; "
             "missing/empty at round start ⇒ that round trains locally.",
    )
    p.add_argument(
        "--hosts-wait-seconds", type=int, default=0,
        help="At round start, wait up to this long for --remote-hosts to appear/fill "
             "before falling back to local training. With timed reveals the field is "
             "only countable ~reveal_margin_blocks before the boundary, so per-round "
             "pods finish booting after the round starts; size this to pod boot + "
             "image pull (e.g. 900).",
    )
    p.add_argument(
        "--plan-only", action="store_true",
        help="Print the upcoming round's eligible field as JSON and exit (no wallet, "
             "no trainer, no GPU) — the input the pod provisioner sizes the fleet "
             "off. Counts only settle once timed reveals have landed (reveals target "
             "boundary − reveal_margin_blocks), so run this at/after the reveal margin.",
    )
    p.add_argument("--base-seed", type=int, default=0, help="Override round base seed (offline).")
    p.add_argument("--offline", action="store_true", help="No chain/GPU; print contract + seeds.")
    p.add_argument(
        "--post-round-benchmarks", action="store_true",
        help="After each round's manifest publishes, benchmark the round's KING "
             "checkpoint (GIFT-Eval/BOOM/TIME) on the idle GPU pod. LOG-ONLY "
             "telemetry — validators still score exclusively on the private eval "
             "pool; this never feeds weights or the throne. Requires --remote-hosts "
             "and pinned benchmark data on the pod (see benchmarks/README).",
    )
    p.add_argument("--bench-suites", default="gift-eval,boom,time",
                   help="Suites for --post-round-benchmarks (size to round cadence: "
                        "the full battery is ~1h on a 4090).")
    p.add_argument("--bench-max-series", type=int, default=0,
                   help="Cap datasets per suite for --post-round-benchmarks (0 = full).")
    p.add_argument("--bench-data-dir", default="/root/bench_data",
                   help="Benchmark data dir on the pod.")
    p.add_argument("--bench-device", default="auto",
                   help="Device for the Cascade king bench eval ([scoring] cascade_enabled). "
                        "'auto' (default) uses the trainer's GPU when one is present, else cpu — "
                        "the eval runs after training finishes, on the same now-idle GPU. Force "
                        "with 'cuda'/'cpu'. The full GIFT-Eval + BOOM + TIME battery is only "
                        "practical on a GPU (BOOM full ≈ 26 min on an RTX 5090).")
    p.add_argument("--bench-interval", type=int, default=0,
                   help="Minimum seconds between benchmark launches (0 = every round). "
                        "Set this above the sweep duration when rounds are tighter than "
                        "the sweep — telemetry then samples every Nth king instead of "
                        "being preempted by every round's training.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_chain_config(args.chain_toml)

    if args.offline:
        from .contract import compute_base_arch_digest

        seeds = RoundSeeds.derive(args.base_seed, cfg.training)
        screen = cfg.screen_contract()
        thrones = cfg.throne_contracts()
        print(f"base_arch:           {cfg.training.base_arch}")
        print(f"round cadence:       1 round / {cfg.round.epoch_blocks} blocks "
              f"(~{cfg.round.round_hours:g}h); finalists {cfg.round.finalists}")
        print(f"screen size:         {screen.arch_preset} "
              f"(heat {cfg.round.heat_train_hours:g}h ≈ "
              f"{screen.tokens_for_hours(cfg.round.heat_train_hours):,} point-passes)")
        print(f"throne sizes:        {', '.join(t.arch_preset for t in thrones)}")
        print(f"available sizes:     {', '.join(cfg.training.size_registry)}")
        for sc in cfg.training.all_sizes():
            computed = compute_base_arch_digest(sc)
            flag = "" if sc.base_arch_digest == computed else "  ← MISMATCH, pin this digest"
            print(f"  [{sc.arch_preset}] base_arch_digest: {computed}{flag}")
            print(f"  [{sc.arch_preset}] final budget:     "
                  f"{cfg.training.target_train_hours:g}h ≈ {sc.train_tokens:,} point-passes")
        print(f"contract_digest:     {contract_digest(cfg.training)}")
        print(f"train_image_digest:  {cfg.training.train_image_digest or '(unpinned)'}")
        print(f"generation_seed:     {seeds.generation_seed}")
        print(f"training_seed:       {seeds.training_seed}")
        print("offline trainer smoke complete")
        return 0

    if args.plan_only:
        import json

        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network)
        print(json.dumps(_plan_payload(cfg, client, args.work_root), sort_keys=True))
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
        from .remote import RemoteDispatchError, load_hosts

        # Best-effort at startup: with an elastic per-round provisioner the file
        # may not exist yet — run_forever re-reads it at every round start.
        try:
            remote_hosts = load_hosts(args.remote_hosts)
            logging.getLogger("cascade.trainer").info(
                "remote training across %d pod(s): %s",
                len(remote_hosts), ", ".join(h.name for h in remote_hosts),
            )
        except RemoteDispatchError as e:
            logging.getLogger("cascade.trainer").warning(
                "remote hosts not ready at startup (%s); re-checking each round", e,
            )

    log = logging.getLogger("cascade.trainer")
    screen_fn, pool_provenance_fn = _build_screen_fn(cfg, cache_dir=args.work_root)

    bench_plan = None
    if args.post_round_benchmarks:
        # Key off the FLAG, not the startup-loaded list: with an elastic fleet
        # the hosts file may be empty until a round's provisioner fills it, and
        # run_forever guards each launch on the round's live fleet anyway.
        if args.remote_hosts is None:
            log.warning("--post-round-benchmarks needs --remote-hosts; disabling")
        else:
            from .bench_hook import BenchPlan

            bench_plan = BenchPlan(
                suites=args.bench_suites,
                max_series=args.bench_max_series,
                data_dir=args.bench_data_dir,
                min_interval_seconds=args.bench_interval,
            )

    # Cascade: score the king's checkpoint on GIFT-Eval/BOOM/TIME and stamp the
    # numbers onto its signed manifest entry so validators promote off one
    # authoritative set. Only wired when [scoring] cascade_enabled.
    bench_eval_fn = None
    cascade_bench_plan = None
    if cfg.scoring.cascade_enabled:
        if remote_hosts:
            # Preferred: bench the king on the pod that just trained it — GPU, and
            # the checkpoint is already at its _train_work path. Reuses the
            # post-round-benchmark remote path; the six numbers still land on the
            # signed manifest via TrainerRunner._stamp_king_bench_scores.
            from .bench_hook import BenchPlan

            wd = remote_hosts[0].workdir
            cascade_bench_plan = BenchPlan(
                suites=args.bench_suites,
                max_series=cfg.eval.cascade_bench_max_series,
                device="cuda",
                data_dir=f"{wd}/bench_data",
                timeout_seconds=1800,  # guard: a capped battery is ~minutes on an L40
            )
            log.info("cascade king bench eval enabled on worker %s (device=cuda, max_series=%s)",
                     remote_hosts[0].name, cfg.eval.cascade_bench_max_series)
        else:
            from .loop import make_bench_eval_fn

            bench_device = args.bench_device
            if bench_device == "auto":
                # Reuse the trainer's GPU when present — the eval runs after training,
                # so that GPU is idle. Falls back to cpu on a GPU-less box.
                try:
                    import torch

                    bench_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:  # noqa: BLE001 — torch missing/broken ⇒ cpu
                    bench_device = "cpu"
            log.info("cascade king bench eval enabled on device=%s (local, no remote host)",
                     bench_device)
            bench_eval_fn = make_bench_eval_fn(cfg, device=bench_device)

    runner = TrainerRunner(
        cfg=cfg,
        base_trainer=base_trainer,
        work_root=args.work_root,
        wallet=client.wallet(),
        remote_hosts=remote_hosts,
        remote_hosts_path=args.remote_hosts,
        hosts_wait_seconds=args.hosts_wait_seconds,
        trainer_spec=args.trainer,
        screen_fn=screen_fn,
        pool_provenance_fn=pool_provenance_fn,
        bench_plan=bench_plan,
        bench_eval_fn=bench_eval_fn,
        cascade_bench_plan=cascade_bench_plan,
    )
    log.info(
        "trainer up: netuid=%s manifest_bucket=%s registry=%s mode=%s screen=%s throne=%s",
        cfg.netuid, cfg.storage.manifest_bucket, cfg.storage.hub_registry_url,
        "remote" if remote_hosts else "local",
        cfg.screen_contract().arch_preset,
        ",".join(t.arch_preset for t in cfg.throne_contracts()),
    )
    # bittensor's logging machine silences all other loggers on import; restore
    # cascade.* levels so the run-loop's progress logs stay visible.
    from ..shared.logging_util import restore_cascade_logging

    restore_cascade_logging(args.log_level)
    runner.run_forever(client)
    return 0


def _plan_payload(cfg, client, work_root: Path | str) -> dict:
    """The upcoming round's eligible field, as one JSON-able dict.

    This is the provisioner's sizing input (``--plan-only``): the same
    eligibility pipeline the round itself will run — resolve reveals, split
    king from challengers, dedup, drop burned hotkeys — so the count matches
    what the heat will actually train, not raw commitments. Every reveal on
    chain now lands strictly before the NEXT boundary, so no cutoff is applied;
    with timed reveals the field only settles once reveals land
    (~reveal_margin_blocks before the boundary).
    """
    from .loop import TrainerRunner, plan_round, resolve_commitments

    block = int(client.current_block())
    epoch_blocks = max(1, cfg.round.epoch_blocks)
    next_boundary = (block // epoch_blocks + 1) * epoch_blocks
    resolved = resolve_commitments(client.poll_commitments())
    plan = plan_round(resolved, client.highest_incentive_hotkey())
    probe = TrainerRunner(cfg=cfg, base_trainer=None, work_root=Path(work_root))
    eligible = probe._filter_burned_challengers(plan.challengers)
    return {
        "block": block,
        "epoch_blocks": epoch_blocks,
        "next_boundary_block": next_boundary,
        "blocks_to_boundary": next_boundary - block,
        "king": plan.king.hotkey if plan.king is not None else None,
        "resolved": len(resolved),
        "challengers": len(plan.challengers),
        "eligible_challengers": len(eligible),
        "heat_train_hours": cfg.round.heat_train_hours,
        "finalists": cfg.round.finalists,
    }


def _build_screen_fn(cfg, *, cache_dir: Path | None):
    """The heat screener plus the eval-pool pin, off one shared pool source.

    Returns ``(screen_fn, pool_provenance_fn)``: the screener ranks heat
    checkpoints on the held-out pool, and the provenance hook reports the
    ``(key, sha256)`` of the snapshot a round screens on so the runner can
    stamp it — signed — into the manifest (validators then verify their own
    snapshot selection against it; see docs/EVAL_POOL.md).

    Loads the same private eval pool the validators use (owner-controlled) and
    scores each heat checkpoint on a per-round-rotated slice, returning
    geomean(CRPS, MASE) (lower is better) so the trainer can rank the field down
    to ``[round] finalists`` before the expensive final. ``block`` (the round's
    epoch boundary) keys a daily-snapshot pool to the same snapshot the
    validator will judge on. Imports torch/pool lazily so the offline smoke and
    unit tests never pull the heavy stacks."""
    from ..eval.scoring import global_geomean
    from ..validator.evaluator import evaluate_checkpoint

    if cfg.storage.pool_bucket:
        from ..validator.pool import load_bucket_pool

        window_source = load_bucket_pool(cfg, cache_dir=cache_dir)
    else:
        from ..validator.pool import load_pool

        window_source = load_pool(cfg, cache_dir=cache_dir)

    n = min(cfg.round.heat_n_windows, cfg.eval.n_windows)
    # The heat only RANKS the field; far fewer samples than the final verdict
    # keeps the sequential CPU screening from rivalling the heat training time.
    num_samples = cfg.round.heat_num_samples or cfg.eval.num_samples

    def screen(ckpt_dir: Path, gen, base_seed: int, block: int | None = None) -> float:
        windows = window_source.windows_for_round(base_seed, n, block=block)
        scores = evaluate_checkpoint(
            ckpt_dir, windows, num_samples=num_samples, device="cpu"
        )
        return global_geomean(scores)

    def pool_provenance(base_seed: int, block: int | None = None) -> tuple[str, str]:
        key, sha = window_source.provenance_for_round(base_seed, block=block)
        return (str(key or ""), str(sha or ""))

    return screen, pool_provenance


if __name__ == "__main__":
    raise SystemExit(main())
