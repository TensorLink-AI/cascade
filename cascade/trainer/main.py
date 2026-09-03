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

from ..shared.config import effective_epoch_blocks, load_chain_config
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
    p.add_argument(
        "--force-rerun-round", default=None, metavar="ROUND_ID",
        help="Approve-tier operator escape hatch: re-run this ONE round_id even though "
             "its manifest is already published, bypassing the restart re-entry guard "
             "(which otherwise skips a finished round after a trainer restart — the "
             "re-entries of 2026-07-23/24 re-published a stale manifest and camped the "
             "next round's final pods). Re-publishing overwrites round-<id>.json AND "
             "latest.json, so use only for a deliberate same-round re-train; remove "
             "the flag after the re-run.",
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
    p.add_argument("--bench-local-data-dir", default="/root/bench_data",
                   help="Benchmark data dir on THIS box to sideload to the pod (tar "
                        "over ssh, ~75s for the 4.4G battery) before each sweep, so "
                        "the sweep never depends on the pod's own HF fetch (which "
                        "cost r45-r48 their reports; the on-pod download remains the "
                        "fallback). Skipped harmlessly when the dir or its suite "
                        "markers are absent, or the pod is already staged. '' disables.")
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
        import os
        import sys

        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network)
        print(json.dumps(_plan_payload(cfg, client, args.work_root), sort_keys=True))
        # os._exit, not return: the dedup screen's abandoned deadline worker is
        # a non-daemon executor thread, and the atexit join would hold this
        # process open past the provisioner's 600s subprocess timeout (r44
        # 2026-08-26: five consecutive plan timeouts from finished plans that
        # could not exit).
        sys.stdout.flush()
        os._exit(0)

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
    screen_fn, runoff_fn, pool_provenance_fn, composition_fn = _build_screen_fn(
        cfg, cache_dir=args.work_root
    )

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
                local_data_dir=args.bench_local_data_dir or None,
            )

    # Cascade: score BOTH duel checkpoints on GIFT-Eval/BOOM/TIME after each
    # manifest publishes and ship the numbers in the round's separate signed
    # bench report (cascade.shared.bench_report) so validators promote off one
    # authoritative set. Only wired when [scoring] cascade_enabled.
    bench_eval_fn = None
    cascade_bench_plan = None
    warm_start_path = None
    promotion = None
    if cfg.scoring.cascade_enabled:
        # Warm-start promotion + consumption (DEC-CA-0005/0012): the trainer is
        # the selection authority — its promotion engine tracks the reign, logs
        # benched duel candidates, fires promotions (signed PromotionRecord),
        # and writes the pointer file every round then trains from (stamping
        # the pin, signed, on the manifest for the fleet's envelope gate).
        from pathlib import Path as _Path

        from .promotion import TrainerPromotion

        warm_start_path = _Path(cfg.validator.warm_start_init_path)
        promotion = TrainerPromotion.load(
            reign_threshold=cfg.scoring.cascade_reign_days,
            k_max=cfg.scoring.cascade_top_k,
            quality_epsilon=cfg.scoring.cascade_quality_epsilon,
            state_path=_Path(args.work_root) / "trainer_promotion.json",
            pointer_path=warm_start_path,
            round_cfg=cfg.round,
            error_vectors_path=_Path(args.work_root) / "promotion_error_vectors.json",
        )
        log.info("cascade promotion engine enabled: generation=%d members=%d "
                 "(pointer file %s, k_max=%d, epsilon=%.3f)",
                 promotion.generation, len(promotion.members), warm_start_path,
                 cfg.scoring.cascade_top_k, cfg.scoring.cascade_quality_epsilon)
        if remote_hosts or args.remote_hosts:
            # Preferred: bench each duel checkpoint on the pod that just trained
            # it — GPU, and the checkpoint is already at its _train_work path.
            # Reuses the post-round-benchmark remote path; the numbers land in
            # the round's signed bench report via
            # TrainerRunner.run_post_publish_bench (strictly after publish).
            #
            # Keyed on the hosts PATH, not the hosts present right now: the live
            # service restarts in the between-rounds idle window, when the
            # elastic fleet is torn down and hosts.toml is EMPTY — the loop
            # re-reads it every round, but a plan wired off startup contents
            # would silently latch the local fallback for the process lifetime
            # (2026-08-05: a full round benched "skipped" on the GPU-less box).
            from .bench_hook import BenchPlan

            # Elastic pods all publish the canonical workdir; fall back to it
            # when the fleet is empty at startup (provisioner default,
            # remote.py). A static heterogeneous-workdir fleet still resolves
            # checkpoint paths per host at bench time — only data_dir uses this.
            wd = remote_hosts[0].workdir if remote_hosts else "/root/cascade"
            # Per-sweep guard: a capped battery is ~minutes on an L40; the full
            # battery ([eval] cascade_bench_max_series = 0) gets the same
            # ceiling as the provisioner's teardown hold — past that the pod is
            # reaped anyway, so a longer timeout only leaks a dead SSH wait.
            timeout_s = (int(cfg.eval.bench_hold_max_hours * 3600)
                         if cfg.eval.cascade_bench_max_series == 0 else 1800)
            cascade_bench_plan = BenchPlan(
                suites=args.bench_suites,
                max_series=cfg.eval.cascade_bench_max_series,
                device="cuda",
                data_dir=f"{wd}/bench_data",
                timeout_seconds=timeout_s,
                local_data_dir=args.bench_local_data_dir or None,
            )
            log.info("cascade duel bench enabled on the final pods (device=cuda, "
                     "max_series=%s, timeout=%ss)",
                     cfg.eval.cascade_bench_max_series, timeout_s)
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
            log.info("cascade duel bench enabled on device=%s (local, no remote host)",
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
        runoff_fn=runoff_fn,
        pool_provenance_fn=pool_provenance_fn,
        composition_fn=composition_fn,
        bench_plan=bench_plan,
        bench_eval_fn=bench_eval_fn,
        cascade_bench_plan=cascade_bench_plan,
        # Live service: report the real round stage (status/round.json) so the
        # dashboards show heat/duel/validation from the trainer, not a
        # wall-clock estimate. Off by default for offline runs and tests.
        publish_stage_status=True,
        warm_start_path=warm_start_path,
        promotion=promotion,
        force_rerun_round=args.force_rerun_round,
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
    king from challengers, dedup (both the same-ref tier and, as
    ``screened_challengers``, the content screen), drop burned hotkeys — so the
    count matches what the heat will actually train, not raw commitments. Every reveal on
    chain now lands strictly before the NEXT boundary, so no cutoff is applied;
    with timed reveals the field only settles once reveals land
    (~reveal_margin_blocks before the boundary).
    """
    from .loop import TrainerRunner, plan_round, resolve_commitments

    block = int(client.current_block())
    epoch_blocks = effective_epoch_blocks(cfg.round, block)
    next_boundary = (block // epoch_blocks + 1) * epoch_blocks
    resolved = resolve_commitments(client.poll_commitments(include_history=True),
                                   floor_block=cfg.round.commit_floor_block)
    # Same king resolution as the live round (loop.py run_forever): without
    # genesis_ref the vacant-throne fallback seats the lowest UID as interim
    # king, which both misreports "king" and undercounts challengers by one.
    plan = plan_round(resolved, client.highest_incentive_hotkey(),
                      genesis_ref=cfg.round.genesis_generator_ref or None)
    probe = TrainerRunner(cfg=cfg, base_trainer=None, work_root=Path(work_root))
    eligible = probe._filter_burned_challengers(plan.challengers)
    # The heat fleet must be sized off the field that will actually train. With
    # [round] dedup_mode = "enforce" the content screen drops a real slice of
    # the field (~29% on the validated round), and a fleet sized before it just
    # rents pods that idle. Static tiers only: fingerprints are fetch+hash, so
    # this stays a read-only sizing path that never executes generator code.
    # Budgeted well under the caller's 600s subprocess timeout (see
    # provision.main.make_plan_fn): a screen that overruns would take the whole
    # plan with it, and a plan that fails rents NO fleet — trading a spam
    # screen for a lost rental window is never the right side of that.
    screened = probe._screen_duplicate_entrants(
        plan.king, eligible, next_boundary, static_only=True, report=False,
        budget_seconds=cfg.round.dedup_plan_seconds)
    return {
        "block": block,
        "epoch_blocks": epoch_blocks,
        "next_boundary_block": next_boundary,
        "blocks_to_boundary": next_boundary - block,
        "king": plan.king.hotkey if plan.king is not None else None,
        "resolved": len(resolved),
        "challengers": len(plan.challengers),
        "eligible_challengers": len(eligible),
        "screened_challengers": len(screened),
        "heat_train_hours": cfg.round.heat_train_hours,
        "finalists": cfg.round.finalists,
        # DEC-CA-0012: the provisioner's final fleet and budget breaker must
        # cover the WORST case the tie-aware advance rule can produce, not the
        # single finalist the plan predicts (JIT rental adapts off the
        # heat_complete marker's actual list either way).
        "max_finalists": cfg.round.max_finalists,
    }


def _build_screen_fn(cfg, *, cache_dir: Path | None):
    """The heat screener plus the eval-pool pin, off one shared pool source.

    Returns ``(screen_fn, pool_provenance_fn, composition_fn)``: the screener
    ranks heat checkpoints on the held-out pool, the provenance hook reports
    the ``(key, sha256)`` of the snapshot a round screens on so the runner can
    stamp it — signed — into the manifest (validators then verify their own
    snapshot selection against it; see docs/EVAL_POOL.md), and the composition
    hook reports the realised jittered-mix breakdown of the round's verdict
    draw for the unsigned manifest block (None while the mix is inactive).

    Loads the same private eval pool the validators use (owner-controlled) and
    scores each heat checkpoint on a per-round-rotated slice, returning the
    per-window scores so the trainer can rank the field down to ``[round]
    finalists`` before the expensive final. The runner reduces them with
    ``global_geomean`` (lower is better) for the ranking, reports their
    ``global_components`` (raw CRPS/MASE) on the heat standings, and additionally
    resamples them for the shadow selection diagnostics — every entrant is
    scored on the SAME slice, which is what makes that bootstrap paired.
    ``block`` (the round's epoch boundary) keys a daily-snapshot pool to the same
    snapshot the validator will judge on. Imports torch/pool lazily so the
    offline smoke and unit tests never pull the heavy stacks."""
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

    def screen(ckpt_dir: Path, gen, base_seed: int, block: int | None = None):
        windows = window_source.windows_for_round(base_seed, n, block=block)
        # Return the per-window scores: the runner ranks on global_geomean,
        # publishes global_components (raw CRPS/MASE), and resamples them for
        # the shadow selection diagnostics.
        return evaluate_checkpoint(
            ckpt_dir, windows, num_samples=num_samples, device="cpu",
            contract=cfg.screen_contract(),
        )

    def runoff(ckpt_dir: Path, gen, base_seed: int, block: int | None = None,
               start: int = 0, stop: int = 0):
        # Tie run-off (DEC-CA-0012): the INCREMENTAL windows [start, stop) of
        # the round's seeded permutation. windows_for_round is a permutation
        # PREFIX, so the heat's slice is windows[:start] of this very list —
        # scoring only the tail keeps the concatenation paired by construction.
        windows = window_source.windows_for_round(base_seed, stop, block=block)[start:]
        return evaluate_checkpoint(
            ckpt_dir, windows, num_samples=num_samples, device="cpu",
            contract=cfg.screen_contract(),
        )

    def pool_provenance(base_seed: int, block: int | None = None) -> tuple[str, str]:
        key, sha = window_source.provenance_for_round(base_seed, block=block)
        return (str(key or ""), str(sha or ""))

    def composition(base_seed: int, block: int | None = None) -> dict | None:
        """Realised composition of the round's VERDICT draw (n_windows, not the
        heat's smaller slice) — None while the jittered mix is inactive, so
        pre-activation manifests carry no new field."""
        from ..validator.pool import mix_params_from_config
        from ..validator.windows import round_composition

        mix = mix_params_from_config(cfg)
        if mix is None or not mix.active(block):
            return None
        windows = window_source.windows_for_round(
            base_seed, cfg.eval.n_windows, block=block
        )
        return round_composition(windows, mix)

    return screen, runoff, pool_provenance, composition


if __name__ == "__main__":
    raise SystemExit(main())
