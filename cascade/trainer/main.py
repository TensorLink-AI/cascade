"""``cascade-trainer`` console-script — the owner's training service.

The base-model training backend is owner-supplied (the GPU boundary), passed as
``--trainer module:Class`` and instantiated with no args. For real runs the
trainer also needs a wallet (to sign manifests, bench reports and promotion
records) and Hub credentials (to upload harvested checkpoints).

The live service drives the rolling scheduler (DEC-CA-0043) through
:class:`~cascade.trainer.loop.TrainerRunner`. ``--offline`` skips the chain and
prints the contract digest and derived seeds — a config + plumbing smoke check
that needs neither GPU nor network.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
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


# The live service's single-instance lock (``<work-root>/trainer.lock``): two
# trainers on one work root would run parallel legs, rent twice and race every
# state file (the r47 stray-trio incident, OPSLOG 2026-08-28). Held — the fd
# kept open — for the life of the process; the kernel drops it on exit.
_INSTANCE_LOCK = None


def acquire_instance_lock(work_root: Path):
    """Take the exclusive, non-blocking ``flock`` on ``work_root/trainer.lock``
    and return its open file, or raise ``RuntimeError`` naming the holder."""
    import fcntl
    import os

    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / "trainer.lock"
    f = open(path, "a+")  # noqa: SIM115 — held open for the process lifetime
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        f.seek(0)
        holder = f.read().strip() or "?"
        f.close()
        raise RuntimeError(
            f"another cascade-trainer already holds {path} (pid {holder}) — refusing to "
            "start a second instance on the same work root") from e
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


def _configure_logging(level: str, log_file: Path | None, *, max_mb: int, backups: int) -> None:
    """Stderr, or — with ``--log-file`` — a size-rotated file (``<file>.1`` …
    ``<file>.<backups>``), so a long-lived service never grows one log forever."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    if log_file is None:
        logging.basicConfig(level=level, format=fmt)
        return
    from logging.handlers import RotatingFileHandler

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=max(1, int(max_mb)) * 1024 * 1024,
                                  backupCount=max(1, int(backups)), encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-trainer", description="cascade trainer service.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--direct-runtime-config", type=Path, default=None, help="Run the durable direct scheduler from explicit JSON configuration.")
    p.add_argument("--direct-allow-paid", action="store_true", help="Explicit paid provisioning authorization for the direct scheduler.")
    p.add_argument("--direct-simulation", type=Path, default=None,
                   help="Run credential-free direct-MoE counter simulation into a NEW directory.")
    p.add_argument("--trainer", default=None, help="BaseTrainer as 'module:Class'.")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"))
    p.add_argument(
        "--lane-b-bench-hosts", type=Path, default=None,
        help="#909: eval-hosts TOML ([[host]], re-read per bench; e.g. targon-eval.toml, "
             "lane A's bench pod) where lane B's king / candidate / anchor public rows are "
             "benched. Unset or empty = lane B fails closed: the trainer box never benches.")
    p.add_argument("--network", default="finney")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument(
        "--remote-hosts", type=Path, default=None,
        help="Trainer-local TOML of SSH GPU hosts ([[host]] tables; written by the "
             "provisioner when it runs): the pod profile rented pods mirror (first "
             "final/any host; a profile-only entry is enough) and the operator FINAL "
             "lanes rolling fallback legs run on (see "
             "trainer/remote.py). Re-read on every use, so a lane or profile change "
             "after startup is seen without a restart; it may not exist at startup.",
    )
    p.add_argument("--base-seed", type=int, default=0, help="Override round base seed (offline).")
    p.add_argument("--offline", action="store_true", help="No chain/GPU; print contract + seeds.")
    p.add_argument("--preflight-v2-baseline", metavar="IMMUTABLE_REF",
                   help="Fetch and sandbox-verify the initial v2 king; record a local launch qualification. "
                        "No rentals. Exit 0 ready to launch, 3 qualified but still blocked "
                        "(the printed blockers), 2 verification failed.")
    p.add_argument("--preflight-v2-baseline-check", metavar="IMMUTABLE_REF",
                   help="Read-only (dry-run) launch check of the initial v2 king: every blocker the "
                        "first king leg would hit, as JSON. Fetches, writes and rents nothing. "
                        "Exit 0 ready, 3 blocked, 2 error.")
    p.add_argument("--bench-suites", default="gift-eval,boom,time",
                   help="Suites the completion bench runs on each checkpoint's pod. "
                        "Lane A only for a trainer-sourced era: an era under [v2] "
                        "bench_source = \"validator\" (DEC-CA-0063) runs no trainer "
                        "lane-A sweep — the main validator benches instead.")
    p.add_argument("--bench-local-data-dir", default="/root/bench_data",
                   help="Benchmark data dir on THIS box to sideload to the pod (tar "
                        "over ssh) before each sweep, so the sweep never depends on "
                        "the pod's own HF fetch (which remains the fallback). Skipped "
                        "harmlessly when the dir or its suite markers are absent, or "
                        "the pod is already staged. '' disables.")
    # Retired with the boundary-round loop: still accepted (and ignored) so a
    # launch line written for the old service keeps starting.
    for flag, kw in (("--hosts-wait-seconds", {"type": int}),
                     ("--force-rerun-round", {}),
                     ("--post-round-benchmarks", {"action": "store_true"}),
                     ("--bench-max-series", {"type": int}),
                     ("--bench-data-dir", {}),
                     ("--bench-device", {}),
                     ("--bench-interval", {"type": int})):
        p.add_argument(flag, help=argparse.SUPPRESS, **kw)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None,
                   help="Write logs to this file, size-rotated (default: stderr).")
    p.add_argument("--log-max-mb", type=int, default=200,
                   help="Rotate --log-file at this size (MiB).")
    p.add_argument("--log-backups", type=int, default=10,
                   help="Rotated --log-file generations kept.")
    return p


PREFLIGHT_READY, PREFLIGHT_ERROR, PREFLIGHT_BLOCKED = 0, 2, 3


def _preflight_v2_baseline(cfg, args) -> int:
    """``--preflight-v2-baseline`` (qualify, then report) and
    ``--preflight-v2-baseline-check`` (report only). Both print JSON and end
    with the read-only :func:`launch_preflight.preflight_report`, so a
    qualification that leaves another blocker (no validator anchor, an
    unqualified worker release) exits :data:`PREFLIGHT_BLOCKED`, never 0."""
    import json

    from .launch_preflight import preflight_report, qualify

    log = logging.getLogger("cascade.trainer")
    ref = args.preflight_v2_baseline or args.preflight_v2_baseline_check
    out: dict = {}
    if args.preflight_v2_baseline:
        try:
            out["qualification"] = qualify(cfg, ref, args.work_root)
        except Exception as exc:
            log.error("v2 baseline preflight failed: %s", exc)
            return PREFLIGHT_ERROR
    try:
        report = preflight_report(cfg, ref, args.work_root)
    except Exception as exc:
        log.error("v2 baseline preflight check failed: %s", exc)
        return PREFLIGHT_ERROR
    out["preflight"] = report
    print(json.dumps(out, allow_nan=False, sort_keys=True))
    if report["ready"]:
        log.info("v2 launch preflight ready: %s", ref)
        return PREFLIGHT_READY
    log.error("v2 launch preflight blocked: %s (%s)", ref, "; ".join(report["blockers"]))
    return PREFLIGHT_BLOCKED


def main(argv: list[str] | None = None) -> int:
    from .operator_legs_cli import COMMANDS as _OPERATOR_LEG_COMMANDS

    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _OPERATOR_LEG_COMMANDS:
        # Owner commands for operator-billed legs: pure work-root file ops,
        # no env, chain or provider (cascade.trainer.operator_legs_cli).
        from .operator_legs_cli import main as operator_legs_main

        return operator_legs_main(raw)
    from ..shared.codepath import verify_release_code_path
    from ..shared.env import load_env_files
    verify_release_code_path("cascade-trainer")
    # DEC-CA-0067: this process votes on / acts on the v2b vote — a config
    # whose pinned feature is missing or wrong refuses to load
    from ..shared.config.loader import require_vote_pin

    require_vote_pin()
    load_env_files()
    args = _build_parser().parse_args(argv)
    if args.direct_runtime_config is not None:
        from ..direct.__main__ import main as direct_main
        return direct_main(["--config", str(args.direct_runtime_config), "--serve"] + (["--allow-paid"] if args.direct_allow_paid else []))
    if args.direct_simulation is not None:
        import json

        from .direct_simulation import simulate
        print(json.dumps(simulate(args.direct_simulation), indent=2))
        return 0
    _configure_logging(args.log_level, args.log_file, max_mb=args.log_max_mb,
                       backups=args.log_backups)
    cfg = load_chain_config(args.chain_toml)
    if args.preflight_v2_baseline or args.preflight_v2_baseline_check:
        return _preflight_v2_baseline(cfg, args)
    if getattr(getattr(cfg, "generator", None), "sandbox_strict", None) is False:
        logging.getLogger("cascade.trainer").warning(
            "sandbox_strict = false: a host without network namespaces runs the generator as a "
            "plain child that can read the worker environment — set sandbox_strict = true on "
            "production ([generator] sandbox_strict in chain.toml)")

    if args.offline:
        from .contract import compute_base_arch_digest

        seeds = RoundSeeds.derive(args.base_seed, cfg.training)
        thrones = cfg.throne_contracts()
        print(f"base_arch:           {cfg.training.base_arch}")
        print(f"settlement cadence:  1 boundary / {cfg.round.epoch_blocks} blocks "
              f"(~{cfg.round.round_hours:g}h); era = {cfg.round.era_settlements} settlements")
        print(f"throne sizes:        {', '.join(t.arch_preset for t in thrones)}")
        print(f"available sizes:     {', '.join(cfg.training.size_registry)}")
        for sc in cfg.training.all_sizes():
            computed = compute_base_arch_digest(sc)
            flag = "" if sc.base_arch_digest == computed else "  ← MISMATCH, pin this digest"
            print(f"  [{sc.arch_preset}] base_arch_digest: {computed}{flag}")
            print(f"  [{sc.arch_preset}] leg budget:       "
                  f"{cfg.training.target_train_hours:g}h ≈ {sc.train_tokens:,} point-passes")
        print(f"contract_digest:     {contract_digest(cfg.training)}")
        print(f"train_image_digest:  {cfg.training.train_image_digest or '(unpinned)'}")
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
    # The orchestrator holds the trainer wallet and the private eval pool:
    # miner generator code runs here only in a hardened sandbox child.
    from ..shared.sandbox_policy import mark_holds_secrets

    mark_holds_secrets("the trainer wallet and the private eval pool (orchestrator)")

    from ..shared.config import LaunchConfigError, assert_launch_ready

    try:
        assert_launch_ready(cfg, role="trainer")
    except LaunchConfigError as e:
        print(e, flush=True)
        return 2

    global _INSTANCE_LOCK
    try:
        _INSTANCE_LOCK = acquire_instance_lock(args.work_root)
    except RuntimeError as e:
        print(f"FATAL: {e}", flush=True)
        return 2

    from ..shared.activation import ActivationStore
    from ..shared.chain import ChainClient
    from .loop import TrainerRunner

    log = logging.getLogger("cascade.trainer")
    base_trainer = _load_trainer(args.trainer)
    client = ChainClient.from_config(
        cfg, network=args.network,
        wallet_name=args.wallet_name, wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
    )

    remote_hosts = None
    if args.remote_hosts is not None:
        from .remote import RemoteDispatchError, load_hosts

        # Best-effort at startup: the file may not exist yet — the lane pool
        # re-reads it on use.
        try:
            remote_hosts = load_hosts(args.remote_hosts)
            log.info("operator hosts on file: %s", ", ".join(h.name for h in remote_hosts))
        except RemoteDispatchError as e:
            log.warning("remote hosts not ready at startup (%s)", e)

    cascade_bench_plan = None
    promotion = None
    if cfg.scoring.cascade_enabled:
        # Warm-start promotion (DEC-CA-0013): the trainer is the selection
        # authority — its engine tracks the reign, logs benched candidates,
        # fires signed promotions and writes the pointer file.
        from .bench_hook import BenchPlan, bench_suites_for
        from .promotion import TrainerPromotion

        work_root = Path(args.work_root)
        warm_start_path = Path(cfg.validator.warm_start_init_path)
        promotion = TrainerPromotion.load(
            reign_threshold=cfg.scoring.cascade_reign_days,
            k_max=cfg.scoring.cascade_top_k,
            quality_epsilon=cfg.scoring.cascade_quality_epsilon,
            state_path=work_root / "trainer_promotion.json",
            pointer_path=warm_start_path,
            round_cfg=cfg.round,
            error_vectors_path=work_root / "promotion_error_vectors.json",
            scoring_cfg=cfg.scoring,
        )
        log.info("cascade promotion engine enabled: generation=%d members=%d "
                 "(pointer file %s, k_max=%d, epsilon=%.3f)",
                 promotion.generation, len(promotion.members), warm_start_path,
                 cfg.scoring.cascade_top_k, cfg.scoring.cascade_quality_epsilon)
        # Bench at completion: each checkpoint is benched on the pod that
        # holds it (GPU, already at its _train_work path). Rented pods all use
        # the canonical workdir; only data_dir depends on it.
        wd = remote_hosts[0].workdir if remote_hosts else "/root/cascade"
        # A capped battery is minutes; the full battery ([eval]
        # cascade_bench_max_series = 0) gets the bench-hold ceiling.
        timeout_s = (int(cfg.eval.bench_hold_max_hours * 3600)
                     if cfg.eval.cascade_bench_max_series == 0 else 1800)
        cascade_bench_plan = BenchPlan(
            # DEC-CA-0061: FEV joins once the four-suite rule is configured.
            suites=bench_suites_for(cfg, args.bench_suites),
            max_series=cfg.eval.cascade_bench_max_series,
            detached=cfg.round.detached_dispatch,
            poll_seconds=cfg.round.dispatch_poll_seconds,
            reattach_grace_seconds=cfg.round.dispatch_reattach_grace_seconds,
            device="cuda",
            data_dir=f"{wd}/bench_data",
            timeout_seconds=timeout_s,
            local_data_dir=args.bench_local_data_dir or None,
        )
        log.info("cascade completion bench enabled (device=cuda, max_series=%s, timeout=%ss)",
                 cfg.eval.cascade_bench_max_series, timeout_s)

    window_source = load_window_source(cfg, cache_dir=args.work_root)
    # DEC-CA-0057: the owner-signed network config, resolved read-only (the
    # trainer never signals); inert while [netconfig] owner_hotkey is empty.
    netconfig = None
    if cfg.netconfig.enabled:
        from ..shared.hippius import open_manifest_store
        from ..shared.netconfig import NetConfigResolver

        netconfig = NetConfigResolver(
            cfg, state_path=Path(args.work_root) / "netconfig_state.json",
            store=lambda: open_manifest_store(cfg.storage), signal=False)
    runner = TrainerRunner(
        cfg=cfg,
        base_trainer=base_trainer,
        work_root=args.work_root,
        # DEC-CA-0045: the rollover the validators decided, persisted beside
        # the trainer's other state; the first loop tick resolves it.
        activation_store=ActivationStore(Path(args.work_root) / "activation_state.json"),
        netconfig=netconfig,
        wallet=client.wallet(),
        remote_hosts=remote_hosts,
        remote_hosts_path=args.remote_hosts,
        # The DEPLOYED chain.toml, pushed onto every pod the trainer rents:
        # the image bakes a pre-release snapshot whose train_image_digest pin
        # is stale.
        deployed_chain_toml=(args.chain_toml or Path("chain.toml")),
        trainer_spec=args.trainer,
        detached_dispatch=cfg.round.detached_dispatch,
        dispatch_poll_seconds=cfg.round.dispatch_poll_seconds,
        dispatch_reattach_grace_seconds=cfg.round.dispatch_reattach_grace_seconds,
        pool_provenance_fn=_build_pool_provenance_fn(window_source),
        cascade_bench_plan=cascade_bench_plan,
        promotion=promotion,
    )
    if cfg.v2.from_block > 0:
        # DEC-CA-0050: the domain-expert lane rides the rolling scheduler,
        # which picks the driver up off the runner. Track v2 lineage tokens
        # before B launches too; tick/settle refuse legs until its own block.
        from .lane_b_driver import LaneBDriver, RunnerLaneBOps

        runner.lane_b_driver = LaneBDriver(
            cfg, RunnerLaneBOps(runner, client, window_source, device=_local_device(),
                                bench=_lane_b_bench(args, cfg, runner),
                                offload=_lane_b_offload(args)),
            Path(args.work_root) / "lane_b_driver.json")
    if int(getattr(getattr(cfg, "lane_c", None), "from_block", 0) or 0) > 0:
        # DEC-CA-0056: the fine-tune lane rides the rolling scheduler the same
        # way (inert until [lane_c] from_block is typed).
        from .lane_c_driver import LaneCDriver, RunnerLaneCOps

        runner.lane_c_driver = LaneCDriver(
            cfg, RunnerLaneCOps(runner, client, device=_local_device()),
            Path(args.work_root) / "lane_c_driver.json")
    log.info("trainer up: netuid=%s manifest_bucket=%s registry=%s throne=%s",
             cfg.netuid, cfg.storage.manifest_bucket, cfg.storage.hub_registry_url,
             ",".join(t.arch_preset for t in cfg.throne_contracts()))
    # bittensor's logging machine silences all other loggers on import; restore
    # cascade.* levels so the run-loop's progress logs stay visible.
    from ..shared.logging_util import restore_cascade_logging

    restore_cascade_logging(args.log_level)
    # A stop must not leave cgen VMs billing until the next start's reaper.
    from .runner_generation import install_generation_shutdown

    install_generation_shutdown(runner)
    runner.run_forever(client)
    return 0


def _lane_b_bench(args, cfg, runner):
    """#909: the eval-pod lane-B bench, or None (no --lane-b-bench-hosts:
    lane B fails closed — the trainer box never benches)."""
    from .lane_b_bench import LaneBBench

    log = logging.getLogger("cascade.trainer")
    path = getattr(args, "lane_b_bench_hosts", None)
    if path is None:
        if cfg.lane_b.require_public:
            log.error("lane B: no --lane-b-bench-hosts — lane-B settlements fail closed "
                      "(public rows are benched only on an eval pod, never on this box)")
        return None
    log.info("lane B: public rows benched on the eval pod(s) in %s", path)
    return LaneBBench(hosts_path=path, cache_dir=Path(args.work_root) / "lane_b_bench_rows",
                      eval_cfg=cfg.eval, fetch=runner._fetch_checkpoint_dir)


def _lane_b_offload(args):
    """Lane B model work (routing probe, lineage read) on the same eval pod(s)
    as the lane-B bench; None ⇒ refused (never run on the trainer box)."""
    from .lane_b_offload import LaneBPodRunner

    path = getattr(args, "lane_b_bench_hosts", None)
    if path is None:
        logging.getLogger("cascade.trainer").error(
            "lane B: no --lane-b-bench-hosts — routing probes are refused (model work runs "
            "only on an eval pod, never on this box)")
        return None
    return LaneBPodRunner.from_hosts_path(path)


def _local_device() -> str:
    """The trainer box's own accelerator for local evals (cpu without one)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — torch missing/broken ⇒ cpu
        return "cpu"


def load_window_source(cfg, *, cache_dir: Path | None):
    """The private eval pool the validators judge on (owner-controlled). The
    pool stack is imported lazily so the offline smoke and unit tests never
    pull it."""
    if cfg.storage.pool_bucket:
        from ..validator.pool import load_bucket_pool

        return load_bucket_pool(cfg, cache_dir=cache_dir)
    from ..validator.pool import load_pool

    return load_pool(cfg, cache_dir=cache_dir)


def _build_pool_provenance_fn(window_source):
    """``(base_seed, block) -> (key, sha256)`` of the private eval-pool
    snapshot validators judge a settlement on — stamped (signed) into the
    manifest so every validator verifies its own snapshot selection against
    it (see docs/EVAL_POOL.md)."""

    def pool_provenance(base_seed: int, block: int | None = None) -> tuple[str, str]:
        key, sha = window_source.provenance_for_round(base_seed, block=block)
        return (str(key or ""), str(sha or ""))

    return pool_provenance


if __name__ == "__main__":
    raise SystemExit(main())
