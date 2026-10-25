"""Local, directional B/C experiments using the production train and verdict paths.

These never call intake, chain writes or provisioners. Data and compute are local.
Private gates, other pending merges and subsequent confirmations remain live-only.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .live_status import LiveStateUnavailable, positive_seconds
from .output import NETWORK, OK, REJECTED, USAGE, reporter


def add_lane_test(sub, lane):
    p = sub.add_parser("test", help=f"Train and evaluate a local lane-{lane.upper()} entry; no submission or rental.")
    p.add_argument("repo_dir", type=Path)
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-hours", type=positive_seconds, default=0.01,
                   help="Local training budget (default 0.01h); does not alter live admission budgets.")
    p.add_argument("--work-dir", type=Path, default=Path(".cascade-work") / f"lane-{lane}")
    if lane == "b":
        p.add_argument("--targets", required=True)
        p.add_argument("--base", required=True, help="Local MoE checkpoint directory, immutable pointer, or live (current king).")
        p.add_argument("--pool-dir", type=Path, required=True, help="Real held-out series with metadata.json domain/source labels.")
        p.add_argument("--at-block", type=int, default=None)
        p.add_argument("--public-project", type=Path, default=None,
                       help="Run the production public benchmark sidecar from this project; otherwise public guards stay unverified.")
        p.set_defaults(func=cmd_lane_b_test, command="lane-b test")
    else:
        p.add_argument("--target", required=True)
        p.add_argument("--base-dir", type=Path, required=True, help="Already downloaded pinned Timestral base; digest verified before loading.")
        p.add_argument("--train-snapshot", type=Path, required=True)
        p.add_argument("--eval-snapshot", type=Path, required=True)
        p.set_defaults(func=cmd_lane_c_test, command="lane-c test")


def cmd_lane_b_test(args):
    from ..shared.config import load_chain_config

    out = reporter(args)
    try:
        data = test_lane_b(args, load_chain_config(args.chain_toml))
    except LiveStateUnavailable as exc:
        return out.fail("live_status_unavailable", str(exc), NETWORK)
    except ImportError as exc:
        return out.fail("missing_train_extra", f"Install .[train,hippius] for local lane-B tests: {exc}", USAGE)
    except (ValueError, OSError) as exc:
        return out.fail("lane_b_test_failed", str(exc), REJECTED)
    out.result(**data)
    out.info("Local lane-B experiment complete; qualification and acceptance remain directional.")
    return OK


def test_lane_b(args, cfg):
    from ..eval.expert_lane import judge_expert_targets
    from ..eval.public_guards import MissingPublicGuard, pair_public_rows
    from ..shared.lane_b import DomainWin
    from ..shared.manifest import contract_digest
    from ..trainer.contract import RoundSeeds
    from ..trainer.expert_lane import probe_target_routing, union_selection
    from ..trainer.toto2_trainer import Toto2Trainer
    from ..validator.evaluator import evaluate_checkpoint
    from ..validator.expert_lane import _arrays, affinity_contexts, verdict_windows, window_horizon
    from ..validator.merge_acceptance import judge_acceptance
    from ..validator.pool import window_source_from_dir
    from .cli import expert_targets
    from .local_corpus import local_round_stream
    from .score import _resolve_warm_start
    from .training_context import configured_context, live_context, with_contract
    from .verify import verify_repo

    if args.base == "live":
        cfg, _, snapshot = live_context(cfg)
        base = (snapshot.get("king") or {}).get("pointer")
        if not base:
            raise LiveStateUnavailable("Current trained king is unavailable for expert selection.")
        block = snapshot["warm_start"]["era_start_block"]
    else:
        base, block = args.base, args.at_block
        if block is not None:
            cfg = configured_context(cfg, block)
        elif cfg.training.v2_contract is not None:
            cfg = with_contract(cfg, cfg.training.v2_contract.primary_size)
    if cfg.training.base_arch != "cascade_moe":
        raise ValueError("Lane-B tests require a configured MoE contract.")
    targets, error = expert_targets(cfg, args.targets)
    if error:
        raise ValueError(error)
    report = verify_repo(args.repo_dir, cfg, quick=True)
    if not report.ok:
        raise ValueError(report.render())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    base_dir, base_label = _resolve_warm_start(cfg, base, cache_dir=args.work_dir)
    if base_dir is None:
        raise ValueError("Lane B requires trained king weights, not random init.")
    source = window_source_from_dir(args.pool_dir, cfg, label=f"local:{args.pool_dir}")
    selections = []
    for target in targets:
        contexts = affinity_contexts(cfg, source, target, block=block)
        if not contexts:
            raise ValueError(f"No affinity contexts for {target}; add domain labels in metadata.json.")
        selections.append(probe_target_routing(base_dir, contexts, cfg.lane_b, target).select(cfg.lane_b))
    selection = union_selection(selections)
    if not any(selection.values()):
        raise ValueError("No routed expert rows selected on this local affinity pool.")
    windows = verdict_windows(cfg, source, args.seed, block=block)
    if not windows:
        raise ValueError("Held-out pool produced no evaluation windows.")
    trainer = Toto2Trainer(device=args.device)
    contract = cfg.training.for_hours(args.train_hours)
    seeds = RoundSeeds.derive(args.seed, contract)
    # Keep each seed's artifacts separate; no published checkpoint or registry writes.
    import tempfile
    with tempfile.TemporaryDirectory(dir=args.work_dir, prefix="test-") as temp:
        checkpoint = Path(temp) / "candidate"
        with local_round_stream(contract.corpus_mode, args.repo_dir, seeds.generation_seed, cfg.generator,
                               generation=cfg.generation,
                               token_budget=contract.train_tokens, use_sandbox=False,
                               blocked=cfg.static_guard.blocked,
                               budget_denomination=contract.budget_denomination,
                               seed_mix=int(contract.gen_seed_mix or 1), corpus_epochs=contract.corpus_epochs,
                               **({"continuation_protocol": contract.continuation_protocol} if getattr(contract, "continuation_protocol", "") else {})) as stream:
            trained = trainer.train(stream.series(), contract, training_seed=seeds.training_seed,
                                    token_budget=contract.train_tokens, out_dir=checkpoint,
                                    warm_start_dir=base_dir, expert_selection=selection)
        before = evaluate_checkpoint(base_dir, windows, num_samples=cfg.eval.num_samples, device=args.device)
        after = evaluate_checkpoint(trained.local_dir, windows, num_samples=cfg.eval.num_samples, device=args.device)
        public_rows = {}
        public, missing = None, list(cfg.lane_b.require_public)
        if args.public_project:
            from ..eval.benchmarks import run_item_rows
            public_rows = {key: run_item_rows(path, project_dir=str(args.public_project), device=args.device) or {}
                           for key, path in (("before", base_dir), ("candidate", trained.local_dir))}
            public_rows["anchor"] = public_rows["before"]
            try:
                paired = pair_public_rows(public_rows["before"], public_rows["candidate"],
                                          require=cfg.lane_b.require_public,
                                          min_items=cfg.lane_b.min_public_items)
                public, missing = paired.public, []
            except MissingPublicGuard as exc:
                missing = [str(exc)]
        k, c = _arrays(before), _arrays(after)
        if k["ids"] != c["ids"]:
            raise ValueError("Base and candidate evaluation windows did not pair.")
        verdict = judge_expert_targets(
            targets=targets, domains=k["domains"], king_qloss=k["qloss"], king_abs_target=k["abs"],
            king_mase=k["mase"], chal_qloss=c["qloss"], chal_abs_target=c["abs"], chal_mase=c["mase"],
            clusters=k["clusters"], public=public, params=cfg.lane_b.verdict_params(), seed=str(args.seed),
            horizons=[window_horizon(sid) for sid, _ in k["ids"]], horizon_targets=cfg.lane_b.horizon_target_names)
        qualified = [DomainWin(domain=t, hotkey="local", uid=0, round_id="local", merge_block=0,
                               targets=targets, target_lcb=verdict.verdicts[t].target_lcb) for t in verdict.won]
        accepted, acceptance = judge_acceptance(qualified, anchor_scores=before, before_scores=before,
                                                candidate_scores=after, public_rows=public_rows,
                                                cfg=cfg.lane_b, seed=f"local:{args.seed}")
    return {"directional_only": True, "base": base_label, "contract_digest": contract_digest(cfg.training),
            "targets": targets, "selection": selection, "seed": args.seed,
            "training": {"seconds": trained.train_seconds, "metrics": trained.metrics},
            "qualification": asdict(verdict), "qualification_complete": not missing, "public_guards_missing": missing,
            "acceptance": acceptance, "accepted_targets": [w.domain for w in accepted] if not missing else [],
            "live_reward_guaranteed": False,
            "limitations": ["Local affinity and held-out pools differ from the private live pools.",
                            "Acceptance tests this contribution alone against its base; pending live merges can change the result."]}


def cmd_lane_c_test(args):
    from ..shared.config import load_chain_config

    out = reporter(args)
    try:
        data = test_lane_c(args, load_chain_config(args.chain_toml))
    except ImportError as exc:
        return out.fail("missing_marketcast_extra", f"Install .[train,marketcast] for local lane-C tests: {exc}", USAGE)
    except (ValueError, OSError) as exc:
        return out.fail("lane_c_test_failed", str(exc), REJECTED)
    out.result(**data)
    out.info("Local MarketCast test complete; a paid live slot still requires forward confirmation.")
    return OK


def test_lane_c(args, cfg):
    from ..eval.timestral_adapter import TimestralQuantileForecaster, verify_checkpoint
    from ..pool.marketcast import load_snapshot, snapshot_windows, window_covariates
    from ..shared.config.marketcast import COVERAGE_GATE_SUITES
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from ..shared.marketcast_report import (
        MktEntryRows,
        MktWindowRow,
        score_windows,
        verdict_from_rows,
    )
    from ..trainer.marketcast_leg import run_mkt_leg
    from ..trainer.marketcast_runner import has_generator_tree

    mc = cfg.marketcast
    recipe = parse_marketcast_recipe(json.loads((args.repo_dir / "marketcast.json").read_text()), mc,
                                    target=args.target, has_generator=has_generator_tree(args.repo_dir),
                                    submission_time=time.time())
    train = load_snapshot(args.train_snapshot, verify_signature=False)
    heldout = load_snapshot(args.eval_snapshot, verify_signature=False)
    from datetime import UTC, datetime
    start = datetime.fromisoformat(recipe.start).replace(tzinfo=UTC).timestamp()
    end = datetime.fromisoformat(recipe.end).replace(tzinfo=UTC).timestamp() + 86400
    coarse = train.meta.get("coarse") or {}
    coarse_start = int(coarse.get("t0", train.t0))
    coarse_end = coarse_start + int(coarse.get("steps", 0)) * int(coarse.get("bar_minutes", 60)) * 60
    if (train.t0 < start or train.end > end or heldout.t0 < train.end
            or coarse_start < start or coarse_end > end or coarse_end > heldout.t0):
        raise ValueError("Training snapshot must lie inside recipe dates; held-out snapshot must start after training ends.")
    windows = snapshot_windows(heldout, mc, recipe.suite, after=train.end)
    if not windows:
        raise ValueError("No forward held-out windows; check snapshot coverage, liquidity filters and context length.")
    verify_checkpoint(args.base_dir / "model.safetensors", mc.base_sha256)
    forecaster = TimestralQuantileForecaster.from_dir(args.base_dir, device=args.device)

    def predict(covariates):
        result = []
        for w in windows:
            past, known = window_covariates(heldout, w, covariates, context_length=mc.eval_context_length)
            result.append(forecaster.forecast_quantiles(w.context, w.steps, past_covariates=past, future_covariates=known))
        return result

    base_q = predict(())
    import tempfile
    from contextlib import ExitStack
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        temp = stack.enter_context(tempfile.TemporaryDirectory(dir=args.work_dir, prefix="test-"))
        synthetic = None
        if recipe.synthetic_fraction:
            from ..trainer.stream import open_round_stream
            from .verify import verify_repo
            report = verify_repo(args.repo_dir, cfg, quick=True)
            if not report.ok:
                raise ValueError(report.render())
            stream = stack.enter_context(open_round_stream(
                "stream_cpu", args.repo_dir, args.seed, cfg.generator, token_budget=cfg.generator.max_total_points,
                use_sandbox=False, blocked=cfg.static_guard.blocked))
            synthetic = stream.series()
        receipt = run_mkt_leg(recipe=recipe, mc=mc, base_dir=args.base_dir, base_fc_inner=forecaster.inner,
                              train_snap=train, out_dir=Path(temp), seed=args.seed,
                              max_seconds=args.train_hours * 3600, synthetic=synthetic)
        forecaster.inner.model.eval()
        candidate_q = predict(recipe.covariates)
    rows = score_windows([w.target for w in windows], base_q, candidate_q,
                         knots=mc.quantile_knots, coverage_levels=mc.coverage_levels,
                         tail_levels=mc.tail_levels, gated=recipe.suite in COVERAGE_GATE_SUITES)
    entry = MktEntryRows(hotkey="local", trained_pointer="local", target=args.target, phase="judge", after=train.end,
                         windows=tuple(MktWindowRow(wid=w.wid, base=b, ft=f, cov_in=ci, cov_n=cn, tail=tail)
                                       for w, (b, f, ci, cn, tail) in zip(windows, rows, strict=True)))
    verdict = verdict_from_rows(entry, mc, suite=recipe.suite, k=1, seed=f"local:{args.seed}")
    return {"directional_only": True, "target": args.target, "seed": args.seed,
            "base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
            "train_digest": train.digest, "eval_digest": heldout.digest, "windows": len(windows),
            "training": receipt, "verdict": asdict(verdict), "confirmation": "not_tested",
            "live_reward_guaranteed": False,
            "limitations": ["Snapshots are local and digest-checked, not validator-signed evidence.",
                            "A later live settlement must confirm a passing entry; this test cannot establish a paid slot."]}
