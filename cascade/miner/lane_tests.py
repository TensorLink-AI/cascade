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
    p.add_argument("repo_dir", type=Path, help="Your generator tree (lane B) or recipe tree (lane C).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--seed", type=int, default=0, help="Local seed for training, windows and the bootstrap.")
    p.add_argument("--device", default="cpu", help="Torch device for local training and scoring (e.g. cuda).")
    p.add_argument("--train-hours", type=positive_seconds, default=0.01,
                   help="Local training budget (default 0.01h); does not alter live admission budgets.")
    p.add_argument("--work-dir", type=Path, default=Path(".cascade-work") / f"lane-{lane}")
    if lane == "b":
        p.add_argument("--targets", required=True,
                       help="Your declared targets, comma-separated (as for `cascade deploy --expert-domain`; "
                            "up to [lane_b] max_domains_per_leg). The preview runs the validators' judge on "
                            "EVERY [lane_b] slot: declared ones at gain_margin, the rest at "
                            "undeclared_gain_margin.")
        p.add_argument("--base", required=True, help="Local MoE checkpoint directory, immutable pointer, or live (current king).")
        p.add_argument("--pool-dir", type=Path, required=True, help="Real held-out series with metadata.json domain/source labels.")
        p.add_argument("--at-block", type=int, default=None)
        p.add_argument("--public-project", type=Path, default=None,
                       help="Run the production public benchmark sidecar from this project; otherwise public guards "
                            "stay unverified and suite:<s> slots are inconclusive.")
        p.add_argument("--anchor", default=None,
                       help="The era's anchor king (local checkpoint directory or immutable pointer): the signed "
                            "manifest's warm_start_base when the init carries lane-B merges, else warm_start_ckpt; "
                            "'live' = the published era init, which is the anchor only when it carries no merges. "
                            "Enables the suite-slot anchor guard; without it suite slots are previewed without "
                            "that guard.")
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
    for line in data.get("slot_report", ()):
        out.info(line)
    out.result(**data)
    out.info("Local lane-B experiment complete; qualification and acceptance remain directional.")
    return OK


def judge_slots_preview(lb, *, declared, before, after, public_rows=None, anchor=None,
                        anchor_label="", seed):
    """The validators' lane-B verdict on local scores: :func:`judge_expert_slots`
    (every slot tested; declared and undeclared max-T families at
    ``gain_margin`` / ``undeclared_gain_margin``; WB26 guard 4 + anchor guard
    on suite slots) called with the same arguments
    :func:`cascade.validator.expert_lane.judge_entry` builds — public pairing,
    horizon labels and the anchor input come from the validator's own helpers.

    ``before`` / ``after`` / ``anchor`` are ``WindowScore`` lists on the SAME
    windows; ``public_rows`` is ``{"before", "candidate", "anchor"}`` per-item
    rows (or empty). Returns ``(TargetsVerdict, public_missing)`` where
    ``public_missing`` is why the public guards could not be paired ("" when
    they were, or nothing was required)."""
    from ..eval.expert_lane import is_suite_target, judge_expert_slots
    from ..eval.public_guards import MissingPublicGuard, pair_public_rows, sentinel_pairing_kw
    from ..validator.expert_lane import _anchor_input, _arrays, _targets, window_horizon

    k, c = _arrays(before), _arrays(after)
    if k["ids"] != c["ids"]:
        raise ValueError("Base and candidate evaluation windows did not pair.")
    rows = dict(public_rows or {})
    public, missing = None, ""
    if rows or lb.require_public:
        try:
            public = pair_public_rows(rows.get("before"), rows.get("candidate"),
                                      min_items=lb.min_public_items,
                                      **sentinel_pairing_kw(rows.get("before"), rows.get("candidate"),
                                                            lb.require_public, None)).public
        except MissingPublicGuard as exc:
            missing = str(exc)
    slots = _targets(lb)
    hts = tuple(getattr(lb, "horizon_target_names", ()) or ())
    extra = {}
    if any(is_suite_target(t) for t in slots):
        extra["anchor"] = (None if anchor is None else
                           lambda: _anchor_input(rows, k, anchor_label or "local-anchor",
                                                 lambda: anchor, lb, None))
    tv = judge_expert_slots(
        slots=slots, declared=tuple(declared), domains=k["domains"],
        king_qloss=k["qloss"], king_abs_target=k["abs"], king_mase=k["mase"],
        chal_qloss=c["qloss"], chal_abs_target=c["abs"], chal_mase=c["mase"],
        clusters=k["clusters"], public=public, params=lb.verdict_params(), seed=str(seed),
        horizons=[window_horizon(sid) for sid, _ in k["ids"]] if hts else None,
        horizon_targets=hts, **extra)
    return tv, missing


def _pct(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if x != x else f"{x * 100:+.2f}%"


def describe_slots(tv, *, anchor_checked: bool, public_missing: str = "") -> list[str]:
    """Plain-English per-slot lines for a :class:`TargetsVerdict` from
    :func:`judge_slots_preview` — reads the verdict, never re-judges."""
    from ..eval.expert_lane import ANCHOR_MISSING, is_suite_target

    lines = ["Lane-B slot preview (the validators' judge_expert_slots):"]
    for fam in ("declared", "undeclared"):
        if fam in tv.maxt_crits:
            lines.append(f"  {fam} family: max-T critical value {tv.maxt_crits[fam]:.3f}")
    if public_missing:
        lines.append(f"  Public guards NOT paired ({public_missing}); a validator would refuse this leg "
                     "until the required benchmarks pair. Pass --public-project.")
    if tv.guard_reason:
        lines.append(f"  Guards FAILED, blocking every slot: {tv.guard_reason}")
    elif tv.guards_computed:
        lines.append("  Guards passed (every domain, horizon and public metric no worse than -tau).")
    if tv.unguarded:
        lines.append(f"  Too few windows/items to guard: {', '.join(tv.unguarded)}")
    suites = [t for t in tv.targets if is_suite_target(t)]
    if suites and not anchor_checked:
        lines.append("  NOTE: suite slots are previewed WITHOUT the anchor guard (no --anchor given); "
                     "a validator also checks them against the era's anchor king.")
    for t in tv.targets:
        v = tv.verdicts[t]
        fam = "declared" if v.declared else "undeclared"
        head = f"  {t} [{fam}, bar {_pct(v.bar)}]"
        if v.inconclusive:
            lines.append(f"{head}: not tested — {v.reason}")
            continue
        metrics = "; ".join(f"{m.upper()} gain {_pct(v.metric_gains.get(m))} (LCB {_pct(v.metric_lcbs.get(m))}, "
                            f"non-inferiority LCB {_pct(v.metric_ni_lcbs.get(m))})"
                            for m in ("crps", "mase"))
        if v.passed:
            verdict = f"CLEARED on {v.gain_metric.upper()}"
        elif v.reason.startswith(ANCHOR_MISSING) and not anchor_checked:
            verdict = ("clears gain, guards and the whole-pool check; the anchor guard was not run "
                       "locally (pass --anchor)")
        elif v.reason.startswith("no gain"):
            over = [m.upper() for m in ("crps", "mase")
                    if v.metric_lcbs.get(m, float("nan")) >= v.bar and v.metric_gains.get(m, 0.0) > 0.0]
            why = (f"{'/'.join(over)} reaches the bar but the other metric fails non-inferiority"
                   if over else "neither CRPS nor MASE LCB reaches the bar")
            verdict = f"not cleared — {why}"
        else:
            verdict = f"not cleared — {v.reason}"
        lines.append(f"{head}: {metrics}: {verdict}")
    return lines


def test_lane_b(args, cfg):
    from ..shared.lane_b import DomainWin
    from ..shared.manifest import contract_digest
    from ..trainer.contract import RoundSeeds
    from ..trainer.expert_lane import probe_target_routing, union_selection
    from ..trainer.toto2_trainer import Toto2Trainer
    from ..validator.evaluator import evaluate_checkpoint
    from ..validator.expert_lane import affinity_contexts, verdict_windows
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
    if cfg.training.base_arch not in ("cascade_moe", "cascade_pyramid"):
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
    anchor_dir, anchor_label = None, ""
    if getattr(args, "anchor", None):
        anchor_dir, anchor_label = _resolve_warm_start(cfg, args.anchor, cache_dir=args.work_dir)
        if anchor_dir is None:
            raise ValueError("--anchor must name trained anchor-king weights, not random init.")
        anchor_label = anchor_label or str(args.anchor)
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
        anchor_scores = (evaluate_checkpoint(anchor_dir, windows, num_samples=cfg.eval.num_samples,
                                             device=args.device) if anchor_dir is not None else None)
        public_rows = {}
        if args.public_project:
            from ..eval.benchmarks import run_item_rows
            paths = [("before", base_dir), ("candidate", trained.local_dir)]
            if anchor_dir is not None:
                paths.append(("anchor", anchor_dir))
            public_rows = {key: run_item_rows(path, project_dir=str(args.public_project), device=args.device) or {}
                           for key, path in paths}
        verdict, public_missing = judge_slots_preview(
            cfg.lane_b, declared=targets, before=before, after=after, public_rows=public_rows,
            anchor=anchor_scores, anchor_label=anchor_label, seed=args.seed)
        missing = [public_missing] if public_missing else (
            [] if public_rows else list(cfg.lane_b.require_public))
        qualified = [DomainWin(domain=t, hotkey="local", uid=0, round_id="local", merge_block=0,
                               targets=targets, target_lcb=verdict.verdicts[t].target_lcb) for t in verdict.won]
        # acceptance's anchor-domain guards: the real anchor when given, else the base
        acc_rows = dict(public_rows)
        if public_rows and "anchor" not in acc_rows:
            acc_rows["anchor"] = public_rows["before"]
        accepted, acceptance = judge_acceptance(qualified, anchor_scores=anchor_scores or before,
                                                before_scores=before, candidate_scores=after,
                                                public_rows=acc_rows, cfg=cfg.lane_b,
                                                seed=f"local:{args.seed}")
    limitations = ["Local affinity and held-out pools differ from the private live pools.",
                   "Acceptance tests this contribution alone against its base; pending live merges can change the result."]
    if anchor_dir is None:
        limitations.append("Suite slots were previewed without the anchor guard (pass --anchor); "
                           "acceptance's original-init guards used the base as the anchor.")
    return {"directional_only": True, "base": base_label, "contract_digest": contract_digest(cfg.training),
            "targets": targets, "selection": selection, "seed": args.seed,
            "training": {"seconds": trained.train_seconds, "metrics": trained.metrics},
            "qualification": asdict(verdict), "qualification_complete": not missing, "public_guards_missing": missing,
            "won_slots": verdict.won, "anchor": anchor_label or None,
            "slot_report": describe_slots(verdict, anchor_checked=anchor_dir is not None,
                                          public_missing="; ".join(missing)),
            "acceptance": acceptance, "accepted_targets": [w.domain for w in accepted] if not missing else [],
            "live_reward_guaranteed": False, "limitations": limitations}


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
    from ..pool.marketcast import load_snapshot, target_windows, window_covariates
    from ..shared.config.marketcast import COVERAGE_GATE_SUITES, base_suite
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from ..shared.marketcast_report import (
        MktEntryRows,
        MktWindowRow,
        baseline_scores,
        guard_scores,
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
    windows = target_windows(heldout, mc, recipe.suite, after=train.end)
    if not windows:
        raise ValueError("No forward held-out windows; check snapshot coverage, liquidity filters and context length.")
    verify_checkpoint(args.base_dir / "model.safetensors", mc.base_sha256)
    forecaster = TimestralQuantileForecaster.from_dir(args.base_dir, device=args.device)

    def predict(covariates):
        result = []
        for w in windows:
            past, known = window_covariates(heldout, w, covariates, context_length=mc.eval_context_length,
                                            event_times=tuple(mc.event_times))
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
                         tail_levels=mc.tail_levels,
                         gated=[base_suite(w.suite) in COVERAGE_GATE_SUITES for w in windows],
                         scales=[w.scale for w in windows])
    gs = guard_scores([w.target for w in windows], base_q, candidate_q, knots=mc.quantile_knots,
                      guards=[mc.guards_of(w.suite) for w in windows], scales=[w.scale for w in windows])
    bls = baseline_scores(windows, knots=mc.quantile_knots)
    entry = MktEntryRows(hotkey="local", trained_pointer="local", target=args.target, phase="judge", after=train.end,
                         windows=tuple(MktWindowRow(wid=w.wid, base=b, ft=f, cov_in=ci, cov_n=cn, tail=tail, **g, bl=x)
                                       for w, (b, f, ci, cn, tail), g, x in zip(windows, rows, gs, bls, strict=True)))
    verdict = verdict_from_rows(entry, mc, suite=recipe.suite, k=1, seed=f"local:{args.seed}")
    return {"directional_only": True, "target": args.target, "seed": args.seed,
            "base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
            "train_digest": train.digest, "eval_digest": heldout.digest, "windows": len(windows),
            "training": receipt, "verdict": asdict(verdict), "confirmation": "not_tested",
            "live_reward_guaranteed": False,
            "limitations": ["Snapshots are local and digest-checked, not validator-signed evidence.",
                            "A later live settlement must confirm a passing entry; this test cannot establish a paid slot."]}
