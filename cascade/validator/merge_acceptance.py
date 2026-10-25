"""Shared trainer/validator decision for one incremental lane-B merge."""

from __future__ import annotations

from dataclasses import asdict, replace

from ..eval.expert_lane import full_pay, judge_expert_targets
from ..eval.public_guards import MissingPublicGuard, pair_public_rows
from .expert_lane import _arrays, window_horizon


def acceptance_seed(block: int, base: str) -> str:
    return f"lane-b-acceptance|{int(block)}|{base}"


def acceptance_block(cfg, manifest):
    """A seed epoch retains its original init even across activation eras."""
    from ..shared.era import EraSpec, era_length_blocks

    era = EraSpec.from_json(manifest.era)
    return min(era.start_block, era.seed_block + era_length_blocks(cfg.round, era.start_block))


def judge_acceptance(qualified, *, anchor_scores, before_scores, candidate_scores,
                     public_rows, cfg, seed):
    """Return accepted target wins plus diagnostic evidence.

    Gain is marginal against the accepted checkpoint so far. Every domain,
    horizon and public guard is also checked against the ORIGINAL init, so
    small regressions cannot accumulate. Only previously qualified targets
    may earn; the original declaration retains its alpha/k correction.
    """
    if not qualified:
        return [], {"reason": "no qualified targets"}
    declarations = {tuple(w.targets or (w.domain,)) for w in qualified}
    if len(declarations) != 1:
        raise ValueError("qualification declarations differ")
    targets = next(iter(declarations))
    if any(t not in cfg.targets for t in targets):
        return [], {"reason": "declared target is no longer configured"}
    anchor, before, candidate = map(_arrays, (anchor_scores, before_scores, candidate_scores))
    if not anchor["ids"] == before["ids"] == candidate["ids"]:
        raise ValueError("acceptance windows are not paired")
    if not (anchor["domains"] == before["domains"] == candidate["domains"]
            and anchor["clusters"] == before["clusters"] == candidate["clusters"]):
        raise ValueError("acceptance domain/cluster labels differ")
    try:
        pubs = [pair_public_rows(public_rows.get(key), public_rows.get("candidate"),
                                 require=cfg.require_public, min_items=cfg.min_public_items).public
                if public_rows or cfg.require_public else None for key in ("before", "anchor")]
    except MissingPublicGuard as e:
        return [], {"reason": f"public guard missing: {e}"}

    def judge(baseline, declared, public, suffix, **kw):
        return judge_expert_targets(
            targets=declared, domains=anchor["domains"], clusters=anchor["clusters"],
            king_qloss=baseline["qloss"], king_abs_target=baseline["abs"], king_mase=baseline["mase"],
            chal_qloss=candidate["qloss"], chal_abs_target=candidate["abs"],
            chal_mase=candidate["mase"], public=public, params=cfg.verdict_params(),
            seed=f"{seed}|{suffix}",
            horizons=[window_horizon(sid) for sid, _ in anchor["ids"]],
            horizon_targets=cfg.horizon_target_names, **kw)

    marginal = judge(before, targets, pubs[0], "marginal")
    anchored = judge(anchor, (), pubs[1], "anchor", guards_only=True)
    detail = {"reason": anchored.guard_reason or marginal.guard_reason,
              "targets": {t: {"passed": v.passed, "lcb": v.target_lcb,
                               "reason": v.reason} for t, v in marginal.verdicts.items()}}
    if anchored.guard_reason:
        return [], detail
    wins = [replace(w, target_lcb=marginal.verdicts[w.domain].target_lcb).with_full_pay(
        full_pay(marginal.verdicts[w.domain].target_lcb, cfg.full_pay_tiers))
        for w in qualified if marginal.verdicts[w.domain].passed]
    if not wins and not detail["reason"]:
        detail["reason"] = "no qualified target clears marginal gain"
    return wins, detail


def verify_acceptance(runner, manifest):
    """Rebuild and re-score accepted steps; return rewards without mutating them.

    Public per-item rows use the existing signed trainer-evidence trust model.
    Private scores and every intermediate checkpoint are independently checked.
    Qualification evidence must predate this warm start (a strict DAG).
    """
    import hashlib
    import json

    import torch
    from safetensors.torch import load_file

    from ..shared.hippius import StorageError
    from ..shared.lane_b import merge_identity
    from ..trainer.expert_lane import _moe_names, apply_merges
    from .expert_lane import verdict_windows
    from .lane_b import LaneBTransient

    block = acceptance_block(runner.cfg, manifest)
    key = hashlib.sha256(json.dumps(
        [manifest.warm_start_ckpt, manifest.warm_start_base, manifest.warm_start_merges, block],
        sort_keys=True).encode()).hexdigest()
    cache = runner.__dict__.setdefault("_merge_acceptance_ok", {})
    if key in cache:
        for merge, _, accepted_block in cache[key]:
            applied = runner._lane_b().accepted.get(json.dumps(
                merge_identity(merge), separators=(",", ":")))
            if applied is not None and applied != accepted_block:
                raise ValueError("expert delta was already accepted at another warm start")
        return cache[key]
    if not runner.cfg.lane_b.acceptance_at_era(block):
        raise ValueError("merge acceptance belongs to a pre-activation warm start")
    st = runner._lane_b()
    king = manifest.entry_for_role("king")
    if king is None:
        raise ValueError("acceptance requires a king entry")

    def directory(pointer):
        return runner._fetch_checkpoint_dir(replace(king, trained_pointer=pointer))

    def load(pointer):
        return load_file(str(directory(pointer) / "weights.safetensors"))

    source = runner._lane_b_window_source
    if source is None:
        raise StorageError("acceptance window source unavailable")
    accepted, before, result, previous_order = [], manifest.warm_start_base, [], None
    paid_targets = set()
    try:
        with runner._settlement_cfg(block + 1):
            cfg = runner.cfg
            windows = verdict_windows(cfg, source, block, block=block)
            contract = cfg.training.at_block(block).primary_size
            scores_cache = {}

            def scores(pointer):
                if pointer not in scores_cache:
                    try:
                        scores_cache[pointer] = runner._evaluate(
                            replace(king, trained_pointer=pointer), windows, contract)
                    except Exception as e:
                        raise StorageError(f"acceptance scoring unavailable: {e}") from e
                return scores_cache[pointer]

            base = load(manifest.warm_start_base)
            names = _moe_names(directory(manifest.warm_start_base))
            anchor_scores = scores(manifest.warm_start_base)
            for merge in manifest.warm_start_merges:
                proof = merge.get("acceptance") or {}
                if proof.get("block") != block or proof.get("before") != before:
                    raise ValueError("acceptance block or checkpoint chain differs")
                applied = st.accepted.get(json.dumps(merge_identity(merge), separators=(",", ":")))
                if applied is not None and applied != block:
                    raise ValueError("expert delta was already accepted at another warm start")
                qualified = st.qualified_for(merge)
                if not qualified:
                    rid = str(merge.get("round_id", ""))
                    source_manifest = runner._lane_b_load_settlement(rid)
                    if (runner._epoch_start_block(source_manifest) > block
                            or (source_manifest.era and
                                int(source_manifest.era["start_block"]) >= block)):
                        raise ValueError("qualification must precede acceptance warm start")
                    runner._lane_b_fold_on_demand(rid)
                    qualified = st.qualified_for(merge)
                if not qualified or any(w.qualified_block > block for w in qualified):
                    raise ValueError("merge has no prior qualified targets")
                from ..shared.verdict_numbers import same_verdict

                if not same_verdict(merge.get("qualified_wins"), [asdict(w) for w in qualified]):
                    raise ValueError("qualification metadata differs from the recorded verdict")
                if paid_targets & {w.domain for w in qualified}:
                    raise ValueError("two accepted merges compete for one reward slot")
                order = (qualified[0].commit_block, qualified[0].hotkey, merge["rebased_pointer"])
                if previous_order is not None and order <= previous_order:
                    raise ValueError("acceptance candidates are not in commit order")
                previous_order = order
                candidate = str(proof.get("candidate", ""))
                want = apply_merges(base, accepted + [merge], load, names, strict=True)
                got = load(candidate)
                if want.keys() != got.keys() or any(not torch.equal(want[n], got[n]) for n in want):
                    raise ValueError("acceptance intermediate checkpoint differs from its deltas")
                wins, detail = judge_acceptance(
                    qualified, anchor_scores=anchor_scores, before_scores=scores(before),
                    candidate_scores=scores(candidate), public_rows=proof.get("public_rows") or {},
                    cfg=cfg.lane_b,
                    seed=f"{acceptance_seed(block, manifest.warm_start_base)}|{merge['rebased_pointer']}")
                if not wins or proof.get("targets") != [w.domain for w in wins]:
                    raise ValueError(f"merge acceptance failed: {detail['reason']}")
                if not same_verdict(proof.get("wins"), [asdict(w) for w in wins]):
                    raise ValueError("acceptance rewards differ from the re-derived verdict")
                provenance = {k: merge[k] for k in (
                    "rebased_pointer", "king_pointer", "selection", "round_id")}
                result.append((provenance, wins, block))
                paid_targets |= {w.domain for w in wins}
                accepted.append(merge)
                before = candidate
    except LaneBTransient as e:
        raise StorageError(f"acceptance evidence unavailable: {e}") from e
    # Read/scoring errors propagate to the caller as transient; ValueError
    # denotes a deterministic refusal of readable, contradictory evidence.
    cache[key] = result
    while len(cache) > 64:
        del cache[next(iter(cache))]
    return result
