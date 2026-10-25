"""Stage winning expert deltas and accept them one at a time at warm start."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from ..shared.lane_b import DomainWin
from ..validator.merge_acceptance import acceptance_seed, judge_acceptance
from .expert_lane import apply_merges, merge_rows


class MergeAcceptancePending(TimeoutError):
    """Acceptance runs in the background; no new-init leg may start yet."""


def accept_candidates(pending, *, init_pointer, init_dir, work_dir, block, ops, cfg, names, load):
    from safetensors.torch import load_file

    from .ckpt_digest import save_tensors_hashed
    from .moe_model import MergeError

    base = load_file(str(init_dir / "weights.safetensors"))
    from ..eval.public_guards import MissingPublicGuard, pair_public_rows

    def require_rows(left, right):
        try:
            pair_public_rows(left, right, require=cfg.require_public,
                             min_items=cfg.min_public_items)
        except MissingPublicGuard as e:
            raise OSError(f"acceptance public benchmarks unavailable or incompatible: {e}") from e

    anchor_scores = ops.merge_scores(init_pointer, block)
    anchor_rows = ops.bench_rows(init_dir) or {}
    require_rows(anchor_rows, anchor_rows)
    before, before_scores, before_rows = init_pointer, anchor_scores, anchor_rows
    accepted, rejected, outcomes, taken, paid_targets = [], [], [], set(), set()
    for i, original in enumerate(pending):
        if taken & merge_rows(original):
            continue  # exclusive rows wait, matching the existing planner
        qualified = [DomainWin(**w) for w in original["qualified_wins"]]
        identity = {"gen_ref": qualified[0].gen_ref,
                    "commit_block": qualified[0].commit_block,
                    "targets": list(qualified[0].targets or (qualified[0].domain,))}
        if paid_targets & {w.domain for w in qualified}:
            continue  # a slot cannot replace another accepted miner in this batch
        proposal = dict(original)
        stage = Path(work_dir) / f"acceptance-{block}-{i}"
        stage.mkdir(parents=True, exist_ok=True)
        for file in ("config.json", "model.py", "forecast_wrapper.py"):
            shutil.copyfile(init_dir / file, stage / file)
        try:
            state = apply_merges(base, accepted + [proposal], load, names, strict=True)
        except MergeError as e:
            rejected.append(original)
            outcomes.append({**identity, "round_id": proposal["round_id"], "hotkey": qualified[0].hotkey,
                             "uid": qualified[0].uid, "domain": qualified[0].domain,
                             "leg_pointer": proposal["rebased_pointer"],
                             "outcome": "acceptance_rejected", "acceptance_block": block,
                             "detail": {"reason": str(e)}})
            continue
        save_tensors_hashed(state, stage / "weights.safetensors")
        pointer = ops.upload(stage, f"ckpt-acceptance-{block}-{i}")
        scores = ops.merge_scores(pointer, block)
        rows = ops.bench_rows(stage) or {}
        require_rows(before_rows, rows)
        require_rows(anchor_rows, rows)
        public = {"anchor": anchor_rows, "before": before_rows, "candidate": rows}
        wins, detail = judge_acceptance(
            qualified, anchor_scores=anchor_scores, before_scores=before_scores,
            candidate_scores=scores, public_rows=public, cfg=cfg,
            seed=f"{acceptance_seed(block, init_pointer)}|{proposal['rebased_pointer']}")
        outcomes.append({**identity, "round_id": proposal["round_id"], "hotkey": qualified[0].hotkey,
                         "uid": qualified[0].uid, "domain": qualified[0].domain,
                         "leg_pointer": proposal["rebased_pointer"],
                         "outcome": "accepted" if wins else "acceptance_rejected",
                         "acceptance_block": block, "detail": detail})
        if not wins:
            rejected.append(original)
            continue
        proposal["acceptance"] = {
            "block": block, "before": before, "candidate": pointer,
            "public_rows": public, "targets": [w.domain for w in wins],
            "wins": [asdict(w) for w in wins],
        }
        accepted.append(proposal)
        taken |= merge_rows(proposal)
        paid_targets |= {w.domain for w in wins}
        before, before_scores, before_rows = pointer, scores, rows
    return accepted, rejected, outcomes
