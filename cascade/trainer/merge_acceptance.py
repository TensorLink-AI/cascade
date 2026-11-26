"""Stage winning expert deltas and accept them one at a time at warm start."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..shared.lane_b import DomainWin
from ..validator.merge_acceptance import acceptance_seed, judge_acceptance
from .expert_lane import merge_rows


class MergeAcceptancePending(TimeoutError):
    """Acceptance runs in the background; no new-init leg may start yet."""


def accept_candidates(pending, *, init_pointer, work_dir, block, ops, cfg, model):
    """Accept the qualified merges one at a time. ``model`` (the driver's
    :class:`~cascade.trainer.lane_b_model.LaneBModel` interface — the eval
    pod for the trainer) stages each ``init + accepted + proposal`` and
    uploads it; ``ops.merge_scores`` scores it (on the pod too). Torch-free
    here: no checkpoint is read on the caller's machine."""
    from ..eval.public_guards import MissingPublicGuard, pair_public_rows
    from .lane_b_model import MergeRefused

    def require_rows(left, right):
        try:
            pair_public_rows(left, right, require=cfg.require_public,
                             min_items=cfg.min_public_items)
        except MissingPublicGuard as e:
            raise OSError(f"acceptance public benchmarks unavailable or incompatible: {e}") from e

    anchor_scores = ops.merge_scores(init_pointer, block)
    anchor_rows = model.rows(init_pointer) or {}
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
        try:
            pointer, handle = model.stage(init_pointer, accepted + [proposal],
                                          out_dir=Path(work_dir) / f"acceptance-{block}-{i}",
                                          repo=f"ckpt-acceptance-{block}-{i}", strict=True)
        except MergeRefused as e:
            rejected.append(original)
            outcomes.append({**identity, "round_id": proposal["round_id"], "hotkey": qualified[0].hotkey,
                             "uid": qualified[0].uid, "domain": qualified[0].domain,
                             "leg_pointer": proposal["rebased_pointer"],
                             "outcome": "acceptance_rejected", "acceptance_block": block,
                             "detail": {"reason": str(e)}})
            continue
        scores = ops.merge_scores(pointer, block)
        rows = model.bench(handle) or {}
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
