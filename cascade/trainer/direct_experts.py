"""Bounded specialist merge construction against one immutable accepted anchor."""
from __future__ import annotations


def combine_experts(base, contributions, names, *, optimizer_policy):
    """Validate every frozen slice against base before constructing the combination.

    Returns edited state and the optimizer rows to clear. Actual combined-model
    evaluation remains mandatory in DirectController.merge; individual verdicts
    cannot authorize it. Stale delta transfer is deliberately unsupported.
    """
    from .moe_model import merge_expert_slices
    if optimizer_policy != "reset_edited_rows":
        raise ValueError("explicit supported optimizer policy required: reset_edited_rows")
    result = {k: v.clone() for k, v in base.items()}
    owned = set()
    reset = {}
    for candidate, selection in contributions:
        overlap = {(layer, row) for layer, rows in selection.items() for row in rows}
        if owned & overlap:
            raise ValueError("overlapping expert contributions")
        owned |= overlap
        checked = merge_expert_slices(base, candidate, selection, names)
        for layer, rows in selection.items():
            for name in names[layer]:
                result[name][rows] = checked[name][rows]
                reset.setdefault(name, []).extend(rows)
    return result, {name: sorted(rows) for name, rows in reset.items()}
