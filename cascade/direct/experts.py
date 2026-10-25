"""Measured specialist selection and complete combined continuation artifacts."""

import torch
from safetensors.torch import load, save

from ..shared.direct_moe import canonical, digest
from ..trainer.direct_experts import combine_experts
from ..trainer.moe_model import domain_expert_affinity, select_domain_experts
from .storage import publish


def select(model, batches, *, anchor, panel_digest, domain, fraction, minimum, maximum, exclude):
    if (
        not anchor
        or len(panel_digest) != 64
        or not domain
        or not 0 < fraction <= 1
        or not 0 < minimum <= maximum <= model.cfg.num_experts
    ):
        raise ValueError("explicit bounded specialist selection policy required")
    counts = domain_expert_affinity(model, batches)
    if (counts.sum(1) == 0).any():
        raise ValueError("empty routing evidence")
    selection = select_domain_experts(
        counts, p=fraction, min_n=minimum, max_n=maximum, exclude=exclude
    )
    if any(len(rows) < minimum for rows in selection.values()):
        raise ValueError("insufficient unowned experts")
    evidence = dict(
        anchor=anchor,
        panel=panel_digest,
        domain=domain,
        counts=counts.tolist(),
        fraction=fraction,
        minimum=minimum,
        maximum=maximum,
        excluded={str(k): sorted(v) for k, v in exclude.items()},
        selection={str(k): v for k, v in selection.items()},
    )
    return selection, evidence


def merge_checkpoint(objects, base, contributions, names, *, optimizer_policy):
    """Validate frozen tensors; reset only edited Muon expert rows on every rank.

    The result is an immutable continuation artifact, not permission to promote.
    The caller must evaluate this exact manifest against the accepted anchor.
    """
    base_weights = load(objects.get(base["ranks"]["0"]["weights"]["uri"]))
    tensors = [
        (load(objects.get(cp["ranks"]["0"]["weights"]["uri"])), selection)
        for cp, selection in contributions
    ]
    combined, reset = combine_experts(
        base_weights, tensors, names, optimizer_policy=optimizer_policy
    )
    weights = publish(objects, save({k: v.contiguous() for k, v in combined.items()}))
    ranks = {}
    for rank, parts in base["ranks"].items():
        optimizer = load(objects.get(parts["optimizer"]["uri"]))
        for name, rows in reset.items():
            keys = [f"muon.{name}.momentum", f"muon.{name}.row_v"]
            # The pinned optimizer stores expert tensors as batched Muon. Refuse
            # an unexpected format rather than resetting an entire shared step.
            unexpected = [k for k in optimizer if name in k and k not in keys]
            if unexpected:
                raise ValueError("unsupported edited expert optimizer state")
            for key in keys:
                if key in optimizer:
                    optimizer[key][rows] = 0
        ranks[rank] = {**parts, "weights": weights, "optimizer": publish(objects, save(optimizer))}
    manifest = {
        **base,
        "ranks": ranks,
        "merge": {
            "anchor": digest(base),
            "contributions": [
                {"checkpoint": digest(cp), "selection": {str(k): v for k, v in selection.items()}}
                for cp, selection in contributions
            ],
            "optimizer_policy": optimizer_policy,
        },
    }
    publish(objects, canonical(manifest).encode())
    return manifest


def verify_selection(evidence, *, anchor, domain, selection, layers, experts):
    if (
        evidence["anchor"] != anchor
        or evidence["domain"] != domain
        or len(evidence["panel"]) != 64
        or len(evidence["counts"]) != layers
        or any(len(row) != experts for row in evidence["counts"])
        or any(type(n) is not int or n < 0 for row in evidence["counts"] for n in row)
        or not 0 < evidence["fraction"] <= 1
        or not 0 < evidence["minimum"] <= evidence["maximum"] <= experts
    ):
        raise ValueError("invalid specialist selection provenance")
    counts = torch.tensor(evidence["counts"], dtype=torch.int64)
    if (counts.sum(1) == 0).any():
        raise ValueError("empty specialist routing evidence")
    actual = select_domain_experts(
        counts,
        p=evidence["fraction"],
        min_n=evidence["minimum"],
        max_n=evidence["maximum"],
        exclude={int(k): set(v) for k, v in evidence["excluded"].items()},
    )
    actual = {str(k): v for k, v in actual.items()}
    if actual != evidence["selection"] or actual != {str(k): v for k, v in selection.items()}:
        raise ValueError("specialist selection does not replay")
    return actual
