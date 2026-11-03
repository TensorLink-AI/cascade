"""The output and batch geometry ``cascade verify`` reports.

Kept out of :mod:`cascade.miner.verify` on purpose: that file's bytes are part
of the v2 launch verifier identity (``cascade.trainer.launch_preflight.
identity``) and ``verify_repo`` runs inside trainer baseline qualification, so
display-only code must neither change those bytes nor add failure modes there.
The CLI attaches this after ``verify_repo`` returns.
"""

from __future__ import annotations


def _contract_geometry(label: str, contract, channels: int | None) -> dict:
    """How the trainer batches a ``(C, L)`` yield under ``contract``
    (:func:`cascade.trainer.toto2_trainer.iter_training_batches`): a
    ``"sequences"`` batch packs ``max(1, batch_size // C)`` series per step,
    a ``"series"`` batch ``batch_size`` series of C channels each."""
    size = int(contract.batch_size)
    denom = str(contract.batch_denomination)
    row = {"contract": label, "batch_size": size, "batch_denomination": denom,
           "micro_batches": int(getattr(contract, "micro_batches", 1) or 1),
           "context_length": int(contract.context_length),
           "long_series": str(getattr(contract, "long_series", "last") or "last"),
           "corpus_mode": str(contract.corpus_mode)}
    if channels:
        series = max(1, size // channels) if denom == "sequences" else size
        row.update(series_per_step=series, channels=channels,
                   channel_sequences_per_step=series * channels)
    return row


def geometry(cfg, across_launch: bool) -> dict:
    """The output and batch geometry a verify pass is checked against, from the
    loaded config (the same policy :func:`cascade.miner.verify.verify_repo`
    applies) — so a miner sees e.g. "exactly 32 channels; 16 series x 32
    channels per step" instead of inferring it."""
    from .verify import _policy

    policy = _policy(cfg, across_launch)
    v2 = cfg.training.v2_contract
    both = across_launch and v2 is not None
    contracts = [("pre-v2" if both else "effective", cfg.screen_contract())]
    if both:
        contracts.append(("v2", v2))
    fixed = int(policy.max_channels) if int(policy.min_channels) == int(policy.max_channels) else None
    return {"min_channels": int(policy.min_channels), "max_channels": int(policy.max_channels),
            "min_length": int(policy.min_length), "max_length": int(policy.max_length),
            "groups_accepted": "groups" in tuple(policy.accepted_fields),
            "packed_sources": str(getattr(cfg.static_guard, "packed_sources", "scan")),
            "training": [_contract_geometry(label, c, fixed) for label, c in contracts]}


def render_geometry(g: dict) -> list[str]:
    lo, hi = g["min_channels"], g["max_channels"]
    width = (f"exactly {hi} channels per yield (pack narrower processes into the {hi} rows "
             "yourself; nothing pads or merges yields)" if lo == hi
             else f"{lo}-{hi} channels per yield")
    lines = [f"  checked geometry: {width}, length {g['min_length']}-{g['max_length']}; "
             + ("optional `groups` packing accepted" if g["groups_accepted"]
                else "values-only (`groups` not accepted by every checked contract)")
             + f"; packed sources: {g['packed_sources']}"]
    for t in g["training"]:
        if "series_per_step" in t and t["batch_denomination"] == "sequences":
            batch = (f"{t['batch_size']} sequences = {t['series_per_step']} series"
                     f" x {t['channels']} channels per step")
        elif "series_per_step" in t:
            batch = f"{t['series_per_step']} series x {t['channels']} channels per step"
        else:
            batch = (f"{t['batch_size']} {t['batch_denomination']} per step"
                     + (" (series = batch_size // C)" if t["batch_denomination"] == "sequences"
                        else " (x C channels each)"))
        lines.append(f"  training [{t['contract']}]: {batch}, {t['micro_batches']} micro-batch(es); "
                     f"context {t['context_length']}, long series: {t['long_series']}; "
                     f"corpus: {t['corpus_mode']}")
    return lines


def safe_geometry(cfg, across_launch: bool) -> dict | None:
    """:func:`geometry`, or ``None`` when the config lacks a field (display only:
    never fails a verify)."""
    try:
        return geometry(cfg, across_launch)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
