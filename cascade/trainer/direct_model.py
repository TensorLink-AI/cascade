"""Explicit model shape interface; intentionally no inferred 220M preset.

Keep the legacy moe_model source bytes unchanged because they fingerprint old
contracts. A direct contract carries the COMPLETE config plus source digest.
"""
from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path


def build_direct_model(architecture, *, device="meta"):
    from . import moe_model
    from .moe_model import CascadeMoEConfig, CascadeMoEModel, param_counts
    required = {f.name for f in fields(CascadeMoEConfig)}
    if set(architecture) != {"config", "source_sha256", "expected_active", "expected_total"}:
        raise ValueError("pin model config/source and measured parameter counts")
    if set(architecture["config"]) != required:
        raise ValueError("direct model requires every architecture field explicitly")
    actual = hashlib.sha256(Path(moe_model.__file__).read_bytes()).hexdigest()
    if architecture["source_sha256"] != actual:
        raise ValueError("model implementation fingerprint mismatch")
    cfg = CascadeMoEConfig.from_dict(architecture["config"])
    if cfg.num_experts < cfg.top_k or cfg.top_k < 1 or cfg.d_model != cfg.num_heads * cfg.head_dim:
        raise ValueError("invalid model dimensions")
    if any(i < 0 or i >= cfg.num_layers for i in (*cfg.moe_blocks, *cfg.variate_blocks)):
        raise ValueError("layer layout outside model")
    import torch
    with torch.device(device):
        model = CascadeMoEModel(cfg)
    counts = param_counts(model)
    if counts != {"active": architecture["expected_active"], "total": architecture["expected_total"]}:
        raise ValueError(f"parameter counts differ: {counts}")
    return model
