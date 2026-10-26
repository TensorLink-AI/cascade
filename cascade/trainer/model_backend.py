"""Resolve a known, trusted model implementation from checkpoint byte identity."""
from __future__ import annotations


def checkpoint_backend(checkpoint_dir):
    from ..eval.checkpoint_guard import verify_checkpoint_code

    arch = verify_checkpoint_code(checkpoint_dir)
    if arch == "cascade_pyramid":
        from . import pyramid_model

        return pyramid_model
    if arch == "cascade_moe":
        from . import moe_model

        return moe_model
    raise ValueError("expert operations require a supported MoE checkpoint")
