"""The pyramid copula (energy-score) loss stays finite on extreme jumps."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
from cascade.trainer.pyramid_trainer import losses


@pytest.mark.parametrize("jump", [1e13, 1e15, 1e20, 3e38])
def test_energy_score_survives_a_huge_jump_after_a_flat_context(jump):
    """uid31's carried king (2026-10-29): a flat context then a ~1e13 jump made
    the float32 energy score inf and aborted the leg at step 47."""
    torch.manual_seed(7)
    model = CascadeMoEModel(CascadeMoEConfig(
        d_model=8, num_layers=3, num_heads=2, head_dim=4, patch_size=4, dense_ffn=12,
        num_experts=8, top_k=2, expert_hidden=5, shared_hidden=7, conv_channels=4,
        moe_blocks=(1, 2), variate_blocks=()))
    v = np.full((1, 1, 64), 5.0)
    v[..., 48:] = jump
    marginal, joint = losses(model, torch.as_tensor(v), torch.zeros(1, 1, 16, dtype=torch.bool),
                             generator=torch.Generator().manual_seed(1))
    assert np.isfinite(float(marginal)) and np.isfinite(float(joint))
    grads = torch.autograd.grad(joint, list(model.parameters()), allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
