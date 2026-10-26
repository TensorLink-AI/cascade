"""Standalone loader for a Pyramid MoE checkpoint; copied as forecast_wrapper.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


class Wrapper:
    def __init__(self, checkpoint_dir, device="cpu"):
        root = Path(checkpoint_dir)
        spec = importlib.util.spec_from_file_location("cascade_pyramid_checkpoint", root / "model.py")
        self.m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.m
        spec.loader.exec_module(self.m)
        obj = json.loads((root / "config.json").read_text())
        self.cfg = self.m.CascadeMoEConfig.from_dict(obj["moe"])
        self.quantile_levels = list(self.m.QUANTILE_LEVELS)
        self.device = device
        self.model = self.m.CascadeMoEModel(self.cfg).to(device).eval()
        from safetensors.torch import load_file
        self.model.load_state_dict(load_file(str(root / "weights.safetensors")), strict=True)

    def _prep(self, histories):
        rows = []
        for history in histories:
            row = np.asarray(history, dtype=np.float64).reshape(-1)
            if not np.isfinite(row).all():
                raise ValueError("non-finite forecast history")
            size = self.cfg.context_length
            if row.size < size:
                row = np.pad(row, (size - row.size, 0), constant_values=row[0] if row.size else 0)
            rows.append(row[-size:])
        return torch.as_tensor(np.stack(rows), device=self.device, dtype=torch.float64)

    def _parameters(self, context, horizon):
        # context B,C,L; horizon is hidden before the scaler and before the stem.
        b, c, length = context.shape
        ps = self.cfg.patch_size
        count = (horizon + ps - 1) // ps
        x = torch.cat([context, context.new_zeros(b, c, count * ps)], -1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[..., length:] = True
        flat = x.reshape(b * c, -1)
        z, loc, scale = self.m.causal_standardize(flat, mask.reshape_as(flat))
        z = z.masked_fill(mask.reshape_as(flat), 0).float().reshape(b, c, -1, ps)
        params = self.model(z, mask=mask.reshape(b, c, -1, ps), with_parameters=True)
        start = length // ps - 1
        params = params[:, :, start:start + count]
        loc = loc.reshape(b, c, -1, ps)[:, :, start:start + count, -1]
        scale = scale.reshape(b, c, -1, ps)[:, :, start:start + count, -1]
        return params, loc.float(), scale.float()

    @torch.no_grad()
    def forecast_quantiles_batch(self, histories, horizon):
        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        context = self._prep(histories)[:, None]
        params, loc, scale = self._parameters(context, int(horizon))
        q = self.m.quantiles(params) * scale[..., None, None] + loc[..., None, None]
        if not torch.isfinite(q).all():
            raise FloatingPointError("non-finite pyramid quantiles")
        return q[:, 0].reshape(len(histories), -1, self.cfg.num_quantiles)[:, :horizon].cpu().numpy().astype(np.float64)

    def forecast_quantiles(self, history, horizon):
        return self.forecast_quantiles_batch([history], horizon)

    @torch.no_grad()
    def forecast_joint(self, history, horizon, num_samples):
        history = np.atleast_2d(np.asarray(history, dtype=np.float64))
        if history.ndim != 2 or horizon < 1 or num_samples < 1:
            raise ValueError("expected C,L history and positive horizon/sample count")
        context = self._prep(history)[None]
        params, loc, scale = self._parameters(context, int(horizon))
        # Same raw window seed on each validator; no learned sample clipping.
        seed_bytes = history.tobytes() + int(horizon).to_bytes(8, "big") + int(num_samples).to_bytes(8, "big")
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        samples = self.m.sample_copula_paths(params, int(num_samples), generator=generator)
        samples = samples * scale[..., None, None] + loc[..., None, None]
        out = samples[0].transpose(1, 2).reshape(history.shape[0], num_samples, -1)[..., :horizon]
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite pyramid forecast")
        return out.cpu().numpy().astype(np.float64)

    def forecast(self, history, horizon, num_samples):
        return self.forecast_joint(np.asarray(history).reshape(1, -1), horizon, num_samples)
