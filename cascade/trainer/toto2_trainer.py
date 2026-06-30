"""Reference :class:`~cascade.trainer.contract.BaseTrainer` — trains a
Toto2-4M backbone **from random initialisation** under cascade's fixed
contract and writes a self-loading checkpoint.

This is the GPU seam made concrete. Plug it into the live trainer with::

    cascade-trainer --trainer cascade.trainer.toto2_trainer:Toto2Trainer \
        --wallet-name owner --wallet-hotkey trainer

It honours the contract that matters for a *controlled* experiment: a fixed
``token_budget`` (point-passes, identical for king and challenger), a shared
``training_seed`` (identical random init + data order), and the 9-quantile pinball
objective that equals the validator's eval metric. The model lives in
``toto2_model.py`` and is copied into each checkpoint so the validator can
reload it.

Validation status: this is a faithful, runnable reference, not a byte-exact
clone of Datadog's released 4M. It needs a GPU to train end-to-end (no GPU in
CI) — validate a real run on your reference box, then pin ``[training]
base_arch_digest`` and ``ref_throughput_tokens_per_s`` to what you launch with.
The u-μP init multipliers and the NorMuon optimiser are approximated (u-μP-style
fan-in init + Muon-orthogonalised matrices + AdamW for the rest); swap in exact
versions before freezing the arch digest if you need bit-fidelity.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..shared.config import TrainingContractConfig
from .contract import TrainLogger, TrainResult

log = logging.getLogger("cascade.trainer.toto2")

LOG_EVERY_STEPS = 50


def _lr_at(token_pos: int, total: int, warmup: int, base_lr: float) -> float:
    """warmup_cosine over the token budget: linear warmup then cosine to 0."""
    if warmup > 0 and token_pos < warmup:
        return base_lr * token_pos / max(1, warmup)
    if total <= warmup:
        return base_lr
    progress = (token_pos - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def iter_training_batches(stream, *, patch_size: int, max_ctx_patches: int, batch_size: int):
    """Yield ``(B, P*patch_size)`` float64 training batches from a series stream.

    Pure numpy (no torch) so it is unit-testable. Each incoming ``(C, L)`` or
    ``(L,)`` series is reduced to channel 0 and its last ``P`` patches are kept,
    where ``P = min(L // patch_size, max_ctx_patches)``; series with fewer than 2
    patches are skipped (a next-patch objective needs at least one input + one
    target patch). Batches are **bucketed by ``P``** so all rows in a batch share
    a length and stack without padding. Full buckets are emitted eagerly; partial
    buckets are flushed when the stream ends. This is what lets the trainer learn
    from realistic series whose length is below the full ``context_length``.
    """
    buckets: dict[int, list[np.ndarray]] = {}
    for series in stream:
        s = np.asarray(series, dtype=np.float64)
        if s.ndim == 2:
            s = s[0]
        p = min(int(s.shape[0]) // patch_size, max_ctx_patches)
        if p < 2:
            continue
        buckets.setdefault(p, []).append(s[-p * patch_size :])
        if len(buckets[p]) >= batch_size:
            yield np.stack(buckets.pop(p), axis=0)
    for items in buckets.values():
        if items:
            yield np.stack(items, axis=0)


def _zeropower_newtonschulz(G, steps: int = 5, eps: float = 1e-7):
    """Newton-Schulz orthogonalisation used by Muon (the 'norm' in NorMuon is the
    operator's refinement). ``G`` is a 2-D tensor."""

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X.to(G.dtype)


class Toto2Trainer:
    """Owner GPU backend. Stateless across king/challenger calls (a fresh model
    is built per :meth:`train`), so the shared ``training_seed`` gives both the
    identical random init the controlled experiment requires.

    With ``deterministic=True`` (the default) the run is **byte-reproducible on a
    fixed GPU model**: deterministic cuBLAS/cuDNN, the math (not flash) attention
    kernel, and all RNGs seeded from ``training_seed``. Combined with running king
    and challenger on the **same pinned GPU SKU** (enforced at the validator gate
    via the recorded ``gpu_name``), a re-derived audit run reproduces the exact
    checkpoint. Pinning the SKU is the operator's job; this class makes the run
    deterministic given it.
    """

    def __init__(self, *, device: str | None = None, dtype: str = "float32",
                 deterministic: bool = True):
        import os

        self.deterministic = deterministic
        if deterministic:
            # Must be set before the first cuBLAS handle is created for
            # deterministic GEMMs; setdefault so an operator override wins.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

    def _enable_determinism(self, torch, seed: int) -> None:
        """Force byte-reproducible kernels + seed every RNG (best-effort across
        torch versions)."""
        if not self.deterministic:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
        # Flash / mem-efficient attention are nondeterministic; force math SDPA.
        for fn, arg in (("enable_flash_sdp", False),
                        ("enable_mem_efficient_sdp", False),
                        ("enable_math_sdp", True)):
            with contextlib.suppress(Exception):
                getattr(torch.backends.cuda, fn)(arg)

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        stream: Iterator[np.ndarray],
        contract: TrainingContractConfig,
        *,
        training_seed: int,
        token_budget: int,
        out_dir: Path,
        logger: TrainLogger | None = None,
    ) -> TrainResult:
        import torch

        from .toto2_model import (
            QUANTILE_LEVELS,
            Toto2Config,
            Toto2Model,
            causal_standardize,
            pinball_loss,
        )

        torch.manual_seed(training_seed)
        np.random.seed(training_seed % (2**32 - 1))
        self._enable_determinism(torch, training_seed)

        cfg = Toto2Config.from_contract(contract)
        model = Toto2Model(cfg).to(self.device, self.dtype)
        model.train()
        levels = QUANTILE_LEVELS[: cfg.num_quantiles] if cfg.num_quantiles <= len(QUANTILE_LEVELS) else QUANTILE_LEVELS
        optimizer = self._build_optimizer(model, contract)

        max_ctx_patches = max(2, cfg.context_length // cfg.patch_size)
        warmup = int(getattr(contract, "warmup_tokens", int(token_budget * 0.05)))

        tokens = 0
        step = 0
        last_loss = float("nan")
        t0 = time.time()
        deadline = t0 + contract.max_train_seconds

        # Bucketed batching: series shorter than the full context still train (the
        # generator's max_length can be < context_length). Each batch holds series
        # of the same patch count P, so a single forward covers them; P caps at
        # max_ctx_patches. Position p predicts patch p+1.
        for arr in iter_training_batches(
            stream, patch_size=cfg.patch_size, max_ctx_patches=max_ctx_patches,
            batch_size=contract.batch_size,
        ):
            x = torch.as_tensor(arr, device=self.device, dtype=self.dtype)  # (B, P*ps)
            z, _, _ = causal_standardize(x)
            num_patches = arr.shape[1] // cfg.patch_size
            patches = z.view(x.shape[0], num_patches, cfg.patch_size)
            pred = model(patches)                       # (B, P, patch_size, num_q)
            pred_q = pred[:, :-1]                        # (B, P-1, patch_size, num_q)
            target = patches[:, 1:]                     # (B, P-1, patch_size)
            loss = pinball_loss(pred_q, target, tuple(levels))

            lr = _lr_at(tokens, token_budget, warmup, contract.base_lr)
            for grp in optimizer.param_groups:
                grp["lr"] = lr * grp.get("lr_scale", 1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            tokens += arr.shape[0] * arr.shape[1]
            step += 1
            if logger is not None and step % LOG_EVERY_STEPS == 0:
                elapsed = max(1e-6, time.time() - t0)
                logger({
                    "event": "step", "step": step, "loss": last_loss, "lr": lr,
                    "tokens": tokens, "tokens_frac": tokens / max(1, token_budget),
                    "throughput_tokens_per_s": tokens / elapsed,
                })
            if tokens >= token_budget or time.time() > deadline:
                break

        train_seconds = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters())
        gpu_name = (
            torch.cuda.get_device_name(0)
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        metrics = {
            "final_loss": last_loss, "steps": step, "tokens_seen": tokens,
            "param_count": param_count,
            "throughput_tokens_per_s": tokens / max(1e-6, train_seconds),
            "gpu_name": gpu_name, "deterministic": self.deterministic,
        }
        if logger is not None:
            logger({"event": "done", **metrics})

        self._save_checkpoint(out_dir, model, cfg, tuple(levels), contract)
        log.info("trained toto2: params=%d steps=%d tokens=%d final_loss=%.4f in %.0fs",
                 param_count, step, tokens, last_loss, train_seconds)
        return TrainResult(
            local_dir=out_dir, param_count=param_count, train_seconds=train_seconds, metrics=metrics
        )

    # ── optimiser (NorMuon ≈ Muon for matrices + AdamW for the rest) ──────────

    def _build_optimizer(self, model, contract: TrainingContractConfig):
        import torch

        if contract.optimizer != "normuon_adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=contract.base_lr, weight_decay=contract.weight_decay
            )
        muon, adamw = [], []
        for name, p in model.named_parameters():
            if p.ndim >= 2 and "embed" not in name and "head" not in name and "pos" not in name:
                muon.append(p)
            else:
                adamw.append(p)
        # Muon params are stepped manually in _MuonAdamW; AdamW handles the rest.
        return _MuonAdamW(muon, adamw, lr=contract.base_lr, weight_decay=contract.weight_decay)

    # ── checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, out_dir, model, cfg, levels, contract) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(out / "weights.safetensors"))

        (out / "config.json").write_text(
            json.dumps({
                "arch": "toto2-4m",
                "toto2": cfg.to_dict(),
                "quantile_levels": list(levels),
                "input_transform": getattr(contract, "input_transform", "arcsinh_causal"),
            }, indent=2),
            encoding="utf-8",
        )
        # Copy the model definition + the loader the validator expects.
        (out / "model.py").write_text(
            (Path(__file__).with_name("toto2_model.py")).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (out / "forecast_wrapper.py").write_text(_FORECAST_WRAPPER_PY, encoding="utf-8")


class _MuonAdamW:
    """Minimal NorMuon-style optimiser: Muon (momentum + Newton-Schulz orthogonal
    update) for hidden weight matrices, AdamW for embeddings/heads/biases.

    A reference, not a tuned implementation — the per-neuron second-moment
    normalisation that distinguishes NorMuon from Muon is left to the operator.
    """

    def __init__(self, muon_params, adamw_params, *, lr: float, weight_decay: float,
                 momentum: float = 0.95):
        import torch

        self._torch = torch
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._bufs: dict = {}
        self.muon_params = list(muon_params)
        self.adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay) if adamw_params else None
        # param_groups so the trainer's LR scheduler can set lr uniformly.
        self.param_groups = [{"params": self.muon_params, "lr": lr, "lr_scale": 1.0}]
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.muon_params:
            p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        for p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad
            buf = self._bufs.get(p)
            if buf is None:
                buf = self._torch.zeros_like(g)
                self._bufs[p] = buf
            buf.mul_(self.momentum).add_(g)
            update = _zeropower_newtonschulz(buf)
            scale = max(1.0, (p.shape[0] / p.shape[1]) ** 0.5)
            p.data.mul_(1.0 - lr * self.weight_decay)
            p.data.add_(update, alpha=-lr * scale)
        if self.adamw is not None:
            self.adamw.step()


# ── the loader written into every checkpoint ─────────────────────────────────
# Self-contained: imports the sibling model.py by file path, rebuilds the arch,
# loads weights, and autoregressively samples forecast paths from the predicted
# quantiles. Matches the contract cascade.validator.evaluator expects:
#   Wrapper(checkpoint_dir, device=...).forecast(history_1d, horizon, num_samples)
#       -> ndarray (1, num_samples, horizon)
_FORECAST_WRAPPER_PY = '''"""Auto-generated by cascade Toto2Trainer. Loads the trained checkpoint and
forecasts by sampling the model's predicted quantiles autoregressively."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def _load_model_module(d: Path):
    spec = importlib.util.spec_from_file_location("cascade_ckpt_model", d / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Wrapper:
    def __init__(self, checkpoint_dir, device: str = "cpu"):
        d = Path(checkpoint_dir)
        self.device = device
        cfg_obj = json.loads((d / "config.json").read_text())
        self.m = _load_model_module(d)
        self.cfg = self.m.Toto2Config(**cfg_obj["toto2"])
        self.levels = torch.tensor(cfg_obj["quantile_levels"], dtype=torch.float32, device=device)
        self.model = self.m.Toto2Model(self.cfg).to(device).eval()
        from safetensors.torch import load_file
        state = load_file(str(d / "weights.safetensors"))
        self.model.load_state_dict(state)

    def _quantiles_to_samples(self, q_vals, num_samples, generator):
        # q_vals: (num_samples, patch_size, num_q) — sample one value per step via
        # the piecewise-linear inverse CDF of the predicted quantiles. ``generator``
        # is seeded per window so every validator draws identical samples (consensus)
        # and king/challenger share uniforms (paired Monte-Carlo, lower variance).
        ns, ps, nq = q_vals.shape
        q_vals, _ = torch.sort(q_vals, dim=-1)  # enforce monotone quantiles
        u = torch.rand(ns, ps, device=q_vals.device, generator=generator)
        levels = self.levels
        idx = torch.searchsorted(levels, u.clamp(levels[0].item(), levels[-1].item()))
        idx = idx.clamp(1, nq - 1)
        lo = (idx - 1)
        hi = idx
        ql = levels[lo]; qh = levels[hi]
        vl = torch.gather(q_vals, -1, lo.unsqueeze(-1)).squeeze(-1)
        vh = torch.gather(q_vals, -1, hi.unsqueeze(-1)).squeeze(-1)
        frac = ((u - ql) / (qh - ql).clamp_min(1e-8)).clamp(0, 1)
        return vl + frac * (vh - vl)

    @torch.no_grad()
    def forecast(self, history, horizon: int, num_samples: int) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64).reshape(-1)
        # Deterministic per-window sampling: seed from the (raw history, horizon,
        # num_samples) so every validator computes identical scores and king vs
        # challenger share the uniform draws (paired Monte-Carlo).
        seed_src = hist.tobytes() + int(horizon).to_bytes(8, "big") + int(num_samples).to_bytes(8, "big")
        seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        ps = self.cfg.patch_size
        n_ctx = max(2, self.cfg.context_length // ps)
        window_len = n_ctx * ps
        # left-pad (with the first value) or truncate to window_len
        if hist.shape[0] < window_len:
            pad = np.full(window_len - hist.shape[0], hist[0] if hist.size else 0.0)
            hist = np.concatenate([pad, hist])
        else:
            hist = hist[-window_len:]
        x = torch.as_tensor(hist, dtype=torch.float32, device=self.device)[None, :]  # (1, L)
        loc = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        z = torch.asinh((x - loc) / scale).repeat(num_samples, 1)  # (ns, L)

        generated = []
        steps_needed = horizon
        while sum(g.shape[1] for g in generated) < steps_needed:
            ctx = z[:, -window_len:]
            patches = ctx.view(num_samples, n_ctx, ps)
            pred = self.model(patches)            # (ns, P, ps, num_q)
            next_q = pred[:, -1]                  # (ns, ps, num_q) — next patch
            nxt = self._quantiles_to_samples(next_q, num_samples, generator)  # (ns, ps)
            z = torch.cat([z, nxt], dim=1)
            generated.append(nxt)
        gen = torch.cat(generated, dim=1)[:, :horizon]  # (ns, horizon)
        out = torch.sinh(gen) * scale + loc             # inverse transform
        return out.detach().cpu().numpy().reshape(1, num_samples, horizon)
'''
