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
Remaining approximation vs the Toto 2.0 report: the u-μP init multipliers and
LR width-scaling rules are fan-in-flavoured (not real unit scaling — the
residual scheme, xPos, and Polar Express are now exact); swap in the full
unit-scaling rules before freezing the arch digest if you need bit-fidelity.
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


def sample_cpm_masks(n_rows: int, n_patches: int, *, c_max: int, p_max: float, rng) -> np.ndarray:
    """Sample per-row contiguous-patch-masking masks: ``(n_rows, n_patches)``
    bool, True = masked (unobserved).

    Mirrors Toto 2.0 §2.1: per row, draw a masked fraction ``p ~ U(0, p_max)``,
    then place random contiguous spans of length ``c ~ U{1..c_max}`` until
    ``~p·P`` patches are masked. Pure numpy (no torch) so it is unit-testable;
    the trainer expands the patch-level mask to the model's per-entry channel.
    """
    masks = np.zeros((n_rows, n_patches), dtype=bool)
    if n_patches <= 1 or c_max < 1 or p_max <= 0:
        return masks
    for r in range(n_rows):
        target = rng.uniform(0.0, p_max) * n_patches
        placed = 0
        while masks[r].sum() < target and placed < 4 * n_patches:
            c = int(rng.integers(1, c_max + 1))
            start = int(rng.integers(0, n_patches))
            masks[r, start : start + c] = True
            placed += 1
    return masks


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


# Polar Express (arXiv 2505.16932, Implementation 1): the minimax-optimal
# degree-5 polynomial composition for polar(M) that Toto 2.0 uses to
# orthogonalize NorMuon updates. Coefficients are the paper's precomputed
# optimal composition for ℓ = 1e-3, with its 1.01 numerical safety factor
# folded in. We run the first 6 iterations in float32 (the paper uses bfloat16
# for speed; float32 preserves cascade's byte-reproducible-on-a-pinned-SKU
# training guarantee).
_POLAR_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    ]
]


def _polar_express(G):
    """Approximate ``polar(G)`` via the Polar Express polynomial iteration.
    ``G`` is a 2-D tensor; only matmuls, so GPU-friendly and deterministic."""
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() * 1.01 + 1e-7)
    for a, b, c in _POLAR_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X  # X ← aX + bX³ + cX⁵
    if transposed:
        X = X.mT
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
            Z_CLAMP,
            Toto2Config,
            Toto2Model,
            causal_standardize,
            patch_anchors,
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
        # The wall-clock cap measures ACTUAL TRAINING TIME: the clock starts at
        # the first training batch, so registry fetch, sandbox boot, and model
        # init never eat the budget (material at testnet-scale budgets). Waits
        # for data DURING training do count — that is the anti-trickler bound —
        # and a first batch that never arrives is killed by the sandbox's
        # max_generate_seconds inactivity timeout, not this deadline.
        t0 = time.time()                     # provisional (re-anchored at first batch)
        deadline: float | None = None

        # CPM masks are drawn per batch from a dedicated generator so the run
        # stays byte-reproducible under the shared training_seed.
        mask_rng = np.random.default_rng(training_seed % (2**63))

        # Bucketed batching: series shorter than the full context still train (the
        # generator's max_length can be < context_length). Each batch holds series
        # of the same patch count P, so a single forward covers them; P caps at
        # max_ctx_patches. Position p predicts patch p+1; CPM zeroes contiguous
        # input spans (mask channel = 1) so the model learns to fill multiple
        # future patches from one forward pass — targets stay unmasked.
        for arr in iter_training_batches(
            stream, patch_size=cfg.patch_size, max_ctx_patches=max_ctx_patches,
            batch_size=contract.batch_size,
        ):
            if deadline is None:             # first batch: training starts NOW
                t0 = time.time()
                deadline = t0 + contract.max_train_seconds
            # Standardize from float64: downcasting the raw series first would
            # quantize away small fluctuations at large levels (float32 has ~7
            # digits) before the scaler ever sees them. Only the O(1)-scale z
            # and targets drop to the model dtype.
            x = torch.as_tensor(arr, device=self.device, dtype=torch.float64)  # (B, P*ps)
            num_patches = arr.shape[1] // cfg.patch_size
            cpm = sample_cpm_masks(
                arr.shape[0], num_patches,
                c_max=cfg.cpm_c_max, p_max=cfg.cpm_p_max, rng=mask_rng,
            )
            mask = torch.as_tensor(cpm, device=self.device)                 # (B, P)
            step_mask = (
                mask[:, :, None].expand(-1, -1, cfg.patch_size).reshape(x.shape)
            ).to(x.dtype)
            # Per-step causal stats over unmasked entries only — masked spans
            # carry the last observed stats forward, exactly like the horizon
            # mask patches at inference.
            z, loc, scale = causal_standardize(x, mask=step_mask)
            patches = z.to(self.dtype).view(x.shape[0], num_patches, cfg.patch_size)
            pred = model(patches, mask=mask)            # (B, P, patch_size, num_q)
            pred_q = pred[:, :-1]                        # (B, P-1, patch_size, num_q)
            # Target patch p+1 is scaled at the anchor closing patch p — the
            # stats known when that patch is forecast, so a target never leaks
            # into its own scaling.
            a_loc, a_scale = patch_anchors(loc, scale, cfg.patch_size)
            raw = x.view(x.shape[0], num_patches, cfg.patch_size)
            # Clamp the target to the same bound as z (toto2_model.Z_CLAMP): the
            # model input and the loss target must share one range, and this is the
            # backstop that keeps a pathological jump from producing an inf/huge
            # target that NaNs or destabilizes the shared training step.
            target = torch.asinh(
                (raw[:, 1:] - a_loc[:, :-1, None]) / a_scale[:, :-1, None]
            ).clamp_(-Z_CLAMP, Z_CLAMP).to(self.dtype)
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

        train_seconds = time.time() - t0     # actual training time (from first batch)
        # First-reached-stops: the loop ends on the token budget OR the wall-clock
        # deadline. A deadline stop leaves the model UNDER the contract's compute
        # — self-penalizing in a heat, but in a final it silently breaks the
        # equal-compute pairing, so it must be loud in the record, never implicit.
        deadline_hit = (
            deadline is not None and tokens < token_budget and time.time() > deadline
        )
        if deadline_hit:
            log.warning(
                "wall-clock deadline (%ds) hit at %d/%d tokens (%.0f%%): checkpoint is "
                "under the contract budget — slow corpus or under-provisioned device",
                contract.max_train_seconds, tokens, token_budget,
                100.0 * tokens / max(1, token_budget),
            )
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
            "deadline_hit": deadline_hit,
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
    """NorMuon optimiser (Toto 2.0 §2.3): Nesterov momentum + Polar Express
    orthogonalisation with per-neuron (row) second-moment normalisation — the
    "Nor" — and cautious weight decay for hidden weight matrices; AdamW for
    embeddings/heads/biases, with no weight decay there (μP++ convention).
    """

    def __init__(self, muon_params, adamw_params, *, lr: float, weight_decay: float,
                 momentum: float = 0.95, beta2: float = 0.95, eps: float = 1e-8):
        import torch

        self._torch = torch
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.beta2 = beta2
        self.eps = eps
        self._bufs: dict = {}
        self._row_v: dict = {}  # per-row second-moment EMA (NorMuon eq. 5)
        self.muon_params = list(muon_params)
        # μP++: no weight decay on biases, norms, or input/output projections.
        self.adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=0.0) if adamw_params else None
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
            update = _polar_express(g.add(buf, alpha=self.momentum))  # Nesterov
            # NorMuon: normalise each row against an EMA of its own squared
            # magnitude, so no handful of neurons dominates the update — and
            # the β₂ variance mechanism pinball training relies on is restored.
            v = self._row_v.get(p)
            if v is None:
                v = self._torch.zeros(p.shape[0], 1, device=p.device, dtype=update.dtype)
                self._row_v[p] = v
            v.mul_(self.beta2).add_(
                (update * update).mean(dim=1, keepdim=True), alpha=1.0 - self.beta2
            )
            # Row-normalise, then restore the orthogonalized update's Frobenius
            # norm (NorMuon alg. 1 step 10): the per-row rebalancing must
            # redistribute the step, not inflate it — without the restore, the
            # zero-init EMA scales the first steps by ~1/sqrt(1-β₂) and
            # steady-state elements to RMS 1 instead of the ~1/sqrt(cols) the
            # Muon-convention base_lr is calibrated for.
            normed = update / (v.sqrt() + self.eps)
            update = normed * (update.norm() / normed.norm().clamp_min(self.eps))
            scale = max(1.0, (p.shape[0] / p.shape[1]) ** 0.5)
            if self.weight_decay:
                # cautious weight decay: only where decay agrees with the update
                cautious = ((update * p.data) > 0).to(p.dtype)
                p.data.mul_(1.0 - lr * self.weight_decay * cautious)
            p.data.add_(update, alpha=-lr * scale)
        if self.adamw is not None:
            self.adamw.step()


# ── the loader written into every checkpoint ─────────────────────────────────
# Self-contained: imports the sibling model.py by file path, rebuilds the arch,
# loads weights, and decodes forecasts via contiguous patch masking — the whole
# horizon in ONE forward pass (masked horizon patches), no autoregressive
# sampling. Matches the contract cascade.validator.evaluator expects:
#   Wrapper(checkpoint_dir, device=...).forecast(history_1d, horizon, num_samples)
#       -> ndarray (1, num_samples, horizon)
# and additionally exposes the quantile head directly (what benchmark CRPS
# consumes), batched across series:
#   forecast_quantiles(history, horizon)          -> (1, horizon, num_q)
#   forecast_quantiles_batch(histories, horizon)  -> (B, horizon, num_q)
_FORECAST_WRAPPER_PY = '''"""Auto-generated by cascade Toto2Trainer. Loads the trained checkpoint and
decodes the full horizon in one forward pass via contiguous patch masking
(CPM) — no autoregressive sampling. Exposes:

  forecast(history, horizon, num_samples) -> (1, num_samples, horizon)
      the cascade validator contract — sample paths drawn once from the
      decoded quantiles (seeded per window for validator consensus).
  forecast_quantiles(history, horizon) -> (1, horizon, num_q)
  forecast_quantiles_batch(histories, horizon) -> (B, horizon, num_q)
      the quantile head directly — what benchmark CRPS consumes; batched
      across series so eval sweeps amortize the forward passes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Single-pass CPM decoding is stable to ~768 steps (Toto 2.0 tech report);
# longer horizons block-decode: commit the median per block, then continue.
STABLE_DECODE_STEPS = 768


def _load_model_module(d: Path):
    spec = importlib.util.spec_from_file_location("cascade_ckpt_model", d / "model.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: model.py defines an @dataclass, and the dataclass
    # machinery does sys.modules.get(cls.__module__).__dict__ during class
    # creation — which is None (AttributeError) unless the module is registered.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Wrapper:
    def __init__(self, checkpoint_dir, device: str = "cpu"):
        d = Path(checkpoint_dir)
        self.device = device
        cfg_obj = json.loads((d / "config.json").read_text())
        self.m = _load_model_module(d)
        self.cfg = self.m.Toto2Config(**cfg_obj["toto2"])
        self.quantile_levels = [float(v) for v in cfg_obj["quantile_levels"]]
        self.levels = torch.tensor(self.quantile_levels, dtype=torch.float32, device=device)
        self.model = self.m.Toto2Model(self.cfg).to(device).eval()
        from safetensors.torch import load_file
        state = load_file(str(d / "weights.safetensors"))
        self.model.load_state_dict(state)

    # ── CPM decoding ──────────────────────────────────────────────────────────

    def _prep(self, histories):
        """Left-pad (with the first value) or truncate each 1-D history to the
        context window. Returns the real-space context ``(B, window_len)`` in
        float64 — standardization happens per decode block, from full
        precision, so large-level series keep their fluctuations."""
        ps = self.cfg.patch_size
        n_ctx = max(2, self.cfg.context_length // ps)
        window_len = n_ctx * ps
        rows = []
        for h in histories:
            h = np.asarray(h, dtype=np.float64).reshape(-1)
            if h.shape[0] < window_len:
                pad = np.full(window_len - h.shape[0], h[0] if h.size else 0.0)
                h = np.concatenate([pad, h])
            else:
                h = h[-window_len:]
            rows.append(h)
        return torch.as_tensor(np.stack(rows), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def _decode_block_z(self, z, block: int):
        """One CPM forward pass: append ``block`` masked patches to the
        normalized context ``(B, L)`` and read their z-space quantiles
        ``(B, block*patch_size, num_q)``."""
        ps = self.cfg.patch_size
        # keep as much context as the positional table allows
        ctx_p = min(z.shape[1] // ps, self.cfg.max_patches - block)
        ctx = z[:, -ctx_p * ps :].view(z.shape[0], ctx_p, ps)
        filler = torch.zeros(z.shape[0], block, ps, dtype=ctx.dtype, device=self.device)
        mask = torch.zeros(z.shape[0], ctx_p + block, dtype=ctx.dtype, device=self.device)
        mask[:, ctx_p:] = 1.0
        pred = self.model(torch.cat([ctx, filler], dim=1), mask=mask)
        # position i predicts patch i+1 → the horizon patches come from
        # positions ctx_p-1 .. ctx_p+block-2.
        q = pred[:, ctx_p - 1 : ctx_p + block - 1]          # (B, block, ps, nq)
        q, _ = torch.sort(q, dim=-1)                        # prevent quantile crossing
        return q.reshape(z.shape[0], block * ps, -1)

    @torch.no_grad()
    def _decode_quantiles(self, x, horizon: int):
        """Block-decode real-space quantiles ``(B, horizon, num_q)`` from the
        real-space context ``x`` ``(B, L)``.

        Each block re-runs the causal scaler over history + committed medians
        and unscales with the resulting end-of-context anchor. Committed
        patches are *observed* context for later blocks, and in training the
        causal stats advance through every observed patch — so the anchor must
        advance with them; reusing the pre-horizon anchor would feed blocks ≥ 2
        a scale/location regime the model never sees in training. Clamp bounds
        are fixed from the original context (min/max ± 1e4x anchor scale, per
        the report) so committed medians can't widen them.
        """
        ps = self.cfg.patch_size
        stable = max(1, min(STABLE_DECODE_STEPS // ps, self.cfg.max_patches - 2))
        remaining = -(-int(horizon) // ps)
        lo = hi = None
        out = []
        while remaining > 0:
            block = min(remaining, stable)
            z, loc_t, scale_t = self.m.causal_standardize(x)
            loc = loc_t[:, -1:].double().unsqueeze(-1)      # (B, 1, 1)
            scale = scale_t[:, -1:].double().unsqueeze(-1)
            if lo is None:
                lo = x.min(dim=-1, keepdim=True).values.unsqueeze(-1) - 1e4 * scale
                hi = x.max(dim=-1, keepdim=True).values.unsqueeze(-1) + 1e4 * scale
            qz = self._decode_block_z(z.to(torch.float32), block)
            q = torch.sinh(qz.double()) * scale + loc       # (B, block*ps, nq)
            q = torch.clamp(q, min=lo, max=hi)
            out.append(q)
            remaining -= block
            if remaining > 0:
                x = torch.cat([x, q[..., q.shape[-1] // 2]], dim=1)
        return torch.cat(out, dim=1)[:, : int(horizon)]

    # ── quantile head (benchmark path) ────────────────────────────────────────

    @torch.no_grad()
    def forecast_quantiles_batch(self, histories, horizon: int) -> np.ndarray:
        """Decode ``len(histories)`` series in one batch → real-space quantiles
        ``(B, horizon, num_q)`` at ``self.quantile_levels``. arcsinh + affine
        are monotone increasing, so quantiles map pointwise."""
        q = self._decode_quantiles(self._prep(list(histories)), horizon)
        return q.detach().cpu().numpy().astype(np.float64)

    def forecast_quantiles(self, history, horizon: int) -> np.ndarray:
        return self.forecast_quantiles_batch([history], horizon)

    # ── validator contract (sample paths) ─────────────────────────────────────

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

        q = self._decode_quantiles(self._prep([hist]), horizon)[0]  # (h, nq) real-space
        # One draw per step per path via the piecewise-linear inverse CDF of the
        # decoded quantiles (already clamped and monotone in the level).
        # Quantiles decode once; samples never feed back.
        nq = q.shape[-1]
        levels = self.levels
        u = torch.rand(int(num_samples), int(horizon), device=self.device, generator=generator)
        idx = torch.searchsorted(levels, u.clamp(levels[0].item(), levels[-1].item()))
        idx = idx.clamp(1, nq - 1)
        i_lo = idx - 1
        i_hi = idx
        qe = q.unsqueeze(0).expand(u.shape[0], -1, -1)      # (ns, h, nq)
        vl = torch.gather(qe, -1, i_lo.unsqueeze(-1)).squeeze(-1)
        vh = torch.gather(qe, -1, i_hi.unsqueeze(-1)).squeeze(-1)
        ql = levels[i_lo].double(); qh = levels[i_hi].double()
        frac = ((u.double() - ql) / (qh - ql).clamp_min(1e-8)).clamp(0, 1)
        out = vl + frac * (vh - vl)                         # (ns, h)
        return out.detach().cpu().numpy().reshape(1, int(num_samples), int(horizon))
'''
