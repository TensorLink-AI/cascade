# ============================================================================
# TRAINING  ·  one run = one Cfg. Resumable; bf16 autocast on Ampere+;
# deterministic paired data via CorpusSampler(step). Every run logs a
# scale-normalized synthetic val CRPS curve; gift_ckpts (15k/30k) snapshot
# weights for the rank-stability check.
# ============================================================================
"""Per-run training loop, held-out synthetic validation, and checkpointing."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from .corpus import CorpusSampler, sample_series
from .device import AMP_DTYPE, DEVICE
from .generators import sample_ensemble
from .model import MiniTSFM2, count_params, sample_cpm_mask
from .optim import build_optimizers, lr_scale, set_lr
from .tokenizers import RobustScaler


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_val_batch(cfg, n=64, seed=777):
    """Fixed held-out synthetic batch from the SAME ensemble prior (never trained on:
    val seed is outside the corpus shard seed range)."""
    rng = np.random.default_rng(seed)
    L = cfg.ctx_span + cfg.train_kmax * cfg.patch
    x = sample_ensemble(n, L, rng)
    return torch.from_numpy(x).unsqueeze(1)                    # [n, 1, L]


@torch.inference_mode()
def val_crps(model, cfg, vb):
    """Scale-normalized CRPS at the full trained tail (kmax patches)."""
    model.eval()
    k = cfg.train_kmax
    ctx = vb[..., :cfg.ctx_span].to(DEVICE)
    tgt = vb[..., cfg.ctx_span:cfg.ctx_span + k * cfg.patch].to(DEVICE)
    q = model.predict(ctx, k)                                  # [n,1,kP,9]
    lv = model.q_levels.to(DEVICE)
    err = tgt[..., None] - q
    pin = 2.0 * err * (lv - (err < 0).float())
    per = pin.mean(dim=(-1, -2))                               # [n,1]
    scale = RobustScaler(ctx.reshape(ctx.shape[0], -1)).scale.clamp_min(1e-6)
    model.train()
    return float((per.squeeze(1) / scale.squeeze(1)).mean())


def _wb(d, step=None):
    """Log to the active W&B run if there is one; silent no-op otherwise."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(d, step=step)
    except Exception:
        pass


def run_dir(cfg, ckpt_dir):
    d = os.path.join(ckpt_dir, "v10_runs", cfg.run_name)
    os.makedirs(d, exist_ok=True)
    return d


def train_one(cfg, ckpt_dir, corpus_dir, verbose=True):
    """Train one arm/seed to ``cfg.steps``, resuming from ``latest.pt`` if present.
    Returns the trained model and its validation log."""
    set_seed(cfg.seed)
    model = MiniTSFM2(cfg).to(DEVICE)
    opts = build_optimizers(model, cfg)
    base_lrs = [cfg.normuon_lr, cfg.lr]
    sampler = CorpusSampler(corpus_dir, cfg.pool, cfg.data_seed)
    vb = make_val_batch(cfg)
    d = run_dir(cfg, ckpt_dir)
    log = {"step": [], "val_crps": [], "train_loss": []}
    start = 0
    # ---- resume -------------------------------------------------------------
    latest = os.path.join(d, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); start = ck["step"]
        for o, s in zip(opts, ck["opts"]): o.load_state_dict(s)
        log = ck.get("log", log)
        print(f"[{cfg.run_name}] resumed at step {start}")
    if start >= cfg.steps:
        print(f"[{cfg.run_name}] already complete"); return model, log
    if verbose:
        print(f"[{cfg.run_name}] params={count_params(model) / 1e6:.2f}M "
              f"tokens/seq={cfg.n_ctx_tokens}+tail device={DEVICE} amp={AMP_DTYPE}")
    mask_rng = np.random.default_rng(cfg.data_seed + 50_000)   # paired masking too
    t0, model_train_loss = time.time(), 0.0
    model.train()
    for step in range(start, cfg.steps):
        x = torch.from_numpy(sample_series(sampler, step, cfg)).to(DEVICE, non_blocking=True)
        k, mp = sample_cpm_mask(cfg, cfg.batch, cfg.n_variates, mask_rng)
        mp = mp.to(DEVICE)
        set_lr(opts, base_lrs, lr_scale(step, cfg.steps, cfg.warmup_frac, cfg.decay_frac))
        for o in opts: o.zero_grad(set_to_none=True)
        if AMP_DTYPE is not None:
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss, *_ = model.forward_train(x, k, mp)
        else:
            loss, *_ = model.forward_train(x, k, mp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts: o.step()
        model_train_loss = 0.98 * model_train_loss + 0.02 * float(loss.detach()) if step > start else float(loss.detach())
        # ---- eval / logging ----------------------------------------------------
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            vc = val_crps(model, cfg, vb)
            log["step"].append(step + 1); log["val_crps"].append(vc)
            log["train_loss"].append(model_train_loss)
            _wb({"train_loss": model_train_loss, "val_crps": vc,
                 "lr_scale": lr_scale(step, cfg.steps, cfg.warmup_frac, cfg.decay_frac)},
                step=step + 1)
            if verbose:
                sps = (step + 1 - start) / max(1e-9, time.time() - t0)
                print(f"[{cfg.run_name}] step {step + 1}/{cfg.steps} "
                      f"loss={model_train_loss:.4f} valCRPS={vc:.4f} ({sps:.1f} it/s)")
        # ---- checkpoints ---------------------------------------------------------
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "opts": [o.state_dict() for o in opts],
                        "cfg": asdict(cfg), "log": log}, latest)
        if (step + 1) in cfg.gift_ckpts:
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "cfg": asdict(cfg)}, os.path.join(d, f"ckpt_{step + 1}.pt"))
    with open(os.path.join(d, "log.json"), "w") as f:
        json.dump({"cfg": asdict(cfg), "log": log}, f)
    return model, log
