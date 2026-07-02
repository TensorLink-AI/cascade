# ============================================================================
# CONFIG  ·  Toto-2-4m-recipe clone + tokenizer switch + run matrix
# 4m recipe anchors: d=256, 4 heads, 4 layers, patch 32, ctx 4096, quantile
# head (9 levels, pinball), CPM (c_max=16, p_max<=0.4), NorMuon+AdamW split,
# arcsinh robust scaling, variate attention in the LAST layer, index RoPE.
# Deviations are flagged inline (V=8 default vs paper's 32 for single-GPU;
# CPM restricted to the fine band + tail so arms stay comparable; no u-muP,
# so LRs are re-calibrated once on T0 and shared).
# ============================================================================
"""Experiment configuration: the ``Cfg`` recipe dataclass and the arm presets.

This is the one module with no torch/numpy dependency, so it can be imported
anywhere (CLI arg parsing, tests, tooling) without pulling in the heavy stack.
"""
from __future__ import annotations

from dataclasses import dataclass

# 9 pinball quantile levels shared by the head, the loss, and the eval CRPS.
STD_LEVELS_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Cfg:
    run_name: str = "T0_fixed"
    # --- tokenizer ---------------------------------------------------------
    tokenizer: str = "fixed"          # fixed | pyramid | adaptive
    patch: int = 32                   # fine/inner patch (matches Toto-2 P=32)
    ctx_span: int = 4096              # raw history steps consumed
    pyramid_levels: tuple = ()        # ((span, plen), ...) oldest->newest; last plen == patch
    adaptive_hist_tokens: int = 96    # equal-surprise segments over pre-fine history
    fine_span: int = 1024             # fine band (patch-32 tokens) = CPM zone
    # --- model (Toto-2 4m) --------------------------------------------------
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    # --- CPM ----------------------------------------------------------------
    train_kmin: int = 4               # tail-mask length in patches
    train_kmax: int = 32
    cpm_cmax: int = 16                # max interior contiguous spans
    cpm_pmax: float = 0.4             # max masked fraction of (zone + tail) tokens
    cpm_span_max: int = 4             # max interior span length (patches)
    # --- train ---------------------------------------------------------------
    steps: int = 30_000
    batch: int = 64
    n_variates: int = 8               # paper: 32 — raise if your GPU allows
    lr: float = 1e-3                  # AdamW group (embed/head/bias/norm)
    normuon_lr: float = 8e-4          # NorMuon group (transformer matrices)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.03
    decay_frac: float = 0.10          # WSD: linear decay tail
    seed: int = 0                     # model init seed
    data_seed: int = 0                # corpus stream seed (paired across arms)
    pool: str = "short"              # short | long corpus pool
    # --- eval / io -----------------------------------------------------------
    eval_every: int = 500
    ckpt_every: int = 2_500
    gift_ckpts: tuple = (15_000, 30_000)

    @property
    def zone_tokens(self):  return self.fine_span // self.patch

    @property
    def n_ctx_tokens(self):
        if self.tokenizer == "fixed":    return self.ctx_span // self.patch
        if self.tokenizer == "pyramid":  return sum(s // p for s, p in self.pyramid_levels)
        return self.adaptive_hist_tokens + self.zone_tokens


# ---------------------------------------------------------------------------- arm presets
def make_arm(name: str, seed: int) -> Cfg:
    """One (arm, seed) recipe. Arms share every hyperparameter except the
    context tokenizer, so a run is a controlled measurement of the tokenizer."""
    base = dict(seed=seed, data_seed=seed)
    if name == "T0":   # fixed-32 control — 4096 steps / 128 tokens
        return Cfg(run_name=f"T0_fixed_s{seed}", tokenizer="fixed",
                   ctx_span=4096, pool="short", **base)
    if name == "T1":   # pyramid iso-context — 4096 steps / 44 tokens
        return Cfg(run_name=f"T1_pyr_ctx_s{seed}", tokenizer="pyramid", ctx_span=4096,
                   pyramid_levels=((2048, 512), (1024, 128), (1024, 32)),
                   fine_span=1024, pool="short", **base)
    if name == "T2":   # pyramid iso-token — 16384 steps / 128 tokens
        return Cfg(run_name=f"T2_pyr_tok_s{seed}", tokenizer="pyramid", ctx_span=16384,
                   pyramid_levels=((8192, 512), (4096, 128), (3072, 64), (1024, 32)),
                   fine_span=1024, pool="long", **base)
    if name == "T3":   # adaptive equal-surprise — 16384 steps / ~128 tokens
        return Cfg(run_name=f"T3_adaptive_s{seed}", tokenizer="adaptive", ctx_span=16384,
                   adaptive_hist_tokens=96, fine_span=1024, pool="long", **base)
    raise ValueError(name)


ARMS = ("T0", "T1", "T2", "T3")
SEEDS = (0, 1, 2)


def default_runs(arms=ARMS, seeds=SEEDS) -> list[Cfg]:
    """The full ablation matrix: every arm x every seed."""
    return [make_arm(a, s) for a in arms for s in seeds]
