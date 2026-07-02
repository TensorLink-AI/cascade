# ============================================================================
# PROBES + AGGREGATION  ·  (a) Toto-2-Fig-10-style long-horizon stability on
# the superimposed periods-(500,100,20) signal: 8192-step block rollout,
# per-bucket Pearson r + amplitude retention. (b) Cross-run aggregate table
# with the tripwire checks (seed noise floor, 15k-vs-30k rank stability).
# ============================================================================
"""Long-horizon stability probe + the cross-run aggregate table and tripwires."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from .device import DEVICE


def make_multiscale(n, rng, periods=(500, 100, 20), noise=0.05):
    t = np.arange(n)
    x = np.zeros(n)
    for p in periods:
        x += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
    return (x + noise * rng.standard_normal(n)).astype(np.float32)


@torch.inference_mode()
def long_horizon_probe(model, n_series=16, total_h=8192, n_buckets=8, seed=123):
    """Block-rollout 8192 steps past a full-context window of the multiscale
    signal; report Pearson r and amplitude retention (pred_std/true_std) per
    horizon bucket. Collapse -> r falls and amp -> 0 in late buckets."""
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    L = cfg.ctx_span + total_h
    xs = np.stack([make_multiscale(L, rng) for _ in range(n_series)])
    x = torch.from_numpy(xs).unsqueeze(1).to(DEVICE)              # [n,1,L]
    ctx, tgt = x[..., :cfg.ctx_span], x[..., cfg.ctx_span:]
    pred = model.rollout(ctx, total_h)                            # [n,1,H] median
    bs = total_h // n_buckets
    pearson, amp = [], []
    for i in range(n_buckets):
        p = pred[..., i * bs:(i + 1) * bs].reshape(-1).float().cpu().numpy()
        t = tgt[..., i * bs:(i + 1) * bs].reshape(-1).float().cpu().numpy()
        pc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        pearson.append(round(pc, 3))
        amp.append(round(float(p.std() / (t.std() + 1e-9)), 3))
    return {"pearson": pearson, "amp_retention": amp}


# ---------------------------------------------------------------------------- aggregation
def load_results(ckpt_dir):
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "v10_runs", "*", "results.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def aggregate_table(rows):
    """Print the per-arm aggregate + tripwires; returns the pandas DataFrame."""
    import pandas as pd
    if not rows:
        print("no results yet"); return None
    df = pd.DataFrame([{
        "arm": r["run_name"].rsplit("_s", 1)[0], "seed": r["cfg"]["seed"],
        "gm_crps_30k": r.get("gift_final", {}).get("gm_crps"),
        "gm_crps_15k": r.get("gift_15k", {}).get("gm_crps"),
        "long_season_30k": r.get("gift_final", {}).get("gm_crps_long_season"),
        "other_30k": r.get("gift_final", {}).get("gm_crps_other"),
        "probe_r_last": (r.get("probe", {}).get("pearson") or [None])[-1],
        "probe_amp_last": (r.get("probe", {}).get("amp_retention") or [None])[-1],
    } for r in rows])
    agg = df.groupby("arm").agg(
        gm_crps_mean=("gm_crps_30k", "mean"), gm_crps_std=("gm_crps_30k", "std"),
        gm_crps_15k=("gm_crps_15k", "mean"),
        long_season=("long_season_30k", "mean"), other=("other_30k", "mean"),
        probe_r=("probe_r_last", "mean"), probe_amp=("probe_amp_last", "mean"),
        n=("seed", "count")).sort_values("gm_crps_mean")
    print(agg.round(4).to_string())

    # ---- tripwires ----------------------------------------------------------
    print("\n--- tripwire checks ---")
    arms = agg.index.tolist()
    sig = float(np.nanmax(agg["gm_crps_std"].values)) if len(agg) else float("nan")
    print(f"seed noise floor (max arm sigma): {sig:.4f}")
    for i in range(len(arms) - 1):
        gap = agg["gm_crps_mean"].iloc[i + 1] - agg["gm_crps_mean"].iloc[i]
        if not np.isfinite(sig):
            ok = "sigma unknown (single seed) — run more seeds before concluding"
        else:
            ok = "RESOLVED" if gap > 2 * sig else "UNRESOLVED (< 2*sigma) — add seeds/steps"
        print(f"  {arms[i]} vs {arms[i + 1]}: gap={gap:.4f} -> {ok}")
    r15 = agg.sort_values("gm_crps_15k").index.tolist()
    r30 = agg.sort_values("gm_crps_mean").index.tolist()
    if r15[:2] != r30[:2]:
        print(f"  RANK FLIP among top-2 between 15k {r15[:2]} and 30k {r30[:2]} "
              f"-> extend to 60k before promoting")
    else:
        print(f"  rank stable 15k->30k: {r30}")
    # T1 compression-tax gate
    if {"T0_fixed", "T1_pyr_ctx"} <= set(agg.index):
        tax = (agg.loc["T1_pyr_ctx", "gm_crps_mean"] / agg.loc["T0_fixed", "gm_crps_mean"] - 1)
        print(f"  H1 compression tax (T1 vs T0): {100 * tax:+.2f}% "
              f"{'(OK, <=2%)' if tax <= 0.02 else '(FAILS H1 gate — read T2 as gain minus tax)'}")
    return agg
