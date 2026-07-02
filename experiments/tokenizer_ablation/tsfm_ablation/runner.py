# ============================================================================
# RUN MATRIX  ·  4 arms x 3 seeds x 30k steps (~2 A100-h per run at the
# default batch 64 x 8 variates; ~1 A100-day total). Fully resumable: rerun
# after any disconnect and it continues where it stopped. Each run:
# train -> dev-GIFT at final + 15k ckpt -> long-horizon probe -> results.json
# -> HF push. Use smoke=True to sanity-check the whole pipeline end-to-end in a
# few minutes (tiny steps, 2 dev datasets, 1 seed).
# ============================================================================
"""Orchestration: the run matrix and the optional LR-calibration sweep."""
from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict

import torch

from .config import make_arm
from .device import DEVICE
from .gift_eval import DEV_SETS, eval_gift_dev
from .hub import push_experiment_card, push_run
from .model import MiniTSFM2
from .probes import long_horizon_probe
from .train import run_dir, train_one


def make_smoke_runs():
    """Tiny 4-arm / 1-seed matrix for an end-to-end pipeline check."""
    runs = [make_arm(a, 0) for a in ("T0", "T1", "T2", "T3")]
    for c in runs:
        c.steps, c.eval_every, c.ckpt_every = 200, 50, 100
        c.gift_ckpts = (100, 200)
        c.batch, c.n_variates = 8, 2
        c.run_name += "_smoke"
    return runs


def run_matrix(runs, ckpt_dir, corpus_dir, gift_eval_dir, *, smoke=False,
               eval_15k=True, do_gift=True, dev_sets=DEV_SETS,
               hf_push=False, hf_repo=None,
               wandb_enable=False, wandb_project="tsfm-tokenizer-ablation-v10"):
    """Train + evaluate every cfg in ``runs``. Resumable; skips runs whose
    ``results.json`` already exists (unless ``smoke``)."""
    if smoke:
        dev_sets = dev_sets[:2]

    for cfg in runs:
        d = run_dir(cfg, ckpt_dir)
        res_path = os.path.join(d, "results.json")
        if os.path.exists(res_path) and not smoke:
            print(f"[skip] {cfg.run_name} — results.json exists")
            continue
        print("=" * 70); print("RUN", cfg.run_name)
        wb = None
        if wandb_enable:
            try:
                import wandb
                wb = wandb.init(project=wandb_project, name=cfg.run_name,
                                id=cfg.run_name, resume="allow",
                                config=asdict(cfg), reinit=True)
            except Exception as ex:
                print(f"[wandb] disabled for this run ({type(ex).__name__}: {ex})")
        model, log = train_one(cfg, ckpt_dir, corpus_dir)
        results = {"run_name": cfg.run_name, "cfg": asdict(cfg), "val_log": log}
        # ---- dev GIFT: final (30k, decayed) + 15k snapshot (undecayed) --------
        if do_gift:
            results["gift_final"] = eval_gift_dev(model, gift_eval_dir, dev_sets,
                                                  tag=f"{cfg.run_name}@final")
            ck15 = os.path.join(d, f"ckpt_{cfg.gift_ckpts[0]}.pt")
            if eval_15k and os.path.exists(ck15):
                m15 = MiniTSFM2(cfg).to(DEVICE)
                m15.load_state_dict(torch.load(ck15, map_location=DEVICE,
                                               weights_only=False)["model"])
                results["gift_15k"] = eval_gift_dev(m15, gift_eval_dir, dev_sets,
                                                    tag=f"{cfg.run_name}@15k")
                del m15
        # ---- long-horizon probe ----------------------------------------------
        results["probe"] = long_horizon_probe(model, total_h=(512 if smoke else 8192))
        print(f"probe pearson per bucket : {results['probe']['pearson']}")
        print(f"probe amp retention      : {results['probe']['amp_retention']}")
        # ---- persist + push ---------------------------------------------------
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        if hf_push:
            push_run(cfg, ckpt_dir, results, repo=hf_repo, enabled=True)
        if wb is not None:
            try:
                wb.summary.update({
                    "gm_crps_final": results.get("gift_final", {}).get("gm_crps"),
                    "gm_crps_long_season": results.get("gift_final", {}).get("gm_crps_long_season"),
                    "gm_crps_other": results.get("gift_final", {}).get("gm_crps_other"),
                    "gm_crps_15k": results.get("gift_15k", {}).get("gm_crps"),
                    "probe_pearson_last": results["probe"]["pearson"][-1],
                    "probe_amp_last": results["probe"]["amp_retention"][-1]})
                wb.finish()
            except Exception:
                pass
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    if hf_push:
        push_experiment_card(ckpt_dir, repo=hf_repo, enabled=True)
    print("\nRUN MATRIX COMPLETE")


def lr_sweep(ckpt_dir, corpus_dir, mults=(0.5, 1.0, 2.0), steps=2_000):
    """3-point LR sweep on the T0 control arm (~25 min on A100). No u-muP here,
    so the Toto-2 LRs don't transfer; this picks the multiplier. All arms share
    widths, so the T0-calibrated LRs transfer across arms to good approximation.
    Apply the winner by editing Cfg defaults (lr, normuon_lr) before the matrix."""
    sweep = {}
    for mult in mults:
        cfg = make_arm("T0", 0)
        cfg.steps, cfg.eval_every = steps, 500
        cfg.ckpt_every, cfg.gift_ckpts = 10**9, ()
        cfg.lr *= mult; cfg.normuon_lr *= mult
        cfg.run_name = f"lrsweep_x{mult}"
        _, log = train_one(cfg, ckpt_dir, corpus_dir)
        sweep[mult] = log["val_crps"][-1]
        print(f"lr multiplier x{mult}: final val CRPS = {sweep[mult]:.4f}")
    best = min(sweep, key=sweep.get)
    print(f"\nbest multiplier: x{best} -> set Cfg.lr={1e-3 * best:g}, "
          f"Cfg.normuon_lr={8e-4 * best:g}, then run the matrix.")
    if best != 1.0 and min(sweep[0.5], sweep[2.0]) < sweep[1.0]:
        print("(edge of grid won — consider extending the sweep one step further)")
    return sweep
