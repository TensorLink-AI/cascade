"""Toto-2-4m-recipe context-tokenizer ablation, as reusable components.

The notebook this came from asked one question: which *context tokenizer* should
the full pretrain use? Horizon decoding is fixed patch-32 + a 9-quantile head in
every arm; only how history is compressed into tokens varies (fixed / pyramid /
adaptive). This package splits that pipeline into importable pieces:

    config       Cfg recipe + arm presets (T0..T3)      — no torch, import-cheap
    generators   TempoPFN-style synthetic prior ensemble — numpy only
    corpus       sharded corpus builder + paired sampler — numpy only
    tokenizers   fixed / pyramid / adaptive + scaler/RoPE
    model        MiniTSFM2 backbone + CPM mask sampler
    optim        NorMuon + AdamW split, WSD schedule
    train        per-run training loop + synthetic val CRPS
    gift_eval    GIFT-Eval dev-subset harness
    probes       long-horizon stability probe + aggregate table
    hub          Hugging Face Hub artifact/card uploads
    runner       run_matrix orchestration + LR sweep
    paths        env-based storage resolution

``config``, ``generators``, ``corpus``, and ``paths`` import without torch. The
rest is loaded lazily on first attribute access so, e.g., building the corpus
does not require the GPU stack.
"""
from __future__ import annotations

import importlib

# Cheap, torch-free surface — safe to import eagerly.
from .config import (
    ARMS,
    SEEDS,
    STD_LEVELS_LIST,
    Cfg,
    default_runs,
    make_arm,
)

__all__ = [
    "ARMS", "SEEDS", "STD_LEVELS_LIST", "Cfg", "default_runs", "make_arm",
    # lazily-resolved (torch / heavy) symbols:
    "MiniTSFM2", "count_params", "sample_cpm_mask",
    "build_tokenizer", "RobustScaler",
    "build_optimizers", "lr_scale", "NorMuon",
    "train_one", "val_crps",
    "eval_gift_dev", "quick_wiring_test", "DEV_SETS",
    "long_horizon_probe", "aggregate_table", "load_results",
    "run_matrix", "lr_sweep", "make_smoke_runs",
    "push_run", "push_experiment_card",
    "sample_ensemble", "set_official_sampler",
    "build_corpus", "CorpusSampler",
    "resolve_storage", "corpus_dir",
]

# Map lazily-exported names to their defining submodule.
_LAZY = {
    "MiniTSFM2": "model", "count_params": "model", "sample_cpm_mask": "model",
    "build_tokenizer": "tokenizers", "RobustScaler": "tokenizers",
    "build_optimizers": "optim", "lr_scale": "optim", "NorMuon": "optim",
    "train_one": "train", "val_crps": "train",
    "eval_gift_dev": "gift_eval", "quick_wiring_test": "gift_eval", "DEV_SETS": "gift_eval",
    "long_horizon_probe": "probes", "aggregate_table": "probes", "load_results": "probes",
    "run_matrix": "runner", "lr_sweep": "runner", "make_smoke_runs": "runner",
    "push_run": "hub", "push_experiment_card": "hub",
    "sample_ensemble": "generators", "set_official_sampler": "generators",
    "build_corpus": "corpus", "CorpusSampler": "corpus",
    "resolve_storage": "paths", "corpus_dir": "paths",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)
