# ============================================================================
# CORPUS  ·  pre-generate the synthetic corpus (resumable, sharded)
# ----------------------------------------------------------------------------
# Two pools:
#   short : len = 4096 + 1024   -> arms T0/T1  (ctx 4096  + train tail)
#   long  : len = 16384 + 1024  -> arms T2/T3  (ctx 16384 + train tail)
# Training samples random crops + affine/noise augmentation, so heavy corpus
# reuse is fine (TempoPFN itself reuses a 10M-series corpus for 4M iters).
# Paired comparisons: arms sharing a pool + data_seed see IDENTICAL batches.
# ============================================================================
"""Sharded on-disk corpus builder + a deterministic paired-batch sampler."""
from __future__ import annotations

import glob
import os

import numpy as np

from .generators import get_sampler

SHORT_LEN, LONG_LEN = 4096 + 1024, 16384 + 1024
N_SHORT, N_LONG = 40_000, 10_000          # ~0.8 GB + ~0.7 GB float32 on disk
SHARD = 2_000

# Default pool geometry: (n_series, length, seed0) keyed by pool name.
POOLS = {
    "short": (N_SHORT, SHORT_LEN, 1000),
    "long": (N_LONG, LONG_LEN, 9000),
}


def build_pool(corpus_dir, name, n_series, length, seed0=1234):
    """Write ``name``'s shards under ``corpus_dir``; skips shards already on disk."""
    os.makedirs(corpus_dir, exist_ok=True)
    n_shards = (n_series + SHARD - 1) // SHARD
    sampler = get_sampler()
    for si in range(n_shards):
        path = os.path.join(corpus_dir, f"{name}_{si:04d}.npy")
        if os.path.exists(path):
            continue
        rng = np.random.default_rng(seed0 + si)
        n = min(SHARD, n_series - si * SHARD)
        x = sampler(n, length, rng)
        np.save(path, x.astype(np.float32))
        print(f"[corpus] {name} shard {si + 1}/{n_shards} written ({n}x{length})")
    print(f"[corpus] {name}: complete ({n_series} series x {length})")


def build_corpus(corpus_dir, pools=("short", "long")):
    """Build every requested pool with the default geometry from ``POOLS``."""
    for name in pools:
        n_series, length, seed0 = POOLS[name]
        build_pool(corpus_dir, name, n_series, length, seed0=seed0)


class CorpusSampler:
    """Memory-maps corpus shards; deterministic paired sampling per (data_seed, step)."""

    def __init__(self, corpus_dir, name, data_seed):
        paths = sorted(glob.glob(os.path.join(corpus_dir, f"{name}_*.npy")))
        assert paths, f"no corpus shards for pool '{name}' in {corpus_dir} — run the corpus builder"
        self.shards = [np.load(p, mmap_mode="r") for p in paths]
        self.counts = np.array([s.shape[0] for s in self.shards])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.total = int(self.counts.sum())
        self.length = self.shards[0].shape[1]
        self.data_seed = data_seed

    def batch(self, step, n, crop_len):
        """[n, crop_len] float32, deterministic in (data_seed, step)."""
        rng = np.random.default_rng((self.data_seed * 1_000_003 + step) % (2**63))
        idx = rng.integers(0, self.total, n)
        starts = rng.integers(0, self.length - crop_len + 1, n)
        out = np.empty((n, crop_len), np.float32)
        for i, (j, s) in enumerate(zip(idx, starts)):
            si = int(np.searchsorted(self.offsets, j, side="right") - 1)
            out[i] = self.shards[si][j - self.offsets[si], s:s + crop_len]
        # augmentation: sign flip, log-scale, offset, small noise
        sgn = np.where(rng.random((n, 1)) < 0.2, -1.0, 1.0)
        a = sgn * np.exp(rng.uniform(np.log(0.5), np.log(2.0), (n, 1)))
        b = rng.normal(0, 1.0, (n, 1)) * (np.abs(out).mean(1, keepdims=True) + 1e-3)
        out = a * out + b + rng.normal(0, 0.01, (n, 1)) * out.std(1, keepdims=True) * rng.standard_normal(out.shape)
        return out.astype(np.float32)


def sample_series(sampler, step, cfg):
    """[B, V, ctx_span + kmax*patch] — variates drawn independently (see notes)."""
    L = cfg.ctx_span + cfg.train_kmax * cfg.patch
    flat = sampler.batch(step, cfg.batch * cfg.n_variates, L)
    return flat.reshape(cfg.batch, cfg.n_variates, L)
