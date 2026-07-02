"""Storage resolution: where checkpoints, the corpus, and the GIFT cache live.

Replaces the notebook's Colab-Drive mount with plain env vars so the same code
runs on a GPU pod, a /data mount, or a laptop:

* ``TSFM_CKPT_DIR`` — checkpoints, corpus, run logs, results (default: /data
  mount if present, else ./tsfm_ckpts).
* ``TSFM_GIFT_DIR`` — GIFT-Eval dataset cache (default: /data mount if present,
  else ./gifteval_data).
* ``TSFM_TIME_DIR`` — Real-TSF/TIME dataset path (read directly by ``time_eval``;
  aliases the ``TIME_DATASET`` env var that ``timebench`` expects).
"""
from __future__ import annotations

import os

CORPUS_SUBDIR = "corpus_v10"


def resolve_storage(ckpt_dir=None, gift_eval_dir=None):
    """Resolve (ckpt_dir, gift_eval_dir), creating ckpt_dir. Explicit args win,
    then env vars, then a /data mount, then a local dir next to the CWD."""
    ckpt_dir = ckpt_dir or os.environ.get("TSFM_CKPT_DIR")
    gift_eval_dir = gift_eval_dir or os.environ.get("TSFM_GIFT_DIR")
    have_data = os.path.isdir("/data")
    if ckpt_dir is None:
        ckpt_dir = "/data/tsfm_ckpts" if have_data else os.path.abspath("tsfm_ckpts")
    if gift_eval_dir is None:
        gift_eval_dir = "/data/gifteval_data" if have_data else os.path.abspath("gifteval_data")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir, gift_eval_dir


def corpus_dir(ckpt_dir):
    """The corpus lives under the checkpoint root so a pod keeps one mount."""
    return os.path.join(ckpt_dir, CORPUS_SUBDIR)
