"""Forge multivariate packing (DEC-CA-0041 Phase 2): a catalog entry's
``mv_channels`` columns pack into one (C, L) series when armed, and project to
one univariate channel until then. Exercises ``TsbenchForgeSource._extract_values``
directly with a synthetic panel-row frame (no forge mirror needed).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cascade.pool.source import HarvestContext
from cascade.pool.sources.tsbench_forge import TsbenchForgeSource

# as_of well after the frame so the freshness cutoff keeps the whole series
CTX = HarvestContext(as_of=dt.date(2027, 3, 1), horizon=16, max_series=100)


def _frame(n=400):
    ts = pd.date_range("2026-01-01", periods=n, freq="D")
    r = np.arange(n)
    return pd.DataFrame({
        "timestamp": ts.astype(str),
        "covid": np.sin(r / 7.0) + 10.0,      # small scale
        "flu": np.cos(r / 7.0) + 100.0,       # mid scale
        "rsv": np.linspace(0.0, 5.0, n) + 1.0,  # different shape
    })


def test_forge_packs_mv_channels_to_C_L():
    src = TsbenchForgeSource(forge_dir="/tmp", mv_pack=True)
    vals = src._extract_values(pd, _frame(), [], CTX, columns=["covid", "flu", "rsv"])
    assert vals is not None
    assert vals.ndim == 2 and vals.shape[0] == 3      # (C=3, L)
    # channels keep their own scale (not collapsed / not averaged)
    assert vals[1].mean() > vals[0].mean() > 1.0


def test_forge_projects_to_one_channel_until_armed():
    src = TsbenchForgeSource(forge_dir="/tmp", mv_pack=False)
    vals = src._extract_values(pd, _frame(), [], CTX, columns=["covid"])
    assert vals is not None and vals.ndim == 1        # univariate, inert


def test_forge_missing_tagged_channel_drops_group():
    src = TsbenchForgeSource(forge_dir="/tmp", mv_pack=True)
    vals = src._extract_values(pd, _frame(), [], CTX, columns=["covid", "not_a_col"])
    assert vals is None                                # a tagged channel must be present


def test_forge_untagged_still_picks_densest_1d():
    src = TsbenchForgeSource(forge_dir="/tmp")
    vals = src._extract_values(pd, _frame(), [], CTX, columns=None)
    assert vals is not None and vals.ndim == 1        # unchanged default behaviour
