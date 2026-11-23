"""Shared spec and fixture builders for W1 (the lane D measurement).

Copied from docs/plans/LANE_D_PLAN.md: the W1 row (forward-eligible counts
and latency per frequency; long-history inventory per context variant),
rev6.12 (a) (e = 48 h for every cadence) and (b) (fresh first, minimum fresh
share 0.5, never relaxed to meet the floors), rev6.11 (a) (max 5000 windows)
and rev6.14 (fresh = data time AND first-seen > T + forward_margin, margin
default 0; judging starts at T + eval_delay_hours, default 6). README.md
states the contract.

Nothing here imports pandas / pyarrow at import time; the mirror builder
imports them inside its body.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "laned_w1_measure.py"
UNIT_TESTS = REPO / "tests" / "unit" / "test_laned_w1_measure.py"
WORKFLOW = REPO / ".github" / "workflows" / "laned-w1-measure.yml"

FORGE_LABEL = "measured from forge mirror"
BOUND_LABEL = "snapshot-diff bound (upper bound on eligible counts)"
FREQ_BUCKETS = ("10s", "1min", "5min", "10min", "15min", "30min", "hourly", "daily", "slower")
CLOCKS = ("T+6h", "T+9h", "T+12h", "T+24h")
CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)
HORIZON = 64
EMBARGO_HOURS = 48
MIN_FRESH_SHARE = 0.5
MAX_WINDOWS = 5000
FORWARD_MARGIN_HOURS = 0
EVAL_DELAY_HOURS = 6
GATE_MIN_SOURCES = 20

DAY0 = dt.datetime(2026, 11, 17, tzinfo=dt.UTC).timestamp()


def load_script():
    """The measurement script as a module (it lives under scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("laned_w1_measure_accept", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def build_mirror(root: Path) -> Path:
    """A tiny forge mirror (``sources.yaml`` + ``data/<feed>/<day>.parquet``):

    * ``m1``: 1-min feed, one partition per day for 4 days, each written
      (mtime) at 23:30 of its day;
    * ``hr``: hourly, 2 panel rows of 5000 points, one history-rich partition;
    * ``wk``: weekly (slower than daily: the reader skips it);
    * ``dis``: disabled.
    """
    import pandas as pd

    root.mkdir(parents=True)
    (root / "sources.yaml").write_text(
        "- {id: m1, frequency: PT1M, domain: energy}\n"
        "- {id: hr, frequency: PT1H, domain: weather}\n"
        "- {id: wk, frequency: P1W, domain: econ}\n"
        "- {id: dis, frequency: PT1H, domain: econ, disabled: true}\n", encoding="utf-8")

    def put(feed, day, frame, mtime):
        d = root / "data" / feed
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{day}.parquet"
        frame.to_parquet(p, index=False)
        os.utime(p, (mtime, mtime))

    rng = np.random.default_rng(0)
    for k in range(4):
        t = DAY0 + k * 86400 + 60 * np.arange(1440)
        put("m1", iso(DAY0 + k * 86400)[:10],
            pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                          "value": rng.normal(size=t.size)}),
            DAY0 + k * 86400 + 23.5 * 3600)
    n = 5000
    t = DAY0 + 3 * 86400 - 3600 * n + 3600 * np.arange(n)
    frames = [pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                            "_panel_station": st, "temp": rng.normal(size=n)})
              for st in ("a", "b")]
    put("hr", "2026-11-20", pd.concat(frames, ignore_index=True), DAY0 + 3 * 86400 + 3600)
    tw = DAY0 - 604800 * np.arange(60)[::-1]
    put("wk", "2026-11-20", pd.DataFrame({"timestamp": pd.to_datetime(tw, unit="s", utc=True),
                                          "v": rng.normal(size=tw.size)}), DAY0 + 3 * 86400)
    put("dis", "2026-11-20", pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                                           "v": rng.normal(size=n)}), DAY0 + 3 * 86400)
    return root
