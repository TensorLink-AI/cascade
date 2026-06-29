"""Diversity sanity check for the cascade base generator (acceptance #5).

Prints summary statistics (mean, std, dominant spectral period, trend slope,
lag-1 autocorrelation) across a sample of the mixed corpus, plus a per-family
breakdown, to confirm multiple regimes appear rather than one degenerate shape.

    PYTHONPATH=/path/to/cascade python base_generator/tests/diversity_check.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("base_generator_div", REPO_DIR / "generator.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    t = np.arange(n)
    slope = np.polyfit(t, x, 1)[0] if n > 2 else 0.0
    xc = x - x.mean()
    # Dominant spectral period (samples), ignoring DC.
    if n > 4 and np.any(xc):
        mag = np.abs(np.fft.rfft(xc))
        mag[0] = 0.0
        k = int(np.argmax(mag))
        period = n / k if k > 0 else float("inf")
    else:
        period = float("inf")
    if xc.std() > 1e-12 and n > 2:
        ac1 = float(np.corrcoef(xc[:-1], xc[1:])[0, 1])
    else:
        ac1 = 0.0
    return {"mean": float(x.mean()), "std": float(x.std()), "slope": float(slope),
            "period": float(period), "ac1": ac1}


def main() -> None:
    m = _load_module()
    Generator = m.Generator

    # ---- mixed corpus sample ----
    gen = Generator(str(REPO_DIR), seed=0)
    sample = [np.asarray(a).ravel() for a in gen.generate(210)]
    S = [_stats(s) for s in sample]
    def col(k):
        return np.array([d[k] for d in S], dtype=np.float64)

    print("=== mixed corpus (n=210) ===")
    for k in ("mean", "std", "slope", "ac1"):
        v = col(k)
        print(f"  {k:6s}: min={v.min():+.4g}  median={np.median(v):+.4g}  max={v.max():+.4g}")
    finite_p = col("period")[np.isfinite(col("period"))]
    print(f"  period: p10={np.percentile(finite_p,10):.1f}  median={np.median(finite_p):.1f}  p90={np.percentile(finite_p,90):.1f} (samples)")
    lengths = np.array([s.size for s in sample])
    print(f"  length: distinct={len(set(lengths.tolist()))}  range=[{lengths.min()},{lengths.max()}]")
    # crude regime spread: std spans orders of magnitude, slopes both signs, AR memory varies
    stds = col("std")
    print(f"  std spans {stds.max()/max(stds.min(),1e-9):.1f}x; "
          f"slopes: {(col('slope')>0).sum()} up / {(col('slope')<0).sum()} down; "
          f"strong-AR(|ac1|>0.9): {(np.abs(col('ac1'))>0.9).sum()}")

    # ---- per-family breakdown (each drawn alone) ----
    print("\n=== per-family means of per-series stats (n=24 each, at full 2048) ===")
    print(f"  {'family':20s} {'mean':>9s} {'std':>9s} {'slope':>10s} {'period':>9s} {'ac1':>7s}")
    for fam in m._FAMILIES:
        g = Generator(str(REPO_DIR), seed=0)
        # generate only this family by overriding weights
        g._weights = {fam: 1.0}
        rows = [np.asarray(a).ravel() for a in g.generate(24)]
        st = [_stats(r) for r in rows]
        agg = {k: np.mean([d[k] for d in st]) for k in ("mean", "std", "slope")}
        per = np.array([d["period"] for d in st]); per = per[np.isfinite(per)]
        ac = np.mean([d["ac1"] for d in st])
        pmed = np.median(per) if per.size else float("inf")
        print(f"  {fam:20s} {agg['mean']:+9.3g} {agg['std']:9.3g} {agg['slope']:+10.3g} {pmed:9.1f} {ac:+7.3f}")


if __name__ == "__main__":
    main()
