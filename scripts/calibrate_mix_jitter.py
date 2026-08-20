"""Calibrate mix_jitter_alpha for cascade's jittered round mix (DEC-TB-0003 port).

Runs the exact allocator that will ship in cascade/validator/windows.py against
the REAL deployed pool snapshot, across candidate (n_windows, alpha, block)
settings. Reports per-domain p1-p99 slot ranges, effective domains
(exp Shannon entropy), P(effective < 4.0), min domain count, distinct
clusters (sources) served, and duplicate stats (none possible in cascade).
"""
from __future__ import annotations

import json
import os
import zlib
from collections import Counter, defaultdict

import numpy as np

POOL_DIR = "/root/cascade/_eval_pool"


# ---------------------------------------------------------------- allocator
# This is the reference implementation to be ported verbatim into
# cascade/validator/windows.py.

def _split(total: int, k: int) -> list[int]:
    if k == 0:
        return []
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _capped_jittered_split(
    n: int,
    caps: list[int],
    alpha: float | None,
    block: int,
    rng: np.random.Generator,
) -> list[int]:
    """Split ``n`` slots into ``len(caps)`` groups, Dirichlet-jittered around
    uniform (``alpha=None`` → exactly uniform), allocated in ``block``-slot
    units with a one-block floor per group, each group hard-capped at
    ``caps[i]`` (cascade serves one window per series — no duplicates), and
    overflow redistributed by remaining weight (water-filling)."""
    k = len(caps)
    if k == 0:
        return []
    n = min(int(n), int(sum(caps)))
    counts = [min(block, c) for c in caps]
    if sum(counts) > n:
        # Too small to floor every group (heat-sized draws on many groups):
        # keep a seeded subset of groups, like the TSBench small-n branch.
        m = max(1, n // block)
        counts = [0] * k
        order = rng.permutation(k)
        left = n
        for i in order[:m]:
            take = min(block if left >= block else left, caps[int(i)])
            counts[int(i)] = take
            left -= take
            if left <= 0:
                break
        return counts
    w = rng.dirichlet(np.full(k, float(alpha))) if alpha is not None else np.full(k, 1.0 / k)
    for _ in range(64):  # water-filling; open-set shrinks, terminates early
        left = n - sum(counts)
        if left <= 0:
            break
        open_ = [i for i in range(k) if counts[i] < caps[i]]
        if not open_:
            break
        if left < block:
            j = max(open_, key=lambda i: (counts[i], -i))
            counts[j] += min(left, caps[j] - counts[j])
            continue
        wsum = float(sum(w[i] for i in open_))
        raw = np.array(
            [left * (float(w[i]) / wsum if wsum > 0 else 1.0 / len(open_)) for i in open_]
        )
        nb = left // block
        base = np.floor(raw / block).astype(int)
        rem = int(nb - base.sum())
        if rem > 0:
            frac = raw / block - base
            order = np.argsort(-frac, kind="stable")
            base[order[:rem]] += 1
        progressed = False
        for pos, i in enumerate(open_):
            add = min(int(base[pos]) * block, caps[i] - counts[i])
            counts[i] += add
            progressed = progressed or add > 0
        if not progressed:
            j = open_[0]
            counts[j] += min(block, caps[j] - counts[j])
    return counts


def _bag_keep(salt: int, key: str, frac: float) -> bool:
    return (zlib.crc32(f"{salt}:{key}".encode()) % 10_000) < int(frac * 10_000)


def draw_round(
    md: dict,
    n: int,
    rng: np.random.Generator,
    *,
    alpha: float | None,
    block: int,
    bag_frac: float,
    keep_frac: float,
) -> list[str]:
    """One round's series-id picks — the full sampler (bag → domain Dirichlet →
    class rotation → capped even class split → without-replacement picks)."""
    ids = sorted(md)
    if bag_frac < 1.0:
        salt = int(rng.integers(0, 2**32))
        bagged = [s for s in ids if _bag_keep(salt, s, bag_frac)]
        if bagged:
            ids = bagged
    # The class tier only exists when the pool carries dgp_class labels;
    # source-as-class would fragment into ~1000 micro-cells and the activation
    # cap would starve rounds below min_windows (measured: 67 of 256).
    has_classes = any(md[s].get("dgp_class") for s in ids)
    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in ids:
        m = md[s]
        cls = str(m.get("dgp_class")) if has_classes else "_all"
        by_cell[(str(m.get("domain", "unknown")), cls)].append(s)
    domains = sorted({d for d, _ in by_cell})
    dom_caps = [sum(len(v) for (d, _), v in by_cell.items() if d == dom) for dom in domains]
    per_domain = _capped_jittered_split(n, dom_caps, alpha, block, rng)
    picks: list[str] = []
    for dom, n_dom in zip(domains, per_domain, strict=True):
        if n_dom == 0:
            continue
        classes = sorted({c for d, c in by_cell if d == dom})
        if has_classes and keep_frac < 1.0 and len(classes) > 1:
            mask = rng.random(len(classes)) < keep_frac
            kept = [c for c, k in zip(classes, mask, strict=True) if k]
            classes = kept or [classes[int(rng.integers(0, len(classes)))]]
            # Rotation must never make the domain quota unfillable: extend the
            # kept set (seeded order over the dropped classes) until active
            # capacity covers n_dom — cascade cannot duplicate-fill.
            capacity = sum(len(by_cell[(dom, c)]) for c in classes)
            if capacity < n_dom:
                dropped = [c for c in sorted({c for d, c in by_cell if d == dom})
                           if c not in classes]
                order = rng.permutation(len(dropped))
                for i in order:
                    c = dropped[int(i)]
                    classes.append(c)
                    capacity += len(by_cell[(dom, c)])
                    if capacity >= n_dom:
                        break
                classes.sort()
        if has_classes:
            # Activation cap: no more classes than can each field one block —
            # but only while capacity still covers the quota.
            cap = max(1, n_dom // block)
            if len(classes) > cap:
                wts = np.array(
                    [min(len(by_cell[(dom, c)]), block) for c in classes], dtype=float
                )
                sel = rng.choice(len(classes), size=cap, replace=False, p=wts / wts.sum())
                trimmed = [classes[int(i)] for i in sorted(int(j) for j in sel)]
                if sum(len(by_cell[(dom, c)]) for c in trimmed) >= n_dom:
                    classes = trimmed
        cls_caps = [len(by_cell[(dom, c)]) for c in classes]
        per_class = _capped_jittered_split(n_dom, cls_caps, None, block, rng)
        for cls, n_cls in zip(classes, per_class, strict=True):
            pool = by_cell[(dom, cls)]
            if n_cls >= len(pool):
                picks.extend(pool)
            else:
                idx = rng.choice(len(pool), size=n_cls, replace=False)
                picks.extend(pool[int(i)] for i in idx)
    return picks


# --------------------------------------------------------------------- sim

def load_md() -> dict:
    latest = sorted(
        os.listdir(POOL_DIR), key=lambda d: os.path.getmtime(f"{POOL_DIR}/{d}")
    )[-1]
    print(f"pool snapshot: {latest}")
    return json.load(open(f"{POOL_DIR}/{latest}/metadata.json"))


def simulate(md, *, n, alpha, block, bag_frac, keep_frac, rounds, seed0=1):
    domains = sorted({str(v.get("domain", "unknown")) for v in md.values()})
    counts_hist = {d: [] for d in domains}
    eff_hist, minc_hist, nclus_hist, npicks_hist = [], [], [], []
    for r in range(rounds):
        rng = np.random.default_rng(seed0 + r)
        picks = draw_round(
            md, n, rng, alpha=alpha, block=block, bag_frac=bag_frac, keep_frac=keep_frac
        )
        c = Counter(str(md[s].get("domain", "unknown")) for s in picks)
        tot = sum(c.values())
        p = np.array([v / tot for v in c.values() if v])
        eff = float(np.exp(-(p * np.log(p)).sum()))
        for d in domains:
            counts_hist[d].append(c.get(d, 0))
        eff_hist.append(eff)
        minc_hist.append(min(c.get(d, 0) for d in domains))
        nclus_hist.append(len({md[s].get("source", s) for s in picks}))
        npicks_hist.append(len(picks))
    eff = np.array(eff_hist)
    print(
        f"n={n} alpha={alpha} block={block} bag={bag_frac} keep={keep_frac} "
        f"rounds={rounds}"
    )
    print(
        f"  served/round: {np.mean(npicks_hist):.0f} "
        f"(min {min(npicks_hist)}, max {max(npicks_hist)})"
    )
    for d in domains:
        a = np.array(counts_hist[d])
        print(
            f"  {d:<13} mean {a.mean():7.1f}  p1 {np.percentile(a,1):6.0f}  "
            f"p99 {np.percentile(a,99):6.0f}"
        )
    print(
        f"  effective domains: mean {eff.mean():.2f}  min {eff.min():.2f}  "
        f"P(<4.0) = {(eff < 4.0).mean():.5f}"
    )
    print(
        f"  min domain count: min {min(minc_hist)}  "
        f"clusters served: min {min(nclus_hist)} mean {np.mean(nclus_hist):.0f}"
    )
    print()


if __name__ == "__main__":
    md = load_md()
    doms = Counter(str(v.get("domain")) for v in md.values())
    print("pool:", dict(doms.most_common()), "total", len(md))
    print()
    R = int(os.environ.get("ROUNDS", "2000"))
    # A: today's config — n=2000 (84% of pool): how much jitter survives?
    simulate(md, n=2000, alpha=4.0, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # B: today's config, no jitter (baseline = uniform permutation proportions)
    simulate(md, n=2000, alpha=None, block=8, bag_frac=1.0, keep_frac=1.0, rounds=200)
    # C: n=1200 — room appears
    simulate(md, n=1200, alpha=4.0, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # D: n=1000, alpha sweep
    for a in (2.0, 4.0, 8.0):
        simulate(md, n=1000, alpha=a, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # E: heat-sized draw (256)
    simulate(md, n=256, alpha=4.0, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # F: block sensitivity at n=1200
    simulate(md, n=1200, alpha=4.0, block=4, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # G: 2x pool growth scenario, n=2000
    md2 = {**md, **{f"{k}__x2": v for k, v in md.items()}}
    simulate(md2, n=2000, alpha=4.0, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
    # H: future pool WITH dgp_class labels (synthetic: hash source -> 36 classes)
    md3 = {k: {**v, "dgp_class": f"cls{zlib.crc32(str(v.get('source', k)).encode()) % 36}"}
           for k, v in md.items()}
    simulate(md3, n=1200, alpha=4.0, block=8, bag_frac=0.7, keep_frac=0.7, rounds=R)
