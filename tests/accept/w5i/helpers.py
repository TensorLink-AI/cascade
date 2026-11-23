"""Spec, fixtures and the reference draw (with backtest cells) for the W5i
tests. Copied from docs/plans/LANE_D_PLAN.md rev6.9 (a) and the W5i row, on
top of rev6.10 (b) / W5j. README.md states the contract.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

HORIZON = 64
ERA = 3600
ARM_BLOCK = 2700 * ERA
HASH_A = "0x" + hashlib.sha256(b"w5i boundary A").hexdigest()
HASH_B = "0x" + hashlib.sha256(b"w5i boundary B").hexdigest()
MIN_HISTORY = 32
T0 = 1_500_000_000
DAY = 86_400

# rev6.9 (a) horizon bins (points), inclusive.
BINS: dict[str, dict[str, tuple[int, int]]] = {}
for _f, _b in (
    (("D",), {"short": (7, 30), "medium": (31, 90), "long": (91, 180)}),
    (("W",), {"short": (4, 13), "medium": (14, 26), "long": (27, 52)}),
    (("M", "ME", "MS"), {"short": (3, 12), "medium": (13, 24), "long": (25, 36)}),
    (("Q", "QE", "QS"), {"short": (2, 4), "medium": (5, 8), "long": (9, 12)}),
    (("A", "Y", "YE"), {"short": (1, 2), "medium": (3, 4), "long": (5, 6)}),
):
    for _x in _f:
        BINS[_x] = dict(_b)

CLASS = {"D": "daily", "W": "weekly", "M": "monthly", "ME": "monthly", "MS": "monthly",
         "Q": "quarterly", "QE": "quarterly", "QS": "quarterly",
         "A": "yearly", "Y": "yearly", "YE": "yearly"}


def lane_d_cfg(**over):
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "horizon": HORIZON, "min_windows": 8, "min_clusters": 2,
           "max_source_share": 1.0, "bootstrap_B": 400, "d_margin": 0.01,
           "max_windows_per_settlement": 60, "min_fresh_window_share": 0.5,
           "backtest_weight_cap": 0.05, "backtest_ratio_floor": 0.85}
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: ERA)


def seed_int(block_hash: str) -> int:
    s = f"{block_hash}:lane_d_windows"
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def cell_key(cell: dict) -> str:
    return f"{cell['timeframe']}|{cell['bin']}"


def ts_of(n: int, step: int = DAY, start: int = T0) -> list[int]:
    return [int(start + i * step) for i in range(n)]


def bt_series(series_id: str, source: str, n: int) -> dict:
    return {"series_id": series_id, "source": source, "ts": ts_of(n)}


def fwd_series(series_id: str, source: str, n: int, step: int = DAY) -> dict:
    return {"series_id": series_id, "source": source, "ts": ts_of(n, step),
            "eligible": [True] * n, "e0": [True] * n}


def cell(freq: str, bin_: str, kind: str, members: list[dict]) -> dict:
    lo, hi = (HORIZON, HORIZON) if bin_ == "forward" else BINS[freq][bin_]
    return {"timeframe": freq, "bin": bin_, "kind": kind, "h_lo": lo, "h_hi": hi,
            "series": members}


# ── reference draw (W5j algorithm + backtest cells) ─────────────────────────


def _runs(flags):
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def capacity(c: dict) -> int:
    lo = int(c["h_lo"])
    if c["kind"] == "backtest":
        return sum(1 for s in c["series"] if len(s["ts"]) - lo >= MIN_HISTORY)
    return sum(r // lo for s in c["series"] for r in _runs(s["eligible"]))


def split_quota(caps: dict[str, int], total: int) -> dict[str, int]:
    quota = {k: 0 for k in caps}
    remaining = dict(sorted(caps.items()))
    want = int(total)
    while remaining and want > 0:
        share = max(1, want // len(remaining))
        key = min(remaining, key=lambda k: (remaining[k], k))
        cap = remaining.pop(key)
        take = min(cap, share if remaining else want)
        quota[key] = take
        want -= take
    return quota


def _record(s, c, i, h, emb):
    return {"window_id": f"{s['series_id']}:{s['ts'][i]}:h{h}", "series_id": s["series_id"],
            "source": s["source"], "timeframe": c["timeframe"], "bin": c["bin"],
            "kind": c["kind"], "h": h, "origin": s["ts"][i], "embargo_window": emb,
            "freq_class": CLASS.get(c["timeframe"], "subdaily")}


def reference_draw(cells, *, block_hash, min_windows=8, min_clusters=2, max_source_share=1.0,
                   fresh_share=0.5, max_windows=60):
    from cascade.validator.lane_d_frozen import cap_source_share

    caps = {cell_key(c): capacity(c) for c in cells}
    quota = split_quota(caps, max_windows)
    rng = np.random.default_rng(seed_int(block_hash))
    taken: dict[str, list[tuple[int, int]]] = {}
    fwd, bt = [], []
    for c in sorted(cells, key=cell_key):
        picked_here: set[str] = set()
        for _ in range(quota[cell_key(c)]):
            lo, hi = int(c["h_lo"]), int(c["h_hi"])
            h = lo if lo == hi else int(rng.integers(lo, hi + 1))
            if c["kind"] == "backtest":
                valid = [s for s in sorted(c["series"], key=lambda x: x["series_id"])
                         if s["series_id"] not in picked_here and len(s["ts"]) - h >= MIN_HISTORY]
                if not valid:
                    break
                s = valid[int(rng.integers(len(valid)))]
                picked_here.add(s["series_id"])
                bt.append(_record(s, c, len(s["ts"]) - h, h, False))
                continue
            valid = []
            for s in sorted(c["series"], key=lambda x: x["series_id"]):
                spans = taken.get(s["series_id"], [])
                for i in range(len(s["eligible"]) - h + 1):
                    if all(s["eligible"][i:i + h]) and not any(i < e and a < i + h
                                                               for a, e in spans):
                        valid.append((s, i))
            fresh = [(s, i) for s, i in valid if all(s["e0"][i:i + h])]
            group = fresh or valid          # W5j: fresh first, then embargo
            if not group:
                break
            s, i = group[int(rng.integers(len(group)))]
            taken.setdefault(s["series_id"], []).append((i, i + h))
            fwd.append(_record(s, c, i, h, not all(s["e0"][i:i + h])))
    seed = f"{block_hash}:lane_d_windows"
    kept_ids = {w["window_id"] for w in cap_source_share(
        [{"window_id": w["window_id"], "source": w["source"]} for w in fwd],
        max_share=max_source_share, seed=seed)}
    kept = [w for w in fwd if w["window_id"] in kept_ids]
    emb = [w for w in kept if w["embargo_window"]]
    other = [w for w in kept if not w["embargo_window"]]
    k = next(k for k in range(len(emb), -1, -1) if len(other) >= fresh_share * (len(other) + k))
    emb = emb[:k]
    forward = other + emb
    final = sorted(forward + bt, key=lambda w: w["window_id"])
    meets = (len(forward) >= min_windows
             and len({w["source"] for w in forward}) >= min_clusters)
    skipped = sorted(({"series_id": s["series_id"], "timeframe": c["timeframe"], "bin": c["bin"],
                       "reason": "backtest_short_history"}
                      for c in cells if c["kind"] == "backtest" for s in c["series"]
                      if len(s["ts"]) - int(c["h_lo"]) < MIN_HISTORY),
                     key=lambda r: (f"{r['timeframe']}|{r['bin']}", r["series_id"]))
    return {"capacity": caps, "quota": quota, "windows": final, "meets_floors": meets,
            "skipped": skipped}


FIELDS = ("window_id", "series_id", "source", "timeframe", "bin", "kind", "h", "origin",
          "embargo_window", "freq_class")


def view(result: dict) -> list[dict]:
    out = []
    for w in result["windows"]:
        missing = [f for f in FIELDS if f not in w]
        assert not missing, f"drawn window {w.get('window_id')!r} lacks {missing}"
        out.append({f: w[f] for f in FIELDS})
    return out


def draw(cells, cfg, block_hash):
    from cascade.validator.lane_d_draw import draw_windows

    return draw_windows(copy.deepcopy(cells), boundary_block_hash=block_hash, cfg=cfg)


# ── statistic ───────────────────────────────────────────────────────────────


def geo(crps, mase) -> float:
    """``cascade.validator.lane_d._geo``."""
    return math.sqrt(math.exp(float(np.mean(np.log(np.maximum(crps, 1e-12)))))
                     * math.exp(float(np.mean(np.log(np.maximum(mase, 1e-9))))))


def ref_gain(king, member, kinds, *, cap, floor) -> float:
    kc, km = (np.asarray(x, dtype=np.float64) for x in king)
    cc, cm = (np.asarray(x, dtype=np.float64) for x in member)
    f = np.array([k == "forward" for k in kinds])
    n_bt, n_fwd = int((~f).sum()), int(f.sum())
    c = 0.0 if n_bt == 0 else min(cap, n_bt / (n_bt + n_fwd))
    r_fwd = geo(cc[f], cm[f]) / geo(kc[f], km[f])
    log_r = (1.0 - c) * math.log(r_fwd)
    if n_bt:
        r_bt = geo(cc[~f], cm[~f]) / geo(kc[~f], km[~f])
        log_r += c * math.log(min(max(r_bt, floor), 1.0 / floor))
    return 1.0 - math.exp(log_r)


def judge(king, members, *, clusters, cfg, seed, kinds=None):
    import inspect

    from cascade.validator.lane_d import judge_cohort

    assert "kinds" in inspect.signature(judge_cohort).parameters, (
        "judge_cohort has no kinds= parameter (forward / backtest per window)")
    kw = {} if kinds is None else {"kinds": list(kinds)}
    return judge_cohort(king, members, clusters=clusters, cfg=cfg, seed=seed, **kw)
