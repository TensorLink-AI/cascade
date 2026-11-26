#!/usr/bin/env python3
"""Lane D null false-positive rate: how often a challenger with NO real
improvement "clears" ``d_margin`` — to calibrate ``d_margin`` to a per-era
target.

REPORT-ONLY. Nothing here is on a consensus path. Two stages, kept apart so
the GPU never runs the statistics and the statistics never need a GPU:

``score`` (GPU, an approved pod)
    Load ONE pinned pool snapshot directory, draw ONE window list
    (:func:`cascade.validator.calibration.load_pool_series` +
    :func:`~cascade.validator.calibration.calib_draw`, which is
    :func:`cascade.validator.windows.horizon_draw`, h = 64), score every
    checkpoint on exactly that list with
    :func:`cascade.validator.evaluator.evaluate_checkpoint` (quantile head +
    fixed batch 32 by default — the live v2 settlement, DEC-CA-0067/0068),
    collapse channels per window (live ``mv_score``), and write the per-window
    rows to ``PREFIX.npz`` + ``PREFIX.json`` (snapshot sha256, host, command
    line, checkpoint digests, the npz's own sha256).

``stats`` (CPU only, consumes the rows file)
    For every ordered null pair (A as king, B as challenger; a model against
    itself is refused):

    1. resampled estimate — ``R`` sets of ``n`` windows, without replacement
       inside a set, each holding at least ``min_clusters`` distinct sources
       (draws under the floor are rejected and redrawn, never scored; a pool
       that cannot reach the floor is reported "below floor: …");
    2. ``--subhourly-only`` restricts the pool to sub-hourly windows (per-window
       ``freq`` metadata), same floor rule;
    3. disjoint spread check — the windows are partitioned into ~N/n disjoint
       sets (no window reused), the null LCB is computed on each and the SD /
       IQR of disjoint vs resampled LCBs and their ratio are reported. No FPR
       is reported from the disjoint sets;
    4. confirmation duel — pairs of NON-overlapping resampled sets; a false
       win only if BOTH clear the margin;
    5. per-entry FPR on a margin grid (Wilson 95 % CIs), the ratio-corrected
       FPR (margins scaled by the disjoint/resampled SD ratio when > 1), null
       LCB quantiles 50/90/95/99, the confirmation-duel FPR, the per-era FPR
       ``1 - (1 - p_conf)^k`` for k in {1, 8, K}, and the ``d_margin`` needed
       for a per-era FPR <= {1 %, 5 %, 10 %} — at every ``min_clusters`` floor.

    The LCB is :func:`cascade.eval.bootstrap.paired_bootstrap_lcb_aggregated`
    (``clusters`` = source, ``wql_mode="geomean"``; sign ``(king - chal) /
    king``, positive = the challenger looks better); the point gain is the
    same relative difference of :func:`cascade.eval.scoring.global_geomean`.
    A "clear" is ``LCB >= margin`` AND point gain > 0. King and challenger rows
    are always the same windows in the same order.

Every output JSON records the command line, the input rows file sha256, the
snapshot sha256, the host, a UTC timestamp and the seeds. Anything that cannot
be computed is the string "below floor: …" or "not run: …" — never a number.

Usage:
  null_fpr.py score --pool-dir SNAP --checkpoint king=CK1 --checkpoint b=CK2 \\
      --out rows/null [--n-windows 0] [--seed 0] [--device cuda]
  null_fpr.py stats --rows rows/null.npz --era-entries K --out null_fpr.json \\
      [--n 1000] [--R 500] [--min-clusters 20 30 40] [--subhourly-only]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import shlex
import socket
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

ROWS_SCHEMA = "cascade-null-fpr-rows/1"
STATS_SCHEMA = "cascade-null-fpr-stats/1"
DEFAULT_HORIZON = 64
DEFAULT_CONTEXT = 4096
DEFAULT_N = 1000
DEFAULT_R = 500
DEFAULT_FLOORS = (20, 30, 40)
DEFAULT_ALPHA = 0.05          # chain.toml [scoring] bootstrap_alpha
DEFAULT_B = 10_000            # chain.toml [scoring] bootstrap_B
DEFAULT_MARGINS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03,
                   0.04, 0.05)
DEFAULT_TARGETS = (0.01, 0.05, 0.10)
NULL_QUANTILES = (0.50, 0.90, 0.95, 0.99)
WILSON_Z = 1.959963984540054
MAX_DRAW_TRIES_PER_SET = 200
# pandas-style frequency units shorter than one hour.
SUBHOURLY_UNITS = frozenset({"S", "s", "T", "min", "L", "ms", "U", "us", "N", "ns"})


# ── provenance ───────────────────────────────────────────────────────────────


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path | str) -> str:
    """sha256 over every regular file under ``root``: sorted relative POSIX
    path, NUL, size, NUL, bytes. A content digest of an UNPACKED directory
    (not the published tar's sha256 — pass that separately when known)."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + str(p.stat().st_size).encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def provenance(argv: Sequence[str] | None) -> dict:
    args = list(sys.argv[1:] if argv is None else argv)
    return {
        "command_line": shlex.join([sys.argv[0] or "null_fpr.py", *args]),
        "host": socket.gethostname(),
        "timestamp_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def sub_seed(*parts) -> int:
    """Stable 64-bit seed from string parts (blake2b), recorded verbatim."""
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


# ── rows container ───────────────────────────────────────────────────────────


class Rows:
    """Per-window rows for M models on one shared window list (same order for
    every model). ``cluster`` is the KOTH cluster key (``source``, else
    ``__series:<id>``); ``freq`` the pool frequency string ('' when absent)."""

    def __init__(self, *, series_id, cluster, source, domain, freq, models: dict,
                 quantile_levels: Sequence[float]):
        self.series_id = np.asarray(series_id, dtype=str)
        self.cluster = np.asarray(cluster, dtype=str)
        self.source = np.asarray(source, dtype=str)
        self.domain = np.asarray(domain, dtype=str)
        self.freq = np.asarray(freq, dtype=str)
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        n = self.series_id.shape[0]
        for name, arr in (("cluster", self.cluster), ("source", self.source),
                          ("domain", self.domain), ("freq", self.freq)):
            if arr.shape != (n,):
                raise ValueError(f"rows: {name} shape {arr.shape}; expected ({n},)")
        self.models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        ref_abs = None
        for name, (q, a, m) in models.items():
            q = np.asarray(q, dtype=np.float64)
            a = np.asarray(a, dtype=np.float64)
            m = np.asarray(m, dtype=np.float64)
            if q.shape != (n, len(self.quantile_levels)) or a.shape != (n,) or m.shape != (n,):
                raise ValueError(f"rows: model {name!r} component shapes {q.shape}/{a.shape}/"
                                 f"{m.shape} do not match {n} windows")
            if ref_abs is not None and not np.allclose(a, ref_abs):
                raise ValueError(f"rows: model {name!r} abs_target differs — windows unpaired")
            ref_abs = a if ref_abs is None else ref_abs
            self.models[name] = (q, a, m)

    def __len__(self) -> int:
        return int(self.series_id.shape[0])

    def window_scores(self, name: str) -> list:
        """The model's rows as :class:`cascade.eval.scoring.WindowScore` s, in
        row order (one per window, ``channel`` 0)."""
        from cascade.eval.scoring import WindowScore

        q, a, m = self.models[name]
        return [WindowScore(series_id=str(self.series_id[i]), mase=float(m[i]),
                            qloss_per_q=q[i], abs_target=float(a[i]),
                            quantile_levels=self.quantile_levels, channel=0,
                            domain=str(self.domain[i]),
                            source=str(self.source[i]) or None)
                for i in range(len(self))]


def save_rows(prefix: Path | str, rows: Rows, manifest: dict) -> tuple[Path, Path]:
    """``PREFIX.npz`` (arrays) + ``PREFIX.json`` (manifest incl. the npz sha256)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows.models)
    arrays = {"series_id": rows.series_id, "cluster": rows.cluster, "source": rows.source,
              "domain": rows.domain, "freq": rows.freq,
              "quantile_levels": np.asarray(rows.quantile_levels, dtype=np.float64)}
    for i, name in enumerate(names):
        q, a, m = rows.models[name]
        arrays[f"m{i}_qloss"], arrays[f"m{i}_abs"], arrays[f"m{i}_mase"] = q, a, m
    npz = prefix.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    doc = dict(manifest, schema=ROWS_SCHEMA, models=names, n_windows=len(rows),
               rows_npz=npz.name, rows_npz_sha256=file_sha256(npz))
    js = prefix.with_suffix(".json")
    js.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return npz, js


def load_rows(npz_path: Path | str) -> tuple[Rows, dict, str]:
    """``(rows, manifest, npz_sha256)``; refuses a manifest whose recorded npz
    sha256 does not match the file."""
    npz_path = Path(npz_path)
    manifest = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("schema") != ROWS_SCHEMA:
        raise ValueError(f"{npz_path}: manifest schema {manifest.get('schema')!r} != {ROWS_SCHEMA}")
    sha = file_sha256(npz_path)
    if manifest.get("rows_npz_sha256") != sha:
        raise ValueError(f"{npz_path}: sha256 {sha} != manifest {manifest.get('rows_npz_sha256')}")
    with np.load(npz_path, allow_pickle=False) as z:
        models = {name: (z[f"m{i}_qloss"], z[f"m{i}_abs"], z[f"m{i}_mase"])
                  for i, name in enumerate(manifest["models"])}
        rows = Rows(series_id=z["series_id"], cluster=z["cluster"], source=z["source"],
                    domain=z["domain"], freq=z["freq"], models=models,
                    quantile_levels=z["quantile_levels"])
    return rows, manifest, sha


# ── score stage (GPU) ────────────────────────────────────────────────────────


def rows_from_scores(windows: list, scores_by_model: dict[str, list]) -> Rows:
    """Build :class:`Rows` from per-model WindowScore lists scored on the same
    ``windows``. Channels are collapsed per window (live ``mv_score``); every
    model must emit the same series_id order (pairing) or this raises."""
    from cascade.eval.koth import _window_clusters
    from cascade.eval.scoring import collapse_channels_by_window, stack_components

    md = {w.series_id: dict(w.metadata or {}) for w in windows}
    ref_ids = None
    comps: dict[str, tuple] = {}
    ref_scores = None
    for name, scores in scores_by_model.items():
        col = collapse_channels_by_window(list(scores))
        ids = [s.series_id for s in col]
        if ref_ids is None:
            ref_ids, ref_scores = ids, col
        elif ids != ref_ids:
            raise ValueError(f"model {name!r} scored a different window order — unpaired")
        comps[name] = stack_components(col)
    clusters, _ = _window_clusters(ref_scores)
    levels = ref_scores[0].quantile_levels if ref_scores else ()
    return Rows(series_id=ref_ids, cluster=clusters,
                source=[s.source or "" for s in ref_scores],
                domain=[s.domain or "" for s in ref_scores],
                freq=[str(md.get(i, {}).get("freq", "") or "") for i in ref_ids],
                models=comps, quantile_levels=levels)


def cmd_score(args, argv) -> int:
    from cascade.validator.calibration import calib_draw, load_pool_series
    from cascade.validator.evaluator import evaluate_checkpoint

    cks: dict[str, Path] = {}
    for spec in args.checkpoint:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--checkpoint must be NAME=DIR; got {spec!r}")
        if name in cks:
            raise SystemExit(f"duplicate checkpoint name {name!r}")
        cks[name] = Path(path)
    if len(cks) < 2:
        raise SystemExit("need at least two checkpoints")
    digests = {n: tree_sha256(p) for n, p in cks.items()}
    dup = [n for n, d in digests.items() if list(digests.values()).count(d) > 1]
    if dup:
        print(f"warning: identical checkpoint bytes for {sorted(dup)}; stats refuses those "
              "pairs as self-pairs", file=sys.stderr)
    pool = Path(args.pool_dir)
    series, _ids, metadata = load_pool_series(pool)
    if not series:
        raise SystemExit(f"{pool}: no series")
    n_draw = int(args.n_windows) if int(args.n_windows) > 0 else len(series)
    windows = calib_draw(series, metadata, horizon=int(args.horizon), n_windows=n_draw,
                         context_length=int(args.context_length), seed=int(args.seed))
    if not windows:
        raise SystemExit(f"{pool}: no eligible series at h={args.horizon}")
    scores = {}
    for name, d in cks.items():
        print(f"scoring {name} on {len(windows)} windows …", file=sys.stderr)
        scores[name] = evaluate_checkpoint(
            d, windows, num_samples=int(args.num_samples), device=args.device,
            trust_checkpoint_code=bool(args.trust_checkpoint_code), workers=int(args.workers),
            quantile_head=not args.sample_path, quantile_batch=int(args.quantile_batch))
    rows = rows_from_scores(windows, scores)
    manifest = dict(provenance(argv),
                    snapshot_dir=str(pool), snapshot_tree_sha256=tree_sha256(pool),
                    snapshot_tar_sha256=args.snapshot_tar_sha256 or
                    "not run: --snapshot-tar-sha256 not given",
                    checkpoints={n: {"dir": str(p), "tree_sha256": digests[n]}
                                 for n, p in cks.items()},
                    draw={"fn": "cascade.validator.calibration.calib_draw",
                          "horizon": int(args.horizon), "n_windows_requested": n_draw,
                          "context_length": int(args.context_length), "seed": int(args.seed)},
                    scorer={"fn": "cascade.validator.evaluator.evaluate_checkpoint",
                            "quantile_head": not args.sample_path,
                            "quantile_batch": int(args.quantile_batch),
                            "num_samples": int(args.num_samples), "device": args.device,
                            "channels_collapsed_per_window": True},
                    seeds={"draw": int(args.seed)})
    npz, js = save_rows(args.out, rows, manifest)
    print(f"wrote {npz} and {js} ({len(rows)} windows, {len(cks)} models)")
    return 0


# ── statistics (CPU) ─────────────────────────────────────────────────────────


def is_subhourly(freq: str) -> bool:
    from cascade.eval.seasonality import _normalise

    f = str(freq or "").strip()
    return bool(f) and _normalise(f) in SUBHOURLY_UNITS


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float]:
    if n <= 0:
        raise ValueError("wilson: n must be positive")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return [lo, hi]


def floor_check(pool_idx: np.ndarray, clusters: np.ndarray, n: int, min_clusters: int,
                *, need_windows: int | None = None) -> str | None:
    """None when the pool can in principle hold a set; else the "below floor"
    string with the pool's window / source counts."""
    need = n if need_windows is None else need_windows
    n_src = len(set(clusters[pool_idx].tolist()))
    if len(pool_idx) < need or n_src < min_clusters:
        return (f"below floor: pool has {len(pool_idx)} windows across {n_src} sources; "
                f"need {need} windows with >= {min_clusters} distinct sources")
    return None


def draw_set(rng: np.random.Generator, pool_idx: np.ndarray, clusters: np.ndarray, n: int,
             min_clusters: int, max_tries: int = MAX_DRAW_TRIES_PER_SET
             ) -> tuple[np.ndarray | None, int]:
    """``(sorted row indices, rejected draws)``: n windows without replacement
    with >= ``min_clusters`` distinct sources; ``None`` when every try fell
    under the floor. Sorted, so the pair keeps the file's row order."""
    rejected = 0
    for _ in range(max_tries):
        idx = np.sort(rng.choice(pool_idx, size=n, replace=False))
        if len(set(clusters[idx].tolist())) >= min_clusters:
            return idx, rejected
        rejected += 1
    return None, rejected


def draw_disjoint_pair(rng: np.random.Generator, pool_idx: np.ndarray, clusters: np.ndarray,
                       n: int, min_clusters: int, max_tries: int = MAX_DRAW_TRIES_PER_SET
                       ) -> tuple[tuple[np.ndarray, np.ndarray] | None, int]:
    """Two NON-overlapping n-window sets (one 2n draw split in half), both at
    the floor; ``None`` when every try fell under it."""
    rejected = 0
    for _ in range(max_tries):
        both = rng.choice(pool_idx, size=2 * n, replace=False)
        a, b = np.sort(both[:n]), np.sort(both[n:])
        if (len(set(clusters[a].tolist())) >= min_clusters
                and len(set(clusters[b].tolist())) >= min_clusters):
            return (a, b), rejected
        rejected += 1
    return None, rejected


def disjoint_partition(rng: np.random.Generator, pool_idx: np.ndarray, n: int
                       ) -> list[np.ndarray]:
    """floor(N / n) disjoint n-window sets from one seeded shuffle (the
    remainder < n is left out); each set sorted. No window appears twice."""
    perm = rng.permutation(pool_idx)
    k = len(perm) // n
    return [np.sort(perm[i * n:(i + 1) * n]) for i in range(k)]


class PairScorer:
    """LCB + point gain of one ordered pair (A = king, B = challenger) on a
    row subset. ``lcb_fn`` is injectable for tests (call counting)."""

    def __init__(self, rows: Rows, king: str, chal: str, *, alpha: float, B: int,
                 lcb_fn: Callable | None = None):
        if king == chal:
            raise ValueError(f"self-pair refused: {king!r} vs itself")
        from cascade.eval.bootstrap import paired_bootstrap_lcb_aggregated

        self.rows, self.king, self.chal = rows, king, chal
        self.alpha, self.B = float(alpha), int(B)
        self.lcb_fn = lcb_fn or paired_bootstrap_lcb_aggregated
        self.k_comp, self.c_comp = rows.models[king], rows.models[chal]
        self.k_scores = rows.window_scores(king)
        self.c_scores = rows.window_scores(chal)
        self.calls = 0

    def score(self, idx: np.ndarray, seed: int) -> tuple[float, float]:
        from cascade.eval.scoring import global_geomean

        self.calls += 1
        kq, ka, km = (c[idx] for c in self.k_comp)
        cq, ca, cm = (c[idx] for c in self.c_comp)
        lcb = float(self.lcb_fn(kq, ka, km, cq, ca, cm, alpha=self.alpha, B=self.B, seed=int(seed),
                                clusters=self.rows.cluster[idx].tolist(), wql_mode="geomean"))
        gk = global_geomean([self.k_scores[i] for i in idx], wql_mode="geomean")
        gc = global_geomean([self.c_scores[i] for i in idx], wql_mode="geomean")
        gain = (gk - gc) / (gk if abs(gk) >= 1e-9 else 1e-9)
        return lcb, float(gain)


def _spread(vals: Sequence[float]) -> dict | str:
    v = np.asarray(vals, dtype=np.float64)
    if v.size < 2:
        return f"below floor: {v.size} set(s); SD/IQR need >= 2"
    q75, q25 = np.quantile(v, [0.75, 0.25])
    return {"sets": int(v.size), "sd": float(np.std(v, ddof=1)), "iqr": float(q75 - q25)}


def _fpr_grid(lcbs: np.ndarray, gains: np.ndarray, margins: Sequence[float],
              scale: float = 1.0) -> list[dict]:
    n = int(lcbs.size)
    out = []
    for m in margins:
        k = int(np.sum((lcbs * scale >= m) & (gains > 0)))
        out.append({"margin": float(m), "clears": k, "sets": n, "fpr": k / n,
                    "wilson95": wilson(k, n)})
    return out


def margin_for_rate(duel_scores: np.ndarray, p_max: float) -> dict:
    """Smallest margin m with ``mean(duel_scores >= m) <= p_max`` (scores are
    min(LCB1, LCB2) when both gains > 0, else -inf). Reported with the
    resolution caveat when ``p_max`` < 1 / pairs."""
    s = np.sort(np.asarray(duel_scores, dtype=np.float64))[::-1]
    n = int(s.size)
    allowed = int(math.floor(p_max * n + 1e-12))
    if allowed >= n or not np.isfinite(s[allowed]):
        raw = 0.0
    else:
        raw = float(np.nextafter(s[allowed], np.inf))
    out = {"d_margin": max(0.0, raw), "raw_threshold": raw, "p_conf_max": p_max,
           "allowed_false_wins": allowed, "pairs": n}
    if p_max * n < 1.0:
        out["resolution"] = (f"p_conf_max {p_max:.3g} < 1/{n} pairs: the margin only bounds an "
                             "observed-zero rate; raise --R for a resolved estimate")
    return out


def analyse_pool(scorer: PairScorer, pool_idx: np.ndarray, *, n: int, R: int,
                 min_clusters: int, margins: Sequence[float], targets: Sequence[float],
                 era_entries: int, seed: str, label: str) -> dict:
    """Everything for one pair × one pool subset × one floor."""
    rows = scorer.rows
    clusters = rows.cluster
    n_src = len(set(clusters[pool_idx].tolist()))
    out: dict = {"pool": {"windows": int(len(pool_idx)), "sources": n_src},
                 "min_clusters": int(min_clusters), "n": int(n), "R": int(R)}
    pair_tag = f"{scorer.king}>{scorer.chal}"
    seeds = {"resampled": sub_seed(seed, label, pair_tag, min_clusters, "resampled"),
             "disjoint": sub_seed(seed, label, pair_tag, min_clusters, "disjoint"),
             "confirm": sub_seed(seed, label, pair_tag, min_clusters, "confirm"),
             "bootstrap": f"blake2b({seed!r}|{label}|{pair_tag}|{min_clusters}|<stage>|<set>)"}
    out["seeds"] = seeds

    # 1. resampled
    why = floor_check(pool_idx, clusters, n, min_clusters)
    res_lcb: np.ndarray | None = None
    if why:
        out["resampled"] = why
    else:
        rng = np.random.default_rng(seeds["resampled"])
        lcbs, gains, rejected, failed = [], [], 0, 0
        for r in range(R):
            idx, rej = draw_set(rng, pool_idx, clusters, n, min_clusters)
            rejected += rej
            if idx is None:
                failed += 1
                continue
            lcb, gain = scorer.score(idx, sub_seed(seed, label, pair_tag, min_clusters, "res", r))
            lcbs.append(lcb)
            gains.append(gain)
        if not lcbs:
            out["resampled"] = (f"below floor: 0 of {R} sets reached {min_clusters} sources in "
                                f"{MAX_DRAW_TRIES_PER_SET} tries each (pool {len(pool_idx)} "
                                f"windows, {n_src} sources)")
        else:
            res_lcb, res_gain = np.asarray(lcbs), np.asarray(gains)
            out["resampled"] = {
                "sets": int(res_lcb.size), "rejected_draws": rejected,
                "sets_failed_floor": failed,
                "null_lcb_quantiles": {f"p{int(q * 100)}": float(np.quantile(res_lcb, q))
                                       for q in NULL_QUANTILES},
                "point_gain_positive_share": float(np.mean(res_gain > 0)),
                "fpr": _fpr_grid(res_lcb, res_gain, margins),
            }

    # 3. disjoint spread
    ratio = None
    why_d = floor_check(pool_idx, clusters, n, min_clusters, need_windows=2 * n)
    if why_d:
        out["disjoint"] = why_d
        out["spread_ratio"] = f"not run: disjoint sets {why_d}"
    else:
        rng = np.random.default_rng(seeds["disjoint"])
        parts = disjoint_partition(rng, pool_idx, n)
        kept = [p for p in parts if len(set(clusters[p].tolist())) >= min_clusters]
        d_lcbs = [scorer.score(p, sub_seed(seed, label, pair_tag, min_clusters, "dis", i))[0]
                  for i, p in enumerate(kept)]
        spread_d = _spread(d_lcbs)
        out["disjoint"] = {"partitions": len(parts), "below_floor_excluded": len(parts) - len(kept),
                           "spread": spread_d,
                           "note": "spread check only; no FPR is estimated from disjoint sets"}
        if res_lcb is None:
            out["spread_ratio"] = "not run: no resampled sets at this floor"
        else:
            spread_r = _spread(res_lcb)
            if isinstance(spread_d, str) or isinstance(spread_r, str):
                out["spread_ratio"] = spread_d if isinstance(spread_d, str) else spread_r
            elif spread_r["sd"] <= 0 or spread_r["iqr"] <= 0:
                out["spread_ratio"] = ("not run: resampled null LCB spread is zero "
                                       f"(sd {spread_r['sd']}, iqr {spread_r['iqr']})")
            else:
                ratio = spread_d["sd"] / spread_r["sd"]
                out["spread_ratio"] = {"resampled": spread_r, "disjoint": spread_d,
                                       "sd_ratio": ratio,
                                       "iqr_ratio": spread_d["iqr"] / spread_r["iqr"]}
    if res_lcb is not None:
        if ratio is None:
            out["fpr_ratio_corrected"] = "not run: disjoint/resampled spread ratio unavailable"
        elif ratio <= 1.0:
            out["fpr_ratio_corrected"] = {"sd_ratio": ratio, "applied": False,
                                          "note": "ratio <= 1: uncorrected FPR stands",
                                          "fpr": out["resampled"]["fpr"]}
        else:
            out["fpr_ratio_corrected"] = {
                "sd_ratio": ratio, "applied": True,
                "note": "margins scaled by sd_ratio: clear iff LCB * sd_ratio >= margin",
                "fpr": _fpr_grid(res_lcb, res_gain, margins, scale=ratio)}

    # 4. confirmation duel
    why_c = floor_check(pool_idx, clusters, n, min_clusters, need_windows=2 * n)
    if why_c:
        out["confirmation"] = why_c
    else:
        rng = np.random.default_rng(seeds["confirm"])
        duel, rejected, failed = [], 0, 0
        for r in range(R):
            pair, rej = draw_disjoint_pair(rng, pool_idx, clusters, n, min_clusters)
            rejected += rej
            if pair is None:
                failed += 1
                continue
            l1, g1 = scorer.score(pair[0], sub_seed(seed, label, pair_tag, min_clusters, "c1", r))
            l2, g2 = scorer.score(pair[1], sub_seed(seed, label, pair_tag, min_clusters, "c2", r))
            duel.append(min(l1, l2) if (g1 > 0 and g2 > 0) else -math.inf)
        if not duel:
            out["confirmation"] = (f"below floor: 0 of {R} duel pairs reached {min_clusters} "
                                   f"sources on both sets (pool {len(pool_idx)} windows, "
                                   f"{n_src} sources)")
        else:
            s = np.asarray(duel)
            ks = sorted({1, 8, int(era_entries)})
            grid = []
            for m in margins:
                k = int(np.sum(s >= m))
                p = k / s.size
                grid.append({"margin": float(m), "false_wins": k, "pairs": int(s.size),
                             "p_conf": p, "wilson95": wilson(k, int(s.size)),
                             "per_era_fpr": {f"k={kk}": 1.0 - (1.0 - p) ** kk for kk in ks}})
            needed = {}
            for t in targets:
                needed[f"per_era<={t:g}"] = {}
                for kk in ks:
                    p_max = 1.0 - (1.0 - float(t)) ** (1.0 / kk)
                    m = margin_for_rate(s, p_max)
                    if ratio is not None and ratio > 1.0:
                        m["d_margin_ratio_corrected"] = m["d_margin"] * ratio
                    else:
                        m["d_margin_ratio_corrected"] = (
                            "not run: spread ratio unavailable" if ratio is None
                            else "ratio <= 1: equals d_margin")
                    needed[f"per_era<={t:g}"][f"k={kk}"] = m
            out["confirmation"] = {"pairs": int(s.size), "rejected_draws": rejected,
                                   "pairs_failed_floor": failed, "fpr": grid,
                                   "d_margin_needed": needed}
    return out


def ordered_null_pairs(models: Sequence[str], pairs: Sequence[str] | None,
                       digests: dict | None = None) -> list[tuple[str, str]]:
    """Every ordered (king, challenger) pair of distinct models, or the
    ``KING:CHAL`` specs given. A model against itself — by name, or by
    identical checkpoint digest — is refused."""
    if pairs:
        out = []
        for spec in pairs:
            a, sep, b = spec.partition(":")
            if not sep or a not in models or b not in models:
                raise ValueError(f"--pair {spec!r}: need KING:CHAL from {list(models)}")
            out.append((a, b))
    else:
        out = list(itertools.permutations(models, 2))
    for a, b in out:
        if a == b:
            raise ValueError(f"self-pair refused: {a!r} vs itself")
        if digests and digests.get(a) and digests.get(a) == digests.get(b):
            raise ValueError(f"self-pair refused: {a!r} and {b!r} are the same checkpoint bytes")
    return out


def run_stats(rows: Rows, *, pairs: list[tuple[str, str]], n: int, R: int,
              floors: Sequence[int], margins: Sequence[float], targets: Sequence[float],
              era_entries: int, alpha: float, B: int, seed: str, subhourly_only: bool,
              lcb_fn: Callable | None = None) -> dict:
    all_idx = np.arange(len(rows))
    if subhourly_only:
        mask = np.asarray([is_subhourly(f) for f in rows.freq.tolist()], dtype=bool)
        pool_idx, label = all_idx[mask], "subhourly"
        pool_note = {"filter": "subhourly", "kept": int(mask.sum()),
                     "dropped": int((~mask).sum()),
                     "unknown_freq": int(sum(1 for f in rows.freq.tolist() if not f))}
    else:
        pool_idx, label = all_idx, "all"
        pool_note = {"filter": "all", "kept": int(len(rows))}
    out = {"pool_filter": pool_note, "pairs": {}}
    for king, chal in pairs:
        scorer = PairScorer(rows, king, chal, alpha=alpha, B=B, lcb_fn=lcb_fn)
        per = {}
        for f in floors:
            per[f"min_clusters={int(f)}"] = analyse_pool(
                scorer, pool_idx, n=n, R=R, min_clusters=int(f), margins=margins,
                targets=targets, era_entries=era_entries, seed=seed, label=label)
        out["pairs"][f"{king}>{chal}"] = {"king": king, "challenger": chal,
                                          "scored_sets": scorer.calls, "floors": per}
    return out


def cmd_stats(args, argv) -> int:
    rows, manifest, sha = load_rows(args.rows)
    digests = {n: (manifest.get("checkpoints", {}).get(n) or {}).get("tree_sha256")
               for n in manifest["models"]}
    pairs = ordered_null_pairs(manifest["models"], args.pair, digests)
    if int(args.era_entries) < 1:
        raise SystemExit("--era-entries must be >= 1")
    result = run_stats(rows, pairs=pairs, n=int(args.n), R=int(args.R), floors=args.min_clusters,
                       margins=args.margins, targets=args.targets,
                       era_entries=int(args.era_entries), alpha=float(args.alpha),
                       B=int(args.B), seed=str(args.seed), subhourly_only=bool(args.subhourly_only))
    doc = dict(provenance(argv), schema=STATS_SCHEMA,
               rows_file=str(args.rows), rows_sha256=sha,
               snapshot_tree_sha256=manifest.get("snapshot_tree_sha256",
                                                 "not run: rows manifest carries no snapshot sha256"),
               snapshot_tar_sha256=manifest.get("snapshot_tar_sha256",
                                                "not run: rows manifest carries no tar sha256"),
               rows_provenance={k: manifest.get(k) for k in
                                ("command_line", "host", "timestamp_utc", "checkpoints",
                                 "draw", "scorer")},
               seeds={"master": str(args.seed), "draw_in_rows": manifest.get("seeds")},
               params={"n": int(args.n), "R": int(args.R), "min_clusters": list(args.min_clusters),
                       "margins": list(args.margins), "targets": list(args.targets),
                       "era_entries": int(args.era_entries), "alpha": float(args.alpha),
                       "B": int(args.B), "wql_mode": "geomean", "clusters": "source",
                       "lcb_sign": "(king - chal) / king",
                       "clear": "LCB >= margin AND point gain > 0"},
               **result)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(doc)
    print(f"wrote {args.out}")
    return 0


def print_summary(doc: dict, out=sys.stdout) -> None:
    print(f"null FPR ({doc['pool_filter']['filter']} windows), rows {doc['rows_sha256'][:16]}…",
          file=out)
    for tag, p in doc["pairs"].items():
        for fl, r in p["floors"].items():
            res = r["resampled"]
            head = f"  {tag} {fl}: "
            if isinstance(res, str):
                print(head + res, file=out)
                continue
            q = res["null_lcb_quantiles"]
            print(head + f"{res['sets']} sets, null LCB p50 {q['p50']:.4f} p99 {q['p99']:.4f}",
                  file=out)
            conf = r["confirmation"]
            if isinstance(conf, str):
                print(f"    confirmation: {conf}", file=out)
                continue
            for t, by_k in conf["d_margin_needed"].items():
                print(f"    d_margin for {t}: " + "  ".join(
                    f"{k} {v['d_margin']:.4f}" for k, v in by_k.items()), file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="GPU: score checkpoints on one pinned window list")
    sc.add_argument("--pool-dir", required=True, help="unpacked, pinned pool snapshot dir")
    sc.add_argument("--checkpoint", action="append", required=True, help="NAME=DIR (>= 2)")
    sc.add_argument("--out", required=True, help="output PREFIX (writes PREFIX.npz/.json)")
    sc.add_argument("--n-windows", type=int, default=0, help="0 = every eligible series")
    sc.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    sc.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    sc.add_argument("--seed", type=int, default=0, help="window draw seed")
    sc.add_argument("--device", default="cuda")
    sc.add_argument("--workers", type=int, default=1)
    sc.add_argument("--quantile-batch", type=int, default=32)
    sc.add_argument("--sample-path", action="store_true",
                    help="score Monte-Carlo samples instead of the quantile head (pre-v2)")
    sc.add_argument("--num-samples", type=int, default=100)
    sc.add_argument("--trust-checkpoint-code", action="store_true",
                    help="skip the ingest guard (ARCHIVED operator checkpoints only)")
    sc.add_argument("--snapshot-tar-sha256", default=None,
                    help="the published snapshot tar's sha256 (pool index), recorded as given")
    st = sub.add_parser("stats", help="CPU: null FPR statistics from a rows file")
    st.add_argument("--rows", required=True, help="PREFIX.npz written by `score`")
    st.add_argument("--out", required=True, help="output JSON")
    st.add_argument("--era-entries", type=int, required=True,
                    help="K: challenger entries per era (per-era FPR at k = 1, 8, K)")
    st.add_argument("--pair", action="append", default=None, help="KING:CHAL (default: all)")
    st.add_argument("--n", type=int, default=DEFAULT_N)
    st.add_argument("--R", type=int, default=DEFAULT_R)
    st.add_argument("--min-clusters", type=int, nargs="+", default=list(DEFAULT_FLOORS))
    st.add_argument("--margins", type=float, nargs="+", default=list(DEFAULT_MARGINS))
    st.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    st.add_argument("--subhourly-only", action="store_true")
    st.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    st.add_argument("--B", type=int, default=DEFAULT_B)
    st.add_argument("--seed", default="lane-d-null-fpr")
    args = ap.parse_args(argv)
    return cmd_score(args, argv) if args.cmd == "score" else cmd_stats(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
