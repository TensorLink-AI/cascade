"""Lane D ``pool_d/1`` snapshot builder (plan W6b; §1.7 "pool_d (D12)").

Runs on the dedicated ``pool_d`` CPU builder, never on the box. One snapshot
holds, for every series the W6a join selected from lane A's ``pool``:

* history up to ``history_length`` (16384) points plus the ``horizon`` (64)
  target, cut from the SAME forge mirror at the snapshot's ``as_of_ts``;
* per series (per channel) the data time and the W6fs first-seen time of
  every target point (``target_ts``, ``first_seen_ts``), ``first_target_ts``
  and ``end_ts``;
* ``frontier``: ``[as_of_ts, end_ts]`` for each earlier retained snapshot that
  held the series (a list, possibly empty, never absent);
* a ``feeds`` entry for every feed: the W6fs ``mirror_lag_hours`` and lead,
  the catalog ``forecast_product`` and the **effective** ``embargo_hours``
  (rev6.10: the catalog value clamped to ``[lane_d] embargo_hours``, the
  global value when the catalog has none, 0 for a publish-ahead feed; rev6.12:
  no sub-hourly zero is written, the pinned value applies to every series).

Only points the W6fs index knows at ``as_of_ts`` (roots committed by then)
are used, so every first-seen time is earlier than the snapshot. A feed whose
catalog ``embargo_hours`` is negative or non-numeric is left out of ``feeds``
together with its series (fail closed; the snapshot stays valid). The files
(``metadata.json``, ``hist.npz``, ``tgt.npz``) are written deterministically,
so the snapshot sha256 is a content address anyone can re-derive. Lane A's
``pool`` is never written or changed here.

W2e note: ``[lane_d] embargo_hours`` is read from the parsed ``[lane_d]``
table when it carries the key, else the plan default 48 (rev6.10 (a)).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import socket
import sys
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..shared.config.lane_d import LaneDConfig
from .builder import _interp_channel, _sanitize_id

POOL_D_SCHEMA = "pool_d/1"
BUILDER_VERSION = "cascade-pool-d/1"
HISTORY_LENGTH = 16_384
# The loader's [lane_d] embargo_hours default (W2e), never a second literal.
DEFAULT_EMBARGO_HOURS = float(LaneDConfig.embargo_hours)
EVIDENCE_SCHEMA = "laned-evidence/w6b/1"
SNAPSHOT_FILES = ("metadata.json", "hist.npz", "tgt.npz")
REPLAY_MISMATCH_EXIT = 3
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _number(value) -> float | None:
    """A finite, non-negative number (bools refused), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    f = float(value)
    if not math.isfinite(f) or f < 0:
        return None
    return f


def lane_d_embargo_hours(lane_d_cfg) -> float:
    """``[lane_d] embargo_hours`` from the parsed table (W2e), else 48.0."""
    raw = getattr(lane_d_cfg, "embargo_hours", DEFAULT_EMBARGO_HOURS)
    value = _number(raw)
    if value is None:
        raise ValueError(f"[lane_d] embargo_hours must be a finite number >= 0; got {raw!r}")
    return value


def effective_embargo_hours(entry: Mapping, *, global_hours: float,
                            publish_ahead: bool) -> float | None:
    """The effective per-feed embargo (rev6.10), or ``None`` (fail closed)
    when the catalog value is negative, non-finite, a bool or not a number."""
    has = "embargo_hours" in entry
    catalog = _number(entry.get("embargo_hours")) if has else None
    if has and catalog is None:
        return None
    if publish_ahead:
        return 0.0
    if catalog is None:
        return float(global_hours)
    return min(catalog, float(global_hours))


@dataclass(frozen=True)
class PoolDBuildConfig:
    history_length: int = HISTORY_LENGTH
    horizon: int = 64
    min_context: int = 256
    max_missing_frac: float = 0.2
    min_std: float = 1e-8
    embargo_hours: float = DEFAULT_EMBARGO_HOURS

    @classmethod
    def from_lane_d(cls, lane_d_cfg) -> PoolDBuildConfig:
        return cls(horizon=int(getattr(lane_d_cfg, "horizon", 64)),
                   embargo_hours=lane_d_embargo_hours(lane_d_cfg))


@dataclass
class PoolDSummary:
    snapshot: str
    sha256: str
    n_series: int
    n_feeds: int
    out_dir: str
    refused: dict = field(default_factory=dict)


# ── deterministic files ─────────────────────────────────────────────────────


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """An ``np.load``-compatible ``.npz`` whose bytes depend only on the
    arrays: sorted members, a fixed zip timestamp, no compression."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]),
                                      allow_pickle=False)
            zf.writestr(info, buf.getvalue())


def snapshot_sha256(pool_dir: Path | str) -> str:
    """The snapshot digest :func:`cascade.validator.lane_d.load_pool` computes."""
    d = Path(pool_dir)
    return hashlib.sha256("".join(hashlib.sha256((d / n).read_bytes()).hexdigest()
                                  for n in SNAPSHOT_FILES).encode()).hexdigest()


# ── forge reading ───────────────────────────────────────────────────────────


def _series_base(feed: str, panel_key: Mapping[str, str]) -> str:
    """The pre-suffix key ``TsbenchForgeSource`` gives a panel row."""
    suffix = "__".join(f"{k}_{v}" for k, v in panel_key.items())
    return _sanitize_id(f"tsforge__{feed}" + (f"__{suffix}" if suffix else ""))


def _densest_name(sub, excluded) -> str | None:
    best, best_n = None, -1
    for c in [c for c in sub.columns if c not in excluded]:
        try:
            v = sub[c].astype(float).to_numpy()
        except (TypeError, ValueError):
            continue
        n = int(np.isfinite(v).sum())
        if n > best_n:
            best, best_n = c, n
    return best if best_n >= 2 else None


def _channel_column(entry: Mapping, sel, sub, excluded) -> str | None:
    mvc = entry.get("mv_channels") or None
    if mvc:
        from .sources.tsbench_forge import MAX_MV_CHANNELS

        names = [str(c) for c in mvc][:MAX_MV_CHANNELS]
        return names[sel.channel] if sel.channel < len(names) else None
    return _densest_name(sub, excluded) if sel.channel == 0 else None


def _cut(values: np.ndarray, cfg: PoolDBuildConfig) -> tuple[np.ndarray, np.ndarray] | str:
    """``(history float32, target float32)`` or a refusal reason."""
    h = int(cfg.horizon)
    if values.size < h + int(cfg.min_context):
        return "too_short"
    tail = values[-(int(cfg.history_length) + h):]
    target, hist = tail[-h:], tail[:-h]
    if not np.isfinite(target).all():
        return "target_gap"
    filled, missing = _interp_channel(hist.astype(np.float64))
    if filled is None or missing > cfg.max_missing_frac:
        return "too_much_missing"
    if np.ptp(filled) == 0.0 or float(np.std(filled)) < cfg.min_std:
        return "degenerate"
    return filled.astype(np.float32), target.astype(np.float32)


def _prior_frontiers(prior_dirs: Sequence[Path | str], as_of_ts: int) -> dict[str, list]:
    out: dict[str, list] = {}
    for d in prior_dirs:
        m = json.loads((Path(d) / "metadata.json").read_text(encoding="utf-8"))
        a = int(m["as_of_ts"])
        if a >= as_of_ts:
            continue
        for s in m.get("series") or []:
            if s.get("end_ts") is not None:
                out.setdefault(str(s["series_id"]), []).append([a, int(s["end_ts"])])
    return {k: sorted(v) for k, v in out.items()}


def build_pool_d(*, forge_dir: Path | str, selection, first_seen_dir: Path | str,
                 as_of_ts: int, out_dir: Path | str, cfg: PoolDBuildConfig | None = None,
                 prior_dirs: Sequence[Path | str] = (), overwrite: bool = False) -> PoolDSummary:
    """Build one ``pool_d/1`` snapshot for the W6a ``selection`` at ``as_of_ts``."""
    from .first_seen import FirstSeenLog, _epoch_seconds, panel_json
    from .sources.tsbench_forge import TsbenchForgeSource, resolve_frequency

    cfg = cfg or PoolDBuildConfig()
    as_of_ts = int(as_of_ts)
    forge = Path(forge_dir)
    pd, pq, yaml = TsbenchForgeSource._require_deps()
    catalog_bytes = (forge / "sources.yaml").read_bytes()
    entries = {str(e["id"]): e for e in (yaml.safe_load(catalog_bytes) or []) if e.get("id")}
    index = FirstSeenLog(first_seen_dir).index(until_ts=as_of_ts)
    lags = index.feed_lag()
    frontiers = _prior_frontiers(prior_dirs, as_of_ts)
    reader = TsbenchForgeSource(forge)
    as_of_day = dt.datetime.fromtimestamp(as_of_ts, tz=dt.UTC).date()

    by_feed: dict[str, list] = {}
    for sel in selection.series:
        by_feed.setdefault(sel.feed, []).append(sel)
    refused: dict[str, str] = {}
    series_out: list[dict] = []
    hist: dict[str, np.ndarray] = {}
    tgt: dict[str, np.ndarray] = {}
    feeds_out: dict[str, dict] = {}
    roots_used: set[str] = set()

    for feed in sorted(by_feed):
        sels = []
        for sel in by_feed[feed]:
            if ".rollup_" in sel.key[1]:
                refused[sel.pool_d_series_id] = "rollup_not_supported"
            else:
                sels.append(sel)
        entry = entries.get(feed)
        mapped = resolve_frequency(str(entry.get("frequency", ""))) if entry else None
        if not sels:
            continue
        if entry is None or mapped is None:
            refused.update({s.pool_d_series_id: "not_in_forge" for s in sels})
            continue
        lag = lags.get(feed, {"mirror_lag_hours": None, "max_lead_hours": None,
                              "publish_ahead": False})
        emb = effective_embargo_hours(entry, global_hours=cfg.embargo_hours,
                                      publish_ahead=bool(lag["publish_ahead"]))
        if emb is None:
            refused.update({s.pool_d_series_id: "bad_embargo" for s in sels})
            continue
        fp = entry.get("forecast_product", False)
        if not isinstance(fp, bool):
            refused.update({s.pool_d_series_id: "bad_forecast_product" for s in sels})
            continue
        freq, seasonal = mapped
        files = reader._snapshot_files(forge / "data" / feed, as_of=as_of_day)
        df = reader._load_frame(pd, pq, files) if files else None
        if df is None or df.empty or "timestamp" not in df.columns:
            refused.update({s.pool_d_series_id: "not_in_forge" for s in sels})
            continue
        ok, secs = _epoch_seconds(pd, df["timestamp"])
        df = df.assign(_ts=secs)[ok]
        panel_cols = [c for c in df.columns if c.startswith("_panel_")]
        excluded = {"timestamp", "_ts", *panel_cols}
        wanted = {s.key[1]: [] for s in sels}
        for s in sels:
            wanted[s.key[1]].append(s)
        found: set[str] = set()
        groups = df.groupby(panel_cols, sort=True) if panel_cols else [((), df)]
        for gkey, sub in groups:
            key_t = gkey if isinstance(gkey, tuple) else (gkey,)
            panel = {c.removeprefix("_panel_"): str(v)
                     for c, v in zip(panel_cols, key_t, strict=False)}
            base = _series_base(feed, panel)
            if base not in wanted:
                continue
            found.add(base)
            pj = panel_json(panel)
            sub = sub.sort_values("_ts", kind="stable")
            ts_all = sub["_ts"].to_numpy(dtype=np.int64)
            seen_all = np.array([index.first_seen(feed, pj, int(t)) or -1 for t in ts_all],
                                dtype=np.int64)
            keep = seen_all >= 0
            sub, ts_arr, seen_arr = sub[keep], ts_all[keep], seen_all[keep]
            lo, hi = reader._freshest_segment(
                ts_arr.astype("datetime64[s]") if ts_arr.size else None, int(ts_arr.size),
                min_len=int(cfg.horizon) + int(cfg.min_context))
            for sel in wanted[base]:
                column = _channel_column(entry, sel, sub, excluded)
                if column is None or column not in sub.columns:
                    refused[sel.pool_d_series_id] = "not_in_forge"
                    continue
                try:
                    vals = sub[column].astype(float).to_numpy()[lo:hi]
                except (TypeError, ValueError):
                    refused[sel.pool_d_series_id] = "not_in_forge"
                    continue
                cut = _cut(vals, cfg)
                if isinstance(cut, str):
                    refused[sel.pool_d_series_id] = cut
                    continue
                h_vals, t_vals = cut
                n_used = h_vals.size + t_vals.size
                seg_ts, seg_seen = ts_arr[lo:hi][-n_used:], seen_arr[lo:hi][-n_used:]
                for t in seg_ts:
                    roots_used.add(index.root_for(feed, pj, int(t)))
                target_ts = [int(t) for t in seg_ts[-cfg.horizon:]]
                sid = sel.pool_d_series_id
                hist[sid], tgt[sid] = h_vals, t_vals
                series_out.append({
                    "series_id": sid, "pool_series_id": sel.pool_series_id,
                    "source": sel.cluster, "feed": feed, "panel": panel,
                    "channel": int(sel.channel), "column": str(column),
                    "freq": freq, "seasonal_period": int(seasonal), "domain": sel.domain,
                    "history_len": int(h_vals.size),
                    "target_ts": target_ts,
                    "first_seen_ts": [int(s) for s in seg_seen[-cfg.horizon:]],
                    "first_target_ts": target_ts[0], "end_ts": target_ts[-1],
                    "frontier": frontiers.get(sid, []),
                })
                feeds_out[feed] = {
                    "mirror_lag_hours": lag["mirror_lag_hours"],
                    "max_lead_hours": lag["max_lead_hours"],
                    "publish_ahead": bool(lag["publish_ahead"]),
                    "forecast_product": fp,
                    "embargo_hours": float(emb),
                }
        for base, sel_list in wanted.items():
            if base not in found:
                refused.update({s.pool_d_series_id: "not_in_forge" for s in sel_list})

    series_out.sort(key=lambda s: s["series_id"])
    meta = {
        "schema": POOL_D_SCHEMA, "snapshot": f"pool_d-{as_of_ts}", "as_of_ts": as_of_ts,
        "horizon": int(cfg.horizon), "history_length": int(cfg.history_length),
        "builder_version": BUILDER_VERSION,
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "embargo_hours_global": float(cfg.embargo_hours),
        "selection_sha256": selection.sha256(),
        "first_seen_roots": sorted(r for r in roots_used if r),
        "feeds": dict(sorted(feeds_out.items())),
        "series": series_out,
        "refused": sorted([sid, reason] for sid, reason in refused.items()),
    }
    if not series_out:
        raise ValueError("refusing to write an empty pool_d snapshot")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / n).exists() for n in SNAPSHOT_FILES) and not overwrite:
        raise FileExistsError(f"{dest} already holds a pool_d snapshot; pass overwrite=True")
    write_npz(dest / "hist.npz", hist)
    write_npz(dest / "tgt.npz", tgt)
    (dest / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    problems = validate_pool_d(dest)
    if problems:
        for n in SNAPSHOT_FILES:
            (dest / n).unlink(missing_ok=True)
        raise ValueError("invalid pool_d snapshot: " + "; ".join(problems[:10]))
    return PoolDSummary(snapshot=meta["snapshot"], sha256=snapshot_sha256(dest),
                        n_series=len(series_out), n_feeds=len(feeds_out), out_dir=str(dest),
                        refused=dict(sorted(refused.items())))


# ── validation ──────────────────────────────────────────────────────────────


def _feed_problems(name: str, f, global_h) -> list[str]:
    if not isinstance(f, Mapping):
        return [f"feed {name!r}: entry is not an object"]
    out = []
    for key in ("mirror_lag_hours", "forecast_product", "embargo_hours"):
        if key not in f:
            out.append(f"feed {name!r}: no {key}")
    lag = f.get("mirror_lag_hours")
    if lag is not None and (isinstance(lag, bool) or not isinstance(lag, int | float)
                            or not math.isfinite(lag)):
        out.append(f"feed {name!r}: mirror_lag_hours {lag!r} is not a number")
    if "forecast_product" in f and not isinstance(f["forecast_product"], bool):
        out.append(f"feed {name!r}: forecast_product is not a bool")
    if "embargo_hours" in f:
        e = _number(f["embargo_hours"])
        if e is None:
            out.append(f"feed {name!r}: embargo_hours {f['embargo_hours']!r} is not >= 0")
        elif global_h is not None and e > global_h:
            out.append(f"feed {name!r}: embargo_hours {e} above the global {global_h}")
        elif f.get("publish_ahead") is True and e != 0.0:
            out.append(f"feed {name!r}: publish-ahead feed with non-zero embargo_hours")
    return out


def _series_problems(s, feeds, horizon: int, as_of, hist, tgt) -> list[str]:
    sid = s.get("series_id")
    out = []
    if s.get("frontier") is None:
        out.append(f"series {sid!r}: no frontier (absent or null)")
    elif not isinstance(s["frontier"], list):
        out.append(f"series {sid!r}: frontier is not a list")
    feed = s.get("feed") or s.get("source")
    if feed not in feeds:
        out.append(f"series {sid!r}: feed {feed!r} has no feeds entry")
    ts, seen = s.get("target_ts"), s.get("first_seen_ts")
    if not isinstance(ts, list) or len(ts) != horizon:
        out.append(f"series {sid!r}: target_ts is not {horizon} points")
        ts = None
    if not isinstance(seen, list) or len(seen) != horizon:
        out.append(f"series {sid!r}: first_seen_ts is not {horizon} points")
        seen = None
    if ts:
        if s.get("first_target_ts") != ts[0] or s.get("end_ts") != ts[-1]:
            out.append(f"series {sid!r}: first_target_ts / end_ts disagree with target_ts")
        if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
            out.append(f"series {sid!r}: target_ts not strictly increasing")
    if seen and as_of is not None and any(int(x) >= as_of for x in seen):
        out.append(f"series {sid!r}: a first_seen_ts is not before as_of_ts")
    if sid not in hist.files or sid not in tgt.files:
        out.append(f"series {sid!r}: no history or target array")
    elif tgt[sid].shape != (horizon,):
        out.append(f"series {sid!r}: target array is not {horizon} points")
    elif hist[sid].shape != (int(s.get("history_len", -1)),):
        out.append(f"series {sid!r}: history array length != history_len")
    return out


def validate_pool_d(pool_dir: Path | str) -> list[str]:
    """Every problem that makes a ``pool_d/1`` snapshot invalid (``[]`` = ok)."""
    d = Path(pool_dir)
    try:
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"metadata.json unreadable: {e}"]
    problems = []
    if meta.get("schema") != POOL_D_SCHEMA:
        problems.append(f"schema {meta.get('schema')!r} is not {POOL_D_SCHEMA}")
    horizon = meta.get("horizon")
    if not isinstance(horizon, int) or horizon < 1:
        return problems + ["no integer horizon"]
    as_of = meta.get("as_of_ts") if isinstance(meta.get("as_of_ts"), int) else None
    if as_of is None:
        problems.append("no integer as_of_ts")
    feeds = meta.get("feeds")
    if not isinstance(feeds, Mapping):
        return problems + ["no feeds block"]
    global_h = _number(meta.get("embargo_hours_global"))
    for name, f in feeds.items():
        problems.extend(_feed_problems(name, f, global_h))
    roots = meta.get("first_seen_roots")
    if not isinstance(roots, list) or roots != sorted(set(roots)):
        problems.append("first_seen_roots is not a sorted list of unique roots")
    series = meta.get("series")
    if not isinstance(series, list) or not series:
        return problems + ["no series"]
    try:
        hist_npz, tgt_npz = np.load(d / "hist.npz"), np.load(d / "tgt.npz")
    except (OSError, ValueError) as e:
        return problems + [f"arrays unreadable: {e}"]
    with hist_npz as hist, tgt_npz as tgt:
        for s in series:
            if not isinstance(s, Mapping):
                problems.append("a series record is not an object")
                continue
            problems.extend(_series_problems(s, feeds, horizon, as_of, hist, tgt))
    return problems


# ── CLI (the owner's builder run) ───────────────────────────────────────────


def _no_network(url, params=None):
    raise RuntimeError(f"the pool_d builder fetches nothing over the network: {url}")


def _pool_files(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())
            if p.is_file() and (p.name == "metadata.json" or p.suffix == ".npy")}


def lane_a_pool_sha256(pool_dir: Path | str) -> str:
    lines = sorted(f"{n} {h}" for n, h in _pool_files(Path(pool_dir)).items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def replay_lane_a(lane_a_pool: Path, forge_dir: Path, *, max_panel_series_per_feed: int | None,
                  mv_pack: bool, max_series_per_source: int):
    """Rebuild lane A from the forge mirror with the snapshot's recorded
    config; ``(selection, mismatches)`` (empty = byte-identical)."""
    from .builder import PoolBuildConfig, write_pool
    from .pool_d_select import collect_with_selection
    from .source import HarvestContext
    from .sources.tsbench_forge import TsbenchForgeSource

    prov = json.loads((lane_a_pool / "provenance.json").read_text(encoding="utf-8"))
    pcfg = PoolBuildConfig(**prov["config"])
    ctx = HarvestContext(as_of=dt.date.fromisoformat(str(prov["summary"]["as_of"])),
                         context_length=pcfg.context_length, horizon=pcfg.horizon,
                         max_series=int(max_series_per_source))
    src = TsbenchForgeSource(forge_dir, mv_pack=mv_pack,
                             panel_sample_seed=(pcfg.selection_seed
                                                if pcfg.selection == "balanced" else None))
    if max_panel_series_per_feed is not None:
        src.max_series_per_source = int(max_panel_series_per_feed)
    records, drops, selection = collect_with_selection([src], ctx, pcfg, _no_network)
    with tempfile.TemporaryDirectory(prefix="pool-d-replay-") as tmp:
        write_pool(records, tmp, as_of=ctx.as_of.isoformat(), cfg=pcfg, drops=drops)
        got, want = _pool_files(Path(tmp)), _pool_files(lane_a_pool)
    mismatches = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
    return selection, mismatches


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.pool_d_build",
                                 description="Build one lane D pool_d/1 snapshot (W6b).")
    ap.add_argument("--forge-dir", type=Path, required=True)
    ap.add_argument("--first-seen-dir", type=Path, default=None,
                    help="default: <forge-dir>/first_seen")
    ap.add_argument("--lane-a-pool", type=Path, required=True,
                    help="unpacked lane A snapshot (metadata.json, *.npy, provenance.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--as-of-ts", type=int, required=True)
    ap.add_argument("--prior-dir", type=Path, action="append", default=[])
    ap.add_argument("--max-panel-series-per-feed", type=int, default=None)
    ap.add_argument("--max-series-per-source", type=int, default=10_000)
    ap.add_argument("--mv-pack", action="store_true")
    ap.add_argument("--chain-toml", type=Path, default=None,
                    help="read [lane_d] horizon / embargo_hours from this file (default: "
                         "the loader's default chain.toml)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--evidence-out", type=Path, default=None)
    args = ap.parse_args(argv)

    from ..shared.config import load_chain_config

    # [lane_d] horizon / embargo_hours always come from a chain config: the
    # named file, else the loader's default chain.toml (item 7: the builder
    # never falls back to a hard-coded embargo).
    cfg = PoolDBuildConfig.from_lane_d(load_chain_config(args.chain_toml).lane_d)
    print(f"pool_d: [lane_d] embargo_hours={cfg.embargo_hours:g} horizon={cfg.horizon} from "
          f"{args.chain_toml or 'the default chain.toml'}", file=sys.stderr)
    selection, mismatches = replay_lane_a(
        args.lane_a_pool, args.forge_dir, max_panel_series_per_feed=args.max_panel_series_per_feed,
        mv_pack=args.mv_pack, max_series_per_source=args.max_series_per_source)
    if mismatches:
        print(f"error: lane_a_replay_mismatch: {len(mismatches)} file(s) differ from "
              f"{args.lane_a_pool}: {', '.join(mismatches[:10])}", file=sys.stderr)
        return REPLAY_MISMATCH_EXIT
    fs_dir = args.first_seen_dir or args.forge_dir / "first_seen"
    try:
        summary = build_pool_d(forge_dir=args.forge_dir, selection=selection,
                               first_seen_dir=fs_dir, as_of_ts=args.as_of_ts, out_dir=args.out,
                               cfg=cfg, prior_dirs=args.prior_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError, OSError) as e:
        print(f"error: pool_d build failed: {e}", file=sys.stderr)
        return 2
    print(f"pool_d {summary.snapshot} sha256={summary.sha256} series={summary.n_series} "
          f"feeds={summary.n_feeds} refused={len(summary.refused)}")
    if args.evidence_out is not None:
        meta = json.loads((args.out / "metadata.json").read_text(encoding="utf-8"))
        doc = {
            "schema": EVIDENCE_SCHEMA, "wp": "W6b", "source": "builder",
            "generated_by": "cascade.pool.pool_d_build", "host": socket.gethostname(),
            "generated_ts": int(time.time()), "snapshot": summary.snapshot,
            "as_of_ts": int(args.as_of_ts), "sha256": summary.sha256,
            "n_series": summary.n_series, "n_feeds": summary.n_feeds,
            "history_length": int(cfg.history_length),
            "max_history_len": max(int(s["history_len"]) for s in meta["series"]),
            "problems": validate_pool_d(args.out), "lane_a_unchanged": True,
            "lane_a_pool_sha256": lane_a_pool_sha256(args.lane_a_pool),
            "catalog_sha256": meta["catalog_sha256"],
            "refused": dict(sorted(Counter(summary.refused.values()).items())),
        }
        doc["digest"] = hashlib.sha256(_canonical(doc)).hexdigest()
        args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BUILDER_VERSION",
    "DEFAULT_EMBARGO_HOURS",
    "HISTORY_LENGTH",
    "POOL_D_SCHEMA",
    "PoolDBuildConfig",
    "PoolDSummary",
    "build_pool_d",
    "effective_embargo_hours",
    "lane_a_pool_sha256",
    "lane_d_embargo_hours",
    "main",
    "replay_lane_a",
    "snapshot_sha256",
    "validate_pool_d",
    "write_npz",
]
