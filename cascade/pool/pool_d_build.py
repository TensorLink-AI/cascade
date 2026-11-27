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

Unreadable partitions: a partition the W6fs quarantine lists (by name, at
``as_of_ts``) is never read, even once it is readable again (its old points
must never count as fresh); any other partition that fails to read is skipped
and logged loudly to stderr. Both lists go into the signed metadata
(``quarantine``, ``unreadable_partitions``) only when non-empty, so a build
without them stays byte-identical. More unreadable partitions than the W6fs
``unreadable_limit`` aborts the build (a systemic read failure).
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
# Owner 2026-11-27 (per-window horizons): the data / first-seen clocks of each
# series' stored target REGION (its newest ``target_points`` points: the tail
# of hist.npz followed by tgt.npz), ``(2, target_points)`` int64 per series.
# Bound by its sha256 in metadata.json (``clocks``), so the snapshot digest
# (over SNAPSHOT_FILES) is unchanged and a pre-band reader loads the snapshot
# exactly as before (its 64-point windows are the region's last 64 points).
CLOCKS_FILE = "clk.npz"
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
    # Per-window horizons (owner 2026-11-27): the GIFT-Eval bands whose
    # targets the region must hold (() = the fixed ``horizon`` target only).
    horizon_bands: tuple[str, ...] = ()
    max_horizon: int = 2048
    # The systemic-read-failure guard (W6fs unreadable_limit defaults).
    max_unreadable_files: int = 50
    max_unreadable_frac: float = 0.001

    @classmethod
    def from_lane_d(cls, lane_d_cfg) -> PoolDBuildConfig:
        return cls(horizon=int(getattr(lane_d_cfg, "horizon", 64)),
                   embargo_hours=lane_d_embargo_hours(lane_d_cfg),
                   horizon_bands=tuple(getattr(lane_d_cfg, "horizon_bands", ()) or ()),
                   max_horizon=int(getattr(lane_d_cfg, "max_horizon", 2048)))


def region_points(freq, raw: np.ndarray, n_used: int, cfg: PoolDBuildConfig) -> int:
    """The stored target region of a series (owner 2026-11-27): the longest
    band horizon of ``freq`` (at least ``horizon``), within the raw segment's
    all-finite suffix (a target point is never interpolated) and leaving
    ``min_context`` history points before it."""
    from ..shared.lane_d_horizons import bands_for

    h = int(cfg.horizon)
    want = max([h, *bands_for(freq, cfg.horizon_bands, max_horizon=cfg.max_horizon).values()])
    bad = np.flatnonzero(~np.isfinite(np.asarray(raw, dtype=np.float64)))
    finite_tail = raw.size - (int(bad[-1]) + 1) if bad.size else raw.size
    return int(max(h, min(want, finite_tail, int(n_used) - int(cfg.min_context))))


@dataclass
class PoolDSummary:
    snapshot: str
    sha256: str
    n_series: int
    n_feeds: int
    out_dir: str
    refused: dict = field(default_factory=dict)
    quarantine: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)


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


def _dedupe_repeated_ts(ts: np.ndarray, seen: np.ndarray, values: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Collapse exact repeated timestamps (``ts`` sorted ascending, stable).

    Each run of equal timestamps keeps its first row when every copy carries
    the same value (NaN equals NaN) and the same first-seen time; any
    disagreement returns ``None`` (the series is refused as
    ``duplicate_ts_conflict``). Without repeats the inputs come back as-is,
    so a regular feed's cut is byte-identical to before.
    """
    if ts.size < 2 or not bool((ts[1:] == ts[:-1]).any()):
        return ts, seen, values
    starts = np.flatnonzero(np.concatenate(([True], ts[1:] != ts[:-1])))
    run_id = np.cumsum(np.concatenate(([True], ts[1:] != ts[:-1]))) - 1
    first_vals, first_seen = values[starts][run_id], seen[starts][run_id]
    same_val = (values == first_vals) | (np.isnan(values) & np.isnan(first_vals))
    if not bool(same_val.all()) or not bool((seen == first_seen).all()):
        return None
    return ts[starts], seen[starts], values[starts]


def _series_invariant_refusal(target_ts: Sequence[int], first_seen_ts: Sequence[int],
                              as_of_ts: int) -> str | None:
    """The typed refusal for a series cut that breaks a per-series snapshot
    invariant :func:`validate_pool_d` checks, else ``None``."""
    if any(b <= a for a, b in zip(target_ts, target_ts[1:], strict=False)):
        return "target_ts_not_increasing"
    if any(int(x) >= int(as_of_ts) for x in first_seen_ts):
        return "first_seen_not_before_as_of"
    return None


def _load_frame_logged(pd, pq, files: Sequence[Path], feed: str, unreadable: list):
    """``TsbenchForgeSource._load_frame`` (same concat + per-panel dedup, so
    the frame is identical), except that each unreadable partition is logged
    loudly and appended to ``unreadable`` as ``[feed, name, reason]``."""
    from .first_seen import unreadable_reason

    frames = []
    for p in files:
        try:
            frames.append(pq.read_table(p).to_pandas())
        except Exception as e:  # noqa: BLE001 — one corrupt partition must not kill the build
            reason = unreadable_reason(e)
            print(f"pool_d: WARNING skipping unreadable partition data/{feed}/{p.name}: "
                  f"{reason}", file=sys.stderr, flush=True)
            unreadable.append([feed, p.name, reason])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        keys = ["timestamp", *[c for c in df.columns if c.startswith("_panel_")]]
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


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
                 prior_dirs: Sequence[Path | str] = (), overwrite: bool = False,
                 lane_a_source: Mapping | None = None) -> PoolDSummary:
    """Build one ``pool_d/1`` snapshot for the W6a ``selection`` at ``as_of_ts``.

    ``lane_a_source`` (``{"mode", "effective_block", "sha256"}``) is recorded
    in ``metadata.json`` as ``lane_a_selection`` when given (absent = the
    pre-W-LAUNCH bytes)."""
    from .first_seen import FirstSeenLog, _epoch_seconds, panel_json
    from .sources.tsbench_forge import TsbenchForgeSource, resolve_frequency

    cfg = cfg or PoolDBuildConfig()
    as_of_ts = int(as_of_ts)
    forge = Path(forge_dir)
    pd, pq, yaml = TsbenchForgeSource._require_deps()
    catalog_bytes = (forge / "sources.yaml").read_bytes()
    entries = {str(e["id"]): e for e in (yaml.safe_load(catalog_bytes) or []) if e.get("id")}
    from .first_seen import unreadable_limit

    fs_log = FirstSeenLog(first_seen_dir)
    # One feed's first-seen points in memory at a time (2026-11-26: the
    # whole-log index of a 20 GB records.jsonl reached 39.3 GB RSS); feeds
    # are walked in order below and every lookup is for the current feed.
    index = fs_log.feed_index(until_ts=as_of_ts)
    quarantined = fs_log.quarantined(until_ts=as_of_ts)
    quarantine_report = fs_log.quarantine_report(until_ts=as_of_ts)
    unreadable: list[list[str]] = []
    n_files_read = 0
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
    clocks: dict[str, np.ndarray] = {}
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
        lag = index.feed_lag([feed]).get(feed, {"mirror_lag_hours": None,
                                                "max_lead_hours": None,
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
        files = [f for f in reader._snapshot_files(forge / "data" / feed, as_of=as_of_day)
                 if (feed, f.name) not in quarantined]
        n_files_read += len(files)
        df = _load_frame_logged(pd, pq, files, feed, unreadable) if files else None
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
        # Two panel rows with one base are lane A's harvest-level key_collision
        # (pool_d_select.select_pool_d refuses them); a snapshot-mode selection
        # cannot see it when lane A kept only one of them, so refuse it here.
        # A replay selection never wants such a base, so its output is unchanged.
        if panel_cols:
            row_bases = Counter(
                _series_base(feed, {c.removeprefix("_panel_"): str(v)
                                    for c, v in zip(panel_cols, row, strict=False)})
                for row in df[panel_cols].drop_duplicates().dropna()
                .itertuples(index=False, name=None))
            for base in [b for b in wanted if row_bases[b] > 1]:
                refused.update({s.pool_d_series_id: "key_collision" for s in wanted.pop(base)})
                found.add(base)
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
            sub, ts_kept, seen_kept = sub[keep], ts_all[keep], seen_all[keep]
            for sel in wanted[base]:
                column = _channel_column(entry, sel, sub, excluded)
                if column is None or column not in sub.columns:
                    refused[sel.pool_d_series_id] = "not_in_forge"
                    continue
                try:
                    col_vals = sub[column].astype(float).to_numpy()
                except (TypeError, ValueError):
                    refused[sel.pool_d_series_id] = "not_in_forge"
                    continue
                # Event / irregular feeds (#841): exact repeated timestamps
                # collapse to one point when every copy agrees, else the
                # series is refused. A feed without repeats is unchanged.
                dd = _dedupe_repeated_ts(ts_kept, seen_kept, col_vals)
                if dd is None:
                    refused[sel.pool_d_series_id] = "duplicate_ts_conflict"
                    continue
                ts_arr, seen_arr, col_vals = dd
                lo, hi = reader._freshest_segment(
                    ts_arr.astype("datetime64[s]") if ts_arr.size else None, int(ts_arr.size),
                    min_len=int(cfg.horizon) + int(cfg.min_context))
                cut = _cut(col_vals[lo:hi], cfg)
                if isinstance(cut, str):
                    refused[sel.pool_d_series_id] = cut
                    continue
                h_vals, t_vals = cut
                n_used = h_vals.size + t_vals.size
                seg_ts, seg_seen = ts_arr[lo:hi][-n_used:], seen_arr[lo:hi][-n_used:]
                target_ts = [int(t) for t in seg_ts[-cfg.horizon:]]
                first_seen_ts = [int(s) for s in seg_seen[-cfg.horizon:]]
                # Per-series snapshot invariants (#841): a series that would
                # fail validate_pool_d is refused here, typed, and the rest
                # of the snapshot is built.
                bad = _series_invariant_refusal(target_ts, first_seen_ts, as_of_ts)
                if bad is not None:
                    refused[sel.pool_d_series_id] = bad
                    continue
                for t in seg_ts:
                    roots_used.add(index.root_for(feed, pj, int(t)))
                sid = sel.pool_d_series_id
                hist[sid], tgt[sid] = h_vals, t_vals
                region = None
                if cfg.horizon_bands:
                    r = region_points(freq, col_vals[lo:hi][-n_used:], n_used, cfg)
                    reg_ts = [int(t) for t in seg_ts[-r:]]
                    reg_seen = [int(x) for x in seg_seen[-r:]]
                    if r > int(cfg.horizon) and _series_invariant_refusal(
                            reg_ts, reg_seen, as_of_ts) is not None:
                        r = int(cfg.horizon)
                        reg_ts, reg_seen = target_ts, first_seen_ts
                    clocks[sid] = np.asarray([reg_ts, reg_seen], dtype=np.int64)
                    region = r
                series_out.append({
                    "series_id": sid, "pool_series_id": sel.pool_series_id,
                    "source": sel.cluster, "feed": feed, "panel": panel,
                    "channel": int(sel.channel), "column": str(column),
                    "freq": freq, "seasonal_period": int(seasonal), "domain": sel.domain,
                    "history_len": int(h_vals.size),
                    "target_ts": target_ts,
                    "first_seen_ts": first_seen_ts,
                    "first_target_ts": target_ts[0], "end_ts": target_ts[-1],
                    "frontier": frontiers.get(sid, []),
                    **({} if region is None else {"target_points": int(region)}),
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
    index.release()

    limit = unreadable_limit(n_files_read, max_files=cfg.max_unreadable_files,
                             max_frac=cfg.max_unreadable_frac)
    if len(unreadable) > limit:
        raise ValueError(f"too_many_unreadable: {len(unreadable)} of {n_files_read} partitions "
                         f"unreadable (limit {limit}); a systemic read failure? First: "
                         + ", ".join(f"{f}/{n}" for f, n, _r in unreadable[:10]))
    unreadable.sort()
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
    # Audit trail for skipped partitions; absent when empty (byte-identical).
    if quarantine_report:
        meta["quarantine"] = quarantine_report
    if lane_a_source is not None:
        meta["lane_a_selection"] = {"mode": str(lane_a_source["mode"]),
                                    "effective_block": int(lane_a_source["effective_block"]),
                                    "sha256": str(lane_a_source["sha256"])}
    if unreadable:
        meta["unreadable_partitions"] = unreadable
    if not series_out:
        raise ValueError("refusing to write an empty pool_d snapshot")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / n).exists() for n in (*SNAPSHOT_FILES, CLOCKS_FILE)) and not overwrite:
        raise FileExistsError(f"{dest} already holds a pool_d snapshot; pass overwrite=True")
    write_npz(dest / "hist.npz", hist)
    write_npz(dest / "tgt.npz", tgt)
    if cfg.horizon_bands:
        kept = {s["series_id"] for s in series_out}
        write_npz(dest / CLOCKS_FILE, {k: v for k, v in clocks.items() if k in kept})
        meta["clocks"] = {"file": CLOCKS_FILE, "sha256": hashlib.sha256(
            (dest / CLOCKS_FILE).read_bytes()).hexdigest()}
        meta["horizon_bands"] = list(cfg.horizon_bands)
        meta["max_horizon"] = int(cfg.max_horizon)
    (dest / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    problems = validate_pool_d(dest)
    if problems:
        for n in (*SNAPSHOT_FILES, CLOCKS_FILE):
            (dest / n).unlink(missing_ok=True)
        raise ValueError("invalid pool_d snapshot: " + "; ".join(problems[:10]))
    return PoolDSummary(snapshot=meta["snapshot"], sha256=snapshot_sha256(dest),
                        n_series=len(series_out), n_feeds=len(feeds_out), out_dir=str(dest),
                        refused=dict(sorted(refused.items())),
                        quarantine=quarantine_report, unreadable=unreadable)


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


def _region_problems(s, horizon: int, as_of, hist, clk) -> list[str]:
    """Owner 2026-11-27: a band snapshot's per-series target region."""
    sid = s.get("series_id")
    r = s.get("target_points")
    if type(r) is not int or r < horizon:
        return [f"series {sid!r}: target_points {r!r} is not an int >= {horizon}"]
    if sid not in clk.files:
        return [f"series {sid!r}: no clocks"]
    c = clk[sid]
    if c.shape != (2, r):
        return [f"series {sid!r}: clocks shape {c.shape} != (2, {r})"]
    out = []
    if hist[sid].shape[-1] + horizon < r:
        out.append(f"series {sid!r}: region of {r} points longer than history + target")
    ts, seen = c[0].tolist(), c[1].tolist()
    if ts[-horizon:] != list(s.get("target_ts") or []) \
            or seen[-horizon:] != list(s.get("first_seen_ts") or []):
        out.append(f"series {sid!r}: region clocks do not end in target_ts / first_seen_ts")
    if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
        out.append(f"series {sid!r}: region target_ts not strictly increasing")
    if as_of is not None and any(int(x) >= as_of for x in seen):
        out.append(f"series {sid!r}: a region first_seen_ts is not before as_of_ts")
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
    clk_rec = meta.get("clocks")
    clk_npz = None
    if clk_rec is not None:
        if not isinstance(clk_rec, Mapping) or clk_rec.get("file") != CLOCKS_FILE:
            return problems + [f"clocks is not {{file: {CLOCKS_FILE!r}, sha256}}"]
        try:
            raw = (d / CLOCKS_FILE).read_bytes()
        except OSError as e:
            return problems + [f"{CLOCKS_FILE} unreadable: {e}"]
        if hashlib.sha256(raw).hexdigest() != clk_rec.get("sha256"):
            return problems + [f"{CLOCKS_FILE} does not hash to metadata clocks.sha256"]
        clk_npz = np.load(d / CLOCKS_FILE)
    with hist_npz as hist, tgt_npz as tgt:
        for s in series:
            if not isinstance(s, Mapping):
                problems.append("a series record is not an object")
                continue
            problems.extend(_series_problems(s, feeds, horizon, as_of, hist, tgt))
            if clk_npz is not None and not problems:
                problems.extend(_region_problems(s, horizon, as_of, hist, clk_npz))
    if clk_npz is not None:
        clk_npz.close()
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


LANE_A_SELECTION_MODES = ("snapshot", "replay")


def lane_a_ref_path(lane_a_pool: Path | str) -> Path:
    """``<lane_a_pool>.ref.json``, written by ``cascade-laned elastic lane-a-pool``."""
    p = Path(lane_a_pool)
    return p.with_name(p.name + ".ref.json")


def verify_lane_a_snapshot(lane_a_pool: Path | str, ref_path: Path | str | None = None) -> dict:
    """Check the unpacked lane A snapshot against its pool index entry.

    The ref (``{"effective_block", "sha256", "key", "tar"}``) names the
    index entry the fetch step resolved. The tar's sha256 must equal the
    index sha256, and the unpacked ``metadata.json`` / ``*.npy`` must equal
    the tar's. Returns ``{"mode": "snapshot", "effective_block", "sha256"}``.
    Raises ``ValueError`` (``lane_a_snapshot_refused: ...``) otherwise."""
    from ..shared.hippius import tar_cid_digest, unpack_tar_to_dir

    lane_a_pool = Path(lane_a_pool)
    ref_path = Path(ref_path) if ref_path is not None else lane_a_ref_path(lane_a_pool)
    try:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        block, want = int(ref["effective_block"]), str(ref["sha256"])
        tar = Path(ref["tar"])
        tar = tar if tar.is_absolute() else ref_path.parent / tar
        data = tar.read_bytes()
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"lane_a_snapshot_refused: ref {ref_path} unreadable: {e}") from e
    got = tar_cid_digest(data)
    if got != want:
        raise ValueError(f"lane_a_snapshot_refused: tar sha256 {got} != index sha256 {want} "
                         f"(block {block})")
    with tempfile.TemporaryDirectory(prefix="pool-d-lane-a-") as tmp:
        try:
            unpack_tar_to_dir(data, tmp)
        except Exception as e:  # noqa: BLE001 — any unreadable tar is a refusal
            raise ValueError(f"lane_a_snapshot_refused: tar unreadable: {e}") from e
        packed, unpacked = _pool_files(Path(tmp)), _pool_files(lane_a_pool)
    if not packed or packed != unpacked:
        diff = sorted(n for n in set(packed) | set(unpacked) if packed.get(n) != unpacked.get(n))
        raise ValueError(f"lane_a_snapshot_refused: {lane_a_pool} differs from the verified "
                         f"tar in {len(diff)} file(s): {', '.join(diff[:10])}")
    return {"mode": "snapshot", "effective_block": block, "sha256": want}


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
    ap.add_argument("--max-unreadable-files", type=int, default=None,
                    help="systemic-read-failure guard (default 50)")
    ap.add_argument("--max-unreadable-frac", type=float, default=None,
                    help="systemic-read-failure guard (default 0.001)")
    ap.add_argument("--lane-a-selection", choices=LANE_A_SELECTION_MODES, default="snapshot",
                    help="snapshot (default, owner 2026-11-25): take the selection from the "
                         "sha-verified lane A snapshot itself; replay: rebuild lane A from the "
                         "forge and require byte-identical files")
    ap.add_argument("--lane-a-ref", type=Path, default=None,
                    help="snapshot mode: the fetch step's ref (default <lane-a-pool>.ref.json)")
    args = ap.parse_args(argv)

    from ..shared.config import load_chain_config

    # [lane_d] horizon / embargo_hours always come from a chain config: the
    # named file, else the loader's default chain.toml (item 7: the builder
    # never falls back to a hard-coded embargo).
    cfg = PoolDBuildConfig.from_lane_d(load_chain_config(args.chain_toml).lane_d)
    guard = {k: v for k, v in (("max_unreadable_files", args.max_unreadable_files),
                               ("max_unreadable_frac", args.max_unreadable_frac))
             if v is not None}
    if guard:
        import dataclasses

        cfg = dataclasses.replace(cfg, **guard)
    print(f"pool_d: [lane_d] embargo_hours={cfg.embargo_hours:g} horizon={cfg.horizon} from "
          f"{args.chain_toml or 'the default chain.toml'}", file=sys.stderr)
    lane_a_source = None
    if args.lane_a_selection == "snapshot":
        from .pool_d_select import select_from_snapshot

        try:
            lane_a_source = verify_lane_a_snapshot(args.lane_a_pool, args.lane_a_ref)
            selection = select_from_snapshot(args.lane_a_pool)
        except (OSError, ValueError, KeyError) as e:
            msg = str(e) if "lane_a_snapshot_refused" in str(e) else \
                f"lane_a_snapshot_refused: snapshot unreadable: {e}"
            print(f"error: {msg}", file=sys.stderr)
            return REPLAY_MISMATCH_EXIT
        print(f"pool_d: lane A selection from snapshot block {lane_a_source['effective_block']} "
              f"sha256 {lane_a_source['sha256'][:16]}… ({len(selection.series)} series, "
              f"{len(selection.refused)} refused)", file=sys.stderr)
    else:
        selection, mismatches = replay_lane_a(
            args.lane_a_pool, args.forge_dir,
            max_panel_series_per_feed=args.max_panel_series_per_feed,
            mv_pack=args.mv_pack, max_series_per_source=args.max_series_per_source)
        if mismatches:
            print(f"error: lane_a_replay_mismatch: {len(mismatches)} file(s) differ from "
                  f"{args.lane_a_pool}: {', '.join(mismatches[:10])}", file=sys.stderr)
            return REPLAY_MISMATCH_EXIT
    fs_dir = args.first_seen_dir or args.forge_dir / "first_seen"
    try:
        summary = build_pool_d(forge_dir=args.forge_dir, selection=selection,
                               first_seen_dir=fs_dir, as_of_ts=args.as_of_ts, out_dir=args.out,
                               cfg=cfg, prior_dirs=args.prior_dir, overwrite=args.overwrite,
                               lane_a_source=lane_a_source)
    except (ValueError, FileExistsError, OSError) as e:
        print(f"error: pool_d build failed: {e}", file=sys.stderr)
        return 2
    print(f"pool_d {summary.snapshot} sha256={summary.sha256} series={summary.n_series} "
          f"feeds={summary.n_feeds} refused={len(summary.refused)}"
          + (f" quarantined={len(summary.quarantine)}" if summary.quarantine else "")
          + (f" unreadable={len(summary.unreadable)}" if summary.unreadable else ""))
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
        if summary.quarantine:
            doc["quarantine"] = summary.quarantine
        if summary.unreadable:
            doc["unreadable_partitions"] = summary.unreadable
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
    "lane_a_ref_path",
    "lane_d_embargo_hours",
    "main",
    "replay_lane_a",
    "snapshot_sha256",
    "validate_pool_d",
    "verify_lane_a_snapshot",
    "write_npz",
]
