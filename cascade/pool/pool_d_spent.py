"""Publish SPENT lane D ``pool_d`` snapshots for miners (owner 2026-11-27:
"it's all ok to publish" prior eval data; 2026-11-28: automate it).

After each settlement the box runs ``publish`` (systemd timer
``cascade-laned-spent-publish``). Every pinned snapshot that is **spent** is
written once, anonymously readable, to the public manifest bucket::

    lane-d/spent/pool-preview-<as_of_ts>/README.md
    lane-d/spent/pool-preview-<as_of_ts>/manifest.json   (written last)
    lane-d/spent/pool-preview-<as_of_ts>/windows.npz
    lane-d/spent/index.json                              (every published pool)

**Spent** (the safety rule, fail closed). Let ``T`` be the earliest reveal
time any entry still to be judged can have::

    T = min(now, state_mtime, reveal_ts of every open entry / house entry) - slack

(``state_mtime``: the lane D state file's last write; an entry revealed since
then is not in it yet, and ``slack`` covers one revealed just before it). A
window of an entry revealed at ``T_E >= T`` is eligible only if every target
point has ``first_seen > b - e`` AND ``data_ts > b - e`` (``lane_d_forward``
rule 4, ``b = max(T_E, T_K) + forward_margin >= T``, ``e <= [lane_d]
embargo_hours``). With ``cut = T - 3600 * E`` (``E`` = the larger of the
deployed and the template ``embargo_hours``) a published value can never
enter an eligible window when

* every published TARGET point has ``first_seen_ts <= cut`` (any window
  holding it fails ``seen_ok``), and
* every published HISTORY point has ``data_ts <= cut``: history precedes the
  target, so ``target_ts[0] <= cut`` suffices (any window holding it fails
  ``data_ok``).

A snapshot is spent when (1) :func:`~cascade.pool.pool_d_publish.retention_plan`
expires it (no open entry's eligible or frozen set can reference it),
(2) ``as_of_ts <= cut`` and (3) EVERY series' fixed window passes the two
rules above (a window whose target lies in the future, a publish-ahead feed's,
is left out of the files rather than holding the snapshot back). A window that fails is never written: :class:`EmbargoViolation`
aborts the build before any upload. Any malformed ``reveal_ts`` in the state
publishes nothing (as ``retention_plan`` keeps everything).

Memory-light: one series at a time (``hist.npz`` members are streamed into
``windows.npz``; the house score cache is read row by row). No torch.

``windows.npz`` (one row per window, the snapshot's fixed ``horizon`` window,
in metadata order; single-channel, usable targets only): ``history``
(float32, concatenated, last ``history_cap`` points), ``history_offsets``,
``target`` (n, h) float32, ``target_ts`` / ``first_seen_ts`` (n, h) int64,
``house_quantiles`` (n, h, 9) float32 (NaN where the house was not forecast
on that window), ``has_house``, ``house_crps`` / ``house_mase`` (NaN when
absent) and the string columns ``window_id``, ``series_id``, ``source``,
``feed``, ``freq``, ``domain`` plus ``seasonal_period``, ``origin_ts``,
``history_len``.

CLI::

    python -P -m cascade.pool.pool_d_spent publish --chain-toml C --pins P \\
        --state S --stage DIR [--score-cache-dir D --house-pointer F] [--dry-run]

Credentials by env NAME only (``HIPPIUS_S3_ACCESS_KEY`` /
``HIPPIUS_S3_SECRET_KEY``, write on ``[storage] manifest_bucket``); values are
never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

SPENT_PREFIX = "lane-d/spent"
PREVIEW_DIR = "pool-preview-"
PREVIEW_SCHEMA = "lane-d-pool-preview/1"
INDEX_SCHEMA = "lane-d-spent-index/1"
INDEX_NAME = "index.json"
FILES = ("windows.npz", "README.md", "manifest.json")   # upload order: manifest last
HISTORY_CAP = 16384
DEFAULT_SLACK_HOURS = 6.0
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HOUR = 3600


class EmbargoViolation(RuntimeError):
    """A window that may still be judged reached a published file."""


# ── the rule ────────────────────────────────────────────────────────────────


def effective_embargo_hours(deployed: float | None) -> float:
    """The larger of the deployed and the template ``[lane_d] embargo_hours``
    (an overlay may lower the deployed value; the rule never relies on it)."""
    from ..shared.config.lane_d import LaneDConfig

    vals = [float(LaneDConfig.embargo_hours)]
    if deployed is not None:
        vals.append(float(deployed))
    return max(vals)


def protect_ts(state: Mapping, *, now: int, state_mtime: int | None,
               slack_hours: float = DEFAULT_SLACK_HOURS) -> int | None:
    """``T``: the earliest reveal time an entry still to be judged can have,
    minus ``slack_hours``. ``None`` (publish nothing) when any open entry's
    ``reveal_ts`` is not an int."""
    from .pool_d_publish import _open_entries

    ts = [int(now)]
    if state_mtime is not None:
        ts.append(int(state_mtime))
    for e in _open_entries(state):
        r = e.get("reveal_ts")
        if isinstance(r, bool) or not isinstance(r, int):
            return None
        ts.append(int(r))
    return min(ts) - round(float(slack_hours) * HOUR)


def leak_cut(protect: int, embargo_hours: float) -> int:
    """``cut = T - 3600 * E``: data (history) and first-seen (target) times at
    or before it can never be in an eligible window."""
    if not math.isfinite(float(embargo_hours)) or float(embargo_hours) < 0:
        raise ValueError(f"embargo_hours must be a non-negative number; got {embargo_hours!r}")
    return int(protect) - round(float(embargo_hours) * HOUR)


def window_spent(target_ts, first_seen_ts, *, cut: int, now: int) -> str | None:
    """``None`` when the window may be published, else why not."""
    ts = [int(t) for t in target_ts or []]
    fs = [int(t) for t in first_seen_ts or []]
    if not ts or len(fs) != len(ts):
        return "missing_clock"
    if max(ts) > int(now):
        return "future_target"
    if ts[0] > int(cut):
        return "history_after_cut"           # history reaches data times after the cut
    if max(fs) > int(cut):
        return "first_seen_after_cut"        # a target point is still fresh
    return None


def assert_window_spent(wid: str, target_ts, first_seen_ts, *, cut: int, now: int) -> None:
    why = window_spent(target_ts, first_seen_ts, cut=cut, now=now)
    if why is not None:
        raise EmbargoViolation(f"{wid}: {why} (cut {cut})")


def _meta(pool_dir: Path) -> dict:
    return json.loads((Path(pool_dir) / "metadata.json").read_text(encoding="utf-8"))


def snapshot_status(pool_dir: Path, *, cut: int, now: int) -> str | None:
    """``None`` when every series' fixed window of the snapshot is spent."""
    if not (Path(pool_dir) / "metadata.json").is_file():
        return "missing_snapshot"
    meta = _meta(pool_dir)
    if int(meta["as_of_ts"]) > int(cut):
        return "as_of_after_cut"
    for s in meta.get("series") or []:
        why = window_spent(s.get("target_ts"), s.get("first_seen_ts"), cut=cut, now=now)
        if why is not None and why not in ("missing_clock", "future_target"):
            return f"{why}:{s['series_id']}"
    return None


def spent_plan(pins_path: Path | str, state: Mapping, *, now: int, state_mtime: int | None,
               embargo_hours: float, slack_hours: float = DEFAULT_SLACK_HOURS,
               published: Iterable[int] = ()) -> dict:
    """``{"protect_ts", "cut_ts", "publish": [pin], "published": [as_of],
    "waiting": [{"snapshot", "reason"}]}`` over the pins file."""
    from .pool_d_publish import _load_pins, retention_plan

    pins_path = Path(pins_path)
    pins = sorted(_load_pins(pins_path)["pins"], key=lambda p: (int(p["as_of_ts"]), p["snapshot"]))
    done = {int(x) for x in published}
    out = {"protect_ts": None, "cut_ts": None, "embargo_hours": float(embargo_hours),
           "slack_hours": float(slack_hours), "publish": [], "published": [], "waiting": []}
    protect = protect_ts(state, now=now, state_mtime=state_mtime, slack_hours=slack_hours)
    if protect is None:
        out["waiting"] = [{"snapshot": p["snapshot"], "reason": "malformed_reveal_ts"}
                          for p in pins]
        return out
    cut = leak_cut(protect, embargo_hours)
    out.update(protect_ts=protect, cut_ts=cut)
    expire = set(retention_plan(pins_path, state, embargo_hours=embargo_hours)["expire"])
    for p in pins:
        if int(p["as_of_ts"]) in done:
            out["published"].append(int(p["as_of_ts"]))
            continue
        if p["snapshot"] not in expire:
            why = "retained"
        else:
            why = snapshot_status(pins_path.parent / p["dir"], cut=cut, now=now)
        if why is None:
            out["publish"].append(p)
        else:
            out["waiting"].append({"snapshot": p["snapshot"], "reason": why})
    return out


# ── streaming .npz ──────────────────────────────────────────────────────────


def _npy_header(fp) -> tuple[tuple, np.dtype, int]:
    """``(shape, dtype, header_end)`` of an open ``.npy`` stream."""
    major, _ = np.lib.format.read_magic(fp)
    if major == 1:
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    if fortran:
        raise ValueError("fortran-ordered arrays are not supported")
    return tuple(shape), np.dtype(dtype), fp.tell()


def _member(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with z.open(name + ".npy") as f:
        return np.lib.format.read_array(f, allow_pickle=False)


def _member_shape(z: zipfile.ZipFile, name: str) -> tuple:
    with z.open(name + ".npy") as f:
        return _npy_header(f)[0]


def _write_npy(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    with zf.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


class _StreamedArray:
    """One 1-D ``.npy`` member written chunk by chunk (length known first)."""

    def __init__(self, zf: zipfile.ZipFile, name: str, dtype, length: int):
        self.dtype, self.left = np.dtype(dtype), int(length)
        self._f = zf.open(name + ".npy", "w", force_zip64=True)
        np.lib.format.write_array_header_1_0(
            self._f, {"descr": np.lib.format.dtype_to_descr(self.dtype),
                      "fortran_order": False, "shape": (int(length),)})

    def write(self, chunk: np.ndarray) -> None:
        c = np.ascontiguousarray(chunk, dtype=self.dtype).reshape(-1)
        self.left -= c.size
        if self.left < 0:
            raise ValueError("streamed array overflow")
        self._f.write(c.tobytes())

    def close(self) -> None:
        self._f.close()
        if self.left != 0:
            raise ValueError(f"streamed array short by {self.left}")


# ── the house's cached forecasts ────────────────────────────────────────────


def hist_key(history: np.ndarray, wid: str) -> str:
    """B6 score-cache key (``lane_d_inputs._row_keys``, one-row window)."""
    h = np.ascontiguousarray(np.asarray(history, dtype=np.float64))
    return f"{wid}:{hashlib.sha256(repr(h.shape).encode() + h.tobytes()).hexdigest()[:16]}"


class HouseCache:
    """A score-cache ``.npz`` (``window_ids``, ``quantiles`` (N, h, 9)) read
    lazily: the id column once, then only the asked rows, in one forward
    pass over ``quantiles``."""

    def __init__(self, path: Path | str, *, digest: str):
        self.path, self.digest = Path(path), str(digest)
        with zipfile.ZipFile(self.path) as z:
            ids = _member(z, "window_ids")
        self.rows = {str(k): i for i, k in enumerate(ids.tolist())}

    def file_sha256(self) -> str:
        return _sha_file(self.path)

    def read_rows(self, rows: Iterable[int]) -> dict[int, np.ndarray]:
        want = sorted(set(int(r) for r in rows))
        out: dict[int, np.ndarray] = {}
        if not want:
            return out
        with zipfile.ZipFile(self.path) as z, z.open("quantiles.npy") as f:
            shape, dtype, start = _npy_header(f)
            per = int(np.prod(shape[1:])) * dtype.itemsize
            pos = start
            for r in want:
                skip = start + r * per - pos
                while skip > 0:                       # forward only (deflate-friendly)
                    got = len(f.read(min(skip, 1 << 20)))
                    if not got:
                        raise ValueError("score cache truncated")
                    skip -= got
                buf = f.read(per)
                pos = start + (r + 1) * per
                out[r] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:]).astype(np.float32)
        return out


def find_house_cache(score_cache_dir: Path | str, house_pointer: Path | str,
                     *, context: int = 4096, horizon: int = 64) -> HouseCache | None:
    digest = json.loads(Path(house_pointer).read_text(encoding="utf-8"))["digest"]
    hits = sorted(Path(score_cache_dir).glob(f"*/{digest}-{int(context)}-{int(horizon)}.npz"))
    return HouseCache(hits[-1], digest=digest) if hits else None


# ── build one preview ───────────────────────────────────────────────────────


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _usable(target: np.ndarray, h: int) -> bool:
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    return t.size == h and bool(np.isfinite(t).all()) and float(np.abs(t).sum()) > 0.0


def _window_id(sid: str, ts0: int, h: int) -> str:
    """§1.3 ``<series_id>:<target_start_ts>:h<h>`` (``lane_d.window_id``)."""
    return f"{sid}:{int(ts0)}:h{int(h)}"


def build_preview(pin: Mapping, pool_dir: Path | str, out_dir: Path | str, *, cut: int,
                  now: int, house: HouseCache | None = None,
                  history_cap: int = HISTORY_CAP, rule: Mapping | None = None) -> dict:
    """Write ``windows.npz``, ``manifest.json`` and ``README.md`` for one spent
    snapshot into ``out_dir``; returns the manifest. Raises
    :class:`EmbargoViolation` (and removes ``out_dir``) on any window that may
    still be judged."""
    pool_dir, out = Path(pool_dir), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        return _build(pin, pool_dir, out, cut=cut, now=now, house=house,
                      history_cap=int(history_cap), rule=dict(rule or {}))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _build(pin, pool_dir: Path, out: Path, *, cut, now, house, history_cap, rule) -> dict:
    meta = _meta(pool_dir)
    if int(meta["as_of_ts"]) > int(cut):
        raise EmbargoViolation(f"{meta['snapshot']}: as_of {meta['as_of_ts']} after cut {cut}")
    h = int(meta["horizon"])
    counts = {"series": 0, "missing_clock": 0, "missing_arrays": 0, "unusable_target": 0,
              "multichannel": 0, "future_target": 0}
    sel: list[dict] = []
    hz = zipfile.ZipFile(pool_dir / "hist.npz")
    tz = zipfile.ZipFile(pool_dir / "tgt.npz")
    try:
        hist_names = {n[:-4] for n in hz.namelist()}
        tgt_names = {n[:-4] for n in tz.namelist()}
        # pass 1: select (one series at a time) and find the house rows
        for s in meta.get("series") or []:
            counts["series"] += 1
            sid = str(s["series_id"])
            ts, fs = s.get("target_ts") or [], s.get("first_seen_ts") or []
            if len(ts) != h or len(fs) != h:
                counts["missing_clock"] += 1
                continue
            wid = _window_id(sid, ts[0], h)
            if max(int(t) for t in ts) > int(now):
                counts["future_target"] += 1          # publish-ahead values: left out
                continue
            assert_window_spent(wid, ts, fs, cut=cut, now=now)
            if sid not in hist_names or sid not in tgt_names:
                counts["missing_arrays"] += 1
                continue
            shape = _member_shape(hz, sid)
            if len(shape) == 2 and shape[0] > 1:
                counts["multichannel"] += 1
                continue
            target = _member(tz, sid)
            if not _usable(target, h):
                counts["unusable_target"] += 1
                continue
            row = None
            if house is not None:
                row = house.rows.get(hist_key(_member(hz, sid), wid))
            sel.append({"s": s, "wid": wid, "n": int(min(int(np.prod(shape)), history_cap)),
                        "row": row, "target": np.asarray(target, np.float32).reshape(-1)})
        qrows = house.read_rows(r["row"] for r in sel if r["row"] is not None) if house else {}
        n = len(sel)
        crps = np.full(n, np.nan, np.float64)
        mase = np.full(n, np.nan, np.float64)
        hq = np.full((n, h, len(QUANTILE_LEVELS)), np.nan, np.float32)
        has = np.zeros(n, bool)
        offsets = np.zeros(n + 1, np.int64)
        offsets[1:] = np.cumsum([r["n"] for r in sel], dtype=np.int64)
        npz = out / "windows.npz"
        with zipfile.ZipFile(npz, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            # pass 2: stream the histories, score the house where it forecast
            sa = _StreamedArray(zf, "history", np.float32, int(offsets[-1]))
            for i, r in enumerate(sel):
                full = np.asarray(_member(hz, str(r["s"]["series_id"])), np.float64).reshape(-1)
                sa.write(full[-r["n"]:] if r["n"] else full[:0])
                if r["row"] is not None:
                    from ..eval.lane_d_scorer import score_window

                    q = qrows[r["row"]]
                    try:
                        c, m = score_window(q.astype(np.float64), r["target"].astype(np.float64),
                                            full, int(r["s"].get("seasonal_period") or 1))
                    except ValueError:
                        continue
                    hq[i], has[i], crps[i], mase[i] = q, True, c, m
            sa.close()
            cols = {
                "history_offsets": offsets,
                "target": np.stack([r["target"] for r in sel]) if sel
                else np.zeros((0, h), np.float32),
                "target_ts": np.asarray([r["s"]["target_ts"] for r in sel], np.int64).reshape(n, h),
                "first_seen_ts": np.asarray([r["s"]["first_seen_ts"] for r in sel],
                                            np.int64).reshape(n, h),
                "house_quantiles": hq, "has_house": has, "house_crps": crps, "house_mase": mase,
                "window_id": np.asarray([r["wid"] for r in sel], dtype=str),
                "origin_ts": np.asarray([int(r["s"]["target_ts"][0]) for r in sel], np.int64),
                "history_len": np.diff(offsets),
                "seasonal_period": np.asarray([int(r["s"].get("seasonal_period") or 1)
                                               for r in sel], np.int64),
            }
            for k in ("series_id", "source", "feed", "freq", "domain"):
                cols[k] = np.asarray([str(r["s"].get(k) or "") for r in sel], dtype=str)
            for k, v in cols.items():
                _write_npy(zf, k, v)
    finally:
        hz.close()
        tz.close()
    # belt and braces: re-read the written clocks against the cut
    with zipfile.ZipFile(npz) as z:
        tts, fss = _member(z, "target_ts"), _member(z, "first_seen_ts")
    if tts.size and (int(tts[:, 0].max()) > cut or int(fss.max()) > cut or int(tts.max()) > now):
        raise EmbargoViolation(f"{meta['snapshot']}: a written window reaches the cut {cut}")

    def gm(x, floor):
        x = x[np.isfinite(x)]
        return float(np.exp(np.log(np.maximum(x, floor)).mean())) if x.size else None

    man = {
        "schema": PREVIEW_SCHEMA,
        "kind": "spent pool_d snapshot (every window of the snapshot, NOT a settlement's draw)",
        "snapshot": str(meta["snapshot"]), "as_of_ts": int(meta["as_of_ts"]),
        "snapshot_sha256": str(pin.get("sha256") or ""), "from_block": pin.get("from_block"),
        "horizon": h, "history_cap": int(history_cap), "built_ts": int(now),
        "spent_rule": {**rule, "cut_ts": int(cut),
                       "statement": "every target point first_seen_ts <= cut_ts and every "
                                    "history point data time < target_ts[0] <= cut_ts: no "
                                    "published value can enter a window still eligible for "
                                    "any entry revealed at or after protect_ts"},
        "max_target_start_ts": int(tts[:, 0].max()) if tts.size else None,
        "max_first_seen_ts": int(fss.max()) if fss.size else None,
        "counts": {"windows": n, "house_windows": int(has.sum()), **counts,
                   "sources": len({r["s"].get("source") for r in sel}),
                   "feeds": len({r["s"].get("feed") for r in sel})},
        "house": None if house is None else {
            "digest": house.digest, "context": 4096, "horizon": h,
            "levels": list(QUANTILE_LEVELS), "score_cache_file_sha256": house.file_sha256(),
            "crps_geomean": gm(crps, 1e-12), "mase_geomean": gm(mase, 1e-9)},
        "files": {"windows.npz": {"sha256": _sha_file(npz), "bytes": npz.stat().st_size}},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")
    (out / "README.md").write_text(render_readme(man), encoding="utf-8")
    return man


def render_readme(man: Mapping) -> str:
    c, r = man["counts"], man["spent_rule"]
    house = man.get("house") or {}
    return f"""# Lane D spent pool `{man['snapshot']}`

Every window of the lane D `pool_d` snapshot taken at `as_of_ts = {man['as_of_ts']}`,
published once no entry can still be judged on any of it. It is **not** a
settlement's draw: settlements draw a subset (see the lane D reports).

- **Windows:** {c['windows']} ({c['sources']} sources, {c['feeds']} feeds), horizon {man['horizon']},
  up to {man['history_cap']} history points each. The house forecast {c['house_windows']} of them.
- **Why it is public:** cut `{r['cut_ts']}` = protect `{r.get('protect_ts')}` minus
  {r.get('embargo_hours')} h embargo. Every target point was first seen at or before the cut
  and every history point is older than its target, so no value here can be in a window
  still eligible for an entry revealed after `protect_ts`.
- **House:** `{house.get('digest', '-')}` (CRPS geomean {house.get('crps_geomean')},
  MASE geomean {house.get('mase_geomean')}).
- **Integrity:** `windows.npz` sha256 `{man['files']['windows.npz']['sha256']}`
  ({man['files']['windows.npz']['bytes']} bytes); snapshot sha256 `{man['snapshot_sha256']}`.

## Load

```python
import numpy as np
z = np.load("windows.npz")
i = 0
lo, hi = z["history_offsets"][i], z["history_offsets"][i + 1]
history, target = z["history"][lo:hi], z["target"][i]          # float32
q = z["house_quantiles"][i]        # (h, 9) at levels 0.1..0.9, NaN when has_house is False
print(z["window_id"][i], z["house_crps"][i], z["house_mase"][i])
```

Columns: `history`, `history_offsets`, `target`, `target_ts`, `first_seen_ts`,
`house_quantiles`, `has_house`, `house_crps`, `house_mase`, `window_id`,
`series_id`, `source`, `feed`, `freq`, `domain`, `seasonal_period`,
`origin_ts`, `history_len`. Consecutive snapshots overlap: dedupe by
`window_id`. Every published pool is listed in `../index.json`.
"""


# ── the store side ──────────────────────────────────────────────────────────


def preview_key(as_of: int, name: str = "") -> str:
    return f"{SPENT_PREFIX}/{PREVIEW_DIR}{int(as_of)}/{name}"


def _as_of_of_key(key: str) -> int | None:
    rest = key[len(f"{SPENT_PREFIX}/{PREVIEW_DIR}"):] if key.startswith(
        f"{SPENT_PREFIX}/{PREVIEW_DIR}") else ""
    head, _, tail = rest.partition("/")
    return int(head) if head.isdigit() and tail == "manifest.json" else None


def list_published(client, bucket: str) -> list[int]:
    """``as_of`` of every ``pool-preview-*`` whose ``manifest.json`` exists."""
    out, token = set(), None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{SPENT_PREFIX}/{PREVIEW_DIR}"}
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        for o in page.get("Contents") or []:
            a = _as_of_of_key(o["Key"])
            if a is not None:
                out.add(a)
        if not page.get("IsTruncated"):
            return sorted(out)
        token = page.get("NextContinuationToken")


def _ctype(name: str) -> str:
    return {"json": "application/json", "md": "text/markdown",
            "npz": "application/octet-stream"}[name.rsplit(".", 1)[-1]]


def upload_preview(client, bucket: str, as_of: int, stage_dir: Path) -> list[str]:
    """``public-read`` PUTs, ``manifest.json`` last (its presence = complete)."""
    keys = []
    for name in FILES:
        key = preview_key(as_of, name)
        client.upload_file(str(Path(stage_dir) / name), bucket, key,
                           ExtraArgs={"ACL": "public-read", "ContentType": _ctype(name)})
        keys.append(key)
    return keys


def index_entry(as_of: int, man: Mapping, *, base_url: str) -> dict:
    files = man.get("files") or {}
    w = files.get("windows.npz") or {}
    return {"as_of_ts": int(as_of), "snapshot": man.get("snapshot") or f"pool_d-{as_of}",
            "schema": man.get("schema"), "windows": (man.get("counts") or {}).get("windows"),
            "windows_sha256": w.get("sha256"), "windows_bytes": w.get("bytes"),
            "built_ts": man.get("built_ts"),
            "prefix": preview_key(as_of),
            "manifest_url": f"{base_url}/{preview_key(as_of, 'manifest.json')}",
            "windows_url": f"{base_url}/{preview_key(as_of, 'windows.npz')}",
            "readme_url": f"{base_url}/{preview_key(as_of, 'README.md')}"}


def build_index(entries: Iterable[Mapping], *, now: int) -> dict:
    pools = sorted((dict(e) for e in entries), key=lambda e: e["as_of_ts"])
    return {"schema": INDEX_SCHEMA, "generated_ts": int(now), "count": len(pools),
            "pools": pools}


def _get_json(client, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as e:                                       # noqa: BLE001
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def refresh_index(client, bucket: str, *, base_url: str, now: int,
                  dry_run: bool = False) -> tuple[dict, bool]:
    """Rebuild ``index.json`` from the published manifests; PUT it only when
    its pool list changed. ``(index, written)``."""
    entries = []
    for a in list_published(client, bucket):
        man = _get_json(client, bucket, preview_key(a, "manifest.json")) or {}
        entries.append(index_entry(a, man, base_url=base_url))
    doc = build_index(entries, now=now)
    old = _get_json(client, bucket, f"{SPENT_PREFIX}/{INDEX_NAME}") or {}
    if old.get("pools") == doc["pools"]:
        return old, False
    if not dry_run:
        client.put_object(Bucket=bucket, Key=f"{SPENT_PREFIX}/{INDEX_NAME}",
                          Body=(json.dumps(doc, indent=1, sort_keys=True) + "\n").encode(),
                          ContentType="application/json", ACL="public-read")
    return doc, True


# ── CLI ─────────────────────────────────────────────────────────────────────


def _client(chain_toml: Path):
    from ..shared.config import load_chain_config
    from ..shared.hippius import S3Config, S3Store

    st = load_chain_config(chain_toml).storage
    bucket = str(getattr(st, "manifest_bucket", "") or "")
    if not bucket:
        raise SystemExit("[storage] manifest_bucket unset")
    endpoint = str(getattr(st, "s3_endpoint", "") or "https://s3.hippius.com").rstrip("/")
    return S3Store(S3Config.from_storage(st, bucket=bucket)), bucket, f"{endpoint}/{bucket}"


def cmd_publish(a) -> int:
    from ..shared.config import load_chain_config
    from ..shared.hippius import StorageError

    now = int(a.now) if a.now is not None else int(time.time())
    state = json.loads(Path(a.state).read_text(encoding="utf-8"))
    mtime = int(os.stat(a.state).st_mtime)
    e = effective_embargo_hours(load_chain_config(a.chain_toml).lane_d.embargo_hours)
    store, bucket, base_url = _client(a.chain_toml)
    try:
        published = list_published(store.client(), bucket)
        remote_ok = True
    except (StorageError, Exception) as ex:                      # noqa: BLE001
        if not a.dry_run:
            print(f"spent-publish: remote listing failed: {type(ex).__name__}", file=sys.stderr)
            return 2
        print(f"spent-publish: remote listing unavailable ({type(ex).__name__}); "
              "dry run assumes none published")
        published, remote_ok = [], False
    plan = spent_plan(a.pins, state, now=now, state_mtime=mtime, embargo_hours=e,
                      slack_hours=a.slack_hours, published=published)
    print("spent-publish: " + json.dumps(
        {k: plan[k] for k in ("protect_ts", "cut_ts", "embargo_hours", "slack_hours",
                              "published", "waiting")}
        | {"publish": [p["snapshot"] for p in plan["publish"]]}, sort_keys=True))
    if plan["cut_ts"] is None:
        return 0
    house = None
    if a.score_cache_dir and a.house_pointer:
        house = find_house_cache(a.score_cache_dir, a.house_pointer)
    rule = {k: plan[k] for k in ("protect_ts", "embargo_hours", "slack_hours")}
    stage = Path(a.stage)
    for p in plan["publish"]:
        d = stage / f"{PREVIEW_DIR}{int(p['as_of_ts'])}"
        man = build_preview(p, Path(a.pins).parent / p["dir"], d, cut=plan["cut_ts"], now=now,
                            house=house, rule=rule)
        files = [f"{preview_key(p['as_of_ts'], n)} ({(d / n).stat().st_size} B)" for n in FILES]
        print(f"spent-publish: {'would upload' if a.dry_run else 'upload'} "
              f"{man['snapshot']} windows={man['counts']['windows']}: " + ", ".join(files))
        if not a.dry_run:
            upload_preview(store.client(), bucket, int(p["as_of_ts"]), d)
        shutil.rmtree(d, ignore_errors=True)
    if remote_ok:
        doc, wrote = refresh_index(store.client(), bucket, base_url=base_url, now=now,
                                   dry_run=bool(a.dry_run))
        print(f"spent-publish: index {SPENT_PREFIX}/{INDEX_NAME} pools={doc.get('count')} "
              + ("unchanged" if not wrote else ("would write" if a.dry_run else "written")))
    return 0


def main(argv: list[str] | None = None) -> int:
    from ..shared import box_guard
    box_guard.install("pool-d-spent-publish")
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.pool_d_spent",
                                 description="publish spent lane D pool_d snapshots")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--chain-toml", type=Path, required=True)
    p.add_argument("--pins", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    p.add_argument("--stage", type=Path, required=True, help="scratch dir (one pool at a time)")
    p.add_argument("--score-cache-dir", type=Path, default=None)
    p.add_argument("--house-pointer", type=Path, default=None)
    p.add_argument("--slack-hours", type=float, default=DEFAULT_SLACK_HOURS)
    p.add_argument("--now", type=int, default=None, help="unix seconds (tests)")
    p.add_argument("--dry-run", action="store_true", help="build + verify; upload nothing")
    a = ap.parse_args(argv)
    return cmd_publish(a)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EmbargoViolation", "HouseCache", "INDEX_SCHEMA", "PREVIEW_SCHEMA", "SPENT_PREFIX",
    "build_index", "build_preview", "effective_embargo_hours", "find_house_cache",
    "leak_cut", "list_published", "protect_ts", "refresh_index", "snapshot_status",
    "spent_plan", "upload_preview", "window_spent",
]
