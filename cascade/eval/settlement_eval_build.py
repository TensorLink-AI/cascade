"""W-EVALPUB: build one lane D settlement's judged eval set on the lane D eval
host (owner 2026-11-28: "publish the eval dataset for lane D after it
occurs"; ``tests/accept/w_evalpub/README.md``).

The validator box (7 GB, live validator) never loads a ``pool_d`` snapshot
(W-OFFBOX, #1000). The box sends this host the settlement's judged windows
(from its signed report) and the leak cut; this host reads them from its own
pinned copies (``<pool_root>/<snapshot>/``, each re-hashed against the
report's ``snapshot_sha256``), one series at a time, checks every window is
**retired** (:func:`window_retired`) and writes ``windows.npz``. Nothing is
written when any window is not retired yet: the reply lists them and the box
waits.

A window is cut exactly as the settlement cut it
(:func:`cascade.validator.lane_d.load_pool`): the series' region is the
``clk.npz`` clock record (a pre-band snapshot's region is its stored target),
``full = concat(hist, tgt)``, the target is the ``h`` points of the region
from the point whose data time is the window's origin, the history every
point before it (at most the snapshot's ``history_length``, then the last
``history_cap``). The window id must equal ``<series_id>:<origin>:h<h>``.

``windows.npz`` (one row per window, ordered by snapshot, series, origin; values float32
as stored; a ``(C, L)`` series is ``C`` channels, flattened channel-major):
``history`` / ``history_offsets`` (values), ``target`` / ``target_offsets``
(values), ``target_ts`` / ``first_seen_ts`` / ``time_offsets`` (int64, one
per target time point), and the columns ``window_id``, ``series_id``,
``snapshot``, ``source``, ``feed``, ``freq``, ``band``, ``h``, ``origin_ts``,
``n_channels``, ``history_len``, ``seasonal_period``, ``embargo_window``.

CLI (the box runs it over SSH; the request JSON on stdin, the reply JSON the
last stdout line)::

    python -m cascade.eval.settlement_eval_build build --pool-root R --out-dir D

Torch-free, numpy only. Arrays are staged in raw temp files and streamed
into the ``.npz``: memory is one series at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REQUEST_SCHEMA = "lane-d-evals-build-request/1"
REPLY_SCHEMA = "lane-d-evals-build-reply/1"
NPZ_NAME = "windows.npz"
HISTORY_CAP = 16384
CLOCKS_FILE = "clk.npz"
_SNAPSHOT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
_STR_COLS = ("window_id", "series_id", "snapshot", "source", "feed", "freq", "band")
_INT_COLS = ("h", "origin_ts", "n_channels", "history_len", "seasonal_period")


class BuildError(RuntimeError):
    """A request that cannot be built (typed ``code``)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


# ── the retirement rule (shared with the box: cascade.pool.settlement_evals) ──


def window_retired(target_ts, first_seen_ts, *, cut: int, now: int) -> str | None:
    """``None`` when a judged window may be published, else why not.

    Retired = every target point's data time AND first-seen time is at or
    before ``cut`` (and none lies in the future). Every published value (the
    history precedes the target) then has data time ``<= cut``, and
    ``lane_d_forward`` rule 4 refuses any window holding such a value in its
    target for every bound ``b`` with ``b - e >= cut`` (``data_ok`` needs
    every target point's data time ``> b - e``)."""
    ts = [int(t) for t in (target_ts if target_ts is not None else [])]
    fs = [int(t) for t in (first_seen_ts if first_seen_ts is not None else [])]
    if not ts or len(fs) != len(ts):
        return "missing_clock"
    if max(ts) > int(now):
        return "future_target"
    if max(ts) > int(cut):
        return "target_after_cut"
    if max(fs) > int(cut):
        return "first_seen_after_cut"
    return None


def parse_window_id(wid: str) -> tuple[str, int, int]:
    """``(series_id, origin, h)`` of ``<series_id>:<origin>:h<h>``."""
    sid, origin, hh = str(wid).rsplit(":", 2)
    if not sid or not hh.startswith("h"):
        raise ValueError(f"bad window id {wid!r}")
    return sid, int(origin), int(hh[1:])


# ── one snapshot ────────────────────────────────────────────────────────────


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_sha256(d: Path) -> str:
    """The digest :func:`cascade.validator.lane_d.load_pool` binds."""
    return hashlib.sha256("".join(_file_sha(Path(d) / n) for n in
                                  ("metadata.json", "hist.npz", "tgt.npz")).encode()).hexdigest()


def _series_clock(s: Mapping, sid: str, clk) -> tuple[np.ndarray, np.ndarray]:
    if clk is not None and sid in clk.files:
        c = np.asarray(clk[sid], dtype=np.int64)
        return c[0], c[1]
    return (np.asarray([int(t) for t in s.get("target_ts") or []], dtype=np.int64),
            np.asarray([int(t) for t in s.get("first_seen_ts") or []], dtype=np.int64))


def cut_window(full: np.ndarray, ts: np.ndarray, seen: np.ndarray, *, origin: int, h: int,
               history_length: int, history_cap: int) -> dict:
    """One window of a series, as the settlement cut it."""
    if not ts.size or ts.size != seen.size or ts.size > full.shape[-1]:
        raise BuildError("bad_clock", "region clocks do not fit the series")
    hit = np.flatnonzero(ts == int(origin))
    if hit.size != 1 or int(hit[0]) + int(h) > ts.size:
        raise BuildError("window_not_in_snapshot", f"origin {origin} h {h}")
    start = int(hit[0])
    cut = full.shape[-1] - ts.size + start
    lo = max(0, cut - int(history_length), cut - int(history_cap))
    return {"history": full[..., lo:cut], "target": full[..., cut:cut + int(h)],
            "target_ts": ts[start:start + int(h)], "first_seen_ts": seen[start:start + int(h)]}


class _Raw:
    """A raw temp file of one dtype, appended in chunks; its length counted."""

    def __init__(self, path: Path, dtype):
        self.path, self.dtype, self.n = Path(path), np.dtype(dtype), 0
        self._f = open(self.path, "wb")  # noqa: SIM115 — closed by close(), in a finally

    def write(self, a) -> int:
        c = np.ascontiguousarray(a, dtype=self.dtype).reshape(-1)
        self._f.write(c.tobytes())
        self.n += c.size
        return c.size

    def close(self) -> None:
        self._f.close()


def _stream_npy(zf: zipfile.ZipFile, name: str, raw: _Raw) -> None:
    with zf.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array_header_1_0(
            f, {"descr": np.lib.format.dtype_to_descr(raw.dtype), "fortran_order": False,
                "shape": (int(raw.n),)})
        with open(raw.path, "rb") as src:
            shutil.copyfileobj(src, f, 1 << 20)


def _write_npy(zf: zipfile.ZipFile, name: str, arr) -> None:
    with zf.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


def _check_request(req: Mapping) -> list[dict]:
    if not isinstance(req, Mapping) or req.get("schema") != REQUEST_SCHEMA:
        raise BuildError("bad_request", f"not a {REQUEST_SCHEMA} request")
    for k in ("cut_ts", "now"):
        if isinstance(req.get(k), bool) or not isinstance(req.get(k), int):
            raise BuildError("bad_request", f"{k} is not an int")
    ws = req.get("windows")
    if not isinstance(ws, list) or not ws:
        raise BuildError("bad_request", "no windows")
    out = []
    for w in ws:
        sid, origin, h = parse_window_id(w["window_id"])
        if str(w.get("series_id", sid)) != sid or int(w.get("h", h)) != h \
                or int(w.get("origin", origin)) != origin:
            raise BuildError("bad_request", f"{w['window_id']}: fields disagree with the id")
        snap = str(w.get("snapshot") or "")
        if not _SNAPSHOT_RE.fullmatch(snap):
            raise BuildError("bad_request", f"{w['window_id']}: bad snapshot {snap!r}")
        out.append({**w, "series_id": sid, "origin": origin, "h": h, "snapshot": snap})
    return sorted(out, key=lambda w: str(w["window_id"]))


def build(req: Mapping, *, pool_root: Path | str, out_dir: Path | str) -> dict:
    """Build ``<out_dir>/windows.npz`` for the request; returns the reply.
    ``ok: False`` (and no file) when any window is not retired yet."""
    ws = _check_request(req)
    cut, now = int(req["cut_ts"]), int(req["now"])
    cap = int(req.get("history_cap") or HISTORY_CAP)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        return _build(ws, Path(pool_root), out, cut=cut, now=now, cap=cap)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _build(ws: list[dict], root: Path, out: Path, *, cut: int, now: int, cap: int) -> dict:
    by_snap: dict[str, list[dict]] = {}
    for w in ws:
        by_snap.setdefault(w["snapshot"], []).append(w)
    raws = {"history": _Raw(out / "history.raw", np.float32),
            "target": _Raw(out / "target.raw", np.float32),
            "target_ts": _Raw(out / "target_ts.raw", np.int64),
            "first_seen_ts": _Raw(out / "first_seen_ts.raw", np.int64)}
    rows: list[dict] = []                 # build order: snapshot, series, origin
    waiting: list[dict] = []
    try:
        for snap in sorted(by_snap):
            d = root / snap
            if not (d / "metadata.json").is_file():
                raise BuildError("snapshot_unavailable", snap)
            want = {str(w["snapshot_sha256"]) for w in by_snap[snap]}
            got = snapshot_sha256(d)
            if want != {got}:
                raise BuildError("snapshot_digest_mismatch", f"{snap}: {got} not {sorted(want)}")
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            series = {str(s["series_id"]): s for s in meta.get("series") or []}
            hist_len = int(meta.get("history_length") or HISTORY_CAP)
            clk = np.load(d / CLOCKS_FILE) if isinstance(meta.get("clocks"), dict) else None
            try:
                with np.load(d / "hist.npz") as hz, np.load(d / "tgt.npz") as tz:
                    for w in sorted(by_snap[snap], key=lambda x: (x["series_id"], x["origin"])):
                        sid = w["series_id"]
                        if sid not in series or sid not in hz.files or sid not in tz.files:
                            raise BuildError("window_not_in_snapshot", w["window_id"])
                        ts, seen = _series_clock(series[sid], sid, clk)
                        full = np.concatenate([np.asarray(hz[sid]), np.asarray(tz[sid])],
                                              axis=-1)
                        c = cut_window(full, ts, seen, origin=w["origin"], h=w["h"],
                                       history_length=hist_len, history_cap=cap)
                        why = window_retired(c["target_ts"], c["first_seen_ts"], cut=cut,
                                             now=now)
                        if why is not None:
                            waiting.append({"window_id": w["window_id"], "reason": why})
                            continue
                        n = {k: raws[k].write(c[k]) for k in raws}
                        rows.append({**w, "s": series[sid], "n": n,
                                     "channels": 1 if c["target"].ndim == 1
                                     else int(c["target"].shape[0]),
                                     "history_len": int(c["history"].shape[-1])})
            finally:
                if clk is not None:
                    clk.close()
    finally:
        for r in raws.values():
            r.close()
    if waiting:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        return {"schema": REPLY_SCHEMA, "ok": False, "code": "not_retired",
                "waiting": sorted(waiting, key=lambda x: x["window_id"])[:50],
                "n_waiting": len(waiting)}
    return _finish(rows, raws, out, cut=cut, now=now, cap=cap)


def _finish(rows: list[dict], raws: dict, out: Path, *, cut: int, now: int, cap: int) -> dict:
    n = len(rows)
    offs = {k: np.zeros(n + 1, np.int64) for k in ("history", "target", "target_ts")}
    for i, r in enumerate(rows):
        for k in offs:
            offs[k][i + 1] = offs[k][i] + r["n"][k]
    cols: dict[str, list] = {k: [] for k in (*_STR_COLS, *_INT_COLS, "embargo_window")}
    for w in rows:
        s = w["s"]
        vals = {"window_id": w["window_id"], "series_id": w["series_id"],
                "snapshot": w["snapshot"], "source": str(s.get("source") or w["series_id"]),
                "feed": str(s.get("feed") or s.get("source") or w["series_id"]),
                "freq": str(s.get("freq") or ""), "band": str(w.get("band") or ""),
                "h": int(w["h"]), "origin_ts": int(w["origin"]),
                "n_channels": int(w["channels"]), "history_len": int(w["history_len"]),
                "seasonal_period": int(s.get("seasonal_period") or 1),
                "embargo_window": bool(w.get("embargo_window", False))}
        for k, v in vals.items():
            cols[k].append(v)
    npz = out / NPZ_NAME
    with zipfile.ZipFile(npz, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k, r in raws.items():
            _stream_npy(zf, k, r)
        _write_npy(zf, "history_offsets", offs["history"])
        _write_npy(zf, "target_offsets", offs["target"])
        _write_npy(zf, "time_offsets", offs["target_ts"])
        for k in _STR_COLS:
            _write_npy(zf, k, np.asarray(cols[k], dtype=str))
        for k in _INT_COLS:
            _write_npy(zf, k, np.asarray(cols[k], dtype=np.int64))
        _write_npy(zf, "embargo_window", np.asarray(cols["embargo_window"], dtype=bool))
    for r in raws.values():
        r.path.unlink(missing_ok=True)
    tts_max, fs_max = _raw_max(npz, "target_ts"), _raw_max(npz, "first_seen_ts")
    if n and (tts_max > cut or fs_max > cut or tts_max > now):
        npz.unlink()
        raise BuildError("embargo_violation", "a written clock reaches the cut")
    return {"schema": REPLY_SCHEMA, "ok": True, "npz": str(npz), "sha256": _file_sha(npz),
            "bytes": npz.stat().st_size, "windows": n, "history_cap": int(cap),
            "max_target_ts": tts_max if n else None, "max_first_seen_ts": fs_max if n else None,
            "snapshots": sorted({w["snapshot"] for w in rows})}


def npy_max(z: zipfile.ZipFile, name: str, *, chunk: int = 1 << 20) -> int | None:
    """The max of a 1-D int64 member, read in chunks (``None`` when empty)."""
    with z.open(name + ".npy") as f:
        major, _ = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if major == 1
                                 else np.lib.format.read_array_header_2_0(f))
        if fortran or len(shape) != 1:
            raise ValueError(f"{name}: not a 1-D C array")
        left, best = int(shape[0]), None
        step = max(1, chunk // np.dtype(dtype).itemsize)
        while left:
            k = min(step, left)
            a = np.frombuffer(f.read(k * np.dtype(dtype).itemsize), dtype=dtype)
            if a.size != k:
                raise ValueError(f"{name}: truncated")
            m = int(a.max())
            best = m if best is None else max(best, m)
            left -= k
        return best


def _raw_max(npz: Path, name: str) -> int:
    with zipfile.ZipFile(npz) as z:
        m = npy_max(z, name)
    return -(1 << 62) if m is None else m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cascade.eval.settlement_eval_build")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build windows.npz from the request on stdin")
    b.add_argument("--pool-root", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        reply = build(json.loads(sys.stdin.buffer.read()), pool_root=a.pool_root,
                      out_dir=a.out_dir)
    except BuildError as e:
        reply = {"schema": REPLY_SCHEMA, "ok": False, "code": e.code, "detail": e.detail}
    except (ValueError, KeyError, TypeError, OSError) as e:
        reply = {"schema": REPLY_SCHEMA, "ok": False, "code": "bad_request",
                 "detail": f"{type(e).__name__}: {e}"}
    print(json.dumps(reply, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BuildError", "REPLY_SCHEMA", "REQUEST_SCHEMA", "build", "cut_window", "npy_max",
           "parse_window_id", "snapshot_sha256", "window_retired"]
