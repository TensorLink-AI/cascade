"""Shared fixtures for the W-EVALPUB tests (README.md has the contract).

``pool_d/1`` snapshots written by hand in the builder's shape (a pre-band one
and a band one with ``clk.npz``), a lane D report carrying the judged windows
the way ``cascade.validator.lane_d`` signs them, an in-memory S3 client and an
in-process eval host. Nothing here imports the new modules: each test reaches
them through :func:`mod`, which ASSERTS they exist, so on main every test
fails by assertion.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HOUR = 3600
DAY = 86400
T0 = 1_795_132_800                 # 2026-11-20 00:00 UTC
HORIZON = 64
REGION = 128                       # band snapshot: the stored target region
LAG = 1800                         # first seen 30 min after the data time
FAR = 4_000_000_000                # a frontier record after every bound


def mod(name: str):
    """The module ``name``; an assertion failure (never an ImportError) when
    the release does not have it."""
    assert importlib.util.find_spec(name) is not None, f"{name} is missing"
    return importlib.import_module(name)


def evals():
    return mod("cascade.pool.settlement_evals")


def builder():
    return mod("cascade.eval.settlement_eval_build")


def lane_d_cfg(**kw):
    from cascade.shared.config.lane_d import LaneDConfig

    return replace(LaneDConfig(), **({"forward_margin_hours": 0.0, "embargo_hours": 48.0,
                                      "max_mirror_lag_hours": 1000.0} | kw))


def values(sid: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(sid)))
    return (10.0 + rng.normal(0, 1.0, n)).astype(np.float32)


SERIES = (("tsforge__a~a.c0", "fa", 400), ("tsforge__b~b.c0", "fb", 300))


def write_pool(d: Path, *, end: int, band: bool, as_of: int | None = None,
               step: int = HOUR, freq: str = "H") -> Path:
    """A snapshot whose newest data point is ``end`` (one point per ``step``
    seconds, each first seen ``LAG`` later). ``band``: a ``clk.npz`` region of
    ``REGION`` points."""
    d.mkdir(parents=True, exist_ok=True)
    as_of = end + LAG + 600 if as_of is None else as_of
    recs, hist, tgt, clk = [], {}, {}, {}
    for sid, feed, n in SERIES:
        full = values(sid, n + HORIZON)
        hist[sid], tgt[sid] = full[:n], full[n:]
        ts = [end - step * (HORIZON - 1 - i) for i in range(HORIZON)]
        recs.append({"series_id": sid, "source": f"curated:{feed}", "feed": feed,
                     "freq": freq, "seasonal_period": 24, "history_len": n,
                     "target_ts": ts, "first_seen_ts": [t + LAG for t in ts],
                     "end_ts": ts[-1], "frontier": [[FAR, 0]]})
        rts = np.asarray([end - step * (REGION - 1 - i) for i in range(REGION)], np.int64)
        clk[sid] = np.stack([rts, rts + LAG])
    feeds = {f: {"mirror_lag_hours": 0.5, "max_lead_hours": -0.5, "publish_ahead": False,
                 "forecast_product": False, "embargo_hours": 48.0} for _, f, _ in SERIES}
    meta = {"schema": "pool_d/1", "snapshot": d.name, "as_of_ts": as_of, "horizon": HORIZON,
            "history_length": 16384, "first_seen_roots": [hashlib.sha256(b"r").hexdigest()],
            "feeds": feeds, "series": recs, "refused": []}
    if band:
        np.savez(d / "clk.npz", **clk)
        meta["clocks"] = {"file": "clk.npz",
                          "sha256": hashlib.sha256((d / "clk.npz").read_bytes()).hexdigest()}
    (d / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    np.savez(d / "hist.npz", **hist)
    np.savez(d / "tgt.npz", **tgt)
    return d


def load(d: Path, *, band: bool) -> dict:
    from cascade.validator.lane_d import load_pool

    if band:
        return load_pool(d, HORIZON, ("short",), max_horizon=2048, slots=2)
    return load_pool(d, HORIZON)


def report_for(boundary: int, pool: dict, windows: list[dict], *, boundary_ts: int,
               embargo_window: bool = False) -> dict:
    """A lane D report's judged-window fields (``lane_d._window_embargo_fields``)."""
    rows = []
    for w in windows:
        band = {"band": str(w["band"])} if w.get("band") else {}
        rows.append({**band, "window_id": w["window_id"], "source": w["source"],
                     "feed": w["feed"], "origin": int(w["target_ts"][0]),
                     "min_target_ts": int(min(w["target_ts"])),
                     "min_first_seen_ts": int(min(w["first_seen_ts"])),
                     "snapshot_sha256": pool["sha256"], "h": int(len(w["target_ts"])),
                     "embargo_window": embargo_window, "embargo_hours": 48.0})
    return {"schema": "lane-d-report/1", "boundary": boundary, "boundary_ts": boundary_ts,
            "pool": {"snapshot": pool["snapshot"], "sha256": pool["sha256"],
                     "snapshots": [{"snapshot": pool["snapshot"], "sha256": pool["sha256"]}]},
            "draw": {"seed": "0x" + "ab" * 32 + ":lane_d_windows"},
            "rows_sha256": "cd" * 32, "windows": rows, "rows": []}


class LocalHost:
    """The eval host in-process (the box side's EvalHost interface)."""

    def __init__(self, pool_root: Path, work: Path):
        self.pool_root, self.work = Path(pool_root), Path(work)
        self.requests: list[dict] = []

    def build(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        return builder().build(request, pool_root=self.pool_root,
                               out_dir=self.work / str(request["boundary"]))

    def fetch(self, boundary, dest):
        shutil.copyfile(self.work / str(boundary) / "windows.npz", dest)

    def cleanup(self, boundary):
        shutil.rmtree(self.work / str(boundary), ignore_errors=True)


class FakeS3:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.puts: list[str] = []
        self.acls: list[tuple[str, str]] = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        return {"Contents": [{"Key": k, "LastModified": self.objects[k].get(
                    "LastModified", datetime.fromtimestamp(T0, UTC))}
                             for k in sorted(self.objects) if k.startswith(Prefix)],
                "IsTruncated": False}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[key] = {"body": Path(path).read_bytes(), **(ExtraArgs or {})}
        self.puts.append(key)

    def put_object(self, Bucket, Key, Body, ContentType=None, ACL=None):
        self.objects[Key] = {"body": Body, "ContentType": ContentType, "ACL": ACL}
        self.puts.append(Key)

    def put_object_acl(self, Bucket, Key, ACL):
        self.objects[Key]["ACL"] = ACL
        self.acls.append((Key, ACL))

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            err = Exception("missing")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
        return {"Body": io.BytesIO(self.objects[Key]["body"])}
