"""W6d: publish SPENT pool_d snapshots for miners (owner 2026-11-27/28).

``cascade.pool.pool_d_spent`` publishes a pinned snapshot only once no entry
still to be judged can see any of its values: retention expired it, and every
target point was first seen (and every history point recorded) at or before
``cut = min(now, state_mtime, open reveals) - slack - embargo``. Torch-free;
the S3 side is an in-memory fake client (no network). On main every test fails
at the import of the new module.
"""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import AS_OF, HORIZON, T0, write_pool_d

HOUR = 3600
REPO = Path(__file__).resolve().parents[3]


def spent_mod():
    return importlib.import_module("cascade.pool.pool_d_spent")


class FakeS3:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.puts: list[str] = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)],
                "IsTruncated": False}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[key] = {"body": Path(path).read_bytes(), **(ExtraArgs or {})}
        self.puts.append(key)

    def put_object(self, Bucket, Key, Body, ContentType=None, ACL=None):
        self.objects[Key] = {"body": Body, "ContentType": ContentType, "ACL": ACL}
        self.puts.append(Key)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            err = Exception("missing")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
        return {"Body": io.BytesIO(self.objects[Key]["body"])}


def _world(tmp_path, n=3):
    """``n`` pins 3 h apart; the newest's data is the freshest."""
    root = tmp_path / "pool_d"
    pins = []
    for i in range(n):
        as_of = AS_OF + 3 * HOUR * i
        name = f"pool_d-{as_of}"
        write_pool_d(root / name, as_of=as_of)
        pins.append({"dir": f"pool_d/{name}", "sha256": f"{i:064x}", "from_block": 10 + i,
                     "snapshot": name, "as_of_ts": as_of})
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"schema": "lane-d-pool-pins/1", "pins": pins}))
    return p, pins


def _max_first_seen(pool_dir: Path) -> int:
    meta = json.loads((pool_dir / "metadata.json").read_text())
    return max(max(s["first_seen_ts"]) for s in meta["series"])


def _now_spending(pins_path, pin, *, slack=6.0, embargo=48.0) -> int:
    """The earliest ``now`` (state written at ``now``) at which ``pin`` is past the cut."""
    fs = _max_first_seen(pins_path.parent / pin["dir"])
    return max(fs, int(pin["as_of_ts"])) + int((slack + embargo) * HOUR)


def test_cut_and_protect_rules():
    m = spent_mod()
    state = {"entries": {"a": {"status": "pending", "reveal_ts": 5000 * HOUR},
                         "b": {"status": "lost", "reveal_ts": 1}}}
    # terminal entries do not protect; open ones, the state's age and now do
    assert m.protect_ts(state, now=6000 * HOUR, state_mtime=7000 * HOUR,
                        slack_hours=6) == 4994 * HOUR
    assert m.protect_ts({"entries": {}}, now=6000 * HOUR, state_mtime=5500 * HOUR,
                        slack_hours=0) == 5500 * HOUR
    # a malformed reveal_ts publishes nothing
    assert m.protect_ts({"entries": {"x": {"status": "pending"}}}, now=1, state_mtime=1) is None
    assert m.leak_cut(100 * HOUR, 48) == 52 * HOUR
    # never below the template embargo, even if an overlay lowers the deployed one
    assert m.effective_embargo_hours(24.0) == 48.0
    assert m.effective_embargo_hours(72.0) == 72.0


def test_window_rule_refuses_anything_that_may_be_judged():
    m = spent_mod()
    ts = [1000 + HOUR * i for i in range(HORIZON)]
    ok_fs = [ts[-1]] * HORIZON
    cut = ts[-1]
    assert m.window_spent(ts, ok_fs, cut=cut, now=cut) is None
    assert m.window_spent(ts, [cut + 1] + ok_fs[1:], cut=cut, now=cut) == "first_seen_after_cut"
    assert m.window_spent(ts, [0] * HORIZON, cut=ts[0] - 1, now=cut) == "history_after_cut"
    assert m.window_spent(ts, ok_fs, cut=cut, now=ts[-1] - 1) == "future_target"
    with pytest.raises(m.EmbargoViolation):
        m.assert_window_spent("w", ts, [cut + 1] * HORIZON, cut=cut, now=cut)


def test_eligible_snapshot_is_never_published(tmp_path):
    """Before the cut passes, nothing is planned and a forced build refuses
    and leaves no files behind."""
    m = spent_mod()
    pins_path, pins = _world(tmp_path)
    now = _now_spending(pins_path, pins[0]) - 1          # one second early
    plan = m.spent_plan(pins_path, {"entries": {}}, now=now, state_mtime=now,
                        embargo_hours=48.0)
    assert plan["publish"] == []
    assert {w["snapshot"] for w in plan["waiting"]} == {p["snapshot"] for p in pins}
    out = tmp_path / "stage" / "x"
    with pytest.raises(m.EmbargoViolation):
        m.build_preview(pins[0], pins_path.parent / pins[0]["dir"], out, cut=plan["cut_ts"],
                        now=now)
    assert not out.exists()


def test_open_entry_and_retention_hold_a_snapshot(tmp_path):
    m = spent_mod()
    pins_path, pins = _world(tmp_path)
    now = _now_spending(pins_path, pins[-1]) + 10 * HOUR
    free = m.spent_plan(pins_path, {"entries": {}}, now=now, state_mtime=now, embargo_hours=48.0)
    # retention always keeps the newest pin
    assert [p["snapshot"] for p in free["publish"]] == [p["snapshot"] for p in pins[:-1]]
    assert free["waiting"] == [{"snapshot": pins[-1]["snapshot"], "reason": "retained"}]
    # an open entry revealed long ago pulls the cut back before every snapshot
    old = {"entries": {"e": {"status": "pending", "reveal_ts": int(pins[0]["as_of_ts"])}}}
    held = m.spent_plan(pins_path, old, now=now, state_mtime=now, embargo_hours=48.0)
    assert held["publish"] == []
    bad = {"entries": {"e": {"status": "pending", "reveal_ts": "soon"}}}
    assert m.spent_plan(pins_path, bad, now=now, state_mtime=now,
                        embargo_hours=48.0)["publish"] == []
    # a stale state file (validator not writing) also holds the cut back
    stale = m.spent_plan(pins_path, {"entries": {}}, now=now, state_mtime=AS_OF,
                         embargo_hours=48.0)
    assert stale["publish"] == []


def _house_cache(tmp_path, pool_dir: Path, sid: str) -> tuple[Path, Path]:
    m = spent_mod()
    meta = json.loads((pool_dir / "metadata.json").read_text())
    s = next(x for x in meta["series"] if x["series_id"] == sid)
    hist = np.load(pool_dir / "hist.npz")[sid]
    wid = f"{sid}:{s['target_ts'][0]}:h{HORIZON}"
    tgt = np.load(pool_dir / "tgt.npz")[sid].astype(np.float64)
    q = np.stack([tgt + d for d in np.linspace(-1, 1, 9)], axis=-1)
    junk = np.zeros((2, HORIZON, 9))
    d = tmp_path / "score_cache" / "eval"
    d.mkdir(parents=True)
    digest = "ab" * 32
    np.savez(d / f"{digest}-4096-{HORIZON}.npz",
             window_ids=np.asarray(["other:1:h64", m.hist_key(hist, wid), "z:2:h64"]),
             quantiles=np.concatenate([junk[:1], q[None], junk[1:]]))
    ptr = tmp_path / "house_pointer.json"
    ptr.write_text(json.dumps({"digest": digest, "reveal_ts": 1}))
    return tmp_path / "score_cache", ptr


def test_future_target_is_left_out_not_published(tmp_path):
    m = spent_mod()
    pins_path, pins = _world(tmp_path, n=2)
    d = pins_path.parent / pins[0]["dir"]
    meta = json.loads((d / "metadata.json").read_text())
    s = meta["series"][0]
    now = _now_spending(pins_path, pins[0])
    s["target_ts"] = [now - HOUR * 10 + HOUR * i for i in range(HORIZON)]   # reaches the future
    s["first_seen_ts"] = [T0] * HORIZON
    (d / "metadata.json").write_text(json.dumps(meta))
    plan = m.spent_plan(pins_path, {"entries": {}}, now=now, state_mtime=now, embargo_hours=48.0)
    assert [p["snapshot"] for p in plan["publish"]] == [pins[0]["snapshot"]]
    man = m.build_preview(pins[0], d, tmp_path / "out", cut=plan["cut_ts"], now=now)
    assert man["counts"]["future_target"] == 1 and man["counts"]["windows"] == 2
    z = np.load(tmp_path / "out" / "windows.npz")
    assert s["series_id"] not in set(z["series_id"].tolist())
    assert int(z["target_ts"].max()) <= now


def test_publish_end_to_end_idempotent(tmp_path, monkeypatch):
    m = spent_mod()
    pins_path, pins = _world(tmp_path, n=2)
    now = _now_spending(pins_path, pins[0])
    state = tmp_path / "lane_d_state.json"
    state.write_text(json.dumps({"entries": {}}))
    import os
    os.utime(state, (now, now))
    fake = FakeS3()

    class Store:
        def client(self):
            return fake

    monkeypatch.setattr(m, "_client", lambda _c: (Store(), "cascade-manifests",
                                                  "https://s3.example/cascade-manifests"))
    sid = "tsforge__meter~meter.c0"
    cache_dir, ptr = _house_cache(tmp_path, pins_path.parent / pins[0]["dir"], sid)
    argv = ["publish", "--chain-toml", str(REPO / "chain.toml"), "--pins", str(pins_path),
            "--state", str(state), "--stage", str(tmp_path / "stage"),
            "--score-cache-dir", str(cache_dir), "--house-pointer", str(ptr), "--now", str(now)]
    # dry run: nothing written
    assert m.main([*argv, "--dry-run"]) == 0
    assert fake.puts == []
    assert m.main(argv) == 0
    pre = f"lane-d/spent/pool-preview-{pins[0]['as_of_ts']}/"
    assert fake.puts == [pre + "windows.npz", pre + "README.md", pre + "manifest.json",
                         "lane-d/spent/index.json"]           # manifest last, then the index
    assert all(fake.objects[k]["ACL"] == "public-read" for k in fake.puts)
    man = json.loads(fake.objects[pre + "manifest.json"]["body"])
    assert man["schema"] == "lane-d-pool-preview/1" and man["counts"]["windows"] == 3
    assert man["max_first_seen_ts"] <= man["spent_rule"]["cut_ts"]
    assert man["max_target_start_ts"] <= man["spent_rule"]["cut_ts"]
    z = np.load(io.BytesIO(fake.objects[pre + "windows.npz"]["body"]))
    assert z["target"].shape == (3, HORIZON) and int(z["has_house"].sum()) == 1
    i = int(np.flatnonzero(z["has_house"])[0])
    assert str(z["series_id"][i]) == sid and np.isfinite(z["house_crps"][i])
    hist = np.load(pins_path.parent / pins[0]["dir"] / "hist.npz")[sid]
    lo, hi = z["history_offsets"][i], z["history_offsets"][i + 1]
    np.testing.assert_array_equal(z["history"][lo:hi], hist)
    assert int(z["first_seen_ts"].max()) <= man["spent_rule"]["cut_ts"]
    idx = json.loads(fake.objects["lane-d/spent/index.json"]["body"])
    assert idx["schema"] == "lane-d-spent-index/1"
    assert [p["as_of_ts"] for p in idx["pools"]] == [pins[0]["as_of_ts"]]
    assert idx["pools"][0]["windows_url"].endswith(pre + "windows.npz")
    # rerun: nothing new, the index is not rewritten
    fake.puts.clear()
    assert m.main(argv) == 0
    assert fake.puts == []


def test_index_keeps_a_hand_made_preview(tmp_path):
    m = spent_mod()
    fake = FakeS3()
    legacy = {"schema": "lane-d-pool-preview/1-sample", "counts": {"windows": 7},
              "files": {"windows.npz": {"sha256": "aa", "bytes": 9}}}
    fake.objects["lane-d/spent/pool-preview-1795759200/manifest.json"] = {
        "body": json.dumps(legacy).encode()}
    fake.objects["lane-d/spent/pool-preview-1795759200/windows.npz"] = {"body": b"x"}
    doc, wrote = m.refresh_index(fake, "b", base_url="https://h/b", now=T0)
    assert wrote and doc["pools"][0]["as_of_ts"] == 1795759200
    assert doc["pools"][0]["windows"] == 7
    assert m.list_published(fake, "b") == [1795759200]
