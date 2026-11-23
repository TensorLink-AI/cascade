"""W6d: pool_d pinning (publish / pin sync / history fetch), digests,
retention and the 3-hourly cadence (plan §3.0 W6d row; rev6.11 (d)).

On main each test fails at its first import of `cascade.pool.pool_d_publish`
(in PATHS).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from pathlib import Path

import pytest

from .helpers import (
    AS_OF,
    HORIZON,
    SERIES,
    Recording,
    pub_mod,
    series_record,
    snapshot_sha,
    write_pool_d,
)

PREFIX = "pool_d"
STEP = 10_800


def _store(tmp_path):
    mod = pub_mod()
    return mod, Recording(mod.FsPoolStore(tmp_path / "bucket"))


def _publish_series(tmp_path, mod, store, n=4, old_series=None):
    out = []
    for k in range(n):
        series = SERIES if (k or old_series is None) else old_series
        pool = write_pool_d(tmp_path / f"snap{k}", as_of=AS_OF + k * STEP, series=series,
                            shift=float(k))
        out.append(mod.publish_snapshot(pool, store, prefix=PREFIX, build_wall_seconds=900 + k))
    return out


def test_constants():
    mod = pub_mod()
    from cascade.validator.lane_d_inputs import POOL_PINS_SCHEMA

    assert mod.CADENCE_SECONDS == 10_800 and mod.MAX_BUILD_WALL_SECONDS == 10_800
    assert mod.PINS_SCHEMA == POOL_PINS_SCHEMA
    assert mod.MANIFEST_SCHEMA == "lane-d-pool-manifest/1"
    assert mod.INDEX_SCHEMA == "lane-d-pool-index/1"
    assert 0 < mod.MULTIPART_CHUNK_BYTES <= 512 * 1024 * 1024
    assert mod.TERMINAL_STATUSES == ("crowned", "lost", "closed")
    assert issubclass(mod.PublishError, ValueError)


def test_due_as_of_cadence():
    mod = pub_mod()
    assert mod.due_as_of(AS_OF) == AS_OF
    assert mod.due_as_of(AS_OF + 1) == AS_OF
    assert mod.due_as_of(AS_OF + STEP - 1) == AS_OF
    assert mod.due_as_of(AS_OF + STEP) == AS_OF + STEP
    dues = sorted({mod.due_as_of(AS_OF + s) for s in range(0, 86_400, 600)})
    assert len(dues) == 8 and all(b - a == STEP for a, b in zip(dues, dues[1:], strict=False))


def test_check_build_wall(caplog):
    mod = pub_mod()
    ok = mod.check_build_wall(as_of_ts=AS_OF, started_ts=AS_OF + 60, finished_ts=AS_OF + 3600)
    assert ok == {"ok": True, "wall_seconds": 3540, "limit_seconds": 10_800,
                  "deadline_ts": AS_OF + STEP}
    with caplog.at_level(logging.WARNING, logger="cascade.pool"):
        slow = mod.check_build_wall(as_of_ts=AS_OF, started_ts=AS_OF,
                                    finished_ts=AS_OF + STEP + 5)
        late = mod.check_build_wall(as_of_ts=AS_OF, started_ts=AS_OF + 9000,
                                    finished_ts=AS_OF + STEP)        # wall short, too late
    assert slow["ok"] is False and slow["wall_seconds"] == STEP + 5
    assert late["ok"] is False and late["wall_seconds"] == 1800
    hits = [r for r in caplog.records if "pool_d_build_wall_exceeded" in r.getMessage()]
    assert len(hits) == 2


def test_publish_writes_manifest_tars_index(tmp_path):
    mod, store = _store(tmp_path)
    pool = write_pool_d(tmp_path / "pool")
    man = mod.publish_snapshot(pool, store, prefix=PREFIX, build_wall_seconds=1234)
    snap = f"pool_d-{AS_OF}"
    meta = json.loads((pool / "metadata.json").read_text())
    assert man["schema"] == mod.MANIFEST_SCHEMA and man["snapshot"] == snap
    assert man["as_of_ts"] == AS_OF and man["sha256"] == snapshot_sha(pool)
    assert man["hist_key"] == f"{PREFIX}/{snap}/hist.tar"
    assert man["tgt_key"] == f"{PREFIX}/{snap}/tgt.tar"
    assert man["catalog_sha256"] == meta["catalog_sha256"]
    assert man["first_seen_roots"] == meta["first_seen_roots"]
    assert man["n_series"] == len(SERIES) and man["build_wall_seconds"] == 1234
    inner = store.inner
    assert hashlib.sha256(inner.get_bytes(man["hist_key"])).hexdigest() == man["hist_sha256"]
    assert hashlib.sha256(inner.get_bytes(man["tgt_key"])).hexdigest() == man["tgt_sha256"]
    assert json.loads(inner.get_bytes(f"{PREFIX}/{snap}/manifest.json")) == man
    index = json.loads(inner.get_bytes(f"{PREFIX}/index.json"))
    assert index == {"schema": mod.INDEX_SCHEMA, "snapshots": [
        {"snapshot": snap, "as_of_ts": AS_OF, "manifest_key": f"{PREFIX}/{snap}/manifest.json"}]}


def test_publish_idempotent_and_conflict(tmp_path):
    mod, store = _store(tmp_path)
    pool = write_pool_d(tmp_path / "pool")
    first = mod.publish_snapshot(pool, store, prefix=PREFIX)
    assert mod.publish_snapshot(pool, store, prefix=PREFIX) == first
    index = json.loads(store.inner.get_bytes(f"{PREFIX}/index.json"))
    assert len(index["snapshots"]) == 1
    other = write_pool_d(tmp_path / "other", shift=3.0)        # same name, other bytes
    with pytest.raises(mod.PublishError) as err:
        mod.publish_snapshot(other, store, prefix=PREFIX)
    assert err.value.code == "snapshot_conflict"
    assert json.loads(store.inner.get_bytes(f"{PREFIX}/{first['snapshot']}/manifest.json")) \
        == first


def test_sync_pins_feeds_pin_file_pool(tmp_path):
    mod, store = _store(tmp_path)
    mans = _publish_series(tmp_path, mod, store, n=3)
    pins_path = tmp_path / "box" / "pool_d_pins.json"
    root = tmp_path / "box" / "pool_d"
    added = mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins_path,
                          from_block=1000)
    assert [p["snapshot"] for p in added] == [m["snapshot"] for m in mans]
    assert mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins_path,
                         from_block=1100) == []
    doc = json.loads(pins_path.read_text())
    assert doc["schema"] == mod.PINS_SCHEMA
    assert [p["as_of_ts"] for p in doc["pins"]] == sorted(p["as_of_ts"] for p in doc["pins"])
    for pin, man in zip(doc["pins"], mans, strict=True):
        assert pin["sha256"] == man["sha256"] and pin["from_block"] == 1000
        assert pin["hist_sha256"] == man["hist_sha256"]
        assert pin["tgt_sha256"] == man["tgt_sha256"]
        assert pin["catalog_sha256"] == man["catalog_sha256"]
        assert not Path(pin["dir"]).is_absolute()
    from cascade.validator.lane_d import load_pool
    from cascade.validator.lane_d_inputs import PinFilePool

    got = PinFilePool(pins_path)(1000)
    assert len(got) == 3
    for pin in got:
        assert load_pool(pin["dir"], HORIZON)["sha256"] == pin["sha256"]
    assert PinFilePool(pins_path)(999) == []
    pool4 = write_pool_d(tmp_path / "snap4", as_of=AS_OF + 3 * STEP, shift=9.0)
    mod.publish_snapshot(pool4, store, prefix=PREFIX)
    new = mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins_path,
                        from_block=2000)
    assert [p["from_block"] for p in new] == [2000]
    assert len(PinFilePool(pins_path)(1500)) == 3 and len(PinFilePool(pins_path)(2000)) == 4


def test_sync_refuses_tampered_tar(tmp_path):
    mod, store = _store(tmp_path)
    (man,) = _publish_series(tmp_path, mod, store, n=1)
    data = bytearray(store.inner.get_bytes(man["tgt_key"]))
    data[-1024] ^= 0xFF
    store.inner.put_bytes(man["tgt_key"], bytes(data))
    pins_path = tmp_path / "box" / "pins.json"
    with pytest.raises(mod.PublishError) as err:
        mod.sync_pins(store, prefix=PREFIX, local_root=tmp_path / "box" / "d",
                      pins_path=pins_path, from_block=5)
    assert err.value.code == "digest_mismatch"
    assert not pins_path.exists() or json.loads(pins_path.read_text())["pins"] == []


def test_fetch_hist_never_reads_targets(tmp_path):
    """Invariant 8 on the eval-host path: only the manifest and hist.tar."""
    mod, store = _store(tmp_path)
    (man,) = _publish_series(tmp_path, mod, store, n=1)
    store.reads.clear()
    path = mod.fetch_hist(store, prefix=PREFIX, snapshot=man["snapshot"],
                          out_dir=tmp_path / "evalhost")
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == man["hist_sha256"]
    assert man["tgt_key"] not in store.reads
    assert not any(k.endswith("tgt.tar") for k in store.reads)
    assert all(not p.name.endswith("tgt.tar") for p in (tmp_path / "evalhost").rglob("*"))
    data = bytearray(store.inner.get_bytes(man["hist_key"]))
    data[100] ^= 0xFF
    store.inner.put_bytes(man["hist_key"], bytes(data))
    with pytest.raises(mod.PublishError) as err:
        mod.fetch_hist(store, prefix=PREFIX, snapshot=man["snapshot"],
                       out_dir=tmp_path / "evalhost2")
    assert err.value.code == "digest_mismatch"


def test_digest_binds_embargo_and_catalog(tmp_path):
    """The pinned sha256 is what carries the effective embargo_hours and the
    catalog sha: change either and the content address changes."""
    mod, store = _store(tmp_path)
    base = write_pool_d(tmp_path / "a")
    meta = json.loads((base / "metadata.json").read_text())
    for edit in ("embargo", "catalog"):
        d = write_pool_d(tmp_path / edit)
        m = json.loads((d / "metadata.json").read_text())
        if edit == "embargo":
            m["feeds"]["meter"]["embargo_hours"] = 12.0
        else:
            m["catalog_sha256"] = hashlib.sha256(b"another catalog").hexdigest()
        (d / "metadata.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        assert snapshot_sha(d) != snapshot_sha(base)
    man = mod.publish_snapshot(base, store, prefix=PREFIX)
    assert "embargo_hours" not in json.dumps(man)          # never re-stated outside the pin
    assert man["catalog_sha256"] == meta["catalog_sha256"]


def _retention_world(tmp_path):
    mod, store = _store(tmp_path)
    old = (("old_feed~oldf.c0", "oldf", 400),)
    mans = _publish_series(tmp_path, mod, store, n=4, old_series=old)
    pins = tmp_path / "box" / "pins.json"
    root = tmp_path / "box" / "pool_d"
    mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins, from_block=10)
    old_wid = f"old_feed~oldf.c0:{series_record('old_feed~oldf.c0', 'oldf', 400)['target_ts'][0]}" \
              f":h{HORIZON}"
    return mod, mans, pins, root, old_wid


def test_retention_plan_rules(tmp_path):
    mod, mans, pins, _root, old_wid = _retention_world(tmp_path)
    names = [m["snapshot"] for m in mans]
    a = [m["as_of_ts"] for m in mans]
    reveal = a[1] + 48 * 3600                       # keeps as_of > a[1]
    state = {"entries": {
        "e1": {"status": "pending", "reveal_ts": reveal, "frozen_set": None},
        "done": {"status": "lost", "reveal_ts": a[0] - 10 * 86_400, "frozen_set": None},
    }}
    plan = mod.retention_plan(pins, state)
    assert plan == {"keep": names[2:], "expire": names[:2]}
    # a frozen window in the oldest snapshot keeps it
    state["entries"]["e2"] = {"status": "hold", "reveal_ts": reveal,
                              "frozen_set": {"frozen": True, "window_ids": [old_wid]}}
    assert mod.retention_plan(pins, state) == {"keep": [names[0], *names[2:]],
                                               "expire": [names[1]]}
    # a terminal entry's frozen set keeps nothing
    state["entries"]["e2"]["status"] = "crowned"
    assert mod.retention_plan(pins, state)["expire"] == names[:2]
    # no open entry: only the newest is kept
    assert mod.retention_plan(pins, {"entries": {}}) == {"keep": names[3:],
                                                         "expire": names[:3]}
    # an open house entry counts; one without reveal_ts keeps everything
    house = {"entries": {}, "house_entry": {"status": "pending", "reveal_ts": a[0] + 48 * 3600}}
    assert mod.retention_plan(pins, house)["expire"] == names[:1]
    blind = {"entries": {"x": {"status": "pending", "frozen_set": None}}}
    assert mod.retention_plan(pins, blind) == {"keep": names, "expire": []}
    # a shorter embargo keeps fewer
    short = {"entries": {"e1": {"status": "pending", "reveal_ts": a[2], "frozen_set": None}}}
    assert mod.retention_plan(pins, short, embargo_hours=0.0)["keep"] == names[3:]


def test_apply_retention(tmp_path):
    mod, mans, pins, root, _old_wid = _retention_world(tmp_path)
    names = [m["snapshot"] for m in mans]
    state = {"entries": {"e1": {"status": "pending", "reveal_ts": mans[1]["as_of_ts"] + 48 * 3600,
                                "frozen_set": None}}}
    plan = mod.retention_plan(pins, state)
    removed = mod.apply_retention(pins, root, plan)
    assert removed == names[:2]
    doc = json.loads(pins.read_text())
    assert [p["snapshot"] for p in doc["pins"]] == names[2:]
    for n in names[:2]:
        assert not (root / n).exists()
    from cascade.validator.lane_d import load_pool
    from cascade.validator.lane_d_inputs import PinFilePool

    for pin in PinFilePool(pins)(10):
        assert load_pool(pin["dir"], HORIZON)["sha256"] == pin["sha256"]
    outside = tmp_path / "keepme"
    outside.mkdir()
    with pytest.raises(mod.PublishError):
        mod.apply_retention(pins, outside, {"keep": [], "expire": names[2:3]})
    assert (root / names[2]).exists()


def test_hippius_store_lazy_and_scoped(monkeypatch):
    mod = pub_mod()
    from cascade.shared.config import load_chain_config

    for k in ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    socket_calls = []
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: socket_calls.append(a) or (_ for _ in ()).throw(
                            OSError("no network in tests")))
    import tomllib

    from cascade.shared.config.lane_d_ops import parse_lane_d_ops

    toml = Path(__file__).resolve().parents[3] / "chain.toml"
    raw = tomllib.loads(toml.read_text(encoding="utf-8"))
    ops = parse_lane_d_ops(raw.get("lane_d_ops"), raw.get("storage"))
    cfg = load_chain_config(toml)
    store = mod.open_pool_store(toml)
    assert isinstance(store, mod.HippiusPoolStore)
    assert store.bucket == ops.pool_d_bucket and store.prefix == ops.pool_d_prefix
    assert store.endpoint == cfg.storage.s3_endpoint
    assert socket_calls == []
    from cascade.shared.hippius import StorageError

    with pytest.raises(StorageError):
        store.get_bytes("pool_d/index.json")                # no credentials in env
    assert socket_calls == []


def test_cli_publish_sync_retention_evidence(tmp_path):
    mod = pub_mod()
    bucket = tmp_path / "bucket"
    for k in range(2):
        pool = write_pool_d(tmp_path / f"p{k}", as_of=AS_OF + k * STEP, shift=float(k))
        assert mod.main(["publish", "--pool-d", str(pool), "--fs-store", str(bucket),
                         "--prefix", PREFIX, "--started-ts", str(AS_OF + k * STEP),
                         "--finished-ts", str(AS_OF + k * STEP + 1200)]) == 0
    pins = tmp_path / "box" / "pins.json"
    assert mod.main(["sync", "--fs-store", str(bucket), "--prefix", PREFIX, "--pins", str(pins),
                     "--local-root", str(tmp_path / "box" / "d"), "--from-block", "7"]) == 0
    assert len(json.loads(pins.read_text())["pins"]) == 2
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"entries": {}}))
    assert mod.main(["retention", "--pins", str(pins), "--local-root", str(tmp_path / "box" / "d"),
                     "--state", str(state)]) == 0
    assert len(json.loads(pins.read_text())["pins"]) == 2           # dry run: nothing removed
    out = tmp_path / "w6d.json"
    assert mod.main(["evidence", "--fs-store", str(bucket), "--prefix", PREFIX,
                     "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    _check_evidence(doc)
    assert [s["build_wall_seconds"] for s in doc["snapshots"]] == [1200, 1200]
    assert mod.main(["retention", "--pins", str(pins), "--local-root", str(tmp_path / "box" / "d"),
                     "--state", str(state), "--apply"]) == 0
    assert len(json.loads(pins.read_text())["pins"]) == 1


# ── evidence (the owner's live run; skipped without $LANED_EVIDENCE) ─────────

def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _check_evidence(doc):
    assert doc["schema"] == "laned-evidence/w6d/1" and doc["wp"] == "W6d"
    assert doc["source"] == "builder"
    assert doc["generated_by"] == "cascade.pool.pool_d_publish evidence"
    body = {k: v for k, v in doc.items() if k != "digest"}
    assert doc["digest"] == hashlib.sha256(_canonical(body)).hexdigest(), \
        "evidence edited after it was generated"
    snaps = doc["snapshots"]
    assert len(snaps) >= 2
    for s in snaps:
        assert s["verified"] is True
        for k in ("sha256", "hist_sha256", "tgt_sha256"):
            assert len(s[k]) == 64 and set(s[k]) <= set("0123456789abcdef")
        assert s["as_of_ts"] % 10_800 == 0
        assert s["snapshot"] == f"pool_d-{s['as_of_ts']}"
    gaps = [b["as_of_ts"] - a["as_of_ts"] for a, b in zip(snaps, snaps[1:], strict=False)]
    assert doc["cadence_ok"] is True and all(g == 10_800 for g in gaps)
    assert doc["walls_ok"] is True
    assert all(s["build_wall_seconds"] is not None and s["build_wall_seconds"] < 10_800
               for s in snaps)


def _evidence_doc():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE unset: the owner's live run writes evidence/w6d.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_evidence_from_live_run():
    pub_mod()
    _check_evidence(_evidence_doc())


def test_evidence_names_the_pool_d_bucket():
    pub_mod()
    doc = _evidence_doc()
    assert doc["bucket"] and doc["prefix"]
    assert doc["bucket"] != "fixture"
