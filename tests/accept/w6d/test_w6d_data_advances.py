"""W6d: a pool_d snapshot whose data never advances is never published or pinned.

2026-11-27 incident: the warm builder never re-pulled its forge mirror (frozen
at 2026-11-25 17:59), so eight 3-hourly snapshots re-cut the same data (newest
first-seen point 2026-11-25 17:34) under new names, all were published and
pinned, and lane D's draw refused every window. Publish (builder) and pin sync
(box) now refuse such a snapshot with ``stale_data``.
"""

from __future__ import annotations

import json

import pytest

from .helpers import AS_OF, HOUR, Recording, pub_mod, write_pool_d

PREFIX = "pool_d"
STEP = 10_800


def _guarded_mod():
    """The module under test; on a base without the guard every test fails
    here by assertion, not by an AttributeError / TypeError further down."""
    mod = pub_mod()
    for name in ("check_data_advances", "max_first_seen_ts", "MAX_DATA_AGE_SECONDS"):
        assert hasattr(mod, name), f"pool_d_publish has no {name}: no stale-data guard"
    import inspect
    for fn, arg in ((mod.publish_snapshot, "allow_stale"), (mod.sync_pins, "refused")):
        assert arg in inspect.signature(fn).parameters, f"{fn.__name__} takes no {arg}="
    return mod


def _store(tmp_path):
    mod = _guarded_mod()
    return mod, Recording(mod.FsPoolStore(tmp_path / "bucket"))


def _index(store):
    try:
        return [e["snapshot"] for e in json.loads(store.inner.get_bytes(f"{PREFIX}/index.json"))
                ["snapshots"]]
    except Exception:
        return []


def test_manifest_records_the_newest_first_seen_point(tmp_path):
    mod, store = _store(tmp_path)
    pool = write_pool_d(tmp_path / "p", first_seen=AS_OF - 2 * HOUR)
    man = mod.publish_snapshot(pool, store, prefix=PREFIX)
    assert man["max_first_seen_ts"] == AS_OF - 2 * HOUR
    assert mod.max_first_seen_ts(json.loads((pool / "metadata.json").read_text())) \
        == AS_OF - 2 * HOUR


def test_publish_refuses_the_same_data_under_a_new_name(tmp_path):
    mod, store = _store(tmp_path)
    frozen = AS_OF - 2 * HOUR
    mod.publish_snapshot(write_pool_d(tmp_path / "a", first_seen=frozen), store, prefix=PREFIX)
    store.writes.clear()
    later = write_pool_d(tmp_path / "b", as_of=AS_OF + STEP, first_seen=frozen)
    with pytest.raises(mod.PublishError) as err:
        mod.publish_snapshot(later, store, prefix=PREFIX)
    assert err.value.code == "stale_data" and "did not advance" in str(err.value)
    assert store.writes == []                       # nothing uploaded, index untouched
    assert _index(store) == [f"pool_d-{AS_OF}"]
    # the operator override still exists, and a re-publish stays idempotent
    man = mod.publish_snapshot(later, store, prefix=PREFIX, allow_stale=True)
    assert mod.publish_snapshot(later, store, prefix=PREFIX) == man


def test_publish_refuses_data_older_than_the_age_limit(tmp_path):
    """The incident numbers: as_of 2026-11-27 06:00, newest first-seen point
    2026-11-25 17:34 (36.4 h), even with no earlier snapshot to compare."""
    mod, store = _store(tmp_path)
    as_of, newest = 1_795_759_200, 1_795_628_056
    pool = write_pool_d(tmp_path / "p", as_of=as_of, first_seen=newest)
    with pytest.raises(mod.PublishError) as err:
        mod.publish_snapshot(pool, store, prefix=PREFIX)
    assert err.value.code == "stale_data" and "36.4 h" in str(err.value)
    assert _index(store) == []
    ok = write_pool_d(tmp_path / "q", as_of=as_of,
                      first_seen=as_of - mod.MAX_DATA_AGE_SECONDS)
    assert mod.publish_snapshot(ok, store, prefix=PREFIX)["snapshot"] == f"pool_d-{as_of}"


def test_cli_publish_exits_nonzero_on_stale_data(tmp_path):
    mod = _guarded_mod()
    bucket = tmp_path / "bucket"
    frozen = AS_OF - 2 * HOUR
    for k in range(2):
        pool = write_pool_d(tmp_path / f"p{k}", as_of=AS_OF + k * STEP, first_seen=frozen)
        rc = mod.main(["publish", "--pool-d", str(pool), "--fs-store", str(bucket),
                       "--prefix", PREFIX])
        assert rc == (0 if k == 0 else 2)


def _publish_unguarded(tmp_path, mod, store, firsts):
    """What a pre-guard builder published: one snapshot per 3 h, given data."""
    for k, fs in enumerate(firsts):
        pool = write_pool_d(tmp_path / f"s{k}", as_of=AS_OF + k * STEP, first_seen=fs,
                            shift=float(k))
        mod.publish_snapshot(pool, store, prefix=PREFIX, allow_stale=True)


def test_sync_never_pins_a_non_advancing_snapshot(tmp_path):
    mod, store = _store(tmp_path)
    frozen = AS_OF - 2 * HOUR
    _publish_unguarded(tmp_path, mod, store, [frozen, frozen, frozen])
    pins = tmp_path / "box" / "pins.json"
    root = tmp_path / "box" / "pool_d"
    refused: list = []
    with pytest.raises(mod.PublishError) as err:     # the newest is stale: fail loudly
        mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins, from_block=10,
                      refused=refused)
    assert err.value.code == "stale_data"
    assert [p["snapshot"] for p in json.loads(pins.read_text())["pins"]] == [f"pool_d-{AS_OF}"]
    assert [r["snapshot"] for r in refused] == [f"pool_d-{AS_OF + STEP}",
                                                 f"pool_d-{AS_OF + 2 * STEP}"]
    assert sorted(p.name for p in root.iterdir()) == [f"pool_d-{AS_OF}"]
    # fresh data arrives: it pins, the old refusals stay unpinned but are no longer fatal
    fresh = write_pool_d(tmp_path / "s3", as_of=AS_OF + 3 * STEP, first_seen=AS_OF + 2 * STEP)
    mod.publish_snapshot(fresh, store, prefix=PREFIX)
    added = mod.sync_pins(store, prefix=PREFIX, local_root=root, pins_path=pins, from_block=20)
    assert [p["snapshot"] for p in added] == [f"pool_d-{AS_OF + 3 * STEP}"]


def test_sync_checks_the_bytes_when_the_manifest_predates_the_field(tmp_path):
    mod, store = _store(tmp_path)
    frozen = AS_OF - 2 * HOUR
    _publish_unguarded(tmp_path, mod, store, [frozen, frozen])
    for k in range(2):                          # strip the field: a pre-guard manifest
        key = f"{PREFIX}/pool_d-{AS_OF + k * STEP}/manifest.json"
        man = json.loads(store.inner.get_bytes(key))
        man.pop("max_first_seen_ts")
        store.inner.put_bytes(key, json.dumps(man).encode())
    pins = tmp_path / "box" / "pins.json"
    with pytest.raises(mod.PublishError) as err:
        mod.sync_pins(store, prefix=PREFIX, local_root=tmp_path / "box" / "d", pins_path=pins,
                      from_block=10)
    assert err.value.code == "stale_data"
    assert len(json.loads(pins.read_text())["pins"]) == 1


def test_healthy_cadence_publishes_and_pins_everything(tmp_path):
    mod, store = _store(tmp_path)
    for k in range(4):
        pool = write_pool_d(tmp_path / f"h{k}", as_of=AS_OF + k * STEP, shift=float(k))
        mod.publish_snapshot(pool, store, prefix=PREFIX)
    added = mod.sync_pins(store, prefix=PREFIX, local_root=tmp_path / "box" / "d",
                          pins_path=tmp_path / "box" / "pins.json", from_block=1)
    assert len(added) == 4
