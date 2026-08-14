"""The public eval-pool composition doc (``cascade.shared.pool_status``) and its
dashboard consumers.

status/pool.json is presentational: aggregate series counts per domain x
granularity for each published snapshot, joined to rounds by the SAME
``effective_block`` rule validators select their scoring pool by. These tests
pin (1) the doc's shape and update semantics (idempotent per block, sorted,
capped), (2) that the round→snapshot mapping matches ``select_snapshot``, and
(3) that the publish CLI writes the doc best-effort — a manifest-bucket failure
warns, never fails the snapshot publish validators depend on.
"""

from __future__ import annotations

import dataclasses
import json

from cascade.shared.hippius import ObjectNotFound, StorageError
from cascade.shared.pool_status import (
    POOL_STATUS_KEY,
    active_snapshot,
    build_pool_status_entry,
    read_pool_status,
    snapshots,
    update_pool_status,
)


class _FakeStore:
    def __init__(self, *, fail: bool = False):
        self.objects: dict[str, bytes] = {}
        self.fail = fail
        self.acls: dict[str, str | None] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        if self.fail:
            raise StorageError(f"s3_put_failed: {key}: 500")
        self.objects[key] = text.encode("utf-8")
        self.acls[key] = acl

    def get_text(self, key):
        if self.fail:
            raise StorageError(f"s3_get_failed: {key}: 500")
        if key not in self.objects:
            raise ObjectNotFound(f"missing: {key}")
        return self.objects[key].decode("utf-8")


def _entry(block=7200, **over):
    base = dict(
        effective_block=block,
        as_of="2026-08-13",
        published_at="2026-08-13T03:12:00+00:00",
        n_series=2987,
        context_length=4096,
        horizon=64,
        sha256="ab12" * 16,
        breakdown={"weather": {"H": 2520}, "web_traffic": {"D": 85}},
    )
    base.update(over)
    return build_pool_status_entry(**base)


def test_entry_shape_and_breakdown_coercion():
    e = _entry(breakdown={"energy": {"15T": "40", "H": 7}})
    assert e["effective_block"] == 7200 and e["n_series"] == 2987
    assert e["breakdown"] == {"energy": {"15T": 40, "H": 7}}
    # entries survive a JSON round-trip unchanged (what the doc stores)
    assert json.loads(json.dumps(e)) == e


def test_update_is_idempotent_sorted_and_capped():
    store = _FakeStore()
    update_pool_status(store, _entry(14_400))
    update_pool_status(store, _entry(7_200))
    doc = read_pool_status(store)
    assert [s["effective_block"] for s in doc["snapshots"]] == [7_200, 14_400]

    # re-publishing a block replaces its entry, never duplicates it
    update_pool_status(store, _entry(7_200, n_series=5))
    doc = read_pool_status(store)
    assert [s["effective_block"] for s in doc["snapshots"]] == [7_200, 14_400]
    assert doc["snapshots"][0]["n_series"] == 5

    # the rolling window keeps the newest max_keep
    for blk in (21_600, 28_800, 36_000):
        update_pool_status(store, _entry(blk), max_keep=3)
    assert [s["effective_block"] for s in read_pool_status(store)["snapshots"]] == [
        21_600, 28_800, 36_000]
    # written public-read (dashboards fetch it anonymously)
    assert store.acls[POOL_STATUS_KEY] == "public-read"


def test_read_degrades_to_empty_on_absence_failure_and_garbage():
    assert read_pool_status(_FakeStore())["snapshots"] == []
    assert read_pool_status(_FakeStore(fail=True))["snapshots"] == []
    broken = _FakeStore()
    broken.objects[POOL_STATUS_KEY] = b"not json"
    assert read_pool_status(broken)["snapshots"] == []
    broken.objects[POOL_STATUS_KEY] = json.dumps({"snapshots": "nope"}).encode()
    assert read_pool_status(broken)["snapshots"] == []


def test_active_snapshot_matches_validator_selection_rule():
    doc = {"snapshots": [_entry(14_400), _entry(7_200), {"bad": True}]}
    # greatest effective_block <= epoch block (select_snapshot's rule)
    assert active_snapshot(doc, 14_400)["effective_block"] == 14_400
    assert active_snapshot(doc, 14_399)["effective_block"] == 7_200
    # a round preceding every snapshot falls back to the earliest
    assert active_snapshot(doc, 0)["effective_block"] == 7_200
    assert active_snapshot({"snapshots": []}, 7_200) is None
    assert snapshots("garbage") == []


def test_publish_cli_writes_composition_doc(tmp_path, monkeypatch):
    from cascade.pool import cli
    from cascade.shared import hippius
    from cascade.shared.config import load_chain_config

    pool_store, manifest_store = _FakeStore(), _FakeStore()
    pool_store.put_bytes = lambda key, data, **kw: pool_store.objects.__setitem__(key, bytes(data))
    cfg = load_chain_config()
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, pool_bucket="cascade-eval-pool"))
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(hippius, "pool_s3_store", lambda *_a, **_k: pool_store)
    monkeypatch.setattr(hippius, "open_manifest_store", lambda *_a, **_k: manifest_store)

    rc = cli.main(
        ["publish", "--sources", "synthetic", "--out", str(tmp_path / "stage"),
         "--effective-block", "7200", "--context-length", "128", "--horizon", "16",
         "--min-context", "32"])
    assert rc == 0
    doc = json.loads(manifest_store.objects[POOL_STATUS_KEY])
    (snap,) = doc["snapshots"]
    assert snap["effective_block"] == 7200
    assert snap["n_series"] > 0
    assert snap["breakdown"], "the domain x granularity breakdown must be published"
    total = sum(n for freqs in snap["breakdown"].values() for n in freqs.values())
    assert total == snap["n_series"]
    # the sha256 lets readers cross-check the manifest's eval_pool_sha256 pin
    index = json.loads(pool_store.objects["pool/index.json"])
    assert snap["sha256"] == index["snapshots"][0]["sha256"]


def test_publish_cli_survives_manifest_bucket_failure(tmp_path, monkeypatch, capsys):
    # The composition doc is presentational; losing it must not fail the
    # snapshot publish that validators score on.
    from cascade.pool import cli
    from cascade.shared import hippius
    from cascade.shared.config import load_chain_config

    pool_store = _FakeStore()
    pool_store.put_bytes = lambda key, data, **kw: pool_store.objects.__setitem__(key, bytes(data))
    cfg = load_chain_config()
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, pool_bucket="cascade-eval-pool"))
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(hippius, "pool_s3_store", lambda *_a, **_k: pool_store)
    monkeypatch.setattr(hippius, "open_manifest_store", lambda *_a, **_k: _FakeStore(fail=True))

    rc = cli.main(
        ["publish", "--sources", "synthetic", "--out", str(tmp_path / "stage"),
         "--effective-block", "7200", "--context-length", "128", "--horizon", "16",
         "--min-context", "32"])
    assert rc == 0
    assert "pool composition doc not published" in capsys.readouterr().err
    assert "pool/index.json" in pool_store.objects
