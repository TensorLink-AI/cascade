"""King archive — throne-history distillation, content-addressed keys, and the
sync orchestration over an in-memory S3 store + a fake Hub fetch. No real Hub /
boto3 / chain needed."""

from __future__ import annotations

import json

import pytest

from cascade.shared import king_archive as ka

SHA = "sha256:" + "a" * 64
SHB = "sha256:" + "b" * 64
SHC = "sha256:" + "c" * 64


def _round(rid, block, *, king_ref, king_hk="hk-king", king_uid=1,
           chal_ref=None, chal_hk="hk-chal", chal_uid=2, won=False, status="scored"):
    return {
        "round_id": rid,
        "epoch_start_block": block,
        "status": status,
        "king_gen_ref": king_ref,
        "king_hotkey": king_hk,
        "king_uid": king_uid,
        "chal_gen_ref": chal_ref,
        "chal_hotkey": chal_hk,
        "chal_uid": chal_uid,
        "challenger_wins_round": won,
        "dethroned": won,
    }


# ───────────────────────────── collect_king_refs ────────────────────────────


def test_collect_counts_reigns_and_orders_by_first_seen():
    doc = {"rounds": [
        _round("r2", 200, king_ref=f"cascade/a@{SHA}"),
        _round("r1", 100, king_ref=f"cascade/a@{SHA}"),
        _round("r3", 300, king_ref=f"cascade/a@{SHA}"),
    ]}
    kings = ka.collect_king_refs(doc)
    assert list(kings) == [f"cascade/a@{SHA}"]
    e = kings[f"cascade/a@{SHA}"]
    assert e["reign_rounds"] == 3
    assert e["first_round_id"] == "r1" and e["first_epoch_start_block"] == 100
    assert e["last_round_id"] == "r3" and e["last_epoch_start_block"] == 300
    assert e["repo"] == "cascade/a" and e["digest"] == SHA
    assert e["hotkey"] == "hk-king" and e["uid"] == 1


def test_collect_captures_winning_challenger_even_without_a_reign_round():
    # a is king in r1 but the challenger b DETHRONES it — b must be archived as a
    # king even though no later round showing b reigning is in the index.
    doc = {"rounds": [
        _round("r1", 100, king_ref=f"cascade/a@{SHA}", chal_ref=f"cascade/b@{SHB}", won=True),
    ]}
    kings = ka.collect_king_refs(doc)
    assert set(kings) == {f"cascade/a@{SHA}", f"cascade/b@{SHB}"}
    b = kings[f"cascade/b@{SHB}"]
    assert b["reign_rounds"] == 0          # never seen reigning yet
    assert b["crowned_round_id"] == "r1"   # but crowned here
    assert b["hotkey"] == "hk-chal" and b["uid"] == 2


def test_collect_skips_rejected_rounds_and_unparseable_refs():
    doc = {"rounds": [
        _round("r1", 100, king_ref=f"cascade/a@{SHA}", status="rejected"),
        _round("r2", 200, king_ref="not-a-ref"),
        _round("r3", 300, king_ref=None),
        _round("r4", 400, king_ref=f"cascade/c@{SHC}"),
    ]}
    kings = ka.collect_king_refs(doc)
    assert list(kings) == [f"cascade/c@{SHC}"]


def test_collect_empty_index():
    assert ka.collect_king_refs({}) == {}
    assert ka.collect_king_refs({"rounds": []}) == {}


# ───────────────────────────── archive addressing ───────────────────────────


def test_archive_key_is_content_addressed_and_stable():
    key = ka.archive_key_for_ref(f"cascade/gen-x@{SHA}")
    assert key == "kings/cascade/gen-x/sha256-" + "a" * 64 + ".tar"
    # stable: same ref → same key (append-only / de-dup anchor)
    assert ka.archive_key_for_ref(f"cascade/gen-x@{SHA}") == key


def test_archive_url_joins_cleanly():
    url = ka.archive_url("https://acct.r2.cloudflarestorage.com/", "cascade-king-archive",
                         "kings/cascade/g/sha256-x.tar")
    assert url == ("https://acct.r2.cloudflarestorage.com/cascade-king-archive/"
                   "kings/cascade/g/sha256-x.tar")


# ────────────────────────────── fakes for sync ──────────────────────────────


class _FakeS3Store:
    """In-memory S3Store stand-in (put_bytes/get_bytes/put_text/get_text)."""

    def __init__(self, seed: dict | None = None):
        self.objects: dict[str, bytes] = dict(seed or {})

    def put_bytes(self, key, data, *, content_type="application/octet-stream", acl=None):
        self.objects[key] = data

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.objects[key] = text.encode("utf-8")

    def get_bytes(self, key):
        from cascade.shared.hippius import ObjectNotFound
        if key not in self.objects:
            raise ObjectNotFound(f"missing: {key}")
        return self.objects[key]

    def get_text(self, key):
        return self.get_bytes(key).decode("utf-8")


def _manifest_store(rounds):
    from cascade.shared.hippius import RECEIPT_INDEX_KEY
    doc = {"schema": 2, "rounds": rounds}
    return _FakeS3Store({RECEIPT_INDEX_KEY: json.dumps(doc).encode("utf-8")})


def _fake_fetch_factory(record):
    """A fetch(ref, dest, hub) that writes a generator.py whose bytes encode the
    ref — so distinct refs pack to distinct tars."""
    def fetch(ref, dest, hub):
        from pathlib import Path
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        (d / "generator.py").write_text(f"# {ref}\n")
        record.append(ref)
        return d
    return fetch


HUB = None  # HubConfig is unused by the fake fetch


def test_sync_archives_new_kings_and_writes_index():
    fetched: list[str] = []
    manifest = _manifest_store([
        _round("r1", 100, king_ref=f"cascade/a@{SHA}", chal_ref=f"cascade/b@{SHB}", won=True),
        _round("r2", 200, king_ref=f"cascade/b@{SHB}"),
    ])
    archive = _FakeS3Store()

    res = ka.sync_kings(
        manifest_store=manifest, archive_store=archive, hub=HUB,
        endpoint="https://acct.r2.cloudflarestorage.com", bucket="cascade-king-archive",
        updated_at="2026-07-23T00:00:00Z", fetch=_fake_fetch_factory(fetched),
    )

    assert res.archived == 2 and res.skipped == 0 and not res.failed
    assert set(fetched) == {f"cascade/a@{SHA}", f"cascade/b@{SHB}"}
    # both tars landed under content-addressed keys
    assert ka.archive_key_for_ref(f"cascade/a@{SHA}") in archive.objects
    assert ka.archive_key_for_ref(f"cascade/b@{SHB}") in archive.objects
    # the index db was written and links each king to its object + url
    idx = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())
    assert idx["schema"] == ka.KING_INDEX_SCHEMA
    assert idx["bucket"] == "cascade-king-archive"
    refs = {k["gen_ref"] for k in idx["kings"]}
    assert refs == {f"cascade/a@{SHA}", f"cascade/b@{SHB}"}
    a_entry = next(k for k in idx["kings"] if k["gen_ref"] == f"cascade/a@{SHA}")
    assert a_entry["archive_url"].endswith(a_entry["archive_key"])
    assert a_entry["tar_sha256"] and a_entry["size_bytes"] > 0
    assert a_entry["reign_rounds"] == 1


def test_sync_is_append_only_second_run_reuploads_nothing():
    fetched: list[str] = []
    rounds = [_round("r1", 100, king_ref=f"cascade/a@{SHA}")]
    manifest = _manifest_store(rounds)
    archive = _FakeS3Store()
    fetch = _fake_fetch_factory(fetched)

    first = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                          endpoint="https://e", bucket="b", updated_at="t1", fetch=fetch)
    assert first.archived == 1
    tar_before = archive.objects[ka.archive_key_for_ref(f"cascade/a@{SHA}")]

    # a later reign extends the throne history but the tar is content-addressed.
    manifest = _manifest_store(rounds + [_round("r2", 200, king_ref=f"cascade/a@{SHA}")])
    fetched.clear()
    second = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                           endpoint="https://e", bucket="b", updated_at="t2", fetch=fetch)
    assert second.archived == 0 and second.skipped == 1
    assert fetched == []  # nothing re-fetched
    assert archive.objects[ka.archive_key_for_ref(f"cascade/a@{SHA}")] == tar_before
    idx = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())
    entry = idx["kings"][0]
    assert entry["reign_rounds"] == 2               # metadata refreshed
    assert entry["archived_at"] == "t1"             # original archive stamp kept


def test_sync_preserves_kings_that_scrolled_off_the_receipt_window():
    # receipts/index.json is a ROLLING window: king `a` reigns, is archived, then
    # falls off the window while a new king `c` appears. The permanent db must
    # keep BOTH — `a` is not re-fetched and not dropped.
    fetched: list[str] = []
    fetch = _fake_fetch_factory(fetched)
    archive = _FakeS3Store()

    m1 = _manifest_store([_round("r1", 100, king_ref=f"cascade/a@{SHA}")])
    ka.sync_kings(manifest_store=m1, archive_store=archive, hub=HUB,
                  endpoint="https://e", bucket="b", updated_at="t1", fetch=fetch)
    a_tar = archive.objects[ka.archive_key_for_ref(f"cascade/a@{SHA}")]

    # window rolled: `a` is gone from the index, `c` is the current king
    fetched.clear()
    m2 = _manifest_store([_round("r9", 900, king_ref=f"cascade/c@{SHC}")])
    res = ka.sync_kings(manifest_store=m2, archive_store=archive, hub=HUB,
                        endpoint="https://e", bucket="b", updated_at="t2", fetch=fetch)

    assert fetched == [f"cascade/c@{SHC}"]                 # only the new king fetched
    assert archive.objects[ka.archive_key_for_ref(f"cascade/a@{SHA}")] == a_tar  # `a` kept
    idx = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())
    refs = {k["gen_ref"] for k in idx["kings"]}
    assert refs == {f"cascade/a@{SHA}", f"cascade/c@{SHC}"}
    assert res.total_kings == 2
    # `a`'s reign count never regresses even though it left the window
    a_entry = next(k for k in idx["kings"] if k["gen_ref"] == f"cascade/a@{SHA}")
    assert a_entry["reign_rounds"] == 1 and a_entry["archived_at"] == "t1"


def test_empty_receipt_read_never_blanks_the_db():
    fetched: list[str] = []
    fetch = _fake_fetch_factory(fetched)
    archive = _FakeS3Store()
    ka.sync_kings(manifest_store=_manifest_store([_round("r1", 100, king_ref=f"cascade/a@{SHA}")]),
                  archive_store=archive, hub=HUB, endpoint="https://e", bucket="b",
                  updated_at="t1", fetch=fetch)

    # a later run sees an empty index (e.g. a transient manifest read) — the db
    # must be left intact, not blanked.
    res = ka.sync_kings(manifest_store=_manifest_store([]), archive_store=archive, hub=HUB,
                        endpoint="https://e", bucket="b", updated_at="t2", fetch=fetch)
    assert res.total_kings == 1
    idx = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())
    assert {k["gen_ref"] for k in idx["kings"]} == {f"cascade/a@{SHA}"}


def test_dry_run_writes_nothing():
    fetched: list[str] = []
    manifest = _manifest_store([_round("r1", 100, king_ref=f"cascade/a@{SHA}")])
    archive = _FakeS3Store()
    res = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                        endpoint="https://e", bucket="b", dry_run=True,
                        fetch=_fake_fetch_factory(fetched))
    assert res.would_archive == 1 and res.archived == 0
    assert fetched == []             # no fetch on a dry run
    assert archive.objects == {}     # nothing written, not even the index


def test_sync_records_failure_without_dropping_prior_entry():
    def boom(ref, dest, hub):
        from cascade.shared.hippius import StorageError
        raise StorageError("hub down")

    manifest = _manifest_store([_round("r1", 100, king_ref=f"cascade/a@{SHA}")])
    archive = _FakeS3Store()
    res = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                        endpoint="https://e", bucket="b", updated_at="t", fetch=boom)
    assert res.failed == [f"cascade/a@{SHA}"] and res.archived == 0
    # index still written (empty kings list), so the run is idempotent-safe
    idx = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())
    assert idx["kings"] == []


def test_king_archive_config_falls_back_to_backup_endpoint(monkeypatch):
    from cascade.shared.config import StorageConfig

    monkeypatch.delenv("KING_ARCHIVE_S3_ACCESS_KEY", raising=False)
    storage = StorageConfig(
        hub_registry_url="", hub_namespace="cascade", s3_endpoint="", s3_region="",
        manifest_bucket="m", logs_bucket="l",
        backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
    )
    cfg, endpoint, bucket = ka.king_archive_config(storage)
    assert endpoint == "https://acct.r2.cloudflarestorage.com"
    assert bucket == "cascade-king-archive"          # default bucket
    assert cfg.region == "auto"                       # R2 default
    assert cfg.access_key_env == "BACKUP_S3_ACCESS_KEY"   # fell back to backup creds


def test_king_archive_config_requires_an_endpoint():
    from cascade.shared.config import StorageConfig
    from cascade.shared.hippius import StorageError

    storage = StorageConfig(
        hub_registry_url="", hub_namespace="cascade", s3_endpoint="", s3_region="",
        manifest_bucket="m", logs_bucket="l",
    )
    with pytest.raises(StorageError):
        ka.king_archive_config(storage)
