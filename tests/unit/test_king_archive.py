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


def test_generator_key_groups_by_hotkey():
    key = ka.generator_key_for_ref(f"cascade/gen-x@{SHA}", "hk-miner")
    assert key == "generators/hk-miner/sha256-" + "a" * 64 + ".tar"
    # same ref, disjoint prefixes: kings/ and generators/ never collide,
    # and two committers of the same ref each keep their own object
    assert key != ka.archive_key_for_ref(f"cascade/gen-x@{SHA}")
    assert key != ka.generator_key_for_ref(f"cascade/gen-x@{SHA}", "hk-other")


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


# ─────────────────── generators/ — every-participant snapshot ───────────────


def _participant(hotkey, uid, ref, commit_block):
    return {"hotkey": hotkey, "uid": uid, "gen_ref": ref, "commit_block": commit_block}


def _manifest_store_with_receipts(rounds):
    """rounds: list of (round_summary, participants|None). Builds the receipt
    index with per-round receipt_key pointers plus the full receipts behind
    them (a None participants list stores NO receipt — an unreadable round)."""
    from cascade.shared.hippius import RECEIPT_INDEX_KEY
    objects = {}
    index_rounds = []
    for rnd, participants in rounds:
        rnd = dict(rnd)
        hk = rnd.setdefault("validator_hotkey", "hk-val")
        key = f"receipts/{hk}/round-{rnd['round_id']}.json"
        rnd["receipt_key"] = key
        index_rounds.append(rnd)
        if participants is not None:
            objects[key] = json.dumps({"participants": participants}).encode("utf-8")
    objects[RECEIPT_INDEX_KEY] = json.dumps(
        {"schema": 2, "rounds": index_rounds}).encode("utf-8")
    return _FakeS3Store(objects)


def test_collect_participants_one_entry_per_hotkey():
    # the same ref committed by two hotkeys — BOTH are recorded, each under its
    # own (hotkey, ref) entry with its own commit_block; a hotkey's re-commit
    # keeps its EARLIEST reveal (a UID recycles; the block is the claim).
    store = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/k@{SHC}"),
         [_participant("hk-late", 9, f"cascade/a@{SHA}", 90),
          _participant("hk-early", 3, f"cascade/a@{SHA}", 50)]),
        (_round("r2", 200, king_ref=f"cascade/k@{SHC}"),
         [_participant("hk-early", 3, f"cascade/a@{SHA}", 120),   # re-commit, later block
          _participant("hk-b", 4, f"cascade/b@{SHB}", 150)]),
    ])
    from cascade.shared.hippius import read_receipt_index

    def read_receipt(key):
        return json.loads(store.get_text(key))

    gens, scanned = ka.collect_participant_refs(read_receipt_index(store), read_receipt)
    assert scanned == ["r1", "r2"]
    early = gens[("hk-early", f"cascade/a@{SHA}")]
    late = gens[("hk-late", f"cascade/a@{SHA}")]
    assert early["commit_block"] == 50      # earliest reveal kept over the re-commit
    assert early["rounds_seen"] == 2        # rounds, not appearances
    assert early["first_round_id"] == "r1" and early["last_round_id"] == "r2"
    assert late["commit_block"] == 90 and late["rounds_seen"] == 1
    assert gens[("hk-b", f"cascade/b@{SHB}")]["rounds_seen"] == 1


def test_collect_participants_unreadable_receipt_stays_unscanned():
    store = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/k@{SHC}"),
         [_participant("hk", 1, f"cascade/a@{SHA}", 50)]),
        (_round("r2", 200, king_ref=f"cascade/k@{SHC}"), None),   # receipt missing
    ])
    from cascade.shared.hippius import ObjectNotFound, read_receipt_index

    def read_receipt(key):
        try:
            return json.loads(store.get_text(key))
        except ObjectNotFound:
            return None

    gens, scanned = ka.collect_participant_refs(read_receipt_index(store), read_receipt)
    assert scanned == ["r1"]                # r2 will be retried next sync
    assert set(gens) == {("hk", f"cascade/a@{SHA}")}


def test_collect_participants_reads_one_receipt_per_round():
    # two validators published the same round — its participants count ONCE.
    reads: list[str] = []
    r1a = dict(_round("r1", 100, king_ref=f"cascade/k@{SHC}"), validator_hotkey="hk-v1")
    r1b = dict(_round("r1", 100, king_ref=f"cascade/k@{SHC}"), validator_hotkey="hk-v2")
    store = _manifest_store_with_receipts([
        (r1a, [_participant("hk", 1, f"cascade/a@{SHA}", 50)]),
        (r1b, [_participant("hk", 1, f"cascade/a@{SHA}", 50)]),
    ])
    from cascade.shared.hippius import read_receipt_index

    def read_receipt(key):
        reads.append(key)
        return json.loads(store.get_text(key))

    gens, scanned = ka.collect_participant_refs(read_receipt_index(store), read_receipt)
    assert scanned == ["r1"] and len(reads) == 1
    assert gens[("hk", f"cascade/a@{SHA}")]["rounds_seen"] == 1


def test_sync_generators_snapshots_every_participant():
    fetched: list[str] = []
    manifest = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-a", 1, f"cascade/a@{SHA}", 40),
          _participant("hk-b", 2, f"cascade/b@{SHB}", 50),
          _participant("hk-c", 3, f"cascade/c@{SHC}", 60)]),
    ])
    archive = _FakeS3Store()

    res = ka.sync_generators(
        manifest_store=manifest, archive_store=archive, hub=HUB,
        endpoint="https://e", bucket="b", updated_at="t1",
        fetch=_fake_fetch_factory(fetched),
    )

    assert res.archived == 3 and res.rounds_scanned == 1 and not res.failed
    for hk, ref in (("hk-a", f"cascade/a@{SHA}"), ("hk-b", f"cascade/b@{SHB}"),
                    ("hk-c", f"cascade/c@{SHC}")):
        assert ka.generator_key_for_ref(ref, hk) in archive.objects
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    assert idx["schema"] == ka.GENERATOR_INDEX_SCHEMA
    assert idx["scanned_rounds"] == ["r1"]
    a = next(g for g in idx["generators"] if g["gen_ref"] == f"cascade/a@{SHA}")
    assert a["archive_key"].startswith("generators/hk-a/")
    assert a["archive_url"].endswith(a["archive_key"])
    assert a["tar_sha256"] and a["size_bytes"] > 0 and a["commit_block"] == 40


def test_sync_generators_shared_ref_lands_under_each_committer():
    # two hotkeys committed the SAME ref — one snapshot per miner dir, one db
    # entry per committer (the archive records, DEC-CA-0008 adjudicates).
    fetched: list[str] = []
    manifest = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-early", 1, f"cascade/a@{SHA}", 40),
          _participant("hk-late", 2, f"cascade/a@{SHA}", 70)]),
    ])
    archive = _FakeS3Store()
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t1",
                             fetch=_fake_fetch_factory(fetched))
    assert res.archived == 2
    early_key = ka.generator_key_for_ref(f"cascade/a@{SHA}", "hk-early")
    late_key = ka.generator_key_for_ref(f"cascade/a@{SHA}", "hk-late")
    assert archive.objects[early_key] == archive.objects[late_key]  # same bytes
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    by_hk = {g["hotkey"]: g for g in idx["generators"]}
    assert by_hk["hk-early"]["commit_block"] == 40
    assert by_hk["hk-late"]["commit_block"] == 70


def test_sync_generators_second_run_rescans_and_reuploads_nothing():
    fetched: list[str] = []
    fetch = _fake_fetch_factory(fetched)
    rounds = [(_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
               [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)])]
    archive = _FakeS3Store()
    ka.sync_generators(manifest_store=_manifest_store_with_receipts(rounds),
                       archive_store=archive, hub=HUB, endpoint="https://e",
                       bucket="b", updated_at="t1", fetch=fetch)
    tar = archive.objects[ka.generator_key_for_ref(f"cascade/a@{SHA}", "hk-a")]

    # a new round joins the window; the same ref participates again
    rounds.append((_round("r2", 200, king_ref=f"cascade/a@{SHA}"),
                   [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)]))
    fetched.clear()
    res = ka.sync_generators(manifest_store=_manifest_store_with_receipts(rounds),
                             archive_store=archive, hub=HUB, endpoint="https://e",
                             bucket="b", updated_at="t2", fetch=fetch)
    assert res.rounds_scanned == 1          # only r2 read; r1 already scanned
    assert res.archived == 0 and res.skipped == 1 and fetched == []
    assert archive.objects[ka.generator_key_for_ref(f"cascade/a@{SHA}", "hk-a")] == tar
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    entry = idx["generators"][0]
    assert entry["rounds_seen"] == 2        # additive across scans
    assert entry["archived_at"] == "t1"     # original snapshot stamp kept
    assert idx["scanned_rounds"] == ["r1", "r2"]


def test_sync_generators_retries_a_failed_fetch_without_rereading_the_round():
    from cascade.shared.hippius import StorageError

    def boom(ref, dest, hub):
        raise StorageError("hub down")

    rounds = [(_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
               [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)])]
    manifest = _manifest_store_with_receipts(rounds)
    archive = _FakeS3Store()
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t1", fetch=boom)
    assert res.failed == [f"hk-a:cascade/a@{SHA}"]
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    # the round IS scanned (attribution recorded) but the tar is still owed
    assert idx["scanned_rounds"] == ["r1"]
    assert idx["generators"][0]["tar_sha256"] is None

    # next run: the Hub recovered — the tar lands even though r1 isn't re-read
    fetched: list[str] = []
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t2",
                             fetch=_fake_fetch_factory(fetched))
    assert res.rounds_scanned == 0 and res.archived == 1 and not res.failed
    assert fetched == [f"cascade/a@{SHA}"]
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    assert idx["generators"][0]["tar_sha256"] and idx["generators"][0]["archived_at"] == "t2"


def test_sync_generators_preserves_entries_that_scrolled_off_the_window():
    fetched: list[str] = []
    fetch = _fake_fetch_factory(fetched)
    archive = _FakeS3Store()
    m1 = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)]),
    ])
    ka.sync_generators(manifest_store=m1, archive_store=archive, hub=HUB,
                       endpoint="https://e", bucket="b", updated_at="t1", fetch=fetch)

    # the window rolled (and a transient read could even be empty) — the db keeps `a`
    m2 = _manifest_store_with_receipts([])
    res = ka.sync_generators(manifest_store=m2, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t2", fetch=fetch)
    assert res.total_generators == 1
    idx = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())
    assert {g["gen_ref"] for g in idx["generators"]} == {f"cascade/a@{SHA}"}
    assert idx["scanned_rounds"] == ["r1"]   # scan history survives too


def test_sync_generators_dry_run_writes_nothing():
    fetched: list[str] = []
    manifest = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)]),
    ])
    archive = _FakeS3Store()
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", dry_run=True,
                             fetch=_fake_fetch_factory(fetched))
    assert res.would_archive == 1 and res.archived == 0
    assert fetched == []
    assert archive.objects == {}             # not even the index / scanned_rounds


def test_kings_and_generators_coexist_in_one_bucket():
    fetched: list[str] = []
    fetch = _fake_fetch_factory(fetched)
    manifest = _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-a", 1, f"cascade/a@{SHA}", 40),
          _participant("hk-b", 2, f"cascade/b@{SHB}", 50)]),
    ])
    archive = _FakeS3Store()
    ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                  endpoint="https://e", bucket="b", updated_at="t", fetch=fetch)
    ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                       endpoint="https://e", bucket="b", updated_at="t", fetch=fetch)
    # the king lands under BOTH prefixes; the non-king only under generators/
    assert ka.archive_key_for_ref(f"cascade/a@{SHA}") in archive.objects
    assert ka.generator_key_for_ref(f"cascade/a@{SHA}", "hk-a") in archive.objects
    assert ka.archive_key_for_ref(f"cascade/b@{SHB}") not in archive.objects
    assert ka.generator_key_for_ref(f"cascade/b@{SHB}", "hk-b") in archive.objects
    assert ka.KING_INDEX_KEY in archive.objects
    assert ka.GENERATOR_INDEX_KEY in archive.objects


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


# ─────────── permanently-gone generators: tombstone, park, stay green ────────


def _gone(message):
    """A fetch that fails the way a deleted/private upstream repo actually does."""
    def fetch(ref, dest, hub):
        from cascade.shared.hippius import StorageError
        raise StorageError(f"fetch of {ref} failed after 1 attempt(s): {message}")
    return fetch


# The four shapes seen in production, verbatim from the scraper's logs.
DELETED_HUB_REV = ("Revision 'sha256:dead' not found in repository 'miner/gen-a'",
                   "gone")
DELETED_HF_REPO = ("404 Client Error. (Request ID: Root=1-6a7810b7-3c2ef5c6)", "gone")
PRIVATE_HUB_REPO = ("Client error '401 Unauthorized' for url "
                    "'https://registry.hippius.com/v2/miner/gen-a/manifests/sha256:dead'",
                    "private")
PRIVATE_HF_REPO = ("403 Client Error. (Request ID: Root=1-6a7810bf-202c13b5)", "private")


@pytest.mark.parametrize("message,reason",
                         [DELETED_HUB_REV, DELETED_HF_REPO, PRIVATE_HUB_REPO, PRIVATE_HF_REPO])
def test_permanent_fetch_failures_are_classified_not_blamed_on_the_network(message, reason):
    from cascade.shared.hippius import StorageError, classify_fetch_failure

    exc = StorageError(f"fetch of miner/gen-a@{SHA} failed after 1 attempt(s): {message}")
    assert classify_fetch_failure(exc) == reason


@pytest.mark.parametrize("message", [
    "read operation timed out",
    "503 Server Error: Service Unavailable",
    "connection reset by peer",
    "the registry said something nobody has seen before",   # unknown ⇒ keep retrying
])
def test_retryable_and_unknown_failures_stay_transient(message):
    from cascade.shared.hippius import FETCH_TRANSIENT, StorageError, classify_fetch_failure

    assert classify_fetch_failure(StorageError(message)) == FETCH_TRANSIENT


def test_hf_request_id_hex_is_not_read_as_a_status_code():
    """A bare '403'/'404' match would fire on the hex request id HF appends."""
    from cascade.shared.hippius import FETCH_TRANSIENT, StorageError, classify_fetch_failure

    exc = StorageError("connection reset (Request ID: Root=1-6a7810bf-403c13b5404efbca)")
    assert classify_fetch_failure(exc) == FETCH_TRANSIENT


def _one_gone_round():
    return _manifest_store_with_receipts([
        (_round("r1", 100, king_ref=f"cascade/a@{SHA}"),
         [_participant("hk-a", 1, f"cascade/a@{SHA}", 40)]),
    ])


def test_gone_generator_is_tombstoned_and_kept_out_of_failed():
    manifest, archive = _one_gone_round(), _FakeS3Store()
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t1",
                             fetch=_gone(DELETED_HUB_REV[0]))

    # NOT a failure: no re-run brings a deleted repo back, so the job stays green.
    assert res.failed == []
    assert [u["ref"] for u in res.unavailable] == [f"hk-a:cascade/a@{SHA}"]
    assert res.unavailable[0]["reason"] == "gone"

    entry = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())["generators"][0]
    assert entry["tar_sha256"] is None
    assert entry["unavailable"]["reason"] == "gone"
    assert entry["unavailable"]["attempts"] == 1
    assert entry["unavailable"]["first_failed_at"] == "t1"
    assert "not found in repository" in entry["unavailable"]["error"]


def test_gone_generator_parks_after_the_attempt_budget():
    manifest, archive = _one_gone_round(), _FakeS3Store()
    fetch = _gone(DELETED_HF_REPO[0])
    attempted = []

    def counting_fetch(ref, dest, hub):
        attempted.append(ref)
        return fetch(ref, dest, hub)

    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS):
        res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                                 endpoint="https://e", bucket="b",
                                 updated_at=f"t{run}", fetch=counting_fetch)
    assert len(attempted) == ka.UNAVAILABLE_MAX_ATTEMPTS
    entry = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())["generators"][0]
    assert entry["unavailable"]["attempts"] == ka.UNAVAILABLE_MAX_ATTEMPTS
    assert entry["unavailable"]["first_failed_at"] == "t0"      # never overwritten
    assert entry["unavailable"]["last_failed_at"] == f"t{ka.UNAVAILABLE_MAX_ATTEMPTS - 1}"

    # budget spent: the next run skips the fetch entirely but still reports it
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t-later",
                             fetch=counting_fetch)
    assert len(attempted) == ka.UNAVAILABLE_MAX_ATTEMPTS      # no new network call
    assert [u["ref"] for u in res.unavailable] == [f"hk-a:cascade/a@{SHA}"]
    assert res.failed == [] and res.archived == 0


def test_retry_unavailable_unparks_and_a_republished_generator_clears_its_tombstone():
    manifest, archive = _one_gone_round(), _FakeS3Store()
    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS):
        ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                           endpoint="https://e", bucket="b", updated_at=f"t{run}",
                           fetch=_gone(PRIVATE_HUB_REPO[0]))

    # the miner made the repo public again — --retry-unavailable picks it back up
    fetched: list[str] = []
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="t-fixed",
                             retry_unavailable=True, fetch=_fake_fetch_factory(fetched))
    assert fetched == [f"cascade/a@{SHA}"]
    assert res.archived == 1 and not res.unavailable and not res.failed
    entry = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())["generators"][0]
    assert entry["tar_sha256"] and "unavailable" not in entry


def test_transient_failures_never_park_and_keep_the_run_red():
    from cascade.shared.hippius import StorageError

    def boom(ref, dest, hub):
        raise StorageError("read operation timed out")

    manifest, archive = _one_gone_round(), _FakeS3Store()
    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS + 2):
        res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                                 endpoint="https://e", bucket="b",
                                 updated_at=f"t{run}", fetch=boom)
        # every single run retries it and reports it as a real failure
        assert res.failed == [f"hk-a:cascade/a@{SHA}"] and res.unavailable == []


def test_gone_king_is_tombstoned_in_the_db_rather_than_dropped():
    manifest = _manifest_store([_round("r1", 100, king_ref=f"cascade/a@{SHA}")])
    archive = _FakeS3Store()
    res = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                        endpoint="https://e", bucket="b", updated_at="t1",
                        fetch=_gone(DELETED_HUB_REV[0]))

    assert res.failed == [] and [u["ref"] for u in res.unavailable] == [f"cascade/a@{SHA}"]
    # "reigned, code lost" is the record the archive exists to keep — the throne
    # attribution is written even though there is no tar behind it.
    kings = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())["kings"]
    assert [k["gen_ref"] for k in kings] == [f"cascade/a@{SHA}"]
    assert kings[0]["tar_sha256"] is None and kings[0]["reign_rounds"] == 1
    assert kings[0]["unavailable"]["reason"] == "gone"

    # and it parks, exactly like a participant generator
    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS):
        res = ka.sync_kings(manifest_store=manifest, archive_store=archive, hub=HUB,
                            endpoint="https://e", bucket="b", updated_at=f"t{run}",
                            fetch=_gone(DELETED_HUB_REV[0]))
    assert res.unavailable and res.failed == []
    kings = json.loads(archive.objects[ka.KING_INDEX_KEY].decode())["kings"]
    assert kings[0]["unavailable"]["attempts"] == ka.UNAVAILABLE_MAX_ATTEMPTS


def test_a_changed_verdict_restarts_the_grace_period():
    """A flaky Hub that later answers a clean 404 must not inherit the timeouts'
    attempt count — the budget measures the condition being waited out."""
    from cascade.shared.hippius import StorageError

    def timeout(ref, dest, hub):
        raise StorageError("read operation timed out")

    manifest, archive = _one_gone_round(), _FakeS3Store()
    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS + 1):
        ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                           endpoint="https://e", bucket="b", updated_at=f"flaky{run}",
                           fetch=timeout)

    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", updated_at="gone0",
                             fetch=_gone(DELETED_HUB_REV[0]))
    entry = json.loads(archive.objects[ka.GENERATOR_INDEX_KEY].decode())["generators"][0]
    assert entry["unavailable"] == {
        "reason": "gone", "attempts": 1,
        "first_failed_at": "gone0", "last_failed_at": "gone0",
        "error": entry["unavailable"]["error"],
    }
    assert res.unavailable and res.failed == []


def test_dry_run_does_not_claim_a_parked_ref_would_be_archived():
    manifest, archive = _one_gone_round(), _FakeS3Store()
    for run in range(ka.UNAVAILABLE_MAX_ATTEMPTS):
        ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                           endpoint="https://e", bucket="b", updated_at=f"t{run}",
                           fetch=_gone(DELETED_HF_REPO[0]))

    fetched: list[str] = []
    res = ka.sync_generators(manifest_store=manifest, archive_store=archive, hub=HUB,
                             endpoint="https://e", bucket="b", dry_run=True,
                             fetch=_fake_fetch_factory(fetched))
    assert res.would_archive == 0 and fetched == []
    assert [u["ref"] for u in res.unavailable] == [f"hk-a:cascade/a@{SHA}"]
