"""W18: the sweeper over the boto3-shaped adapter (``S3Store``) and the
versioned in-memory :class:`helpers.FakeS3`. Contract: README.md."""

from __future__ import annotations

import hashlib

import pytest

from .helpers import (
    ARCHIVE_BUCKET,
    DUE,
    EXPECTED_KEEP,
    NOW,
    ORPHAN_KEY,
    STRAY_KEY,
    VAULT_BUCKET,
    WRITE_OPS,
    Entry,
    FakeS3,
    akey,
    days,
    keep_reason,
    ledger,
    log_problems,
    no_network_in_process,
    ops_of,
    policy,
    run_sweep,
    s3_stores,
    scenario_entries,
    seeded_fake,
    vkey,
    vs,
    write_plan,
)


def _setup(entries=None, clock=NOW):
    entries = entries or scenario_entries()
    fake = seeded_fake(entries, clock)
    vault, archive = s3_stores(fake)
    return entries, fake, vault, archive


def _clean(fake: FakeS3):
    assert fake.forbidden == [], f"S3 calls outside the allowed surface: {fake.forbidden}"
    assert fake.violations == [], "\n".join(fake.violations)


# ── the happy path ───────────────────────────────────────────────────────────


def test_due_losers_archived_then_every_vault_version_removed(tmp_path):
    entries, fake, vault, archive = _setup()
    with no_network_in_process():
        log = run_sweep(vault, archive, ledger(entries), tmp_path)
    assert log_problems(log) == []
    _clean(fake)
    for tag in DUE:
        e = entries[tag]
        for name, data in e.files.items():
            rel = e.key(name)
            assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) == data, (
                f"{tag}: archive copy of {rel} missing or wrong bytes")
            assert fake.versions(VAULT_BUCKET, vkey(rel)) == [], (
                f"{tag}: {rel} still has vault versions or a delete marker "
                f"{fake.versions(VAULT_BUCKET, vkey(rel))}: an Object-Lock vault is versioned, "
                "so the sweeper deletes every version by VersionId")
            assert ("archive", "done") in ops_of(log, rel), ops_of(log, rel)
            assert ("delete", "done") in ops_of(log, rel), ops_of(log, rel)
            arch = [a for a in log["actions"] if a["key"] == rel and a["op"] == "archive"][0]
            assert arch["sha256"] == hashlib.sha256(data).hexdigest()
    assert log["counts"]["archived"] == log["counts"]["deleted"] == 2 * len(DUE)
    assert log["counts"]["failed"] == 0


def test_archive_verified_by_reading_it_back_before_any_vault_delete(tmp_path):
    entries, fake, vault, archive = _setup()
    run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    for tag in DUE:
        for rel in entries[tag].rels():
            calls = fake.calls
            first_delete = next(i for i, (op, kw) in enumerate(calls)
                                if op == "delete_object" and kw.get("Bucket") == VAULT_BUCKET
                                and kw.get("Key") == vkey(rel)
                                or op == "delete_objects" and kw.get("Bucket") == VAULT_BUCKET
                                and any(o.get("Key") == vkey(rel)
                                        for o in kw["Delete"]["Objects"]))
            read_back = [i for i, (op, kw) in enumerate(calls[:first_delete])
                         if op == "get_object" and kw.get("Bucket") == ARCHIVE_BUCKET
                         and kw.get("Key") == akey(rel)]
            assert read_back, (f"{rel}: vault delete at call {first_delete} with no earlier "
                               "GetObject of the archive copy (verify by sha256 before delete)")


def test_retained_objects_kept_with_typed_reasons(tmp_path):
    entries, fake, vault, archive = _setup()
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    for tag, reason in EXPECTED_KEEP.items():
        for rel in entries[tag].rels():
            assert keep_reason(log, rel) == reason, (tag, rel, ops_of(log, rel))
            assert fake.live_data(VAULT_BUCKET, vkey(rel)) is not None, f"{tag}: {rel} removed"
            assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) is None, f"{tag}: {rel} archived"
    for rel in (ORPHAN_KEY, STRAY_KEY):
        assert keep_reason(log, rel) == "unreferenced", (rel, ops_of(log, rel))
        assert fake.live_data(VAULT_BUCKET, vkey(rel)) is not None
    assert VAULT_BUCKET not in {kw.get("Bucket") for _, kw in fake.writes()}, (
        "the sweeper never writes to the vault")


def test_ttl_boundary_is_inclusive_and_read_from_the_policy(tmp_path):
    entries, fake, vault, archive = _setup()
    log = run_sweep(vault, archive, ledger(entries), tmp_path, pol=policy(ttl=3))
    rel = entries["lost_young"].key("manifest.json")   # 20 - 17 = 3 >= 3
    assert ("delete", "done") in ops_of(log, rel), ops_of(log, rel)
    entries2, fake2, vault2, archive2 = _setup()
    log2 = run_sweep(vault2, archive2, ledger(entries2), tmp_path / "b", pol=policy(ttl=5))
    rel2 = entries2["lost_due"].key("manifest.json")    # 20 - 16 = 4 < 5
    assert keep_reason(log2, rel2) == "ttl_not_expired", ops_of(log2, rel2)
    assert log2["policy"] == {"loser_ttl_eras": 5, "archive_retention_days": 14}


# ── Object Lock ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["COMPLIANCE", "GOVERNANCE"])
def test_locked_object_never_deleted_then_swept_after_retention(tmp_path, mode):
    e = Entry("locked_loser", "lost", 1)
    entries = {e.tag: e}
    fake = FakeS3()
    until = NOW + days(1)
    locked_rel = e.key("model.safetensors")
    sib_rel = e.key("manifest.json")
    fake.seed(VAULT_BUCKET, vkey(locked_rel), e.files["model.safetensors"], mode=mode,
              retain_until=until)
    fake.seed(VAULT_BUCKET, vkey(sib_rel), e.files["manifest.json"])
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path)
    _clean(fake)
    assert log_problems(log) == []
    assert vkey(locked_rel) not in fake.deleted_keys(VAULT_BUCKET), (
        "a delete was ATTEMPTED on an object under Object Lock retention")
    kept = [a for a in log["actions"] if a["key"] == locked_rel and a["op"] == "keep"]
    assert kept and kept[0]["reason"] == "object_lock", ops_of(log, locked_rel)
    assert kept[0]["retain_until"][:19] == until.strftime("%Y-%m-%dT%H:%M:%S")
    assert fake.live_data(VAULT_BUCKET, vkey(sib_rel)) is not None, (
        "an entry is swept whole: a locked file keeps its siblings in the vault this run")
    assert fake.live_data(VAULT_BUCKET, vkey(locked_rel)) is not None

    later = NOW + days(2)
    fake.clock = later
    log2 = run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path / "r2", now=later)
    _clean(fake)
    for rel in (locked_rel, sib_rel):
        assert ("delete", "done") in ops_of(log2, rel), ops_of(log2, rel)
        assert fake.versions(VAULT_BUCKET, vkey(rel)) == []
        assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) == e.files[rel.rsplit("/", 1)[1]]


def test_locked_noncurrent_version_blocks_the_key(tmp_path):
    e = Entry("old_locked", "lost", 1)
    fake = FakeS3()
    rel = e.key("model.safetensors")
    fake.seed(VAULT_BUCKET, vkey(rel), b"older-version", mode="COMPLIANCE",
              retain_until=NOW + days(3))
    fake.seed(VAULT_BUCKET, vkey(rel), e.files["model.safetensors"])
    fake.seed(VAULT_BUCKET, vkey(e.key("manifest.json")), e.files["manifest.json"])
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path)
    _clean(fake)
    assert keep_reason(log, rel) == "object_lock", ops_of(log, rel)
    assert len(fake.versions(VAULT_BUCKET, vkey(rel))) == 2


# ── failures ─────────────────────────────────────────────────────────────────


def test_failed_archive_copy_means_no_delete(tmp_path):
    entries, fake, vault, archive = _setup()
    e = entries["lost_due"]
    bad = e.key("model.safetensors")
    fake.fail_put.add((ARCHIVE_BUCKET, akey(bad)))
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    assert log_problems(log) == []
    fails = [a for a in log["actions"] if a["key"] == bad and a["outcome"] == "failed"]
    assert fails and fails[0]["op"] == "archive" and fails[0]["reason"] == "archive_copy_failed"
    for rel in e.rels():
        assert vkey(rel) not in fake.deleted_keys(VAULT_BUCKET), (
            f"{rel} deleted although its entry's archive copy failed")
        assert fake.live_data(VAULT_BUCKET, vkey(rel)) == e.files[rel.rsplit("/", 1)[1]]
    sib = e.key("manifest.json")
    assert keep_reason(log, sib) == "entry_incomplete", ops_of(log, sib)
    for tag in ("unjudged_due", "expired_due"):
        for rel in entries[tag].rels():
            assert ("delete", "done") in ops_of(log, rel), "one failure stalls only its entry"
    assert log["counts"]["failed"] >= 1


def test_archive_sha256_mismatch_means_no_delete(tmp_path):
    entries, fake, vault, archive = _setup()
    e = entries["expired_due"]
    bad = e.key("manifest.json")
    fake.corrupt.add((ARCHIVE_BUCKET, akey(bad)))
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    fails = [a for a in log["actions"] if a["key"] == bad and a["outcome"] == "failed"]
    assert fails and fails[0]["reason"] == "archive_verify_failed", ops_of(log, bad)
    for rel in e.rels():
        assert vkey(rel) not in fake.deleted_keys(VAULT_BUCKET)
        assert fake.live_data(VAULT_BUCKET, vkey(rel)) is not None


def test_rerun_after_partial_failure_resumes_without_double_archive(tmp_path):
    entries, fake, vault, archive = _setup()
    e = entries["voided_due"]
    stuck = e.key("model.safetensors")
    fake.fail_delete.add((VAULT_BUCKET, vkey(stuck)))
    log1 = run_sweep(vault, archive, ledger(entries), tmp_path / "r1")
    assert any(a["key"] == stuck and a["op"] == "delete" and a["outcome"] == "failed"
               and a["reason"] == "delete_failed" for a in log1["actions"]), ops_of(log1, stuck)
    assert fake.live_data(VAULT_BUCKET, vkey(stuck)) is not None
    writes_after_r1 = len(fake.writes(ARCHIVE_BUCKET))

    fake.fail_delete.clear()
    log2 = run_sweep(vault, archive, ledger(entries), tmp_path / "r2")
    _clean(fake)
    assert log_problems(log2) == []
    assert ("archive", "already") in ops_of(log2, stuck), ops_of(log2, stuck)
    assert ("delete", "done") in ops_of(log2, stuck), ops_of(log2, stuck)
    assert len(fake.writes(ARCHIVE_BUCKET)) == writes_after_r1, (
        "the resumed run re-wrote an archive copy that was already verified")
    assert fake.versions(VAULT_BUCKET, vkey(stuck)) == []
    per_key = {}
    for op, kw in fake.writes(ARCHIVE_BUCKET):
        if op in ("put_object", "copy_object", "create_multipart_upload"):
            per_key[kw["Key"]] = per_key.get(kw["Key"], 0) + 1
    assert all(n == 1 for n in per_key.values()), f"archived more than once: {per_key}"


def test_second_clean_run_changes_nothing(tmp_path):
    entries, fake, vault, archive = _setup()
    run_sweep(vault, archive, ledger(entries), tmp_path / "r1")
    n_writes, n_deletes = len(fake.writes()), len(fake.deletes())
    log2 = run_sweep(vault, archive, ledger(entries), tmp_path / "r2")
    _clean(fake)
    assert len(fake.writes()) == n_writes and len(fake.deletes()) == n_deletes
    assert log2["counts"]["archived"] == log2["counts"]["deleted"] == 0
    assert log2["counts"]["failed"] == 0


def test_a_changed_partial_archive_copy_is_rewritten(tmp_path):
    entries, fake, vault, archive = _setup()
    e = entries["lost_due"]
    rel = e.key("model.safetensors")
    fake.seed(ARCHIVE_BUCKET, akey(rel), b"truncated")
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    assert ("archive", "done") in ops_of(log, rel), ops_of(log, rel)
    assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) == e.files["model.safetensors"]
    assert ("delete", "done") in ops_of(log, rel)


# ── owner-only archive ───────────────────────────────────────────────────────


def test_archive_writes_are_private_and_owner_only(tmp_path):
    entries, fake, vault, archive = _setup()
    run_sweep(vault, archive, ledger(entries), tmp_path)
    _clean(fake)
    writes = fake.writes(ARCHIVE_BUCKET)
    assert writes, "nothing was written to the archive"
    for op, kw in writes:
        if op in ("put_object", "copy_object", "create_multipart_upload"):
            assert kw.get("ACL") == "private", f"{op} {kw.get('Key')}: ACL={kw.get('ACL')!r}"
            assert not [k for k in kw if k.startswith("Grant")], kw
            assert str(kw.get("Key", "")).startswith("lane-d-archive/"), kw.get("Key")
    assert {kw.get("Bucket") for _, kw in fake.writes()} == {ARCHIVE_BUCKET}


def test_archive_overlapping_the_vault_is_refused(tmp_path):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    m = vs()
    vault = m.S3Store(fake, VAULT_BUCKET, "lane-d")
    for archive in (m.S3Store(fake, VAULT_BUCKET, "lane-d"),
                    m.S3Store(fake, VAULT_BUCKET, "lane-d/archive"),
                    m.S3Store(fake, VAULT_BUCKET, "")):
        with pytest.raises(m.SweepRefused) as ei:
            run_sweep(vault, archive, ledger(entries), tmp_path)
        assert ei.value.code == "archive_overlaps_vault"
    assert fake.writes() == [] and fake.deletes() == []


# ── dry-run ──────────────────────────────────────────────────────────────────


def test_dry_run_lists_actions_and_changes_nothing(tmp_path):
    entries, fake, vault, archive = _setup()
    before = {b: {k: [dict(v) for v in vs_] for k, vs_ in objs.items()}
              for b, objs in fake.buckets.items()}
    log = run_sweep(vault, archive, ledger(entries), tmp_path, live=False, plan=None)
    _clean(fake)
    assert log_problems(log) == []
    assert log["mode"] == "dry-run"
    assert [op for op, _ in fake.calls if op in WRITE_OPS or op.startswith("delete")
            or op == "abort_multipart_upload"] == [], "a dry-run wrote or deleted"
    assert fake.buckets == before
    for tag in DUE:
        for rel in entries[tag].rels():
            assert ops_of(log, rel) == [("archive", "planned"), ("delete", "planned")], (
                tag, ops_of(log, rel))
    for tag, reason in EXPECTED_KEEP.items():
        for rel in entries[tag].rels():
            assert keep_reason(log, rel) == reason
    assert log["counts"]["planned"] == 4 * len(DUE)
    assert log["counts"]["archived"] == log["counts"]["deleted"] == 0


# ── the D16 gate ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("plan_kind", ["open", "withdrawn", "deferred-to-stage", "absent_row",
                                       "no_register", "missing_file", "none"])
def test_live_refused_unless_d16_decided(tmp_path, plan_kind):
    entries, fake, vault, archive = _setup()
    m = vs()
    if plan_kind == "none":
        plan = None
    elif plan_kind == "missing_file":
        plan = tmp_path / "no_such_plan.md"
    elif plan_kind == "absent_row":
        plan = write_plan(tmp_path / "plan.md", None)
    elif plan_kind == "no_register":
        plan = write_plan(tmp_path / "plan.md", None, register=False)
    else:
        plan = write_plan(tmp_path / "plan.md", plan_kind)
    with pytest.raises(m.SweepRefused) as ei:
        run_sweep(vault, archive, ledger(entries), tmp_path, plan=plan)
    assert ei.value.code == "d16_not_decided", ei.value.code
    assert fake.calls == [], f"a refused live run touched storage: {fake.calls[:5]}"


def test_live_runs_when_the_register_says_decided(tmp_path):
    entries, fake, vault, archive = _setup()
    log = run_sweep(vault, archive, ledger(entries), tmp_path,
                    plan=write_plan(tmp_path / "plan.md", "decided"))
    assert log["mode"] == "live" and log["d16"] == "decided"
    assert log["counts"]["deleted"] > 0


# ── archive expiry (D8: owner-only archive 14 days, then delete) ─────────────


def test_archive_expires_after_retention_days_except_disputed(tmp_path):
    e = Entry("exp", "lost", 1)
    d = Entry("exp_disputed", "lost", 1)
    entries = {e.tag: e, d.tag: d}
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path / "r0")
    for x in (e, d):
        for rel in x.rels():
            assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) is not None

    t13 = NOW + days(13)
    fake.clock = t13
    log13 = run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path / "r13", now=t13)
    for rel in e.rels():
        assert keep_reason(log13, rel, "archive") == "archive_retention_not_expired", (
            ops_of(log13, rel, "archive"))
        assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) is not None

    t15 = NOW + days(15)
    fake.clock = t15
    log15 = run_sweep(vault, archive, ledger(entries, disputes=("exp_disputed",)),
                      tmp_path / "r15", now=t15)
    _clean(fake)
    assert log_problems(log15) == []
    for rel in e.rels():
        assert ("delete", "done") in ops_of(log15, rel, "archive"), ops_of(log15, rel, "archive")
        assert fake.versions(ARCHIVE_BUCKET, akey(rel)) == []
    for rel in d.rels():
        assert keep_reason(log15, rel, "archive") == "dispute_open"
        assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) is not None
    assert log15["counts"]["expired"] == len(e.rels())


def test_archive_retention_days_comes_from_the_policy(tmp_path):
    e = Entry("short", "lost", 1)
    entries = {e.tag: e}
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    pol = policy(days=2)
    run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path / "r0", pol=pol)
    t3 = NOW + days(3)
    fake.clock = t3
    log = run_sweep(vault, archive, ledger(entries, disputes=()), tmp_path / "r3", now=t3,
                    pol=pol)
    for rel in e.rels():
        assert ("delete", "done") in ops_of(log, rel, "archive"), ops_of(log, rel, "archive")


# ── the ledger ───────────────────────────────────────────────────────────────


def _bad_ledgers(entries):
    good = ledger(entries)
    out = {}
    x = dict(good, schema="laned_vault_ledger/0")
    out["schema"] = x
    x = dict(good, era=-1)
    out["era"] = x
    x = dict(good, entries=[dict(good["entries"][0], digest="sha256:abc")] + good["entries"][1:])
    out["digest"] = x
    x = dict(good, entries=[dict(good["entries"][0], closed_era=None)] + good["entries"][1:])
    out["loser_without_closed_era"] = x
    x = dict(good, disputes=["not-a-digest"])
    out["dispute"] = x
    x = dict(good)
    x.pop("entries")
    out["no_entries"] = x
    return out


@pytest.mark.parametrize("which", ["schema", "era", "digest", "loser_without_closed_era",
                                   "dispute", "no_entries"])
def test_malformed_ledger_refused_before_any_storage_call(tmp_path, which):
    entries, fake, vault, archive = _setup()
    m = vs()
    with pytest.raises(m.SweepRefused) as ei:
        run_sweep(vault, archive, _bad_ledgers(entries)[which], tmp_path)
    assert ei.value.code == "ledger_invalid"
    assert fake.writes() == [] and fake.deletes() == []
