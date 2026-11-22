"""W18 (RV6 B1): a storage error at any point of a live sweep never escapes
without the action log. The run stops at the first storage error (no further
storage write or delete), records it as a ``failed`` action typed
``storage_error``, writes the complete strict-JSON log of everything done so
far, and only then returns or raises. The CLI exits 1 and names the log file.
A re-run resumes idempotently. Contract: README.md "Storage errors"."""

from __future__ import annotations

import json

import pytest

from .helpers import (
    ARCHIVE_BUCKET,
    ARCHIVE_PREFIX,
    CHAIN_TOML,
    DUE,
    NOW,
    VAULT_BUCKET,
    VAULT_PREFIX,
    akey,
    all_vault_objects,
    iso,
    ledger,
    log_problems,
    ops_of,
    run_sweep,
    s3_stores,
    scenario_entries,
    seeded_fake,
    strict_loads,
    sweep_logged,
    vkey,
    vs,
    write_plan,
)

DELETE_OPS = ("delete_object", "delete_objects")


def _gone(fake, e) -> bool:
    return all(not fake.versions(VAULT_BUCKET, vkey(r)) for r in e.rels())


def _after_first_entry_deleted(entries):
    due = [entries[t] for t in DUE]
    return lambda fake: any(_gone(fake, e) for e in due)


# step -> the S3 operations that fail (503 SlowDown) once the first due entry is
# entirely gone from the vault. Each is a call every correct sweeper must make
# for the NEXT due entry (README: entries are swept one at a time, archive ->
# verify -> delete, before the next entry starts).
STEPS = {
    "vault_read": [("get_object", VAULT_BUCKET)],
    "archive_put": [("put_object", ARCHIVE_BUCKET), ("copy_object", ARCHIVE_BUCKET),
                    ("create_multipart_upload", ARCHIVE_BUCKET)],
    "archive_read_back": [("get_object", ARCHIVE_BUCKET)],
    "vault_version_list": [("list_object_versions", VAULT_BUCKET)],
    "vault_delete": [(op, VAULT_BUCKET) for op in DELETE_OPS],
}


def _check_log_after_storage_error(fake, log, entries, *, listed_vault: bool = True):
    """The invariants of a run that hit a storage error (README "Storage errors")."""
    assert log_problems(log) == [], log_problems(log)
    assert log["mode"] == "live"
    assert fake.injected, "the injected storage error never fired (test setup)"
    first_fault = fake.injected[0]

    # nothing destructive after the first storage error
    late = [(i, op, kw.get("Bucket"), kw.get("Key")) for i, (op, kw) in enumerate(fake.calls)
            if i > first_fault and op in DELETE_OPS]
    assert late == [], (f"W18: delete call(s) after the storage error at call {first_fault} "
                        f"({fake.calls[first_fault][0]}): {late[:4]}; the run must stop there")

    # the failure is on record, typed
    errs = [a for a in log["actions"]
            if a["outcome"] == "failed" and a["reason"] == "storage_error"]
    assert errs, ("W18: no action typed storage_error in the log of a run that hit a storage "
                  f"error; failed actions: {[a for a in log['actions'] if a['outcome'] == 'failed']}")
    assert log["counts"]["failed"] >= 1

    # every successful vault delete is on record (the only record of the deletion):
    # delete/done, or delete/failed storage_error when the error hit the
    # post-delete check of that very key
    vault_done = {a["key"] for a in log["actions"]
                  if a["store"] == "vault" and a["op"] == "delete" and (
                      a["outcome"] == "done" or a["reason"] == "storage_error")}
    archived = {a["key"] for a in log["actions"] if a["store"] == "vault"
                and a["op"] == "archive" and a["outcome"] in ("done", "already")}
    pre = VAULT_PREFIX + "/"
    deleted_rels = {k[len(pre):] for b, k, _ in fake.deleted_ok if b == VAULT_BUCKET}
    missing = sorted(deleted_rels - vault_done)
    assert missing == [], f"W18: vault objects deleted with no delete/done record: {missing}"
    assert deleted_rels <= archived, sorted(deleted_rels - archived)

    # every due entry is either swept whole (and fully recorded) or untouched
    whole = 0
    for tag in DUE:
        e = entries[tag]
        if _gone(fake, e):
            whole += 1
            for rel in e.rels():
                assert rel in vault_done, (tag, rel, ops_of(log, rel))
                assert fake.live_data(ARCHIVE_BUCKET, akey(rel)) == e.files[rel.rsplit("/", 1)[1]]
        else:
            for rel in e.rels():
                assert fake.live_data(VAULT_BUCKET, vkey(rel)) == e.files[rel.rsplit("/", 1)[1]], (
                    f"{tag}: {rel} half-swept: an entry is swept whole or not at all")

    # every vault object listed at the start has a record (unprocessed: run_aborted)
    if listed_vault:
        recorded = {a["key"] for a in log["actions"] if a["store"] == "vault"}
        unrecorded = sorted(set(all_vault_objects(entries)) - recorded)
        assert unrecorded == [], f"W18: listed vault objects with no record: {unrecorded}"
    return whole


@pytest.mark.parametrize("step", sorted(STEPS))
def test_storage_error_after_an_entry_was_deleted_still_writes_the_log(tmp_path, step):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    when = _after_first_entry_deleted(entries)
    for op, bucket in STEPS[step]:
        fake.outage(op, bucket, when=when)
    log, _raised = sweep_logged(vault, archive, ledger(entries), tmp_path / "r1")
    assert fake.forbidden == [], fake.forbidden
    whole = _check_log_after_storage_error(fake, log, entries)
    assert whole >= 1, "test setup: the first due entry was not deleted before the outage"
    assert whole < len(DUE), "the outage fired but every due entry was still swept"
    fault_rec = [a for a in log["actions"] if a["outcome"] == "failed"
                 and a["reason"] == "storage_error"]
    due_keys = {r for t in DUE for r in entries[t].rels()}
    assert any(a["store"] == "vault" and a["key"] in due_keys for a in fault_rec), (
        f"the storage_error record names no key of the entry being swept: {fault_rec}")


def test_retention_outage_keeps_every_later_entry_as_lock_unknown(tmp_path):
    """RV6's reproduction: 503 SlowDown on GetObjectRetention and HEAD from the
    second due entry on. Unreadable lock state is a keep (``lock_unknown``),
    so the run ends with a complete log and deletes nothing more."""
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    when = _after_first_entry_deleted(entries)
    fake.outage("get_object_retention", VAULT_BUCKET, when=when)
    fake.outage("head_object", VAULT_BUCKET, when=when)
    log, _ = sweep_logged(vault, archive, ledger(entries), tmp_path)
    assert fake.injected, "test setup: no retention read after the first entry"
    first = fake.injected[0]
    late = [op for i, (op, _) in enumerate(fake.calls) if i > first and op in DELETE_OPS]
    assert late == [], f"W18: delete(s) after the lock state became unreadable: {late}"
    assert log_problems(log) == []
    for tag in DUE:
        e = entries[tag]
        if _gone(fake, e):
            continue
        reasons = {a["reason"] for r in e.rels() for a in log["actions"]
                   if a["store"] == "vault" and a["key"] == r and a["outcome"] in ("kept", "failed")}
        assert reasons & {"lock_unknown", "storage_error", "run_aborted"}, (tag, reasons)
        assert reasons <= {"lock_unknown", "storage_error", "run_aborted", "entry_incomplete"}, (
            tag, reasons)


def test_listing_error_at_the_start_writes_a_log_and_changes_nothing(tmp_path):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    fake.outage("list_objects_v2", VAULT_BUCKET)
    log, _ = sweep_logged(vault, archive, ledger(entries), tmp_path)
    assert log_problems(log) == []
    assert fake.writes() == [] and fake.deletes() == []
    recs = [a for a in log["actions"] if a["op"] == "list"]
    assert recs and recs[0]["store"] == "vault" and recs[0]["outcome"] == "failed" \
        and recs[0]["reason"] == "storage_error", log["actions"]
    assert log["counts"]["failed"] >= 1


def test_archive_listing_error_writes_a_log_of_everything_done(tmp_path):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    fake.outage("list_objects_v2", ARCHIVE_BUCKET)
    log, _ = sweep_logged(vault, archive, ledger(entries), tmp_path)
    vault_listed = any(op == "list_objects_v2" and kw.get("Bucket") == VAULT_BUCKET
                       for op, kw in fake.calls[:fake.injected[0]])
    _check_log_after_storage_error(fake, log, entries, listed_vault=vault_listed)
    recs = [a for a in log["actions"] if a["op"] == "list"]
    assert recs and recs[0]["store"] == "archive" and recs[0]["reason"] == "storage_error", (
        log["actions"][-3:])
    assert [k for b, k, _ in fake.deleted_ok if b == ARCHIVE_BUCKET] == []


@pytest.mark.parametrize("step", sorted(STEPS))
def test_rerun_after_a_storage_error_resumes_idempotently(tmp_path, step):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    vault, archive = s3_stores(fake)
    for op, bucket in STEPS[step]:
        fake.outage(op, bucket, when=_after_first_entry_deleted(entries))
    sweep_logged(vault, archive, ledger(entries), tmp_path / "r1")
    fake.clear_faults()
    log2 = run_sweep(vault, archive, ledger(entries), tmp_path / "r2")
    assert fake.forbidden == [] and fake.violations == [], (fake.forbidden, fake.violations)
    assert log_problems(log2) == []
    assert log2["counts"]["failed"] == 0, [a for a in log2["actions"] if a["outcome"] == "failed"]
    for tag in DUE:
        e = entries[tag]
        for rel in e.rels():
            assert fake.versions(VAULT_BUCKET, vkey(rel)) == [], (tag, rel)
            vers = fake.versions(ARCHIVE_BUCKET, akey(rel))
            assert len(vers) == 1 and vers[0]["data"] == e.files[rel.rsplit("/", 1)[1]], (
                f"{tag}: {rel} archived {len(vers)} times (resume never archives twice)")


def _cli_args(tmp_path, entries):
    tmp_path.mkdir(parents=True, exist_ok=True)
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(ledger(entries)), encoding="utf-8")
    plan = write_plan(tmp_path / "plan.md", "decided")
    log_dir = tmp_path / "logs"
    return log_dir, ["--s3", "--archive-bucket", ARCHIVE_BUCKET, "--archive-prefix",
                     ARCHIVE_PREFIX, "--ledger", str(led), "--config", str(CHAIN_TOML),
                     "--log-dir", str(log_dir), "--live", "--plan", str(plan), "--now", iso(NOW)]


def test_cli_exits_1_and_names_the_log_after_a_storage_error(tmp_path, capsys):
    entries = scenario_entries()
    fake = seeded_fake(entries)
    fake.outage("delete_object", VAULT_BUCKET, when=_after_first_entry_deleted(entries))
    fake.outage("delete_objects", VAULT_BUCKET, when=_after_first_entry_deleted(entries))
    log_dir, args = _cli_args(tmp_path, entries)
    rc = vs().main(args, client_factory=lambda storage: fake)
    out = capsys.readouterr().out
    assert rc == 1, (rc, out[-2000:])
    files = sorted(log_dir.glob("vault-sweep-*.json")) if log_dir.is_dir() else []
    assert len(files) == 1, f"W18: the CLI's storage-error exit wrote {len(files)} log file(s)"
    try:
        obj = strict_loads(out)
    except ValueError as e:
        raise AssertionError(f"stdout is not one strict JSON object ({e}): {out[-2000:]!r}") from None
    assert isinstance(obj, dict) and obj.get("error") == "storage_failed", obj
    assert obj.get("log") == str(files[0]), (obj, str(files[0]))
    log = strict_loads(files[0].read_text(encoding="utf-8"))
    _check_log_after_storage_error(fake, log, entries)

    fake.clear_faults()
    log_dir2, args2 = _cli_args(tmp_path / "r2", entries)
    rc2 = vs().main(args2, client_factory=lambda storage: fake)
    out2 = capsys.readouterr().out
    assert rc2 == 0, out2[-2000:]
    for tag in DUE:
        for rel in entries[tag].rels():
            assert fake.versions(VAULT_BUCKET, vkey(rel)) == []
