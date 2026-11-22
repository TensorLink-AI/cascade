"""W18 (RV6 B2): lock detection fails closed. Only an explicit "no retention
configured" answer (``NoSuchObjectLockConfiguration``) or a retention whose
``RetainUntilDate`` has passed counts as unlocked. Any error reading a
version's retention (``InvalidRequest``, ``AccessDenied``, 5xx) or a malformed
answer leaves the object kept as ``lock_unknown``: no archive copy and no
``delete_object`` for that key, even when the version is in truth unlocked.
A legal hold ``ON`` keeps it as ``legal_hold``. Contract: README.md
"What is kept, and why"."""

from __future__ import annotations

import pytest

from .helpers import (
    ARCHIVE_BUCKET,
    NOW,
    VAULT_BUCKET,
    WRITE_OPS,
    Entry,
    FakeS3,
    akey,
    days,
    keep_reason,
    ledger,
    log_problems,
    ops_of,
    run_sweep,
    s3_stores,
    sweep_logged,
    vkey,
)

ANSWERS = {
    "InvalidRequest_400": ("error", "InvalidRequest", 400),
    "AccessDenied_403": ("error", "AccessDenied", 403),
    "InternalError_500": ("error", "InternalError", 500),
    "SlowDown_503": ("error", "SlowDown", 503),
    "malformed_no_mode": ("response", {"Retention": {},
                                       "ResponseMetadata": {"HTTPStatusCode": 200}}),
    "malformed_no_date": ("response", {"Retention": {"Mode": "COMPLIANCE"},
                                       "ResponseMetadata": {"HTTPStatusCode": 200}}),
    "malformed_bad_date": ("response", {"Retention": {"Mode": "GOVERNANCE",
                                                      "RetainUntilDate": "not-a-date"},
                                        "ResponseMetadata": {"HTTPStatusCode": 200}}),
    "malformed_empty": ("response", {"ResponseMetadata": {"HTTPStatusCode": 200}}),
}
TRUTH = ("unlocked", "COMPLIANCE", "GOVERNANCE")


def _fixture(truth: str):
    e = Entry("lock_probe", "lost", 1)
    other = Entry("lock_probe_other", "lost", 1)
    fake = FakeS3()
    rel = e.key("model.safetensors")
    if truth == "unlocked":
        fake.seed(VAULT_BUCKET, vkey(rel), e.files["model.safetensors"])
    else:
        fake.seed(VAULT_BUCKET, vkey(rel), e.files["model.safetensors"], mode=truth,
                  retain_until=NOW + days(5))
    fake.seed(VAULT_BUCKET, vkey(e.key("manifest.json")), e.files["manifest.json"])
    for name, data in other.files.items():
        fake.seed(VAULT_BUCKET, vkey(other.key(name)), data)
    return e, other, fake, rel


def _archive_writes_for(fake, rels):
    keys = {akey(r) for r in rels}
    return [(op, kw.get("Key")) for op, kw in fake.calls
            if op in WRITE_OPS and kw.get("Key") in keys]


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("answer", sorted(ANSWERS))
def test_unreadable_retention_is_never_treated_as_no_lock(tmp_path, answer, truth):
    e, other, fake, rel = _fixture(truth)
    fake.retention_override[(VAULT_BUCKET, vkey(rel))] = ANSWERS[answer]
    vault, archive = s3_stores(fake)
    entries = {e.tag: e, other.tag: other}
    log, raised = sweep_logged(vault, archive, ledger(entries, disputes=()), tmp_path)
    assert fake.forbidden == [], fake.forbidden
    assert fake.violations == [], "\n".join(fake.violations)
    assert vkey(rel) not in fake.deleted_keys(VAULT_BUCKET), (
        f"W18: delete_object sent for {rel} whose retention answer was {answer}: an unreadable "
        "or malformed lock state is LOCKED/unknown (fail closed), never 'no lock'")
    assert raised is None, f"a per-object unreadable lock is a keep, not a run error: {raised!r}"
    assert log_problems(log) == []
    allowed = {"lock_unknown"} if truth == "unlocked" else {"lock_unknown", "object_lock"}
    assert keep_reason(log, rel) in allowed, (answer, truth, ops_of(log, rel))
    sib = e.key("manifest.json")
    assert keep_reason(log, sib) == "entry_incomplete", ops_of(log, sib)
    assert vkey(sib) not in fake.deleted_keys(VAULT_BUCKET), "an entry is swept whole"
    assert _archive_writes_for(fake, e.rels()) == [], (
        "a kept object gets no archive copy (never archived-then-deleted)")
    for r in e.rels():
        assert fake.live_data(VAULT_BUCKET, vkey(r)) is not None
    for r in other.rels():
        assert ("delete", "done") in ops_of(log, r), (
            "one unreadable lock stalls only its own entry", ops_of(log, r))


@pytest.mark.parametrize("mode", ["COMPLIANCE", "GOVERNANCE"])
def test_invalid_request_on_a_locked_version_sends_no_delete(tmp_path, mode):
    """RV6's reproduction: COMPLIANCE/GOVERNANCE-locked versions whose
    GetObjectRetention answered InvalidRequest were archived and then sent
    delete_object."""
    e = Entry("ir_locked", "lost", 1)
    fake = FakeS3()
    for name, data in e.files.items():
        fake.seed(VAULT_BUCKET, vkey(e.key(name)), data, mode=mode, retain_until=NOW + days(2))
        fake.retention_override[(VAULT_BUCKET, vkey(e.key(name)))] = (
            "error", "InvalidRequest", 400)
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path)
    sent = [kw.get("Key") for op, kw in fake.calls if op in ("delete_object", "delete_objects")]
    assert sent == [], f"W18: delete call(s) on {mode}-locked versions: {sent}"
    assert fake.violations == [], fake.violations
    for r in e.rels():
        assert keep_reason(log, r) in ("lock_unknown", "object_lock"), ops_of(log, r)
    assert _archive_writes_for(fake, e.rels()) == []


def test_legal_hold_on_blocks_the_delete(tmp_path):
    e = Entry("held", "lost", 1)
    fake = FakeS3()
    rel = e.key("model.safetensors")
    fake.seed(VAULT_BUCKET, vkey(rel), e.files["model.safetensors"], hold="ON")
    fake.seed(VAULT_BUCKET, vkey(e.key("manifest.json")), e.files["manifest.json"])
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path)
    assert vkey(rel) not in fake.deleted_keys(VAULT_BUCKET), (
        "W18: delete sent for a version under a legal hold")
    assert fake.violations == [], fake.violations
    assert log_problems(log) == []
    assert keep_reason(log, rel) == "legal_hold", ops_of(log, rel)
    assert keep_reason(log, e.key("manifest.json")) == "entry_incomplete"
    assert _archive_writes_for(fake, e.rels()) == []


def test_legal_hold_off_does_not_block(tmp_path):
    e = Entry("hold_off", "lost", 1)
    fake = FakeS3()
    for name, data in e.files.items():
        fake.seed(VAULT_BUCKET, vkey(e.key(name)), data, hold="OFF")
    vault, archive = s3_stores(fake)
    log = run_sweep(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path)
    assert fake.violations == [], fake.violations
    for r in e.rels():
        assert ("delete", "done") in ops_of(log, r), ops_of(log, r)


@pytest.mark.parametrize("answer", ["InvalidRequest_400", "AccessDenied_403", "SlowDown_503",
                                    "malformed_no_mode"])
def test_archive_expiry_fails_closed_on_unreadable_retention(tmp_path, answer):
    e = Entry("arch_exp", "lost", 1)
    fake = FakeS3()
    for name, data in e.files.items():
        fake.seed(VAULT_BUCKET, vkey(e.key(name)), data)
    vault, archive = s3_stores(fake)
    run_sweep(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path / "r0")
    for r in e.rels():
        assert fake.live_data(ARCHIVE_BUCKET, akey(r)) is not None
        fake.retention_override[(ARCHIVE_BUCKET, akey(r))] = ANSWERS[answer]
    later = NOW + days(15)
    fake.clock = later
    log, raised = sweep_logged(vault, archive, ledger({e.tag: e}, disputes=()), tmp_path / "r15",
                               now=later)
    assert raised is None, raised
    assert fake.deleted_keys(ARCHIVE_BUCKET) == [], (
        f"W18: archive copies expired although their retention answer was {answer}")
    for r in e.rels():
        assert keep_reason(log, r, "archive") == "lock_unknown", ops_of(log, r, "archive")
        assert fake.live_data(ARCHIVE_BUCKET, akey(r)) is not None
