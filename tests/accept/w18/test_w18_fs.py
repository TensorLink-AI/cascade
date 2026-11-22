"""W18: the sweeper over the filesystem adapter (``FsStore``), the CI stand-in
for the vault and the owner-only archive. Contract: README.md."""

from __future__ import annotations

import os
import stat

import pytest

from .helpers import (
    DUE,
    EXPECTED_KEEP,
    NOW,
    ORPHAN_KEY,
    STRAY_KEY,
    Entry,
    days,
    fs_seed,
    keep_reason,
    ledger,
    log_problems,
    ops_of,
    run_sweep,
    scenario_entries,
    tree,
    vs,
)


@pytest.fixture
def umask022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def _stores(tmp_path, entries):
    m = vs()
    vroot, aroot = tmp_path / "vault", tmp_path / "archive"
    fs_seed(vroot, entries)
    return vroot, aroot, m.FsStore(vroot), m.FsStore(aroot)


def test_fs_due_losers_archived_and_removed_rest_kept(tmp_path, umask022):
    entries = scenario_entries()
    vroot, aroot, vault, archive = _stores(tmp_path, entries)
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    assert log_problems(log) == []
    assert log["vault"] == {"kind": "fs", "root": str(vroot)}
    assert log["archive"] == {"kind": "fs", "root": str(aroot)}
    for tag in DUE:
        e = entries[tag]
        for name, data in e.files.items():
            rel = e.key(name)
            assert (aroot / rel).read_bytes() == data, f"{tag}: archive copy of {rel}"
            assert not (vroot / rel).exists(), f"{tag}: {rel} still in the vault"
    for tag, reason in EXPECTED_KEEP.items():
        for rel in entries[tag].rels():
            assert (vroot / rel).is_file() and not (aroot / rel).exists()
            assert keep_reason(log, rel) == reason, (tag, ops_of(log, rel))
    for rel in (ORPHAN_KEY, STRAY_KEY):
        assert (vroot / rel).is_file() and keep_reason(log, rel) == "unreferenced"


def test_fs_archive_is_owner_only(tmp_path, umask022):
    entries = scenario_entries()
    vroot, aroot, vault, archive = _stores(tmp_path, entries)
    run_sweep(vault, archive, ledger(entries), tmp_path)
    assert aroot.is_dir()
    bad = []
    for p in [aroot, *aroot.rglob("*")]:
        mode = stat.S_IMODE(p.lstat().st_mode)
        want = 0o700 if p.is_dir() else 0o600
        if mode != want:
            bad.append(f"{p.relative_to(tmp_path)}: {oct(mode)} (want {oct(want)})")
    assert bad == [], "the owner-only archive must be 0700 dirs / 0600 files even under " \
                      "umask 022:\n  " + "\n  ".join(bad)


def test_fs_dry_run_changes_nothing(tmp_path, umask022):
    entries = scenario_entries()
    vroot, aroot, vault, archive = _stores(tmp_path, entries)
    before_v, before_a = tree(vroot), tree(aroot)
    log = run_sweep(vault, archive, ledger(entries), tmp_path, live=False, plan=None)
    assert log["mode"] == "dry-run"
    assert tree(vroot) == before_v, "a dry-run changed the vault"
    assert tree(aroot) == before_a, "a dry-run created or changed the archive"
    for tag in DUE:
        for rel in entries[tag].rels():
            assert ops_of(log, rel) == [("archive", "planned"), ("delete", "planned")]


def test_fs_failed_archive_copy_means_no_delete(tmp_path, umask022):
    entries = scenario_entries()
    vroot, aroot, vault, archive = _stores(tmp_path, entries)
    e = entries["lost_due"]
    aroot.mkdir(mode=0o700)
    (aroot / e.hotkey).write_bytes(b"a file where the entry's archive directory goes")
    log = run_sweep(vault, archive, ledger(entries), tmp_path)
    for rel in e.rels():
        assert (vroot / rel).read_bytes() == e.files[rel.rsplit("/", 1)[1]], (
            f"{rel} left the vault although its archive copy could not be written")
        assert any(a["op"] == "archive" and a["outcome"] == "failed"
                   and a["reason"] == "archive_copy_failed"
                   for a in log["actions"] if a["key"] == rel), ops_of(log, rel)
    for rel in entries["unjudged_due"].rels():
        assert not (vroot / rel).exists(), "one failed entry stalls only itself"


def test_fs_locked_object_never_deleted_then_swept(tmp_path, umask022):
    m = vs()
    e = Entry("fs_locked", "lost", 1)
    vroot, aroot = tmp_path / "vault", tmp_path / "archive"
    vault, archive = m.FsStore(vroot), m.FsStore(aroot)
    locked = e.key("model.safetensors")
    vault.put(locked, e.files["model.safetensors"], lock_mode="COMPLIANCE",
              retain_until=NOW + days(1))
    vault.put(e.key("manifest.json"), e.files["manifest.json"])
    led = ledger({e.tag: e}, disputes=())
    log = run_sweep(vault, archive, led, tmp_path / "r1")
    assert keep_reason(log, locked) == "object_lock", ops_of(log, locked)
    assert (vroot / locked).read_bytes() == e.files["model.safetensors"]
    assert (vroot / e.key("manifest.json")).is_file()
    with pytest.raises(m.ObjectLocked):
        vault.delete(locked, now=NOW)
    later = NOW + days(2)
    log2 = run_sweep(vault, archive, led, tmp_path / "r2", now=later)
    for rel in e.rels():
        assert ("delete", "done") in ops_of(log2, rel), ops_of(log2, rel)
        assert not (vroot / rel).exists()
        assert (aroot / rel).read_bytes() == e.files[rel.rsplit("/", 1)[1]]


def test_fs_archive_overlapping_the_vault_is_refused(tmp_path):
    m = vs()
    entries = scenario_entries()
    vroot = tmp_path / "vault"
    fs_seed(vroot, entries)
    before = tree(vroot)
    for aroot in (vroot, vroot / "archive", tmp_path):
        with pytest.raises(m.SweepRefused) as ei:
            run_sweep(m.FsStore(vroot), m.FsStore(aroot), ledger(entries), tmp_path / "x")
        assert ei.value.code == "archive_overlaps_vault"
    assert tree(vroot) == before
