"""W3a (RV4): evidence is re-verified against the live bucket; relabelling is refused.

Plan §17.1 L4: "evidence JSON present, and the verifier passes against the
live resource". RV4 showed a ``--dry-run --fixture`` run relabelled as live
(source, git_dirty, request ids, passed) passed the offline verifier. The
fix: ``python -P -m cascade.laned.storage_probe verify --evidence FILE``
(owner, ``HIPPIUS_*`` set, before the locks expire) re-reads each
COMPLIANCE-locked object from the bucket (HEAD, GetObjectRetention and GET
of the recorded version), compares version id, mode, retain-until, ETag,
size and sha256 with the evidence, lists the run root to confirm cleanup,
and only then writes a ``verified_live`` record into the evidence, bound to
the evidence body by ``body_sha256`` (helpers.canonical_body). The offline
L4 check requires that record, so:

* relabelled fixture evidence has no record, and ``verify_live`` refuses it
  against any bucket that does not hold its locked objects;
* editing the evidence after verification breaks ``body_sha256``.

Pinned surface: ``storage_probe.verify_live(evidence, client, *, now=None)``
returns the record or raises ``storage_probe.VerifyRefused`` (``.code``);
``client`` is boto3-shaped (:class:`helpers.FakeS3` here). Contract:
README.md.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

import pytest

from .helpers import (
    CREDENTIAL_ENV,
    TEST_BUCKET,
    FakeS3,
    body_sha256,
    evidence_problems,
    fixture_evidence,
    live_pair,
    probe,
    probe_module,
    relabel_as_live,
    run_probe,
    verified_live_problems,
    write_json,
)

CANARY = {"HIPPIUS_S3_ACCESS_KEY": "W3ACANARYACCESSKEY0001",
          "HIPPIUS_S3_SECRET_KEY": "w3a-canary-secret-value-0002"}


def _full(ev: dict) -> list[str]:
    return evidence_problems(ev, check_git=False, live_sizes=False)


def _relabelled(tmp_path, name: str = "ev.json") -> dict:
    return relabel_as_live(fixture_evidence(tmp_path, name=name))


def _refused(sp, ev: dict, fake: FakeS3, code: str, **kw) -> None:
    refused = sp.VerifyRefused
    with pytest.raises(refused) as ei:
        sp.verify_live(ev, fake, **kw)
    assert getattr(ei.value, "code", None) == code, (getattr(ei.value, "code", None), str(ei.value))
    assert fake.forbidden == [], f"verify must be read-only, it called {fake.forbidden}"


def _verified(sp, tmp_path) -> tuple[dict, FakeS3]:
    ev, fake = live_pair(_relabelled(tmp_path))
    rec = sp.verify_live(ev, fake)
    ev["verified_live"] = rec
    return ev, fake


# ── a bucket that really holds the locked objects ────────────────────────────


def test_verify_against_a_bucket_holding_the_locked_objects_is_accepted(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    verify = sp.verify_live
    ev, fake = live_pair(_relabelled(tmp_path))
    assert _full(ev) != [], "without verified_live the evidence is refused"
    rec = verify(ev, fake)
    assert isinstance(rec, dict) and rec.get("verified") is True, rec
    assert rec.get("body_sha256") == body_sha256(ev), "the record is bound to the evidence body"
    ev["verified_live"] = rec
    assert _full(ev) == [], _full(ev)
    assert json.loads(json.dumps(ev)) == ev, "the record is plain JSON"
    assert fake.forbidden == [], f"verify must be read-only, it called {fake.forbidden}"


def test_verify_reads_each_locked_version_and_lists_the_run_root(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    verify = sp.verify_live
    ev, fake = live_pair(_relabelled(tmp_path))
    verify(ev, fake)
    for lo in ev["locked_objects"]:
        want = {"Bucket": ev["bucket"], "Key": lo["key"], "VersionId": lo["version_id"]}
        for op in ("head_object", "get_object_retention", "get_object"):
            assert any(o == op and all(kw.get(k) == v for k, v in want.items())
                       for o, kw in fake.calls), f"verify did not {op} {want}: {fake.calls}"
    root = f"{ev['prefix']}{ev['run_id']}/"
    assert any(o in ("list_object_versions", "list_objects_v2", "get_paginator")
               for o, _ in fake.calls), fake.calls
    assert any(kw.get("Prefix") == root for o, kw in fake.calls
               if o in ("list_object_versions", "list_objects_v2")), (
        f"verify lists the run root {root}: {fake.calls}")


def test_verify_record_shape(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, _ = _verified(sp, tmp_path)
    rec = ev["verified_live"]
    assert rec.get("schema") == "laned_w3a_verify/1", rec.get("schema")
    assert rec.get("bucket") == ev["bucket"] and rec.get("run_id") == ev["run_id"]
    at = datetime.fromisoformat(rec["verified_at"])
    assert at.tzinfo is not None
    assert {o["key"] for o in rec["objects"]} == {lo["key"] for lo in ev["locked_objects"]}
    for o in rec["objects"]:
        assert o["request_ids"] and all(str(r).startswith("live-fake-") for r in o["request_ids"]), o
    assert set(rec["listing"]["keys"]) == {lo["key"] for lo in ev["locked_objects"]}
    assert verified_live_problems(ev, check_git=False) == []


# ── relabelled fixture evidence is refused ───────────────────────────────────


def test_relabelled_fixture_evidence_is_refused_on_an_empty_bucket(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _relabelled(tmp_path)
    _refused(sp, ev, FakeS3(ev["bucket"]), "locked_object_missing")


def test_relabelled_fixture_evidence_is_refused_on_another_runs_bucket(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _relabelled(tmp_path, name="a.json")
    _, other_bucket = live_pair(_relabelled(tmp_path, name="b.json"))
    _refused(sp, ev, other_bucket, "locked_object_missing")


def test_fixture_source_is_refused_before_any_bucket_read(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    ev["source"] = "fixture"
    _refused(sp, ev, fake, "not_live")
    assert fake.calls == [], f"a fixture source is refused before touching the bucket: {fake.calls}"


def test_evidence_without_locked_objects_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    ev["locked_objects"] = []
    _refused(sp, ev, fake, "no_locked_objects")


def _edit_evidence_etag(ev, fake):
    ev["locked_objects"][0]["etag"] = '"' + "0" * 32 + '"'


def _bucket_other_bytes(ev, fake):
    lo = ev["locked_objects"][0]
    v = fake.objects[lo["key"]][-1]
    v["data"] = bytes(len(v["data"]))


def _bucket_other_retention(ev, fake):
    lo = ev["locked_objects"][0]
    fake.objects[lo["key"]][-1]["retain_until"] += timedelta(hours=3)


def _bucket_governance(ev, fake):
    fake.objects[ev["locked_objects"][0]["key"]][-1]["mode"] = "GOVERNANCE"


def _bucket_unlocked(ev, fake):
    fake.objects[ev["locked_objects"][0]["key"]][-1]["mode"] = None


def _edit_evidence_size(ev, fake):
    ev["locked_objects"][0]["size"] += 1


@pytest.mark.parametrize("change", [_edit_evidence_etag, _bucket_other_bytes,
                                    _bucket_other_retention, _bucket_governance,
                                    _bucket_unlocked, _edit_evidence_size],
                         ids=lambda f: f.__name__.lstrip("_"))
def test_a_locked_object_that_differs_from_the_evidence_is_refused(tmp_path, monkeypatch, change):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    change(ev, fake)
    sp_code = "locked_object_mismatch"
    if change is _bucket_unlocked:
        # GetObjectRetention answers 404 for an unlocked version
        refused = sp.VerifyRefused
        with pytest.raises(refused) as ei:
            sp.verify_live(ev, fake)
        assert getattr(ei.value, "code", None) in ("locked_object_mismatch",
                                                   "locked_object_missing"), str(ei.value)
        assert fake.forbidden == []
        return
    _refused(sp, ev, fake, sp_code)


def test_a_wrong_version_id_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    ev["locked_objects"][0]["version_id"] = "not-the-locked-version"
    _refused(sp, ev, fake, "locked_object_missing")


def test_verify_after_the_lock_expired_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    last = max(datetime.fromisoformat(lo["retain_until"]) for lo in ev["locked_objects"])
    _refused(sp, ev, fake, "lock_expired", now=last + timedelta(seconds=1))


def test_an_unlocked_leftover_under_the_run_root_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    key = probe(ev, "single_put_limit")["objects"][0]["key"]
    fake.put_version(key, "v-left", b"left behind")
    _refused(sp, ev, fake, "run_root_not_clean")


def test_verify_bucket_is_the_evidence_bucket(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(_relabelled(tmp_path))
    fake.bucket = "some-other-bucket"
    assert ev["bucket"] == TEST_BUCKET
    _refused(sp, ev, fake, "locked_object_missing")


# ── the record is hash-bound: no edit after verification ─────────────────────


def test_editing_the_evidence_after_verification_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, _ = _verified(sp, tmp_path)
    assert _full(ev) == []
    for edit in (lambda e: probe(e, "presigned_upload_part").update(reason="edited"),
                 lambda e: e.update(recommended_path="upload_relay"),
                 lambda e: e["bucket_lock"].update(default_retention=None, request_id="other"),
                 lambda e: e["cleanup"].update(deleted_versions=e["cleanup"]["deleted_versions"] + 1)):
        e2 = copy.deepcopy(ev)
        edit(e2)
        probs = verified_live_problems(e2, check_git=False)
        assert any("body_sha256" in p for p in probs), probs


def test_editing_the_record_is_refused(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, _ = _verified(sp, tmp_path)
    cases = {
        "etag": lambda r: r["objects"][0].update(etag='"' + "1" * 32 + '"'),
        "sha256": lambda r: r["objects"][0].update(sha256="1" * 64),
        "retain_until": lambda r: r["objects"][0].update(retain_until="2099-01-01T00:00:00+00:00"),
        "request_ids": lambda r: r["objects"][0].update(request_ids=["fixture-3"]),
        "expired": lambda r: r.update(verified_at="2099-01-01T00:00:00+00:00"),
        "objects": lambda r: r.update(objects=r["objects"][:1]),
        "listing": lambda r: r["listing"]["keys"].append(f"{ev['prefix']}{ev['run_id']}/x.bin"),
        "verified": lambda r: r.update(verified=False),
        "body_sha256": lambda r: r.update(body_sha256="0" * 64),
        "schema": lambda r: r.update(schema="laned_w3a_verify/0"),
    }
    for name, edit in cases.items():
        e2 = copy.deepcopy(ev)
        edit(e2["verified_live"])
        probs = verified_live_problems(e2, check_git=False)
        assert probs, f"an edited record ({name}) must be refused"
    e2 = copy.deepcopy(ev)
    del e2["verified_live"]
    assert any(p.startswith("verified_live") for p in _full(e2))


# ── the `verify` CLI refuses before any network use ──────────────────────────


def _cli(tmp_path, *args, env_extra=None):
    return run_probe(tmp_path, "verify", *args, env_extra=env_extra)


def test_verify_cli_needs_the_main_credentials(tmp_path):
    f = write_json(tmp_path / "ev.json", _relabelled(tmp_path))
    before = f.read_bytes()
    proc = _cli(tmp_path, "--evidence", str(f))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])
    for name in CREDENTIAL_ENV[:2]:
        assert name in proc.stderr, f"the refusal names {name}: {proc.stderr[-2000:]}"
    assert f.read_bytes() == before, "a refused verify leaves the evidence untouched"


def test_verify_cli_refuses_fixture_evidence_before_any_network_use(tmp_path):
    f = write_json(tmp_path / "ev.json", fixture_evidence(tmp_path))
    before = f.read_bytes()
    proc = _cli(tmp_path, "--evidence", str(f), env_extra=CANARY)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])
    assert "not_live" in proc.stderr, proc.stderr[-2000:]
    assert f.read_bytes() == before
    for v in CANARY.values():
        assert v not in proc.stdout + proc.stderr


def test_verify_cli_refuses_a_missing_or_non_json_evidence_file(tmp_path):
    proc = _cli(tmp_path, "--evidence", str(tmp_path / "absent.json"), env_extra=CANARY)
    assert proc.returncode == 2 and "absent.json" in proc.stderr, proc.stderr[-2000:]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    proc = _cli(tmp_path, "--evidence", str(bad), env_extra=CANARY)
    assert proc.returncode == 2 and "evidence" in proc.stderr.lower(), proc.stderr[-2000:]
    assert bad.read_text() == "{not json"
