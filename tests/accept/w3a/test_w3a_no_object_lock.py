"""W3a: ``--no-object-lock``, the owner's explicit flag for a bucket without Object Lock.

Owner decision 2026-11-23: Hippius has NO Object Lock (a full-access
bucket-scoped sub-token and the main key both get AccessDenied 403 on
GetObjectLockConfiguration / GetBucketVersioning; the console offers no lock
permission). The probe is re-run without the lock tests and the upload path
is picked from that evidence. Kings are not COMPLIANCE-locked; immutability
comes from the content-addressed path + the ingest re-hash + the R2/HF
mirrors.

With ``--no-object-lock`` the probe:

* never calls GetObjectLockConfiguration and never sends a lock header or a
  retention (no ``ObjectLock*`` on any write, no PutObjectRetention);
* records ``bucket_lock: {"lock_enabled": false, "asserted_by": "owner_flag"}``;
* records ``object_lock_multipart`` and ``compliance_retention_unshortenable``
  as ``result: "skipped", reason: "no_object_lock"`` (never ``pass``), with no
  request, object or observation;
* still runs ``upload_part_copy_2_5gb``: the copy never used a lock (its
  destination is a plain multipart upload), so it runs unlocked as before;
* leaves one small UNLOCKED anchor object (``anchor_objects``) after cleanup,
  which ``storage_probe verify`` re-reads in place of the locked objects
  (record ``binding: "unlocked_anchor"``).

Without the flag nothing changes (the fail-closed ``bucket_lock_unreadable``
exit 2 stays). ``recommended_path`` is unchanged. Contract: README.md "No
Object Lock".
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path

import pytest

from .helpers import (
    ANCHOR_BINDING,
    BUCKET_LOCK_KEYS,
    LOCK_PROBES,
    MAX_ANCHOR_BYTES,
    NO_LOCK_FLAG,
    OWNER_FLAG_LOCK,
    PROBES,
    SKIP_REASON,
    TEST_PREFIX,
    FakeClientError,
    FakeS3,
    anchor_pair,
    base_args,
    body_sha256,
    dry_run,
    evidence_problems,
    fixture_evidence,
    leak_problems,
    model_file,
    probe,
    probe_module,
    relabel_as_live,
    verified_live_problems,
)

CANARY = {"HIPPIUS_S3_ACCESS_KEY": "W3ACANARYACCESSKEY0001",
          "HIPPIUS_S3_SECRET_KEY": "w3a-canary-secret-value-0002",
          "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY": "W3ACANARYSUBTOKENKEY03",
          "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY": "w3a-canary-subtoken-secret-0004"}
OTHER_PROBES = tuple(n for n in PROBES if n not in LOCK_PROBES)
LOCK_WORDS = re.compile(r"ObjectLock|Retention|COMPLIANCE", re.I)

# a Hippius-shaped bucket: no Object Lock at all
HIPPIUS = {"bucket_lock_enabled": False}


def _nolock(tmp_path: Path, change: dict | None = None, *, name: str = "nl.json",
            expect_rc: int = 0, env_extra: dict | None = None):
    """A ``--no-object-lock`` fixture run: (stdout JSON, evidence, process)."""
    model = model_file(tmp_path, (lambda m: m.update(change)) if change else None,
                       name=f"model-{name}")
    out = tmp_path / name
    printed, ev, proc = dry_run(tmp_path, NO_LOCK_FLAG, fixture=model, out=out,
                                expect_rc=expect_rc, env_extra=env_extra)
    assert ev is not None, "a --no-object-lock dry-run with --fixture writes its evidence"
    return printed, ev, proc


def _body(ev: dict) -> list[str]:
    return evidence_problems(ev, check_git=False, live_sizes=False, require_verified=False)


def _full(ev: dict) -> list[str]:
    return evidence_problems(ev, check_git=False, live_sizes=False)


def _live_nolock(tmp_path: Path, change: dict | None = None, name: str = "nl.json") -> dict:
    return relabel_as_live(_nolock(tmp_path, change, name=name)[1])


# ── the plan ─────────────────────────────────────────────────────────────────


def test_no_lock_plan_skips_the_lock_probes_and_plans_no_lock_call(tmp_path):
    printed, ev, _ = dry_run(tmp_path, NO_LOCK_FLAG, out=tmp_path / "o.json", expect_rc=0)
    assert ev is None and not (tmp_path / "o.json").exists()
    plan = printed["would_do"]
    assert isinstance(plan, dict) and set(plan) == set(PROBES), plan
    for n in LOCK_PROBES:
        assert isinstance(plan[n], list) and plan[n], (n, plan[n])
        text = " ".join(plan[n])
        assert "skipped" in text and SKIP_REASON in text, (n, plan[n])
    blob = json.dumps(plan)
    assert not LOCK_WORDS.search(blob), f"the no-lock plan still plans a lock call: {plan}"
    assert not leak_problems(printed)


# ── the evidence a --no-object-lock run writes ───────────────────────────────


@pytest.mark.parametrize("change", [None, HIPPIUS], ids=["lock-capable-model", "hippius-model"])
def test_no_lock_evidence_records_the_owner_flag_and_skips_the_lock_probes(tmp_path, change):
    printed, ev, _ = _nolock(tmp_path, change)
    assert ev.get("bucket_lock") == OWNER_FLAG_LOCK, ev.get("bucket_lock")
    for n in LOCK_PROBES:
        row = probe(ev, n)
        assert row["result"] == "skipped" and row["passed"] is False, (n, row)
        assert row["reason"] == SKIP_REASON, (n, row["reason"])
        assert row["requests"] == [] and row["objects"] == [] and row["observations"] == {}, row
    for n in OTHER_PROBES:
        assert probe(ev, n)["result"] == "pass", (n, probe(ev, n)["reason"])
    assert ev["locked_objects"] == []
    assert ev["cleanup"]["remaining_unlocked"] == [] and ev["cleanup"]["remaining_locked"] == []
    assert ev["recommended_path"] == "presigned_multipart"
    assert ev["source"] == "fixture" and ev["passed"] is False
    assert printed["complete"] is True, "skipped lock probes do not make the run incomplete"
    assert printed["results"] == {n: ("skipped" if n in LOCK_PROBES else "pass") for n in PROBES}
    assert printed["locked_objects"] == []


def test_no_lock_run_keeps_the_unlocked_upload_part_copy(tmp_path):
    """The copy probe never locked anything (plain multipart destination), so
    with no Object Lock it still runs and records pass/fail."""
    _, ev, _ = _nolock(tmp_path, HIPPIUS)
    row = probe(ev, "upload_part_copy_2_5gb")
    assert row["result"] == "pass", row["reason"]
    ops = [r["op"] for r in row["requests"]]
    assert any(op.startswith("UploadPartCopy") for op in ops), ops
    obs = row["observations"]
    assert obs["dest_sha256"] == obs["source_sha256"] and obs["dest_size"] == obs["source_size"]
    _, ev, _ = _nolock(tmp_path, {**HIPPIUS, "upload_part_copy": "rejected"}, name="r.json")
    assert probe(ev, "upload_part_copy_2_5gb")["result"] == "fail"


def test_no_lock_run_sends_no_lock_request_at_all(tmp_path):
    _, ev, _ = _nolock(tmp_path)
    ops = [r["op"] for p in ev["probes"] for r in p["requests"]]
    assert ops, "the other probes still record their requests"
    bad = [op for op in ops if LOCK_WORDS.search(op)]
    assert not bad, f"a --no-object-lock run made lock requests: {bad}"


def test_no_lock_run_leaves_one_small_unlocked_anchor(tmp_path):
    _, ev, _ = _nolock(tmp_path, HIPPIUS)
    anchors = ev.get("anchor_objects")
    assert isinstance(anchors, list) and len(anchors) == 1, anchors
    a = anchors[0]
    root = f"{TEST_PREFIX}{ev['run_id']}/"
    probe_keys = {o["key"] for p in ev["probes"] for o in p["objects"]}
    assert a["key"].startswith(root) and a["key"] not in probe_keys, a
    assert 0 < a["size"] <= MAX_ANCHOR_BYTES, a
    assert re.fullmatch(r"[0-9a-f]{64}", str(a.get("sha256"))), a
    assert isinstance(a.get("etag"), str) and a["etag"].strip('"'), a
    assert "version_id" in a, a
    assert a["key"] not in ev["cleanup"]["remaining_unlocked"]


@pytest.mark.parametrize("change,path", [
    ({"presigned_upload_part": "rejected"}, "single_presigned_put"),
    ({"presigned_upload_part": "rejected", "single_put_max_bytes": 600000}, "upload_relay"),
    ({"lock_on_multipart": "rejected", "compliance": "shortenable"}, "presigned_multipart"),
], ids=["presigned-rejected", "no-presigned-path", "lock-model-irrelevant"])
def test_no_lock_recommended_path_is_unchanged(tmp_path, change, path):
    _, ev, _ = _nolock(tmp_path, {**HIPPIUS, **change})
    assert ev["recommended_path"] == path
    for n in LOCK_PROBES:
        assert probe(ev, n)["result"] == "skipped"
    live = relabel_as_live(ev)
    assert live["passed"] is True
    assert _body(live) == [], _body(live)


def test_no_lock_run_leaks_no_credential_url_or_signature(tmp_path):
    printed, ev, proc = _nolock(tmp_path, HIPPIUS, env_extra=CANARY)
    blob = proc.stdout + proc.stderr + json.dumps(ev)
    for name, value in CANARY.items():
        assert value not in blob, f"the value of {name} leaked"
    assert leak_problems(ev) == [] and leak_problems(printed) == []
    assert not re.search(r"x-amz-signature|x-amz-credential|https?://", blob, re.I)


# ── in-process: no lock call, no lock header, no retention ───────────────────


def _recording_backend(base):
    calls: list[tuple[str, object]] = []

    def wrap(name):
        orig = getattr(base, name)

        def f(self, *a, **k):
            try:
                bound = inspect.signature(orig).bind(self, *a, **k)
                lock = bound.arguments.get("lock")
            except TypeError:
                lock = None
            calls.append((name, lock))
            return orig(self, *a, **k)
        return f

    names = ("get_object_lock_configuration", "bucket_lock_enabled", "put_object_retention",
             "create_multipart", "put_object", "presigned_put", "presigned_upload_part",
             "upload_part", "upload_part_copy", "copy_object", "delete_object",
             "complete_multipart")
    cls = type("RecordingFixtureBackend", (base,),
               {n: wrap(n) for n in names if hasattr(base, n)})
    return cls, calls


def _main(sp, monkeypatch, tmp_path: Path, model: Path, *extra: str):
    cls, calls = _recording_backend(sp.FixtureBackend)
    monkeypatch.setattr(sp, "FixtureBackend", cls)
    out = tmp_path / "e.json"
    rc = sp.main([*base_args(), "--dry-run", "--fixture", str(model), "--out", str(out), *extra])
    return rc, calls, out


@pytest.mark.parametrize("change", [None, HIPPIUS], ids=["lock-capable-model", "hippius-model"])
def test_no_lock_makes_no_lock_call_and_sends_no_lock_header(tmp_path, monkeypatch, capsys, change):
    sp = probe_module(monkeypatch)
    preflight_calls = []
    monkeypatch.setattr(sp, "bucket_lock_preflight",
                        lambda *a, **k: preflight_calls.append(a) or OWNER_FLAG_LOCK)
    model = model_file(tmp_path, (lambda m: m.update(change)) if change else None)
    rc, calls, out = _main(sp, monkeypatch, tmp_path, model, NO_LOCK_FLAG)
    capsys.readouterr()
    assert rc == 0 and out.is_file(), f"a --no-object-lock fixture run exits 0, got {rc}"
    assert preflight_calls == [], "with --no-object-lock the lock preflight is never called"
    names = [n for n, _ in calls]
    for forbidden in ("get_object_lock_configuration", "bucket_lock_enabled",
                      "put_object_retention"):
        assert forbidden not in names, f"--no-object-lock called {forbidden}: {names}"
    locked = [(n, lk) for n, lk in calls if lk]
    assert locked == [], f"lock headers were sent: {locked}"
    assert "put_object" in names and "upload_part_copy" in names, names


def _hippius_lock_read(sp, monkeypatch):
    """GetObjectLockConfiguration answers AccessDenied 403, as Hippius does."""
    def denied(self, Bucket):  # noqa: N803 — boto3 keyword shape
        raise sp.FixtureClientError("AccessDenied", {"HTTPStatusCode": 403,
                                                     "RequestId": "fixture-denied"})
    monkeypatch.setattr(sp.FixtureBackend, "get_object_lock_configuration", denied)


def test_hippius_403_lock_read_still_fails_closed_without_the_flag(tmp_path, monkeypatch, capsys):
    """Byte-for-byte as before: no flag, a 403 lock read is exit 2
    `bucket_lock_unreadable`, nothing written, no evidence."""
    sp = probe_module(monkeypatch)
    _hippius_lock_read(sp, monkeypatch)
    rc, calls, out = _main(sp, monkeypatch, tmp_path, model_file(tmp_path, lambda m: m.update(HIPPIUS)))
    err = capsys.readouterr().err
    assert rc == 2 and "bucket_lock_unreadable" in err, (rc, err[-2000:])
    writes = [n for n, _ in calls if n != "get_object_lock_configuration"]
    assert writes == [] and not out.exists(), writes


def test_hippius_403_lock_read_is_never_made_with_the_flag(tmp_path, monkeypatch, capsys):
    sp = probe_module(monkeypatch)
    _hippius_lock_read(sp, monkeypatch)
    rc, calls, out = _main(sp, monkeypatch, tmp_path,
                           model_file(tmp_path, lambda m: m.update(HIPPIUS)), NO_LOCK_FLAG)
    capsys.readouterr()
    assert rc == 0 and out.is_file(), f"--no-object-lock never reads the lock config; rc {rc}"
    assert "get_object_lock_configuration" not in [n for n, _ in calls]
    ev = json.loads(out.read_text())
    assert ev["bucket_lock"] == OWNER_FLAG_LOCK


def test_without_the_flag_the_evidence_is_unchanged(tmp_path):
    ev = fixture_evidence(tmp_path)
    assert "anchor_objects" not in ev, "a run without the flag leaves no anchor object"
    assert set(ev["bucket_lock"]) == set(BUCKET_LOCK_KEYS), ev["bucket_lock"]
    assert all(probe(ev, n)["result"] == "pass" for n in PROBES)
    assert len(ev["locked_objects"]) == 2


# ── the evidence verifier on --no-object-lock evidence ───────────────────────


def test_verifier_accepts_a_no_lock_body_but_requires_verified_live(tmp_path):
    ev = _live_nolock(tmp_path)
    assert ev["passed"] is True
    assert _body(ev) == [], _body(ev)
    assert any(p.startswith("verified_live") for p in _full(ev)), _full(ev)


@pytest.mark.parametrize("name", LOCK_PROBES)
def test_verifier_rejects_a_skipped_lock_probe_edited_to_pass(tmp_path, name):
    ev = _live_nolock(tmp_path)
    row = probe(ev, name)
    row["result"], row["passed"], row["reason"] = "pass", True, "the capability behaves as lane D needs"
    probs = _body(ev)
    assert any(name in p for p in probs), probs


def test_verifier_rejects_a_skip_without_the_owner_flag(tmp_path):
    ev = relabel_as_live(fixture_evidence(tmp_path))
    row = probe(ev, "compliance_retention_unshortenable")
    row.update(result="skipped", passed=False, reason=SKIP_REASON, requests=[], objects=[],
               observations={})
    ev["locked_objects"] = [lo for lo in ev["locked_objects"]
                            if not lo["key"].endswith("compliance.bin")]
    ev["cleanup"]["remaining_locked"] = [lo["key"] for lo in ev["locked_objects"]]
    probs = _body(ev)
    assert any("skipped" in p for p in probs), probs


def test_verifier_rejects_skipping_a_non_lock_probe(tmp_path):
    ev = _live_nolock(tmp_path)
    row = probe(ev, "upload_part_copy_2_5gb")
    row.update(result="skipped", passed=False, reason=SKIP_REASON, requests=[], objects=[],
               observations={})
    probs = _body(ev)
    assert any("upload_part_copy_2_5gb" in p and "skipped" in p for p in probs), probs


def test_verifier_rejects_a_skipped_row_that_did_something(tmp_path):
    for edit in (lambda r: r.update(reason="no lock"),
                 lambda r: r.update(passed=True),
                 lambda r: r.update(requests=[{"op": "GetObjectLockConfiguration",
                                               "http_status": 403, "request_id": "x-1"}]),
                 lambda r: r.update(observations={"bucket_lock_enabled": False})):
        ev = _live_nolock(tmp_path)
        edit(probe(ev, "object_lock_multipart"))
        probs = _body(ev)
        assert any("object_lock_multipart" in p for p in probs), probs


def test_verifier_rejects_an_edited_owner_flag_record(tmp_path):
    for bl in ({"lock_enabled": True, "asserted_by": "owner_flag"},
               {"lock_enabled": False, "asserted_by": "someone"},
               {"lock_enabled": False, "asserted_by": "owner_flag", "http_status": 200}):
        ev = _live_nolock(tmp_path)
        ev["bucket_lock"] = bl
        assert any("bucket_lock" in p for p in _body(ev)), bl


def test_verifier_rejects_missing_or_bad_anchor_objects(tmp_path):
    ev = _live_nolock(tmp_path)
    ev["anchor_objects"] = []
    assert any("anchor_objects" in p for p in _body(ev))
    ev = _live_nolock(tmp_path)
    del ev["anchor_objects"]
    assert any("anchor_objects" in p for p in _body(ev))
    ev = _live_nolock(tmp_path)
    ev["anchor_objects"][0]["key"] = probe(ev, "subtoken_scope")["objects"][0]["key"]
    assert any("anchor_objects[0]" in p for p in _body(ev))
    ev = _live_nolock(tmp_path)
    ev["anchor_objects"][0]["key"] = "lane-d/elsewhere/anchor.bin"
    assert any("outside" in p for p in _body(ev))
    for k, bad in (("sha256", None), ("etag", ""), ("size", 2684354560)):
        ev = _live_nolock(tmp_path)
        ev["anchor_objects"][0][k] = bad
        assert any(p.startswith("anchor_objects[0]") and k in p for p in _body(ev)), (k, _body(ev))
    ev = relabel_as_live(fixture_evidence(tmp_path))
    ev["anchor_objects"] = [{"key": f"{ev['prefix']}{ev['run_id']}/anchor.bin", "version_id": "v1",
                             "size": 1024, "etag": '"x"', "sha256": "0" * 64}]
    assert any("anchor_objects" in p for p in _body(ev)), "no anchor without the flag"


def test_verifier_rejects_locked_objects_in_a_no_lock_run(tmp_path):
    ev = _live_nolock(tmp_path)
    ev["locked_objects"] = [{"key": f"{ev['prefix']}{ev['run_id']}/compliance.bin",
                             "version_id": "v1", "mode": "COMPLIANCE",
                             "retain_until": ev["finished_at"], "size": 1024, "etag": '"x"',
                             "sha256": "0" * 64}]
    assert _body(ev) != []


# ── storage_probe verify on --no-object-lock evidence ────────────────────────


def _refused(sp, ev: dict, fake: FakeS3, code: str) -> None:
    with pytest.raises(sp.VerifyRefused) as ei:
        sp.verify_live(ev, fake)
    assert getattr(ei.value, "code", None) == code, (getattr(ei.value, "code", None), str(ei.value))
    assert fake.forbidden == [], f"verify must be read-only, it called {fake.forbidden}"


def _verified(sp, tmp_path, **kw) -> tuple[dict, FakeS3]:
    ev, fake = anchor_pair(_live_nolock(tmp_path), **kw)
    ev["verified_live"] = sp.verify_live(ev, fake)
    return ev, fake


def test_verify_a_no_lock_run_against_its_anchor_is_accepted(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    rec = sp.verify_live(ev, fake)
    assert isinstance(rec, dict) and rec.get("verified") is True, rec
    assert rec.get("binding") == ANCHOR_BINDING, rec.get("binding")
    assert rec.get("body_sha256") == body_sha256(ev)
    assert {o["key"] for o in rec["objects"]} == {a["key"] for a in ev["anchor_objects"]}
    for o in rec["objects"]:
        assert o["mode"] is None and o["retain_until"] is None, o
        assert o["request_ids"] and all(str(r).startswith("live-fake-") for r in o["request_ids"])
    ev["verified_live"] = rec
    assert _full(ev) == [], _full(ev)
    assert verified_live_problems(ev, check_git=False) == []
    assert fake.forbidden == []
    for a in ev["anchor_objects"]:
        want = {"Bucket": ev["bucket"], "Key": a["key"], "VersionId": a["version_id"]}
        for op in ("head_object", "get_object"):
            assert any(o == op and all(kw.get(k) == v for k, v in want.items())
                       for o, kw in fake.calls), f"verify did not {op} {want}: {fake.calls}"
    root = f"{ev['prefix']}{ev['run_id']}/"
    assert any(kw.get("Prefix") == root for o, kw in fake.calls
               if o in ("list_object_versions", "list_objects_v2")), fake.calls


def test_verify_an_unversioned_anchor_sends_no_version_id(tmp_path, monkeypatch):
    """Hippius denies GetBucketVersioning: an anchor may carry no version id,
    and verify then reads it without a VersionId parameter."""
    sp = probe_module(monkeypatch)
    ev, fake = _verified(sp, tmp_path, version_ids=False)
    assert _full(ev) == [], _full(ev)
    reads = [kw for o, kw in fake.calls if o in ("head_object", "get_object")]
    assert reads and all("VersionId" not in kw for kw in reads), reads


class _NoVersionListing(FakeS3):
    def list_object_versions(self, **kw):
        self.calls.append(("list_object_versions", dict(kw)))
        raise self._err("ListObjectVersions", "NotImplemented", 501)

    def get_paginator(self, name: str):
        if name == "list_object_versions":
            self.calls.append(("get_paginator", {"name": name}))

            class _P:
                def paginate(_s, **kw):
                    return iter([self.list_object_versions(**kw)])
            return _P()
        return super().get_paginator(name)


def test_verify_a_no_lock_run_falls_back_to_a_plain_listing(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, plain = anchor_pair(_live_nolock(tmp_path))
    fake = _NoVersionListing(plain.bucket, lock_enabled=False)
    fake.objects = plain.objects
    rec = sp.verify_live(ev, fake)
    assert rec.get("verified") is True and fake.forbidden == []
    assert any(o == "list_objects_v2" for o, _ in fake.calls), fake.calls


def test_verify_a_no_lock_run_refusals(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _live_nolock(tmp_path)
    _refused(sp, ev, FakeS3(ev["bucket"], lock_enabled=False), "anchor_object_missing")
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    v = fake.objects[ev["anchor_objects"][0]["key"]][-1]
    v["data"] = bytes(len(v["data"]))
    _refused(sp, ev, fake, "anchor_object_mismatch")
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    ev["anchor_objects"][0]["etag"] = '"' + "0" * 32 + '"'
    _refused(sp, ev, fake, "anchor_object_mismatch")
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    ev["anchor_objects"] = []
    _refused(sp, ev, fake, "no_anchor_objects")
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    fake.put_version(probe(ev, "single_put_limit")["objects"][0]["key"], "v-left", b"left")
    _refused(sp, ev, fake, "run_root_not_clean")
    ev, fake = anchor_pair(_live_nolock(tmp_path))
    fake.bucket = "some-other-bucket"
    _refused(sp, ev, fake, "anchor_object_missing")


def test_a_no_lock_record_is_hash_bound_and_tamper_evident(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, _ = _verified(sp, tmp_path)
    assert _full(ev) == []
    for edit in (lambda e: probe(e, "presigned_upload_part").update(reason="edited"),
                 lambda e: e.update(recommended_path="upload_relay"),
                 lambda e: e["anchor_objects"][0].update(size=e["anchor_objects"][0]["size"] + 1)):
        e2 = copy.deepcopy(ev)
        edit(e2)
        assert any("body_sha256" in p for p in verified_live_problems(e2, check_git=False))
    for edit in (lambda r: r.pop("binding"),
                 lambda r: r["objects"][0].update(sha256="1" * 64),
                 lambda r: r["objects"][0].update(request_ids=["fixture-3"]),
                 lambda r: r.update(objects=[]),
                 lambda r: r["listing"]["keys"].append(f"{ev['prefix']}{ev['run_id']}/x.bin")):
        e2 = copy.deepcopy(ev)
        edit(e2["verified_live"])
        assert verified_live_problems(e2, check_git=False), "an edited record must be refused"


def test_lock_evidence_still_needs_its_locked_objects(tmp_path, monkeypatch):
    """The anchor binding is only for --no-object-lock evidence: lock-mode
    evidence without locked objects is still `no_locked_objects`."""
    sp = probe_module(monkeypatch)
    ev = relabel_as_live(fixture_evidence(tmp_path))
    ev["locked_objects"] = []
    ev["anchor_objects"] = [{"key": f"{ev['prefix']}{ev['run_id']}/anchor.bin",
                             "version_id": "v1", "size": 1024, "etag": '"x"', "sha256": "0" * 64}]
    _refused(sp, ev, FakeS3(ev["bucket"]), "no_locked_objects")


def test_fake_client_error_shape_is_boto3_like():
    """Guard for the helpers this file relies on."""
    e = FakeClientError("HeadObject", "NoSuchKey", 404, "live-fake-1")
    assert e.response["Error"]["Code"] == "NoSuchKey"
