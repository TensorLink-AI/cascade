"""W3a (RV4): the probe refuses a bucket with a DEFAULT Object Lock retention rule.

On a lock-enabled bucket whose ``ObjectLockConfiguration.Rule`` carries a
``DefaultRetention`` (any mode, any period), every object the probe writes
would inherit the lock: ~11 GB of test payloads undeletable for the rule's
period. The probe reads the bucket's lock configuration
(GetObjectLockConfiguration) BEFORE writing anything and refuses such a
bucket with the typed code ``bucket_default_retention`` (exit 2, no evidence,
no object written). With no default rule it proceeds and records the read in
the evidence (``bucket_lock``). So the probe can never lock more than its two
<= 4096-byte objects, for <= 1 day.

Pinned surfaces: ``storage_probe.bucket_lock_preflight(client, bucket)`` on a
boto3-shaped client (:class:`helpers.FakeS3`), the fixture model's
``default_retention`` field, and ``storage_probe.main`` with a write-counting
``FixtureBackend``. Contract: README.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import (
    DEFAULT_RETENTION_CODE,
    TEST_BUCKET,
    FakeS3,
    base_args,
    dry_run,
    model_file,
    probe_module,
    run_probe,
)

# The FixtureBackend methods that create, change or delete an object, an
# upload or a bucket policy.
WRITE_OPS = ("presigned_upload_part", "presigned_put", "create_multipart", "upload_part",
             "upload_part_copy", "complete_multipart", "put_object", "copy_object",
             "put_object_retention", "delete_object", "put_bucket_policy",
             "delete_bucket_policy", "abort_multipart")

RULES = [{"Mode": "COMPLIANCE", "Days": 1}, {"Mode": "COMPLIANCE", "Days": 3650},
         {"Mode": "GOVERNANCE", "Days": 30}, {"Mode": "COMPLIANCE", "Years": 1},
         {"Mode": "GOVERNANCE", "Years": 5}]
MODEL_RULES = [{"mode": "COMPLIANCE", "days": 30}, {"mode": "GOVERNANCE", "days": 1},
               {"mode": "COMPLIANCE", "years": 1}]


def _code(exc: BaseException) -> str | None:
    return getattr(exc, "code", None)


# ── bucket_lock_preflight on a boto3-shaped client ───────────────────────────


@pytest.mark.parametrize("rule", RULES, ids=lambda r: "-".join(f"{k}{v}" for k, v in r.items()))
def test_preflight_refuses_any_default_retention_rule(monkeypatch, rule):
    sp = probe_module(monkeypatch)
    preflight = sp.bucket_lock_preflight
    fake = FakeS3(TEST_BUCKET, default_retention=rule)
    with pytest.raises(sp.UsageError) as ei:
        preflight(fake, TEST_BUCKET)
    assert _code(ei.value) == DEFAULT_RETENTION_CODE, (_code(ei.value), str(ei.value))
    assert DEFAULT_RETENTION_CODE in str(ei.value), str(ei.value)
    assert fake.calls == [("get_object_lock_configuration", {"Bucket": TEST_BUCKET})], fake.calls
    assert fake.forbidden == [], f"the preflight wrote to the bucket: {fake.forbidden}"


def test_preflight_without_a_default_rule_proceeds_and_reports_the_read(monkeypatch):
    sp = probe_module(monkeypatch)
    preflight = sp.bucket_lock_preflight
    for fake in (FakeS3(TEST_BUCKET), FakeS3(TEST_BUCKET, rule_present=True)):
        got = preflight(fake, TEST_BUCKET)
        assert isinstance(got, dict), got
        assert got.get("lock_enabled") is True and got.get("default_retention") is None, got
        assert got.get("http_status") == 200, got
        assert str(got.get("request_id", "")).startswith("live-fake-"), got
        assert fake.ops() == ["get_object_lock_configuration"] and fake.forbidden == []


def test_preflight_on_a_bucket_without_object_lock_proceeds_unlocked(monkeypatch):
    sp = probe_module(monkeypatch)
    preflight = sp.bucket_lock_preflight
    fake = FakeS3(TEST_BUCKET, lock_enabled=False)
    got = preflight(fake, TEST_BUCKET)
    assert isinstance(got, dict), got
    assert got.get("lock_enabled") is False and got.get("default_retention") is None, got
    assert got.get("http_status") == 404, got
    assert fake.forbidden == []


@pytest.mark.parametrize("err", [("AccessDenied", 403), ("InternalError", 500)])
def test_preflight_refuses_an_unreadable_lock_configuration(monkeypatch, err):
    """Can't tell whether a default rule exists -> refuse (fail closed)."""
    sp = probe_module(monkeypatch)
    preflight = sp.bucket_lock_preflight
    fake = FakeS3(TEST_BUCKET, lock_error=err)
    with pytest.raises(sp.UsageError) as ei:
        preflight(fake, TEST_BUCKET)
    assert _code(ei.value) == "bucket_lock_unreadable", (_code(ei.value), str(ei.value))
    assert fake.forbidden == []


# ── the CLI with a fixture model that has a default rule ─────────────────────


@pytest.mark.parametrize("rule", MODEL_RULES, ids=lambda r: "-".join(f"{k}{v}" for k, v in r.items()))
def test_fixture_bucket_with_default_retention_is_refused_and_nothing_written(tmp_path, rule):
    model = model_file(tmp_path, lambda m: m.update(default_retention=rule))
    out = tmp_path / "o.json"
    _, _, proc = dry_run(tmp_path, fixture=model, out=out, expect_rc=2)
    assert DEFAULT_RETENTION_CODE in proc.stderr, proc.stderr[-2000:]
    assert not out.exists(), "a refused run writes no evidence"


@pytest.mark.parametrize("bad", ["yes", 30, {"mode": "COMPLIANCE"}, {"mode": "LEGAL", "days": 1},
                                 {"mode": "COMPLIANCE", "days": 0},
                                 {"mode": "COMPLIANCE", "days": 1, "years": 1}])
def test_bad_fixture_default_retention_is_refused(tmp_path, bad):
    model = model_file(tmp_path, lambda m: m.update(default_retention=bad))
    out = tmp_path / "o.json"
    proc = run_probe(tmp_path, *base_args(), "--dry-run", "--fixture", str(model),
                     "--out", str(out))
    assert proc.returncode == 2, (bad, proc.returncode, proc.stderr[-2000:])
    assert not out.exists()


def test_fixture_without_default_retention_key_means_none(tmp_path):
    model = model_file(tmp_path, lambda m: m.pop("default_retention", None))
    _, ev, _ = dry_run(tmp_path, fixture=model, out=tmp_path / "e.json", expect_rc=0)
    bl = ev.get("bucket_lock")
    assert isinstance(bl, dict) and bl.get("default_retention") is None, bl


# ── in-process: the refusal comes before ANY write ───────────────────────────


def _counting_backend(base):
    writes: list[str] = []

    def wrap(name):
        orig = getattr(base, name)

        def f(self, *a, **k):
            writes.append(name)
            return orig(self, *a, **k)
        return f

    cls = type("CountingFixtureBackend", (base,),
               {n: wrap(n) for n in WRITE_OPS if hasattr(base, n)})
    return cls, writes


def _main(sp, monkeypatch, tmp_path: Path, model: Path) -> tuple[int, list[str], Path]:
    cls, writes = _counting_backend(sp.FixtureBackend)
    monkeypatch.setattr(sp, "FixtureBackend", cls)
    out = tmp_path / "e.json"
    rc = sp.main([*base_args(), "--dry-run", "--fixture", str(model), "--out", str(out)])
    return rc, writes, out


def test_default_retention_refusal_happens_before_any_write(tmp_path, monkeypatch, capsys):
    sp = probe_module(monkeypatch)
    model = model_file(tmp_path, lambda m: m.update(
        default_retention={"mode": "COMPLIANCE", "days": 30}))
    rc, writes, out = _main(sp, monkeypatch, tmp_path, model)
    err = capsys.readouterr().err
    assert rc == 2, f"a default-retention bucket must be refused (exit 2), got {rc}"
    assert writes == [], f"the probe wrote before refusing: {writes}"
    assert DEFAULT_RETENTION_CODE in err, err[-2000:]
    assert not out.exists()


def test_counting_backend_sees_writes_on_a_normal_bucket(tmp_path, monkeypatch, capsys):
    """Control for the test above: the same harness does see the writes."""
    sp = probe_module(monkeypatch)
    model = model_file(tmp_path)
    rc, writes, out = _main(sp, monkeypatch, tmp_path, model)
    capsys.readouterr()
    assert rc == 0 and out.is_file()
    assert "put_object" in writes and "create_multipart" in writes, writes
