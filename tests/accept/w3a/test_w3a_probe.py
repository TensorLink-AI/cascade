"""W3a: the owner's live Hippius storage probe, dry-run/fixture mode only.

``python -P -m cascade.laned.storage_probe --bucket B --prefix P/
--other-bucket O --out evidence/w3a_probe.json --confirm-live`` runs seven
probes against live Hippius S3 and writes ``laned_w3a_probe/1`` evidence.
These tests never reach Hippius: every run is ``--dry-run`` with boto3,
botocore and bittensor unimportable and every socket connect refused.
Contract: README.md.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from .helpers import (
    MAX_LOCKED_BYTES,
    MODULE,
    PROBES,
    REPO,
    TEST_PREFIX,
    as_live_evidence,
    base_args,
    dry_run,
    evidence_problems,
    fixture_evidence,
    leak_problems,
    model_file,
    probe,
    production_buckets,
    require_probe,
    run_probe,
)

CANARY = {"HIPPIUS_S3_ACCESS_KEY": "W3ACANARYACCESSKEY0001",
          "HIPPIUS_S3_SECRET_KEY": "w3a-canary-secret-value-0002",
          "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY": "W3ACANARYSUBTOKENKEY03",
          "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY": "w3a-canary-subtoken-secret-0004"}


def _usage(tmp_path: Path, *args: str, env_extra: dict | None = None):
    require_probe()
    proc = run_probe(tmp_path, *args, env_extra=env_extra)
    assert proc.returncode == 2, (args, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert proc.stderr.strip(), f"a refusal says why on stderr: {args}"
    return proc


# ── the module and its surface ───────────────────────────────────────────────


def test_storage_probe_module_exists():
    require_probe()


def test_probe_makes_no_bucket_admin_calls_and_reads_no_dotenv():
    require_probe()
    src = Path(importlib.util.find_spec(MODULE).origin).read_text()
    tree = ast.parse(src)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
              and isinstance(n.value, str)}
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imports.add(n.module or "")
            names |= {a.name for a in n.names}
    admin = {"create_bucket", "delete_bucket", "put_bucket_versioning",
             "put_object_lock_configuration", "put_bucket_acl", "put_object_acl",
             "put_public_access_block", "delete_public_access_block",
             "put_bucket_lifecycle_configuration", "put_bucket_lifecycle"}
    used = (attrs | names | consts) & admin
    assert not used, f"bucket administration in the probe: {used}"
    assert not ({"S3Store", "_ensure_bucket"} & (attrs | names)), \
        "S3Store/_ensure_bucket create missing buckets; the probe must not use them"
    assert not any(m == "dotenv" or m.startswith("dotenv.") for m in imports), imports
    assert "load_dotenv" not in attrs | names
    assert not any(m == "bittensor" or m.startswith("bittensor.") for m in imports), imports
    for env_name in ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY",
                     "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY", "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY"):
        assert env_name in src, f"credentials come from the environment by name: {env_name}"


# ── dry-run without a fixture: the plan ──────────────────────────────────────


def test_dry_run_without_fixture_prints_the_plan_and_writes_nothing(tmp_path):
    out = tmp_path / "w3a_probe.json"
    printed, ev, _ = dry_run(tmp_path, out=out, expect_rc=0)
    assert ev is None and not out.exists(), "a dry-run without --fixture writes no evidence"
    assert printed["dry_run"] is True and printed["fixture"] is None
    plan = printed["would_do"]
    assert isinstance(plan, dict) and set(plan) == set(PROBES), plan
    for name in PROBES:
        assert isinstance(plan[name], list) and plan[name], f"no plan for {name}"
        assert all(isinstance(s, str) and s for s in plan[name]), plan[name]
    assert not leak_problems(printed)


# ── refusals (exit 2, before any network use) ────────────────────────────────


def test_live_mode_requires_out_and_confirm_live(tmp_path):
    _usage(tmp_path, *base_args(), "--confirm-live", env_extra=CANARY)
    _usage(tmp_path, *base_args(), "--out", str(tmp_path / "o.json"), env_extra=CANARY)
    assert not (tmp_path / "o.json").exists()


def test_live_mode_without_credentials_refuses_and_names_the_variables(tmp_path):
    proc = _usage(tmp_path, *base_args(), "--out", str(tmp_path / "o.json"), "--confirm-live")
    assert "HIPPIUS_S3_ACCESS_KEY" in proc.stderr and "HIPPIUS_S3_SECRET_KEY" in proc.stderr
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("bad", ["", "/abs/", "no-trailing-slash", "lane-d/x/", "a/../b/",
                                 "UPPER/", "a//b/", "./x/"])
def test_bad_prefix_is_a_usage_error(tmp_path, bad):
    model = model_file(tmp_path)
    _usage(tmp_path, *base_args(prefix=bad), "--dry-run", "--fixture", str(model),
           "--out", str(tmp_path / "o.json"))
    assert not (tmp_path / "o.json").exists()


def test_chain_toml_buckets_are_refused(tmp_path):
    prod = production_buckets()
    assert {"cascade-manifests", "cascade-lane-d-vault"} <= prod, prod
    model = model_file(tmp_path)
    for b in ("cascade-manifests", "cascade-lane-d-vault"):
        _usage(tmp_path, *base_args(bucket=b), "--dry-run", "--fixture", str(model),
               "--out", str(tmp_path / "o.json"))
    assert not (tmp_path / "o.json").exists()


def test_other_bucket_must_differ_from_bucket(tmp_path):
    model = model_file(tmp_path)
    _usage(tmp_path, *base_args(other="cascade-w3a-probe-test"), "--dry-run", "--fixture",
           str(model), "--out", str(tmp_path / "o.json"))


@pytest.mark.parametrize("endpoint", ["https://user:pw@s3.hippius.com",
                                      "https://s3.hippius.com/?X-Amz-Credential=a",
                                      "https://s3.hippius.com/#frag", "ftp://s3.hippius.com"])
def test_endpoint_with_credentials_or_query_is_refused(tmp_path, endpoint):
    model = model_file(tmp_path)
    proc = _usage(tmp_path, *base_args(), "--endpoint", endpoint, "--dry-run", "--fixture",
                  str(model), "--out", str(tmp_path / "o.json"))
    assert "pw@" not in proc.stderr and "pw@" not in proc.stdout


@pytest.mark.parametrize("secs", ["59", "86401", "31536000", "0", "-5"])
def test_lock_retention_outside_one_minute_to_one_day_is_refused(tmp_path, secs):
    model = model_file(tmp_path)
    _usage(tmp_path, *base_args(), "--lock-retention-seconds", secs, "--dry-run", "--fixture",
           str(model), "--out", str(tmp_path / "o.json"))


def test_fixture_payloads_are_capped_at_16_mib(tmp_path):
    model = model_file(tmp_path)
    _usage(tmp_path, *base_args(), "--big-object-bytes", "2684354560", "--dry-run",
           "--fixture", str(model), "--out", str(tmp_path / "o.json"))
    _usage(tmp_path, *base_args(), "--single-put-bytes", "1048576,33554432", "--dry-run",
           "--fixture", str(model), "--out", str(tmp_path / "o.json"))


def test_non_empty_prefix_is_refused_and_nothing_is_written(tmp_path):
    model = model_file(tmp_path, lambda m: m.update(prefix_object_count=3))
    _usage(tmp_path, *base_args(), "--dry-run", "--fixture", str(model),
           "--out", str(tmp_path / "o.json"))
    assert not (tmp_path / "o.json").exists()


def test_bad_fixture_schema_is_refused(tmp_path):
    model = model_file(tmp_path, lambda m: m.update(schema="laned_w3a_fixture/0"))
    _usage(tmp_path, *base_args(), "--dry-run", "--fixture", str(model),
           "--out", str(tmp_path / "o.json"))


# ── dry-run with a fixture: the evidence it would write ──────────────────────


def test_fixture_evidence_schema_and_every_probe(tmp_path):
    out = tmp_path / "w3a_probe.json"
    printed, ev, _ = dry_run(tmp_path, fixture=model_file(tmp_path), out=out, expect_rc=0)
    assert ev is not None, "a dry-run with --fixture writes the evidence it would write"
    assert ev["schema"] == "laned_w3a_probe/1" and ev["wp"] == "W3a"
    assert ev["source"] == "fixture" and ev["passed"] is False, (
        "fixture evidence must never pass: source 'fixture', passed false")
    assert ev["endpoint_host"] == "s3.hippius.com"
    assert ev["bucket"] == "cascade-w3a-probe-test" and ev["prefix"] == TEST_PREFIX
    assert re.fullmatch(r"[0-9a-f]{16}", ev["run_id"]), ev["run_id"]
    assert [p["name"] for p in ev["probes"]] and sorted(p["name"] for p in ev["probes"]) \
        == sorted(PROBES)
    for p in ev["probes"]:
        assert p["result"] == "pass" and p["passed"] is True, (p["name"], p["reason"])
        assert p["requests"] and p["objects"], p["name"]
        for r in p["requests"]:
            if r["http_status"] is not None:
                assert str(r["request_id"]).startswith("fixture-"), r
        for o in p["objects"]:
            assert o["key"].startswith(f"{TEST_PREFIX}{ev['run_id']}/"), o
    assert ev["recommended_path"] == "presigned_multipart"
    assert ev["params"]["big_object_bytes"] == 1048576
    assert ev["params"]["part_bytes"] == 262144
    assert ev["params"]["single_put_bytes"] == [524288, 1048576]
    assert ev["params"]["lock_retention_seconds"] == 86400
    assert printed["dry_run"] is True and printed["complete"] is True
    assert printed["passed"] is False
    assert printed["results"] == {n: "pass" for n in PROBES}
    assert printed["recommended_path"] == "presigned_multipart"
    assert printed["run_id"] == ev["run_id"]


def test_fixture_evidence_relabelled_live_passes_the_verifier(tmp_path):
    ev = as_live_evidence(fixture_evidence(tmp_path))
    assert ev["passed"] is True
    problems = evidence_problems(ev, check_git=False, live_sizes=False)
    assert problems == [], problems


def test_locked_objects_are_tiny_short_lived_and_the_only_leftovers(tmp_path):
    ev = fixture_evidence(tmp_path)
    locked = ev["locked_objects"]
    lock_keys = {o["key"] for n in ("object_lock_multipart", "compliance_retention_unshortenable")
                 for o in probe(ev, n)["objects"]}
    assert len(locked) == 2 and {lo["key"] for lo in locked} == lock_keys
    started = datetime.fromisoformat(ev["started_at"])
    finished = datetime.fromisoformat(ev["finished_at"])
    for lo in locked:
        assert lo["mode"] == "COMPLIANCE" and lo["size"] <= MAX_LOCKED_BYTES, lo
        ru = datetime.fromisoformat(lo["retain_until"])
        assert started < ru <= finished + timedelta(seconds=86400 + 60), lo
    assert ev["cleanup"]["remaining_unlocked"] == []
    assert sorted(ev["cleanup"]["remaining_locked"]) == sorted(lock_keys)
    assert ev["cleanup"]["deleted_versions"] >= 5


def test_shorter_lock_retention_is_used(tmp_path):
    model = model_file(tmp_path)
    _, ev, _ = dry_run(tmp_path, "--lock-retention-seconds", "300", fixture=model,
                       out=tmp_path / "e.json", expect_rc=0)
    assert ev["params"]["lock_retention_seconds"] == 300
    finished = datetime.fromisoformat(ev["finished_at"])
    for lo in ev["locked_objects"]:
        assert datetime.fromisoformat(lo["retain_until"]) <= finished + timedelta(seconds=360)


FLIPS = [
    ("presigned_upload_part", {"presigned_upload_part": "rejected"}, "single_presigned_put"),
    ("single_put_limit", {"single_put_max_bytes": 600000}, "presigned_multipart"),
    ("object_lock_multipart", {"lock_on_multipart": "ignored"}, "presigned_multipart"),
    ("object_lock_multipart", {"lock_on_multipart": "rejected"}, "presigned_multipart"),
    ("prefix_bucket_policy", {"bucket_policy": "accepted_not_enforced"}, "presigned_multipart"),
    ("prefix_bucket_policy", {"bucket_policy": "rejected"}, "presigned_multipart"),
    ("subtoken_scope", {"subtoken": "unscoped"}, "presigned_multipart"),
    ("upload_part_copy_2_5gb", {"upload_part_copy": "rejected"}, "presigned_multipart"),
    ("compliance_retention_unshortenable", {"compliance": "shortenable"}, "presigned_multipart"),
]


@pytest.mark.parametrize("name,change,path", FLIPS, ids=[f"{n}-{list(c.values())[0]}"
                                                       for n, c, _ in FLIPS])
def test_each_capability_flip_fails_only_its_probe(tmp_path, name, change, path):
    ev = fixture_evidence(tmp_path, lambda m: m.update(change))
    got = {p["name"]: p["result"] for p in ev["probes"]}
    assert got[name] == "fail", (name, got, probe(ev, name)["reason"])
    assert all(r == "pass" for n, r in got.items() if n != name), got
    assert ev["recommended_path"] == path
    assert ev["cleanup"]["remaining_unlocked"] == []
    live = as_live_evidence(ev)
    assert live["passed"] is True, "a recorded fail is still a recorded live result"
    assert evidence_problems(live, check_git=False, live_sizes=False) == []


def test_no_presigned_path_at_all_recommends_the_relay(tmp_path):
    ev = fixture_evidence(tmp_path, lambda m: m.update(presigned_upload_part="rejected",
                                                       single_put_max_bytes=600000))
    assert ev["recommended_path"] == "upload_relay"
    assert probe(ev, "single_put_limit")["observations"]["max_accepted_bytes"] == 524288
    assert probe(ev, "single_put_limit")["observations"]["min_rejected_bytes"] == 1048576


def test_rejected_multipart_uploads_are_aborted(tmp_path):
    ev = fixture_evidence(tmp_path, lambda m: m.update(presigned_upload_part="rejected",
                                                       upload_part_copy="rejected"))
    assert ev["cleanup"]["aborted_uploads"] >= 2
    assert ev["cleanup"]["remaining_unlocked"] == []


def test_ignored_lock_leaves_nothing_locked_from_that_probe(tmp_path):
    ev = fixture_evidence(tmp_path, lambda m: m.update(lock_on_multipart="ignored"))
    mpu_keys = {o["key"] for o in probe(ev, "object_lock_multipart")["objects"]}
    assert not mpu_keys & {lo["key"] for lo in ev["locked_objects"]}
    assert len(ev["locked_objects"]) == 1


def test_missing_subtoken_is_an_error_and_exit_1(tmp_path):
    model = model_file(tmp_path, lambda m: m.update(subtoken=None))
    printed, ev, _ = dry_run(tmp_path, fixture=model, out=tmp_path / "e.json", expect_rc=1)
    assert probe(ev, "subtoken_scope")["result"] == "error"
    assert printed["complete"] is False
    assert as_live_evidence(ev)["passed"] is False


def test_preexisting_policy_is_an_error_and_left_untouched(tmp_path):
    model = model_file(tmp_path, lambda m: m.update(bucket_has_policy=True))
    _, ev, _ = dry_run(tmp_path, fixture=model, out=tmp_path / "e.json", expect_rc=1)
    row = probe(ev, "prefix_bucket_policy")
    assert row["result"] == "error"
    assert row["observations"]["preexisting_policy"] is True
    ops = [r["op"] for r in row["requests"]]
    assert not any("PutBucketPolicy" in op or "DeleteBucketPolicy" in op for op in ops), ops


def test_bucket_without_object_lock_errors_both_lock_probes(tmp_path):
    model = model_file(tmp_path, lambda m: m.update(bucket_lock_enabled=False))
    _, ev, _ = dry_run(tmp_path, fixture=model, out=tmp_path / "e.json", expect_rc=1)
    for n in ("object_lock_multipart", "compliance_retention_unshortenable"):
        assert probe(ev, n)["result"] == "error", n
    assert ev["locked_objects"] == []
    assert ev["cleanup"]["remaining_locked"] == []


def test_no_credentials_urls_or_signatures_leak(tmp_path):
    model = model_file(tmp_path)
    out = tmp_path / "e.json"
    printed, ev, proc = dry_run(tmp_path, fixture=model, out=out, expect_rc=0, env_extra=CANARY)
    blob = proc.stdout + proc.stderr + out.read_text()
    for name, value in CANARY.items():
        assert value not in blob, f"the value of {name} leaked"
    assert leak_problems(ev) == [] and leak_problems(printed) == []
    assert not re.search(r"x-amz-signature|x-amz-credential|https?://", blob, re.I), \
        "presigned URLs or signatures leaked"


def test_fixture_tool_sha_is_this_checkout(tmp_path):
    ev = fixture_evidence(tmp_path)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ev["tool"]["module"] == MODULE
    assert ev["tool"]["git_sha"] == head
    assert isinstance(ev["tool"]["git_dirty"], bool)


def test_evidence_file_is_one_json_object(tmp_path):
    ev = fixture_evidence(tmp_path)
    assert json.loads(json.dumps(ev)) == ev
