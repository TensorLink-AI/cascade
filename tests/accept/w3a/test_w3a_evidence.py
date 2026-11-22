"""W3a: the live probe evidence verifier (plan §3 conventions, §17.1 L4).

``scripts/laned/accept W3a --evidence evidence/w3a_probe.json`` exports the
file as ``$LANED_EVIDENCE``. The ``test_evidence_*`` tests verify it: schema
``laned_w3a_probe/1``, ``source: "live"``, one recorded result per probe,
every verdict re-derived from its observations, the pre-write lock read
showing no bucket default retention, only COMPLIANCE-locked objects left
behind, no credential anywhere, and a ``verified_live`` record (written by
``storage_probe verify`` after re-reading the locked objects from the live
bucket) hash-bound to the evidence body.

Without ``$LANED_EVIDENCE`` they SKIP with a clear reason, so the probe can
reach main (L2/L3) before the owner's run. When it IS set, any missing,
malformed, fixture or hand-edited evidence fails hard. The
``test_verifier_*`` tests prove the verifier rejects tampering, offline; they
check the evidence BODY of a relabelled fixture run (``relabel_as_live``),
which the full verifier itself refuses (test_w3a_live_verify.py).
Contract: README.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .helpers import (
    PROBES,
    evidence_problems,
    fixture_evidence,
    probe,
    relabel_as_live,
    verified_live_problems,
)


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no owner evidence to verify yet. L4 needs "
                    "`scripts/laned/accept W3a --evidence evidence/w3a_probe.json` green on main "
                    "(run `python -P -m cascade.laned.storage_probe ... --out "
                    "evidence/w3a_probe.json --confirm-live` first; see tests/accept/w3a/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


def _good(tmp_path: Path, mutate=None) -> dict:
    """A relabelled fixture body: consistent, but never full-verifier evidence."""
    return relabel_as_live(fixture_evidence(tmp_path, mutate))


def _probs(ev: dict) -> list[str]:
    return evidence_problems(ev, check_git=False, live_sizes=False, require_verified=False)


# ── the owner's evidence ─────────────────────────────────────────────────────


def test_evidence_schema_and_verdicts_reproduced():
    problems = evidence_problems(_evidence())
    assert problems == [], "W3a evidence rejected:\n  " + "\n  ".join(problems)


def test_evidence_every_probe_has_a_live_result():
    ev = _evidence()
    assert ev.get("source") == "live", ev.get("source")
    res = {p.get("name"): p.get("result") for p in ev.get("probes", [])}
    assert set(res) == set(PROBES), sorted(res)
    errored = {n: r for n, r in res.items() if r not in ("pass", "fail")}
    assert not errored, f"probes without a recorded live result (re-run them): {errored}"
    assert ev.get("passed") is True, ev.get("passed")


def test_evidence_was_verified_against_the_live_bucket():
    ev = _evidence()
    problems = verified_live_problems(ev)
    assert problems == [], ("W3a evidence is not verified against the live bucket:\n  "
                            + "\n  ".join(problems))


def test_evidence_bucket_had_no_default_retention():
    ev = _evidence()
    bl = ev.get("bucket_lock")
    assert isinstance(bl, dict), f"the evidence records no pre-write lock read: {bl!r}"
    assert bl.get("default_retention") is None, bl


def test_evidence_cleanup_left_only_locked_objects():
    ev = _evidence()
    cleanup = ev.get("cleanup", {})
    assert cleanup.get("remaining_unlocked") == [], (
        f"unlocked probe objects were left behind: {cleanup.get('remaining_unlocked')}")
    locked = {lo["key"] for lo in ev.get("locked_objects", [])}
    assert set(cleanup.get("remaining_locked", [])) <= locked


# ── the verifier itself (offline) ────────────────────────────────────────────


def test_verifier_accepts_an_untampered_body(tmp_path):
    assert _probs(_good(tmp_path)) == []


def test_verifier_refuses_a_relabelled_body_without_verified_live(tmp_path):
    ev = _good(tmp_path)
    assert _probs(ev) == []
    probs = evidence_problems(ev, check_git=False, live_sizes=False)
    assert any(p.startswith("verified_live") for p in probs), probs


def test_verifier_rejects_fixture_source(tmp_path):
    ev = _good(tmp_path)
    ev["source"] = "fixture"
    assert any(p.startswith("source") for p in _probs(ev))


def test_verifier_rejects_live_sizes_below_2_5_gb(tmp_path):
    ev = _good(tmp_path)
    probs = evidence_problems(ev, check_git=False, live_sizes=True)
    assert any("big_object_bytes" in p for p in probs), probs


def test_verifier_rejects_a_missing_or_duplicate_probe(tmp_path):
    ev = _good(tmp_path)
    ev["probes"] = [p for p in ev["probes"] if p["name"] != "subtoken_scope"]
    assert any("subtoken_scope" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["probes"].append(dict(probe(ev, "single_put_limit")))
    assert any("single_put_limit" in p for p in _probs(ev))


@pytest.mark.parametrize("name,change", [
    ("presigned_upload_part", {"presigned_upload_part": "rejected"}),
    ("prefix_bucket_policy", {"bucket_policy": "accepted_not_enforced"}),
    ("subtoken_scope", {"subtoken": "unscoped"}),
    ("upload_part_copy_2_5gb", {"upload_part_copy": "rejected"}),
    ("compliance_retention_unshortenable", {"compliance": "shortenable"}),
    ("object_lock_multipart", {"lock_on_multipart": "ignored"}),
    ("single_put_limit", {"single_put_max_bytes": 600000}),
])
def test_verifier_rejects_a_fail_edited_to_pass(tmp_path, name, change):
    ev = _good(tmp_path, lambda m: m.update(change))
    row = probe(ev, name)
    assert row["result"] == "fail"
    row["result"], row["passed"] = "pass", True
    probs = _probs(ev)
    assert any(name in p and "result" in p for p in probs), probs


def test_verifier_rejects_an_edited_recommended_path(tmp_path):
    ev = _good(tmp_path, lambda m: m.update(presigned_upload_part="rejected"))
    assert ev["recommended_path"] == "single_presigned_put"
    ev["recommended_path"] = "presigned_multipart"
    assert any(p.startswith("recommended_path") for p in _probs(ev))


def test_verifier_rejects_an_error_hidden_as_passed(tmp_path):
    ev = relabel_as_live(fixture_evidence(tmp_path, lambda m: m.update(subtoken=None),
                                          expect_rc=1))
    assert ev["passed"] is False
    ev["passed"] = True
    assert any(p.startswith("passed") for p in _probs(ev))


def test_verifier_rejects_leftover_unlocked_objects(tmp_path):
    ev = _good(tmp_path)
    ev["cleanup"]["remaining_unlocked"] = [probe(ev, "single_put_limit")["objects"][0]["key"]]
    assert any(p.startswith("passed") for p in _probs(ev))
    ev["passed"] = False
    assert _probs(ev) == []  # consistent, but the L4 gate test refuses it


def test_verifier_rejects_unreported_or_long_locks(tmp_path):
    ev = _good(tmp_path)
    ev["locked_objects"] = ev["locked_objects"][:1]
    ev["cleanup"]["remaining_locked"] = [ev["locked_objects"][0]["key"]]
    assert any("not reported" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["locked_objects"][0]["retain_until"] = "2099-01-01T00:00:00+00:00"
    assert any("shortest retention" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["params"]["lock_retention_seconds"] = 31536000
    assert any("lock_retention_seconds" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["locked_objects"][0]["size"] = 2684354560
    assert any("bytes" in p for p in _probs(ev))


def test_verifier_rejects_locked_objects_without_etag_or_sha(tmp_path):
    for k, bad in (("etag", None), ("etag", ""), ("sha256", None), ("sha256", "0" * 64),
                   ("version_id", None)):
        ev = _good(tmp_path)
        ev["locked_objects"][0][k] = bad
        probs = _probs(ev)
        assert any(p.startswith("locked_objects[0]") and k in p for p in probs), (k, bad, probs)


def test_verifier_rejects_a_default_retention_bucket_or_missing_lock_read(tmp_path):
    ev = _good(tmp_path)
    bl = ev.get("bucket_lock")
    assert isinstance(bl, dict), f"fixture evidence records no pre-write lock read: {bl!r}"
    bl["default_retention"] = {"mode": "COMPLIANCE", "days": 30}
    assert any("bucket_default_retention" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev.pop("bucket_lock", None)
    assert any("bucket_lock" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["bucket_lock"]["request_id"] = "fixture-1"
    assert any("bucket_lock.request_id" in p for p in _probs(ev))


def test_verifier_rejects_credentials_urls_and_signatures(tmp_path):
    for leak in ("https://s3.hippius.com/b/k?X-Amz-Signature=abc",
                 "X-Amz-Credential=AKIA/2026", "Authorization: AWS4-HMAC-SHA256"):
        ev = _good(tmp_path)
        probe(ev, "presigned_upload_part")["reason"] = leak
        assert any(p.startswith("leak") for p in _probs(ev)), leak
    ev = _good(tmp_path)
    ev["endpoint_host"] = "user:pw@s3.hippius.com"
    assert any(p.startswith("endpoint_host") for p in _probs(ev))


def test_verifier_rejects_a_hippius_env_value_in_the_evidence(tmp_path, monkeypatch):
    ev = _good(tmp_path)
    monkeypatch.setenv("HIPPIUS_S3_SECRET_KEY", "w3a-verifier-canary-secret")
    probe(ev, "subtoken_scope")["reason"] = "w3a-verifier-canary-secret"
    assert any(p.startswith("leak") for p in _probs(ev))


def test_verifier_rejects_objects_outside_the_run_prefix(tmp_path):
    ev = _good(tmp_path)
    probe(ev, "subtoken_scope")["objects"][0]["key"] = "lane-d/5Fhot/abc/weights.safetensors"
    assert any("outside" in p for p in _probs(ev))
    ev = _good(tmp_path)
    ev["prefix"] = "lane-d/"
    assert any(p.startswith("prefix") for p in _probs(ev))
    ev = _good(tmp_path)
    ev["bucket"] = "cascade-lane-d-vault"
    assert any("chain.toml bucket" in p for p in _probs(ev))


def test_verifier_rejects_missing_or_fixture_request_ids(tmp_path):
    ev = _good(tmp_path)
    row = probe(ev, "upload_part_copy_2_5gb")
    req = next(r for r in row["requests"] if r["http_status"] is not None)
    req["request_id"] = None
    assert any("request id" in p for p in _probs(ev))
    ev = _good(tmp_path)
    row = probe(ev, "upload_part_copy_2_5gb")
    req = next(r for r in row["requests"] if r["http_status"] is not None)
    req["request_id"] = "fixture-7"
    assert any("fixture request id" in p for p in _probs(ev))


def test_verifier_rejects_a_digest_mismatch_marked_pass(tmp_path):
    ev = _good(tmp_path)
    obs = probe(ev, "upload_part_copy_2_5gb")["observations"]
    obs["dest_sha256"] = "0" * 63 + "1"
    assert any("upload_part_copy_2_5gb" in p and "result" in p for p in _probs(ev))
    ev = _good(tmp_path)
    obs = probe(ev, "presigned_upload_part")["observations"]
    obs["authorization_header_sent"] = True
    assert any("authorization_header_sent" in p for p in _probs(ev))


def test_verifier_rejects_a_probe_policy_left_on_the_bucket(tmp_path):
    ev = _good(tmp_path)
    probe(ev, "prefix_bucket_policy")["observations"]["policy_removed"] = False
    assert any("policy_removed" in p for p in _probs(ev))


def test_verifier_rejects_a_dirty_or_unknown_tool_sha(tmp_path):
    ev = _good(tmp_path)
    ev["tool"]["git_dirty"] = True
    assert any(p.startswith("tool.git_dirty") for p in _probs(ev))
    ev = _good(tmp_path)
    ev["tool"]["git_sha"] = "0123456789abcdef0123456789abcdef01234567"
    assert any("not a commit" in p for p in evidence_problems(ev, check_git=True,
                                                               live_sizes=False))
