"""Shared helpers for the W3a acceptance tests (live Hippius probe, owner WP).

Four parts:

* the probe ``python -P -m cascade.laned.storage_probe`` is driven as a
  subprocess in ``--dry-run`` mode ONLY, with ``boto3``/``botocore``/
  ``bittensor`` made unimportable and every socket connect refused, so no
  test can reach Hippius;
* :class:`FakeS3` is a boto3-shaped, in-memory S3 client (reads only; any
  write call is recorded in ``forbidden``). The in-process tests hand it to
  ``storage_probe.bucket_lock_preflight`` and ``storage_probe.verify_live``;
  it can model a bucket default retention rule;
* :func:`evidence_problems` is the evidence verifier. It never trusts the
  probe: it re-derives every probe verdict, the recommended path and the
  overall ``passed`` from the recorded observations, so an edited verdict
  fails. By default it also requires the ``verified_live`` record that
  ``storage_probe verify`` writes after re-reading the locked objects from
  the live bucket, hash-bound to the evidence body, so fixture evidence
  relabelled as live (or evidence edited after verification) is refused;
* :func:`decision_problems` checks ``decisions/STORAGE-LANE-D.md`` against
  the evidence. Contract: README.md.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODULE = "cascade.laned.storage_probe"
MODULE_PATH = "cascade/laned/storage_probe.py"
SCHEMA = "laned_w3a_probe/1"
WP = "W3a"
DECISION = REPO / "decisions" / "STORAGE-LANE-D.md"
EVIDENCE_REL = "evidence/w3a_probe.json"

PROBES = ("presigned_upload_part", "single_put_limit", "object_lock_multipart",
          "prefix_bucket_policy", "subtoken_scope", "upload_part_copy_2_5gb",
          "compliance_retention_unshortenable")
LOCK_PROBES = ("object_lock_multipart", "compliance_retention_unshortenable")
RESULTS = ("pass", "fail", "error")
PATH_IDS = ("presigned_multipart", "single_presigned_put", "upload_relay")

LIVE_MIN_BIG_BYTES = 2_500_000_000
MIN_PART_BYTES = 5 * 1024 * 1024
MAX_PART_BYTES = 512 * 1024 * 1024
MAX_PARTS = 10_000
MIN_LOCK_S, MAX_LOCK_S = 60, 86_400
MAX_LOCKED_BYTES = 4096
MAX_PRESIGN_S = 604_800

TEST_BUCKET = "cascade-w3a-probe-test"
TEST_PREFIX = "w3a-probe/"
OTHER_BUCKET = "cascade-w3a-other-test"

SHA_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[0-9a-f]{16}")
HOST_RE = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*")
PREFIX_RE = re.compile(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*/")
LEAK_RES = [re.compile(p, re.I) for p in (
    r"https?://", r"x-amz-signature", r"x-amz-credential", r"x-amz-security-token",
    r"awsaccesskeyid", r"signature=", r"authorization:")]
TOP_KEYS = ("schema", "wp", "source", "endpoint_host", "region", "bucket", "prefix", "run_id",
            "started_at", "finished_at", "tool", "params", "bucket_lock", "probes",
            "locked_objects", "cleanup", "recommended_path", "passed")
BUCKET_LOCK_KEYS = ("lock_enabled", "default_retention", "http_status", "request_id")
LOCKED_KEYS = ("key", "version_id", "mode", "retain_until", "size", "etag", "sha256")
VERIFY_SCHEMA = "laned_w3a_verify/1"
VERIFY_KEYS = ("schema", "body_sha256", "verified_at", "tool", "endpoint_host", "bucket",
               "run_id", "objects", "listing", "verified")
VERIFY_OBJECT_KEYS = ("key", "version_id", "mode", "retain_until", "size", "etag", "sha256",
                      "request_ids")
DEFAULT_RETENTION_CODE = "bucket_default_retention"
VERIFY_READ_OPS = {"head_object", "get_object_retention", "get_object", "list_object_versions",
                   "list_objects_v2", "get_object_lock_configuration", "get_paginator"}
PROBE_KEYS = ("name", "result", "passed", "reason", "requests", "objects", "observations")
CREDENTIAL_ENV = ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY",
                  "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY", "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY")

# `--no-object-lock` (owner 2026-11-23: Hippius has no Object Lock; README
# "No Object Lock"). The owner asserts the bucket has no Object Lock, so the
# probe never reads the lock configuration, never sends a lock header or a
# retention, skips the two lock probes and leaves one small UNLOCKED anchor
# object for `storage_probe verify` to re-read instead of the locked ones.
NO_LOCK_FLAG = "--no-object-lock"
SKIP_RESULT = "skipped"
SKIP_REASON = "no_object_lock"
OWNER_FLAG_LOCK = {"lock_enabled": False, "asserted_by": "owner_flag"}
ANCHOR_KEYS = ("key", "version_id", "size", "etag", "sha256")
ANCHOR_BINDING = "unlocked_anchor"
MAX_ANCHOR_BYTES = 4096


# ── driving the probe (dry-run only) ─────────────────────────────────────────


def require_probe() -> None:
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError as e:
        raise AssertionError(f"{MODULE} is missing (W3a PATHS): {e}") from None
    assert spec is not None, f"{MODULE} is missing (W3a PATHS)"


def load_model() -> dict:
    return json.loads((FIXTURES / "hippius_model.json").read_text())


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path


def model_file(tmp_path: Path, mutate=None, name: str = "model.json") -> Path:
    m = load_model()
    if mutate:
        mutate(m)
    return write_json(tmp_path / name, m)


_SITECUSTOMIZE = '''\
import os, socket
def _blocked(*a, **k):
    raise OSError("W3a test: network is blocked; --dry-run must not open a connection")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
_m = os.environ.get("W3A_NET_BLOCK_MARKER")
if _m:
    open(_m, "w").write("blocked")
'''


def _sandbox_dir(tmp_path: Path) -> Path:
    """First on PYTHONPATH: unimportable boto3/botocore/bittensor and a
    sitecustomize that refuses every socket connect."""
    d = tmp_path / "w3a_sandbox"
    for pkg in ("boto3", "botocore", "bittensor"):
        (d / pkg).mkdir(parents=True, exist_ok=True)
        (d / pkg / "__init__.py").write_text(
            f"raise ImportError('W3a test: {pkg} is blocked; --dry-run must not touch Hippius')\n")
    (d / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    return d


def run_probe(tmp_path: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("AWS_"))}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join([str(_sandbox_dir(tmp_path)), str(REPO)])
    marker = tmp_path / "net_block_marker"
    env["W3A_NET_BLOCK_MARKER"] = str(marker)
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=180, check=False)
    assert marker.is_file(), "the network block (sitecustomize) did not load in the probe run"
    return proc


def base_args(bucket: str = TEST_BUCKET, prefix: str = TEST_PREFIX,
              other: str = OTHER_BUCKET) -> list[str]:
    return ["--bucket", bucket, "--prefix", prefix, "--other-bucket", other]


def dry_run(tmp_path: Path, *extra: str, fixture: Path | None = None, out: Path | None = None,
            expect_rc: int, args: list[str] | None = None,
            env_extra: dict | None = None) -> tuple[dict, dict | None, subprocess.CompletedProcess]:
    """Run a dry-run; returns (stdout JSON, evidence JSON or None, process)."""
    require_probe()
    argv = (base_args() if args is None else list(args)) + ["--dry-run"]
    if fixture is not None:
        argv += ["--fixture", str(fixture)]
    if out is not None:
        argv += ["--out", str(out)]
    argv += list(extra)
    proc = run_probe(tmp_path, *argv, env_extra=env_extra)
    assert proc.returncode == expect_rc, (
        f"`python -P -m {MODULE} {' '.join(argv)}` exited {proc.returncode}, want {expect_rc}\n"
        f"stdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}")
    try:
        printed = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object ({e}): {proc.stdout[-2000:]!r}\n"
                             f"stderr: {proc.stderr[-2000:]}") from None
    assert isinstance(printed, dict), printed
    evidence = json.loads(out.read_text()) if out is not None and out.is_file() else None
    return printed, evidence, proc


def fixture_evidence(tmp_path: Path, mutate=None, expect_rc: int = 0,
                     name: str = "ev.json") -> dict:
    model = model_file(tmp_path, mutate)
    _, ev, _ = dry_run(tmp_path, fixture=model, out=tmp_path / name, expect_rc=expect_rc)
    assert ev is not None, "a dry-run with --fixture writes the evidence it would write"
    return ev


def probe(ev: dict, name: str) -> dict:
    rows = [p for p in ev["probes"] if p.get("name") == name]
    assert len(rows) == 1, f"want exactly one {name} probe, got {len(rows)}"
    return rows[0]


def relabel_as_live(evidence: dict) -> dict:
    """A fixture-made evidence file RELABELLED as a live run: source ``live``,
    a clean tool checkout, ``fixture-`` request ids renamed, and ``passed``
    restated from the recomputed rule.

    This is the forgery RV4 found. Its body is self-consistent, so the body
    checks (``evidence_problems(..., require_verified=False)``) still run on
    it, which is what the tampering tests need; but the FULL verifier refuses
    it (no ``verified_live`` record), and ``storage_probe.verify_live``
    refuses it against any bucket that does not hold its locked objects.
    No test may show relabelled evidence passing the full verifier without a
    bucket that really holds those objects."""
    ev = copy.deepcopy(evidence)
    ev["source"] = "live"
    if isinstance(ev.get("tool"), dict):
        ev["tool"]["git_dirty"] = False

    def rename(rec: object) -> None:
        if isinstance(rec, dict):
            rid = rec.get("request_id")
            if isinstance(rid, str) and rid.startswith("fixture-"):
                rec["request_id"] = "relabelled-" + rid[len("fixture-"):]

    for p in ev.get("probes", []):
        for r in p.get("requests", []):
            rename(r)
    rename(ev.get("bucket_lock"))
    ev.pop("verified_live", None)
    ev["passed"] = passed_rule(ev)
    return ev


# ── the in-process module, offline ───────────────────────────────────────────


def probe_module(monkeypatch=None):
    """``cascade.laned.storage_probe`` imported in-process. With
    ``monkeypatch``, every socket connect is refused and boto3/botocore are
    made unimportable first, so an in-process call can't reach Hippius."""
    require_probe()
    if monkeypatch is not None:
        def _blocked(*a, **k):
            raise OSError("W3a test: network is blocked in-process")
        monkeypatch.setattr(socket.socket, "connect", _blocked)
        monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        monkeypatch.setattr(socket, "getaddrinfo", _blocked)
        for pkg in ("boto3", "botocore"):
            monkeypatch.setitem(sys.modules, pkg, None)
    return importlib.import_module(MODULE)


# ── a boto3-shaped fake S3 client (reads only) ───────────────────────────────


class FakeClientError(Exception):
    """Shaped like ``botocore.exceptions.ClientError``: ``.response`` carries
    ``Error.Code`` and ``ResponseMetadata.HTTPStatusCode``."""

    def __init__(self, op: str, code: str, status: int, request_id: str):
        super().__init__(f"An error occurred ({code}) when calling the {op} operation")
        self.operation_name = op
        self.response = {"Error": {"Code": code, "Message": code},
                         "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": request_id,
                                              "HTTPHeaders": {"x-amz-request-id": request_id}}}


class _Paginator:
    def __init__(self, fn):
        self._fn = fn

    def paginate(self, **kw):
        yield self._fn(**kw)


def _etag(data: bytes) -> str:
    return '"' + hashlib.md5(data).hexdigest() + '"'


def _dt(x) -> datetime:
    return x if isinstance(x, datetime) else datetime.fromisoformat(str(x))


class FakeS3:
    """An in-memory S3 bucket behind a boto3-shaped client: the read calls
    ``storage_probe`` may make (HEAD, GetObjectRetention, GET, version
    listing, GetObjectLockConfiguration). Every call is recorded in
    ``calls`` as ``(op, kwargs)``; any other attribute (a write such as
    ``put_object``, ``delete_object``, ``put_object_retention``) is recorded
    in ``forbidden`` and raises. Request ids are ``live-fake-<n>``.

    ``default_retention`` models the bucket's Object Lock default rule, in
    the GetObjectLockConfiguration shape (``{"Mode": "COMPLIANCE",
    "Days": 30}``); ``lock_enabled=False`` answers
    ``ObjectLockConfigurationNotFoundError`` (404); ``lock_error`` =
    ``(code, status)`` answers that error instead."""

    def __init__(self, bucket: str, *, lock_enabled: bool = True,
                 default_retention: dict | None = None, rule_present: bool | None = None,
                 lock_error: tuple[str, int] | None = None):
        self.bucket = bucket
        self.lock_enabled = lock_enabled
        self.default_retention = default_retention
        self.rule_present = rule_present if rule_present is not None else default_retention is not None
        self.lock_error = lock_error
        self.objects: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, dict]] = []
        self.forbidden: list[str] = []
        self._n = 0

    # state
    def put_version(self, key: str, version_id: str, data: bytes, *, mode: str | None = None,
                    retain_until=None, etag: str | None = None) -> None:
        self.objects.setdefault(key, []).append(
            {"vid": version_id, "data": data, "mode": mode,
             "retain_until": _dt(retain_until) if retain_until is not None else None,
             "etag": etag or _etag(data)})

    def _meta(self, status: int = 200) -> dict:
        self._n += 1
        rid = f"live-fake-{self._n}"
        return {"HTTPStatusCode": status, "RequestId": rid, "HTTPHeaders": {"x-amz-request-id": rid}}

    def _err(self, op: str, code: str, status: int):
        self._n += 1
        return FakeClientError(op, code, status, f"live-fake-{self._n}")

    def _check_bucket(self, op: str, bucket) -> None:
        if bucket != self.bucket:
            raise self._err(op, "NoSuchBucket", 404)

    def _find(self, op: str, key, version_id) -> dict:
        vs = self.objects.get(key) or []
        if version_id is None:
            if vs:
                return vs[-1]
        else:
            for v in vs:
                if v["vid"] == version_id:
                    return v
        raise self._err(op, "NoSuchKey" if op == "GetObject" else "404", 404)

    # the boto3-shaped read surface
    def get_object_lock_configuration(self, **kw):
        self.calls.append(("get_object_lock_configuration", dict(kw)))
        self._check_bucket("GetObjectLockConfiguration", kw.get("Bucket"))
        if self.lock_error:
            raise self._err("GetObjectLockConfiguration", *self.lock_error)
        if not self.lock_enabled:
            raise self._err("GetObjectLockConfiguration", "ObjectLockConfigurationNotFoundError", 404)
        cfg: dict = {"ObjectLockEnabled": "Enabled"}
        if self.rule_present:
            cfg["Rule"] = ({"DefaultRetention": dict(self.default_retention)}
                           if self.default_retention is not None else {})
        return {"ObjectLockConfiguration": cfg, "ResponseMetadata": self._meta()}

    def head_object(self, **kw):
        self.calls.append(("head_object", dict(kw)))
        self._check_bucket("HeadObject", kw.get("Bucket"))
        v = self._find("HeadObject", kw.get("Key"), kw.get("VersionId"))
        out = {"ContentLength": len(v["data"]), "ETag": v["etag"], "VersionId": v["vid"],
               "ResponseMetadata": self._meta()}
        if v["mode"]:
            out["ObjectLockMode"] = v["mode"]
            out["ObjectLockRetainUntilDate"] = v["retain_until"]
        return out

    def get_object_retention(self, **kw):
        self.calls.append(("get_object_retention", dict(kw)))
        self._check_bucket("GetObjectRetention", kw.get("Bucket"))
        v = self._find("GetObjectRetention", kw.get("Key"), kw.get("VersionId"))
        if not v["mode"]:
            raise self._err("GetObjectRetention", "NoSuchObjectLockConfiguration", 404)
        return {"Retention": {"Mode": v["mode"], "RetainUntilDate": v["retain_until"]},
                "ResponseMetadata": self._meta()}

    def get_object(self, **kw):
        self.calls.append(("get_object", dict(kw)))
        self._check_bucket("GetObject", kw.get("Bucket"))
        v = self._find("GetObject", kw.get("Key"), kw.get("VersionId"))
        return {"Body": io.BytesIO(v["data"]), "ContentLength": len(v["data"]),
                "ETag": v["etag"], "VersionId": v["vid"], "ResponseMetadata": self._meta()}

    def list_object_versions(self, **kw):
        self.calls.append(("list_object_versions", dict(kw)))
        self._check_bucket("ListObjectVersions", kw.get("Bucket"))
        pre = kw.get("Prefix", "")
        vers = [{"Key": k, "VersionId": v["vid"], "IsLatest": i == len(vs) - 1,
                 "ETag": v["etag"], "Size": len(v["data"])}
                for k, vs in sorted(self.objects.items()) if k.startswith(pre)
                for i, v in enumerate(vs)]
        return {"Versions": vers, "DeleteMarkers": [], "IsTruncated": False,
                "ResponseMetadata": self._meta()}

    def list_objects_v2(self, **kw):
        self.calls.append(("list_objects_v2", dict(kw)))
        self._check_bucket("ListObjectsV2", kw.get("Bucket"))
        pre = kw.get("Prefix", "")
        keys = [k for k in sorted(self.objects) if k.startswith(pre)]
        return {"Contents": [{"Key": k, "Size": len(self.objects[k][-1]["data"])} for k in keys],
                "KeyCount": len(keys), "IsTruncated": False, "ResponseMetadata": self._meta()}

    def get_paginator(self, name: str):
        self.calls.append(("get_paginator", {"name": name}))
        if name not in ("list_object_versions", "list_objects_v2"):
            self.forbidden.append(f"get_paginator({name})")
            raise AssertionError(f"FakeS3: no paginator {name!r}")
        return _Paginator(getattr(self, name))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        self.forbidden.append(name)
        raise AssertionError(f"FakeS3: {name} is not a read; W3a verify/preflight are read-only")

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]


def _object_bytes(key: str, size: int) -> bytes:
    seed = hashlib.sha256(f"w3a-live-fake:{key}".encode()).digest()
    return (seed * (size // len(seed) + 1))[:size]


def live_pair(evidence: dict) -> tuple[dict, FakeS3]:
    """A live-shaped evidence body and a :class:`FakeS3` bucket that really
    holds its COMPLIANCE-locked objects: the offline stand-in for "the owner
    ran the probe on Hippius". Each locked object gets deterministic bytes of
    its recorded size; the evidence's ``etag``/``sha256`` (in
    ``locked_objects`` and in the lock probe's ``objects``) are set to what
    the bucket holds, and the bucket's version id, mode and retain-until are
    the evidence's. The run root holds nothing else (cleanup was clean)."""
    ev = copy.deepcopy(evidence)
    fake = FakeS3(ev.get("bucket"))
    for lo in ev.get("locked_objects", []):
        data = _object_bytes(lo["key"], int(lo.get("size") or 0))
        sha, etag = hashlib.sha256(data).hexdigest(), _etag(data)
        lo["sha256"], lo["etag"] = sha, etag
        for p in ev.get("probes", []):
            if p.get("name") in LOCK_PROBES:
                for o in p.get("objects", []):
                    if o.get("key") == lo["key"]:
                        o["sha256"] = sha
        fake.put_version(lo["key"], lo["version_id"], data, mode=lo.get("mode"),
                         retain_until=lo.get("retain_until"), etag=etag)
    return ev, fake


def no_lock_evidence(ev: dict) -> bool:
    """The evidence of a ``--no-object-lock`` run (README "No Object Lock")."""
    return isinstance(ev, dict) and isinstance(ev.get("bucket_lock"), dict) \
        and "asserted_by" in ev["bucket_lock"]


def anchor_pair(evidence: dict, *, version_ids: bool = True) -> tuple[dict, FakeS3]:
    """The ``--no-object-lock`` counterpart of :func:`live_pair`: a FakeS3
    bucket that holds the run's UNLOCKED anchor object(s) and nothing else
    under the run root, with the evidence's ``etag``/``sha256`` set to what the
    bucket holds. ``version_ids=False`` models a bucket without versioning
    (Hippius denies GetBucketVersioning): the anchors' ``version_id`` is null."""
    ev = copy.deepcopy(evidence)
    fake = FakeS3(ev.get("bucket"), lock_enabled=False)
    for i, a in enumerate(ev.get("anchor_objects") or []):
        data = _object_bytes(a["key"], int(a.get("size") or 0))
        a["sha256"], a["etag"] = hashlib.sha256(data).hexdigest(), _etag(data)
        if not version_ids:
            a["version_id"] = None
        fake.put_version(a["key"], a["version_id"] or f"null-{i}", data, etag=a["etag"])
    return ev, fake


def canonical_body(evidence: dict) -> bytes:
    """The bytes ``verified_live.body_sha256`` hashes: the evidence object
    WITHOUT its ``verified_live`` key, as ``json.dumps(sort_keys=True,
    separators=(",", ":"), ensure_ascii=True)`` encoded UTF-8."""
    body = {k: v for k, v in evidence.items() if k != "verified_live"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def body_sha256(evidence: dict) -> str:
    return hashlib.sha256(canonical_body(evidence)).hexdigest()


# ── chain.toml buckets ───────────────────────────────────────────────────────


def production_buckets() -> set[str]:
    """Every non-empty string value of a key ending in ``bucket``, any table."""
    out: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and k.endswith("bucket") and v:
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for name in ("chain.toml", "chain.testnet.toml"):
        p = REPO / name
        if p.is_file():
            walk(tomllib.loads(p.read_text()))
    return out


# ── the evidence verifier ────────────────────────────────────────────────────


class _Bad(Exception):
    pass


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _status(x, where: str):
    if x is None:
        return None
    if not (_is_int(x) and 100 <= x <= 599):
        raise _Bad(f"{where}: want an HTTP status int or null, got {x!r}")
    return x


def _ts(x, where: str) -> datetime:
    try:
        t = datetime.fromisoformat(str(x))
    except ValueError:
        raise _Bad(f"{where}: not ISO 8601: {x!r}") from None
    if t.tzinfo is None:
        raise _Bad(f"{where}: must carry a UTC offset")
    return t


def _same_instant(a, b, where: str) -> bool:
    if a is None or b is None:
        return False
    return abs((_ts(a, where) - _ts(b, where)).total_seconds()) <= 1


def _hex_or_none(x, where: str):
    if x is not None and not (isinstance(x, str) and HEX64_RE.fullmatch(x)):
        raise _Bad(f"{where}: want 64 lowercase hex or null, got {x!r}")
    return x


def _obs(o: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in o]
    if missing:
        raise _Bad(f"observations missing keys {missing}")


def _d_presigned(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "create_status", "url_expires_seconds", "authorization_header_sent", "parts",
         "complete_status", "uploaded_sha256", "object_size", "object_sha256")
    if o["authorization_header_sent"] is not False:
        raise _Bad("authorization_header_sent must be false (a presigned PUT carries no credential)")
    exp = o["url_expires_seconds"]
    if not (_is_int(exp) and 1 <= exp <= MAX_PRESIGN_S):
        raise _Bad(f"url_expires_seconds: want 1..{MAX_PRESIGN_S}, got {exp!r}")
    if _status(o["create_status"], "create_status") != 200:
        return "error"
    parts = o["parts"]
    if not isinstance(parts, list):
        raise _Bad("parts must be a list")
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            raise _Bad(f"parts[{i}] must be an object")
        _obs(p, "part_number", "size", "http_status", "etag_returned")
        _status(p["http_status"], f"parts[{i}].http_status")
    _hex_or_none(o["uploaded_sha256"], "uploaded_sha256")
    _hex_or_none(o["object_sha256"], "object_sha256")
    ok = (len(parts) >= 2
          and [p["part_number"] for p in parts] == list(range(1, len(parts) + 1))
          and all(p["http_status"] == 200 and p["etag_returned"] is True for p in parts)
          and _status(o["complete_status"], "complete_status") == 200
          and o["object_size"] == sum(p["size"] for p in parts)
          and o["object_sha256"] is not None
          and o["object_sha256"] == o["uploaded_sha256"])
    return "pass" if ok else "fail"


def _d_single_put(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "required_bytes", "authorization_header_sent", "attempts", "max_accepted_bytes",
         "min_rejected_bytes")
    if o["authorization_header_sent"] is not False:
        raise _Bad("authorization_header_sent must be false (a presigned PUT carries no credential)")
    if o["required_bytes"] != params.get("big_object_bytes"):
        raise _Bad(f"required_bytes {o['required_bytes']!r} != params.big_object_bytes")
    atts = o["attempts"]
    if not isinstance(atts, list) or not atts:
        raise _Bad("attempts must be a non-empty list")
    sizes = []
    accepted, rejected = [], []
    for i, a in enumerate(atts):
        if not isinstance(a, dict):
            raise _Bad(f"attempts[{i}] must be an object")
        _obs(a, "size", "http_status", "error_code", "object_size", "sha256")
        _status(a["http_status"], f"attempts[{i}].http_status")
        _hex_or_none(a["sha256"], f"attempts[{i}].sha256")
        if rejected:
            raise _Bad("attempts continue after the first rejection")
        sizes.append(a["size"])
        if a["http_status"] == 200 and a["object_size"] == a["size"]:
            if a["sha256"] is None:
                raise _Bad(f"attempts[{i}]: an accepted PUT records its sha256")
            accepted.append(a["size"])
        else:
            rejected.append(a["size"])
    ladder = params.get("single_put_bytes")
    if not isinstance(ladder, list) or sizes != ladder[:len(sizes)]:
        raise _Bad(f"attempt sizes {sizes} are not a prefix of params.single_put_bytes {ladder}")
    want_max = max(accepted) if accepted else None
    want_min = min(rejected) if rejected else None
    if o["max_accepted_bytes"] != want_max or o["min_rejected_bytes"] != want_min:
        raise _Bad(f"max_accepted/min_rejected: attempts give {want_max}/{want_min}, recorded "
                   f"{o['max_accepted_bytes']}/{o['min_rejected_bytes']}")
    return "pass" if want_max is not None and want_max >= o["required_bytes"] else "fail"


def _d_lock_mpu(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "bucket_lock_enabled", "requested_mode", "requested_retain_until", "create_status",
         "complete_status", "head_mode", "head_retain_until", "object_size")
    if o["bucket_lock_enabled"] is not True:
        return "error"
    if o["requested_mode"] != "COMPLIANCE":
        raise _Bad(f"requested_mode: want COMPLIANCE, got {o['requested_mode']!r}")
    _ts(o["requested_retain_until"], "requested_retain_until")
    size = o["object_size"]
    if size is not None and not (_is_int(size) and 0 <= size <= MAX_LOCKED_BYTES):
        raise _Bad(f"object_size: the lock probe object must be <= {MAX_LOCKED_BYTES} bytes, got {size!r}")
    ok = (_status(o["create_status"], "create_status") == 200
          and _status(o["complete_status"], "complete_status") == 200
          and o["head_mode"] == "COMPLIANCE"
          and _same_instant(o["head_retain_until"], o["requested_retain_until"], "head_retain_until"))
    return "pass" if ok else "fail"


def _d_policy(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "preexisting_policy", "policy", "credential", "put_policy_status",
         "denied_put_status", "allowed_put_status", "policy_removed")
    if o["preexisting_policy"] is True:
        return "error"
    if o["preexisting_policy"] is not False:
        raise _Bad("preexisting_policy must be a bool")
    if o["policy_removed"] is not True:
        raise _Bad("policy_removed: the probe policy must be deleted from the bucket")
    if o["credential"] not in ("subtoken", "main"):
        raise _Bad(f"credential: want subtoken|main, got {o['credential']!r}")
    pol = o["policy"]
    want_res = f"arn:aws:s3:::{ctx['bucket']}/{ctx['prefix']}{ctx['run_id']}/"
    stmts = pol.get("Statement") if isinstance(pol, dict) else None
    if not isinstance(stmts, list):
        raise _Bad("policy: want an AWS policy document with a Statement list")

    def res_list(s):
        r = s.get("Resource")
        return [r] if isinstance(r, str) else (r if isinstance(r, list) else [])

    denies = [s for s in stmts if isinstance(s, dict) and s.get("Effect") == "Deny"]
    if not denies or not all(isinstance(r, str) and r.startswith(want_res)
                             for s in denies for r in res_list(s)) \
            or not any(res_list(s) for s in denies):
        raise _Bad(f"policy: want a Deny scoped under {want_res}")
    ok = (_status(o["put_policy_status"], "put_policy_status") in (200, 204)
          and _status(o["denied_put_status"], "denied_put_status") == 403
          and _status(o["allowed_put_status"], "allowed_put_status") == 200)
    return "pass" if ok else "fail"


def _d_subtoken(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "subtoken_present", "other_bucket", "other_bucket_exists", "in_bucket_put_status",
         "in_bucket_get_status", "other_bucket_list_status")
    if o["subtoken_present"] is not True:
        return "error"
    if not isinstance(o["other_bucket"], str) or not o["other_bucket"] \
            or o["other_bucket"] == ctx["bucket"]:
        raise _Bad(f"other_bucket: want a bucket other than {ctx['bucket']!r}, got {o['other_bucket']!r}")
    if o["other_bucket_exists"] is not True:
        return "error"
    ok = (_status(o["in_bucket_put_status"], "in_bucket_put_status") == 200
          and _status(o["in_bucket_get_status"], "in_bucket_get_status") == 200
          and _status(o["other_bucket_list_status"], "other_bucket_list_status") == 403)
    return "pass" if ok else "fail"


def _d_part_copy(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "source_status", "source_size", "source_sha256", "copy_object_status",
         "copy_object_error_code", "parts", "complete_status", "dest_size", "dest_sha256")
    if _status(o["source_status"], "source_status") != 200:
        return "error"
    if o["source_size"] != params.get("big_object_bytes"):
        raise _Bad(f"source_size {o['source_size']!r} != params.big_object_bytes")
    _status(o["copy_object_status"], "copy_object_status")
    _hex_or_none(o["source_sha256"], "source_sha256")
    _hex_or_none(o["dest_sha256"], "dest_sha256")
    if o["source_sha256"] is None:
        raise _Bad("source_sha256: a completed source records its sha256")
    parts = o["parts"]
    if not isinstance(parts, list):
        raise _Bad("parts must be a list")
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            raise _Bad(f"parts[{i}] must be an object")
        _obs(p, "part_number", "range_start", "range_end", "http_status")
        _status(p["http_status"], f"parts[{i}].http_status")
    cover, nxt = bool(parts), 0
    for i, p in enumerate(parts):
        if p["part_number"] != i + 1 or p["range_start"] != nxt or not _is_int(p["range_end"]) \
                or p["range_end"] < p["range_start"] \
                or p["range_end"] - p["range_start"] + 1 > params.get("part_bytes", 0):
            cover = False
            break
        nxt = p["range_end"] + 1
    cover = cover and nxt == o["source_size"]
    ok = (cover and all(p["http_status"] == 200 for p in parts)
          and _status(o["complete_status"], "complete_status") == 200
          and o["dest_size"] == o["source_size"]
          and o["dest_sha256"] == o["source_sha256"])
    return "pass" if ok else "fail"


def _d_compliance(o: dict, params: dict, ctx: dict) -> str:
    _obs(o, "bucket_lock_enabled", "put_status", "mode", "retain_until", "shorten_to",
         "shorten_status", "shorten_bypass_status", "delete_version_status",
         "retain_until_after", "object_size")
    if o["bucket_lock_enabled"] is not True or _status(o["put_status"], "put_status") != 200 \
            or o["mode"] != "COMPLIANCE":
        return "error"
    size = o["object_size"]
    if not (_is_int(size) and 0 <= size <= MAX_LOCKED_BYTES):
        raise _Bad(f"object_size: the compliance object must be <= {MAX_LOCKED_BYTES} bytes, got {size!r}")
    if not _ts(o["shorten_to"], "shorten_to") < _ts(o["retain_until"], "retain_until"):
        raise _Bad("shorten_to must be earlier than retain_until")
    refused = all((_status(o[k], k) or 0) >= 400
                  for k in ("shorten_status", "shorten_bypass_status", "delete_version_status"))
    ok = refused and _same_instant(o["retain_until_after"], o["retain_until"], "retain_until_after")
    return "pass" if ok else "fail"


DERIVE = {"presigned_upload_part": _d_presigned, "single_put_limit": _d_single_put,
          "object_lock_multipart": _d_lock_mpu, "prefix_bucket_policy": _d_policy,
          "subtoken_scope": _d_subtoken, "upload_part_copy_2_5gb": _d_part_copy,
          "compliance_retention_unshortenable": _d_compliance}


def recommended_path(ev: dict) -> str:
    res = {p.get("name"): p.get("result") for p in ev.get("probes", []) if isinstance(p, dict)}
    if res.get("presigned_upload_part") == "pass":
        return "presigned_multipart"
    if res.get("single_put_limit") == "pass":
        return "single_presigned_put"
    return "upload_relay"


def passed_rule(ev: dict) -> bool:
    rows = [p for p in ev.get("probes", []) if isinstance(p, dict)]
    no_lock = no_lock_evidence(ev)

    def recorded(p: dict) -> bool:
        if no_lock and p.get("name") in LOCK_PROBES:
            return p.get("result") == SKIP_RESULT
        return p.get("result") in ("pass", "fail")

    cleanup = ev.get("cleanup") if isinstance(ev.get("cleanup"), dict) else {}
    return (ev.get("source") == "live" and len(rows) == len(PROBES)
            and all(recorded(p) for p in rows)
            and cleanup.get("remaining_unlocked") == []
            and (not no_lock or bool(ev.get("anchor_objects"))))


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=False)


def _tool_problems(tool: object, check_git: bool) -> list[str]:
    if not isinstance(tool, dict):
        return ["tool: must be an object"]
    out = []
    if tool.get("module") != MODULE:
        out.append(f"tool.module: want {MODULE!r}, got {tool.get('module')!r}")
    sha = tool.get("git_sha")
    if not (isinstance(sha, str) and SHA_RE.fullmatch(sha)):
        out.append(f"tool.git_sha: want 40 lowercase hex, got {sha!r}")
    elif check_git:
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            out.append(f"tool.git_sha {sha} is not a commit in this repository")
        elif _git("cat-file", "-e", f"{sha}:{MODULE_PATH}").returncode != 0:
            out.append(f"tool.git_sha {sha} has no {MODULE_PATH}")
    if tool.get("git_dirty") is not False:
        out.append(f"tool.git_dirty: the run must be from a clean checkout, got {tool.get('git_dirty')!r}")
    return out


def _params_problems(p: object, live_sizes: bool) -> list[str]:
    if not isinstance(p, dict):
        return ["params: must be an object"]
    keys = ("big_object_bytes", "part_bytes", "single_put_bytes", "lock_retention_seconds",
            "presign_expires_seconds")
    missing = [k for k in keys if k not in p]
    if missing:
        return [f"params: missing keys {missing}"]
    out = []
    big, part = p["big_object_bytes"], p["part_bytes"]
    if not (_is_int(big) and _is_int(part) and big > 0 and part > 0):
        return ["params: big_object_bytes and part_bytes must be ints > 0"]
    if live_sizes:
        if big < LIVE_MIN_BIG_BYTES:
            out.append(f"params.big_object_bytes: a live run probes >= {LIVE_MIN_BIG_BYTES} "
                       f"(~2.5 GB BF16), got {big}")
        if not MIN_PART_BYTES <= part <= MAX_PART_BYTES:
            out.append(f"params.part_bytes: want {MIN_PART_BYTES}..{MAX_PART_BYTES}, got {part}")
        if math.ceil(big / part) > MAX_PARTS:
            out.append(f"params: {big} / {part} needs more than {MAX_PARTS} parts")
    ladder = p["single_put_bytes"]
    if not (isinstance(ladder, list) and ladder and all(_is_int(x) and x > 0 for x in ladder)
            and ladder == sorted(set(ladder))):
        out.append(f"params.single_put_bytes: want a strictly ascending list of ints, got {ladder!r}")
    lr = p["lock_retention_seconds"]
    if not (_is_int(lr) and MIN_LOCK_S <= lr <= MAX_LOCK_S):
        out.append(f"params.lock_retention_seconds: want {MIN_LOCK_S}..{MAX_LOCK_S}, got {lr!r}")
    pe = p["presign_expires_seconds"]
    if not (_is_int(pe) and 1 <= pe <= MAX_PRESIGN_S):
        out.append(f"params.presign_expires_seconds: want 1..{MAX_PRESIGN_S}, got {pe!r}")
    return out


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def leak_problems(data: object, extra_text: str = "") -> list[str]:
    """Credential-shaped strings, URLs, or a set HIPPIUS_* value anywhere."""
    out = []
    secrets = [v for k, v in os.environ.items() if k.startswith("HIPPIUS_") and len(v) >= 8]
    for s in list(_strings(data)) + ([extra_text] if extra_text else []):
        for rx in LEAK_RES:
            if rx.search(s):
                out.append(f"leak: a string matches {rx.pattern!r}: {s[:80]!r}")
        if any(v in s for v in secrets):
            out.append("leak: a HIPPIUS_* environment value appears in the evidence")
    return out


def _skipped_row_problems(where: str, row: dict, no_lock: bool) -> list[str]:
    """A lock probe of a ``--no-object-lock`` run: ``skipped`` with reason
    ``no_object_lock``, never ``pass``, and nothing done (no request, object
    or observation). Any other probe, or any probe of a run without the
    flag, can't be skipped."""
    if not no_lock:
        return [f"{where}.result: 'skipped' is only valid for a lock probe of a "
                f"{NO_LOCK_FLAG} run (bucket_lock.asserted_by: owner_flag)"]
    if row["name"] not in LOCK_PROBES:
        return [f"{where}.result: only the lock probes {LOCK_PROBES} may be skipped "
                f"in a {NO_LOCK_FLAG} run"]
    out = []
    if row["passed"] is not False:
        out.append(f"{where}.passed: a skipped probe never passed")
    if row["reason"] != SKIP_REASON:
        out.append(f"{where}.reason: want {SKIP_REASON!r}, got {row['reason']!r}")
    if row["requests"] != [] or row["objects"] != [] or row["observations"] != {}:
        out.append(f"{where}: a skipped lock probe makes no request, creates no object and "
                   "records no observation")
    return out


def _probe_row_problems(i: int, row: object, ctx: dict, params: dict, live: bool,
                        no_lock: bool = False) -> list[str]:
    where = f"probes[{i}]"
    if not isinstance(row, dict):
        return [f"{where}: must be an object"]
    missing = [k for k in PROBE_KEYS if k not in row]
    if missing:
        return [f"{where}: missing keys {missing}"]
    name = row["name"]
    where = f"probes[{name}]"
    out = []
    if row["result"] == SKIP_RESULT:
        return _skipped_row_problems(where, row, no_lock)
    if no_lock and name in LOCK_PROBES:
        return [f"{where}.result: a {NO_LOCK_FLAG} run never runs a lock probe; want "
                f"'skipped' ({SKIP_REASON}), got {row['result']!r}"]
    if row["result"] not in RESULTS:
        return [f"{where}.result: want one of {RESULTS}, got {row['result']!r}"]
    if row["passed"] is not (row["result"] == "pass"):
        out.append(f"{where}.passed must equal result == 'pass'")
    if not isinstance(row["reason"], str):
        out.append(f"{where}.reason must be a string")
    if row["result"] == "error":
        return out
    reqs, objs = row["requests"], row["objects"]
    if not isinstance(reqs, list) or not reqs:
        out.append(f"{where}.requests: a recorded result needs its requests")
        reqs = []
    for j, r in enumerate(reqs):
        if not (isinstance(r, dict) and isinstance(r.get("op"), str) and r.get("op")):
            out.append(f"{where}.requests[{j}]: want {{op, http_status, request_id}}")
            continue
        try:
            st = _status(r.get("http_status"), f"{where}.requests[{j}].http_status")
        except _Bad as e:
            out.append(str(e))
            continue
        rid = r.get("request_id")
        if st is not None and not (isinstance(rid, str) and rid.strip()):
            out.append(f"{where}.requests[{j}] ({r['op']}): a response with a status carries "
                       "its request id")
        if live and isinstance(rid, str) and rid.startswith("fixture-"):
            out.append(f"{where}.requests[{j}]: fixture request id in live evidence")
    if not isinstance(objs, list) or not objs:
        out.append(f"{where}.objects: a recorded result names the objects it created")
        objs = []
    run_root = f"{ctx['prefix']}{ctx['run_id']}/"
    for j, o in enumerate(objs):
        if not isinstance(o, dict) or not isinstance(o.get("key"), str):
            out.append(f"{where}.objects[{j}]: want {{key, size, sha256}}")
            continue
        if not o["key"].startswith(run_root) or ".." in o["key"].split("/"):
            out.append(f"{where}.objects[{j}].key {o['key']!r} is outside {run_root}")
        if not (_is_int(o.get("size")) and o["size"] >= 0):
            out.append(f"{where}.objects[{j}].size must be an int >= 0")
        try:
            _hex_or_none(o.get("sha256"), f"{where}.objects[{j}].sha256")
        except _Bad as e:
            out.append(str(e))
    obs = row["observations"]
    if not isinstance(obs, dict):
        return out + [f"{where}.observations must be an object"]
    try:
        want = DERIVE[name](obs, params, ctx)
    except _Bad as e:
        return out + [f"{where}.observations: {e}"]
    except (TypeError, KeyError, AttributeError) as e:
        return out + [f"{where}.observations malformed: {e!r}"]
    if want != row["result"]:
        out.append(f"{where}.result: the observations give {want!r}, recorded {row['result']!r}")
    return out


def _locked_problems(data: dict, rows: dict, started, finished, params: dict) -> list[str]:
    out = []
    locked = data["locked_objects"]
    if not isinstance(locked, list):
        return ["locked_objects must be a list"]
    lock_keys = {o.get("key") for n in LOCK_PROBES if n in rows
                 for o in (rows[n].get("objects") or []) if isinstance(o, dict)}
    retention = params.get("lock_retention_seconds") if isinstance(params, dict) else None
    keys = []
    for i, lo in enumerate(locked):
        where = f"locked_objects[{i}]"
        if not isinstance(lo, dict):
            out.append(f"{where}: must be an object")
            continue
        miss = [k for k in LOCKED_KEYS if k not in lo]
        if miss:
            out.append(f"{where}: missing keys {miss}")
            continue
        keys.append(lo["key"])
        if lo["key"] not in lock_keys:
            out.append(f"{where}.key {lo['key']!r} is not an object of the two lock probes")
        if not (isinstance(lo["version_id"], str) and lo["version_id"]):
            out.append(f"{where}.version_id: a locked object names its version (verify HEADs it)")
        if not (isinstance(lo["etag"], str) and lo["etag"].strip('"')):
            out.append(f"{where}.etag: want the HEAD ETag of the locked version")
        if not (isinstance(lo["sha256"], str) and HEX64_RE.fullmatch(lo["sha256"])):
            out.append(f"{where}.sha256: want 64 lowercase hex (verify re-reads the bytes)")
        else:
            probe_shas = {o.get("sha256") for n in LOCK_PROBES if n in rows
                          for o in (rows[n].get("objects") or [])
                          if isinstance(o, dict) and o.get("key") == lo["key"]}
            if probe_shas and lo["sha256"] not in probe_shas:
                out.append(f"{where}.sha256 differs from the lock probe's object sha256")
        if lo["mode"] != "COMPLIANCE":
            out.append(f"{where}.mode: want COMPLIANCE, got {lo['mode']!r}")
        if not (_is_int(lo["size"]) and 0 <= lo["size"] <= MAX_LOCKED_BYTES):
            out.append(f"{where}.size: locked objects must be <= {MAX_LOCKED_BYTES} bytes")
        try:
            ru = _ts(lo["retain_until"], f"{where}.retain_until")
        except _Bad as e:
            out.append(str(e))
            continue
        if started is not None and ru <= started:
            out.append(f"{where}.retain_until is not after started_at")
        if finished is not None and _is_int(retention) and \
                ru > finished + timedelta(seconds=retention + 60):
            out.append(f"{where}.retain_until is later than finished_at + "
                       f"lock_retention_seconds: the run must use the shortest retention")
    for n in LOCK_PROBES:
        row = rows.get(n)
        if not row or row.get("result") == "error" or not isinstance(row.get("observations"), dict):
            continue
        obs = row["observations"]
        mode = obs.get("head_mode") if n == "object_lock_multipart" else obs.get("mode")
        if mode == "COMPLIANCE":
            mine = {o.get("key") for o in row.get("objects") or [] if isinstance(o, dict)}
            if not mine & set(keys):
                out.append(f"locked_objects: the COMPLIANCE object of {n} is not reported")
    cleanup = data["cleanup"]
    if not isinstance(cleanup, dict):
        return out + ["cleanup must be an object"]
    miss = [k for k in ("deleted_versions", "aborted_uploads", "remaining_unlocked",
                        "remaining_locked") if k not in cleanup]
    if miss:
        return out + [f"cleanup: missing keys {miss}"]
    for k in ("deleted_versions", "aborted_uploads"):
        if not (_is_int(cleanup[k]) and cleanup[k] >= 0):
            out.append(f"cleanup.{k} must be an int >= 0")
    if not isinstance(cleanup["remaining_unlocked"], list):
        out.append("cleanup.remaining_unlocked must be a list")
    rl = cleanup["remaining_locked"]
    if not isinstance(rl, list) or not set(rl) <= set(keys):
        out.append("cleanup.remaining_locked must list only reported locked objects")
    return out


def _bucket_lock_problems(data: dict, rows: dict, live: bool) -> list[str]:
    """The pre-write GetObjectLockConfiguration read (README "bucket_lock")."""
    bl = data["bucket_lock"]
    if not isinstance(bl, dict):
        return ["bucket_lock must be an object"]
    if "asserted_by" in bl:
        # --no-object-lock: the owner's assertion, not a read (none was made).
        if bl != OWNER_FLAG_LOCK:
            return [f"bucket_lock: a {NO_LOCK_FLAG} run records exactly {OWNER_FLAG_LOCK}, "
                    f"got {bl!r}"]
        return []
    miss = [k for k in BUCKET_LOCK_KEYS if k not in bl]
    if miss:
        return [f"bucket_lock: missing keys {miss}"]
    out = []
    if bl["default_retention"] is not None:
        out.append(f"bucket_lock.default_retention: the bucket has a default retention rule "
                   f"({bl['default_retention']!r}); the probe must refuse it before writing "
                   f"({DEFAULT_RETENTION_CODE})")
    if not isinstance(bl["lock_enabled"], bool):
        out.append("bucket_lock.lock_enabled must be a bool")
    try:
        st = _status(bl["http_status"], "bucket_lock.http_status")
    except _Bad as e:
        return out + [str(e)]
    if st is None:
        out.append("bucket_lock.http_status: the pre-write lock read was not made")
    rid = bl["request_id"]
    if not (isinstance(rid, str) and rid.strip()):
        out.append("bucket_lock.request_id: the lock read carries its request id")
    elif live and rid.startswith("fixture-"):
        out.append("bucket_lock.request_id: fixture request id in live evidence")
    for n in LOCK_PROBES:
        obs = rows.get(n, {}).get("observations")
        if isinstance(obs, dict) and "bucket_lock_enabled" in obs and \
                obs["bucket_lock_enabled"] is not bl["lock_enabled"]:
            out.append(f"bucket_lock.lock_enabled disagrees with {n}.bucket_lock_enabled")
    return out


def _rid_ok(rid: object) -> bool:
    return isinstance(rid, str) and bool(rid.strip()) and not rid.startswith("fixture-")


def _verified_anchor_problems(data: dict, rec: dict) -> list[str]:
    """The ``verified_live`` record of a ``--no-object-lock`` run: the bucket
    re-read the run's UNLOCKED anchor object(s) instead of locked ones
    (binding ``unlocked_anchor``; there is no lock, so no expiry)."""
    out = []
    if rec.get("binding") != ANCHOR_BINDING:
        out.append(f"verified_live.binding: a {NO_LOCK_FLAG} run is verified by its anchor "
                   f"object(s): want {ANCHOR_BINDING!r}, got {rec.get('binding')!r}")
    anchors = {a.get("key"): a for a in data.get("anchor_objects") or [] if isinstance(a, dict)}
    if not anchors:
        out.append("verified_live: the evidence has no anchor objects to verify against the bucket")
    try:
        at = _ts(rec["verified_at"], "verified_live.verified_at")
        if at < _ts(data.get("finished_at"), "finished_at"):
            out.append("verified_live.verified_at is before the run finished")
    except _Bad as e:
        out.append(str(e))
    objs = rec["objects"]
    if not isinstance(objs, list):
        return out + ["verified_live.objects must be a list"]
    seen = []
    for i, o in enumerate(objs):
        where = f"verified_live.objects[{i}]"
        if not isinstance(o, dict) or [k for k in VERIFY_OBJECT_KEYS if k not in o]:
            out.append(f"{where}: want the keys {VERIFY_OBJECT_KEYS}")
            continue
        seen.append(o["key"])
        a = anchors.get(o["key"])
        if a is None:
            out.append(f"{where}.key {o['key']!r} is not an anchor object of the evidence")
            continue
        for k in ("version_id", "etag", "sha256", "size"):
            if o[k] != a.get(k):
                out.append(f"{where}.{k}: the bucket gave {o[k]!r}, the evidence records "
                           f"{a.get(k)!r}")
        if o["mode"] is not None or o["retain_until"] is not None:
            out.append(f"{where}: an anchor object is unlocked (mode and retain_until null)")
        rids = o["request_ids"]
        if not (isinstance(rids, list) and rids and all(_rid_ok(r) for r in rids)):
            out.append(f"{where}.request_ids: want the live HEAD/GET request ids (non-empty, "
                       f"never fixture-), got {rids!r}")
    if sorted(seen) != sorted(k for k in anchors if isinstance(k, str)):
        out.append(f"verified_live.objects {sorted(seen)} must be exactly the anchor objects "
                   f"{sorted(k for k in anchors if isinstance(k, str))}")
    lst = rec["listing"]
    run_root = f"{data.get('prefix')}{data.get('run_id')}/"
    if not (isinstance(lst, dict) and all(k in lst for k in ("prefix", "keys", "request_id"))):
        return out + ["verified_live.listing: want {prefix, keys, request_id}"]
    if lst["prefix"] != run_root:
        out.append(f"verified_live.listing.prefix: want the run root {run_root!r}, got "
                   f"{lst['prefix']!r}")
    if not isinstance(lst["keys"], list) or set(lst["keys"]) != set(anchors):
        out.append(f"verified_live.listing.keys: the run root must hold exactly the anchor "
                   f"objects, got {lst['keys']!r}")
    if not _rid_ok(lst["request_id"]):
        out.append("verified_live.listing.request_id: want the live listing's request id")
    return out


def _anchor_problems(data: dict, rows: dict) -> list[str]:
    """``anchor_objects`` of a ``--no-object-lock`` run: at least one small
    unlocked object under the run root, written after cleanup, that no probe
    created, with its version id (null without versioning), HEAD ETag and
    sha256. Without the flag the key must be absent."""
    if not no_lock_evidence(data):
        return (["anchor_objects: only a --no-object-lock run leaves an anchor object"]
                if "anchor_objects" in data else [])
    out = []
    if data["locked_objects"] != []:
        out.append(f"locked_objects: a {NO_LOCK_FLAG} run locks nothing, got "
                   f"{data['locked_objects']!r}")
    cleanup = data["cleanup"] if isinstance(data["cleanup"], dict) else {}
    if cleanup.get("remaining_locked") not in (None, []):
        out.append(f"cleanup.remaining_locked: a {NO_LOCK_FLAG} run locks nothing")
    anchors = data.get("anchor_objects")
    if not isinstance(anchors, list) or not anchors:
        return out + [f"anchor_objects: a {NO_LOCK_FLAG} run leaves one unlocked anchor object "
                      "for `storage_probe verify` to re-read (none recorded)"]
    run_root = f"{data['prefix']}{data['run_id']}/"
    probe_keys = {o.get("key") for r in rows.values() for o in (r.get("objects") or [])
                  if isinstance(o, dict)}
    for i, a in enumerate(anchors):
        where = f"anchor_objects[{i}]"
        if not isinstance(a, dict) or [k for k in ANCHOR_KEYS if k not in a]:
            out.append(f"{where}: want the keys {ANCHOR_KEYS}")
            continue
        key = a["key"]
        if not (isinstance(key, str) and key.startswith(run_root)) or ".." in str(key).split("/"):
            out.append(f"{where}.key {key!r} is outside {run_root}")
        elif key in probe_keys:
            out.append(f"{where}.key {key!r} is a probe object, not a dedicated anchor")
        if a["version_id"] is not None and not (isinstance(a["version_id"], str) and a["version_id"]):
            out.append(f"{where}.version_id: want a version id string or null")
        if not (isinstance(a["etag"], str) and a["etag"].strip('"')):
            out.append(f"{where}.etag: want the HEAD ETag of the anchor")
        if not (isinstance(a["sha256"], str) and HEX64_RE.fullmatch(a["sha256"])):
            out.append(f"{where}.sha256: want 64 lowercase hex (verify re-reads the bytes)")
        if not (_is_int(a["size"]) and 0 < a["size"] <= MAX_ANCHOR_BYTES):
            out.append(f"{where}.size: an anchor object is 1..{MAX_ANCHOR_BYTES} bytes")
    return out


def verified_live_problems(data: dict, *, check_git: bool = True) -> list[str]:
    """The ``verified_live`` record (README "storage_probe verify"): written
    by ``python -P -m cascade.laned.storage_probe verify --evidence FILE``
    after it re-read every COMPLIANCE-locked object from the live bucket. It
    is hash-bound to the evidence body (``body_sha256``), so any edit to the
    evidence after verification fails, and every re-read value must equal
    what the evidence recorded."""
    rec = data.get("verified_live")
    if rec is None:
        return ["verified_live: missing: the evidence was never re-checked against the live "
                "bucket (run `python -P -m cascade.laned.storage_probe verify --evidence "
                "evidence/w3a_probe.json` with the HIPPIUS_* environment set, before the locks "
                "expire); fixture evidence relabelled as live never gets one"]
    if not isinstance(rec, dict):
        return ["verified_live must be an object"]
    miss = [k for k in VERIFY_KEYS if k not in rec]
    if miss:
        return [f"verified_live: missing keys {miss}"]
    out = []
    if rec["schema"] != VERIFY_SCHEMA:
        out.append(f"verified_live.schema: want {VERIFY_SCHEMA!r}, got {rec['schema']!r}")
    if rec["verified"] is not True:
        out.append(f"verified_live.verified: want true, got {rec['verified']!r}")
    want_sha = body_sha256(data)
    if rec["body_sha256"] != want_sha:
        out.append(f"verified_live.body_sha256 {rec['body_sha256']!r} is not the sha256 of the "
                   f"evidence body {want_sha}: the evidence changed after verification")
    for k in ("endpoint_host", "bucket", "run_id"):
        if rec[k] != data.get(k):
            out.append(f"verified_live.{k} {rec[k]!r} != evidence {k} {data.get(k)!r}")
    out += [f"verified_live.{p}" for p in _tool_problems(rec["tool"], check_git)]
    if no_lock_evidence(data):
        return out + _verified_anchor_problems(data, rec)
    locked = {lo.get("key"): lo for lo in data.get("locked_objects") or [] if isinstance(lo, dict)}
    if not locked:
        out.append("verified_live: the evidence has no locked objects to verify against the bucket")
    try:
        at = _ts(rec["verified_at"], "verified_live.verified_at")
    except _Bad as e:
        return out + [str(e)]
    try:
        if at < _ts(data.get("finished_at"), "finished_at"):
            out.append("verified_live.verified_at is before the run finished")
    except _Bad as e:
        out.append(str(e))
    objs = rec["objects"]
    if not isinstance(objs, list):
        return out + ["verified_live.objects must be a list"]
    seen = []
    for i, o in enumerate(objs):
        where = f"verified_live.objects[{i}]"
        if not isinstance(o, dict):
            out.append(f"{where}: must be an object")
            continue
        miss = [k for k in VERIFY_OBJECT_KEYS if k not in o]
        if miss:
            out.append(f"{where}: missing keys {miss}")
            continue
        seen.append(o["key"])
        lo = locked.get(o["key"])
        if lo is None:
            out.append(f"{where}.key {o['key']!r} is not a locked object of the evidence")
            continue
        for k in ("version_id", "etag", "sha256", "size"):
            if o[k] != lo.get(k):
                out.append(f"{where}.{k}: the bucket gave {o[k]!r}, the evidence records "
                           f"{lo.get(k)!r}")
        if o["mode"] != "COMPLIANCE" or o["mode"] != lo.get("mode"):
            out.append(f"{where}.mode: want COMPLIANCE as recorded, got {o['mode']!r}")
        try:
            if not _same_instant(o["retain_until"], lo.get("retain_until"), f"{where}.retain_until"):
                out.append(f"{where}.retain_until: the bucket gave {o['retain_until']!r}, the "
                           f"evidence records {lo.get('retain_until')!r}")
            elif not at < _ts(o["retain_until"], f"{where}.retain_until"):
                out.append(f"{where}: verified after the lock expired (verified_at >= "
                           "retain_until): the bucket no longer vouches for the object")
        except _Bad as e:
            out.append(str(e))
        rids = o["request_ids"]
        if not (isinstance(rids, list) and rids and all(_rid_ok(r) for r in rids)):
            out.append(f"{where}.request_ids: want the live HEAD/GetObjectRetention/GET "
                       f"request ids (non-empty, never fixture-), got {rids!r}")
    if sorted(seen) != sorted(locked):
        out.append(f"verified_live.objects {sorted(seen)} must be exactly the locked objects "
                   f"{sorted(k for k in locked if isinstance(k, str))}")
    lst = rec["listing"]
    run_root = f"{data.get('prefix')}{data.get('run_id')}/"
    if not (isinstance(lst, dict) and all(k in lst for k in ("prefix", "keys", "request_id"))):
        return out + ["verified_live.listing: want {prefix, keys, request_id}"]
    if lst["prefix"] != run_root:
        out.append(f"verified_live.listing.prefix: want the run root {run_root!r}, got "
                   f"{lst['prefix']!r}")
    if not isinstance(lst["keys"], list) or set(lst["keys"]) != set(locked):
        out.append(f"verified_live.listing.keys: the run root must hold exactly the locked "
                   f"objects, got {lst['keys']!r}")
    if not _rid_ok(lst["request_id"]):
        out.append("verified_live.listing.request_id: want the live listing's request id")
    return out


def evidence_problems(data: object, *, check_git: bool = True,
                      live_sizes: bool = True, require_verified: bool = True) -> list[str]:
    """Every reason ``data`` is not acceptable W3a evidence (``[]`` = pass).

    ``require_verified=False`` checks the evidence BODY only (the offline
    tampering tests); acceptance (L4) always uses the default, which also
    requires the hash-bound ``verified_live`` record."""
    if not isinstance(data, dict):
        return ["evidence: top level must be a JSON object"]
    missing = [k for k in TOP_KEYS if k not in data]
    if missing:
        return [f"evidence: missing keys {missing}"]
    out = []
    if data["schema"] != SCHEMA:
        out.append(f"schema: want {SCHEMA!r}, got {data['schema']!r}")
    if data["wp"] != WP:
        out.append(f"wp: want {WP!r}, got {data['wp']!r}")
    if data["source"] != "live":
        out.append(f"source: want 'live' (a live Hippius run), got {data['source']!r}")
    host = data["endpoint_host"]
    if not (isinstance(host, str) and HOST_RE.fullmatch(host)):
        out.append(f"endpoint_host: want a bare lowercase host name, got {host!r}")
    if not isinstance(data["region"], str):
        out.append("region must be a string")
    bucket, prefix, run_id = data["bucket"], data["prefix"], data["run_id"]
    if not (isinstance(bucket, str) and bucket):
        out.append("bucket must be a non-empty string")
    elif bucket in production_buckets():
        out.append(f"bucket {bucket!r} is a chain.toml bucket, not a throwaway")
    if not (isinstance(prefix, str) and PREFIX_RE.fullmatch(prefix)) or \
            str(prefix).startswith("lane-d/"):
        out.append(f"prefix: want a throwaway prefix (not lane-d/), got {prefix!r}")
    if not (isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id)):
        out.append(f"run_id: want 16 lowercase hex, got {run_id!r}")
    started = finished = None
    try:
        started = _ts(data["started_at"], "started_at")
        finished = _ts(data["finished_at"], "finished_at")
        if finished < started:
            out.append("finished_at is before started_at")
    except _Bad as e:
        out.append(str(e))
    out += _tool_problems(data["tool"], check_git)
    params = data["params"]
    out += _params_problems(params, live_sizes)
    out += leak_problems(data)
    if out:
        return out
    ctx = {"bucket": bucket, "prefix": prefix, "run_id": run_id}
    rows_list = data["probes"]
    if not isinstance(rows_list, list):
        return ["probes must be a list"]
    names = [r.get("name") if isinstance(r, dict) else None for r in rows_list]
    for n in PROBES:
        if names.count(n) != 1:
            out.append(f"probes: want exactly one {n!r} result, got {names.count(n)}")
    extra = [n for n in names if n not in PROBES]
    if extra:
        out.append(f"probes: unknown probe names {extra}")
    if out:
        return out
    no_lock = no_lock_evidence(data)
    for i, row in enumerate(rows_list):
        out += _probe_row_problems(i, row, ctx, params, live=data["source"] == "live",
                                   no_lock=no_lock)
    rows = {r["name"]: r for r in rows_list}
    out += _bucket_lock_problems(data, rows, live=data["source"] == "live")
    out += _locked_problems(data, rows, started, finished, params)
    out += _anchor_problems(data, rows)
    want_path = recommended_path(data)
    if data["recommended_path"] != want_path:
        out.append(f"recommended_path: the results give {want_path!r}, recorded "
                   f"{data['recommended_path']!r}")
    want_pass = passed_rule(data)
    if data["passed"] is not want_pass:
        out.append(f"passed: the rule gives {want_pass}, recorded {data['passed']!r}")
    if require_verified:
        out += verified_live_problems(data, check_git=check_git)
    return out


# ── the decision record ──────────────────────────────────────────────────────

CHOSEN_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?\**Chosen path:\**(.*)$", re.M)
SHA_LINE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?\**Evidence sha256:\**(.*)$", re.M)


def _clean(v: str) -> str:
    return v.strip().strip("`*").strip()


def decision_template_problems(text: str) -> list[str]:
    out = []
    chosen = CHOSEN_RE.findall(text)
    if len(chosen) != 1:
        out.append(f"decision record: want exactly one 'Chosen path:' line, got {len(chosen)}")
    shas = SHA_LINE_RE.findall(text)
    if len(shas) != 1:
        out.append(f"decision record: want exactly one 'Evidence sha256:' line, got {len(shas)}")
    for n in PROBES + PATH_IDS:
        if n not in text:
            out.append(f"decision record: does not name {n!r}")
    if EVIDENCE_REL not in text:
        out.append(f"decision record: does not point at {EVIDENCE_REL}")
    return out


def supported_paths(ev: dict) -> set[str]:
    res = {p.get("name"): p.get("result") for p in ev.get("probes", []) if isinstance(p, dict)}
    out = {"upload_relay"}
    if res.get("presigned_upload_part") == "pass":
        out.add("presigned_multipart")
    if res.get("single_put_limit") == "pass":
        out.add("single_presigned_put")
    return out


def decision_problems(text: str, evidence: dict, evidence_bytes: bytes) -> list[str]:
    out = decision_template_problems(text)
    if out:
        return out
    chosen = _clean(CHOSEN_RE.findall(text)[0])
    if chosen not in PATH_IDS:
        named = [p for p in PATH_IDS if p in chosen]
        return [f"decision record: 'Chosen path:' must be exactly one of {PATH_IDS}, got "
                f"{chosen!r} (names {len(named)} path ids): the owner fills it after the live run"]
    if chosen not in supported_paths(evidence):
        out.append(f"decision record: chosen path {chosen!r} is not supported by the evidence "
                   f"(supported: {sorted(supported_paths(evidence))})")
    sha = _clean(SHA_LINE_RE.findall(text)[0])
    want = hashlib.sha256(evidence_bytes).hexdigest()
    if sha != want:
        out.append(f"decision record: 'Evidence sha256:' {sha!r} is not the evidence file's "
                   f"sha256 {want}")
    if lock_probes_skipped(evidence):
        out += no_object_lock_problems(text)
    return out


# The "No Object Lock" consequence (README "No Object Lock"): required when the
# evidence's lock probes are `skipped: no_object_lock`.
NO_LOCK_LINE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?\**No Object Lock:\**(.*)$", re.M)
NO_LOCK_NEEDS = (
    (re.compile(r"\bkings?\b", re.I), "kings"),
    (re.compile(r"\b(?:not|never)\s+(?:be\s+)?(?:compliance[- ])?locked\b", re.I),
     "that kings are NOT locked (by copy at crowning)"),
    (re.compile(r"content[- ]address", re.I), "the content-addressed path"),
    (re.compile(r"\bre-?hash", re.I), "the ingest re-hash"),
    (re.compile(r"\bmirror", re.I), "the R2/HF mirrors"),
)
LOCK_CLAIM_RES = (
    re.compile(r"\b(?:are|is|get|gets|will\s+be|stay|stays|remain|remains)\s+"
               r"(?:compliance[- ])?locked\b", re.I),
    re.compile(r"(?<!not )(?<!never )\blocked\s+(?:by\s+copy|at\s+crowning)\b", re.I),
)


def lock_probes_skipped(evidence: dict) -> bool:
    """Evidence of a ``--no-object-lock`` run: a lock probe ``skipped``."""
    rows = evidence.get("probes", []) if isinstance(evidence, dict) else []
    return any(isinstance(p, dict) and p.get("name") in LOCK_PROBES
               and p.get("result") == SKIP_RESULT for p in rows)


def no_object_lock_problems(text: str) -> list[str]:
    """With skipped lock probes the record carries exactly one ``No Object
    Lock:`` line (kings are not locked by copy; immutability comes from the
    content-addressed path + ingest re-hash + the R2/HF mirrors), and no other
    line claims kings are locked."""
    out = []
    lines = NO_LOCK_LINE_RE.findall(text)
    if len(lines) != 1:
        out.append(f"decision record: the evidence skipped the lock probes ({SKIP_REASON}), so it "
                   f"needs exactly one 'No Object Lock:' consequence line, got {len(lines)}")
    else:
        val = _clean(lines[0])
        missing = [what for rx, what in NO_LOCK_NEEDS if not rx.search(val)]
        if missing:
            out.append(f"decision record: the 'No Object Lock:' line must state {missing}: "
                       f"{val!r}")
    for ln in text.splitlines():
        if NO_LOCK_LINE_RE.match(ln):
            continue
        plain = ln.replace("*", "").replace("`", "")
        if re.search(r"\bkings?\b", plain, re.I) and any(rx.search(plain) for rx in LOCK_CLAIM_RES):
            out.append(f"decision record: claims kings are locked, but the evidence skipped the "
                       f"lock probes ({SKIP_REASON}): {ln.strip()[:120]!r}")
    return out
