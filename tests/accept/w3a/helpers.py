"""Shared helpers for the W3a acceptance tests (live Hippius probe, owner WP).

Three parts:

* the probe ``python -P -m cascade.laned.storage_probe`` is driven as a
  subprocess in ``--dry-run`` mode ONLY, with ``boto3``/``botocore``/
  ``bittensor`` made unimportable and every socket connect refused, so no
  test can reach Hippius;
* :func:`evidence_problems` is the evidence verifier. It never trusts the
  probe: it re-derives every probe verdict, the recommended path and the
  overall ``passed`` from the recorded observations, so an edited verdict
  fails;
* :func:`decision_problems` checks ``decisions/STORAGE-LANE-D.md`` against
  the evidence. Contract: README.md.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
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
            "started_at", "finished_at", "tool", "params", "probes", "locked_objects",
            "cleanup", "recommended_path", "passed")
PROBE_KEYS = ("name", "result", "passed", "reason", "requests", "objects", "observations")
CREDENTIAL_ENV = ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY",
                  "HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY", "HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY")


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


def as_live_evidence(evidence: dict) -> dict:
    """A fixture-made evidence file relabelled as a live run, to exercise the
    verifier offline: source ``live``, a clean tool checkout, ``fixture-``
    request ids renamed, and ``passed`` restated from the recomputed rule."""
    ev = copy.deepcopy(evidence)
    ev["source"] = "live"
    ev["tool"]["git_dirty"] = False
    for p in ev["probes"]:
        for r in p.get("requests", []):
            rid = r.get("request_id")
            if isinstance(rid, str) and rid.startswith("fixture-"):
                r["request_id"] = "relabelled-" + rid[len("fixture-"):]
    ev["passed"] = passed_rule(ev)
    return ev


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
    res = [p.get("result") for p in ev.get("probes", []) if isinstance(p, dict)]
    cleanup = ev.get("cleanup") if isinstance(ev.get("cleanup"), dict) else {}
    return (ev.get("source") == "live" and len(res) == len(PROBES)
            and all(r in ("pass", "fail") for r in res)
            and cleanup.get("remaining_unlocked") == [])


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


def _probe_row_problems(i: int, row: object, ctx: dict, params: dict, live: bool) -> list[str]:
    where = f"probes[{i}]"
    if not isinstance(row, dict):
        return [f"{where}: must be an object"]
    missing = [k for k in PROBE_KEYS if k not in row]
    if missing:
        return [f"{where}: missing keys {missing}"]
    name = row["name"]
    where = f"probes[{name}]"
    out = []
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
        miss = [k for k in ("key", "version_id", "mode", "retain_until", "size") if k not in lo]
        if miss:
            out.append(f"{where}: missing keys {miss}")
            continue
        keys.append(lo["key"])
        if lo["key"] not in lock_keys:
            out.append(f"{where}.key {lo['key']!r} is not an object of the two lock probes")
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


def evidence_problems(data: object, *, check_git: bool = True,
                      live_sizes: bool = True) -> list[str]:
    """Every reason ``data`` is not acceptable W3a evidence (``[]`` = pass)."""
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
    for i, row in enumerate(rows_list):
        out += _probe_row_problems(i, row, ctx, params, live=data["source"] == "live")
    rows = {r["name"]: r for r in rows_list}
    out += _locked_problems(data, rows, started, finished, params)
    want_path = recommended_path(data)
    if data["recommended_path"] != want_path:
        out.append(f"recommended_path: the results give {want_path!r}, recorded "
                   f"{data['recommended_path']!r}")
    want_pass = passed_rule(data)
    if data["passed"] is not want_pass:
        out.append(f"passed: the rule gives {want_pass}, recorded {data['passed']!r}")
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
    return out
