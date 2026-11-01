"""Checkpoint transport pod → Cloudflare R2 → Hippius (``[round]
funded_harvest_transport = "r2"``).

2026-10-31: several finished legs' checkpoints (~14 GB each) were pulled by
tar-over-ssh onto the trainer box at once, filled its disk, and the teardown
that followed crashed on ENOSPC while the pods kept billing their miners. In
r2 mode the checkpoint never lands under ``work_root``:

1. **Plan** (box, operator R2 keys): one presigned ``upload_part`` link per
   512 MiB part of every checkpoint file (``create_multipart_upload``), or one
   presigned ``put_object`` link for a file of a single part. A link authorises
   one part of one object for ~2 h — a capability, never an account key; NO
   operator credential reaches the pod (miners may have root there).
2. **Upload** (pod, over the already-pinned ssh): per part
   ``tail -c +OFF FILE | head -c N | curl -X PUT -T - URL`` — the link travels
   on stdin, the ETag comes back from the response headers. A few parts at a
   time, each retried.
3. **Complete** (box): ``list_parts`` must show exactly the planned part
   numbers, sizes and ETags, else the upload is aborted; the finished object's
   size is re-checked.
4. **Verify** (box): every file is sha256-hashed by STREAMING it from R2
   (nothing written to disk) and checked against the worker receipt's
   in-memory tensor digests — the same tamper verdict the ssh harvest gives;
   the contract guard runs on a sparse skeleton (small files + safetensors
   headers, tensor bodies as holes — KB on disk, outside ``work_root``).
5. The caller tears the pod down, THEN **pushes to Hippius** from R2
   (:func:`push_from_r2`): on a configured operator relay host (presigned
   GETs, re-verified on the relay's own disk) or, without one, through a
   bounded local temp only when free disk allows — otherwise the push is
   requeued (the pending record names the R2 keys, so a retry needs no pod).

The validator-facing interface is unchanged: the manifest still carries the
Hippius ``repo@sha256:`` ref.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import struct
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..shared.hippius import (
    ALLOW_PATTERNS,
    HubConfig,
    HubRef,
    HubUpload,
    StorageError,
    backup_s3_store,
    upload_dir_to_hub_or_hf,
)

log = logging.getLogger("cascade.trainer.r2")

PART_BYTES = 512 * 1024 * 1024      # one presigned PUT per 512 MiB
UPLOAD_EXPIRES_S = 2 * 3600         # a whole checkpoint (~14 GB) fits easily
DOWNLOAD_EXPIRES_S = 2 * 3600
PARALLEL_PARTS = 4                  # concurrent part uploads per leg
PART_ATTEMPTS = 3
PART_TIMEOUT_S = 1800               # one 512 MiB part, pod uplink at >= ~0.3 MB/s
PART_RETRY_BACKOFF_S = 10.0
# Any non-tensor file larger than this is not a config/code file: the skeleton
# holds a sparse placeholder for it (the contract guard then refuses it by
# content, exactly as it would the real bytes).
SMALL_FILE_MAX = 8 * 1024 * 1024
# A checkpoint dir summing past this is not one the contract can produce (the
# 1.23B preset with its optimizer is ~20 GB): refused before a byte moves.
MAX_CHECKPOINT_BYTES = 128 * 1024 ** 3
RESULT_MARK = "CASCADE_R2_RELAY_RESULT "
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_ETAG_RE = re.compile(r"^etag:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


class R2TransferError(RuntimeError):
    """The pod → R2 move failed on the transport (infra; retried / requeued)."""


class R2StorageError(R2TransferError):
    """The OPERATOR's R2 side failed (keys, bucket, API): an operator fault."""


class CheckpointTooLarge(RuntimeError):
    """The pod's checkpoint dir is larger than any contract can produce."""


class R2PushDeferred(RuntimeError):
    """No relay and not enough local disk: the Hippius push waits (requeued)."""


# ── where the objects live ───────────────────────────────────────────────────


def ckpt_store(cfg, *, bucket: str | None = None):
    """The R2 store for checkpoints: the ``[storage]`` backup endpoint and its
    ``BACKUP_S3_*`` keys, bucket ``[round] ckpt_r2_bucket`` (else the backup
    bucket, else the manifest bucket — :func:`backup_s3_store`'s own rule)."""
    st = cfg.storage
    store = backup_s3_store(st, bucket=getattr(st, "manifest_bucket", "cascade-manifests"))
    if store is None:
        raise R2StorageError("r2 transport needs [storage] backup_s3_endpoint")
    want = bucket or str(getattr(cfg.round, "ckpt_r2_bucket", "") or "")
    if want and want != store.cfg.bucket:
        store = type(store)(replace(store.cfg, bucket=want))
    return store


def ckpt_prefix(cfg, base_seed, hotkey: str, job_id: str) -> str:
    """``<ckpt_r2_prefix><base_seed>/<hotkey>/<job_id>/`` — one leg's objects."""
    root = str(getattr(cfg.round, "ckpt_r2_prefix", "ckpt/") or "")
    return f"{root}{base_seed}/{hotkey}/{job_id}/"


def checkpoint_file_allowed(name: str) -> bool:
    """A top-level checkpoint file the Hippius push would carry (its
    ``ALLOW_PATTERNS``) with a name safe to use in a shell and an object key."""
    return bool(_SAFE_NAME.match(name)) and any(fnmatch.fnmatch(name, p) for p in ALLOW_PATTERNS)


# ── 1. plan / 3. complete (box side, operator keys) ──────────────────────────


@dataclass
class PartPlan:
    number: int
    offset: int
    size: int
    url: str = field(repr=False)        # a capability: never logged


@dataclass
class FilePlan:
    name: str
    key: str
    size: int
    upload_id: str                       # "" ⇒ one presigned put_object
    parts: list[PartPlan]


def part_layout(size: int, part_bytes: int = PART_BYTES) -> list[tuple[int, int, int]]:
    """``[(part_number, offset, size)]`` covering ``size`` bytes, 1-based."""
    size = int(size)
    if size <= part_bytes:
        return [(1, 0, size)]
    n = math.ceil(size / part_bytes)
    return [(i + 1, i * part_bytes, min(part_bytes, size - i * part_bytes)) for i in range(n)]


def plan_file_upload(store, name: str, key: str, size: int, *,
                     part_bytes: int = PART_BYTES,
                     expires_s: int = UPLOAD_EXPIRES_S) -> FilePlan:
    """Presigned upload links for one file (multipart past one part)."""
    client = store.client()
    bucket = store.cfg.bucket
    layout = part_layout(size, part_bytes)
    try:
        if len(layout) == 1:
            url = client.generate_presigned_url(
                "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=int(expires_s))
            return FilePlan(name, key, int(size), "", [PartPlan(1, 0, int(size), url)])
        upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
        parts = [PartPlan(n, off, sz, client.generate_presigned_url(
            "upload_part", Params={"Bucket": bucket, "Key": key, "UploadId": upload_id,
                                   "PartNumber": n}, ExpiresIn=int(expires_s)))
            for n, off, sz in layout]
    except Exception as e:  # noqa: BLE001 — our keys / bucket / API
        raise R2StorageError(f"r2 upload plan for {key}: {e}") from e
    return FilePlan(name, key, int(size), upload_id, parts)


def _norm_etag(tag: str) -> str:
    return str(tag or "").strip().strip('"').lower()


def verify_parts(plan: FilePlan, listed: list[dict], etags: dict[int, str]) -> list[str]:
    """Why R2's ``list_parts`` view disagrees with the plan (empty = agrees):
    exactly the planned part numbers, each of the planned size, each carrying
    the ETag its PUT returned."""
    problems: list[str] = []
    got = {int(p["PartNumber"]): p for p in listed}
    want = {p.number: p for p in plan.parts}
    if sorted(got) != sorted(want):
        problems.append(f"{plan.name}: R2 lists {len(got)} part(s) {sorted(got)[:8]}, "
                        f"planned {len(want)}")
    for n, p in sorted(want.items()):
        row = got.get(n)
        if row is None:
            continue
        if int(row.get("Size", -1)) != p.size:
            problems.append(f"{plan.name}: part {n} is {row.get('Size')} bytes on R2, "
                            f"planned {p.size}")
        sent = etags.get(n)
        if sent and _norm_etag(row.get("ETag", "")) != _norm_etag(sent):
            problems.append(f"{plan.name}: part {n} ETag on R2 differs from the PUT's")
    return problems


def _list_parts(client, bucket: str, key: str, upload_id: str) -> list[dict]:
    parts: list[dict] = []
    marker = 0
    while True:
        resp = client.list_parts(Bucket=bucket, Key=key, UploadId=upload_id,
                                 PartNumberMarker=marker)
        parts += list(resp.get("Parts") or [])
        if not resp.get("IsTruncated"):
            return parts
        marker = int(resp.get("NextPartNumberMarker") or 0)
        if not marker:
            return parts


def abort_file_upload(store, plan: FilePlan) -> None:
    """Best-effort abort of an unfinished multipart upload (no stored parts)."""
    if not plan.upload_id:
        return
    try:
        store.client().abort_multipart_upload(Bucket=store.cfg.bucket, Key=plan.key,
                                              UploadId=plan.upload_id)
    except Exception as e:  # noqa: BLE001 — the bucket lifecycle rule is the backstop
        log.warning("r2: abort of %s failed (%s); lifecycle expiry cleans it", plan.key, e)


def _object_size(store, key: str) -> int:
    try:
        return int(store.client().head_object(Bucket=store.cfg.bucket, Key=key)["ContentLength"])
    except Exception as e:  # noqa: BLE001
        raise R2TransferError(f"r2 head of {key}: {e}") from e


def complete_file_upload(store, plan: FilePlan, etags: dict[int, str]) -> None:
    """Finish one file: check ``list_parts`` against the plan (abort on any
    mismatch), complete it, and check the object's final size."""
    if plan.upload_id:
        client = store.client()
        try:
            listed = _list_parts(client, store.cfg.bucket, plan.key, plan.upload_id)
        except Exception as e:  # noqa: BLE001
            abort_file_upload(store, plan)
            raise R2StorageError(f"r2 list_parts of {plan.key}: {e}") from e
        problems = verify_parts(plan, listed, etags)
        if problems:
            abort_file_upload(store, plan)
            raise R2TransferError("r2 upload incomplete: " + "; ".join(problems))
        try:
            client.complete_multipart_upload(
                Bucket=store.cfg.bucket, Key=plan.key, UploadId=plan.upload_id,
                MultipartUpload={"Parts": [
                    {"PartNumber": int(p["PartNumber"]), "ETag": p["ETag"]}
                    for p in sorted(listed, key=lambda r: int(r["PartNumber"]))]})
        except Exception as e:  # noqa: BLE001
            abort_file_upload(store, plan)
            raise R2StorageError(f"r2 complete of {plan.key}: {e}") from e
    size = _object_size(store, plan.key)
    if size != plan.size:
        raise R2TransferError(f"r2 object {plan.name} is {size} bytes, the pod listed {plan.size}")


# ── 4. read back without touching the disk ───────────────────────────────────


def stream_sha256(store, key: str, *, chunk_bytes: int = 8 << 20) -> tuple[str, int]:
    """``(sha256 hex, bytes)`` of an R2 object, streamed — nothing written."""
    try:
        body = store.client().get_object(Bucket=store.cfg.bucket, Key=key)["Body"]
        h = hashlib.sha256()
        n = 0
        for chunk in iter(lambda: body.read(chunk_bytes), b""):
            h.update(chunk)
            n += len(chunk)
    except Exception as e:  # noqa: BLE001
        raise R2TransferError(f"r2 read of {key}: {e}") from e
    return h.hexdigest(), n


def read_range(store, key: str, start: int, length: int) -> bytes:
    if length <= 0:
        return b""
    try:
        resp = store.client().get_object(Bucket=store.cfg.bucket, Key=key,
                                         Range=f"bytes={int(start)}-{int(start) + int(length) - 1}")
        return resp["Body"].read()
    except Exception as e:  # noqa: BLE001
        raise R2TransferError(f"r2 range read of {key}: {e}") from e


@dataclass
class R2Checkpoint:
    """One leg's checkpoint on R2: ``files`` = {name: {key, size, sha256}}.
    JSON-safe (it rides in the pending-upload record), carries no secret."""

    bucket: str
    prefix: str
    files: dict[str, dict]

    def to_record(self) -> dict:
        return {"bucket": self.bucket, "prefix": self.prefix,
                "files": {n: dict(m) for n, m in sorted(self.files.items())}}

    @classmethod
    def from_record(cls, rec: dict) -> R2Checkpoint:
        return cls(str(rec["bucket"]), str(rec["prefix"]),
                   {str(n): dict(m) for n, m in dict(rec["files"]).items()})

    @property
    def total_bytes(self) -> int:
        return sum(int(m["size"]) for m in self.files.values())

    def digests(self) -> dict[str, str]:
        return {n: str(m["sha256"]) for n, m in self.files.items()}


def materialize_skeleton(store, ckpt: R2Checkpoint, dest: Path | str) -> Path:
    """A sparse stand-in of the checkpoint for the contract guard
    (:func:`cascade.eval.checkpoint_guard.verify_checkpoint`): small files
    in full (hash-checked against the streamed digest), every safetensors
    file as its real header followed by a HOLE of the real length, any other
    large file as a hole. KB on disk; ``verify_weights_finite`` must NEVER run
    on it (holes read as zeros) — that check runs where the full bytes land
    (the push)."""
    from ..eval.checkpoint_guard import MAX_HEADER_BYTES

    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    for name, meta in sorted(ckpt.files.items()):
        size = int(meta["size"])
        p = d / name
        if name.endswith(".safetensors"):
            head = read_range(store, meta["key"], 0, min(8, size))
            if len(head) == 8:
                (n,) = struct.unpack("<Q", head)
                if 0 < n <= MAX_HEADER_BYTES and 8 + n <= size:
                    head += read_range(store, meta["key"], 8, n)
            with open(p, "wb") as f:
                f.write(head)
                f.truncate(size)
        elif size <= SMALL_FILE_MAX:
            data = read_range(store, meta["key"], 0, size)
            if len(data) != size or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise R2TransferError(f"r2 re-read of {name} disagrees with its streamed digest")
            p.write_bytes(data)
        else:
            with open(p, "wb") as f:
                f.truncate(size)
    return d


def delete_checkpoint(store, ckpt: R2Checkpoint) -> None:
    """Best-effort delete of a pushed checkpoint's objects (lifecycle backstop)."""
    keys = [{"Key": m["key"]} for m in ckpt.files.values()]
    if not keys:
        return
    try:
        store.client().delete_objects(Bucket=store.cfg.bucket, Delete={"Objects": keys})
    except Exception as e:  # noqa: BLE001
        log.warning("r2: deleting %s* failed (%s); lifecycle expiry cleans it", ckpt.prefix, e)


# ── 2. the pod side, over the pinned ssh ─────────────────────────────────────


def _ssh(runner):
    from .remote import run_ssh

    return runner or run_ssh


def _in_dir(host, remote_dir: str) -> str:
    # The receipt's checkpoint dir is relative to the worker's workdir (see
    # harvest_remote_dir); an absolute one makes the first cd a no-op.
    return f"cd {shlex.quote(host.workdir)} && cd {shlex.quote(remote_dir)}"


def list_command(host, remote_dir: str) -> str:
    return f"{_in_dir(host, remote_dir)} && find . -maxdepth 1 -type f -printf '%f\\t%s\\n'"


def list_remote_files(host, remote_dir: str, *, runner=None, timeout: int = 120) -> dict[str, int]:
    """``{name: size}`` of the regular top-level checkpoint files on the pod
    the Hippius push would carry (symlinks and subdirectories never count)."""
    from .remote import RemoteDispatchError, build_ssh_argv

    proc = _ssh(runner)(build_ssh_argv(host, list_command(host, remote_dir)), timeout)
    if proc.returncode != 0:
        raise RemoteDispatchError(
            f"listing the checkpoint on {host.name}:{remote_dir} failed "
            f"(rc={proc.returncode}): {(proc.stderr or '')[-200:]}", returncode=proc.returncode)
    files: dict[str, int] = {}
    skipped: list[str] = []
    for line in (proc.stdout or "").splitlines():
        name, sep, size = line.rpartition("\t")
        if not sep or not size.strip().isdigit():
            continue
        if checkpoint_file_allowed(name):
            files[name] = int(size)
        else:
            skipped.append(name)
    if skipped:
        log.info("r2 harvest from %s: %d file(s) outside the push allow-list left on the "
                 "pod: %s", host.name, len(skipped), ", ".join(sorted(skipped)[:6]))
    return files


def part_upload_command(host, remote_dir: str, name: str, part: PartPlan) -> str:
    """Stream bytes ``[offset, offset+size)`` of ``name`` into the part's link.
    The link is read from stdin (never on the ssh command line); curl's
    response headers come back on stdout for the ETag."""
    if not _SAFE_NAME.match(name):
        raise ValueError(f"unsafe checkpoint file name {name!r}")
    return (f"{_in_dir(host, remote_dir)} && IFS= read -r url && "
            f"tail -c +{int(part.offset) + 1} -- {shlex.quote(name)} | head -c {int(part.size)} | "
            f"curl -fsS --connect-timeout 30 -X PUT -H 'Content-Length: {int(part.size)}' "
            f"-H 'Transfer-Encoding:' -H 'Expect:' -T - -D - -o /dev/null \"$url\"")


def parse_etag(stdout: str) -> str:
    """The ETag of the LAST response in curl's ``-D -`` dump ("" if none)."""
    found = _ETAG_RE.findall((stdout or "").replace("\r", ""))
    return _norm_etag(found[-1]) if found else ""


def upload_part_via_ssh(host, remote_dir: str, name: str, part: PartPlan, *, runner=None,
                        attempts: int = PART_ATTEMPTS, timeout: int = PART_TIMEOUT_S,
                        sleep=time.sleep) -> str:
    """One part pod → R2, retried; returns its ETag."""
    import subprocess

    from .remote import RemoteDispatchError, build_ssh_argv

    argv = build_ssh_argv(host, part_upload_command(host, remote_dir, name, part))
    last, rc = "", None
    for i in range(attempts):
        try:
            proc = _ssh(runner)(argv, timeout, part.url + "\n")
            rc = proc.returncode
            etag = parse_etag(proc.stdout) if rc == 0 else ""
            if etag:
                return etag
            last = (proc.stderr or "")[-200:] or "no ETag in the response"
        except subprocess.TimeoutExpired:
            rc, last = None, f"timed out after {timeout}s"
        log.warning("r2 harvest %s: %s part %d attempt %d/%d failed: %s", host.name, name,
                    part.number, i + 1, attempts, last)
        if i + 1 < attempts:
            sleep(PART_RETRY_BACKOFF_S * (2 ** i))
    raise RemoteDispatchError(
        f"checkpoint upload to R2 from {host.name}: {name} part {part.number} failed after "
        f"{attempts} attempts: {last}", returncode=rc)


def upload_parts_via_ssh(host, remote_dir: str, plans: list[FilePlan], *, runner=None,
                         parallel: int = PARALLEL_PARTS, sleep=time.sleep) -> dict[str, dict[int, str]]:
    """Every part of every plan, ``parallel`` at a time → {name: {part: etag}}."""
    jobs = [(fp, part) for fp in plans for part in fp.parts]
    etags: dict[str, dict[int, str]] = {fp.name: {} for fp in plans}
    lock = threading.Lock()

    def one(job):
        fp, part = job
        tag = upload_part_via_ssh(host, remote_dir, fp.name, part, runner=runner, sleep=sleep)
        with lock:
            etags[fp.name][part.number] = tag

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(parallel)),
                                               thread_name_prefix="r2-part") as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                for other in futures:
                    other.cancel()
                raise exc
    return etags


def harvest_to_r2(host, remote_dir: str, store, prefix: str, *, runner=None,
                  parallel: int = PARALLEL_PARTS, part_bytes: int = PART_BYTES,
                  max_bytes: int = MAX_CHECKPOINT_BYTES, sleep=time.sleep) -> R2Checkpoint:
    """Move ``host:remote_dir`` to R2 under ``prefix`` and hash it back:
    list → plan → pod uploads → complete (part-checked) → stream sha256."""
    files = list_remote_files(host, remote_dir, runner=runner)
    total = sum(files.values())
    if total > max_bytes:
        raise CheckpointTooLarge(f"checkpoint dir on the pod is {total} bytes "
                                 f"(> {max_bytes}) — not a contract checkpoint")
    plans: list[FilePlan] = []
    done: set[str] = set()
    try:
        for name, size in sorted(files.items()):
            plans.append(plan_file_upload(store, name, prefix + name, size,
                                          part_bytes=part_bytes))
        etags = upload_parts_via_ssh(host, remote_dir, plans, runner=runner,
                                     parallel=parallel, sleep=sleep)
        for fp in plans:
            complete_file_upload(store, fp, etags.get(fp.name, {}))
            done.add(fp.name)
    except BaseException:
        for fp in plans:
            if fp.name not in done:
                abort_file_upload(store, fp)
        raise
    out: dict[str, dict] = {}

    def digest(fp: FilePlan):
        sha, n = stream_sha256(store, fp.key)
        if n != fp.size:
            raise R2TransferError(f"r2 object {fp.name} streamed {n} bytes, planned {fp.size}")
        return fp, sha

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(parallel)),
                                               thread_name_prefix="r2-hash") as pool:
        for fp, sha in pool.map(digest, plans):
            out[fp.name] = {"key": fp.key, "size": fp.size, "sha256": sha}
    log.info("r2 harvest from %s: %d file(s), %.2f GB on r2://%s/%s", host.name, len(out),
             total / 1e9, store.cfg.bucket, prefix)
    return R2Checkpoint(store.cfg.bucket, prefix, out)


# ── 5. the Hippius push from R2 (relay host, or bounded local temp) ──────────


def presigned_get_plan(store, ckpt: R2Checkpoint,
                       expires_s: int = DOWNLOAD_EXPIRES_S) -> dict[str, dict]:
    client = store.client()
    try:
        return {name: {"url": client.generate_presigned_url(
            "get_object", Params={"Bucket": store.cfg.bucket, "Key": m["key"]},
            ExpiresIn=int(expires_s)), "size": int(m["size"]), "sha256": str(m["sha256"])}
            for name, m in sorted(ckpt.files.items())}
    except Exception as e:  # noqa: BLE001
        raise R2StorageError(f"r2 download links for {ckpt.prefix}: {e}") from e


def _copy_hashed(src, dst: Path, name: str, size: int, sha: str) -> None:
    h = hashlib.sha256()
    n = 0
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as f:
        for chunk in iter(lambda: src.read(8 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            f.write(chunk)
    if n != int(size) or h.hexdigest() != str(sha):
        tmp.unlink(missing_ok=True)
        raise R2TransferError(f"download of {name} does not match what was verified "
                              f"({n} bytes)")
    tmp.replace(dst)


def url_fetcher(timeout: float = 600.0):
    """``fetch(name, meta, dest)`` over presigned GET links (the relay)."""
    def fetch(name: str, meta: dict, dest: Path) -> None:
        with urllib.request.urlopen(meta["url"], timeout=timeout) as resp:  # noqa: S310 — operator-signed
            _copy_hashed(resp, dest, name, meta["size"], meta["sha256"])
    return fetch


def store_fetcher(store, ckpt: R2Checkpoint):
    """``fetch(name, meta, dest)`` straight from the store (the box's local push)."""
    def fetch(name: str, meta: dict, dest: Path) -> None:
        body = store.client().get_object(Bucket=store.cfg.bucket,
                                         Key=ckpt.files[name]["key"])["Body"]
        _copy_hashed(body, dest, name, meta["size"], meta["sha256"])
    return fetch


def relay_push(plan: dict, *, workdir: Path | str, fetch) -> dict:
    """Download → re-verify → push to Hippius → delete the copy. Runs on the
    relay host (``python -m cascade.trainer.r2_transfer relay``) and in the
    box's bounded local fallback. Returns ``{"ok": True, "ref", "size_bytes"}``
    or ``{"ok": False, "kind": "tamper"|"diverged"|"storage", "why"}``."""
    from ..eval.checkpoint_guard import (
        CheckpointDiverged,
        CheckpointTampered,
        verify_checkpoint,
        verify_weights_finite,
    )

    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="r2push-", dir=root))
    try:
        for name, meta in sorted(dict(plan["files"]).items()):
            if not checkpoint_file_allowed(name):
                return {"ok": False, "kind": "storage", "why": f"bad file name {name!r}"}
            fetch(name, meta, tmp / name)
        try:
            # The bytes are the ones the box verified (sha256 above); the code
            # check refreshes a historical wrapper like the ssh path, the
            # finite check needs the real tensors (never the box's skeleton).
            verify_checkpoint(tmp)
            verify_weights_finite(tmp)
        except CheckpointTampered as e:
            return {"ok": False, "kind": "tamper", "why": str(e)}
        except CheckpointDiverged as e:
            return {"ok": False, "kind": "diverged", "why": str(e)}
        hub = HubConfig(**dict(plan.get("hub") or {}))
        up = upload_dir_to_hub_or_hf(tmp, str(plan["repo"]), hub,
                                     hf_repo=plan.get("hf_repo") or None)
        return {"ok": True, "ref": up.ref.immutable_ref, "size_bytes": int(up.size_bytes)}
    except Exception as e:  # noqa: BLE001 — storage / transport: retried by the box
        return {"ok": False, "kind": "storage", "why": f"{type(e).__name__}: {e}"[-400:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def relay_command(relay) -> str:
    return (f"cd {shlex.quote(relay.workdir)} && {shlex.quote(relay.remote_python)} -m "
            f"cascade.trainer.r2_transfer relay --workdir "
            f"{shlex.quote(relay.workdir.rstrip('/') + '/_r2_relay')}")


def push_via_relay(relay, plan: dict, *, runner=None, timeout: int = 4 * 3600) -> dict:
    """Run :func:`relay_push` on the operator relay over ssh; the plan (links
    and any forwarded env) travels on stdin."""
    from .remote import build_ssh_argv

    proc = _ssh(runner)(build_ssh_argv(relay, relay_command(relay)), timeout, json.dumps(plan))
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(RESULT_MARK):
            try:
                return json.loads(line[len(RESULT_MARK):])
            except ValueError:
                break
    return {"ok": False, "kind": "storage",
            "why": f"relay {relay.name} rc={proc.returncode}: {(proc.stderr or '')[-300:]}"}


def _relay_host(runner, name: str):
    from .remote import load_hosts

    hosts = list(runner.remote_hosts or [])
    path = getattr(runner, "remote_hosts_path", None)
    if path is not None:
        with contextlib.suppress(Exception):
            hosts = load_hosts(path)
    return next((h for h in hosts if h.name == name), None)


def push_from_r2(runner, record: dict) -> HubUpload:
    """The Hippius push of a verified R2 checkpoint (``record["r2"]``) →
    :class:`HubUpload`. Relay host when ``[round] ckpt_relay_host`` is set;
    else a local temp under ``work_root/_r2_push`` only when free disk stays
    above ``min_free_disk_gb`` after the download; else :class:`R2PushDeferred`.
    A tamper / diverged verdict at the push raises the leg's own exceptions."""
    from .faults import _FundedDiverged, _FundedTamper

    ckpt = R2Checkpoint.from_record(record["r2"])
    store = getattr(runner, "_r2_store", None) or ckpt_store(runner.cfg, bucket=ckpt.bucket)
    repo = record["ckpt_repo"]
    hub = runner.hub()
    plan = {"repo": repo, "hf_repo": runner._hf_ckpt_repo(repo),
            "hub": {"registry_url": getattr(hub, "registry_url", HubConfig.registry_url),
                    "namespace": getattr(hub, "namespace", HubConfig.namespace)}}
    rnd = runner.cfg.round
    relay_name = str(getattr(rnd, "ckpt_relay_host", "") or "")
    if relay_name:
        relay = _relay_host(runner, relay_name)
        if relay is None:
            raise R2PushDeferred(f"[round] ckpt_relay_host={relay_name!r} is not in the hosts file")
        plan["files"] = presigned_get_plan(store, ckpt)
        if relay.forward_env and not relay.isolated:
            plan["env"] = {n: os.environ[n] for n in relay.forward_env if n in os.environ}
        result = push_via_relay(relay, plan, runner=getattr(runner, "_r2_ssh_runner", None))
    else:
        from .runner_common import DISK_USAGE

        gb = float(getattr(rnd, "min_free_disk_gb", 0.0) or 0.0)
        work = Path(runner.work_root)
        free = int((DISK_USAGE or shutil.disk_usage)(work).free)
        need = ckpt.total_bytes + int(gb * (1 << 30))
        if free < need:
            raise R2PushDeferred(
                f"no [round] ckpt_relay_host and {free / (1 << 30):.1f} GiB free < "
                f"{ckpt.total_bytes / (1 << 30):.1f} GiB checkpoint + {gb:g} GiB floor — "
                "Hippius push requeued from R2")
        plan["files"] = {n: {"size": int(m["size"]), "sha256": m["sha256"]}
                         for n, m in ckpt.files.items()}
        result = relay_push(plan, workdir=work / "_r2_push", fetch=store_fetcher(store, ckpt))
    if result.get("ok"):
        return HubUpload(ref=HubRef.parse(str(result["ref"])),
                         size_bytes=int(result.get("size_bytes", 0) or 0))
    kind, why = result.get("kind"), str(result.get("why", ""))
    if kind == "tamper":
        raise _FundedTamper(f"checkpoint: {why}")
    if kind == "diverged":
        raise _FundedDiverged(why)
    raise StorageError(f"hippius push from r2 failed: {why}")


# ── the leg's harvest step (runner glue) ─────────────────────────────────────


def harvest_checkpoint_r2(runner, host, receipt, contract, seeds, suffix: str, *,
                          release=None):
    """r2-mode :meth:`TrainerRunner._harvest_funded_checkpoint`: pod → R2,
    verify (sizes, tensor digests by streaming hash, contract guard on a
    sparse skeleton), ``release()`` the pod, then the Hippius push via
    :func:`cascade.trainer.loop.upload_harvested_checkpoint` (its retries and
    pending record — here naming the R2 keys). Same verdicts as the ssh
    harvest: deviations raise ``_FundedTamper`` / ``_FundedDiverged``, our
    storage ``_OperatorFault``, the pod's transport a plain error (infra)."""
    from ..eval.checkpoint_guard import CheckpointTampered, verify_checkpoint
    from .ckpt_digest import listing_mismatches
    from .faults import _FundedTamper, _OperatorFault
    from .loop import upload_harvested_checkpoint
    from .runner_common import harvest_job_id

    job_id = harvest_job_id(receipt, suffix)
    try:
        store = getattr(runner, "_r2_store", None) or ckpt_store(runner.cfg)
    except R2StorageError as e:
        raise _OperatorFault(str(e)) from e
    prefix = ckpt_prefix(runner.cfg, seeds.base_seed, receipt.miner_hotkey, job_id)
    try:
        ckpt = harvest_to_r2(host, receipt.checkpoint_dir, store, prefix,
                             runner=getattr(runner, "_r2_ssh_runner", None),
                             sleep=getattr(runner, "_upload_retry_sleep", None) or time.sleep)
    except CheckpointTooLarge as e:
        raise _FundedTamper(f"checkpoint: {e}") from e
    except R2StorageError as e:
        raise _OperatorFault(f"checkpoint upload to R2: {e}") from e
    expected = dict(getattr(receipt, "tensor_digests", None) or {})
    if expected:
        bad = listing_mismatches(ckpt.digests(), expected)
        if bad:
            raise _FundedTamper("checkpoint: " + "; ".join(bad))
    elif runner.cfg.round.funded_require_tensor_digests:
        raise _FundedTamper("checkpoint: the worker receipt carries no tensor digests")
    else:
        log.warning("funded leg %s: worker receipt carries no tensor digests (worker image "
                    "predates them) — checkpoint digest check skipped", receipt.miner_hotkey)
    # Contract guard on a sparse skeleton in the SYSTEM temp dir (KB on disk),
    # never under work_root; the finite-weights check runs at the push.
    skeleton = Path(tempfile.mkdtemp(prefix="cascade-r2-skel-"))
    try:
        materialize_skeleton(store, ckpt, skeleton)
        verify_checkpoint(skeleton, contract)
    except CheckpointTampered as e:
        raise _FundedTamper(f"checkpoint: {e}") from e
    finally:
        shutil.rmtree(skeleton, ignore_errors=True)
    if release is not None:
        # Every byte the leg produced is on R2 and verified: the pod (and its
        # bill) goes NOW, before the Hippius push, whatever the push does.
        release()
    size = contract.arch_preset
    record = {
        "r2": ckpt.to_record(), "harvest_job_id": job_id,
        "ckpt_repo": f"{runner.hub().namespace}/ckpt-r{seeds.base_seed}-{receipt.role}-{size}{suffix}",
        "suffix": suffix, "size": size,
        "miner_hotkey": receipt.miner_hotkey, "miner_uid": int(receipt.miner_uid),
        "role": receipt.role, "gen_ref": receipt.gen_ref,
        "corpus_digest": receipt.corpus_digest, "train_block": int(receipt.train_block),
        "gpu_name": receipt.gpu_name, "receipt_size": receipt.size or "",
        "warm_start_ref": receipt.init_pointer,
        "contract_digest": receipt.training_contract_digest,
    }
    return upload_harvested_checkpoint(runner, record, base_seed=seeds.base_seed)


# ── relay entry point ────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cascade.trainer.r2_transfer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("relay", help="push one R2 checkpoint to Hippius (plan JSON on stdin)")
    rp.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    plan = json.loads(sys.stdin.read())
    for k, v in dict(plan.pop("env", None) or {}).items():
        os.environ[str(k)] = str(v)
    result = relay_push(plan, workdir=args.workdir, fetch=url_fetcher())
    print(RESULT_MARK + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover — the relay's CLI
    sys.exit(main())
