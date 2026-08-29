"""Private submission store: code comes to the gateway, only champions leave.

DEC-CA-0036's second half. Instead of miners hosting their own Hub repos (the
account-watching and delete-my-repo-after-losing problems), the generator ZIP
rides the SAME authenticated intake request that carries the Lium key. The
operator stores it privately, content-addressed by the ZIP's sha256; the miner
commits a **vault ref** on chain; and code goes public only when (and if) it
takes the throne — merged into the operator's public ``champions/`` prefix,
the way sn100's PRISM publishes ``top-model/``. Losers stay private forever.

The vault ref deliberately rides the EXISTING Hub ``repo@digest`` grammar
under a reserved repo id (:data:`VAULT_REPO`), so the on-chain payload is a
byte-ordinary ``metro-v1:gen:hippius:vault/direct@sha256:<64hex>``:

* ``parse_commit`` accepts it unchanged on every deployed validator —
  participant sets (and therefore receipts) cannot fork on the scheme, and
  validators never fetch generator code anyway.
* Everything downstream that treats a ref as an opaque ``repo@digest``
  (manifests, dedup keys, dashboards) keeps working.
* Only the FETCH path branches: ``fetch_from_hub`` resolves ``vault/…`` from
  the local vault dir (:func:`fetch_vault_snapshot`) — the orchestrator's
  store, or the single staged ZIP a dispatch placed on a pod — falling back
  to the published ``champions/`` object for code that has gone public.

Ownership is part of the store, not a convention: ``put`` records the
uploading hotkey, and the trainer refuses a vault ref whose digest was
uploaded by someone else — a copied digest (e.g. the published champion's)
can never enter a round as a different miner's submission. ZIP contents are
treated as hostile: extraction admits regular files only, resolved strictly
inside the destination.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from ..shared.hippius import FETCH_COMPLETE_MARKER, HubRef, StorageError

__all__ = [
    "CHAMPION_INDEX_KEY",
    "VAULT_DIR_ENV",
    "CHAMPION_BASE_ENV",
    "VAULT_REPO",
    "SubmissionStore",
    "champion_zip_key",
    "extract_zip_safely",
    "fetch_vault_snapshot",
    "is_vault_ref",
    "parse_vault_ref",
    "vault_ref",
]

# The reserved repo id vault refs ride under. Never a real Hub repo: the fetch
# path intercepts it before any registry call, and the intake is the only
# writer of digests underneath it.
VAULT_REPO = "vault/direct"

# Where a fetch looks for `<digest>.zip`: the orchestrator's store dir, or the
# per-dispatch staging dir on a pod (which must hold ONLY that entry's ZIP —
# a funded pod's payer can read the box, so the whole vault never ships).
VAULT_DIR_ENV = "CASCADE_VAULT_DIR"
# Anonymous https base (`{s3_endpoint}/{bucket}`) for the published champion
# objects — the public fallback that lets anyone re-derive the king.
CHAMPION_BASE_ENV = "CASCADE_CHAMPION_BASE"

CHAMPION_INDEX_KEY = "champions/index.json"

# Matches [generator] max_repo_mb's ceiling; the store takes an explicit cap
# so the intake can pass the configured value.
DEFAULT_MAX_ZIP_BYTES = 128 * 1024 * 1024


def vault_ref(digest_hex: str) -> str:
    """``vault/direct@sha256:<hex>`` for a stored ZIP's digest."""
    if len(digest_hex) != 64 or any(c not in "0123456789abcdef" for c in digest_hex):
        raise StorageError(f"not a sha256 hex digest: {digest_hex!r}")
    return f"{VAULT_REPO}@sha256:{digest_hex}"


def parse_vault_ref(ref: str) -> str | None:
    """The digest hex of a vault ref, or ``None`` for any other ref."""
    try:
        parsed = HubRef.parse(ref)
    except StorageError:
        return None
    if parsed.repo != VAULT_REPO or not parsed.digest.startswith("sha256:"):
        return None
    return parsed.digest[len("sha256:"):]


def is_vault_ref(ref: str) -> bool:
    return parse_vault_ref(ref) is not None


def champion_zip_key(digest_hex: str) -> str:
    return f"champions/{digest_hex}.zip"


def extract_zip_safely(data: bytes, dest_dir: Path | str) -> Path:
    """Extract hostile ZIP bytes: regular files only, strictly inside ``dest``.

    Mirrors ``unpack_tar_to_dir``'s posture: no symlinks (a ZIP "symlink" is
    an external-attr trick — everything is written as a plain file from the
    member's bytes), no absolute paths, no ``..`` escapes, and the resolved
    parent of every write must stay inside the destination.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise StorageError(f"not a valid zip: {e}") from e
    with zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue                      # directories materialise via parents
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise StorageError(f"zip member escapes the destination: {name!r}")
            target = dest / p
            if not target.resolve().parent.is_relative_to(dest_resolved) \
                    and target.resolve().parent != dest_resolved:
                raise StorageError(f"zip member escapes the destination: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    return dest


class SubmissionStore:
    """Content-addressed private ZIP store: ``<dir>/<sha256>.zip`` + meta.

    Files land 0600 under a 0700 dir (the payer-vault posture: operator-local,
    off git, off any rsync'd pod tree). ``put`` validates before it stores —
    size cap, real ZIP, no hostile member paths — so everything IN the store
    is extractable, and records the uploading hotkey as the ownership fact
    the trainer enforces.
    """

    def __init__(self, dir: Path | str, *, max_bytes: int = DEFAULT_MAX_ZIP_BYTES,
                 clock: Callable[[], float] = time.time) -> None:
        self.dir = Path(dir)
        self.max_bytes = int(max_bytes)
        self.clock = clock
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)

    def _zip_path(self, digest_hex: str) -> Path:
        return self.dir / f"{digest_hex}.zip"

    def _meta_path(self, digest_hex: str) -> Path:
        return self.dir / f"{digest_hex}.json"

    def put(self, zip_bytes: bytes, hotkey: str) -> str:
        """Store a submission; returns its sha256 hex digest.

        Idempotent for the SAME uploader (re-POSTing your own ZIP is a no-op
        that returns the same digest); a digest already owned by a DIFFERENT
        hotkey is refused — content-addressing must never let a second miner
        claim bytes the first uploaded (earliest upload wins; the dedup screen
        separately kills byte-copies that arrive as distinct uploads).
        """
        if len(zip_bytes) > self.max_bytes:
            raise StorageError(
                f"zip_too_large: {len(zip_bytes)} bytes > cap {self.max_bytes}")
        if not zip_bytes:
            raise StorageError("empty submission body")
        # Validate contents BEFORE storing: everything in the store must be
        # extractable later, when the miner is no longer on the wire to fix it.
        import tempfile

        with tempfile.TemporaryDirectory(dir=self.dir) as probe:
            extract_zip_safely(zip_bytes, Path(probe) / "x")
        digest = hashlib.sha256(zip_bytes).hexdigest()
        existing = self.owner(digest)
        if existing is not None:
            if existing != hotkey:
                raise StorageError(
                    "digest_owned: identical bytes were already submitted by "
                    "another hotkey (earliest upload owns the content)")
            return digest
        zp, mp = self._zip_path(digest), self._meta_path(digest)
        tmp = zp.with_suffix(".zip.tmp")
        tmp.write_bytes(zip_bytes)
        os.chmod(tmp, 0o600)
        os.replace(tmp, zp)
        mtmp = mp.with_suffix(".json.tmp")
        mtmp.write_text(json.dumps({
            "hotkey": hotkey, "uploaded_at": self.clock(), "size": len(zip_bytes),
        }), encoding="utf-8")
        os.chmod(mtmp, 0o600)
        os.replace(mtmp, mp)
        return digest

    def has(self, digest_hex: str) -> bool:
        return self._zip_path(digest_hex).is_file()

    def owner(self, digest_hex: str) -> str | None:
        """The uploading hotkey, or ``None`` when the digest is not stored."""
        mp = self._meta_path(digest_hex)
        if not mp.is_file():
            return None
        try:
            return str(json.loads(mp.read_text(encoding="utf-8"))["hotkey"])
        except (ValueError, KeyError):
            return None

    def zip_bytes(self, digest_hex: str) -> bytes:
        zp = self._zip_path(digest_hex)
        if not zp.is_file():
            raise StorageError(f"vault digest not stored: {digest_hex}")
        data = zp.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest_hex:
            raise StorageError(
                f"vault corruption: {digest_hex}.zip hashes to {actual}")
        return data

    def extract(self, digest_hex: str, dest_dir: Path | str) -> Path:
        return extract_zip_safely(self.zip_bytes(digest_hex), dest_dir)

    def stage_for_dispatch(self, digest_hex: str, staging_dir: Path | str) -> Path:
        """Copy ONE entry's ZIP into a per-dispatch staging dir.

        This — never a sync of the whole store — is what may ship to a pod:
        a funded pod's payer can read the box, so it must only ever hold its
        own submission. No meta file travels (the pod needs bytes, not
        ownership records).
        """
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        data = self.zip_bytes(digest_hex)     # integrity-checked read
        target = staging / f"{digest_hex}.zip"
        tmp = target.with_suffix(".zip.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return target


def fetch_vault_snapshot(ref: HubRef, dest_dir: Path | str) -> Path:
    """Materialise a vault ref into ``dest_dir`` (the ``fetch_from_hub`` branch).

    Resolution order:

    1. ``$CASCADE_VAULT_DIR/<digest>.zip`` — the orchestrator's store, or the
       single ZIP a dispatch staged onto this pod;
    2. the published champion object at
       ``$CASCADE_CHAMPION_BASE/champions/<digest>.zip`` (anonymous GET) —
       how ANYONE re-derives the king once its code has gone public.

    Same completion-marker contract as the Hub fetch: a dir already carrying
    the marker is reused (the digest pins the bytes), and the extract lands in
    a temp dir renamed into place so concurrent lanes never read a torn tree.
    """
    digest = parse_vault_ref(ref.immutable_ref)
    if digest is None:
        raise StorageError(f"not a vault ref: {ref.immutable_ref}")
    dest = Path(dest_dir)
    if (dest / FETCH_COMPLETE_MARKER).exists():
        return dest
    data: bytes | None = None
    vault_dir = os.environ.get(VAULT_DIR_ENV, "")
    if vault_dir:
        zp = Path(vault_dir) / f"{digest}.zip"
        if zp.is_file():
            data = zp.read_bytes()
    if data is None:
        base = os.environ.get(CHAMPION_BASE_ENV, "").rstrip("/")
        if base:
            url = f"{base}/{champion_zip_key(digest)}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — operator-configured base
                    data = resp.read()
            except OSError as e:
                raise StorageError(
                    f"generator_artifact_unreachable: vault digest {digest} is not "
                    f"in ${VAULT_DIR_ENV} and the champion fetch failed: {e}") from e
    if data is None:
        raise StorageError(
            f"generator_artifact_unreachable: vault digest {digest} not present in "
            f"${VAULT_DIR_ENV} and no ${CHAMPION_BASE_ENV} configured — private "
            "submissions resolve only where their ZIP was stored or staged")
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise StorageError(f"vault fetch of {digest} hashed to {actual} — refusing")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.vault-{os.getpid()}"
    try:
        extract_zip_safely(data, tmp)
        (tmp / FETCH_COMPLETE_MARKER).touch()
        try:
            os.rename(tmp, dest)
        except OSError:
            if not (dest / FETCH_COMPLETE_MARKER).exists():
                shutil.rmtree(dest, ignore_errors=True)
                os.rename(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest
