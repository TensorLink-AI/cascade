"""Lane D ingest guards (LANE_D_PLAN §1.1, §1.2, §3.2 invariant 3; W4).

A lane D checkpoint is miner bytes. Ingest takes a revealed entry to a
verified local checkpoint directory and NEVER imports, unpickles or executes
anything in it. Two transports share one guard pipeline:

* :func:`ingest_entry` reads the vault through a :class:`LaneDStorage`
  adapter (the box, CI and the walking skeleton);
* :func:`ingest_presigned` reads it through presigned GETs only (the eval
  pod, which never holds a credential, §1.2).

The guard order is pinned and fails closed at the first problem with a typed
:class:`IngestError` (each refusal also logs WARNING
``lane_d_ingest_refused <code> ...``):

1. the manifest hashes to the entry digest (``digest_mismatch``);
2. the manifest's file names are exactly the allowlist
   (``file_not_allowed``); storage only: no stray object under the prefix
   (``stray_object``), every object present at its size;
3. the declared weights size is within the budget (``oversize``) — so an
   oversized or disallowed entry is refused before any object is fetched;
4. every object is fetched size-capped at its manifest size and re-hashed
   (``oversize`` / ``size_mismatch`` / ``digest_mismatch`` / ``fetch_failed``);
5. ``model.py`` and ``forecast_wrapper.py`` equal the repo's bytes
   (``code_mismatch``);
6. ``config.json``'s ``"moe"`` object equals the contract's (``config_mismatch``);
7. the safetensors header, parsed by hand as JSON without touching tensor
   data: framing and offsets (``bad_header``), the tensor name set
   (``tensor_set_mismatch``), dtypes (``dtype_mismatch``), shapes
   (``shape_mismatch``);
8. every floating weight is finite (``non_finite``) — the only step that
   imports torch (safetensors loads the now fully bounded file).

Lane A's guard (:func:`cascade.eval.checkpoint_guard.verify_checkpoint`) is
not used and not changed: lane D's expectation is lane A's spec with BF16 in
place of F32 (:func:`expectation_for`). Only after every step does the
directory appear at ``dest_dir/<digest>``; a refusal leaves nothing behind.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from ..lane_d.intake import (
    ALLOWED_FILES,
    MANIFEST_OBJECT,
    IntakeError,
    canonical_manifest,
    check_object_list,
    entry_prefix,
)
from ..lane_d.storage import FetchError, LaneDStorage, fetch_presigned, fetch_presigned_bytes
from .checkpoint_guard import MAX_HEADER_BYTES, WEIGHTS_FILE

log = logging.getLogger("cascade.eval.lane_d_ingest")

CODE_FILES = ("forecast_wrapper.py", "model.py")
CONFIG_FILE = "config.json"
CONFIG_KEY = "moe"
LANE_D_ARCH = "cascade_pyramid"
# The manifest of four files is a few hundred bytes; anything far larger is
# not a manifest.
MAX_MANIFEST_BYTES = 64 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HOTKEY = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
# safetensors dtype -> bytes per element (the format's own table).
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
                "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
                "F8_E4M3": 1, "F8_E5M2": 1}


class IngestCode(enum.StrEnum):
    """Why an entry was refused at ingest (the miner's fault, recorded as
    ``checkpoint_fault`` by the settlement)."""

    DIGEST_MISMATCH = "digest_mismatch"
    FILE_NOT_ALLOWED = "file_not_allowed"
    STRAY_OBJECT = "stray_object"
    MISSING_OBJECT = "missing_object"
    SIZE_MISMATCH = "size_mismatch"
    MISSING_MANIFEST = "missing_manifest"
    BAD_MANIFEST = "bad_manifest"
    BAD_IDENTITY = "bad_identity"
    OVERSIZE = "oversize"
    CODE_MISMATCH = "code_mismatch"
    CONFIG_MISMATCH = "config_mismatch"
    BAD_HEADER = "bad_header"
    DTYPE_MISMATCH = "dtype_mismatch"
    SHAPE_MISMATCH = "shape_mismatch"
    TENSOR_SET_MISMATCH = "tensor_set_mismatch"
    NON_FINITE = "non_finite"
    FETCH_FAILED = "fetch_failed"


INGEST_CODES = tuple(c.value for c in IngestCode)


class IngestError(RuntimeError):
    """A typed ingest refusal; ``code`` is an :class:`IngestCode` member."""

    def __init__(self, code: IngestCode | str, detail: str = ""):
        self.code = IngestCode(code)
        self.detail = detail
        super().__init__(f"{self.code.value}: {detail}" if detail else self.code.value)


@dataclass(frozen=True)
class LaneDExpectation:
    """What an entry must match: the exact code bytes, the ``"moe"`` config
    object, the tensor spec ``{name: (dtype, shape)}`` and the weights byte
    budget."""

    code: dict
    config: dict
    tensors: dict
    max_weights_bytes: int

    def __post_init__(self):
        code = {str(k): bytes(v) for k, v in dict(self.code).items()}
        if set(code) != set(CODE_FILES):
            raise ValueError(f"code must hold exactly {CODE_FILES}; got {sorted(code)}")
        tensors = {str(k): (str(dt), tuple(int(x) for x in shape))
                   for k, (dt, shape) in dict(self.tensors).items()}
        if not tensors:
            raise ValueError("tensors must not be empty")
        if int(self.max_weights_bytes) <= 0:
            raise ValueError("max_weights_bytes must be positive")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "max_weights_bytes", int(self.max_weights_bytes))


def expectation_for(contract, *, dtype: str = "BF16") -> LaneDExpectation:
    """The lane D expectation for ``contract`` (needs torch): the pyramid
    code bytes, ``CascadeMoEConfig.from_contract(contract).to_dict()``, lane
    A's pinned tensor spec with every F32 replaced by ``dtype``, and a budget
    of at least lane A's F32 file size (so lane A's dtype is refused as
    ``dtype_mismatch``) and at most 3x it."""
    from ..trainer.pyramid_model import CascadeMoEConfig
    from .checkpoint_guard import BUDGET_FACTOR, expected_checkpoint_code, lane_d_state_spec

    arch = str(getattr(contract, "base_arch", "") or "")
    if arch != LANE_D_ARCH:
        raise ValueError(f"lane D checkpoints are {LANE_D_ARCH}; the contract pins {arch!r}")
    tensors, f32_bytes = lane_d_state_spec(contract, dtype=dtype)
    # F32 file = 8-byte length + header (<= MAX_HEADER_BYTES) + data.
    budget = min(BUDGET_FACTOR * f32_bytes, f32_bytes + 8 + MAX_HEADER_BYTES)
    return LaneDExpectation(code=expected_checkpoint_code(LANE_D_ARCH),
                            config=CascadeMoEConfig.from_contract(contract).to_dict(),
                            tensors=tensors, max_weights_bytes=budget)


# ── guard steps ─────────────────────────────────────────────────────────────


def _identity(hotkey, digest) -> tuple[str, str]:
    hk, dg = str(hotkey or ""), str(digest or "")
    if not _HOTKEY.match(hk) or hk in (".", "..") or not _HEX64.match(dg):
        raise IngestError(IngestCode.BAD_IDENTITY, f"hotkey={hk[:64]!r} digest={dg[:80]!r}")
    return hk, dg


def _manifest(raw: bytes, digest: str) -> dict:
    """Step 1: the manifest bytes hash to ``digest`` (as stored, or in their
    canonical form) and describe files as ``{name: {sha256, size}}``."""
    parsed = None
    if hashlib.sha256(raw).hexdigest() != digest:
        try:
            parsed = json.loads(raw.decode("utf-8"))
            ok = hashlib.sha256(canonical_manifest(parsed)).hexdigest() == digest
        except (UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError):
            ok = False
        if not ok:
            raise IngestError(IngestCode.DIGEST_MISMATCH,
                              "the manifest does not hash to the entry digest")
    if parsed is None:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise IngestError(IngestCode.BAD_MANIFEST, "unparseable manifest") from e
    files = parsed.get("files") if isinstance(parsed, dict) else None
    if not isinstance(files, dict) or not files:
        raise IngestError(IngestCode.BAD_MANIFEST, "no files")
    out = {}
    for name, meta in files.items():
        sha = meta.get("sha256") if isinstance(meta, dict) else None
        size = meta.get("size") if isinstance(meta, dict) else None
        if (not isinstance(name, str) or not isinstance(sha, str) or not _HEX64.match(sha)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise IngestError(IngestCode.BAD_MANIFEST, f"malformed entry for {str(name)[:64]!r}")
        out[name] = {"sha256": sha, "size": size}
    return out


def _allowlist(files: dict) -> None:
    """Step 2: the file names are exactly the allowlist."""
    names = set(files)
    if names != set(ALLOWED_FILES):
        extra = sorted(names - set(ALLOWED_FILES))
        missing = sorted(set(ALLOWED_FILES) - names)
        raise IngestError(IngestCode.FILE_NOT_ALLOWED,
                          f"extra={[n[:64] for n in extra[:5]]} missing={missing}")


def _budget(files: dict, expect: LaneDExpectation) -> None:
    """Step 3: the declared weights size is within the budget."""
    size = files[WEIGHTS_FILE]["size"]
    if size > expect.max_weights_bytes:
        raise IngestError(IngestCode.OVERSIZE, f"{WEIGHTS_FILE} declares {size} bytes; "
                                               f"budget {expect.max_weights_bytes}")


def _check_code(d: Path, expect: LaneDExpectation) -> None:
    """Step 5: the code files equal the repo's bytes exactly."""
    for name in CODE_FILES:
        if (d / name).read_bytes() != expect.code[name]:
            raise IngestError(IngestCode.CODE_MISMATCH,
                              f"{name} differs from this release's copy")


def _check_config(d: Path, expect: LaneDExpectation) -> None:
    """Step 6: ``config.json``'s model object equals the expectation."""
    try:
        obj = json.loads((d / CONFIG_FILE).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise IngestError(IngestCode.CONFIG_MISMATCH, f"{CONFIG_FILE} is unparseable") from e
    shipped = obj.get(CONFIG_KEY) if isinstance(obj, dict) else None
    if not isinstance(shipped, dict):
        raise IngestError(IngestCode.CONFIG_MISMATCH, f"{CONFIG_FILE} has no {CONFIG_KEY!r} object")
    if shipped != expect.config:
        diff = sorted(str(k)[:64] for k in set(shipped) | set(expect.config)
                      if shipped.get(k) != expect.config.get(k))
        raise IngestError(IngestCode.CONFIG_MISMATCH, f"{CONFIG_FILE} differs on {diff[:10]}")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _read_header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Step 7a: parse the safetensors framing by hand (u64 LE length, JSON
    header, data) and check every tensor's offsets: in bounds, sized by its
    dtype x shape, contiguous and covering the data exactly."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise IngestError(IngestCode.BAD_HEADER, "truncated safetensors length")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > MAX_HEADER_BYTES or 8 + n > size:
            raise IngestError(IngestCode.BAD_HEADER, f"safetensors header length {n} out of bounds")
        blob = f.read(n)
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise IngestError(IngestCode.BAD_HEADER, "unparseable safetensors header") from e
    if not isinstance(header, dict):
        raise IngestError(IngestCode.BAD_HEADER, "safetensors header is not an object")
    meta = header.pop("__metadata__", None)
    if meta is not None and not (isinstance(meta, dict)
                                 and all(isinstance(v, str) for v in meta.values())):
        raise IngestError(IngestCode.BAD_HEADER, "malformed __metadata__")
    data_len = size - 8 - n
    spans, out = [], {}
    for name, entry in header.items():
        label = str(name)[:64]
        if not isinstance(entry, dict):
            raise IngestError(IngestCode.BAD_HEADER, f"tensor {label!r}: malformed entry")
        dtype, shape, offs = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise IngestError(IngestCode.BAD_HEADER, f"tensor {label!r}: unknown dtype")
        if not isinstance(shape, list) or not all(_is_int(x) and x >= 0 for x in shape):
            raise IngestError(IngestCode.BAD_HEADER, f"tensor {label!r}: malformed shape")
        if (not isinstance(offs, list) or len(offs) != 2 or not all(_is_int(x) for x in offs)
                or not 0 <= offs[0] <= offs[1] <= data_len):
            raise IngestError(IngestCode.BAD_HEADER, f"tensor {label!r}: malformed data_offsets")
        numel = 1
        for x in shape:
            numel *= x
        if offs[1] - offs[0] != numel * _DTYPE_BYTES[dtype]:
            raise IngestError(IngestCode.BAD_HEADER, f"tensor {label!r}: data_offsets do not "
                                                     "match its dtype and shape")
        spans.append((offs[0], offs[1]))
        out[name] = (dtype, tuple(shape))
    pos = 0
    for begin, end in sorted(spans):
        if begin != pos:
            raise IngestError(IngestCode.BAD_HEADER, "tensor data overlaps or leaves gaps")
        pos = end
    if pos != data_len:
        raise IngestError(IngestCode.BAD_HEADER, "tensor data does not cover the file exactly")
    return out


def _check_header(d: Path, expect: LaneDExpectation) -> None:
    """Step 7: framing, then name set, then dtypes, then shapes."""
    got = _read_header(d / WEIGHTS_FILE)
    want = expect.tensors
    if set(got) != set(want):
        raise IngestError(IngestCode.TENSOR_SET_MISMATCH,
                          f"missing={sorted(set(want) - set(got))[:5]} "
                          f"extra={[str(k)[:64] for k in sorted(set(got) - set(want))[:5]]}")
    wrong = sorted(k for k in want if got[k][0] != want[k][0])
    if wrong:
        raise IngestError(IngestCode.DTYPE_MISMATCH,
                          f"{len(wrong)} tensor(s) not in the pinned dtype, e.g. {wrong[:3]} "
                          f"is {got[wrong[0]][0]}, want {want[wrong[0]][0]}")
    wrong = sorted(k for k in want if got[k][1] != want[k][1])
    if wrong:
        raise IngestError(IngestCode.SHAPE_MISMATCH,
                          f"{len(wrong)} tensor(s) with the wrong shape, e.g. {wrong[:3]}")


def _check_finite(d: Path) -> None:
    """Step 8: every floating weight finite (the only torch step; the file is
    fully bounded by step 7 before safetensors opens it)."""
    from .checkpoint_guard import CheckpointDiverged, verify_weights_finite

    try:
        verify_weights_finite(d)
    except CheckpointDiverged as e:
        raise IngestError(IngestCode.NON_FINITE, str(e)) from e
    except (OSError, ValueError, RuntimeError) as e:
        raise IngestError(IngestCode.BAD_HEADER, f"weights unreadable: {type(e).__name__}") from e


def _check_checkpoint(d: Path, expect: LaneDExpectation) -> None:
    """Steps 5-8 over a fully fetched, re-hashed staging directory."""
    _check_code(d, expect)
    _check_config(d, expect)
    _check_header(d, expect)
    _check_finite(d)


# ── the two transports ──────────────────────────────────────────────────────


def _resolve(contract, expect: LaneDExpectation | None) -> LaneDExpectation:
    if expect is not None:
        return expect
    if contract is None:
        raise ValueError("ingest needs the entry's contract or an explicit expectation")
    return expectation_for(contract)


def _staged(dest_dir, digest: str, fill) -> Path:
    """Run ``fill(staging)`` then publish it at ``dest_dir/<digest>``; a
    failure removes the staging directory."""
    dest = Path(dest_dir)
    final = dest / digest
    staging = dest / f".{digest}.staging-{os.getpid()}"
    created = []          # directories this call creates, deepest first
    p = dest
    while not p.exists():
        created.append(p)
        p = p.parent
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        fill(staging)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        for d in created:
            try:
                d.rmdir()
            except OSError:
                break
        raise
    return final


def _refused(e: IngestError, hotkey, digest) -> None:
    log.warning("lane_d_ingest_refused %s hotkey=%s digest=%s detail=%s", e.code.value,
                str(hotkey)[:64], str(digest)[:64], e.detail)


def ingest_entry(storage: LaneDStorage, entry: dict, dest_dir: str | os.PathLike, *,
                 contract, expect: LaneDExpectation | None = None) -> Path:
    """Ingest ``entry`` (``{"hotkey", "digest", ...}``) from the vault
    adapter; returns ``dest_dir/<digest>`` holding exactly the allowlisted
    files. With ``expect`` given, ``contract`` is unused."""
    hotkey = entry.get("hotkey") if isinstance(entry, dict) else None
    digest = entry.get("digest") if isinstance(entry, dict) else None
    try:
        hk, dg = _identity(hotkey, digest)
        want = _resolve(contract, expect)
        prefix = entry_prefix(hk, dg)
        try:
            raw = storage.get(prefix + MANIFEST_OBJECT)
        except KeyError as e:
            raise IngestError(IngestCode.MISSING_MANIFEST, prefix + MANIFEST_OBJECT) from e
        if len(raw) > MAX_MANIFEST_BYTES:
            raise IngestError(IngestCode.BAD_MANIFEST, f"manifest is {len(raw)} bytes")
        files = _manifest(raw, dg)
        _allowlist(files)
        try:
            check_object_list(storage, hk, dg, {"files": files})
        except IntakeError as e:
            raise IngestError(e.code, e.detail) from e
        _budget(files, want)

        def fill(staging: Path) -> None:
            for name in sorted(files):
                meta = files[name]
                try:
                    data = storage.get(prefix + name)
                except KeyError as e:
                    raise IngestError(IngestCode.MISSING_OBJECT, prefix + name) from e
                if len(data) > meta["size"]:
                    raise IngestError(IngestCode.OVERSIZE, f"{name}: {len(data)} > {meta['size']}")
                if len(data) != meta["size"]:
                    raise IngestError(IngestCode.SIZE_MISMATCH, f"{name}: {len(data)} != {meta['size']}")
                if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                    raise IngestError(IngestCode.DIGEST_MISMATCH,
                                      f"{name} does not hash to its manifest sha256")
                (staging / name).write_bytes(data)
                del data
            _check_checkpoint(staging, want)

        final = _staged(dest_dir, dg, fill)
    except IngestError as e:
        _refused(e, hotkey, digest)
        raise
    log.info("lane_d_ingest_ok hotkey=%s digest=%s transport=storage", hk, dg)
    return final


def ingest_presigned(grant: dict, dest_dir: str | os.PathLike, *, contract,
                     expect: LaneDExpectation | None = None) -> Path:
    """Ingest an entry through presigned GETs only: ``grant = {"hotkey",
    "digest", "manifest_url", "files": {name: url}}``. No credential is read
    or sent. Returns ``dest_dir/<digest>``; with ``expect`` given,
    ``contract`` is unused."""
    hotkey = grant.get("hotkey") if isinstance(grant, dict) else None
    digest = grant.get("digest") if isinstance(grant, dict) else None
    try:
        hk, dg = _identity(hotkey, digest)
        want = _resolve(contract, expect)
        urls = grant.get("files")
        manifest_url = grant.get("manifest_url")
        if not isinstance(urls, dict) or not isinstance(manifest_url, str):
            raise IngestError(IngestCode.BAD_IDENTITY, "grant lacks manifest_url or files")
        try:
            raw = fetch_presigned_bytes(manifest_url, max_size=MAX_MANIFEST_BYTES)
        except FetchError as e:
            if e.code == "oversize":
                raise IngestError(IngestCode.BAD_MANIFEST, e.detail) from e
            raise IngestError(IngestCode.FETCH_FAILED, f"{MANIFEST_OBJECT}: {e.detail}") from e
        files = _manifest(raw, dg)
        _allowlist(files)
        missing = sorted(n for n in files if not isinstance(urls.get(n), str))
        if missing:
            raise IngestError(IngestCode.MISSING_OBJECT, f"no presigned URL for {missing}")
        _budget(files, want)

        def fill(staging: Path) -> None:
            for name in sorted(files):
                meta = files[name]
                try:
                    fetch_presigned(urls[name], staging / name, sha256=meta["sha256"],
                                    size=meta["size"])
                except FetchError as e:
                    raise IngestError(e.code, f"{name}: {e.detail}") from e
            _check_checkpoint(staging, want)

        final = _staged(dest_dir, dg, fill)
    except IngestError as e:
        _refused(e, hotkey, digest)
        raise
    log.info("lane_d_ingest_ok hotkey=%s digest=%s transport=presigned", hk, dg)
    return final


__all__ = [
    "CODE_FILES", "INGEST_CODES", "IngestCode", "IngestError", "LaneDExpectation",
    "MAX_MANIFEST_BYTES", "expectation_for", "ingest_entry", "ingest_presigned",
]
