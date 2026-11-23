"""The signed house pointer (LANE_D_PLAN §1.4 "The house", §2 items 1-2; W5f).

The house is the checkpoint digest lane A currently treats as king. It enters
lane D from ONE signed pointer naming that digest and its T (the block time at
which the digest was published; D46, the latest commit in the house's
lineage, is computed by whoever produces the pointer and taken here as
given). This module signs, verifies and loads that pointer, and checks the
verified house record ``{"digest", "reveal_ts"}`` that
:func:`~cascade.validator.lane_d.settle_lane_d` takes.

The pointer is exactly :data:`HOUSE_FIELDS`. ``signature`` is the sr25519
signature (64 bytes as 128 lower-case hex, no ``0x``) of ``signer`` over the
canonical body: the pointer without ``signature`` as ``json.dumps(sort_keys=
True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`` in UTF-8,
the lane D report's canonical JSON, so a satellite builds and checks it on
its own. sr25519 signatures are randomised; the house a pointer verifies to
is not.

Everything fails closed with a typed :class:`LaneDHouseError`: a pointer
that cannot be read or verified is refused, never repaired and never
replaced by a default.

Light imports only: this module is on the validator package's import path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HOUSE_SCHEMA = "lane-d-house/1"
HOUSE_FIELDS = ("schema", "digest", "reveal_ts", "signer", "signature")
HOUSE_ERROR_CODES = ("unreadable", "bad_json", "duplicate_key", "not_object",
                     "unknown_schema", "missing_field", "unknown_field", "bad_value",
                     "bad_signature", "wrong_signer")
# The verified house record settle_lane_d takes.
HOUSE_RECORD_FIELDS = ("digest", "reveal_ts")

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")


class LaneDHouseError(ValueError):
    """A refused house pointer or house record; ``code`` is in
    :data:`HOUSE_ERROR_CODES`."""

    def __init__(self, code: str, message: str):
        if code not in HOUSE_ERROR_CODES:
            raise ValueError(f"LaneDHouseError: unknown code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


def canonical_body(pointer: dict) -> bytes:
    """The signed bytes: ``pointer`` without ``signature`` as canonical JSON."""
    body = {k: v for k, v in pointer.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _is_digest(value) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _is_reveal_ts(value) -> bool:
    return type(value) is int and value >= 0


def _check_keys(obj, fields: tuple[str, ...], what: str) -> None:
    if not isinstance(obj, dict):
        raise LaneDHouseError("not_object", f"the {what} is {type(obj).__name__}, "
                              "not an object")
    missing = [k for k in fields if k not in obj]
    if missing:
        raise LaneDHouseError("missing_field", f"the {what} lacks {missing}")
    extra = sorted(str(k) for k in obj if k not in fields)
    if extra:
        raise LaneDHouseError("unknown_field", f"the {what} carries unknown fields {extra}")


def _check_house_values(digest, reveal_ts, what: str) -> None:
    if not _is_digest(digest):
        raise LaneDHouseError("bad_value", f"the {what} digest {digest!r} is not 64 "
                              "lower-case hex")
    if not _is_reveal_ts(reveal_ts):
        raise LaneDHouseError("bad_value", f"the {what} reveal_ts {reveal_ts!r} is not a "
                              "non-negative int")


def check_house(house) -> dict:
    """The verified house record ``{"digest", "reveal_ts"}`` as a new dict,
    or :class:`LaneDHouseError` (``not_object``, ``missing_field``,
    ``unknown_field``, ``bad_value``). The input is never mutated."""
    _check_keys(house, HOUSE_RECORD_FIELDS, "house record")
    _check_house_values(house["digest"], house["reveal_ts"], "house record")
    return {"digest": house["digest"], "reveal_ts": house["reveal_ts"]}


def verify_house_pointer(pointer, *, signer_ss58: str) -> dict:
    """The house a signed pointer enters: ``{"digest", "reveal_ts"}``, or
    :class:`LaneDHouseError`. In order: ``not_object``, ``missing_field``,
    ``unknown_field``, ``unknown_schema``, ``bad_value`` (digest, reveal_ts,
    signer, signature encoding), ``wrong_signer`` (``signer`` is not
    ``signer_ss58``), ``bad_signature`` (no sr25519 signature of
    ``signer_ss58`` over the canonical body). The input is never mutated."""
    _check_keys(pointer, HOUSE_FIELDS, "house pointer")
    if pointer["schema"] != HOUSE_SCHEMA:
        raise LaneDHouseError("unknown_schema", f"house pointer schema "
                              f"{pointer['schema']!r} is not {HOUSE_SCHEMA}")
    _check_house_values(pointer["digest"], pointer["reveal_ts"], "house pointer")
    signer, sig = pointer["signer"], pointer["signature"]
    if not isinstance(signer, str) or not signer:
        raise LaneDHouseError("bad_value", f"the house pointer signer {signer!r} is not a "
                              "non-empty str")
    if not isinstance(sig, str) or _SIGNATURE_RE.fullmatch(sig) is None:
        raise LaneDHouseError("bad_value", "the house pointer signature is not 128 "
                              "lower-case hex")
    if signer != signer_ss58:
        raise LaneDHouseError("wrong_signer", f"the house pointer is signed by {signer!r}, "
                              f"not the pinned {signer_ss58!r}")
    from ..shared.sr25519 import verify_hotkey

    if not verify_hotkey(signer_ss58, canonical_body(pointer), sig):
        raise LaneDHouseError("bad_signature", f"the house pointer signature does not "
                              f"verify against {signer_ss58}")
    return {"digest": pointer["digest"], "reveal_ts": pointer["reveal_ts"]}


def _signature_hex(sig) -> str:
    if isinstance(sig, (bytes, bytearray)):
        raw = bytes(sig)
    else:
        text = str(sig)
        raw = bytes.fromhex(text[2:] if text[:2] in ("0x", "0X") else text)
    out = raw.hex()
    if _SIGNATURE_RE.fullmatch(out) is None:
        raise ValueError(f"sign_house_pointer: the signer produced {len(raw)} signature "
                         "bytes, want 64 (sr25519)")
    return out


def sign_house_pointer(*, digest: str, reveal_ts: int, signer) -> dict:
    """The house pointer for ``digest`` / ``reveal_ts`` signed by ``signer``
    (a keypair, or a wallet whose ``hotkey`` is one): ``signer`` is the key's
    ss58 and ``signature`` its sr25519 signature over the canonical body. A
    digest or T outside the pointer's rules is ``bad_value`` (nothing is
    signed)."""
    _check_house_values(digest, reveal_ts, "house pointer")
    key = getattr(signer, "hotkey", signer)
    pointer = {"schema": HOUSE_SCHEMA, "digest": digest, "reveal_ts": reveal_ts,
               "signer": str(key.ss58_address)}
    pointer["signature"] = _signature_hex(key.sign(canonical_body(pointer)))
    return pointer


def _parse(raw: bytes, where) -> object:
    """``raw`` as JSON, strictly: bad UTF-8, bad JSON or NaN/Infinity is
    ``bad_json``; a key repeated in any object at any depth is
    ``duplicate_key``."""

    def pairs_hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise LaneDHouseError("duplicate_key", f"{where}: key {k!r} repeats")
            out[k] = v
        return out

    def refuse_constant(name):
        raise LaneDHouseError("bad_json", f"{where}: non-finite number {name}")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise LaneDHouseError("bad_json", f"{where}: not UTF-8 ({e})") from e
    try:
        return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=refuse_constant)
    except LaneDHouseError:
        raise
    except ValueError as e:
        raise LaneDHouseError("bad_json", f"{where}: not JSON ({e})") from e


def load_house_pointer(path: str | os.PathLike, *, signer_ss58: str) -> dict:
    """The house the pointer file at ``path`` enters (see
    :func:`verify_house_pointer`), read strictly first: an absent path or one
    that is not a readable regular file is ``unreadable``; then ``bad_json``,
    ``duplicate_key``, ``not_object`` and the verify codes."""
    p = Path(path)
    if not p.is_file():
        raise LaneDHouseError("unreadable", f"{p}: the house pointer is not a readable file")
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise LaneDHouseError("unreadable", f"{p}: cannot read the house pointer ({e})") from e
    pointer = _parse(raw, p)
    if not isinstance(pointer, dict):
        raise LaneDHouseError("not_object", f"{p}: the house pointer is "
                              f"{type(pointer).__name__}, not an object")
    return verify_house_pointer(pointer, signer_ss58=signer_ss58)


__all__ = ["HOUSE_ERROR_CODES", "HOUSE_FIELDS", "HOUSE_RECORD_FIELDS", "HOUSE_SCHEMA",
           "LaneDHouseError", "canonical_body", "check_house", "load_house_pointer",
           "sign_house_pointer", "verify_house_pointer"]
