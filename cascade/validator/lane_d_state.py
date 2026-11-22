"""``lane_d_state.json`` (LANE_D_PLAN §1.5 "State"; W7c).

Holds KING_D, the prior miner kings, the house, and every entry's progress
(status, typed reason, the windows it has been judged on, its confirmation
record, and ``frozen_set``: the W5b accumulating/frozen window set from
:func:`~cascade.validator.lane_d_frozen.advance_frozen_set`). Protected like
``cascade_state.json`` (snapshot before restarts, never hand-edited). A
missing file is the genesis state (the house holds KING_D) only while no
settlement has been persisted: :func:`load_prior_state` (what
``settle_lane_d`` reads) refuses a lost state (``state_missing``) or a
rolled-back one (``state_digest_mismatch``) once a report exists, and
:func:`restore_state` is the only way back (an operator backup bound to the
last receipt's posterior and to the persisted report).

W7c pins the file and the digests a settlement's receipt binds:

- The file is exactly :func:`canonical_state` of the state (sorted keys,
  compact separators, ASCII, no NaN, no trailing newline), so the sha256 of
  the file bytes is :func:`state_digest`. :func:`save_state` writes it
  atomically (temp file in the same directory, fsync, ``os.replace``).
- :func:`load_state` is strict: bad UTF-8/JSON, a repeated key at any depth,
  non-canonical bytes, an unknown schema or an unexpected digest are refused
  with a typed :class:`LaneDStateError`; nothing is ever silently repaired.
- A settlement's signed report is persisted at :func:`report_path` (next to
  the state, so it is snapshotted with it) before the state that binds it.
  ``last_report`` in the state holds the sha256 of the report's
  :func:`~cascade.validator.lane_d_report.canonical_body` (never the
  randomized sr25519 signature), so two runs give byte-identical states.
- The receipt record (:func:`make_receipt`) binds ``prior_state_sha256``
  (the state as loaded), ``posterior_state_sha256`` (the state written) and
  ``report_sha256``; :func:`verify_receipt` and :func:`verify_receipt_chain`
  check it with a typed :class:`LaneDReceiptError`.
- Every armed report binds the prior state digest inside its signed body
  (``extensions.prior_state_sha256``, :data:`PRIOR_STATE_EXTENSION`). After a
  crash between the report write and the state write, recovery adopts the
  persisted report only once :func:`check_persisted_report` accepts it
  (pinned signer ``report_unverified``; boundary and bound prior digest
  ``report_conflict``); it never re-sweeps the bench (RV7 B1).

No migration from the pre-W7c layout (``indent=1``, a signature in
``last_report``): lane D has never been armed on any deployment
(``[lane_d] from_block = 0`` refuses activation, and no validator path calls
``settle_lane_d`` yet), so no such file exists to migrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from ..shared.atomic import write_text_atomic
from . import lane_d_report

STATE_SCHEMA = "lane-d-state/1"
RECEIPT_SCHEMA = "lane-d-receipt/1"
# §1.5 typed statuses and reasons an entry can carry. One source of truth: the
# W7a report enums (``lane_d_report``, equal to the schema file), plus the
# reasons state can hold before the report schema lists them.
ENTRY_STATUSES = tuple(lane_d_report.ENTRY_STATUSES)
# ``lane_closed`` (T18, W5b): every open entry closes at ``[lane_d] until_block``.
STATE_ONLY_REASONS = ("lane_closed",)
ENTRY_REASONS = tuple(lane_d_report.ENTRY_REASONS) + tuple(
    r for r in STATE_ONLY_REASONS if r not in lane_d_report.ENTRY_REASONS)

STATE_ERROR_CODES = ("bad_json", "duplicate_key", "not_canonical", "unknown_schema",
                     "state_digest_mismatch", "stale_boundary", "report_conflict",
                     # RV6 B3: a lost or unreadable state after a settlement
                     "state_missing", "state_unreadable",
                     # RV7 B1: a persisted report recovery cannot verify
                     "report_unverified")
# RV7 B1: where an armed report binds, inside its signed body, the digest of
# the state it was settled from (W7a froze the top-level keys, so it sits in
# the open ``extensions`` map).
PRIOR_STATE_EXTENSION = "prior_state_sha256"
# A persisted report's file name under ``report_path``'s directory.
_REPORT_NAME = re.compile(r"lane_d_report_([0-9]+)\.json")
RECEIPT_ERROR_CODES = ("not_object", "unknown_schema", "missing_field", "unknown_field",
                       "bad_value", "boundary_mismatch", "report_mismatch", "state_mismatch",
                       "boundary_order", "chain_break")
# The lane D receipt record, exactly these keys (README W7c "The receipt record").
RECEIPT_FIELDS = ("schema", "boundary", "prior_state_sha256", "posterior_state_sha256",
                  "report_sha256")
_RECEIPT_DIGESTS = ("prior_state_sha256", "posterior_state_sha256", "report_sha256")
_HEX = frozenset("0123456789abcdef")


class LaneDStateError(ValueError):
    """A refused state or report file; ``code`` is in :data:`STATE_ERROR_CODES`."""

    def __init__(self, code: str, message: str):
        if code not in STATE_ERROR_CODES:
            raise ValueError(f"LaneDStateError: unknown code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


class LaneDReceiptError(ValueError):
    """A refused receipt; ``code`` is in :data:`RECEIPT_ERROR_CODES`."""

    def __init__(self, code: str, message: str):
        if code not in RECEIPT_ERROR_CODES:
            raise ValueError(f"LaneDReceiptError: unknown code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


# ── canonical bytes and the digest ──────────────────────────────────────────


def canonical_state(state: dict) -> bytes:
    """``sort_keys``, compact separators, ASCII-escaped, UTF-8; a non-finite
    number raises ``ValueError``. This is the state file's exact content."""
    return lane_d_report.canonical_json(state)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_digest(state: dict) -> str:
    """sha256 hex of :func:`canonical_state` (key order never changes it)."""
    return _sha256(canonical_state(state))


def report_digest(report: dict) -> str:
    """sha256 hex of the report's canonical body (every field but
    ``signature``): a re-signed report keeps its digest."""
    return _sha256(lane_d_report.canonical_body(report))


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


# ── strict parsing ──────────────────────────────────────────────────────────


def _parse_object(raw: bytes, where) -> dict:
    """``raw`` as a JSON object, refusing bad UTF-8, bad JSON, NaN/Infinity,
    a non-object top level (``bad_json``) and a key repeated in any object at
    any depth (``duplicate_key``, checked while parsing)."""

    def pairs_hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise LaneDStateError("duplicate_key", f"{where}: key {k!r} repeats")
            out[k] = v
        return out

    def refuse_constant(name):
        raise LaneDStateError("bad_json", f"{where}: non-finite number {name}")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise LaneDStateError("bad_json", f"{where}: not UTF-8 ({e})") from e
    try:
        obj = json.loads(text, object_pairs_hook=pairs_hook, parse_constant=refuse_constant)
    except LaneDStateError:
        raise
    except ValueError as e:
        raise LaneDStateError("bad_json", f"{where}: not JSON ({e})") from e
    if not isinstance(obj, dict):
        raise LaneDStateError("bad_json", f"{where}: not a JSON object")
    return obj


# ── the state file ──────────────────────────────────────────────────────────


def genesis(house: dict) -> dict:
    """The state before any lane D settlement: the house holds KING_D."""
    h = {"digest": str(house["digest"]), "reveal_ts": int(house["reveal_ts"])}
    return {"schema": STATE_SCHEMA, "boundary": 0, "house": dict(h),
            "king": {**h, "hotkey": None, "house": True},
            "prior_kings": [], "entries": {}}


def _read_state_file(p: Path) -> tuple[dict, bytes]:
    """The state file at ``p`` and its exact bytes, strictly: absent is
    ``state_missing``, not a readable regular file is ``state_unreadable``,
    then ``bad_json``, ``duplicate_key``, ``not_canonical``, ``unknown_schema``."""
    if not os.path.lexists(p):
        raise LaneDStateError("state_missing", f"{p}: the lane D state file is absent")
    if not p.is_file():
        raise LaneDStateError("state_unreadable", f"{p}: the lane D state path is not a "
                              "regular file")
    try:
        raw = p.read_bytes()
    except FileNotFoundError as e:
        raise LaneDStateError("state_missing", f"{p}: the lane D state file is absent") from e
    except OSError as e:
        raise LaneDStateError("state_unreadable", f"{p}: cannot read the lane D state "
                              f"({e})") from e
    state = _parse_object(raw, p)
    if canonical_state(state) != raw:
        raise LaneDStateError("not_canonical", f"{p}: the bytes are not the canonical "
                              "serialisation of the state (re-indented or hand-edited)")
    if state.get("schema") != STATE_SCHEMA:
        raise LaneDStateError("unknown_schema",
                              f"{p}: lane D state schema {state.get('schema')!r} "
                              f"is not {STATE_SCHEMA}")
    return state, raw


def load_state(path: str | os.PathLike, house: dict, *, expect_sha256: str | None = None
               ) -> dict:
    """The state at ``path`` (``genesis(house)`` when nothing is at ``path``),
    refused with a :class:`LaneDStateError` unless the bytes are exactly the
    canonical serialisation of a ``STATE_SCHEMA`` object with no repeated
    key. A path that exists but cannot be read as a file is
    ``state_unreadable``, never genesis. With ``expect_sha256`` the file's
    digest (the genesis digest for a missing file) must equal it
    (``state_digest_mismatch``). The lost-state check (a missing file after a
    persisted settlement) is :func:`load_prior_state`'s."""
    p = Path(path)
    if not os.path.lexists(p):
        state = genesis(house)
        raw = canonical_state(state)
    else:
        state, raw = _read_state_file(p)
    if expect_sha256 is not None and _sha256(raw) != expect_sha256:
        raise LaneDStateError("state_digest_mismatch",
                              f"{p}: state digest {_sha256(raw)} != expected {expect_sha256!r}")
    return state


def persisted_report_boundaries(state_path: str | os.PathLike) -> list[int]:
    """The boundaries of every persisted report (a ``lane_d_report_<int>.json``
    entry in :func:`report_path`'s directory), ascending. A missing or empty
    directory is none; a reports path that is not a listable directory is
    ``state_unreadable`` (fail closed: it may hide a settlement)."""
    d = report_path(state_path, 0).parent
    if not os.path.lexists(d):
        return []
    if not d.is_dir():
        raise LaneDStateError("state_unreadable", f"{d}: the lane D reports path is not a "
                              "directory")
    try:
        names = sorted(os.listdir(d))
    except OSError as e:
        raise LaneDStateError("state_unreadable", f"{d}: cannot list the lane D reports "
                              f"({e})") from e
    return sorted({int(m.group(1)) for m in map(_REPORT_NAME.fullmatch, names) if m})


def load_prior_state(state_path: str | os.PathLike, house: dict, boundary: int, *,
                     expect_prior_sha256: str | None = None) -> dict:
    """The prior state of the settlement at ``boundary``, failing closed on a
    lost or rolled-back state (README W7c "Lost state fails closed"), in order:

    1. the state file, strictly (:func:`load_state`'s codes); when nothing is
       at the path it is genesis only if no report is persisted at a boundary
       other than ``boundary`` (a first settlement, or its recovery), else
       ``state_missing``, whatever ``expect_prior_sha256`` says;
    2. a report persisted above the state's boundary, other than the one this
       call recovers, is a rolled-back state: ``state_digest_mismatch``;
    3. with ``expect_prior_sha256``, the loaded state's digest must equal it
       (``state_digest_mismatch``).

    Nothing is written."""
    p = Path(state_path)
    b = _boundary_int(boundary)
    reports = persisted_report_boundaries(p)
    if not os.path.lexists(p):
        others = [r for r in reports if r != b]
        if others:
            raise LaneDStateError("state_missing", f"{p}: the lane D state file is absent "
                                  f"but settlements are persisted at {others}; restore it "
                                  "from a backup (restore_state), never genesis")
        state = genesis(house)
    else:
        state, _ = _read_state_file(p)
        settled = state.get("boundary")
        if isinstance(settled, bool) or not isinstance(settled, int):
            raise LaneDStateError("state_digest_mismatch", f"{p}: the state boundary "
                                  f"{settled!r} is not an int")
        ahead = [r for r in reports if r > settled and r != b]
        if ahead:
            raise LaneDStateError("state_digest_mismatch", f"{p}: the state is at boundary "
                                  f"{settled} but settlements are persisted at {ahead} "
                                  "(a rolled-back state)")
    got = state_digest(state)
    if expect_prior_sha256 is not None and got != expect_prior_sha256:
        raise LaneDStateError("state_digest_mismatch", f"{p}: state digest {got} != the "
                              f"expected prior {expect_prior_sha256!r}")
    return state


def restore_state(state_path: str | os.PathLike, backup_path: str | os.PathLike, *,
                  expect_sha256: str) -> dict:
    """Install an operator backup of the state at ``state_path`` (the only way
    back from ``state_missing``). In order:

    1. the backup is read strictly (``state_missing`` if absent,
       ``state_unreadable``, ``bad_json``, ``duplicate_key``,
       ``not_canonical``, ``unknown_schema``);
    2. the sha256 of its bytes equals ``expect_sha256`` (the last published
       receipt's ``posterior_state_sha256``), else ``state_digest_mismatch``;
    3. at a boundary > 0, its ``last_report.report_sha256`` equals the body
       digest of the report persisted at that boundary (which must exist),
       else ``report_conflict``.

    Then the backup bytes are written atomically, byte for byte, to
    ``state_path`` and the parsed state is returned. On refusal nothing is
    written; persisted reports are never touched."""
    bp = Path(backup_path)
    state, raw = _read_state_file(bp)
    got = _sha256(raw)
    if got != expect_sha256:
        raise LaneDStateError("state_digest_mismatch", f"{bp}: backup digest {got} != "
                              f"expected {expect_sha256!r}")
    b = state.get("boundary")
    if isinstance(b, bool) or not isinstance(b, int) or b < 0:
        raise LaneDStateError("report_conflict", f"{bp}: the backup's boundary {b!r} is not "
                              "a non-negative int")
    if b > 0:
        rp = report_path(state_path, b)
        last = state.get("last_report")
        want = last.get("report_sha256") if isinstance(last, dict) else None
        if not _is_hex64(want) or last.get("boundary") != b:
            raise LaneDStateError("report_conflict", f"{bp}: the backup has no well-formed "
                                  f"last_report at boundary {b}")
        if not rp.is_file():
            raise LaneDStateError("report_conflict", f"{rp}: no report is persisted at the "
                                  f"backup's boundary {b}")
        try:
            report = load_report_file(rp)
            body = report_digest(report)
        except (LaneDStateError, OSError, TypeError, ValueError) as e:
            raise LaneDStateError("report_conflict", f"{rp}: the persisted report cannot be "
                                  f"read ({e})") from e
        if report.get("boundary") != b or body != want:
            raise LaneDStateError("report_conflict", f"{rp}: the persisted report (body "
                                  f"sha256 {body}) is not the one the backup bound ({want})")
    # The bytes are canonical, hence ASCII: the encode is byte for byte.
    write_text_atomic(state_path, raw.decode("ascii"))
    return state


def save_state(path: str | os.PathLike, state: dict) -> None:
    """Write :func:`canonical_state` atomically (temp + fsync + ``os.replace``)."""
    write_text_atomic(path, canonical_state(state).decode("ascii"))


def refresh_house(state: dict, house: dict) -> None:
    """§1.4: the house follows lane A's king digest. While the house holds
    KING_D the throne moves with it; a miner KING_D keeps its throne."""
    h = {"digest": str(house["digest"]), "reveal_ts": int(house["reveal_ts"])}
    if state["house"]["digest"] == h["digest"]:
        return
    state["house"] = dict(h)
    if state["king"].get("house"):
        state["king"] = {**h, "hotkey": None, "house": True}


# ── the persisted report ────────────────────────────────────────────────────


def _boundary_int(boundary) -> int:
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise ValueError(f"lane D boundary {boundary!r} is not a non-negative int")
    return boundary


def report_path(state_path: str | os.PathLike, boundary: int) -> Path:
    """Where the signed report of the settlement at ``boundary`` is persisted:
    ``<state dir>/<state stem>_reports/lane_d_report_<boundary>.json``, under
    the state's directory so it is snapshotted with the state."""
    p = Path(state_path)
    b = _boundary_int(boundary)
    return p.parent / f"{p.stem}_reports" / f"lane_d_report_{b}.json"


def save_report_file(path: str | os.PathLike, report: dict) -> None:
    """Write the signed report's canonical JSON atomically."""
    write_text_atomic(path, lane_d_report.canonical_json(report).decode("ascii"))


def load_report_file(path: str | os.PathLike) -> dict:
    """A persisted report, parsed strictly (``bad_json``, ``duplicate_key`` at
    any depth); any layout of a clean report loads equal to it."""
    p = Path(path)
    return _parse_object(p.read_bytes(), p)


def bound_prior_state(report: dict) -> str | None:
    """The prior state digest bound in ``report``'s signed body
    (``extensions.prior_state_sha256``), or ``None`` if it carries none."""
    ext = report.get("extensions") if isinstance(report, dict) else None
    got = ext.get(PRIOR_STATE_EXTENSION) if isinstance(ext, dict) else None
    return got if _is_hex64(got) else None


def check_persisted_report(report: dict, *, signer_ss58: str, boundary: int,
                           prior_state_sha256: str, where) -> None:
    """RV7 B1: may recovery adopt ``report`` (persisted at ``where``, already
    parsed strictly by :func:`load_report_file`) for the settlement at
    ``boundary`` from the state whose digest is ``prior_state_sha256``? In
    order: its body is signed by ``signer_ss58`` (the settling validator's
    key; a foreign signer, a bad signature or an edit after signing is
    ``report_unverified``); its ``boundary`` is ``boundary`` and the prior
    state digest bound in its signed body is ``prior_state_sha256`` (else
    ``report_conflict``: a genuine report of another settlement); and it
    passes the full :func:`~cascade.validator.lane_d_report.verify_report`
    (a schema defect is ``report_unverified``). Reads only."""
    if not _signed_by(report, signer_ss58):
        raise LaneDStateError("report_unverified", f"{where}: the persisted report is not "
                              f"signed by the settling validator's key {signer_ss58}")
    if report.get("boundary") != _boundary_int(boundary):
        raise LaneDStateError("report_conflict", f"{where}: the persisted report's boundary "
                              f"{report.get('boundary')!r} is not {boundary}")
    bound = bound_prior_state(report)
    if bound != prior_state_sha256:
        raise LaneDStateError("report_conflict", f"{where}: the persisted report was settled "
                              f"from state {bound!r}, not the loaded state "
                              f"{prior_state_sha256}")
    if not lane_d_report.verify_report(report, signer_ss58):
        raise LaneDStateError("report_unverified", f"{where}: the persisted report does not "
                              f"verify (lane-d-report/1) against {signer_ss58}")


def _signed_by(report: dict, signer_ss58: str) -> bool:
    """``report["signature"]`` is ``signer_ss58``'s sr25519 signature over
    the report's canonical body and ``report["signer"]`` names that key
    (the signature half of ``verify_report``, without the schema)."""
    sig = report.get("signature")
    if (not isinstance(signer_ss58, str) or not signer_ss58
            or report.get("signer") != signer_ss58 or not isinstance(sig, str)
            or not re.fullmatch(lane_d_report.SIGNATURE_PATTERN, sig)):
        return False
    try:
        body = lane_d_report.canonical_body(report)
    except (TypeError, ValueError):
        return False
    from ..shared.sr25519 import verify_hotkey

    return verify_hotkey(signer_ss58, body, sig)


# ── the receipt record ──────────────────────────────────────────────────────


def _check_receipt(receipt, where: str) -> dict:
    if not isinstance(receipt, dict):
        raise LaneDReceiptError("not_object", f"{where}: not an object "
                                f"({type(receipt).__name__})")
    missing = [k for k in RECEIPT_FIELDS if k not in receipt]
    if missing:
        raise LaneDReceiptError("missing_field", f"{where}: missing {missing}")
    extra = sorted(str(k) for k in receipt if k not in RECEIPT_FIELDS)
    if extra:
        raise LaneDReceiptError("unknown_field", f"{where}: unknown fields {extra}")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise LaneDReceiptError("unknown_schema", f"{where}: receipt schema "
                                f"{receipt['schema']!r} is not {RECEIPT_SCHEMA}")
    b = receipt["boundary"]
    if isinstance(b, bool) or not isinstance(b, int) or b < 0:
        raise LaneDReceiptError("bad_value", f"{where}: boundary {b!r} is not a "
                                "non-negative int")
    for k in _RECEIPT_DIGESTS:
        if not _is_hex64(receipt[k]):
            raise LaneDReceiptError("bad_value", f"{where}: {k} {receipt[k]!r} is not "
                                    "64 lower-case hex")
    return receipt


def make_receipt(*, boundary: int, prior_state_sha256: str, posterior_state_sha256: str,
                 report_sha256: str) -> dict:
    """The lane D receipt record of one settlement (structure-checked)."""
    receipt = {"schema": RECEIPT_SCHEMA, "boundary": boundary,
               "prior_state_sha256": prior_state_sha256,
               "posterior_state_sha256": posterior_state_sha256,
               "report_sha256": report_sha256}
    return dict(_check_receipt(receipt, "make_receipt"))


def verify_receipt(receipt, *, report, state=None) -> None:
    """Refuse (``LaneDReceiptError``) unless, in order: the receipt is well
    formed; ``report["boundary"]`` is the receipt's (an old report replayed at
    a later settlement is ``boundary_mismatch``); the report's canonical body
    hashes to ``report_sha256`` (``report_mismatch``); and, when ``state`` is
    given, it hashes to ``posterior_state_sha256`` (``state_mismatch``). The
    report's signature is W7b's ``verify_report``, not checked here."""
    _check_receipt(receipt, "receipt")
    rb = report.get("boundary") if isinstance(report, dict) else None
    if isinstance(rb, bool) or not isinstance(rb, int) or rb != receipt["boundary"]:
        raise LaneDReceiptError("boundary_mismatch", f"report boundary {rb!r} != receipt "
                                f"boundary {receipt['boundary']}")
    try:
        got = report_digest(report)
    except (TypeError, ValueError) as e:
        raise LaneDReceiptError("report_mismatch", f"report body is not canonical JSON: {e}"
                                ) from e
    if got != receipt["report_sha256"]:
        raise LaneDReceiptError("report_mismatch", f"report body sha256 {got} != receipt "
                                f"report_sha256 {receipt['report_sha256']}")
    if state is not None:
        try:
            got = state_digest(state)
        except (TypeError, ValueError) as e:
            raise LaneDReceiptError("state_mismatch", f"state is not canonical JSON: {e}"
                                    ) from e
        if got != receipt["posterior_state_sha256"]:
            raise LaneDReceiptError("state_mismatch", f"state digest {got} != receipt "
                                    f"posterior {receipt['posterior_state_sha256']}")


def verify_receipt_chain(receipts) -> None:
    """Refuse (``LaneDReceiptError``) unless every receipt is well formed, the
    boundaries strictly increase (``boundary_order``) and every prior digest
    is the previous receipt's posterior (``chain_break``: a dropped receipt or
    a state file edited between two settlements)."""
    if isinstance(receipts, dict | str | bytes) or not hasattr(receipts, "__iter__"):
        raise LaneDReceiptError("not_object", "receipts must be a sequence of receipts")
    seq = list(receipts)
    for i, r in enumerate(seq):
        _check_receipt(r, f"receipt[{i}]")
    for i, (a, b) in enumerate(zip(seq, seq[1:], strict=False), start=1):
        if b["boundary"] <= a["boundary"]:
            raise LaneDReceiptError("boundary_order", f"receipt[{i}] boundary {b['boundary']} "
                                    f"does not follow {a['boundary']}")
    for i, (a, b) in enumerate(zip(seq, seq[1:], strict=False), start=1):
        if b["prior_state_sha256"] != a["posterior_state_sha256"]:
            raise LaneDReceiptError("chain_break", f"receipt[{i}] prior "
                                    f"{b['prior_state_sha256']} != receipt[{i - 1}] posterior "
                                    f"{a['posterior_state_sha256']}")


__all__ = ["ENTRY_REASONS", "ENTRY_STATUSES", "PRIOR_STATE_EXTENSION", "RECEIPT_ERROR_CODES",
           "RECEIPT_FIELDS", "RECEIPT_SCHEMA", "STATE_ERROR_CODES", "STATE_ONLY_REASONS",
           "STATE_SCHEMA", "LaneDReceiptError", "LaneDStateError", "bound_prior_state",
           "canonical_state", "check_persisted_report", "genesis",
           "load_prior_state", "load_report_file", "load_state", "make_receipt",
           "persisted_report_boundaries", "refresh_house", "report_digest", "report_path",
           "restore_state", "save_report_file", "save_state", "state_digest",
           "verify_receipt", "verify_receipt_chain"]


def main(argv: list[str] | None = None) -> int:
    """Operator CLI: ``python -P -m cascade.validator.lane_d_state restore
    --state <path> --backup <path> --expect-sha256 <hex>``. Exit 0 on a
    restore, 1 on a typed refusal (printed as ``refused: <code>: ...``)."""
    ap = argparse.ArgumentParser(prog="python -P -m cascade.validator.lane_d_state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("restore", help="install a backup bound to the last receipt")
    rs.add_argument("--state", required=True, help="the live lane_d_state.json path")
    rs.add_argument("--backup", required=True, help="the operator's backup of the state")
    rs.add_argument("--expect-sha256", required=True,
                    help="the last published receipt's posterior_state_sha256")
    args = ap.parse_args(argv)
    try:
        state = restore_state(args.state, args.backup, expect_sha256=args.expect_sha256)
    except LaneDStateError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"restored {args.state} at boundary {state['boundary']} "
          f"(sha256 {args.expect_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
