"""``lane_d_state.json`` (LANE_D_PLAN §1.5 "State"; W7c).

Holds KING_D, the prior miner kings, the house, and every entry's progress
(status, typed reason, the windows it has been judged on, its confirmation
record, and ``frozen_set``: the W5b accumulating/frozen window set from
:func:`~cascade.validator.lane_d_frozen.advance_frozen_set`). Protected like
``cascade_state.json`` (snapshot before restarts, never hand-edited). A
missing file is the genesis state: the house holds KING_D.

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

No migration from the pre-W7c layout (``indent=1``, a signature in
``last_report``): lane D has never been armed on any deployment
(``[lane_d] from_block = 0`` refuses activation, and no validator path calls
``settle_lane_d`` yet), so no such file exists to migrate.
"""

from __future__ import annotations

import hashlib
import json
import os
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
                     "state_digest_mismatch", "stale_boundary", "report_conflict")
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


def load_state(path: str | os.PathLike, house: dict, *, expect_sha256: str | None = None
               ) -> dict:
    """The state at ``path`` (``genesis(house)`` when the file is missing),
    refused with a :class:`LaneDStateError` unless the bytes are exactly the
    canonical serialisation of a ``STATE_SCHEMA`` object with no repeated
    key. With ``expect_sha256`` the file's digest (the genesis digest for a
    missing file) must equal it (``state_digest_mismatch``)."""
    p = Path(path)
    if not p.exists():
        state = genesis(house)
        raw = canonical_state(state)
    else:
        raw = p.read_bytes()
        state = _parse_object(raw, p)
        if canonical_state(state) != raw:
            raise LaneDStateError("not_canonical", f"{p}: the bytes are not the canonical "
                                  "serialisation of the state (re-indented or hand-edited)")
        if state.get("schema") != STATE_SCHEMA:
            raise LaneDStateError("unknown_schema",
                                  f"{p}: lane D state schema {state.get('schema')!r} "
                                  f"is not {STATE_SCHEMA}")
    if expect_sha256 is not None and _sha256(raw) != expect_sha256:
        raise LaneDStateError("state_digest_mismatch",
                              f"{p}: state digest {_sha256(raw)} != expected {expect_sha256!r}")
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


__all__ = ["ENTRY_REASONS", "ENTRY_STATUSES", "RECEIPT_ERROR_CODES", "RECEIPT_FIELDS",
           "RECEIPT_SCHEMA", "STATE_ERROR_CODES", "STATE_ONLY_REASONS", "STATE_SCHEMA",
           "LaneDReceiptError", "LaneDStateError", "canonical_state", "genesis",
           "load_report_file", "load_state", "make_receipt", "refresh_house", "report_digest",
           "report_path", "save_report_file", "save_state", "state_digest", "verify_receipt",
           "verify_receipt_chain"]
