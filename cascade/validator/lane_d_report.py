"""The signed lane D report, schema ``lane-d-report/1`` (LANE_D_PLAN §1.5,
§14.3, B7, B12, §3.2 invariant 11).

The report is a plain JSON dict. Its signature is the main validator's sr25519
signature over the canonical body (every field except ``signature``,
``sort_keys``, compact separators, no NaN), the MarketCast report pattern
(``cascade/shared/marketcast_report.py``) verified through
``cascade.shared.sr25519``. ``rows_sha256`` is the sha256 of the canonical
serialisation of ``rows`` (A9): anyone holding the rows can prove they are
the ones the verdict used.

The published contract is the JSON Schema (draft 2020-12) file
``schemas/lane_d_report_1.json``. Every object in it is closed; the one open
map is the optional top-level ``extensions``, which is never consensus.
:func:`validate_report` enforces that file with a stdlib interpreter of the
keyword subset the file uses (``type``, ``const``, ``enum``, ``pattern``,
``minLength``, ``minimum``, ``required``, ``properties``,
``additionalProperties``, ``propertyNames``, ``items``, ``$ref`` and
``if``/``then``/``else``), plus two checks JSON Schema cannot state: numbers
are finite, and ``rows_sha256`` is the canonical digest of ``rows`` (checked
last). The signature is :func:`verify_report`'s job.

Light imports only: this module sits on the validator package's import path.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import re
from pathlib import Path

REPORT_SCHEMA = "lane-d-report/1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "lane_d_report_1.json"

# ── the typed enums (each equals the schema file's ``$defs.<name>.enum``) ────

ENTRY_STATUSES = ("pending", "hold", "confirm_pending", "crowned", "lost", "closed")
SKIP_REASONS = ("pending_floor", "coldkey_cap_settlement", "coldkey_unresolved",
                "coldkey_changed", "king_unscored", "bench_hold", "deregistered",
                "ready_expired", "reveal_delay_exceeded", "duplicate_digest",
                "hotkey_has_open_entry", "context_not_armed", "checkpoint_fault",
                "mirror_lag_exceeded")
ENTRY_REASONS = SKIP_REASONS + ("confirm_pending", "lost_duel", "lost_confirmation",
                                "lowfreq_regression", "crowned")
BURN_REASONS = ("house_king", "deregistered")
CONFIRMATION_STAGES = ("first", "confirm")
CONFIRM_RULES = ("restart", "original")
LOWFREQ_RESULTS = ("pass", "fail", "no_windows", "not_candidate")
VERDICT_OUTCOMES = ("lost", "confirm_pending", "confirmed", "crowned")
SCHEMA_ERROR_CODES = ("not_object", "unknown_schema", "missing_field", "unknown_field",
                      "bad_type", "bad_value", "rows_sha256_mismatch")


def _window_refusal_reasons() -> tuple[str, ...]:
    from .lane_d_forward import REFUSAL_REASONS

    return tuple(str(r) for r in REFUSAL_REASONS)


WINDOW_REFUSAL_REASONS = _window_refusal_reasons()


class ReportSchemaError(ValueError):
    """A report outside ``lane-d-report/1``: ``code`` is one of
    ``SCHEMA_ERROR_CODES`` and ``field`` the dotted path of the offending
    field (``[i]`` for list indices)."""

    def __init__(self, code: str, field: str, detail: str = ""):
        self.code = code
        self.field = field
        super().__init__(f"{code} at {field or '<report>'}" + (f": {detail}" if detail else ""))


@functools.lru_cache(maxsize=1)
def _schema_text() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def load_schema() -> dict:
    """The parsed schema file (a fresh copy on every call)."""
    return json.loads(_schema_text())


def _join(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _is_type(value, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"lane-d-report schema uses an unsupported type {name!r}")


@functools.cache
def _pattern(p: str) -> re.Pattern:
    # ECMA-262 ``$`` never matches before a trailing newline; Python's does.
    return re.compile(p[:-1] + r"\Z" if p.endswith("$") and not p.endswith(r"\$") else p)


class _Validator:
    def __init__(self, root: dict):
        self.root = root

    def resolve(self, node: dict) -> dict:
        while "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/"):
                raise ValueError(f"lane-d-report schema has a non-local $ref {ref!r}")
            target = self.root
            for part in ref[2:].split("/"):
                target = target[part]
            node = target
        return node

    def check(self, value, node: dict, path: str) -> None:
        node = self.resolve(node)
        types = node.get("type")
        if types is not None:
            names = [types] if isinstance(types, str) else list(types)
            if not any(_is_type(value, t) for t in names):
                raise ReportSchemaError("bad_type", path, f"want {'|'.join(names)}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ReportSchemaError("bad_value", path, "non-finite number")
        if "const" in node and value != node["const"]:
            raise ReportSchemaError("bad_value", path, f"want {node['const']!r}")
        if "enum" in node and not any(_same(value, v) for v in node["enum"]):
            raise ReportSchemaError("bad_value", path, f"{value!r} not in the enum")
        if isinstance(value, str):
            if len(value) < int(node.get("minLength", 0)):
                raise ReportSchemaError("bad_value", path, "too short")
            if "pattern" in node and not _pattern(node["pattern"]).search(value):
                raise ReportSchemaError("bad_value", path, f"does not match {node['pattern']}")
        if _is_type(value, "number") and "minimum" in node and value < node["minimum"]:
            raise ReportSchemaError("bad_value", path, f"below {node['minimum']}")
        if isinstance(value, list) and "items" in node:
            for i, item in enumerate(value):
                self.check(item, node["items"], _join(path, i))
        if isinstance(value, dict):
            self.check_object(value, node, path)
        if "if" in node:
            branch = node.get("then") if self.holds(value, node["if"], path) else node.get("else")
            if branch is not None:
                try:
                    self.check(value, branch, path)
                except ReportSchemaError as e:
                    # An if/then/else branch states a consistency rule between
                    # fields: breaking it is a value defect, whatever the type.
                    raise ReportSchemaError("bad_value", e.field, str(e)) from None

    def check_object(self, value: dict, node: dict, path: str) -> None:
        props = node.get("properties") or {}
        for name in node.get("required") or ():
            if name not in value:
                raise ReportSchemaError("missing_field", _join(path, name))
        extra = node.get("additionalProperties", True)
        names = node.get("propertyNames")
        for key in value:
            if names is not None:
                self.check(key, names, _join(path, key))
            if key not in props and extra is False:
                raise ReportSchemaError("unknown_field", _join(path, key))
        for key, sub in props.items():
            if key in value:
                self.check(value[key], sub, _join(path, key))
        if isinstance(extra, dict):
            for key in value:
                if key not in props:
                    self.check(value[key], extra, _join(path, key))

    def holds(self, value, node: dict, path: str) -> bool:
        try:
            self.check(value, node, path)
        except ReportSchemaError:
            return False
        return True


def _same(a, b) -> bool:
    """JSON equality: ``True`` is not ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def validate_report(report) -> None:
    """Raise :class:`ReportSchemaError` on the first defect of ``report``
    against ``lane-d-report/1``: not an object, then the ``schema`` field,
    then every structural rule of the schema file, then ``rows_sha256``."""
    if not isinstance(report, dict):
        raise ReportSchemaError("not_object", "", type(report).__name__)
    if report.get("schema") != REPORT_SCHEMA or not isinstance(report.get("schema"), str):
        raise ReportSchemaError("unknown_schema", "schema", repr(report.get("schema")))
    schema = load_schema()
    _Validator(schema).check(report, schema, "")
    if rows_sha256(report["rows"]) != report["rows_sha256"]:
        raise ReportSchemaError("rows_sha256_mismatch", "rows_sha256")


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def rows_sha256(rows: list) -> str:
    """sha256 hex of the canonical serialisation of ``rows`` (in order)."""
    return hashlib.sha256(canonical_json(list(rows))).hexdigest()


def canonical_body(report: dict) -> bytes:
    """The signed bytes: the report without its ``signature``."""
    body = {k: v for k, v in report.items() if k != "signature"}
    return canonical_json(body)


def sign_report(report: dict, signer) -> dict:
    """A copy of ``report`` signed by ``signer`` (a keypair, or a wallet whose
    ``hotkey`` is one). ``signer`` is stamped as the report's ``signer``."""
    key = getattr(signer, "hotkey", signer)
    out = copy.deepcopy(report)
    out.pop("signature", None)
    out["signer"] = str(key.ss58_address)
    sig = key.sign(canonical_body(out))
    out["signature"] = sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return out


def verify_report(report: dict, signer_ss58: str) -> bool:
    """True iff ``report`` carries ``signer_ss58``'s valid signature over its
    canonical body. Raises ``SignatureVerifierUnavailable`` when the SDK
    cannot verify (never a silent False)."""
    from ..shared.sr25519 import verify_hotkey

    sig = report.get("signature") if isinstance(report, dict) else None
    if not sig or not signer_ss58 or report.get("signer") != signer_ss58:
        return False
    try:
        body = canonical_body(report)
    except (TypeError, ValueError):
        return False
    return verify_hotkey(signer_ss58, body, str(sig))


__all__ = ["BURN_REASONS", "CONFIRMATION_STAGES", "CONFIRM_RULES", "ENTRY_REASONS",
           "ENTRY_STATUSES", "LOWFREQ_RESULTS", "REPORT_SCHEMA", "SCHEMA_ERROR_CODES",
           "SCHEMA_PATH", "SKIP_REASONS", "VERDICT_OUTCOMES", "WINDOW_REFUSAL_REASONS",
           "ReportSchemaError", "canonical_body", "canonical_json", "load_schema", "rows_sha256",
           "sign_report", "validate_report", "verify_report"]
