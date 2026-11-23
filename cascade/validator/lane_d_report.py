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
``minLength``, ``minimum``, ``exclusiveMinimum``, ``maximum``, ``minItems``,
``maxItems``, ``required``, ``properties``, ``additionalProperties``,
``propertyNames``, ``items``, ``$ref`` and ``if``/``then``/``else``), plus the
checks JSON Schema cannot state: numbers are finite (``extensions``
included), the cross-field rules of :func:`_cross_field` (snapshot binding,
a crowned report carries a gate record (W5e, RV5) holding a ``pass`` for the
crowned digest, the gate record's consistency with the report, the
W5c house bar's ``house_digest`` / ``verdicts[].house`` binding,
``gate_weights`` keys, the W3b5 ``verdicts[].lowfreq`` LCBs present exactly
for ``pass``/``fail`` and consistent with ``tau_public``), and
``rows_sha256`` is the canonical digest of ``rows`` (checked last). The
signature is :func:`verify_report`'s job.

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

# ── the signature (versioned by ``schema``: any change is a schema bump) ─────

# schnorrkel sr25519, substrate signing context, via ``cascade.shared.sr25519``.
SIGNATURE_SCHEME = "sr25519"
# The 64 signature bytes as lower-case hex; equals the schema file's
# ``properties.signature.pattern``. One spelling per signature.
SIGNATURE_PATTERN = r"^[0-9a-f]{128}$"
_SIGNATURE_BYTES = 64
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")

# ── the typed enums (each equals the schema file's ``$defs.<name>.enum``) ────

ENTRY_STATUSES = ("pending", "hold", "confirm_pending", "crowned", "lost", "closed")
SKIP_REASONS = ("pending_floor", "coldkey_cap_settlement", "coldkey_unresolved",
                "coldkey_changed", "king_unscored", "bench_hold", "deregistered",
                "ready_expired", "reveal_delay_exceeded", "duplicate_digest",
                "hotkey_has_open_entry", "context_not_armed", "checkpoint_fault",
                "mirror_lag_exceeded", "lane_closed")
# ``gate_regression``: a judged loss on the §1.3b gate (like
# ``lowfreq_regression``), so an entry reason and never a skip reason.
ENTRY_REASONS = SKIP_REASONS + ("confirm_pending", "lost_duel", "lost_confirmation",
                                "lowfreq_regression", "gate_regression", "crowned")
BURN_REASONS = ("house_king", "deregistered")
# The §1.3b paired gate's outcomes and references: W3b2's
# ``lane_d_bench.OUTCOMES`` / ``REFS`` (stated here, not imported, to keep this
# module's imports light; the acceptance tests pin the equality).
GATE_OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
GATE_REFS = ("king_d", "house")
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
        if _is_type(value, "number"):
            if "minimum" in node and value < node["minimum"]:
                raise ReportSchemaError("bad_value", path, f"below {node['minimum']}")
            if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
                raise ReportSchemaError("bad_value", path,
                                        f"not above {node['exclusiveMinimum']}")
            if "maximum" in node and value > node["maximum"]:
                raise ReportSchemaError("bad_value", path, f"above {node['maximum']}")
        if isinstance(value, list):
            if len(value) < int(node.get("minItems", 0)):
                raise ReportSchemaError("bad_value", path, f"fewer than {node['minItems']} items")
            if "maxItems" in node and len(value) > int(node["maxItems"]):
                raise ReportSchemaError("bad_value", path, f"more than {node['maxItems']} items")
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


def _check_finite(value, path: str) -> None:
    """Every number under ``value`` (the open ``extensions`` map, which the
    schema does not describe) is finite: what validates must also sign."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportSchemaError("bad_value", path, "non-finite number")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_finite(v, _join(path, str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_finite(v, _join(path, i))


def _check_snapshots(report: dict) -> None:
    """Invariant 4 (RV4): ``pool.snapshots`` lists each snapshot once (by
    name, then by sha256), the newest (``pool.snapshot`` / ``pool.sha256``)
    among them, and every window names a listed snapshot."""
    pool = report["pool"]
    names: set[str] = set()
    shas: set[str] = set()
    for i, snap in enumerate(pool["snapshots"]):
        at = _join("pool.snapshots", i)
        if snap["snapshot"] in names:
            raise ReportSchemaError("bad_value", _join(at, "snapshot"), "snapshot listed twice")
        if snap["sha256"] in shas:
            raise ReportSchemaError("bad_value", _join(at, "sha256"), "sha256 listed twice")
        names.add(snap["snapshot"])
        shas.add(snap["sha256"])
    newest = {"snapshot": pool["snapshot"], "sha256": pool["sha256"]}
    if not any(s["snapshot"] == newest["snapshot"] and s["sha256"] == newest["sha256"]
               for s in pool["snapshots"]):
        raise ReportSchemaError("bad_value", "pool.snapshots",
                                "the newest snapshot (pool.snapshot / pool.sha256) is not listed")
    for i, w in enumerate(report["windows"]):
        if w["snapshot_sha256"] not in shas:
            raise ReportSchemaError("bad_value", _join(_join("windows", i), "snapshot_sha256"),
                                    "not one of pool.snapshots[].sha256")


def _check_gate(report: dict) -> None:
    """The §1.3b gate record agrees with its report: drawn at the report's
    boundary, the drawn suite is eligible, one result per candidate digest."""
    gate = report["gate"]
    if gate["block"] != report["boundary"]:
        raise ReportSchemaError("bad_value", "gate.block", "not the report's boundary")
    if gate["suite"] not in gate["eligible"]:
        raise ReportSchemaError("bad_value", "gate.suite", "not one of gate.eligible")
    seen: set[str] = set()
    judged = {v["digest"] for v in report["verdicts"]}
    for i, res in enumerate(gate["results"]):
        if res["digest"] in seen:
            raise ReportSchemaError("bad_value", f"gate.results[{i}].digest",
                                    "digest swept twice")
        seen.add(res["digest"])
        # RV7 B2: only a held candidate (no verdict here) names its hold,
        # an earlier settlement's. Whether every held result carries one is
        # the audit's to check (check_lane_d), with the chain behind it.
        at = f"gate.results[{i}].held_from"
        if "held_from" not in res:
            continue
        if res["digest"] in judged:
            raise ReportSchemaError("unknown_field", at,
                                    "a result with a verdict this settlement is not held")
        if res["held_from"]["boundary"] >= report["boundary"]:
            raise ReportSchemaError("bad_value", f"{at}.boundary",
                                    "the hold is not an earlier settlement")
    crowned = report["crowned"]
    if crowned is not None and not any(res["digest"] == crowned and res["outcome"] == "pass"
                                       for res in gate["results"]):
        raise ReportSchemaError("bad_value", "gate.results",
                                "no pass result for the crowned digest")


def _check_house(report: dict) -> None:
    """W5c change 3: a cohort's ``house_digest`` is the report's house and
    not its KING_D; a verdict carries ``house`` iff its cohort names a
    ``house_digest``, and then names that digest."""
    house = report["house"]["digest"]
    cohorts = report["cohorts"]
    for i, c in enumerate(cohorts):
        if "house_digest" not in c:
            continue
        at = _join(_join("cohorts", i), "house_digest")
        if c["house_digest"] != house:
            raise ReportSchemaError("bad_value", at, "not the report's house digest")
        if c["house_digest"] == c["king_digest"]:
            raise ReportSchemaError("bad_value", at, "the house holds KING_D: no house bar")
    for i, v in enumerate(report["verdicts"]):
        at = _join(_join("verdicts", i), "house")
        if not 0 <= v["cohort"] < len(cohorts):
            raise ReportSchemaError("bad_value", _join(_join("verdicts", i), "cohort"),
                                    "not a cohort index")
        bar = cohorts[v["cohort"]].get("house_digest")
        if bar is None:
            if "house" in v:
                raise ReportSchemaError("unknown_field", at, "the cohort has no house bar")
        elif "house" not in v:
            raise ReportSchemaError("missing_field", at)
        elif v["house"]["digest"] != bar:
            raise ReportSchemaError("bad_value", _join(at, "digest"),
                                    "not the cohort's house_digest")


def _check_lowfreq(report: dict) -> None:
    """W3b5 (A10): a verdict's ``lowfreq.lcb`` is present iff its result is
    ``pass`` or ``fail``; an evaluated block has at least one window and an
    empty one (``no_windows``) none; ``fail`` iff some LCB is below
    ``-tau_public``; and a ``fail`` is a cleared confirmation that is
    ``lost`` (never crowned or held)."""
    for i, v in enumerate(report["verdicts"]):
        at = _join(_join("verdicts", i), "lowfreq")
        lf = v["lowfreq"]
        evaluated = lf["result"] in ("pass", "fail")
        if evaluated and "lcb" not in lf:
            raise ReportSchemaError("missing_field", _join(at, "lcb"))
        if not evaluated and "lcb" in lf:
            raise ReportSchemaError("unknown_field", _join(at, "lcb"),
                                    f"result {lf['result']} carries no LCBs")
        if evaluated and lf["n_windows"] < 1:
            raise ReportSchemaError("bad_value", _join(at, "n_windows"),
                                    "an evaluated slice has at least one window")
        if lf["result"] == "no_windows" and lf["n_windows"] != 0:
            raise ReportSchemaError("bad_value", _join(at, "n_windows"), "no_windows is empty")
        if not evaluated:
            continue
        worst = min(float(lf["lcb"][r][m]) for r in GATE_REFS for m in ("crps", "mase"))
        want = "fail" if worst < -float(lf["tau_public"]) else "pass"
        if lf["result"] != want:
            raise ReportSchemaError("bad_value", _join(at, "result"),
                                    f"the worst LCB {worst} gives {want}")
        if want == "fail" and not (v["stage"] == "confirm" and v["cleared"]
                                   and v["outcome"] == "lost"):
            raise ReportSchemaError("bad_value", _join(_join("verdicts", i), "outcome"),
                                    "a lowfreq fail is a cleared confirmation that is lost")


def _check_gate_weights(report: dict) -> None:
    """``params.gate_weights`` names only ``params.gate_suites`` members."""
    suites = set(report["params"]["gate_suites"])
    for suite in report["params"]["gate_weights"]:
        if suite not in suites:
            raise ReportSchemaError("bad_value", _join("params.gate_weights", suite),
                                    "not one of params.gate_suites")


def _check_armed(report: dict) -> None:
    """rev6.6: ``armed_contexts`` and ``armed_evidence`` come together; the
    evidence covers ``params.contexts`` exactly (ascending), states
    ``params.min_long_series``, each variant's ``armed`` is its two gates and
    the armed set is the armed variants, ascending."""
    has_set, has_ev = "armed_contexts" in report, "armed_evidence" in report
    if not has_set and not has_ev:
        return
    if not has_set:
        raise ReportSchemaError("missing_field", "armed_contexts", "armed_evidence without it")
    if not has_ev:
        raise ReportSchemaError("missing_field", "armed_evidence", "armed_contexts without it")
    ev, params = report["armed_evidence"], report["params"]
    if ev["min_long_series"] != params["min_long_series"]:
        raise ReportSchemaError("bad_value", "armed_evidence.min_long_series",
                                "not params.min_long_series")
    contexts = sorted(params["contexts"])
    if [v["context"] for v in ev["variants"]] != contexts:
        raise ReportSchemaError("bad_value", "armed_evidence.variants",
                                f"must cover params.contexts {contexts} ascending")
    armed = []
    for i, v in enumerate(ev["variants"]):
        gates = (v["series"] >= ev["min_long_series"] and v["sources"] >= ev["min_sources"]
                 and v["t7_fits"])
        if v["armed"] is not gates:
            raise ReportSchemaError("bad_value", _join(_join("armed_evidence.variants", i), "armed"),
                                    "disagrees with its gates")
        if gates:
            armed.append(v["context"])
    if report["armed_contexts"] != armed:
        raise ReportSchemaError("bad_value", "armed_contexts",
                                f"not the armed variants {armed} of armed_evidence")


def _cross_field(report: dict) -> None:
    """The rules between fields, run after every structural check passes."""
    _check_armed(report)
    _check_gate_weights(report)
    _check_snapshots(report)
    _check_house(report)
    _check_lowfreq(report)
    if report["crowned"] is not None and not isinstance(report.get("gate"), dict):
        # W5e (RV5): nothing crowns ungated (the schema's top-level
        # conditional states the same rule).
        raise ReportSchemaError("missing_field", "gate", "a crowned report needs its gate record")
    if "gate" in report:
        _check_gate(report)


def validate_report(report) -> None:
    """Raise :class:`ReportSchemaError` on the first defect of ``report``
    against ``lane-d-report/1``: not an object, then the ``schema`` field,
    then every structural rule of the schema file, then finite numbers in
    ``extensions``, then the cross-field rules, then ``rows_sha256``."""
    if not isinstance(report, dict):
        raise ReportSchemaError("not_object", "", type(report).__name__)
    if report.get("schema") != REPORT_SCHEMA or not isinstance(report.get("schema"), str):
        raise ReportSchemaError("unknown_schema", "schema", repr(report.get("schema")))
    schema = load_schema()
    _Validator(schema).check(report, schema, "")
    if "extensions" in report:
        _check_finite(report["extensions"], "extensions")
    _cross_field(report)
    if rows_sha256(report["rows"]) != report["rows_sha256"]:
        raise ReportSchemaError("rows_sha256_mismatch", "rows_sha256")


def canonical_json(obj) -> bytes:
    """``sort_keys``, compact separators, ASCII-escaped, no NaN, UTF-8.
    A non-finite number raises ``ValueError``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def rows_sha256(rows: list) -> str:
    """sha256 hex of the canonical serialisation of ``rows`` (in order)."""
    return hashlib.sha256(canonical_json(list(rows))).hexdigest()


def canonical_body(report: dict) -> bytes:
    """The signed bytes: the report without its ``signature`` (``signer`` and
    the optional ``extensions`` included), as :func:`canonical_json`. No
    domain prefix: ``schema`` is inside the body. A non-finite number raises
    ``ValueError``."""
    body = {k: v for k, v in report.items() if k != "signature"}
    return canonical_json(body)


def _signature_hex(sig) -> str:
    """``sig`` (the keypair's output) in :data:`SIGNATURE_PATTERN` form."""
    if isinstance(sig, (bytes, bytearray)):
        raw = bytes(sig)
    else:
        text = str(sig)
        raw = bytes.fromhex(text[2:] if text[:2] in ("0x", "0X") else text)
    out = raw.hex()
    if not _SIGNATURE_RE.fullmatch(out):
        raise ValueError(f"sign_report: the signer produced {len(raw)} signature bytes, "
                         f"want {_SIGNATURE_BYTES} ({SIGNATURE_SCHEME})")
    return out


def sign_report(report: dict, signer) -> dict:
    """A copy of ``report`` signed by ``signer`` (a keypair, or a wallet whose
    ``hotkey`` is one). The input is never mutated and any old ``signature``
    is dropped; ``signer`` is stamped with the key's ss58 (so the stated
    signer is itself signed) and the signature is stored in
    :data:`SIGNATURE_PATTERN` form."""
    key = getattr(signer, "hotkey", signer)
    out = copy.deepcopy(report)
    out.pop("signature", None)
    out["signer"] = str(key.ss58_address)
    out["signature"] = _signature_hex(key.sign(canonical_body(out)))
    return out


def verify_report(report, signer_ss58: str) -> bool:
    """True iff all hold, in this order: ``report`` passes
    :func:`validate_report` (schema, signature encoding, ``rows_sha256``);
    ``signer_ss58`` is a non-empty string equal to ``report["signer"]``; and
    ``signature`` is ``signer_ss58``'s sr25519 signature over
    :func:`canonical_body`. Every other input is ``False``, never an exception
    and never a mutation.

    Validation runs first, so an invalid report is refused without reaching
    the sr25519 verifier. A valid report whose signature cannot be checked
    raises ``SignatureVerifierUnavailable`` (DEC-CA-0051: never a silent
    False)."""
    if not isinstance(report, dict) or not isinstance(signer_ss58, str) or not signer_ss58:
        return False
    try:
        validate_report(report)
    except (TypeError, ValueError):
        return False
    sig = report["signature"]
    if report["signer"] != signer_ss58 or not isinstance(sig, str) \
            or not _SIGNATURE_RE.fullmatch(sig):
        return False
    try:
        body = canonical_body(report)
    except (TypeError, ValueError):
        return False
    from ..shared.sr25519 import verify_hotkey

    return verify_hotkey(signer_ss58, body, sig)


def pinned_report_signer(cfg) -> str:
    """The only hotkey a lane D report is trusted from: ``[manifest]
    validator_hotkey`` (the main validator), stripped. A blank pin raises
    ``ValueError``; there is no fallback to the report's self-declared
    ``signer``. Callers verify with
    ``verify_report(report, pinned_report_signer(cfg))``."""
    manifest = getattr(cfg, "manifest", None)
    pin = str(getattr(manifest, "validator_hotkey", "") or "").strip()
    if not pin:
        raise ValueError("lane D report signer is unpinned: set [manifest] validator_hotkey "
                         "(the main validator hotkey); the report's own signer is never trusted")
    return pin


__all__ = ["BURN_REASONS", "CONFIRMATION_STAGES", "CONFIRM_RULES", "ENTRY_REASONS",
           "ENTRY_STATUSES", "GATE_OUTCOMES", "GATE_REFS", "LOWFREQ_RESULTS", "REPORT_SCHEMA", "SCHEMA_ERROR_CODES",
           "SCHEMA_PATH", "SIGNATURE_PATTERN", "SIGNATURE_SCHEME", "SKIP_REASONS",
           "VERDICT_OUTCOMES", "WINDOW_REFUSAL_REASONS", "ReportSchemaError", "canonical_body",
           "canonical_json", "load_schema", "pinned_report_signer", "rows_sha256",
           "sign_report", "validate_report", "verify_report"]
