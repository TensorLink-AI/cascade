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
``gate_weights`` keys, rev6.18 F4's per-half / per-source / per-domain
point gains (:func:`_check_halves`: the public half present exactly at
``public_weight > 0``, a clear needs ``lcb >= d_margin`` and both halves'
gains > 0) and the public rows' coverage), and ``rows_sha256`` is the
canonical digest of ``rows`` and ``public.rows_sha256`` of ``public.rows``
(checked last). The
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

# rev6.11 (b): no confirmation stage, so no ``confirm_pending`` status or
# reason and no ``lost_confirmation``; a bench_hold candidate is ``hold``.
ENTRY_STATUSES = ("pending", "hold", "crowned", "lost", "closed")
SKIP_REASONS = ("pending_floor", "coldkey_cap_settlement", "coldkey_unresolved",
                "coldkey_changed", "king_unscored", "bench_hold", "deregistered",
                "ready_expired", "reveal_delay_exceeded", "duplicate_digest",
                "hotkey_has_open_entry", "context_not_armed", "checkpoint_fault",
                "mirror_lag_exceeded", "lane_closed", "context_below_armed",
                "before_eval_delay", "boundary_ts_missing")
# ``gate_regression``: a judged loss on the §1.3b gate, so an entry reason
# and never a skip reason. rev6.18 A7: ``lowfreq_regression`` is gone.
ENTRY_REASONS = SKIP_REASONS + ("lost_duel", "gate_regression", "crowned")
BURN_REASONS = ("house_king", "deregistered")
# The §1.3b paired gate's outcomes and references: W3b2's
# ``lane_d_bench.OUTCOMES`` / ``REFS`` (stated here, not imported, to keep this
# module's imports light; the acceptance tests pin the equality).
GATE_OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
GATE_REFS = ("king_d", "house")
# rev6.11 (b): a crown is decided at the first verdict; ``cleared`` is a
# first clear that was not crowned (held by the gate, or a passing non-best
# candidate).
CONFIRMATION_STAGES = ("first",)
VERDICT_OUTCOMES = ("lost", "cleared", "crowned")
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


def _gain_close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def _check_halves(report: dict) -> None:
    """rev6.18 A3/A8 (F4): per verdict (and its house judgement) the public
    half is present iff ``params.public_weight > 0``; ``sources.live`` is
    the live half; the public sources' and the domains' window counts add up
    to their halves'; a clear has ``lcb >= d_margin`` and every half's gain
    > 0. The ``public`` block is present iff ``public_weight > 0`` and its
    rows name only kept public windows, each judged digest on every one."""
    w = float(report["params"]["public_weight"])
    dm = float(report["params"]["d_margin"])
    if (w > 0.0) != ("public" in report):
        raise ReportSchemaError("missing_field" if w > 0.0 else "unknown_field", "public",
                                "the public block is present iff public_weight > 0")
    for i, v in enumerate(report["verdicts"]):
        at = _join("verdicts", i)
        hv, src, dom = v["halves"], v["sources"], v["domains"]
        if ("public" in hv) != (w > 0.0):
            raise ReportSchemaError("bad_value", _join(at, "halves"),
                                    "a public half is present iff public_weight > 0")
        if v["cleared"] and not (float(v["lcb"]) >= dm
                                 and all(float(x["gain"]) > 0.0 for x in hv.values())):
            raise ReportSchemaError("bad_value", _join(at, "cleared"),
                                    "a clear needs lcb >= d_margin and a gain > 0 on every half")
        house = v.get("house")
        if house is not None and house["cleared"] and float(house["lcb"]) < dm:
            raise ReportSchemaError("bad_value", _join(_join(at, "house"), "cleared"),
                                    "a house clear needs lcb >= d_margin")
        if set(src) - {"live"} and w <= 0.0 or "live" not in src \
                or src["live"] != hv["live"]:
            raise ReportSchemaError("bad_value", _join(at, "sources"),
                                    "sources.live is the live half; public sources only at "
                                    "public_weight > 0")
        n_pub = sum(x["n"] for k, x in src.items() if k != "live")
        if w > 0.0 and n_pub != hv["public"]["n"]:
            raise ReportSchemaError("bad_value", _join(at, "sources"),
                                    "the public sources' windows are not the public half's")
        n_live_dom = sum(x["n"] for k, x in dom.items() if k.startswith("live|"))
        n_pub_dom = sum(x["n"] for k, x in dom.items() if not k.startswith("live|"))
        if n_live_dom != hv["live"]["n"] or n_pub_dom != (hv["public"]["n"] if w > 0.0 else 0):
            raise ReportSchemaError("bad_value", _join(at, "domains"),
                                    "the domains' windows are not their halves'")
    if w <= 0.0:
        return
    kept = []
    for s_ in report["public"]["sources"]:
        gone = {d["window_id"] for d in s_["dropped"]}
        kept += [x["window_id"] for x in s_["windows"] if x["window_id"] not in gone]
    kept.sort()
    rows = report["public"]["rows"]
    digests = sorted({r["digest"] for r in rows})
    want = [(wid, dg) for wid in kept for dg in digests]
    if [(r["window_id"], r["digest"]) for r in rows] != want:
        raise ReportSchemaError("bad_value", "public.rows",
                                "not every scored digest on every kept public window, in "
                                "window_id then digest order")
    judged = set()
    for c in report["cohorts"]:
        judged |= {c["king_digest"], *c["members"],
                   *([c["house_digest"]] if "house_digest" in c else [])}
    if set(digests) != judged:
        raise ReportSchemaError("bad_value", "public.rows",
                                "the public rows' digests are not the judged cohorts' models")


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


DRAW_SEED_TAG = "lane_d_windows"
_SUBSCORE_RTOL = 1e-9


def _check_judging_start(report: dict) -> None:
    """rev6.14 (a): every cohort was frozen at a boundary stamped at or after
    its judging start ``t_max + 3600 * eval_delay_hours`` (so never without
    ``boundary_ts``); a ``before_eval_delay`` skip names an entry whose start
    is still after ``boundary_ts``; ``boundary_ts_missing`` only without it."""
    delay = 3600.0 * float(report["params"]["eval_delay_hours"])
    if float(report["params"]["forward_margin_hours"]) > float(report["params"]["eval_delay_hours"]):
        raise ReportSchemaError("bad_value", "params.forward_margin_hours",
                                "above params.eval_delay_hours (b > j)")
    bts = report.get("boundary_ts")
    for i, c in enumerate(report["cohorts"]):
        if bts is None:
            raise ReportSchemaError("missing_field", "boundary_ts",
                                    "a cohort was frozen without the boundary timestamp")
        if bts < int(c["t_max"]) + delay:
            raise ReportSchemaError("bad_value", _join("cohorts", i),
                                    "frozen before its judging start (eval_delay_hours)")
    reveal = {e["digest"]: int(e["reveal_ts"]) for e in report["entries"]}
    king_t = int(report["king_before"]["reveal_ts"])
    for i, sk in enumerate(report["skipped"]):
        at = _join(_join("skipped", i), "reason")
        if sk["reason"] == "boundary_ts_missing" and bts is not None:
            raise ReportSchemaError("bad_value", at, "boundary_ts is present")
        if sk["reason"] == "before_eval_delay":
            t = reveal.get(sk["digest"])
            if bts is None or t is None or bts >= max(t, king_t) + delay:
                raise ReportSchemaError("bad_value", at, "the judging start has passed")


def _check_below_armed(report: dict) -> None:
    """rev6.15: a ``context_below_armed`` skip (or entry reason) needs the
    signed armed set, and no armed variant may be <= the entry's declared
    context."""
    below = {s["digest"] for s in report["skipped"] if s["reason"] == "context_below_armed"}
    below |= {e["digest"] for e in report["entries"] if e["reason"] == "context_below_armed"}
    if not below:
        return
    if "armed_contexts" not in report:
        raise ReportSchemaError("bad_value", "skipped",
                                "context_below_armed without the signed armed set")
    armed = [int(c) for c in report["armed_contexts"]]
    for i, e in enumerate(report["entries"]):
        if e["digest"] in below and any(c <= int(e["context"]) for c in armed):
            raise ReportSchemaError("bad_value", _join(_join("entries", i), "reason"),
                                    "an armed variant is <= the declared context")


CHANNEL_FIELDS = ("series_id", "n_channels_raw", "empty_channels", "n_channels",
                  "channels", "channels_sha256")


def _check_window_channels(report: dict) -> None:
    """rev6.18 A4 (F2): a window carries all of :data:`CHANNEL_FIELDS` or
    none; its dropped channels are distinct, ascending and inside
    ``n_channels_raw``, and it scores at most the channels left."""
    for i, w in enumerate(report["windows"]):
        at = _join("windows", i)
        have = [k for k in CHANNEL_FIELDS if k in w]
        if not have:
            continue
        if len(have) != len(CHANNEL_FIELDS):
            missing = next(k for k in CHANNEL_FIELDS if k not in w)
            raise ReportSchemaError("missing_field", _join(at, missing),
                                    "a multichannel window carries every channel field")
        raw, empty = int(w["n_channels_raw"]), list(w["empty_channels"])
        if empty != sorted(set(empty)) or any(not 0 <= int(c) < raw for c in empty):
            raise ReportSchemaError("bad_value", _join(at, "empty_channels"),
                                    "not distinct ascending channel indices below n_channels_raw")
        if int(w["n_channels"]) > raw - len(empty):
            raise ReportSchemaError("bad_value", _join(at, "n_channels"),
                                    "more channels scored than are non-empty")
        chans = [int(c) for c in w["channels"]]
        if (len(chans) != int(w["n_channels"]) or chans != sorted(set(chans))
                or any(not 0 <= c < raw or c in empty for c in chans)):
            raise ReportSchemaError("bad_value", _join(at, "channels"),
                                    "not n_channels distinct ascending non-empty channel indices")
        body = json.dumps(chans, separators=(",", ":")).encode()
        if hashlib.sha256(body).hexdigest() != w["channels_sha256"]:
            raise ReportSchemaError("bad_value", _join(at, "channels_sha256"),
                                    "not the sha256 of the channels list")


def _check_windows_embargo(report: dict) -> None:
    """rev6.10 (a) / rev6.12 / W5j: each window's embargo record and draw cell
    agree with themselves, its id and the config echo."""
    cap = float(report["params"]["embargo_hours"])
    for i, w in enumerate(report["windows"]):
        at = _join("windows", i)
        e = float(w["embargo_hours"])
        if e > cap:
            raise ReportSchemaError("bad_value", _join(at, "embargo_hours"),
                                    f"above params.embargo_hours {cap}")
        if w["rule"] != ("embargo" if e > 0.0 else "embargo0"):
            raise ReportSchemaError("bad_value", _join(at, "rule"),
                                    f"rule {w['rule']!r} for an effective embargo of {e} h")
        if w["min_release_ts"] != int(w["min_first_seen_ts"]) + int(round(3600.0 * e)):
            raise ReportSchemaError("bad_value", _join(at, "min_release_ts"),
                                    "not min_first_seen_ts + 3600 * embargo_hours")
        if w["rule"] == "embargo0" and w["embargo_window"]:
            raise ReportSchemaError("bad_value", _join(at, "embargo_window"),
                                    "an embargo0 window is fresh")
        parts = str(w["window_id"]).rsplit(":", 2)
        if len(parts) == 3 and parts[2] != f"h{w['h']}":
            raise ReportSchemaError("bad_value", _join(at, "h"), "not the window id's h")
        if len(parts) == 3 and parts[1] != str(w["origin"]):
            raise ReportSchemaError("bad_value", _join(at, "origin"), "not the window id's origin")


def _check_cohort_split(report: dict) -> None:
    """rev6.12 (b): each cohort's fresh / embargo counts match its windows'
    flags and the fresh share holds; W5j: the judged windows stay within
    ``params.max_windows_per_settlement``."""
    flag = {w["window_id"]: bool(w["embargo_window"]) for w in report["windows"]}
    share = float(report["params"]["min_fresh_window_share"])
    for i, c in enumerate(report["cohorts"]):
        at = _join("cohorts", i)
        ids = [x for x in c["window_ids"] if x in flag]
        n_emb = sum(1 for x in ids if flag[x])
        if c["fresh_windows"] != len(ids) - n_emb:
            raise ReportSchemaError("bad_value", _join(at, "fresh_windows"),
                                    "not the count of its fresh windows")
        if c["embargo_windows"] != n_emb:
            raise ReportSchemaError("bad_value", _join(at, "embargo_windows"),
                                    "not the count of its embargo windows")
        if ids and c["fresh_windows"] < share * len(ids):
            raise ReportSchemaError("bad_value", _join(at, "embargo_windows"),
                                    f"the fresh share is below {share}")
    judged = {x for c in report["cohorts"] for x in c["window_ids"]}
    if len(judged) > int(report["params"]["max_windows_per_settlement"]):
        raise ReportSchemaError("bad_value", "cohorts",
                                "more judged windows than max_windows_per_settlement")


def _geo(rows: list) -> float | None:
    if not rows:
        return None
    lc = sum(math.log(max(float(r["crps"]), 1e-12)) for r in rows) / len(rows)
    lm = sum(math.log(max(float(r["mase"]), 1e-9)) for r in rows) / len(rows)
    return math.sqrt(math.exp(lc) * math.exp(lm))


def _check_subscores(report: dict) -> None:
    """rev6.12 (b): each verdict's fresh / embargo sub-score is its rows on
    its cohort's fresh / embargo windows (``geomean`` null iff n = 0)."""
    flag = {w["window_id"]: bool(w["embargo_window"]) for w in report["windows"]}
    cohorts = report["cohorts"]
    for i, v in enumerate(report["verdicts"]):
        c = cohorts[v["cohort"]] if 0 <= v["cohort"] < len(cohorts) else None
        if c is None:
            continue
        ids = set(c["window_ids"])
        mine = [r for r in report["rows"] if r["digest"] == v["digest"]
                and r["cohort"] == v["cohort"] and r["window_id"] in ids]
        for name, emb in (("fresh", False), ("embargo", True)):
            rs = [r for r in mine if flag.get(r["window_id"]) is emb]
            at = _join(_join("verdicts", i), name)
            if v[name]["n"] != len(rs):
                raise ReportSchemaError("bad_value", _join(at, "n"), "not its row count")
            want, got = _geo(rs), v[name]["geomean"]
            if (want is None) is not (got is None) or (
                    want is not None and abs(float(got) - want) > _SUBSCORE_RTOL * want):
                raise ReportSchemaError("bad_value", _join(at, "geomean"), "not its rows' geomean")


def _check_draw(report: dict) -> None:
    """W5j: the draw record is bound to its block hash, the budget and the
    report's cohorts; per cell, quota = the cohort's judged windows of that
    cell and quota <= capacity."""
    if "draw" not in report:
        return
    d = report["draw"]
    if d["seed"] != f"{d['boundary_block_hash']}:{DRAW_SEED_TAG}":
        raise ReportSchemaError("bad_value", "draw.seed", "not <boundary_block_hash>:lane_d_windows")
    if d["max_windows_per_settlement"] != report["params"]["max_windows_per_settlement"]:
        raise ReportSchemaError("bad_value", "draw.max_windows_per_settlement",
                                "not params.max_windows_per_settlement")
    cells = {w["window_id"]: f"{w['timeframe']}|{w['bin']}" for w in report["windows"]}
    by_index = {c["index"]: c for c in report["cohorts"]}
    seen: set[int] = set()
    for i, dc in enumerate(d["cohorts"]):
        at = _join(_join("draw", "cohorts"), i)
        c = by_index.get(dc["index"])
        if c is None or dc["index"] in seen:
            raise ReportSchemaError("bad_value", _join(at, "index"), "not a cohort of the report")
        seen.add(dc["index"])
        count: dict[str, int] = {}
        for x in c["window_ids"]:
            k = cells.get(x)
            if k is not None:
                count[k] = count.get(k, 0) + 1
        for k in sorted(set(dc["quota"]) | set(count)):
            q = dc["quota"].get(k, 0)
            if q > dc["capacity"].get(k, 0) or q != count.get(k, 0):
                raise ReportSchemaError("bad_value", _join(_join(at, "quota"), k),
                                        "not the cohort's drawn windows of that cell, or above "
                                        "its capacity")


PUBLIC_SEED_TAG = "lane_d_public"
_BLOCK_HASH_RE = re.compile(r"0x[0-9a-f]{64}")


def _check_public(report: dict) -> None:
    """rev6.18 A1/A2/A6 (F3, W5k): each public source once, sorted; its seed
    is ``<block hash>:lane_d_public:<source>`` on the settlement's block hash
    (the draw's / gate's when present); per cell the drawn windows are at
    most the quota and the quota at most the capacity, the quotas sum to at
    most ``n_windows``; window ids are unique, sorted, of the form
    ``<source>:<series_id>:<cut>:h<h>`` with ``h = horizon`` and ``min_history
    <= cut <= length - h``; a series' targets never overlap; every dropped
    window is a drawn one, listed once."""
    if "public" not in report:
        return
    bhs = {d.get("boundary_block_hash") for d in (report.get("draw"),) if isinstance(d, dict)}
    bhs |= {g.get("block_hash") for g in (report.get("gate"),) if isinstance(g, dict)}
    srcs = report["public"]["sources"]
    names = [s["source"] for s in srcs]
    if names != sorted(set(names)):
        raise ReportSchemaError("bad_value", "public.sources", "sources not unique and sorted")
    for i, s in enumerate(srcs):
        at = _join(_join("public", "sources"), i)
        src, h = s["source"], int(s["horizon"])
        bh, _, tail = s["seed"].partition(":")
        if (tail != f"{PUBLIC_SEED_TAG}:{src}" or not _BLOCK_HASH_RE.fullmatch(bh)
                or (bhs and bh not in bhs) or len(bhs) > 1):
            raise ReportSchemaError("bad_value", _join(at, "seed"),
                                    "not <boundary_block_hash>:lane_d_public:<source>")
        if sum(s["quota"].values()) > int(s["n_windows"]):
            raise ReportSchemaError("bad_value", _join(at, "quota"), "quotas exceed n_windows")
        for k, q in s["quota"].items():
            if q > s["capacity"].get(k, 0):
                raise ReportSchemaError("bad_value", _join(_join(at, "quota"), k),
                                        "above its capacity")
        count: dict[str, int] = {}
        ids = [w["window_id"] for w in s["windows"]]
        if ids != sorted(set(ids)):
            raise ReportSchemaError("bad_value", _join(at, "windows"),
                                    "window ids not unique and sorted")
        spans: dict[str, list[tuple[int, int]]] = {}
        for j, w in enumerate(s["windows"]):
            wat = _join(_join(at, "windows"), j)
            k = f"{w['family']}|{w['freq']}"
            count[k] = count.get(k, 0) + 1
            cut = int(w["cut"])
            head = f"{src}:"
            tail = f":{cut}:h{h}"
            wid = w["window_id"]
            if (w["h"] != h or not wid.startswith(head) or not wid.endswith(tail)
                    or len(wid) <= len(head) + len(tail)):
                raise ReportSchemaError("bad_value", _join(wat, "window_id"),
                                        "not <source>:<series_id>:<cut>:h<horizon>")
            if not int(s["min_history"]) <= cut <= int(w["length"]) - h:
                raise ReportSchemaError("bad_value", _join(wat, "cut"),
                                        "outside [min_history, length - h]")
            sid = wid[len(head):-len(tail)]
            if any(cut < e and a < cut + h for a, e in spans.get(sid, ())):
                raise ReportSchemaError("bad_value", _join(wat, "cut"),
                                        "target overlaps another window of its series")
            spans.setdefault(sid, []).append((cut, cut + h))
        for k, n in count.items():
            if n > s["quota"].get(k, 0):
                raise ReportSchemaError("bad_value", _join(_join(at, "quota"), k),
                                        "fewer than the cell's drawn windows")
        dropped = [d["window_id"] for d in s["dropped"]]
        if dropped != sorted(set(dropped)) or not set(dropped) <= set(ids):
            raise ReportSchemaError("bad_value", _join(at, "dropped"),
                                    "dropped windows not unique, sorted and drawn")


def _check_host(report: dict) -> None:
    """rev6.18 B7: a ``host`` record's ``identity`` hashes its other
    fields (whether its env is qualified is the audit's check: it depends on
    the boundary and the committed environment list)."""
    if "host" not in report:
        return
    from ..shared.config.lane_d import host_record_problem

    why = host_record_problem(report["host"])
    if why:
        raise ReportSchemaError("bad_value", "host", why)


def _cross_field(report: dict) -> None:
    """The rules between fields, run after every structural check passes."""
    _check_public(report)
    _check_armed(report)
    _check_host(report)
    _check_below_armed(report)
    _check_judging_start(report)
    _check_windows_embargo(report)
    _check_window_channels(report)
    _check_cohort_split(report)
    _check_subscores(report)
    _check_draw(report)
    _check_gate_weights(report)
    _check_snapshots(report)
    _check_house(report)
    _check_halves(report)
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
    if "public" in report and rows_sha256(report["public"]["rows"]) \
            != report["public"]["rows_sha256"]:
        raise ReportSchemaError("rows_sha256_mismatch", "public.rows_sha256")


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


__all__ = ["BURN_REASONS", "CONFIRMATION_STAGES", "ENTRY_REASONS",
           "ENTRY_STATUSES", "GATE_OUTCOMES", "GATE_REFS", "REPORT_SCHEMA", "SCHEMA_ERROR_CODES",
           "SCHEMA_PATH", "SIGNATURE_PATTERN", "SIGNATURE_SCHEME", "SKIP_REASONS",
           "VERDICT_OUTCOMES", "WINDOW_REFUSAL_REASONS", "ReportSchemaError", "canonical_body",
           "canonical_json", "load_schema", "pinned_report_signer", "rows_sha256",
           "sign_report", "validate_report", "verify_report"]
