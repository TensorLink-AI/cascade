"""Decided on-chain votes vs the shipped ``chain.toml`` — the drift guard.

Stake-weighted activation (DEC-CA-0045, :mod:`cascade.shared.activation`)
lets the fleet DECIDE a feature's block on chain. Once decided, the block is
meant to be TYPED into the release ``chain.toml`` (the owner override): a
typed block is behaviour-identical to the resolved one and frees the
feature's segment of the 128-byte readiness note. Until 2026-11-13 that typing
happened only on the operator's deployed file and never reached ``main`` —
an external validator that pulled ``main`` overflowed its note and refused to
start (OPSLOG 2026-11-13; PR #457 typed the v2b block).

This module makes the gap visible in CI and on the box:

* :data:`DECIDED_FILE` (``decided_votes.toml`` beside ``chain.toml``) is the
  one place in git that records what the fleet decided — every vote's
  feature name, lock block and activation block, which ``chain.toml`` key
  types it, and the worker-image pins the network trains on.
* :func:`drift_report` compares a loaded config against that record and
  returns findings: a decided block the config has not typed (with the
  exact line to type), a typed block that disagrees with the decision, a
  pin that drifted, a record entry whose feature name the config no longer
  computes (changed vote params = a fresh vote; the record is stale).
* :func:`state_report` compares a node's persisted ``activation_*_state.json``
  records against the record: a vote this node resolved that the record
  lacks, or resolved to another block (the rolling-era-king case: nodes
  that restarted late adopted a different block).
* :func:`type_hint` is the one-line "type this" the resolver logs at lock-in.

``cascade-audit decided`` runs both reports; ``tests/unit/test_decided_votes.py``
runs :func:`drift_report` on the repo's own files, so a vote that resolves
and is not typed turns CI red on the next PR, whatever that PR is about.

Nothing here changes consensus: it reads config and state and reports.
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import activation as A
from .config import ChainConfig

DECIDED_FILE = "decided_votes.toml"

# Feature-name prefix → the chain.toml key (dotted section path) that types it.
# ``cascade.shared.activation`` names every feature ``<prefix>[-<params hash>]``;
# a new voted feature adds one row here and one ``*_feature`` entry in
# :func:`configured_features`.
TYPED_KEY_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("rolling-era-king", "round.rolling_from_block"),
    ("forfeit-", "scoring.forfeit_from_block"),
    ("margin-v2-", "scoring.margin_v2_from_block"),
    ("prelaunch-", "v2.prelaunch_burn_from_block"),
    ("v2b-", "v2.image_decoupled_from_block"),
    ("lad-", "eval.shared_ladder_from_block"),
)

# The image pins the network trains on; a release that moves one without
# updating the record (or the record without the release) drifts.
PIN_KEYS: tuple[str, ...] = (
    "training.train_image_digest",
    "round.funded_pod_image",
    "round.worker_code_fingerprint",
    "v2.training.train_image_digest",
)

# The activation_*_state.json files a node keeps, by feature prefix.
STATE_FILES: tuple[tuple[str, str], ...] = (
    ("rolling-era-king", "activation_state.json"),
    ("forfeit-", "activation_forfeit_state.json"),
    ("margin-v2-", "activation_margin_v2_state.json"),
    ("prelaunch-", "activation_prelaunch_burn_state.json"),
    ("v2b-", "activation_image_decoupled_state.json"),
    ("lad-", "activation_shared_ladder_state.json"),
)


def typed_key_for(feature: str) -> str:
    """The dotted ``chain.toml`` key that types ``feature``, or ``""`` when the
    code knows no such feature (a retired vote's state file, say)."""
    for prefix, key in TYPED_KEY_BY_PREFIX:
        if feature == prefix or (prefix.endswith("-") and feature.startswith(prefix)):
            return key
    return ""


def toml_line(key: str, value: Any) -> str:
    """``[section]`` + ``key = value`` for a dotted key, as it is typed."""
    section, _, name = key.rpartition(".")
    rendered = json.dumps(value) if isinstance(value, str) else str(int(value))
    return f"[{section}]\n{name} = {rendered}"


def type_hint(feature: str, block: int) -> str:
    """The one-line instruction logged at lock-in and printed by the audit."""
    key = typed_key_for(feature)
    if not key:
        return (f"{feature} decided at block {int(block)} — no chain.toml key types this "
                "feature in this release")
    return (f"DECIDED: {feature} applies at block {int(block)} — once that block has passed, "
            f"type `{key.rpartition('.')[2]} = {int(block)}` under [{key.rpartition('.')[0]}] "
            f"in the release chain.toml and record it in {DECIDED_FILE} (PR to main); the "
            "typed block is behaviour-identical and frees the feature's note segment")


def config_value(cfg: Any, key: str) -> Any:
    """Read a dotted key off the loaded config (``v2.training.x`` →
    ``cfg.training.v2_contract.x``; everything else walks attributes)."""
    if key.startswith("v2.training."):
        contract = getattr(getattr(cfg, "training", None), "v2_contract", None)
        return getattr(contract, key[len("v2.training."):], None)
    obj: Any = cfg
    for part in key.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ── the record ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecidedVote:
    feature: str
    lock_block: int
    activation_block: int
    source: str = ""
    # False = nodes resolved this vote to DIFFERENT blocks (or the loader
    # cannot take the typed value yet); the guard never asks to type it and
    # reports the known disagreement instead. ``reason`` says why.
    type_it: bool = True
    reason: str = ""

    @property
    def key(self) -> str:
        return typed_key_for(self.feature)


@dataclass(frozen=True)
class DecidedRecord:
    as_of_block: int
    votes: tuple[DecidedVote, ...] = ()
    pins: dict[str, str] = field(default_factory=dict)

    def vote(self, feature: str) -> DecidedVote | None:
        for v in self.votes:
            if v.feature == feature:
                return v
        return None


def parse_decided(text: str) -> DecidedRecord:
    raw = tomllib.loads(text)
    votes = []
    for row in raw.get("vote", ()) or ():
        votes.append(DecidedVote(
            feature=str(row["feature"]), lock_block=int(row.get("lock_block", 0) or 0),
            activation_block=int(row["activation_block"]), source=str(row.get("source", "")),
            type_it=bool(row.get("type_it", True)), reason=str(row.get("reason", ""))))
    names = [v.feature for v in votes]
    if len(set(names)) != len(names):
        raise ValueError(f"{DECIDED_FILE}: duplicate feature rows")
    for v in votes:
        if v.activation_block <= 0:
            raise ValueError(f"{DECIDED_FILE}: {v.feature}: activation_block must be > 0")
        if v.lock_block and v.lock_block > v.activation_block:
            raise ValueError(f"{DECIDED_FILE}: {v.feature}: lock_block {v.lock_block} after "
                             f"activation_block {v.activation_block}")
    pins = {str(k): str(v) for k, v in (raw.get("pins", {}) or {}).items()}
    unknown = sorted(set(pins) - set(PIN_KEYS))
    if unknown:
        raise ValueError(f"{DECIDED_FILE}: unknown pin key(s) {unknown}; known: {list(PIN_KEYS)}")
    return DecidedRecord(as_of_block=int(raw.get("as_of_block", 0) or 0),
                         votes=tuple(votes), pins=pins)


def load_decided(path: Path | str) -> DecidedRecord:
    return parse_decided(Path(path).read_text(encoding="utf-8"))


def default_decided_path(chain_toml: Path | str | None = None) -> Path:
    """``decided_votes.toml`` beside the given ``chain.toml`` (default: the
    repo's / package's own)."""
    from .config import DEFAULT_CHAIN_TOML

    shipped = Path(DEFAULT_CHAIN_TOML).resolve().parent / DECIDED_FILE
    if chain_toml is None:
        return shipped
    beside = Path(chain_toml).resolve().parent / DECIDED_FILE
    # a deployed chain.toml usually has no record beside it: the release's is
    # the one to judge it by
    return beside if beside.is_file() else shipped


# ── what the config itself says ─────────────────────────────────────────────


def configured_features(cfg: ChainConfig) -> dict[str, A.FeatureSpec]:
    """Every voted feature this config computes, by name — the names the
    record must match (changed vote params change the name: a fresh vote)."""
    out: dict[str, A.FeatureSpec] = {}
    primary = A.primary_feature(cfg)
    if primary.name:
        out[primary.name] = primary
    for fn in (A.forfeit_feature, A.margin_v2_feature, A.prelaunch_burn_feature,
               A.image_decoupled_feature, A.shared_ladder_feature):
        spec = fn(cfg)
        if spec is not None and spec.name:
            out[spec.name] = spec
    return out


# ── findings ────────────────────────────────────────────────────────────────

TYPE = "TYPE"            # decided, applied, not typed → type it (CI FAIL)
MISMATCH = "MISMATCH"    # typed to a block other than the decided one (CI FAIL)
PIN_DRIFT = "PIN_DRIFT"  # a pin differs between config and record (CI FAIL)
STALE = "STALE"          # record names a feature the config no longer computes (CI FAIL)
PENDING = "PENDING"      # decided but not yet applied as of the record's block (info)
HELD = "HELD"            # record says type_it = false (nodes disagree / owner decision pending)
DISAGREE = "DISAGREE"    # this node resolved a vote to another block than the record (info)
UNRECORDED = "UNRECORDED"  # this node resolved a vote the record lacks (CI FAIL on the box)
UNKNOWN = "UNKNOWN"      # a state file for a feature this release does not know (info)
OK = "OK"

FAILING = frozenset({TYPE, MISMATCH, PIN_DRIFT, STALE, UNRECORDED})


@dataclass(frozen=True)
class Finding:
    kind: str
    feature: str
    key: str = ""
    expected: Any = None
    actual: Any = None
    detail: str = ""
    fix: str = ""

    @property
    def fails(self) -> bool:
        return self.kind in FAILING

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "feature": self.feature, "key": self.key,
                "expected": self.expected, "actual": self.actual, "detail": self.detail,
                "fix": self.fix}


def drift_report(cfg: ChainConfig, record: DecidedRecord, *,
                 as_of_block: int | None = None) -> list[Finding]:
    """Compare a loaded ``chain.toml`` with the decided-votes record.

    ``as_of_block`` (default: the record's) is "now": a decided block at or
    before it has APPLIED and must be typed; a later one is pending (typing
    it early would give the fleet no notice — see
    :func:`cascade.shared.activation.v2b_typed_notice_problem`)."""
    now = int(record.as_of_block if as_of_block is None else as_of_block)
    features = configured_features(cfg)
    out: list[Finding] = []
    for v in record.votes:
        key = v.key
        if not key:
            out.append(Finding(UNKNOWN, v.feature, detail="no chain.toml key types this "
                               "feature in this release; drop the row or add the mapping"))
            continue
        typed = int(config_value(cfg, key) or 0)
        if v.feature not in features:
            if typed and typed == v.activation_block:
                # typed and therefore not computed as an open vote (forfeit
                # with its hotkeys, say) — that is the goal state
                out.append(Finding(OK, v.feature, key, v.activation_block, typed))
                continue
            if typed:
                out.append(Finding(MISMATCH, v.feature, key, v.activation_block, typed,
                                   detail="typed to a block the fleet did not decide",
                                   fix=toml_line(key, v.activation_block)))
                continue
            out.append(Finding(STALE, v.feature, key, v.activation_block, typed,
                               detail="the config computes no feature of this name (vote "
                               "params changed = a fresh vote, or the vote was retired); "
                               f"configured: {sorted(features)}",
                               fix=f"update or drop the {v.feature!r} row in {DECIDED_FILE}"))
            continue
        if typed and typed != v.activation_block:
            out.append(Finding(MISMATCH, v.feature, key, v.activation_block, typed,
                               detail="typed block differs from the decided one",
                               fix=toml_line(key, v.activation_block)))
            continue
        if typed:
            out.append(Finding(OK, v.feature, key, v.activation_block, typed))
            continue
        if not v.type_it:
            out.append(Finding(HELD, v.feature, key, v.activation_block, typed,
                               detail=v.reason or "held: not to be typed (see the record)"))
            continue
        if v.activation_block > now:
            out.append(Finding(PENDING, v.feature, key, v.activation_block, typed,
                               detail=f"applies at {v.activation_block} > as-of {now}; type it "
                               "once that block has passed"))
            continue
        out.append(Finding(TYPE, v.feature, key, v.activation_block, typed,
                           detail=f"decided (lock {v.lock_block}, applied {v.activation_block} "
                           f"<= as-of {now}) but the config still votes on it",
                           fix=toml_line(key, v.activation_block)))
    recorded = {v.feature for v in record.votes}
    for name, spec in features.items():
        if name not in recorded and spec.decided_on_chain:
            out.append(Finding(PENDING, name, typed_key_for(name),
                               detail="open vote, no decision recorded"))
    for key in PIN_KEYS:
        want = record.pins.get(key)
        if want is None:
            continue
        have = config_value(cfg, key)
        have_s = "" if have is None else str(have)
        if have_s != want:
            out.append(Finding(PIN_DRIFT, "pins", key, want, have_s,
                               detail="the config's pin differs from the recorded live pin",
                               fix=toml_line(key, want)))
        else:
            out.append(Finding(OK, "pins", key, want, have_s))
    return out


def read_state_records(state_dir: Path | str) -> dict[str, A.ActivationRecord]:
    """Every ``activation_*_state.json`` under ``state_dir`` that parses, by
    file name. Reads only; a torn file is skipped (the services own repair)."""
    out: dict[str, A.ActivationRecord] = {}
    for p in sorted(Path(state_dir).glob("activation_*_state.json")) + \
            [Path(state_dir) / "activation_state.json"]:
        if not p.is_file() or p.name in out:
            continue
        try:
            rec = A.ActivationStore(p).load()
        except (A.ActivationStateUnreadable, OSError, ValueError):
            continue
        if rec.feature:
            out[p.name] = rec
    return out


def state_report(records: Iterable[A.ActivationRecord] | dict[str, A.ActivationRecord],
                 record: DecidedRecord) -> list[Finding]:
    """Compare a node's persisted lock-ins with the record: a locked vote the
    record lacks must be added; one the record names at another block is a
    known disagreement (or a stale record) and is reported, never resolved."""
    recs = list(records.values()) if isinstance(records, dict) else list(records)
    out: list[Finding] = []
    for rec in recs:
        if not rec.locked:
            continue
        key = typed_key_for(rec.feature)
        if not key:
            out.append(Finding(UNKNOWN, rec.feature, actual=rec.activation_block,
                               detail="this node holds a lock-in for a feature this release "
                               "does not know (retired vote); nothing to type"))
            continue
        known = record.vote(rec.feature)
        if known is None:
            out.append(Finding(UNRECORDED, rec.feature, key, actual=rec.activation_block,
                               detail=f"this node locked in {rec.feature} (lock "
                               f"{rec.lock_block}, applies {rec.activation_block}, via "
                               f"{rec.source}) but {DECIDED_FILE} has no row for it",
                               fix=record_row(rec)))
        elif int(known.activation_block) != int(rec.activation_block):
            out.append(Finding(DISAGREE, rec.feature, key, known.activation_block,
                               rec.activation_block,
                               detail="this node resolved the vote to a different block than "
                               "the record names — nodes disagree (or the record is stale); "
                               "do NOT type either value until the fleet agrees"))
        else:
            out.append(Finding(OK, rec.feature, key, known.activation_block,
                               rec.activation_block))
    return out


def record_row(rec: A.ActivationRecord) -> str:
    """The ``[[vote]]`` row to add to the record for a node's lock-in."""
    return (f"[[vote]]\nfeature = {json.dumps(rec.feature)}\nlock_block = {int(rec.lock_block)}\n"
            f"activation_block = {int(rec.activation_block)}\nsource = {json.dumps(rec.source)}")


def render(findings: Iterable[Finding]) -> str:
    rows = list(findings)
    lines = []
    for f in rows:
        head = f"[{f.kind}] {f.feature}"
        if f.key:
            head += f"  {f.key}"
        if f.kind != OK:
            if f.expected is not None:
                head += f"  decided={f.expected}"
            if f.actual is not None:
                head += f"  config={f.actual}"
            if f.detail:
                head += f"\n    {f.detail}"
            if f.fix:
                head += "\n    → type:\n" + "\n".join("      " + ln for ln in f.fix.splitlines())
        lines.append(head)
    bad = [f for f in rows if f.fails]
    lines.append("" if not bad else f"{len(bad)} finding(s) need typing — the release chain.toml "
                 f"and {DECIDED_FILE} via a PR to main; a deployed file at its next quiet restart.")
    if not bad:
        lines.append("decided votes and pins: in sync.")
    return "\n".join(lines)
