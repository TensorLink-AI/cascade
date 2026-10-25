"""Owner-signed network config (DEC-CA-0057).

Every verdict / payout constant a validator reads from its local
``chain.toml`` that is not code, a locked training term or a security root
can be changed by an OVERLAY the owner signs, instead of a fleet release:

1. **Overlay.** A signed, hash-chained body ``{protocol, seq, prev,
   effective_block, overrides}`` (``overrides`` is a TOML-shaped table of
   ``{table: {key: value}}``), published to the manifest bucket at
   ``config/network/seq-<n>.json`` and copied to ``config/network/latest.json``.
   Signed by ``[netconfig] owner_hotkey`` with the DEC-CA-0052 envelope
   (:mod:`cascade.direct.signing`, scope :data:`SCOPE`). Snapshots, not
   deltas: the live config is ``chain.toml`` + the ONE overlay in force. A
   lower ``seq`` than one seen, two bodies under one ``seq`` and a broken
   ``prev`` chain are refused.
2. **Allow-list** (:data:`ALLOWED`): ``[scoring]``, ``[eval]``, the ``[round]``
   schedule keys, ``[v2]`` runtime policy (never
   ``[v2.training]``), ``[lane_b]``, ``[lane_c]`` (of ``[lane_c.competitions]``
   only a configured competition's ``weight`` / ``until_block`` —
   :func:`lane_c_competition_changes`), and the listed ``[marketcast]``
   numeric verdict, payout and liveness controls. The DEC-CA-0045 "typed
   blocks" live in those tables (``rolling_from_block``,
   ``forfeit_from_block``, ``margin_v2_from_block``,
   ``prelaunch_burn_from_block``). Everything else is refused — ``[training]``,
   ``[manifest]``, ``[storage]``, ``[chain]``, ``[netconfig]``,
   ``[activation]`` (the tally that adopts overlays) included. The merged
   table is re-parsed by :func:`cascade.shared.config.parse_chain_config`, so
   every load check (era-boundary ``from_block``, seams, lanes) applies.
3. **Adoption.** A node that fetched and verified an overlay signals
   ``cfg-<8 hex of its digest>`` as a DEC-CA-0045 note segment
   (:func:`cfg_feature`); 51 % of permit stake locks it in (``align="era"``,
   :func:`cascade.shared.activation.resolve_activation`).
4. **When.** :func:`apply_block_for`: ``effective_block`` when the lock-in
   lands before it, else the first era start after the lock-in. An overlay
   whose changed ``*_from_block`` / ``*_activation_block`` values would lie
   before that block, or that would move the grid before it, is refused
   (:func:`retroactive_changes`) — never applied retroactively.
5. **Block-indexed.** :class:`NetConfigResolver` keeps the adopted history
   (persisted atomically) and :meth:`NetConfigResolver.config_at` returns the
   config in force at a block — identical object to the base config when no
   overlay is in force, so an empty ``owner_hotkey`` is bit-identical.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .activation import (
    ActivationRecord,
    FeatureSpec,
    ReadySignal,
    activation_block_for,
    resolve_activation,
)
from .atomic import read_text_with_backup, write_text_atomic
from .config import ChainConfig, effective_epoch_blocks, parse_chain_config
from .config.reliability import DEFAULTS as _RELIABILITY_DEFAULTS
from .direct_moe import canonical, digest
from .era import era_length_blocks

log = logging.getLogger("cascade.netconfig")

PROTOCOL = "cascade-netconfig/2"
SCOPE = "netconfig-overlay"
PREFIX = "config/network"
LATEST_KEY = f"{PREFIX}/latest.json"
FEATURE_PREFIX = "cfg-"
BODY_FIELDS = frozenset({"protocol", "seq", "prev", "effective_block", "overrides", "network"})
OPTIONAL_FIELDS = frozenset({"comment"})
# A first-sight walk of the hash chain reads every version; past this many the
# publisher is doing something wrong and the node refuses rather than loop.
MAX_CHAIN_WALK = 1000

# table -> allowed keys (None = every key of the table). DEC-CA-0057 §2.
ALLOWED: dict[str, frozenset[str] | None] = {
    "scoring": None,
    "eval": None,
    "lane_b": None,
    "lane_c": None,
    # the [round] SCHEDULE: the grid and its seam, the DEC-CA-0043 rollover,
    # the era length, the blocked-hotkey gate and the commit floor
    "round": frozenset({
        "epoch_blocks", "epoch_blocks_prev", "epoch_activation_block",
        "rolling_from_block", "era_settlements", "round_hours",
        "blocked_hotkeys", "blocked_from_block", "commit_floor_block",
        "funded_field_cap",
    }),
    # Existing v2 runtime policy; the nested [v2.training] contract is refused.
    # Sources select between ALREADY pinned trainer/main-validator identities;
    # neither identity can be replaced through an overlay.
    "v2": frozenset({
        "from_block", "bench_gate_mode", "random_init_eras", "reign_warm_start_blocks",
        "bench_gate_v2_from_block", "bench_source", "bench_source_from_block",
        "bench_report_timeout_settlements", "bench_budget_hours", "bench_sweep_hours",
        "prelaunch_burn", "prelaunch_burn_from_block", "carryover_king_share",
        "king_ladder", "win_margin_start", "win_margin_end", "margin_warmup_blocks",
        "king_check_margin", "ladder_start_tiers", "ladder_start_level_bp",
        "ladder_step_bp", "ladder_cap_bp", "ladder_fail_bp", "key_slash_bp", "court_bp", "court_blocks", "key_slash_min_eras",
    }),
    # Base, activation, suite/universe, data sources, window construction
    # and recipe bounds remain release-only. These controls change how an
    # already-supported, pinned competition is judged, paid or timed out.
    "marketcast": frozenset({
        "alpha", "bootstrap_B", "gain_margin", "full_pay_tiers", "min_windows", "min_clusters",
        "min_time_blocks", "coverage_tolerance", "coverage_tolerance_over",
        "tail_tolerance", "calibration_z", "zero_scale_frac", "cell_ratio_cap",
        "near_dup_tol", "report_timeout_settlements", "max_infra_deferrals",
    }),
}
# Recovery controls are trainer-only policy; satellites can ignore their values.

ALLOWED["round"] = ALLOWED["round"] | frozenset(_RELIABILITY_DEFAULTS)

# Release-only, with the reason the refusal prints.
REFUSED_TABLES: dict[str, str] = {
    "training": "locked training terms ship with a release",
    "manifest": "the trainer hotkey is a security root",
    "storage": "validator-read storage keys (incl. where this overlay lives) are security roots",
    "chain": "the SDK pin is consensus-adjacent",
    "netconfig": "the owner config key is a security root",
    "activation": "the tally that adopts overlays cannot be changed by one",
}
# Block-valued switch keys checked for retroactivity (§4).
_BLOCK_KEY = re.compile(r"(?:^|_)from_block(?:_v2)?$|activation_block2?$")
# The DEC-CA-0045 typed blocks: an overlay that SETS one of these makes that
# feature typed in the config it builds (the resolved flag is not carried),
# which also drops its note segment (§7).
_ROLLOVER_KEYS = (("round", "rolling_from_block"), ("scoring", "era_king_from_block"),
                  ("scoring", "tenure_blocks_from_block"))


class NetConfigError(ValueError):
    """An overlay (or the chain of overlays) that must not be adopted."""


def network_identity(cfg: ChainConfig) -> dict:
    return {"chain": cfg.netconfig.network_id, "netuid": cfg.subnet.netuid}


# ── the body ─────────────────────────────────────────────────────────────────


def overlay_policy() -> dict:
    """Machine-readable capabilities of this installed release; no I/O.

    ``*`` means existing parser-supported keys, not arbitrary new logic. All
    values still pass the loader, cross-field checks and historical guards.
    """
    return {
        "protocol": PROTOCOL,
        "overlay_keys": {table: ["*"] if keys is None else sorted(keys)
                         for table, keys in sorted(ALLOWED.items())},
        "release_only_tables": dict(sorted(REFUSED_TABLES.items())),
        "default_policy": "Keys not listed are release-only.",
        "wildcard_policy": "Existing parser-supported keys only; loader checks still apply.",
        "lane_c_competitions": "Existing competitions: weight and until_block only.",
        "adoption": "Owner signature, permit-stake lock-in, then an era boundary.",
        "operator_only": ["generation"],
        "training": "Trainer-declared recipe/budget changes need no fleet update; "
                    "locked training terms and new model code require a release.",
    }


def body_digest(body: dict) -> str:
    """sha256 of the canonical body — the overlay's identity (``prev`` links,
    the ``cfg-`` note segment and the receipt stamp all name it)."""
    return digest(body)


def cfg_feature_name(body_or_digest: dict | str) -> str:
    d = body_or_digest if isinstance(body_or_digest, str) else body_digest(body_or_digest)
    return FEATURE_PREFIX + str(d)[:8]


def cfg_feature(body: dict) -> FeatureSpec:
    """The DEC-CA-0045 feature that adopts ``body``: ``cfg-<8 hex>``, era-aligned
    (a lock-in's act block is the first era start after it)."""
    return FeatureSpec(name=cfg_feature_name(body), align="era")


def seq_key(seq: int) -> str:
    return f"{PREFIX}/seq-{int(seq)}.json"


def check_overrides(overrides: object) -> dict:
    """Refuse anything outside :data:`ALLOWED`; returns ``overrides`` unchanged."""
    if not isinstance(overrides, dict):
        raise NetConfigError("overrides must be a table of tables")
    bad: list[str] = []
    for table, kv in overrides.items():
        if table in REFUSED_TABLES:
            bad.append(f"[{table}] ({REFUSED_TABLES[table]})")
            continue
        if table not in ALLOWED:
            bad.append(f"[{table}] (not on the DEC-CA-0057 allow-list)")
            continue
        if not isinstance(kv, dict):
            bad.append(f"[{table}] must be a table")
            continue
        allowed = ALLOWED[table]
        for key, value in kv.items():
            if (table, key) in (("scoring", "burn_uid"), ("lane_b", "remainder")):
                bad.append(f"[{table}] {key} (burn destination is release-only)")
            elif isinstance(value, dict) and table in ("v2",):
                bad.append(f"[{table}.{key}] (nested tables are release-only)")
            elif allowed is not None and key not in allowed:
                bad.append(f"[{table}] {key} (not an overlay key)")
    if bad:
        raise NetConfigError("overlay refused — release-only keys: " + "; ".join(bad))
    return overrides


def check_body(body: object) -> dict:
    """Shape of an overlay body; returns it unchanged."""
    if not isinstance(body, dict):
        raise NetConfigError("overlay body must be an object")
    keys = set(body)
    if not keys >= BODY_FIELDS or keys - BODY_FIELDS - OPTIONAL_FIELDS:
        raise NetConfigError(f"overlay fields must be {sorted(BODY_FIELDS)} (+ optional "
                             f"{sorted(OPTIONAL_FIELDS)}); got {sorted(keys)}")
    if body["protocol"] != PROTOCOL:
        raise NetConfigError(f"overlay protocol {body['protocol']!r} != {PROTOCOL!r}")
    network = body["network"]
    if (not isinstance(network, dict) or set(network) != {"netuid", "chain"}
            or type(network["netuid"]) is not int or network["netuid"] < 0
            or not isinstance(network["chain"], str) or not network["chain"]):
        raise NetConfigError("overlay network must bind a chain identity and netuid")
    if type(body["seq"]) is not int or body["seq"] < 1:
        raise NetConfigError("overlay seq must be a positive integer")
    if (body["seq"] == 1) != (body["prev"] is None):
        raise NetConfigError("overlay prev must be null exactly at seq 1")
    if body["prev"] is not None and not (isinstance(body["prev"], str) and len(body["prev"]) == 64):
        raise NetConfigError("overlay prev must be a sha256 hex digest")
    if type(body["effective_block"]) is not int or body["effective_block"] <= 0:
        raise NetConfigError("overlay effective_block must be a positive integer")
    if "comment" in body and not isinstance(body["comment"], str):
        raise NetConfigError("overlay comment must be a string")
    check_overrides(body["overrides"])
    canonical(body)                     # strict JSON (no NaN) or raise
    return body


def next_overlay(overrides: dict, effective_block: int, previous: dict | None, *,
                 network: dict, comment: str = "") -> dict:
    """The next version after ``previous`` (``None`` = the first)."""
    body: dict = {
        "protocol": PROTOCOL,
        "network": dict(network),
        "seq": 1 if previous is None else int(previous["seq"]) + 1,
        "prev": None if previous is None else body_digest(previous),
        "effective_block": int(effective_block),
        "overrides": json.loads(json.dumps(overrides)),
    }
    if comment:
        body["comment"] = str(comment)
    return check_body(body)


def sign_overlay(body: dict, keypair: Any) -> dict:
    from ..direct.signing import sign

    return sign(check_body(body), keypair, scope=SCOPE)


def verify_overlay(envelope: object, *, owner: str) -> dict:
    """The verified body of ``envelope`` signed by ``owner``, else raise."""
    from ..direct.signing import verify

    if not owner:
        raise NetConfigError("no [netconfig] owner_hotkey to verify against")
    if not isinstance(envelope, dict):
        raise NetConfigError("overlay envelope must be an object")
    try:
        body = verify(envelope, expected_signer=owner, scope=SCOPE)
    except (KeyError, TypeError) as e:
        raise NetConfigError(f"malformed overlay envelope ({e})") from e
    except ValueError as e:
        raise NetConfigError(str(e)) from e
    return check_body(body)


# ── merging and validation ───────────────────────────────────────────────────


def merge_raw(base_raw: dict, overrides: dict) -> dict:
    """``base_raw`` with every override written over it (deep copies; a list
    or table VALUE replaces the base value whole)."""
    out = copy.deepcopy(base_raw)
    for table, kv in overrides.items():
        tgt = out.setdefault(table, {})
        for key, value in kv.items():
            tgt[key] = copy.deepcopy(value)
    return out


def effective_raw(cfg: ChainConfig) -> dict:
    """``cfg.raw`` with the blocks the fleet resolved on chain (DEC-CA-0045:
    rollover, forfeiture, margin v2, pre-launch burn) written in — the table
    the owner would have typed for the config this node actually runs."""
    raw = copy.deepcopy(cfg.raw or {})
    ac = cfg.activation
    if int(ac.resolved_block or 0):
        r, s = cfg.round, cfg.scoring
        raw.setdefault("round", {}).update(
            rolling_from_block=int(r.rolling_from_block), epoch_blocks=int(r.epoch_blocks),
            epoch_blocks_prev=int(r.epoch_blocks_prev),
            epoch_activation_block=int(r.epoch_activation_block))
        raw.setdefault("scoring", {}).update(
            era_king_from_block=int(s.era_king_from_block),
            tenure_blocks_from_block=int(s.tenure_blocks_from_block),
            cohort_maxt_from_block=int(s.cohort_maxt_from_block),
            cohort_maxt_increment_from_block=int(s.cohort_maxt_increment_from_block))
    if int(ac.resolved_forfeit_block or 0):
        raw.setdefault("scoring", {})["forfeit_from_block"] = int(cfg.scoring.forfeit_from_block)
    if int(ac.resolved_margin_v2_block or 0):
        raw.setdefault("scoring", {})["margin_v2_from_block"] = int(
            cfg.scoring.margin_v2_from_block)
    if int(ac.resolved_prelaunch_burn_block or 0):
        raw.setdefault("v2", {})["prelaunch_burn_from_block"] = int(
            cfg.v2.prelaunch_burn_from_block)
    return raw


def _sets(overrides: dict, table: str, key: str) -> bool:
    return key in (overrides.get(table) or {})


def _comparable(cfg: ChainConfig) -> ChainConfig:
    return replace(cfg, raw={})


def build_config(base: ChainConfig, body: dict, *, applied_block: int = 0) -> ChainConfig:
    """The config ``body`` puts in force on top of ``base`` (the node's live
    config, resolved activation blocks included), through the SAME parser and
    load checks as ``chain.toml``. Raises :class:`NetConfigError` on a refused
    key, a key no parser reads, or a merged table the loader refuses."""
    if body.get("network") != network_identity(base):
        raise NetConfigError("overlay network mismatch (chain identity or netuid)")
    ov = check_overrides(body["overrides"])
    raw = merge_raw(effective_raw(base), ov)
    try:
        cfg = parse_chain_config(raw)
    except (ValueError, KeyError, TypeError) as e:
        raise NetConfigError(f"merged config refused by the loader: {e}") from e
    # A dated launch re-parses this overlay with v2 armed when it resolves; an
    # overlay that only loads while v2 is inert would then fail every poll.
    from .calendar_launch import check_launch_loads

    head = max(int(applied_block or 0), int(body["effective_block"]))
    try:
        check_launch_loads(cfg, head)
    except ValueError as e:
        try:
            # judged from the same table the overlay was merged into
            check_launch_loads(parse_chain_config(effective_raw(base)), head)
        except (ValueError, KeyError, TypeError):
            pass    # the BASE cannot launch either: the service-start check owns that
        else:
            raise NetConfigError(f"merged config refused once the dated launch resolves: {e}") from e
    # A key the base table does not carry must change SOMETHING, or it is a
    # typo the loader would silently ignore.
    for table, kv in ov.items():
        base_tbl = (base.raw or {}).get(table) or {}
        for key in kv:
            if key in base_tbl:
                continue
            without = copy.deepcopy(raw)
            del without[table][key]
            try:
                same = _comparable(parse_chain_config(without)) == _comparable(cfg)
            except (ValueError, KeyError, TypeError):
                same = False
            if same:
                raise NetConfigError(f"[{table}] {key} has no effect (unknown key, or equal "
                                     "to its default)")
    why = lane_c_competition_changes(base, cfg)
    if why:
        raise NetConfigError("overlay refused — " + "; ".join(why))
    ac, bac = cfg.activation, base.activation
    flags = {
        "resolved_block": (0 if any(_sets(ov, t, k) for t, k in _ROLLOVER_KEYS)
                           else int(bac.resolved_block)),
        "resolved_forfeit_block": (0 if _sets(ov, "scoring", "forfeit_from_block")
                                   else int(bac.resolved_forfeit_block)),
        "resolved_margin_v2_block": (0 if _sets(ov, "scoring", "margin_v2_from_block")
                                     else int(bac.resolved_margin_v2_block)),
        "resolved_prelaunch_burn_block": (0 if _sets(ov, "v2", "prelaunch_burn_from_block")
                                          else int(bac.resolved_prelaunch_burn_block)),
    }
    return replace(
        cfg, activation=replace(ac, **flags),
        # runtime-only operator switches (--force-burn) ride on the base
        validator=base.validator,
        netconfig=replace(cfg.netconfig, seq=int(body["seq"]), digest=body_digest(body),
                          applied_block=int(applied_block)))


def lane_c_competition_changes(base: ChainConfig, new: ChainConfig) -> list[str]:
    """Concurrent lane-C competitions (owner 2026-10-20): an overlay may
    change only a CONFIGURED competition's ``weight`` / ``until_block``
    (:data:`~cascade.shared.config.lane_c.COMPETITION_OVERLAY_KEYS`) —
    exactly as it may change ``[lane_c] targets``. Adding or removing a
    competition (a new kind is code), its start (``from_block``, or the
    sibling table that owns it) and its wind-down rule are release-only.
    Returns the refusals (empty = allowed)."""
    from dataclasses import fields

    from .config.lane_c import COMPETITION_OVERLAY_KEYS, LaneCCompetition

    a = {c.kind: c for c in base.lane_c.competition_list}
    b = {c.kind: c for c in new.lane_c.competition_list}
    if set(a) != set(b):
        return [f"[lane_c] competitions {sorted(a)} -> {sorted(b)}: adding or removing a "
                "competition is release-only (a new competition kind is code)"]
    out = []
    for k in sorted(a):
        for f in fields(LaneCCompetition):
            if f.name == "kind" or f.name in COMPETITION_OVERLAY_KEYS:
                continue
            if getattr(a[k], f.name) != getattr(b[k], f.name):
                out.append(f"[lane_c.competitions.{k}] {f.name} is release-only (an overlay "
                           f"may change only {list(COMPETITION_OVERLAY_KEYS)})")
    return out


def is_era_start(cfg: ChainConfig, block: int) -> bool:
    b = int(block)
    return b > 0 and b % int(era_length_blocks(cfg.round, b)) == 0


def _block_items(raw: dict) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for table in ALLOWED:
        for key, value in (raw.get(table) or {}).items():
            if _BLOCK_KEY.search(key) and isinstance(value, int) and not isinstance(value, bool):
                out[(table, key)] = int(value)
    return out


def retroactive_changes(prev: ChainConfig, new: ChainConfig, apply_block: int) -> list[str]:
    """Why ``new`` cannot take over from ``prev`` at ``apply_block`` (empty =
    it can): a switch block whose value changed to one before ``apply_block``,
    or a grid / era length that differs from ``prev``'s anywhere before it."""
    apply_block = int(apply_block)
    out: list[str] = []
    old_b, new_b = _block_items(effective_raw(prev)), _block_items(effective_raw(new))
    for (table, key), v in sorted(new_b.items()):
        if v != old_b.get((table, key), 0) and 0 < v < apply_block:
            out.append(f"[{table}] {key} = {v} lies before the block the overlay applies "
                       f"at ({apply_block})")
    # a lane-C competition's end (concurrent competitions): never moved to,
    # or away from, a block before the apply block
    old_c = {c.kind: int(c.until_block) for c in prev.lane_c.competition_list}
    for c in new.lane_c.competition_list:
        was, now = old_c.get(c.kind, 0), int(c.until_block)
        if was != now and (0 < now < apply_block or 0 < was < apply_block):
            out.append(f"[lane_c.competitions.{c.kind}] until_block {was} -> {now} crosses a "
                       f"block before the one the overlay applies at ({apply_block})")
    probes = {0, max(0, apply_block - 1)}
    for c in (prev, new):
        act = int(c.round.epoch_activation_block or 0)
        if 0 < act < apply_block:
            probes |= {act - 1, act}
    for b in sorted(probes):
        if b >= apply_block:
            continue
        if (effective_epoch_blocks(prev.round, b) != effective_epoch_blocks(new.round, b)
                or era_length_blocks(prev.round, b) != era_length_blocks(new.round, b)):
            out.append(f"the settlement grid / era length at block {b} would change "
                       f"(before the apply block {apply_block})")
            break
    return out


def validate_overlay(base: ChainConfig, body: dict) -> ChainConfig:
    """Everything checkable before lock-in: shape, allow-list, the merged load
    checks, and ``effective_block`` an era start on both sides. Returns the
    built config (``applied_block`` = ``effective_block``)."""
    check_body(body)
    eff = int(body["effective_block"])
    cfg = build_config(base, body, applied_block=eff)
    if not is_era_start(base, eff) or not is_era_start(cfg, eff):
        raise NetConfigError(f"effective_block {eff} is not an era start (era length "
                             f"{era_length_blocks(base.round, eff)} blocks at it)")
    return cfg


def apply_block_for(body: dict, record: ActivationRecord, round_cfg: Any) -> int:
    """§4: ``effective_block`` when the lock-in boundary lies before it, else
    the first era start after the lock-in."""
    eff = int(body["effective_block"])
    lock = int(record.lock_block or 0)
    if lock and lock < eff:
        return eff
    act = int(record.activation_block or 0) or activation_block_for(round_cfg, lock, "era")
    return max(eff, act)


def diff_tables(base_raw: dict, merged_raw: dict) -> list[str]:
    """``[table] key: old -> new`` for every changed overlay-reachable key."""
    out = []
    for table in sorted(set(base_raw) | set(merged_raw)):
        a, b = base_raw.get(table), merged_raw.get(table)
        if not isinstance(a, dict) and not isinstance(b, dict):
            continue
        a, b = a or {}, b or {}
        for key in sorted(set(a) | set(b)):
            if isinstance(b.get(key), dict) and isinstance(a.get(key), dict):
                continue
            if a.get(key, "<unset>") != b.get(key, "<unset>"):
                out.append(f"[{table}] {key}: {a.get(key, '<unset>')!r} -> "
                           f"{b.get(key, '<unset>')!r}")
    return out


# ── storage ──────────────────────────────────────────────────────────────────


def publish_overlay(store: Any, envelope: dict) -> str:
    """Write ``seq-<n>.json`` (immutable), then advance ``latest.json``."""
    body = check_body(envelope["body"])
    key = seq_key(body["seq"])
    text = canonical(envelope)
    store.put_text(key, text, content_type="application/json", acl="public-read")
    store.put_text(LATEST_KEY, text, content_type="application/json", acl="public-read")
    return key


def _read_envelope(store: Any, key: str) -> dict | None:
    from .hippius import ObjectNotFound

    try:
        return json.loads(store.get_text(key))
    except ObjectNotFound:
        return None


def fetch_overlays(store: Any, *, owner: str, last: dict | None = None) -> list[dict]:
    """Verified envelopes newer than ``last`` (a body this node accepted),
    oldest first; ``[]`` when nothing new (or nothing published). Raises
    :class:`NetConfigError` on a rollback, a fork (two bodies under one
    ``seq``) or a broken ``prev`` chain."""
    env = _read_envelope(store, LATEST_KEY)
    if env is None:
        return []
    head = verify_overlay(env, owner=owner)
    n = int(head["seq"])
    if last is not None:
        if n < int(last["seq"]):
            raise NetConfigError(f"overlay rolled back: latest seq {n} < accepted {last['seq']}")
        if n == int(last["seq"]):
            if body_digest(head) != body_digest(last):
                raise NetConfigError(f"two different overlays share seq {n}")
            return []
    start = 1 if last is None else int(last["seq"]) + 1
    if n - start >= MAX_CHAIN_WALK:
        raise NetConfigError(f"overlay chain walk {start}..{n} exceeds {MAX_CHAIN_WALK}")
    prev_body = last
    out: list[dict] = []
    for k in range(start, n + 1):
        e = env if k == n else _read_envelope(store, seq_key(k))
        if e is None:
            raise NetConfigError(f"overlay seq {k} missing from {PREFIX}/ (broken chain)")
        body = verify_overlay(e, owner=owner)
        if int(body["seq"]) != k:
            raise NetConfigError(f"{seq_key(k)} carries seq {body['seq']}")
        want = None if prev_body is None else body_digest(prev_body)
        if body["prev"] != want:
            raise NetConfigError(f"overlay seq {k} does not chain from seq {k - 1}")
        out.append(e)
        prev_body = body
    return out


def fetch_overlay_by_digest(store: Any, *, seq: int, digest_hex: str,
                            owner: str = "") -> dict:
    """The body stamped on a receipt (``netconfig_seq`` / ``netconfig_digest``),
    read from ``seq-<n>.json`` and checked against the digest (and the owner's
    signature when ``owner`` is given)."""
    env = _read_envelope(store, seq_key(seq))
    if env is None:
        raise NetConfigError(f"{seq_key(seq)} not found")
    body = verify_overlay(env, owner=owner) if owner else check_body(env.get("body"))
    if body_digest(body) != str(digest_hex):
        raise NetConfigError(f"{seq_key(seq)} digest {body_digest(body)[:12]} != receipt "
                             f"{str(digest_hex)[:12]}")
    return body


# ── adopted history ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Adopted:
    """One adopted overlay: in force from ``apply_block`` until the next one."""

    apply_block: int
    seq: int
    digest: str
    envelope: dict
    lock_block: int = 0

    @property
    def body(self) -> dict:
        return self.envelope["body"]


@dataclass
class NetConfigState:
    last: dict | None = None             # the newest body accepted (seq/prev anchor)
    candidate: dict | None = None        # its envelope, when it validated
    refused: list[str] = field(default_factory=list)   # digests never to apply
    history: list[Adopted] = field(default_factory=list)
    record: ActivationRecord = field(default_factory=ActivationRecord)

    def to_json(self) -> str:
        return json.dumps({
            "last": self.last, "candidate": self.candidate, "refused": list(self.refused),
            "history": [asdict(h) for h in self.history], "record": asdict(self.record),
        }, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> NetConfigState:
        obj = json.loads(text)
        return cls(
            last=obj.get("last"), candidate=obj.get("candidate"),
            refused=[str(x) for x in obj.get("refused", [])],
            history=[Adopted(**h) for h in obj.get("history", [])],
            record=ActivationRecord(**(obj.get("record") or {})),
        )


class NetConfigStateUnreadable(RuntimeError):
    """The state file exists but neither it nor its ``.bak`` parses: refuse
    to start rather than forget an adopted overlay."""


def load_state(path: Path | str | None) -> NetConfigState:
    if path is None:
        return NetConfigState()
    st, source = read_text_with_backup(path, NetConfigState.from_json)
    if source == "absent":
        return NetConfigState()
    if source == "unreadable" or st is None:
        raise NetConfigStateUnreadable(f"netconfig state {path} (and .bak) does not parse")
    if source == "backup":
        log.warning("netconfig state %s did not parse; resumed from its .bak", path)
    return st


def config_from_history(base: ChainConfig, history: list[Adopted], block: int | None,
                        cache: dict | None = None) -> ChainConfig:
    """The config in force at ``block`` (the shared core of ``config_at``):
    ``base`` itself — same object — when no adopted overlay applies at or
    before it."""
    if block is None or not history:
        return base
    live = [h for h in history if h.apply_block <= int(block)]
    if not live:
        return base
    h = max(live, key=lambda x: (x.apply_block, x.seq))
    key = (h.digest, id(base))
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] is base:
            return hit[1]
    cfg = build_config(base, h.body, applied_block=h.apply_block)
    if cache is not None:
        if len(cache) > 16:
            cache.clear()
        cache[key] = (base, cfg)
    return cfg


class NetConfigResolver:
    """Fetch → verify → validate → (signal) → adopt → ``config_at``.

    One per process. The validator signals (``signal=True``: its note carries
    :meth:`segment`); the trainer runs the same object read-only
    (``signal=False``): it resolves the fleet's lock-in from the notes and
    applies the same overlay at the same block, never writing anything on
    chain. ``store`` duck-types ``get_text`` (a callable returning one is
    opened lazily). Never raises from :meth:`tick`."""

    def __init__(self, cfg: ChainConfig, *, state_path: Path | str | None = None,
                 store: Any = None, signal: bool = False,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.owner = str(cfg.netconfig.owner_hotkey or "")
        self.poll_seconds = int(cfg.netconfig.poll_seconds)
        self.state_path = Path(state_path) if state_path is not None else None
        self._store = store
        self.signal = bool(signal)
        self.clock = clock
        self._fetched_at = float("-inf")
        self._cache: dict = {}
        self._persistence_error: str = ""
        self.state = load_state(self.state_path) if self.enabled else NetConfigState()

    @property
    def enabled(self) -> bool:
        return bool(self.owner)

    # ── reading ──────────────────────────────────────────────────────────────

    def config_at(self, cfg: ChainConfig, block: int | None) -> ChainConfig:
        """The config a settlement whose boundary is ``block`` is judged under:
        ``cfg`` (the node's live base config) with the overlay in force there.
        ``cfg`` itself when none is — always, while the owner key is empty."""
        if not self.enabled:
            return cfg
        self._require_durable()
        return config_from_history(cfg, self.state.history, block, self._cache)

    def in_force(self, block: int | None) -> Adopted | None:
        self._require_durable()
        live = [h for h in self.state.history if block is not None and h.apply_block <= block]
        return max(live, key=lambda x: (x.apply_block, x.seq)) if live else None

    def candidate_body(self) -> dict | None:
        env = self.state.candidate
        if env is None:
            return None
        d = body_digest(env["body"])
        return None if d in self.state.refused else env["body"]

    def segment(self) -> ReadySignal | None:
        """This node's ``cfg-`` note segment (``None`` = nothing to signal)."""
        if not (self.enabled and self.signal) or self._persistence_error:
            return None
        body = self.candidate_body()
        if body is None:
            return None
        rec = self.state.record
        name = cfg_feature_name(body)
        if rec.feature == name and rec.locked and rec.lock_block:
            return ReadySignal(name, lock_block=rec.lock_block,
                               activation_block=rec.activation_block)
        return ReadySignal(name)

    def history_identity(self) -> str:
        """``""`` with no adopted overlay, else the digests in force order."""
        return ",".join(f"{h.apply_block}:{h.digest[:16]}" for h in self.state.history)

    # ── advancing ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if self.state_path is None:
            return
        try:
            write_text_atomic(self.state_path, self.state.to_json(), backup=True)
        except OSError as e:
            self._persistence_error = str(e)
            raise NetConfigError(f"netconfig state {self.state_path} failed to persist: {e}") from e
        self._persistence_error = ""

    def _require_durable(self) -> None:
        if self._persistence_error:
            raise NetConfigError("netconfig persistence pending; judgement held: "
                                 + self._persistence_error)

    def _open_store(self) -> Any:
        if callable(self._store) and not hasattr(self._store, "get_text"):
            self._store = self._store()
        return self._store

    def refresh(self, cfg: ChainConfig, *, force: bool = False) -> bool:
        """Re-read ``latest.json`` (every ``poll_seconds``); verify the chain
        and validate the newest body against ``cfg``. True when the candidate
        changed."""
        if not self.enabled:
            return False
        if self._persistence_error:
            # Retry the exact in-memory state even when latest is unchanged
            # or storage is unavailable; never signal/judge until it is durable.
            self._persist()
        now = self.clock()
        if not force and now - self._fetched_at < self.poll_seconds:
            return False
        self._fetched_at = now
        try:
            store = self._open_store()
            if store is None:
                return False
            envs = fetch_overlays(store, owner=self.owner, last=self.state.last)
        except NetConfigError as e:
            log.error("netconfig: overlay chain REFUSED (%s); keeping the config in force", e)
            return False
        except Exception as e:  # noqa: BLE001 — storage flake: retry next poll
            log.warning("netconfig: overlay fetch failed (%s); retrying next poll", e)
            return False
        if not envs:
            return False
        newest = envs[-1]
        body = newest["body"]
        d = body_digest(body)
        self.state.last = body
        try:
            validate_overlay(cfg, body)
            self.state.candidate = newest
            log.warning("netconfig: overlay seq %d (%s, effective %d) verified — signalling %s",
                        body["seq"], d[:12], body["effective_block"], cfg_feature_name(d))
        except NetConfigError as e:
            self.state.candidate = None
            if d not in self.state.refused:
                self.state.refused.append(d)
            log.error("netconfig: overlay seq %d (%s) REFUSED: %s", body["seq"], d[:12], e)
        self._persist()
        return True

    def tick(self, cfg: ChainConfig, client: Any, *, now_block: int) -> bool:
        """One pass: refresh, resolve the candidate's lock-in, adopt it. True
        when the adopted history changed. Never raises."""
        if not self.enabled:
            return False
        try:
            self.refresh(cfg)
            return self.resolve(cfg, client, now_block=now_block)
        except Exception as e:  # noqa: BLE001 — config adoption must never sink a poll
            log.warning("netconfig step failed (%s); retrying next poll", e)
            return False

    def resolve(self, cfg: ChainConfig, client: Any, *, now_block: int) -> bool:
        body = self.candidate_body()
        if body is None:
            return False
        d = body_digest(body)
        if any(h.digest == d for h in self.state.history):
            return False
        spec = cfg_feature(body)
        rec = self.state.record if self.state.record.feature == spec.name else ActivationRecord()
        res = resolve_activation(cfg, client, now_block=int(now_block), record=rec, feature=spec)
        if res.changed or rec is not self.state.record:
            self.state.record = res.record
            self._persist()
        rec = self.state.record
        if not (rec.locked and rec.source != "config"):
            return False
        return self.adopt(cfg, self.state.candidate, rec)

    def adopt(self, cfg: ChainConfig, envelope: dict, rec: ActivationRecord) -> bool:
        """Put a locked-in overlay into the history at its apply block, or
        refuse it for good (logged) when it would apply retroactively."""
        body = envelope["body"]
        d = body_digest(body)
        apply = apply_block_for(body, rec, cfg.round)
        try:
            new = build_config(cfg, body, applied_block=apply)
            prev = self.config_at(cfg, apply - 1)
            why = retroactive_changes(prev, new, apply)
            if not (is_era_start(prev, apply) and is_era_start(new, apply)):
                why.append(f"apply block {apply} is not an era start on both sides")
        except NetConfigError as e:
            why = [str(e)]
        if why:
            if d not in self.state.refused:
                self.state.refused.append(d)
            self._persist()
            log.error("netconfig: overlay seq %d (%s) locked in but REFUSED at apply block %d "
                      "— never applied (republish with later blocks): %s", body["seq"], d[:12],
                      apply, "; ".join(why))
            return False
        # Entries that would only start at or after this one never judged
        # anything (apply blocks are never in the past): the newer snapshot
        # replaces them.
        self.state.history = sorted(
            [h for h in self.state.history if h.apply_block < apply]
            + [Adopted(apply_block=apply, seq=int(body["seq"]), digest=d,
                       envelope=envelope, lock_block=int(rec.lock_block))],
            key=lambda h: (h.apply_block, h.seq))
        self._cache.clear()
        self._persist()
        log.warning("netconfig: overlay seq %d (%s) ADOPTED — in force from block %d "
                    "(locked in at %d via %s)", body["seq"], d[:12], apply, rec.lock_block,
                    rec.source)
        return True

    def summary(self) -> dict:
        """Presentational block for ``status/chain.json``."""
        body = self.candidate_body()
        return {
            "owner_hotkey": self.owner,
            "candidate": (None if body is None else {
                "seq": body["seq"], "digest": body_digest(body),
                "effective_block": body["effective_block"],
                "feature": cfg_feature_name(body)}),
            "record": asdict(self.state.record),
            "history": [{"apply_block": h.apply_block, "seq": h.seq, "digest": h.digest}
                        for h in self.state.history],
            "refused": list(self.state.refused),
        }


def apply_receipt_netconfig(cfg: ChainConfig, receipt: Any, *, store: Any) -> ChainConfig:
    """For the audit: the config a receipt was judged under — ``cfg`` (already
    carrying the receipt's DEC-CA-0045 blocks) with the overlay the receipt
    stamps, fetched by seq and checked against its digest. A receipt without
    the stamp returns ``cfg`` unchanged."""
    seq = int(getattr(receipt, "netconfig_seq", 0) or 0)
    d = str(getattr(receipt, "netconfig_digest", "") or "")
    if not seq:
        return cfg
    body = fetch_overlay_by_digest(store, seq=seq, digest_hex=d,
                                   owner=str(cfg.netconfig.owner_hotkey or ""))
    return build_config(cfg, body)


# ── the owner's publish tool ─────────────────────────────────────────────────


@dataclass
class OverlayPlan:
    """What ``cascade-validator --netconfig-overlay`` would publish."""

    body: dict
    diff: list[str]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        b = self.body
        lines = [f"overlay seq {b['seq']}  prev {str(b['prev'])[:16] if b['prev'] else '-'}  "
                 f"effective_block {b['effective_block']}",
                 f"digest {body_digest(b)}  note segment {cfg_feature_name(b)}",
                 "", "changes vs chain.toml:"]
        lines += [f"  {d}" for d in self.diff] or ["  (none)"]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        for e in self.errors:
            lines.append(f"REFUSED: {e}")
        lines.append("VALID" if self.ok else "INVALID — not publishable")
        return "\n".join(lines)


def plan_overlay(cfg: ChainConfig, overrides: dict, effective_block: int,
                 previous: dict | None, *, comment: str = "",
                 head_block: int | None = None) -> OverlayPlan:
    """Build the next overlay and run every check a validator will run on it
    (allow-list, merged load checks, era-start ``effective_block``, and the
    retroactivity check assuming it applies AT ``effective_block``). Pure."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        body = next_overlay(overrides, effective_block, previous, network=network_identity(cfg), comment=comment)
    except NetConfigError as e:
        body = {"protocol": PROTOCOL, "seq": 1 if previous is None else previous["seq"] + 1,
                "prev": None if previous is None else body_digest(previous),
                "effective_block": int(effective_block), "overrides": overrides}
        return OverlayPlan(body, [], [str(e)], [])
    diff = diff_tables(cfg.raw or {}, merge_raw(cfg.raw or {}, overrides))
    try:
        new = validate_overlay(cfg, body)
        prev_cfg = cfg
        if previous is not None:
            try:
                prev_cfg = build_config(cfg, previous)
            except NetConfigError as e:
                warnings.append(f"the previous overlay no longer builds on this chain.toml ({e})")
        errors += retroactive_changes(prev_cfg, new, int(effective_block))
    except NetConfigError as e:
        errors.append(str(e))
    if head_block is not None:
        era = int(era_length_blocks(cfg.round, int(head_block)))
        if int(effective_block) - int(head_block) < era:
            warnings.append(f"effective_block is less than one era ({era} blocks) after the "
                            f"head ({head_block}): if the 51 % lock-in lands after it, the "
                            "overlay applies at the first era start after the lock-in, and "
                            "any of its switch blocks before that are then refused")
    warnings.append("validated against chain.toml as typed: blocks the fleet resolved on chain "
                    "(DEC-CA-0045) are re-checked by every validator at adoption")
    return OverlayPlan(body, diff, errors, warnings)
