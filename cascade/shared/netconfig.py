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
   On top of the loader, :mod:`cascade.shared.netconfig_bounds` (rev6.24,
   W-OV) refuses values outside a per-key overlay envelope, release-only
   keys inside the allowed tables, uncontent-addressed data references and
   lane shares the carve would clamp. The envelope is code: no overlay
   changes it.
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
from .config import RETIRED_KEYS, ChainConfig, effective_epoch_blocks, parse_chain_config
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
    # remainder stays release-only below; DEC-CA-0075: the WB10
    # contamination_hashset pin is overlay-settable as a content hash (+ its
    # storage key): every reader verifies the bytes' sha256 against the pin
    "lane_b": None,
    "lane_c": None,
    # LANE_D_PLAN §14.2: every lane D tuning key (activation, kill switch,
    # judging, court, rate limits, TTLs); remainder stays release-only below.
    "lane_d": None,
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
        # LANE_D_PLAN §16 / §14.2: the cross-lane sentinel arms by overlay
        "sentinel_from_block", "sentinel_suites", "sentinel_ratchet", "sentinel_tau",
    }),
    # Base, activation, suite/universe, data sources, window construction
    # and recipe bounds remain release-only. These controls change how an
    # already-supported, pinned competition is judged, paid or timed out.
    "marketcast": frozenset({
        "alpha", "bootstrap_B", "gain_margin", "min_windows", "min_clusters",
        "min_time_blocks", "coverage_tolerance", "coverage_tolerance_over",
        "tail_tolerance", "calibration_z", "zero_scale_frac", "cell_ratio_cap",
        "near_dup_tol", "report_timeout_settlements", "max_infra_deferrals",
        # rev6.24 (W-OV, owner 2026-11-24): lane C's eval DATA — the asset
        # universe, the timeframes, the composite's suite weights and the
        # grid density. Still release-only: the model pins, from_block, the
        # suite list (it is lane C's slot list), data_sources (each is
        # code), pool_prefix and context lengths.
        "assets", "frequencies", "horizons", "composite_suite_weights", "windows_per_cell",
        # DEC-CA-0075 (W-CMP): the embargo is overlay-settable — the trainer
        # records the value in force at each leg's DISPATCH in the signed
        # leg record and the draw / scan / verdict use THAT, so a change
        # never scores a leg on its own training data (code-bounded).
        "embargo_days", "test_family", "pay_curve",
        # DEC-CA-0074: the embargo eval-draw length, backtest windows per cell
        # and the <= 13 h inconclusive wait (code-bounded by the loader)
        "embargo_eval_days", "backtest_windows_per_cell", "verdict_wait_blocks",
        # containment (owner 2026-11-24): MarketCast's start is movable LATER
        # than the typed block, forward-only (:func:`start_refusals`,
        # :func:`retroactive_changes`) — lane C's launch is overlay-gated
        "from_block",
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
# Operator-only: per-deployment knobs no overlay ever carries (each operator
# sets them on their own box; nothing consensus reads them), with the reason
# the refusal prints.
OPERATOR_ONLY_TABLES: dict[str, str] = {
    "generation": "operator-only generation policy, never an overlay",
    "lane_d_ops": "operator-only lane D ops table (LANE_D_PLAN §14.2), never an overlay",
}
# Block-valued switch keys checked for retroactivity (§4).
_BLOCK_KEY = re.compile(r"(?:^|_)from_block(?:_v2)?$|activation_block2?$")
# Containment (owner 2026-11-24): lane C's / MarketCast's typed launch may be
# moved LATER by overlay, never earlier than the typed value, never once it
# has started (:func:`start_refusals`, :func:`retroactive_changes`). An overlay
# that moves ``[lane_c] from_block`` alone carries MarketCast's start along
# (:func:`_carry_lane_c_start`).
_LATER_ONLY_STARTS = (("lane_c", "from_block"), ("marketcast", "from_block"))
# The containment kill switches: never moved to, or away from, a block before
# the apply block (a closed lane is never re-opened retroactively).
_UNTIL_KEYS = (("lane_b", "until_block"), ("lane_c", "until_block"),
               ("lane_d", "until_block"))
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
    from .netconfig_bounds import policy as bounds_policy

    return {
        "protocol": PROTOCOL,
        "envelope": bounds_policy(),
        "overlay_keys":{table: ["*"] if keys is None else sorted(keys)
                         for table, keys in sorted(ALLOWED.items())},
        "release_only_tables": dict(sorted(REFUSED_TABLES.items())),
        "default_policy": "Keys not listed are release-only.",
        "wildcard_policy": "Existing parser-supported keys only; loader checks still apply.",
        "lane_c_competitions": "Existing competitions: weight and until_block only; "
                               "[lane_c] specs compose registry blocks (DEC-CA-0075).",
        "competition_registry": _competition_registry(),
        "adoption": "Owner signature, permit-stake lock-in, then an era boundary.",
        "operator_only": sorted(OPERATOR_ONLY_TABLES),
        "training": "Trainer-declared recipe/budget changes need no fleet update; "
                    "locked training terms and new model code require a release.",
    }


def _competition_registry() -> dict:
    from .competition_spec import registry

    return registry()


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


def check_overrides(overrides: object, *, in_force: bool = False) -> dict:
    """Refuse anything outside :data:`ALLOWED`; returns ``overrides`` unchanged.

    ``in_force`` (#891): the body is an overlay already adopted (or a
    receipt's stamp), so the W-OV per-key envelope (release-only keys,
    content-addressed pins) is not a refusal here — :func:`build_config`
    resolves those keys to the release value instead."""
    if not isinstance(overrides, dict):
        raise NetConfigError("overrides must be a table of tables")
    bad: list[str] = []
    for table, kv in overrides.items():
        if table in REFUSED_TABLES:
            bad.append(f"[{table}] ({REFUSED_TABLES[table]})")
            continue
        if table in OPERATOR_ONLY_TABLES:
            bad.append(f"[{table}] ({OPERATOR_ONLY_TABLES[table]})")
            continue
        if table not in ALLOWED:
            bad.append(f"[{table}] (not on the DEC-CA-0057 allow-list)")
            continue
        if not isinstance(kv, dict):
            bad.append(f"[{table}] must be a table")
            continue
        allowed = ALLOWED[table]
        for key, value in kv.items():
            if (table, key) in (("scoring", "burn_uid"), ("lane_b", "remainder"),
                                ("lane_c", "remainder"), ("lane_d", "remainder")):
                bad.append(f"[{table}] {key} (burn destination is release-only)")
            elif (table in ("lane_b", "lane_c", "marketcast")
                  and key in RETIRED_KEYS.get(table, ())):
                # the loader ignores a retired key, so an overlay naming one
                # would silently change nothing (DEC-CA-0073: lane C pay;
                # LANE_D_PLAN rev6.22: lane B pay)
                bad.append(f"[{table}] {key} (retired)")
            elif isinstance(value, dict) and table in ("v2",):
                bad.append(f"[{table}.{key}] (nested tables are release-only)")
            elif allowed is not None and key not in allowed:
                bad.append(f"[{table}] {key} (not an overlay key)")
    from .netconfig_bounds import key_refusals, pin_refusals

    if not in_force:
        bad += key_refusals(overrides) + pin_refusals(overrides)
    if bad:
        raise NetConfigError("overlay refused — release-only keys: " + "; ".join(bad))
    return overrides


def check_body(body: object, *, in_force: bool = False) -> dict:
    """Shape of an overlay body; returns it unchanged (``in_force``: see
    :func:`check_overrides`)."""
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
    check_overrides(body["overrides"], in_force=in_force)
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


def verify_overlay(envelope: object, *, owner: str, in_force: bool = False) -> dict:
    """The verified body of ``envelope`` signed by ``owner``, else raise
    (``in_force``: see :func:`check_overrides`)."""
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
    return check_body(body, in_force=in_force)


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
    if int(getattr(ac, "resolved_image_decoupled_block", 0) or 0):
        raw.setdefault("v2", {})["image_decoupled_from_block"] = int(
            cfg.v2.image_decoupled_from_block)
    if int(getattr(ac, "resolved_shared_ladder_block", 0) or 0):
        raw.setdefault("eval", {})["shared_ladder_from_block"] = int(
            cfg.eval.shared_ladder_from_block)
    return raw


def _sets(overrides: dict, table: str, key: str) -> bool:
    return key in (overrides.get(table) or {})


def _comparable(cfg: ChainConfig) -> ChainConfig:
    return replace(cfg, raw={})


def _typed_start(base: ChainConfig, table: str) -> int:
    return int(getattr(getattr(base, table), "from_block", 0) or 0)


def start_refusals(base: ChainConfig, overrides: dict) -> list[str]:
    """Containment (owner 2026-11-24): an overlay may move lane C's /
    MarketCast's start LATER than the typed (``chain.toml``) value or to 0
    (off), never earlier. Empty = allowed. A typed 0 (off) may be armed —
    the retroactivity guard then keeps it at or after the apply block."""
    out = []
    for table, key in _LATER_ONLY_STARTS:
        if not _sets(overrides, table, key):
            continue
        v = overrides[table][key]
        typed = _typed_start(base, table)
        if type(v) is int and 0 < v < typed:
            out.append(f"[{table}] {key} = {v} is earlier than the typed launch {typed} "
                       "(an overlay only postpones it)")
    return out


def _carry_lane_c_start(base: ChainConfig, overrides: dict, raw: dict) -> dict:
    """An overlay that postpones ``[lane_c] from_block`` without naming
    ``[marketcast] from_block`` postpones MarketCast with it (MarketCast
    never starts before lane C), so the one-key postpone loads. Pure."""
    if not _sets(overrides, "lane_c", "from_block") or _sets(overrides, "marketcast",
                                                             "from_block"):
        return raw
    v = overrides["lane_c"]["from_block"]
    mc = int(base.marketcast.from_block or 0)
    if type(v) is not int or v <= 0 or not 0 < mc < v:
        return raw
    out = copy.deepcopy(raw)
    out.setdefault("marketcast", {})["from_block"] = int(v)
    return out


def build_config(base: ChainConfig, body: dict, *, applied_block: int = 0,
                 in_force: bool = False) -> ChainConfig:
    """The config ``body`` puts in force on top of ``base`` (the node's live
    config, resolved activation blocks included), through the SAME parser and
    load checks as ``chain.toml``. Raises :class:`NetConfigError` on a refused
    key, a key no parser reads, or a merged table the loader refuses.

    ``in_force`` (#891): ``body`` is ALREADY in force (adopted, locked in, or
    stamped on a receipt), possibly under an older release whose W-OV
    envelope (:mod:`cascade.shared.netconfig_bounds`) was looser. Every key
    this release's envelope refuses is resolved to the RELEASE value (the
    override is dropped; the rest of the overlay stands), logged once per
    key and listed by :func:`out_of_bounds_report`, instead of raising —
    so an upgrade never stalls rounds on an overlay nobody can un-adopt.
    The rule is a pure function of (release, base, body): every node, the
    satellites and the audit replay resolve the same config. With no key out
    of bounds the result is identical to ``in_force=False``. A NEW overlay
    being validated for adoption is never built with it (still refused)."""
    if body.get("network") != network_identity(base):
        raise NetConfigError("overlay network mismatch (chain identity or netuid)")
    if in_force:
        ov = check_overrides(_resolve_in_force(base, body), in_force=False)
    else:
        ov = check_overrides(body["overrides"])
    early = start_refusals(base, ov)
    if early:
        raise NetConfigError("overlay refused — " + "; ".join(early))
    raw = _carry_lane_c_start(base, ov, merge_raw(effective_raw(base), ov))
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
    # rev6.24 (W-OV): the overlay-only envelope, after the loader accepted
    # the merged table (a value the loader refuses keeps its loader message)
    from .netconfig_bounds import share_refusals, value_refusals

    carve = any(_sets(ov, t, k) for t in ("lane_b", "lane_c", "lane_d")
                for k in ("share", "takes_from"))
    from .competition_spec import bound_refusals

    out_of_bounds = value_refusals(ov) + (share_refusals(cfg) if carve else [])
    if _sets(ov, "lane_c", "specs"):
        # DEC-CA-0075: a spec's KNOWN keys out of their code bounds are
        # refused here (unknown block names stay a typed refusal at load)
        out_of_bounds += bound_refusals(ov["lane_c"]["specs"])
    if out_of_bounds:
        raise NetConfigError("overlay refused — out of bounds: " + "; ".join(out_of_bounds))
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
        "resolved_image_decoupled_block": (
            0 if _sets(ov, "v2", "image_decoupled_from_block")
            else int(getattr(bac, "resolved_image_decoupled_block", 0) or 0)),
        "resolved_shared_ladder_block": (
            0 if _sets(ov, "eval", "shared_ladder_from_block")
            else int(getattr(bac, "resolved_shared_ladder_block", 0) or 0)),
    }
    return replace(
        cfg, activation=replace(ac, **flags),
        # runtime-only operator switches (--force-burn) ride on the base
        validator=base.validator,
        netconfig=replace(cfg.netconfig, seq=int(body["seq"]), digest=body_digest(body),
                          applied_block=int(applied_block)))


# #891: ``digest -> ("[table] key ... using release value ...", ...)`` for each
# in-force overlay this process resolved with keys out of the current
# envelope; the logged set keeps the loud line to once per (overlay, key).
_OOB_REPORT: dict[str, tuple[str, ...]] = {}
_OOB_LOGGED: set[tuple[str, str, str]] = set()
_CARVE_KEYS = tuple((t, k) for t in ("lane_b", "lane_c", "lane_d")
                    for k in ("share", "takes_from"))


def _release_value(base: ChainConfig, table: str, key: str) -> object:
    return (effective_raw(base).get(table) or {}).get(key, "<release default>")


def _without(overrides: dict, drop: list[tuple[str, str, str]]) -> dict:
    gone = {(t, k) for t, k, _ in drop}
    out: dict = {}
    for table, kv in overrides.items():
        if not isinstance(kv, dict):
            out[table] = kv
            continue
        keep = {k: v for k, v in kv.items() if (table, k) not in gone}
        if keep:
            out[table] = keep
    return out


def in_force_out_of_bounds(base: ChainConfig, overrides: dict) -> list[tuple[str, str, str]]:
    """#891: ``(table, key, why)`` for every override of an IN-FORCE overlay
    that this release's W-OV envelope refuses, in a fixed order: the per-key
    envelope (:func:`~cascade.shared.netconfig_bounds.out_of_bounds_keys`),
    then — on the remainder — lane shares the carve would clamp (every
    ``share`` / ``takes_from`` override of lanes B/C/D) and ``[lane_c]
    specs`` with a spec out of its code bounds (the whole specs table). A
    merged table the loader refuses is left to :func:`build_config` (not an
    envelope matter). Pure."""
    from .netconfig_bounds import out_of_bounds_keys, share_refusals

    drop = out_of_bounds_keys(overrides)
    rest = _without(overrides, drop)
    carve = [(t, k) for t, k in _CARVE_KEYS if _sets(rest, t, k)]
    specs = _sets(rest, "lane_c", "specs")
    if carve or specs:
        try:
            cfg = parse_chain_config(_carry_lane_c_start(base, rest,
                                                         merge_raw(effective_raw(base), rest)))
        except (ValueError, KeyError, TypeError):
            return drop                 # the loader's refusal stands (build_config)
        why = share_refusals(cfg) if carve else []
        if why:
            drop += [(t, k, "; ".join(why)) for t, k in carve]
        if specs:
            from .competition_spec import bound_refusals

            sw = bound_refusals(rest["lane_c"]["specs"])
            if sw:
                drop.append(("lane_c", "specs", "; ".join(sw)))
    return drop


def _resolve_in_force(base: ChainConfig, body: dict) -> dict:
    """The overrides :func:`build_config` applies for an in-force ``body``
    (#891): :func:`in_force_out_of_bounds` keys dropped (release value),
    recorded for :func:`out_of_bounds_report` and logged once per key."""
    ov = body["overrides"]
    if not isinstance(ov, dict):
        return ov
    drop = in_force_out_of_bounds(base, ov)
    if not drop:
        return ov
    d = body_digest(body)
    lines = []
    for table, key, why in drop:
        line = (f"[{table}] {key} = {ov[table][key]!r} ({why}) — using release value "
                f"{_release_value(base, table, key)!r}")
        lines.append(line)
        if (d, table, key) not in _OOB_LOGGED:
            _OOB_LOGGED.add((d, table, key))
            log.error("netconfig: overlay key out of current bounds: %s; overlay seq %s (%s) "
                      "stays in force for its other keys — publish a compliant overlay",
                      line, body.get("seq"), d[:12])
    _OOB_REPORT[d] = tuple(lines)
    return _without(ov, drop)


def out_of_bounds_report() -> dict[str, list[str]]:
    """#891, presentational: ``{overlay digest: [resolved key lines]}`` for
    every in-force overlay this process resolved with keys outside the
    current envelope (empty = none). Status / readiness surfaces read it."""
    return {d: list(v) for d, v in sorted(_OOB_REPORT.items())}


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

    # W-CMP2 (DEC-CA-0075): a SPEC INSTANCE is added, changed or removed by
    # its spec (its start is guarded by retroactive_changes)
    inst = {t[0] for t in tuple(base.lane_c.instances) + tuple(new.lane_c.instances)}
    a = {c.kind: c for c in base.lane_c.competition_list if c.kind not in inst}
    b = {c.kind: c for c in new.lane_c.competition_list if c.kind not in inst}
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
    # containment (owner 2026-11-24): a lane C / MarketCast start that has
    # already passed is never moved to another nonzero block (that would
    # un-start a running lane; use until_block), and a kill switch is never
    # moved to, or away from, a block before the apply block
    for table, key in _LATER_ONLY_STARTS:
        was = int(getattr(getattr(prev, table), key, 0) or 0)
        now = int(getattr(getattr(new, table), key, 0) or 0)
        if was != now and now > 0 and 0 < was < apply_block:
            out.append(f"[{table}] {key} {was} -> {now}: it started before the block the "
                       f"overlay applies at ({apply_block}); end it with until_block instead")
    for table, key in _UNTIL_KEYS:
        was = int(getattr(getattr(prev, table), key, 0) or 0)
        now = int(getattr(getattr(new, table), key, 0) or 0)
        if was != now and (0 < now < apply_block or 0 < was < apply_block):
            out.append(f"[{table}] {key} {was} -> {now} crosses a block before the one the "
                       f"overlay applies at ({apply_block})")
    # W-CMP2: a spec instance's start never lies before the apply block when
    # it is added or moved (no retroactive competition)
    old_f = {c.kind: int(c.from_block) for c in prev.lane_c.competition_list}
    inst = {t[0] for t in new.lane_c.instances}
    for c in new.lane_c.competition_list:
        if c.kind in inst and old_f.get(c.kind) != int(c.from_block) and \
                0 < int(c.from_block) < apply_block:
            out.append(f"[lane_c.specs.{c.kind}] from_block {c.from_block} lies before the "
                       f"block the overlay applies at ({apply_block})")
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
    # the receipt stamps an overlay that WAS in force (#891: in_force)
    body = (verify_overlay(env, owner=owner, in_force=True) if owner
            else check_body(env.get("body"), in_force=True))
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
    # #891: an adopted overlay is in force — keys this release's envelope
    # refuses resolve to the release value instead of failing every round
    cfg = build_config(base, h.body, applied_block=h.apply_block, in_force=True)
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
        and adopt as the candidate the NEWEST body that validates against
        ``cfg`` (containment, owner 2026-11-24): the new bodies newest first,
        then the candidate already held. A refused newer body never drops an
        older valid one that is still being signalled (a publishing-order
        mistake cannot strand a step). True when the state changed."""
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
        self.state.last = envs[-1]["body"]
        held = self.state.candidate
        chosen = None
        for env in reversed(envs):
            body = env["body"]
            d = body_digest(body)
            if d in self.state.refused:
                continue
            try:
                validate_overlay(cfg, body)
            except NetConfigError as e:
                self.state.refused.append(d)
                log.error("netconfig: overlay seq %d (%s) REFUSED: %s", body["seq"], d[:12], e)
                continue
            chosen = env
            log.warning("netconfig: overlay seq %d (%s, effective %d) verified — signalling %s",
                        body["seq"], d[:12], body["effective_block"], cfg_feature_name(d))
            break
        if chosen is None and held is not None:
            hb = held["body"]
            hd = body_digest(hb)
            try:
                if hd in self.state.refused:
                    raise NetConfigError("refused earlier")
                validate_overlay(cfg, hb)
            except NetConfigError as e:
                log.error("netconfig: held overlay seq %d (%s) no longer validates (%s)",
                          hb["seq"], hd[:12], e)
            else:
                chosen = held
                log.warning("netconfig: no newer overlay validates; still signalling seq %d "
                            "(%s)", hb["seq"], hd[:12])
        self.state.candidate = chosen
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
            # locked in by the fleet: in force from ``apply`` (#891)
            new = build_config(cfg, body, applied_block=apply, in_force=True)
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
            # #891: in-force overlay keys resolved to the release value
            "out_of_bounds": {d: v for d, v in out_of_bounds_report().items()
                              if any(h.digest == d for h in self.state.history)},
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
    return build_config(cfg, body, in_force=True)


# ── the owner's publish tool ─────────────────────────────────────────────────


@dataclass
class OverlayPlan:
    """What ``cascade-validator --netconfig-overlay`` would publish."""

    body: dict
    diff: list[str]
    errors: list[str]
    warnings: list[str]
    # #911: what the plan was judged against and what it changes THERE
    in_force: str = "chain.toml alone"
    changes: list[str] = field(default_factory=list)
    lanes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        b = self.body
        lines = [f"overlay seq {b['seq']}  prev {str(b['prev'])[:16] if b['prev'] else '-'}  "
                 f"effective_block {b['effective_block']}",
                 f"digest {body_digest(b)}  note segment {cfg_feature_name(b)}",
                 f"judged against the config in force: {self.in_force}",
                 "", "changes vs the config in force (effective values):"]
        lines += [f"  {d}" for d in self.changes] or ["  (none)"]
        lines += [f"  {x}" for x in self.lanes]
        lines += ["", "changes vs chain.toml:"]
        lines += [f"  {d}" for d in self.diff] or ["  (none)"]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        for e in self.errors:
            lines.append(f"REFUSED: {e}")
        lines.append("VALID" if self.ok else "INVALID — not publishable")
        return "\n".join(lines)


def newest_valid(cfg: ChainConfig, bodies: list[dict]) -> tuple[dict | None, list[str]]:
    """#911, owner tool: of the published chain ``bodies`` (oldest first), the
    newest one :func:`validate_overlay` accepts on ``cfg`` — the body a node
    on this release keeps signalling / in force when newer ones are refused
    (:meth:`NetConfigResolver.refresh`) — and one line per newer body skipped.
    ``None`` = no published body validates (chain.toml alone). Pure."""
    skipped: list[str] = []
    for body in reversed(list(bodies)):
        try:
            validate_overlay(cfg, body)
        except NetConfigError as e:
            skipped.append(f"published seq {body.get('seq')} ({body_digest(body)[:12]}) does "
                           f"not validate on this chain.toml, so nodes refused it: {e}")
            continue
        return body, skipped
    return None, skipped


# #911: the lanes whose on/off state the plan reports, and the keys that
# switch each one (an overlay that sets none of them turns it off only by
# dropping a restatement — implicitly). MarketCast is switched with lane C.
LANE_LABELS = (("lane_b", "LANE B"), ("lane_c", "LANE C"), ("lane_d", "LANE D"),
               ("marketcast", "MARKETCAST"))
_LANE_SWITCHES = {
    "lane_b": (("lane_b", "from_block"), ("lane_b", "until_block"), ("lane_b", "share")),
    "lane_c": (("lane_c", "from_block"), ("lane_c", "until_block"), ("lane_c", "share")),
    "lane_d": (("lane_d", "from_block"), ("lane_d", "until_block"), ("lane_d", "share")),
    "marketcast": (("marketcast", "from_block"), ("marketcast", "until_block"),
                   ("lane_c", "from_block"), ("lane_c", "until_block"), ("lane_c", "share")),
}
_UNSET = "<release default>"


def lane_on(cfg: ChainConfig, table: str, block: int) -> bool:
    """Presentational (owner tool): lane ``table`` is armed to pay at some
    settlement after ``block`` (started or scheduled, share > 0, not ended at
    or before ``block``)."""
    lane = getattr(cfg, table)
    fb = int(getattr(lane, "from_block", 0) or 0)
    ub = int(getattr(lane, "until_block", 0) or 0)
    share = getattr(lane, "share", None)
    return fb > 0 and (share is None or float(share) > 0) and (ub <= 0 or ub > int(block))


def _flat(raw: dict, prefix: str = "") -> dict[tuple[str, str], object]:
    out: dict[tuple[str, str], object] = {}
    for table, kv in raw.items():
        if not isinstance(kv, dict):
            continue
        name = f"{prefix}{table}"
        for key, value in kv.items():
            if isinstance(value, dict):
                out.update(_flat({key: value}, prefix=f"{name}."))
            else:
                out[(name, key)] = value
    return out


def _resolved_overrides(cfg: ChainConfig, body: dict | None) -> dict:
    """The overrides a node runs for an in-force ``body`` (#891 resolution,
    without the loud log line). ``{}`` for ``None``."""
    if body is None or not isinstance(body.get("overrides"), dict):
        return {}
    ov = body["overrides"]
    return _without(ov, in_force_out_of_bounds(cfg, ov))


def _merged(cfg: ChainConfig, overrides: dict) -> dict:
    if not all(isinstance(kv, dict) for kv in overrides.values()):
        return effective_raw(cfg)
    return _carry_lane_c_start(cfg, overrides, merge_raw(effective_raw(cfg), overrides))


def effective_changes(cfg: ChainConfig, in_force_overrides: dict,
                      overrides: dict) -> list[str]:
    """#911: ``[table] key: in force -> new`` for every key whose EFFECTIVE
    value changes when ``overrides`` replaces the in-force overlay on ``cfg``
    — a key the in-force overlay restated and this one drops (it reverts to
    chain.toml) is marked ``DROPPED``. Pure, presentational."""
    before = _flat(_merged(cfg, in_force_overrides))
    after = _flat(_merged(cfg, overrides))
    out = []
    for name, key in sorted(set(before) | set(after)):
        a, b = before.get((name, key), _UNSET), after.get((name, key), _UNSET)
        if a == b:
            continue
        top, _, rest = name.partition(".")
        first = rest.split(".")[0] if rest else key
        line = f"[{name}] {key}: {a!r} -> {b!r}".replace(repr(_UNSET), _UNSET)
        if _sets(in_force_overrides, top, first) and not _sets(overrides, top, first):
            line += (" (DROPPED: restated by the overlay in force, not by this one; "
                     "reverts to chain.toml)")
        out.append(line)
    return out


def lane_switches(cfg: ChainConfig, prev_cfg: ChainConfig, new: ChainConfig,
                  in_force_overrides: dict, overrides: dict,
                  block: int) -> tuple[list[str], list[str]]:
    """#911: (report lines, implicit switch-offs). A lane ON in ``prev_cfg``
    and OFF in ``new`` is implicit unless ``overrides`` SETS one of its switch
    keys to a value that changes it (a kill switch, a zero share)."""
    before = _merged(cfg, in_force_overrides)
    after = _merged(cfg, overrides)
    lines: list[str] = []
    implicit: list[str] = []
    for table, label in LANE_LABELS:
        was, now = lane_on(prev_cfg, table, block), lane_on(new, table, block)
        if was != now:
            explicit = [f"[{t}] {k}" for t, k in _LANE_SWITCHES[table]
                        if _sets(overrides, t, k)
                        and (before.get(t) or {}).get(k) != (after.get(t) or {}).get(k)]
            line = f"{label}: {'ON' if was else 'OFF'} -> {'ON' if now else 'OFF'}"
            if was and not explicit:
                line += " (IMPLICIT: no switch key of this lane is set by this overlay)"
                implicit.append(label)
            elif explicit:
                line += f" (set by {', '.join(explicit)})"
            lines.append(line)
        lp, ln = getattr(prev_cfg, table), getattr(new, table)
        for attr in ("share", "takes_from"):
            a, b = getattr(lp, attr, None), getattr(ln, attr, None)
            if a != b:
                lines.append(f"{label} {attr.upper()}: {a!r} -> {b!r}")
    lines += _lane_a_moves(prev_cfg, new, block)
    return lines, implicit


def lane_a_share(cfg: ChainConfig, boundary: int) -> float | None:
    """Presentational: lane A's pool at settlement ``boundary`` (``1 - b - c -
    d`` over the lanes active there, before lane B's unpaid refund). Mirrors
    ``lane_c.lane_pool_shares`` without importing it: the satellite import
    closure (INV-12) must not reach that producer module."""
    b = float(cfg.lane_b.share) if cfg.lane_b.active(boundary) else 0.0
    c = 0.0
    if cfg.lane_c.active(boundary):
        if cfg.lane_c.takes_from == "lane_a":
            c = min(float(cfg.lane_c.share), max(0.0, 1.0 - b))
        else:
            c = min(float(cfg.lane_c.share), b)
            b -= c
    d = 0.0
    if cfg.lane_d.active(boundary):
        want, src = max(0.0, float(cfg.lane_d.share)), str(cfg.lane_d.takes_from)
        if src == "lane_a":
            d = min(want, max(0.0, 1.0 - b - c))
        elif src == "lane_b":
            d = min(want, max(0.0, b))
            b -= d
        elif src == "lane_c":
            d = min(want, max(0.0, c))
            c -= d
        else:
            return None
    return round(1.0 - b - c - d, 6)


def _lane_a_moves(prev_cfg: ChainConfig, new: ChainConfig, block: int) -> list[str]:
    """``LANE A SHARE: a -> b (from settlement S)`` at the first settlement
    after ``block`` or after a lane start past it where lane A's pool moves."""
    starts = {int(block)}
    for c in (prev_cfg, new):
        for table, _ in LANE_LABELS:
            fb = int(getattr(getattr(c, table), "from_block", 0) or 0)
            if fb > int(block):
                starts.add(fb)
    for s in sorted(starts):
        probe = s + int(effective_epoch_blocks(new.round, s))
        a, b = lane_a_share(prev_cfg, probe), lane_a_share(new, probe)
        if a != b:
            return [f"LANE A SHARE: {a!r} -> {b!r} (from settlement {probe})"]
    return []


_PREVIOUS = object()


def plan_overlay(cfg: ChainConfig, overrides: dict, effective_block: int,
                 previous: dict | None, *, comment: str = "",
                 head_block: int | None = None, in_force: Any = _PREVIOUS,
                 allow_lane_off: bool = False) -> OverlayPlan:
    """Build the next overlay and run every check a validator will run on it
    (allow-list, merged load checks, era-start ``effective_block``, and the
    retroactivity check assuming it applies AT ``effective_block``). Pure.

    ``previous`` (the latest PUBLISHED body) gives ``seq`` / ``prev`` only.
    ``in_force`` (#911; default ``previous``, ``None`` = chain.toml alone) is
    the body nodes run when this one takes over — after a refused overlay,
    the newest one that validates (:func:`newest_valid`). Retroactivity, the
    effective-change report and the lane switch guard are judged against it.
    A lane it arms that this overlay turns off without setting a switch key
    (a dropped restatement) is refused unless ``allow_lane_off``. Owner-side
    only: nothing here is what a validator runs to accept an overlay."""
    errors: list[str] = []
    warnings: list[str] = []
    if in_force is _PREVIOUS:
        in_force = previous
    label = ("chain.toml alone" if in_force is None else
             f"seq {in_force.get('seq')} ({body_digest(in_force)[:12]})")
    try:
        body = next_overlay(overrides, effective_block, previous, network=network_identity(cfg), comment=comment)
    except NetConfigError as e:
        body = {"protocol": PROTOCOL, "seq": 1 if previous is None else previous["seq"] + 1,
                "prev": None if previous is None else body_digest(previous),
                "effective_block": int(effective_block), "overrides": overrides}
        return OverlayPlan(body, [], [str(e)], [], in_force=label)
    diff = diff_tables(cfg.raw or {}, merge_raw(cfg.raw or {}, overrides))
    in_force_ov = _resolved_overrides(cfg, in_force)
    changes = effective_changes(cfg, in_force_ov, overrides)
    lanes: list[str] = []
    try:
        new = validate_overlay(cfg, body)
        prev_cfg = cfg
        if in_force is not None:
            try:
                prev_cfg = build_config(cfg, in_force)
            except NetConfigError as e:
                warnings.append(f"the overlay in force no longer builds on this chain.toml ({e})")
                # #891: nodes on this release run it with the out-of-bounds
                # keys at the release value; judge retroactivity against that
                try:
                    prev_cfg = build_config(cfg, in_force, in_force=True)
                except NetConfigError:
                    prev_cfg = cfg
        errors += retroactive_changes(prev_cfg, new, int(effective_block))
        lanes, implicit = lane_switches(cfg, prev_cfg, new, in_force_ov, overrides,
                                        int(effective_block))
        for lane in implicit:
            why = (f"{lane} is armed by the config in force ({label}) and this overlay turns "
                   "it OFF without setting any of its switch keys (a dropped restatement): "
                   "restate the in-force keys")
            if allow_lane_off:
                warnings.append(why + " — allowed (--allow-lane-off)")
            else:
                errors.append(why + ", or pass --allow-lane-off to turn it off on purpose")
    except NetConfigError as e:
        errors.append(str(e))
    if head_block is not None:
        era = int(era_length_blocks(cfg.round, int(head_block)))
        # containment: a postponed lane C / MarketCast start lies at least one
        # era past the head (the lock-in needs that long)
        for table, key in _LATER_ONLY_STARTS:
            v = (overrides.get(table) or {}).get(key)
            if type(v) is int and 0 < v < int(head_block) + era:
                errors.append(f"[{table}] {key} = {v} is less than one era ({era} blocks) "
                              f"after the head ({head_block})")
        if int(effective_block) - int(head_block) < era:
            warnings.append(f"effective_block is less than one era ({era} blocks) after the "
                            f"head ({head_block}): if the 51 % lock-in lands after it, the "
                            "overlay applies at the first era start after the lock-in, and "
                            "any of its switch blocks before that are then refused")
    warnings.append("validated against chain.toml as typed: blocks the fleet resolved on chain "
                    "(DEC-CA-0045) are re-checked by every validator at adoption")
    return OverlayPlan(body, diff, errors, warnings, in_force=label, changes=changes,
                       lanes=lanes)
