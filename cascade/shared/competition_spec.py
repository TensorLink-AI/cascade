"""Declarative lane-C competitions (DEC-CA-0075, W-CMP).

Owner 2026-11-24: "new test or formula and embargo days etc. should be up to
our choice, and we should bring new lane C competitions without validators
updating." Validators never execute owner-supplied code (security), so a
competition is a SPEC composed only of BLOCKS this release ships, selected
by a signed overlay (DEC-CA-0057, ``[lane_c] specs``)::

    [lane_c.specs.mkt]
    runner = "marketcast"                        # RUNNERS
    pool = { prefix = "pool_mkt", signer = "owner", schema = "cascade-std-pool/1" }
    select = { domain = "market" }               # standard-pool series tags
    universe = { crypto = ["BTC", "ETH"] }       # tag-class -> assets
    timeframes = { returns = ["5m", "1h"] }      # per metric
    horizons = { returns = ["1h", "1d"] }        # per metric
    metrics = ["returns", "vol"]                 # SERIES_METRICS
    score = "crps"                               # SCORES
    guards = ["mase"]                            # GUARD_METRICS
    metric_weights = { returns = 0.5, vol = 0.5 }
    gain_margin = 0.01
    alpha = 0.05
    test_family = "bonferroni"                   # rules_menu.TEST_FAMILIES
    pay_curve = "fixed_term"                     # rules_menu.PAY_CURVES
    weight = 1.0                                 # its part of lane C's share
    until_block = 0
    embargo_days = 30
    embargo_eval_days = 28

Every key but ``runner`` is optional; an absent key keeps the value the
runner's own table carries (``[marketcast]`` for the ``marketcast`` runner),
so a spec restating today's values is byte-identical
(:func:`marketcast_spec` exports it; ``tests/accept/w_cmp`` pins it).

Data: every competition reads a content-addressed, signed STANDARD POOL
(:mod:`cascade.shared.std_pool`) named by prefix + signer + schema. Source
code (APIs, relays, scrapers) lives only in pool builders; a new domain is a
new builder job plus a spec — no validator code.

Refusal, never a crash or a fork: :func:`parse_spec` returns a
:class:`SpecRefusal` (``code``, ``reason``) for an unknown block name, a
missing runner, a pool schema this release cannot read or a value outside
the code bounds (:data:`SPEC_BOUNDS`). A refused competition judges nothing
and its share burns; the refusal is logged and recorded in the lane-C
config record (:func:`specs_record`). The overlay path additionally refuses
out-of-bounds values before any node signals the overlay (W-OV style), but
lets unknown NAMES through as a typed refusal, so a node one release behind
burns that one competition's share instead of rejecting the whole overlay.

What still needs a release: a new BLOCK (a new metric, test family, pay
curve, pool schema version or runner).

W-CMP2 (plan rev6.25, owner 2026-11-24): the ``marketcast`` runner is
MULTI-INSTANCE. ``mkt`` is its legacy instance (its tables are ``[marketcast]``
and ``[lane_c.competitions.mkt]``); any other id is a spec INSTANCE — its own
targets ``<id>:<name>``, legs, rows, reports (``reports[<id>]`` in the lane-C
manifest), verdicts, slots, payouts and audit, keyed by the id. An instance
spec must name its pool, metrics, universe, timeframes, horizons and
``from_block`` (:data:`INSTANCE_REQUIRED`); everything else inherits the
deployed ``[marketcast]`` table (the model pins, the recipe bounds, the
verdict statistics). ``channels`` maps a metric's channel name to the pool's
(``{close = "load"}``: an energy pool's load series is read as the close).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from .rules_menu import LANE_DEFAULTS, PAY_CURVES, TEST_FAMILIES
from .std_pool import SCHEMAS as STD_POOL_SCHEMAS

# ── the registries (code; an overlay only names entries) ────────────────────

# Series metrics: what a competition forecasts, each a pure transform of the
# standard-pool CHANNELS it names (cascade.pool.marketcast.asset_series).
SERIES_METRICS: dict[str, tuple[str, ...]] = {
    "returns": ("close",),
    "vol": ("open", "high", "low", "close"),
    "range": ("open", "high", "low"),
    "high": ("open", "high", "low"),
    "low": ("open", "high", "low"),
    "liquidity": ("volume",),
    "ohlcv": ("open", "high", "low", "close", "volume"),
    "funding": ("funding",),
    "alpha": ("price",),
}
# Primary scores (the gain is a ratio of these) and no-regression guards.
SCORES: tuple[str, ...] = ("crps",)
GUARD_METRICS: tuple[str, ...] = ("mase", "qlike", "tail")
# Pool readers: the standard-pool schema versions this release reads, plus
# "" = the legacy MarketCast snapshot OR a standard pool (today's behaviour).
POOL_SCHEMAS: tuple[str, ...] = ("",) + tuple(STD_POOL_SCHEMAS)
# Runners: the judge / leg code a spec is executed by, and the competition
# ids each can host in this release.
# ids each runner hosts: "*" = any id matching :data:`SPEC_ID_RE` that is
# not reserved (W-CMP2: the marketcast runner is multi-instance)
RUNNERS: dict[str, tuple[str, ...]] = {"marketcast": ("*",)}
LEGACY_INSTANCE = "mkt"
# ids that name another lane-C kind (retired DEC-CA-0056 kinds) or a table
RESERVED_IDS = frozenset({"domain", "freq", "horizon", "lane_a", "lane_b", "lane_c", "lane_d",
                          "market", "burn"})
SPEC_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,23}$")
SPEC_KEYS: tuple[str, ...] = (
    "runner", "pool", "select", "universe", "timeframes", "horizons", "metrics", "score",
    "guards", "metric_weights", "gain_margin", "alpha", "test_family", "pay_curve", "weight",
    "until_block", "embargo_days", "embargo_eval_days", "from_block", "channels")
# what a NEW instance must name (the legacy ``mkt`` inherits [marketcast])
INSTANCE_REQUIRED: tuple[str, ...] = ("pool", "metrics", "universe", "timeframes", "horizons",
                                      "from_block")
# keys only an instance sets (mkt's start / channels are [marketcast]'s own)
INSTANCE_ONLY: tuple[str, ...] = ("from_block", "channels")
# channel names the series metrics read (SERIES_METRICS values)
CHANNEL_NAMES = ("open", "high", "low", "close", "volume", "funding", "price")
POOL_KEYS: tuple[str, ...] = ("prefix", "signer", "schema")

REFUSAL_CODES = ("unknown_key", "unknown_block", "runner_capacity", "bad_value",
                 "out_of_bounds", "runner_mismatch")


@dataclass(frozen=True)
class SpecBound:
    lo: float
    hi: float
    integer: bool = False
    lo_open: bool = False

    def problem(self, v: object) -> str | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"must be a number; got {v!r}"
        if self.integer and not isinstance(v, int):
            return f"must be an integer; got {v!r}"
        x = float(v)
        if not math.isfinite(x) or x < self.lo or x > self.hi or (self.lo_open and x == self.lo):
            return f"{v!r} outside {'(' if self.lo_open else '['}{self.lo}, {self.hi}]"
        return None


# Code bounds (W-OV style: the overlay envelope of a spec; changing one is a
# release). They contain every shipped value.
SPEC_BOUNDS: dict[str, SpecBound] = {
    "gain_margin": SpecBound(0.002, 0.25),
    "alpha": SpecBound(0.0, 0.10, lo_open=True),
    "weight": SpecBound(0.0, 100.0),
    "until_block": SpecBound(0, 10**12, integer=True),
    "from_block": SpecBound(1, 10**12, integer=True),
    "embargo_days": SpecBound(2, 120, integer=True),
    "embargo_eval_days": SpecBound(1, 119, integer=True),
}
MAX_METRIC_WEIGHT = 100.0
MAX_LIST = 512


@dataclass(frozen=True)
class SpecRefusal:
    id: str
    code: str
    reason: str

    def record(self) -> dict:
        return {"id": self.id, "refused": self.code, "reason": self.reason}


@dataclass(frozen=True)
class CompetitionSpec:
    """A parsed, accepted spec: ``id``, ``runner`` and the raw (validated)
    key → value table it sets (absent keys inherit the runner's table)."""
    id: str
    runner: str
    values: tuple[tuple[str, object], ...]

    def get(self, key: str, default=None):
        return dict(self.values).get(key, default)

    def record(self) -> dict:
        return {"id": self.id, "runner": self.runner,
                **{k: v for k, v in self.values if k != "runner"}}

    @property
    def digest(self) -> str:
        return spec_digest(self.record())


def spec_digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _str_list(v, what: str) -> list[str]:
    if isinstance(v, str) or not isinstance(v, (list, tuple)) or len(v) > MAX_LIST or not all(
            isinstance(x, str) and x for x in v) or len(set(v)) != len(v):
        raise _Bad("bad_value", f"{what} must be a list of unique non-empty strings")
    return list(v)


def _str_table(v, what: str, *, of_lists: bool) -> dict:
    if not isinstance(v, dict) or len(v) > MAX_LIST:
        raise _Bad("bad_value", f"{what} must be a table")
    out = {}
    for k, x in v.items():
        if not isinstance(k, str) or not k:
            raise _Bad("bad_value", f"{what} keys must be non-empty strings")
        if of_lists:
            out[k] = _str_list(x, f"{what}.{k}")
        elif isinstance(x, str) and x:
            out[k] = x
        else:
            raise _Bad("bad_value", f"{what}.{k} must be a non-empty string")
    return out


class _Bad(Exception):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


def parse_spec(sid: str, raw: object) -> CompetitionSpec | SpecRefusal:
    """One ``[lane_c.specs.<sid>]`` table → a spec or a typed refusal. Never
    raises (a refusal is the outcome, not an error)."""
    try:
        return _parse(str(sid), raw)
    except _Bad as e:
        return SpecRefusal(str(sid), e.code, e.reason)


def _parse(sid: str, raw: object) -> CompetitionSpec:
    if not isinstance(raw, dict):
        raise _Bad("bad_value", "a spec must be a table")
    extra = sorted(set(raw) - set(SPEC_KEYS))
    if extra:
        raise _Bad("unknown_key", f"unknown spec keys {extra} (this release knows "
                                  f"{list(SPEC_KEYS)})")
    runner = raw.get("runner")
    if runner not in RUNNERS:
        raise _Bad("unknown_block", f"runner {runner!r} is not in this release "
                                    f"({sorted(RUNNERS)})")
    hosts = RUNNERS[runner]
    if sid != LEGACY_INSTANCE and (sid not in hosts and "*" not in hosts
                                   or sid in RESERVED_IDS or not SPEC_ID_RE.match(sid)):
        raise _Bad("runner_capacity", f"runner {runner!r} cannot host id {sid!r} (ids are "
                                      "[a-z][a-z0-9_]{1,23}, not reserved)")
    if sid == LEGACY_INSTANCE:
        extra = [k for k in INSTANCE_ONLY if k in raw]
        if extra:
            raise _Bad("unknown_key", f"{extra} are instance keys; mkt's start and channels "
                                      "are [marketcast]'s own")
    else:
        miss = [k for k in INSTANCE_REQUIRED if k not in raw]
        if miss:
            raise _Bad("bad_value", f"a new competition must name {miss}")
    vals: dict[str, object] = {"runner": runner}
    if "channels" in raw:
        ch = _str_table(raw["channels"], "channels", of_lists=False)
        bad = [k for k in ch if k not in CHANNEL_NAMES]
        if bad:
            raise _Bad("unknown_block", f"channels maps {bad}; metric channels are "
                                        f"{list(CHANNEL_NAMES)}")
        vals["channels"] = dict(sorted(ch.items()))
    if "pool" in raw:
        p = raw["pool"]
        if not isinstance(p, dict) or set(p) - set(POOL_KEYS):
            raise _Bad("unknown_key", f"pool keys must be {list(POOL_KEYS)}")
        pool = {}
        for k in POOL_KEYS:
            if k in p:
                if not isinstance(p[k], str):
                    raise _Bad("bad_value", f"pool.{k} must be a string")
                pool[k] = p[k]
        if "schema" in pool and pool["schema"] not in POOL_SCHEMAS:
            raise _Bad("unknown_block", f"pool schema {pool['schema']!r} is not readable by "
                                        f"this release ({list(POOL_SCHEMAS)})")
        if "prefix" in pool and not pool["prefix"].strip():
            raise _Bad("bad_value", "pool.prefix must be non-empty")
        if sid != LEGACY_INSTANCE and not pool.get("prefix"):
            raise _Bad("bad_value", "a new competition must name pool.prefix")
        vals["pool"] = dict(sorted(pool.items()))
    if "select" in raw:
        vals["select"] = dict(sorted(_str_table(raw["select"], "select", of_lists=False).items()))
    for k in ("universe", "timeframes", "horizons"):
        if k in raw:
            vals[k] = {a: b for a, b in sorted(_str_table(raw[k], k, of_lists=True).items())}
    if "metrics" in raw:
        ms = _str_list(raw["metrics"], "metrics")
        bad = [m for m in ms if m not in SERIES_METRICS]
        if bad:
            raise _Bad("unknown_block", f"metrics {bad} are not in this release's registry "
                                        f"({sorted(SERIES_METRICS)})")
        if not ms:
            raise _Bad("bad_value", "metrics must be non-empty")
        vals["metrics"] = ms
    if "score" in raw:
        if raw["score"] not in SCORES:
            raise _Bad("unknown_block", f"score {raw['score']!r} not in {list(SCORES)}")
        vals["score"] = raw["score"]
    if "guards" in raw:
        gs = _str_list(raw["guards"], "guards")
        bad = [g for g in gs if g not in GUARD_METRICS]
        if bad:
            raise _Bad("unknown_block", f"guards {bad} not in {list(GUARD_METRICS)}")
        vals["guards"] = gs
    if "metric_weights" in raw:
        mw = raw["metric_weights"]
        if not isinstance(mw, dict) or not mw:
            raise _Bad("bad_value", "metric_weights must be a non-empty table")
        out = {}
        for m, w in sorted(mw.items()):
            if m not in SERIES_METRICS:
                raise _Bad("unknown_block", f"metric_weights names {m!r}, not in the registry")
            why = SpecBound(0.0, MAX_METRIC_WEIGHT, lo_open=True).problem(w)
            if why:
                raise _Bad("out_of_bounds", f"metric_weights.{m} {why}")
            out[m] = float(w)
        vals["metric_weights"] = out
    for k in ("test_family", "pay_curve"):
        if k in raw:
            menu = TEST_FAMILIES if k == "test_family" else PAY_CURVES
            if raw[k] not in menu:
                raise _Bad("unknown_block", f"{k} {raw[k]!r} not on the menu {list(menu)}")
            vals[k] = raw[k]
    for k, b in SPEC_BOUNDS.items():
        if k in raw:
            why = b.problem(raw[k])
            if why:
                raise _Bad("out_of_bounds", f"{k} {why}")
            vals[k] = int(raw[k]) if b.integer else float(raw[k])
    if "embargo_eval_days" in vals and int(vals["embargo_eval_days"]) >= int(
            vals.get("embargo_days", 10**9)):
        raise _Bad("out_of_bounds", "embargo_eval_days must be < embargo_days (the draw stays "
                                    "inside the embargo)")
    return CompetitionSpec(sid, str(runner), tuple(sorted(vals.items())))


def parse_specs(raw: object) -> tuple[tuple[CompetitionSpec, ...], tuple[SpecRefusal, ...]]:
    """``[lane_c] specs`` → ``(accepted, refused)``, id order. A non-table
    value refuses every spec it may have meant (``_all_``)."""
    if raw is None:
        return (), ()
    if not isinstance(raw, dict):
        return (), (SpecRefusal("_all_", "bad_value", "[lane_c] specs must be a table "
                                "<id> -> spec"),)
    ok, bad = [], []
    for sid in sorted(raw):
        r = parse_spec(sid, raw[sid])
        (bad if isinstance(r, SpecRefusal) else ok).append(r)
    return tuple(ok), tuple(bad)


def bound_refusals(raw: object) -> list[str]:
    """Overlay-path refusals (W-OV style): out-of-bounds / malformed values of
    KNOWN keys. Unknown names are not listed (they are a typed refusal at
    load, so a node one release behind burns only that competition)."""
    _ok, bad = parse_specs(raw)
    return [f"[lane_c.specs.{r.id}] {r.reason}" for r in bad
            if r.code in ("out_of_bounds", "bad_value")]


# ── the marketcast runner: compile a spec onto its tables ───────────────────


def compile_marketcast(spec: CompetitionSpec | None, mc_raw: dict, comp_raw: dict | None,
                       ) -> tuple[dict, dict | None]:
    """``([marketcast] raw, [lane_c.competitions.mkt] raw)`` with ``spec``
    written over them (None = unchanged). Pure; the loader then parses the
    result with the SAME parser and checks as a hand-written table, so a
    spec can never reach a value the tables could not."""
    if spec is None:
        return mc_raw, comp_raw
    mc = dict(mc_raw or {})
    v = dict(spec.values)
    pool = dict(v.get("pool") or {})
    if "prefix" in pool:
        mc["pool_prefix"] = pool["prefix"]
    if "signer" in pool:
        mc["pool_signer"] = pool["signer"]
    if "schema" in pool:
        mc["pool_schema"] = pool["schema"]
    if "select" in v:
        mc["pool_select"] = dict(v["select"])
    for sk, mk in (("universe", "assets"), ("timeframes", "frequencies"),
                   ("horizons", "horizons"), ("metrics", "suites"), ("guards", "guards"),
                   ("metric_weights", "composite_suite_weights"), ("gain_margin", "gain_margin"),
                   ("alpha", "alpha"), ("test_family", "test_family"),
                   ("pay_curve", "pay_curve"), ("embargo_days", "embargo_days"),
                   ("embargo_eval_days", "embargo_eval_days")):
        if sk in v:
            mc[mk] = v[sk]
    comp = None if comp_raw is None else dict(comp_raw)
    for k in ("weight", "until_block"):
        if k in v:
            comp = dict(comp or {})
            comp[k] = v[k]
    return mc, comp


# [marketcast] keys a spec INSTANCE never inherits from the legacy table: its
# data (it names its own), the market feeds and relays, its start, and the
# legacy competition's own derived lists
_NOT_INHERITED = ("from_block", "suites", "assets", "frequencies", "horizons", "data_sources",
                  "pool_prefix", "pool_signer", "pool_schema", "pool_select", "pool_channels",
                  "refused", "bar_relay", "native_intervals", "native_history_days",
                  "composite", "composite_suite_weights", "forward_horizons", "gap_horizons",
                  "min_dollar_volume_24h", "event_times",
                  # trainer rent policy (owner 2026-11-24), never a compiled table's
                  "funded_pod_skus")


def compile_instance(spec: CompetitionSpec, mc_raw: dict) -> tuple[dict, dict]:
    """W-CMP2: ``(instance [marketcast]-shaped raw table, its
    [lane_c.competitions.<id>] raw)`` of a spec INSTANCE: the legacy table's
    model pins, recipe bounds and verdict statistics, with the spec's data,
    metrics and rules written over them. Parsed with
    ``parse_marketcast(..., open_classes=True)`` by the loader."""
    base = {k: v for k, v in dict(mc_raw or {}).items() if k not in _NOT_INHERITED}
    base["min_dollar_volume_24h"] = 0.0      # no market volume in a non-market pool
    out, comp = compile_marketcast(spec, base, {})
    v = dict(spec.values)
    out["composite"] = len(list(v.get("metrics") or ())) >= 2
    if "channels" in v:
        out["pool_channels"] = dict(v["channels"])
    comp = dict(comp or {})
    comp["from_block"] = int(v["from_block"])
    return out, comp


def marketcast_spec(cfg) -> dict:
    """The ``[lane_c.specs.mkt]`` table that restates ``cfg``'s MarketCast
    competition exactly (every spec key). Compiling it back is the identity
    (``tests/accept/w_cmp``)."""
    mc, lc = cfg.marketcast, cfg.lane_c
    comp = lc.competition("mkt")
    out = {
        "runner": "marketcast",
        "pool": {"prefix": mc.pool_prefix, "signer": mc.pool_signer or "owner",
                 "schema": mc.pool_schema},
        "universe": {c: list(a) for c, a in mc.assets},
        "timeframes": {s: list(f) for s, f in mc.frequencies},
        "horizons": {s: list(h) for s, h in mc.horizons},
        "metrics": list(mc.suites),
        "score": "crps",
        "guards": list(mc.guards),
        "metric_weights": {s: float(w) for s, w in mc.composite_suite_weights},
        "gain_margin": float(mc.gain_margin), "alpha": float(mc.alpha),
        "test_family": mc.test_family, "pay_curve": mc.pay_curve,
        "embargo_days": int(mc.embargo_days), "embargo_eval_days": int(mc.embargo_eval_days),
    }
    if mc.pool_select:
        out["select"] = dict(mc.pool_select)
    if comp is not None and lc.competitions:
        out["weight"] = float(comp.weight)
        out["until_block"] = int(comp.until_block)
    return out


def specs_record(accepted, refused) -> dict:
    """What the lane-C config record carries: each accepted spec's digest
    and each refusal (so the audit re-derives which competitions ran)."""
    return {"specs": {s.id: s.digest for s in accepted},
            "refused": [r.record() for r in refused]}


def registry() -> dict:
    """The block registry as the overlay policy publishes it."""
    from .rules_menu import menu

    return {"runners": {k: list(v) for k, v in sorted(RUNNERS.items())},
            "series_metrics": {k: list(v) for k, v in sorted(SERIES_METRICS.items())},
            "scores": list(SCORES), "guards": list(GUARD_METRICS),
            "pool_schemas": [s for s in POOL_SCHEMAS if s], "spec_keys": list(SPEC_KEYS),
            "bounds": {k: [b.lo, b.hi] for k, b in sorted(SPEC_BOUNDS.items())},
            "menu": menu(), "lane_c_defaults": dict(zip(("test_family", "pay_curve"),
                                                        LANE_DEFAULTS["lane_c"], strict=True))}


__all__ = ["INSTANCE_REQUIRED", "LEGACY_INSTANCE", "RESERVED_IDS", "compile_instance",
           "GUARD_METRICS", "POOL_SCHEMAS", "REFUSAL_CODES", "RUNNERS", "SCORES",
           "SERIES_METRICS", "SPEC_BOUNDS", "SPEC_KEYS", "CompetitionSpec", "SpecRefusal",
           "bound_refusals", "compile_marketcast", "marketcast_spec", "parse_spec",
           "parse_specs", "registry", "spec_digest", "specs_record"]
