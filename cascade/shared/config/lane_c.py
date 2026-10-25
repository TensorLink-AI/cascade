"""[lane_c] fine-tuning lane config (DEC-CA-0056).

Lane C mirrors lane B (DEC-CA-0050): miner-funded legs, a separate signed
manifest bound into the lane-A manifest, per-target emission slots with the
lane-B decay schedule and TTL. It is a container for fine-tuning
COMPETITIONS; a target's kind names its competition and only a competition
with a forward-only eval is admitted (MarketCast-only lane C, owner
2026-10-20: today that is MarketCast, ``mkt:<suite>``, DEC-CA-0064).

Activation: ``from_block`` is a TYPED era-boundary block (0 = off, the
shipped state on mainnet and testnet). Like ``[v2] from_block`` the era
STARTING at it is the first lane-C era; a settlement at boundary ``B``
belongs to the era containing ``B − 1`` (DEC-CA-0043), so
:meth:`LaneCConfig.active` is ``B > from_block`` — the settlement AT the
block stays a pre-lane-C receipt byte-for-byte.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from .marketcast import DEFAULT_SUITES as MKT_DEFAULT_SUITES
from .v2 import FULL_PAY_TIERS

# ── competitions (MarketCast-only lane C, owner 2026-10-20) ─────────────────
#
# Lane C is a CONTAINER for fine-tuning competitions: a target is
# ``"<kind>:<value>"`` and its kind names the competition that judges it. A
# competition is admitted only with a FORWARD-ONLY eval — windows that close
# after the entry is trained, so nothing a miner can see before submitting is
# ever scored. The DEC-CA-0056 generic kinds (domain / freq / horizon on the
# public-feed ``pool_c`` snapshot) score the last horizon of public series
# revealed ~48 h later: scrapeable and memorisable (red-team 2026-10-20), so
# they are registered as NOT forward-only and refused everywhere. Today the
# only competition is MarketCast (DEC-CA-0064, ``mkt:<suite>``). A future
# competition is added by registering its kind + eval below.

PUBLIC_SUITES = ("gift", "boom", "time", "fev")
LANE_C_REMAINDERS = ("burn",)
LANE_C_TAKES_FROM = ("lane_b", "lane_a")
RECIPE_SCHEDULES = ("cosine", "constant", "linear")

_LABEL_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")

FORWARD_ONLY_REQUIRED = ("lane-C competitions need a forward-only eval — only MarketCast "
                         "(mkt:) has one; see DEC-CA-0056")


@dataclass(frozen=True)
class CompetitionKind:
    """One lane-C competition: its target prefix (``kind``), the eval that
    judges it and whether that eval is FORWARD-ONLY (the admission rule).

    ``from_block_table`` (concurrent competitions, owner 2026-10-20) names the
    sibling table that OWNS the competition's start block when its eval has
    its own go-live with its own arming checks — ``"marketcast"`` for
    ``mkt:`` (``[marketcast] from_block``, DEC-CA-0064) — so there is one
    source of truth; ``""`` = the competition's own ``[lane_c.competitions.
    <kind>] from_block``. ``parse_value`` validates a target's value (None =
    the kind has no generic parser; ``mkt`` is parsed against its suites)."""
    kind: str
    eval: str
    forward_only: bool
    from_block_table: str = ""
    parse_value: Callable[[str], str] | None = None


COMPETITION_KINDS: dict[str, CompetitionKind] = {}
# the sibling tables a competition's start block may be read from
FROM_BLOCK_TABLES = ("", "marketcast")


def register_competition_kind(kind: str, *, eval: str, forward_only: bool,
                              from_block_table: str = "",
                              parse_value: Callable[[str], str] | None = None
                              ) -> CompetitionKind:
    """Register a lane-C competition kind. ``forward_only`` must be given as
    a bool — a kind with no declared eval property is refused. A new kind is
    CODE (its eval ships with a release); configuring it is
    ``[lane_c.competitions.<kind>]``."""
    if not isinstance(forward_only, bool):
        raise ValueError(f"lane-C competition {kind!r} must declare forward_only as a bool; "
                         f"got {forward_only!r}")
    if not _LABEL_RE.match(str(kind)) or not str(eval or "").strip():
        raise ValueError(f"lane-C competition {kind!r} needs a [a-z0-9_-] kind and an eval")
    if from_block_table not in FROM_BLOCK_TABLES:
        raise ValueError(f"lane-C competition {kind!r}: from_block_table must be one of "
                         f"{FROM_BLOCK_TABLES}")
    if kind in COMPETITION_KINDS:
        raise ValueError(f"lane-C competition {kind!r} is already registered")
    ck = CompetitionKind(kind=str(kind), eval=str(eval), forward_only=forward_only,
                         from_block_table=str(from_block_table), parse_value=parse_value)
    COMPETITION_KINDS[ck.kind] = ck
    return ck


# DEC-CA-0064 MarketCast: ``mkt:<suite>`` is a lane-C target ONLY for the
# suites the caller admits (``mkt_suites``: ``[marketcast] suites`` at config
# load, ``[marketcast] target_suites`` — empty while MarketCast is inert — on
# the intake / commit / recipe surfaces).
MKT_KIND = "mkt"
register_competition_kind(MKT_KIND, eval="marketcast (DEC-CA-0064)", forward_only=True,
                          from_block_table="marketcast")
# DEC-CA-0056's generic targets: public-feed pool, NOT forward-only ⇒ refused.
for _k in ("domain", "freq", "horizon"):
    register_competition_kind(_k, eval="pool_c public-feed snapshot (retired)",
                              forward_only=False)
del _k

# The shipped targets: one per default [marketcast] suite.
DEFAULT_TARGETS: tuple[str, ...] = tuple(f"{MKT_KIND}:{s}" for s in MKT_DEFAULT_SUITES)


def is_mkt_target(spec: object) -> bool:
    """Whether ``spec`` is syntactically a MarketCast ``mkt:<suite>`` target
    (no validation — a routing test for the lane-C surfaces)."""
    return str(spec or "").strip().startswith(MKT_KIND + ":")


def parse_target(spec: str, *, mkt_suites: tuple[str, ...] = ()) -> tuple[str, str]:
    """``"<kind>:<value>"`` → ``(kind, value)``. The kind must be a
    registered FORWARD-ONLY competition (today only ``mkt``); an ``mkt``
    target's suite must be in ``mkt_suites``. Raises ``ValueError`` on
    anything else."""
    s = str(spec or "").strip()
    kind, sep, val = s.partition(":")
    ck = COMPETITION_KINDS.get(kind) if sep else None
    if ck is None:
        raise ValueError(f"lane-C target {spec!r} must be <competition>:<value> with a "
                         f"registered competition {tuple(COMPETITION_KINDS)}; "
                         f"{FORWARD_ONLY_REQUIRED}")
    if not ck.forward_only:
        raise ValueError(f"lane-C target {spec!r}: {FORWARD_ONLY_REQUIRED} ({kind}: is "
                         f"judged on {ck.eval}, which is not forward-only)")
    if kind == MKT_KIND:
        if not mkt_suites:
            raise ValueError(f"lane-C target {spec!r}: MarketCast admits no targets here "
                             "([marketcast] is off)")
        if val not in tuple(mkt_suites):
            raise ValueError(f"lane-C target {spec!r}: MarketCast suite must be one of "
                             f"{tuple(mkt_suites)}")
        return kind, val
    if ck.parse_value is not None:
        return kind, str(ck.parse_value(val))
    raise ValueError(f"lane-C target {spec!r}: competition {kind!r} has no parser")


def kind_of(target: object) -> str:
    """The competition kind of a target spec (the part before ``:``)."""
    return str(target or "").strip().partition(":")[0]


# ── concurrent competitions (owner 2026-10-20) ──────────────────────────────
#
# Several competitions may run in lane C at once. Lane C's share is split by
# competition WEIGHT first (normalised over every CONFIGURED competition —
# pending and ended ones included, so one competition starting or ending
# never changes another's pay: an idle competition's part BURNS), then
# equally among that competition's targets. Each competition has its own
# start (``from_block``; for ``mkt`` it IS ``[marketcast] from_block``) and an
# optional end (``until_block``, 0 = none). After the end nothing new is
# admitted or judged; existing wins wind down per ``wind_down``.

WIND_DOWN_MODES = ("finish_full_pay", "stop")
COMPETITION_STATES = ("pending", "active", "winding_down", "ended")
COMPETITION_KEYS = ("weight", "from_block", "until_block", "wind_down", "wind_down_max_eras")
# DEC-CA-0057: what an overlay may change on a CONFIGURED competition (the
# set of competitions, their start and their wind-down rule are
# release-only — a new kind needs code, a start has its own arming checks,
# and the wind-down rule is the promise a winner was paid under)
COMPETITION_OVERLAY_KEYS = ("weight", "until_block")
DEFAULT_WIND_DOWN = "finish_full_pay"
DEFAULT_WIND_DOWN_MAX_ERAS = 6


@dataclass(frozen=True)
class LaneCCompetition:
    """``[lane_c.competitions.<kind>]`` — one configured competition.

    ``weight`` (>= 0) is its part of lane C's share, normalised over every
    configured competition. ``from_block`` is its first era (0 = not
    started); a kind whose :attr:`CompetitionKind.from_block_table` is set
    reads it there instead (``mkt``: ``[marketcast] from_block``) and keeps 0
    here. ``until_block`` (0 = no end) is the era boundary it ends at: the
    settlement AT it is its last judged one. ``wind_down``: ``"finish_full_pay"``
    — a win held at the end keeps full pay while it is inside its own tiered
    full-pay period AND within ``wind_down_max_eras`` eras of the end, then
    stops (no decay tail); ``"stop"`` — pay stops at the end."""
    kind: str
    weight: float = 1.0
    from_block: int = 0
    until_block: int = 0
    wind_down: str = DEFAULT_WIND_DOWN
    wind_down_max_eras: int = DEFAULT_WIND_DOWN_MAX_ERAS

    def record(self) -> dict:
        return {"kind": self.kind, "weight": float(self.weight),
                "from_block": int(self.from_block), "until_block": int(self.until_block),
                "wind_down": self.wind_down, "wind_down_max_eras": int(self.wind_down_max_eras)}


def check_target(spec: str, *, mkt_suites: tuple[str, ...] = ()) -> str:
    """:func:`parse_target` → the canonical spec string."""
    kind, val = parse_target(spec, mkt_suites=mkt_suites)
    return f"{kind}:{val}"


@dataclass(frozen=True)
class LaneCConfig:
    """``[lane_c]`` — the fine-tuning lane (DEC-CA-0056).

    CONSENSUS terms (every validator must agree; replayed by cascade-audit):
    ``from_block``, ``share`` / ``takes_from`` / ``remainder`` (the emission
    split), ``targets`` (one emission slot each), the payout schedule
    (``full_pay_tiers`` / ``half_life_rounds`` / ``floor`` in ERAS,
    ``win_ttl_eras``), the verdict (``alpha``, ``bootstrap_B``,
    ``gain_margin``, ``min_target_windows``, ``min_target_clusters``,
    ``n_windows``, ``guard_public``, ``tau_public``, ``min_public_items``).
    MarketCast's windows come from ``[marketcast] pool_prefix``; lane C has
    no pool of its own (the DEC-CA-0056 ``pool_c`` is retired).
    TRAINER POLICY (no consensus impact): ``leg_hours``, ``legs_per_target``,
    ``max_base_age_eras``, ``max_lr_scale``, ``max_steps_fraction`` — the
    recipe bounds (:func:`cascade.shared.lane_c.parse_recipe`).
    """

    from_block: int = 0
    share: float = 0.25
    takes_from: str = "lane_b"
    remainder: str = "burn"
    targets: tuple[str, ...] = DEFAULT_TARGETS
    # payout schedule in ERAS (owner-tunable), exactly lane B's rule: the
    # full-pay period is GAIN-TIERED (owner 2026-10-20) — the ``full_eras``
    # of the highest ``full_pay_tiers`` threshold the confirmed win's gain LCB
    # meets, fixed at the win — then the decay below. An ``mkt:`` win uses
    # ``[marketcast] full_pay_tiers`` instead (DEC-CA-0064: a different
    # statistic), on this TTL and decay.
    full_pay_tiers: tuple[tuple[float, int], ...] = FULL_PAY_TIERS
    half_life_rounds: float = 1.0
    floor: float = 0.1
    win_ttl_eras: int = 28
    # verdict: gain LCB_α of (base − fine-tune)/base on the target's windows
    # must clear ``gain_margin`` (> 0: a null fine-tune must not win ~α of the
    # time); optional public non-inferiority guard per target
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    gain_margin: float = 0.01
    min_target_windows: int = 100
    min_target_clusters: int = 5     # distinct sources among the target windows
    n_windows: int = 1000            # windows drawn per target per settlement
    guard_public: tuple[tuple[str, tuple[str, ...]], ...] = ()
    tau_public: float = 0.01
    min_public_items: int = 10
    # trainer policy
    leg_hours: float = 1.0
    legs_per_target: int = 2
    max_base_age_eras: int = 2
    max_lr_scale: float = 1.0
    max_steps_fraction: float = 1.0
    # Legacy recovery hint, retained for config compatibility. Recovery now
    # walks to the saved boundary / activation; TTL cannot bound slot state.
    replay_settlements: int = 64
    # concurrent competitions (owner 2026-10-20), CONSENSUS: one entry per
    # configured competition, in kind order; () = derived — one weight-1,
    # never-ending competition per kind in ``targets`` (:attr:`competition_list`)
    competitions: tuple[LaneCCompetition, ...] = ()

    @property
    def competition_list(self) -> tuple[LaneCCompetition, ...]:
        """The configured competitions (``competitions``, or the derived
        default: one weight-1, never-ending competition per target kind)."""
        if self.competitions:
            return tuple(self.competitions)
        return tuple(LaneCCompetition(kind=k)
                     for k in sorted(dict.fromkeys(kind_of(t) for t in self.targets)))

    def competition(self, kind: str) -> LaneCCompetition | None:
        return next((c for c in self.competition_list if c.kind == str(kind)), None)

    def competition_of(self, target: str) -> LaneCCompetition | None:
        return self.competition(kind_of(target))

    def competition_targets(self, kind: str) -> tuple[str, ...]:
        """``kind``'s targets, in ``targets`` order."""
        return tuple(t for t in self.targets if kind_of(t) == str(kind))

    def active(self, block: int | None) -> bool:
        """Whether the SETTLEMENT at boundary ``block`` judges lane C (its era
        starts at or after ``from_block``)."""
        return self.from_block > 0 and block is not None and int(block) > self.from_block

    def era_active(self, era_start: int | None) -> bool:
        """Whether the era STARTING at ``era_start`` is a lane-C era (the
        trainer's leg-scheduling test)."""
        return (self.from_block > 0 and era_start is not None
                and int(era_start) >= self.from_block)

    @property
    def guard_public_map(self) -> dict[str, tuple[str, ...]]:
        return {t: tuple(s) for t, s in self.guard_public}

    def guard_suites(self, target: str) -> tuple[str, ...]:
        return self.guard_public_map.get(str(target), ())

    def verdict_params(self, k: int = 1):
        """The expert-lane params; ``k`` > 1 Bonferroni-corrects ``alpha`` to
        ``alpha / k`` (one target's ``k`` judged entries at one settlement)."""
        from ...eval.expert_lane import ExpertLaneParams

        return ExpertLaneParams(
            alpha=self.alpha / max(1, int(k)), B=self.bootstrap_B,
            gain_margin=self.gain_margin, tau_public=self.tau_public,
            min_target_windows=self.min_target_windows,
            min_target_clusters=self.min_target_clusters,
            min_public_items=self.min_public_items,
        )


def _check_listed(spec: str, mkt_suites: tuple[str, ...]) -> str:
    try:
        return check_target(spec, mkt_suites=mkt_suites)
    except ValueError as e:
        raise ValueError(f"[lane_c] {e}") from e


def _num(raw, name: str, *, integer: bool = False):
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or (
            integer and not isinstance(raw, int)):
        raise ValueError(f"[lane_c] {name} must be {'an integer' if integer else 'a number'}; "
                         f"got {raw!r}")
    return int(raw) if integer else float(raw)


def _parse_competitions(raw, targets: tuple[str, ...], *, lane_c_from: int,
                        kind_from_blocks: dict[str, int], era_len: Callable[[int], int]
                        ) -> tuple[LaneCCompetition, ...]:
    """``[lane_c.competitions]`` → the configured competitions (absent ⇒ ()
    = derived). Every target's kind must be configured; a configured kind
    must be a registered forward-only competition; ``weight`` >= 0 with a
    positive total; ``from_block`` / ``until_block`` era boundaries with
    ``[lane_c] from_block <= from_block < until_block`` (0 = unset)."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ValueError(f"[lane_c] competitions must be a table <kind> -> {{weight, "
                         f"from_block, until_block, ...}}; got {raw!r}")
    out: list[LaneCCompetition] = []
    for kind in sorted(raw):
        where = f"[lane_c.competitions.{kind}]"
        ck = COMPETITION_KINDS.get(str(kind))
        if ck is None:
            raise ValueError(f"{where}: unknown competition {kind!r} — a competition kind is "
                             f"code (registered: {tuple(COMPETITION_KINDS)}); "
                             f"{FORWARD_ONLY_REQUIRED}")
        if not ck.forward_only:
            raise ValueError(f"{where}: {FORWARD_ONLY_REQUIRED} ({kind}: is judged on "
                             f"{ck.eval}, which is not forward-only)")
        t = raw[kind]
        if not isinstance(t, dict):
            raise ValueError(f"{where} must be a table")
        extra = sorted(set(t) - set(COMPETITION_KEYS))
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}; allowed {list(COMPETITION_KEYS)}")
        if ck.from_block_table and "from_block" in t:
            raise ValueError(f"{where}: from_block is [{ck.from_block_table}] from_block "
                             "(one source of truth) — set it there")
        w = _num(t.get("weight", 1.0), f"competitions.{kind}.weight")
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"{where}: weight must be a finite number >= 0; got {w}")
        mode = str(t.get("wind_down", DEFAULT_WIND_DOWN))
        if mode not in WIND_DOWN_MODES:
            raise ValueError(f"{where}: wind_down must be one of {WIND_DOWN_MODES}; got {mode!r}")
        c = LaneCCompetition(
            kind=str(kind), weight=w,
            from_block=_num(t.get("from_block", 0), f"competitions.{kind}.from_block",
                            integer=True),
            until_block=_num(t.get("until_block", 0), f"competitions.{kind}.until_block",
                             integer=True),
            wind_down=mode,
            wind_down_max_eras=_num(t.get("wind_down_max_eras", DEFAULT_WIND_DOWN_MAX_ERAS),
                                    f"competitions.{kind}.wind_down_max_eras", integer=True))
        if c.from_block < 0 or c.until_block < 0 or c.wind_down_max_eras < 0:
            raise ValueError(f"{where}: from_block, until_block and wind_down_max_eras must "
                             "be >= 0")
        fb = (int(kind_from_blocks.get(c.kind, 0) or 0) if ck.from_block_table
              else c.from_block)
        src = (f"[{ck.from_block_table}] from_block" if ck.from_block_table
               else f"{where} from_block")
        for name, b in (("from_block", c.from_block), ("until_block", c.until_block)):
            eb = int(era_len(b)) if b else 0
            if b and eb > 0 and b % eb:
                raise ValueError(f"{where}: {name}={b} must be an era boundary (a multiple of "
                                 f"the era length {eb} in force at it)")
        if c.from_block and (not lane_c_from or c.from_block < int(lane_c_from)):
            raise ValueError(f"{where}: from_block={c.from_block} must be >= [lane_c] "
                             f"from_block ({lane_c_from or 'unset'})")
        if c.until_block and (not fb or c.until_block <= fb):
            raise ValueError(f"{where}: until_block={c.until_block} must be after the "
                             f"competition's start ({src} = {fb or 'unset'})")
        out.append(c)
    kinds = {c.kind for c in out}
    for tgt in targets:
        if kind_of(tgt) not in kinds:
            raise ValueError(f"[lane_c] target {tgt!r}: its competition {kind_of(tgt)!r} has "
                             f"no [lane_c.competitions.{kind_of(tgt)}] entry")
    if out and not sum(c.weight for c in out) > 0:
        raise ValueError("[lane_c] competitions: the weights must sum to > 0")
    return tuple(out)


def parse_lane_c(lc: dict, *, v2_from_block: int = 0, era_blocks: int = 0,
                 era_settlements: int = 0, lane_b_share: float | None = None,
                 mkt_suites: tuple[str, ...] | None = None,
                 kind_from_blocks: dict[str, int] | None = None,
                 era_blocks_at: Callable[[int], int] | None = None) -> LaneCConfig:
    """Parse + guard ``[lane_c]`` (absent section ⇒ inert defaults).

    ``era_blocks`` (the era length on the grid in force at ``from_block``) and
    ``v2_from_block`` gate an armed block: an era boundary, at or after the v2
    go-live (lane C launches after lane B, with its allocation taken from
    ``takes_from``). ``mkt_suites`` (``[marketcast] suites``,
    DEC-CA-0064 — the CONFIGURED suites, default its default ones, so the
    shipped targets load while MarketCast is off; :func:`cascade.shared.lane_c.active_targets` then
    counts none of them) admits ``mkt:<suite>`` targets. Every target must
    belong to a forward-only competition (:data:`COMPETITION_KINDS`); a
    public guard on an ``mkt:`` target is refused, and ``pool_prefix`` is
    retired.

    ``[lane_c.competitions.<kind>]`` (concurrent competitions, owner
    2026-10-20; :func:`_parse_competitions`): ``kind_from_blocks`` gives the
    start block of a kind whose start lives in a sibling table (``mkt`` →
    ``[marketcast] from_block``), ``era_blocks_at(block)`` the era length in
    force at a competition block (default ``era_blocks``)."""
    from .v2 import _refuse_full_rounds, lane_b_replay_cover, parse_full_pay_tiers

    mkt_suites = tuple(MKT_DEFAULT_SUITES if mkt_suites is None else mkt_suites)
    lc = dict(lc or {})
    _refuse_full_rounds(lc, "lane_c")
    if "pool_prefix" in lc:
        raise ValueError("[lane_c] pool_prefix is retired: lane C has no pool of its own "
                         "(MarketCast reads [marketcast] pool_prefix; see DEC-CA-0056)")
    fb = _num(lc.get("from_block", 0), "from_block", integer=True)
    if fb < 0:
        raise ValueError(f"[lane_c] from_block must be >= 0; got {fb}")
    ttl = _num(lc.get("win_ttl_eras", 28), "win_ttl_eras", integer=True)
    cover = lane_b_replay_cover(ttl, era_settlements)
    if "replay_settlements" in lc:
        replay = _num(lc["replay_settlements"], "replay_settlements", integer=True)
        if cover and era_settlements and replay < cover:
            raise ValueError(
                f"[lane_c] replay_settlements={replay} does not cover the payout TTL "
                f"(= {cover} settlements)")
    else:
        replay = cover if (cover and era_settlements) else 64
    raw_targets = lc.get("targets", DEFAULT_TARGETS)
    if isinstance(raw_targets, str) or not isinstance(raw_targets, (list, tuple)):
        raise ValueError(f"[lane_c] targets must be a list of target specs; got {raw_targets!r}")
    targets = tuple(_check_listed(str(t), mkt_suites) for t in raw_targets)
    gp_raw = lc.get("guard_public", {}) or {}
    if not isinstance(gp_raw, dict):
        raise ValueError(f"[lane_c] guard_public must be a table target -> [suites]; "
                         f"got {gp_raw!r}")
    guard: list[tuple[str, tuple[str, ...]]] = []
    for t, suites in sorted(gp_raw.items()):
        spec = _check_listed(str(t), mkt_suites)
        if is_mkt_target(spec):
            raise ValueError(f"[lane_c] guard_public names {spec!r}: public bench guards are "
                             "off for MarketCast targets (DEC-CA-0064)")
        if spec not in targets:
            raise ValueError(f"[lane_c] guard_public names {spec!r}, which is not a target")
        if isinstance(suites, str) or not isinstance(suites, (list, tuple)):
            raise ValueError(f"[lane_c] guard_public[{spec!r}] must be a list of suites")
        ss = tuple(str(s) for s in suites)
        bad = [s for s in ss if s not in PUBLIC_SUITES]
        if bad or len(set(ss)) != len(ss):
            raise ValueError(f"[lane_c] guard_public[{spec!r}] suites must be unique members "
                             f"of {PUBLIC_SUITES}; got {list(ss)}")
        if ss:
            guard.append((spec, ss))
    cfg = LaneCConfig(
        from_block=fb,
        share=_num(lc.get("share", 0.25), "share"),
        takes_from=str(lc.get("takes_from", "lane_b")),
        remainder=str(lc.get("remainder", "burn")),
        targets=targets,
        full_pay_tiers=parse_full_pay_tiers(lc.get("full_pay_tiers", FULL_PAY_TIERS),
                                            section="lane_c", win_ttl_eras=ttl),
        half_life_rounds=_num(lc.get("half_life_rounds", 1.0), "half_life_rounds"),
        floor=_num(lc.get("floor", 0.1), "floor"),
        win_ttl_eras=ttl,
        alpha=_num(lc.get("alpha", 0.05), "alpha"),
        bootstrap_B=_num(lc.get("bootstrap_B", 10_000), "bootstrap_B", integer=True),
        gain_margin=_num(lc.get("gain_margin", 0.01), "gain_margin"),
        min_target_windows=_num(lc.get("min_target_windows", 100), "min_target_windows",
                                integer=True),
        min_target_clusters=_num(lc.get("min_target_clusters", 5), "min_target_clusters",
                                 integer=True),
        n_windows=_num(lc.get("n_windows", 1000), "n_windows", integer=True),
        guard_public=tuple(guard),
        tau_public=_num(lc.get("tau_public", 0.01), "tau_public"),
        min_public_items=_num(lc.get("min_public_items", 10), "min_public_items", integer=True),
        leg_hours=_num(lc.get("leg_hours", 1.0), "leg_hours"),
        legs_per_target=_num(lc.get("legs_per_target", 2), "legs_per_target", integer=True),
        max_base_age_eras=_num(lc.get("max_base_age_eras", 2), "max_base_age_eras",
                               integer=True),
        max_lr_scale=_num(lc.get("max_lr_scale", 1.0), "max_lr_scale"),
        max_steps_fraction=_num(lc.get("max_steps_fraction", 1.0), "max_steps_fraction"),
        replay_settlements=replay,
        competitions=_parse_competitions(
            lc.get("competitions"), targets, lane_c_from=fb,
            kind_from_blocks=dict(kind_from_blocks or {}),
            era_len=era_blocks_at or (lambda _b: int(era_blocks))),
    )
    if not 0.0 <= cfg.share <= 1.0:
        raise ValueError(f"[lane_c] share must be in [0, 1]; got {cfg.share}")
    if cfg.takes_from not in LANE_C_TAKES_FROM:
        raise ValueError(f"[lane_c] takes_from must be one of {LANE_C_TAKES_FROM}")
    if cfg.remainder not in LANE_C_REMAINDERS:
        raise ValueError(f"[lane_c] remainder must be one of {LANE_C_REMAINDERS}")
    if len(set(cfg.targets)) != len(cfg.targets):
        raise ValueError("[lane_c] targets must be unique")
    if not 0.0 < cfg.floor <= 1.0:
        raise ValueError(f"[lane_c] floor must be in (0, 1]; got {cfg.floor}")
    if cfg.half_life_rounds <= 0:
        raise ValueError("[lane_c] half_life_rounds must be > 0")
    if cfg.win_ttl_eras < 0:
        raise ValueError(f"[lane_c] win_ttl_eras={cfg.win_ttl_eras} must be 0 (no expiry) "
                         "or > 0")
    if not 0.0 < cfg.alpha < 0.5:
        raise ValueError(f"[lane_c] alpha must be in (0, 0.5); got {cfg.alpha}")
    if cfg.bootstrap_B < 100:
        raise ValueError(f"[lane_c] bootstrap_B must be >= 100; got {cfg.bootstrap_B}")
    if not 0.0 < cfg.gain_margin < 1.0:
        raise ValueError(f"[lane_c] gain_margin must be in (0, 1) — a zero bar lets a null "
                         f"fine-tune win ~alpha of the time; got {cfg.gain_margin}")
    if cfg.min_target_windows < 1 or cfg.n_windows < cfg.min_target_windows:
        raise ValueError("[lane_c] need 1 <= min_target_windows <= n_windows")
    if not 1 <= cfg.min_target_clusters <= cfg.min_target_windows:
        raise ValueError("[lane_c] need 1 <= min_target_clusters <= min_target_windows")
    if not 0.0 <= cfg.tau_public < 1.0 or cfg.min_public_items < 1:
        raise ValueError("[lane_c] tau_public must be in [0, 1) and min_public_items >= 1")
    if cfg.leg_hours <= 0 or cfg.legs_per_target < 1 or cfg.max_base_age_eras < 0:
        raise ValueError("[lane_c] leg_hours > 0, legs_per_target >= 1, "
                         "max_base_age_eras >= 0")
    if not 0.0 < cfg.max_lr_scale <= 10.0 or not 0.0 < cfg.max_steps_fraction <= 1.0:
        raise ValueError("[lane_c] max_lr_scale must be in (0, 10] and max_steps_fraction "
                         "in (0, 1]")
    if cfg.replay_settlements < 0:
        raise ValueError("[lane_c] replay_settlements must be >= 0")
    if fb:
        if not cfg.targets:
            raise ValueError("[lane_c] from_block set but no targets")
        if not int(v2_from_block):
            raise ValueError(f"[lane_c] from_block={fb} needs [v2] from_block: lane C "
                             "runs on the v2 era schedule")
        if fb < int(v2_from_block):
            raise ValueError(f"[lane_c] from_block={fb} precedes [v2] from_block="
                             f"{v2_from_block}")
        if int(era_blocks) > 0 and fb % int(era_blocks):
            raise ValueError(f"[lane_c] from_block={fb} must be an era boundary (a multiple "
                             f"of the era length {era_blocks} in force at it)")
    # Checked armed or not, so an overlay (DEC-CA-0057) can never stage a
    # composition that would only fail at the block that arms it.
    if lane_b_share is not None:
        available = (1.0 - float(lane_b_share) if cfg.takes_from == "lane_a"
                     else float(lane_b_share))
        if cfg.share > available + 1e-12:
            raise ValueError(f"[lane_c] share={cfg.share} exceeds the available "
                             f"{cfg.takes_from} share={available} it is taken from")
    return cfg


__all__ = [
    "COMPETITION_KEYS", "COMPETITION_KINDS", "COMPETITION_OVERLAY_KEYS", "COMPETITION_STATES",
    "CompetitionKind", "DEFAULT_TARGETS", "DEFAULT_WIND_DOWN", "DEFAULT_WIND_DOWN_MAX_ERAS",
    "FORWARD_ONLY_REQUIRED", "FROM_BLOCK_TABLES", "LANE_C_REMAINDERS", "LANE_C_TAKES_FROM",
    "LaneCCompetition", "LaneCConfig", "MKT_KIND", "PUBLIC_SUITES", "RECIPE_SCHEDULES",
    "WIND_DOWN_MODES", "check_target", "is_mkt_target", "kind_of", "parse_lane_c",
    "parse_target", "register_competition_kind",
]
