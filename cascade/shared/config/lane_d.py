"""``[lane_d]`` — lane D "open weights" (LANE_D_PLAN §1, §14.2; DEC-CA-0062).

The table parses and lands on :attr:`ChainConfig.lane_d`. Activation is a
signed-overlay ``from_block`` at an era boundary (D10, D45) — never a
chain.toml edit and never a DEC-CA-0045 vote — and the kill switch is an
overlay ``until_block`` or ``share = 0`` (T21). Every key except
``remainder`` (the burn destination, release-only; ``netconfig``) is
overlay-settable. With ``from_block = 0`` (shipped) the lane is never active
and no consensus surface reads the table, so an unarmed table moves no
digest, receipt, verdict or weight vector (invariant 1).

Retired (rev6.18 B11): ``confirm_on_king_change`` and the DEC-CA-0062 keys
(``targets``, ``teacher_*``, ``real_corpus_ref``, ``data_sources``,
``augmentations``, ``expert_choice``, ``max_context_length``,
``pool_prefix``, ``gain_margin``, the payout schedule). No lane D code read
them; the parser ignores them (:data:`RETIRED_LANE_D_KEYS`, listed in the
loader's ``RETIRED_KEYS``) so a deployed file that still carries them loads.
``notes`` is still carried for the dashboards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

LANE_D_TAKES_FROM = ("lane_a", "lane_b", "lane_c")
LANE_D_REMAINDERS = ("burn",)
# D2 (rev6.6): the context variants an entry may DECLARE, all of them from
# launch; 131072 is the cap. Arming is computed per settlement from its two
# gates (cascade.eval.lane_d_scorer.armed_record), never configured here.
LANE_D_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)
LANE_D_MAX_CONTEXT = 131072
# §1.1: any lane D horizon is at most 16 CPM patches x 32 points.
LANE_D_MAX_HORIZON = 512
# D14: BF16 only.
LANE_D_DTYPES = ("BF16",)
# rev6.18 B11: keys a deployed [lane_d] table may still carry that no code
# reads. The parser ignores them; never re-use one of these names for a knob.
RETIRED_LANE_D_KEYS = frozenset({
    "confirm_on_king_change",
    "targets", "teacher_ref", "teacher_api", "teacher_mode", "teacher_models",
    "real_corpus_ref", "data_sources", "augmentations", "expert_choice",
    "max_context_length", "pool_prefix", "gain_margin", "full_rounds",
    "half_life_rounds", "floor", "win_ttl_eras",
})
# §16 suite names (data, not a closed list: any non-empty unique names load).
LANE_D_GATE_SUITES = ("gift", "boom", "time", "fev")
# rev6.18 F4 (A7, D24): keys a deployed [lane_d] table may still carry that no
# code reads any more: the W5i backtest cap / floor (D24, backtest dropped)
# and the A10 low-frequency slice (A7, the check removed). The parser
# ignores them (listed in the loader's RETIRED_KEYS) so such a file loads;
# never re-use one of these names for a knob.
RETIRED_LANE_D_F4_KEYS = frozenset({"backtest_weight_cap", "backtest_ratio_floor",
                                    "lowfreq_freqs"})


@dataclass(frozen=True)
class LaneDConfig:
    # Activation (D10/D45): 0 = off; an era-boundary block arms it. The lane
    # closes after ``until_block`` (0 = no end; the T21 kill switch).
    from_block: int = 0
    until_block: int = 0
    # Emissions: ``share`` of the total, carved from ``takes_from`` (D1/D48
    # move it to lane_a by overlay); the unpaid remainder burns (D21).
    share: float = 0.0
    takes_from: str = "lane_b"
    remainder: str = "burn"
    # What an entry is judged at (§1.1, D2, D14).
    contexts: tuple[int, ...] = LANE_D_CONTEXTS
    horizon: int = 64
    dtypes: tuple[str, ...] = LANE_D_DTYPES
    # Forward-only judging (§1.3, D5).
    forward_margin_hours: float = 6.0
    min_windows: int = 1000
    min_clusters: int = 20
    min_long_series: int = 300
    max_source_share: float = 0.25
    d_margin: float = 0.01
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    # W2e (rev6.10 (a)/(b), rev6.11 (a), rev6.12 (b)): the per-series embargo
    # top-up (0 = the rev3 rule), the per-settlement window cap (>= min_windows)
    # and the minimum fresh share of a frozen set (1 = fresh windows only).
    embargo_hours: float = 48.0
    max_windows_per_settlement: int = 5000
    min_fresh_window_share: float = 0.5
    # W2e (rev6.18 A3, F4): the public half's weight in the combined score
    # L = (1 - w) L_live + w L_pub (0 = the public half off: not drawn, its
    # point-gain condition not applied) and the per-window ratio clamp
    # [floor, 1 / floor]. Both overlay-settable; no cross-key rule.
    public_weight: float = 0.5
    window_ratio_floor: float = 0.5
    # Rotating external-benchmark gate (§1.3b, D2b).
    gate_suites: tuple[str, ...] = LANE_D_GATE_SUITES
    gate_weights: tuple[tuple[str, float], ...] = tuple((s, 1.0) for s in LANE_D_GATE_SUITES)
    tau_public: float = 0.01
    # Court (§1.4, D7).
    reward_prior_kings: int = 2
    king_decay: float = 0.5
    # Rate limits, queue and TTLs (D3, D8, A1, A8, A9, B10).
    max_entries_per_settlement: int = 8
    max_open_per_coldkey: int = 2
    max_judged_per_coldkey_per_settlement: int = 2
    max_coldkey_unresolved_settlements: int = 4
    max_reveal_delay_days: int = 14
    max_bench_holds: int = 4
    loser_ttl_eras: int = 4
    max_mirror_lag_hours: float = 6.0
    ready_ttl_blocks: int = 7200
    # Free-form owner notes carried through for the dashboards.
    notes: dict = field(default_factory=dict)

    def active(self, boundary: int | None) -> bool:
        """Whether the SETTLEMENT at ``boundary`` judges lane D: its era starts
        at or after ``from_block`` (the settlement AT ``from_block`` belongs to
        the era before it, as in lane C) and it is not past ``until_block``."""
        if self.from_block <= 0 or boundary is None:
            return False
        b = int(boundary)
        return b > self.from_block and (self.until_block <= 0 or b <= self.until_block)


def _int(raw: object, name: str, *, lo: int, hi: int | None = None) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"[lane_d] {name} must be an integer; got {raw!r}")
    if raw < lo or (hi is not None and raw > hi):
        span = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise ValueError(f"[lane_d] {name} must be {span}; got {raw!r}")
    return int(raw)


def _float(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"[lane_d] {name} must be a number; got {raw!r}")
    return float(raw)


def _open_unit(raw: object, name: str) -> float:
    """A float in (0, 1)."""
    v = _float(raw, name)
    if not 0.0 < v < 1.0:
        raise ValueError(f"[lane_d] {name} must be in (0, 1); got {raw!r}")
    return v


def _half_open_unit(raw: object, name: str) -> float:
    """A float in (0, 1]."""
    v = _float(raw, name)
    if not 0.0 < v <= 1.0:
        raise ValueError(f"[lane_d] {name} must be in (0, 1]; got {raw!r}")
    return v


def _names(raw: object, name: str) -> tuple[str, ...]:
    """A non-empty list of unique non-empty strings."""
    if (not isinstance(raw, (list, tuple)) or not raw
            or not all(isinstance(x, str) and x.strip() for x in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError(f"[lane_d] {name} must be a non-empty list of unique non-empty "
                         f"strings; got {raw!r}")
    return tuple(raw)


def _contexts(raw: object) -> tuple[int, ...]:
    if (not isinstance(raw, (list, tuple)) or not raw
            or not all(type(c) is int and c in LANE_D_CONTEXTS for c in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError(f"[lane_d] contexts must be a non-empty list of unique values from "
                         f"{list(LANE_D_CONTEXTS)} (cap {LANE_D_MAX_CONTEXT}); got {raw!r}")
    return tuple(int(c) for c in raw)


def _dtypes(raw: object) -> tuple[str, ...]:
    if (not isinstance(raw, (list, tuple)) or not raw
            or not all(isinstance(d, str) and d in LANE_D_DTYPES for d in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError(f"[lane_d] dtypes must be a non-empty subset of {list(LANE_D_DTYPES)} "
                         f"(D14: BF16 only); got {raw!r}")
    return tuple(raw)


def _gate_weights(raw: object, suites: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    """``{suite = weight}`` over ``gate_suites`` (absent ⇒ equal over them),
    held as ``(suite, weight)`` pairs in ``gate_suites`` order."""
    if raw is None:
        return tuple((s, 1.0) for s in suites)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"[lane_d] gate_weights must be a table suite -> weight; got {raw!r}")
    unknown = sorted(str(k) for k in raw if k not in suites)
    if unknown:
        raise ValueError(f"[lane_d] gate_weights names {unknown}, not in gate_suites "
                         f"{list(suites)}")
    out = []
    for s in suites:
        if s not in raw:
            continue
        w = _float(raw[s], f"gate_weights.{s}")
        if w < 0.0:
            raise ValueError(f"[lane_d] gate_weights.{s} must be >= 0; got {w!r}")
        out.append((s, w))
    if sum(w for _, w in out) <= 0.0:
        raise ValueError("[lane_d] gate_weights must have a positive sum")
    return tuple(out)


def parse_lane_d(raw: object, *, era_blocks_at: Callable[[int], int] | None = None
                 ) -> LaneDConfig:
    """Parse + guard ``[lane_d]`` (absent table ⇒ the defaults, never active).

    ``era_blocks_at(block)`` is the era length on the grid in force at
    ``block`` (the loader passes it): an armed ``from_block`` and a set
    ``until_block`` must be era boundaries on it, with ``until_block`` after
    ``from_block``. Without it (a bare call) only those orderings are checked."""
    ld = raw if isinstance(raw, dict) else {}

    def on_era(name: str, block: int) -> None:
        era = max(1, int(era_blocks_at(block))) if era_blocks_at is not None else 1
        if block % era:
            raise ValueError(f"[lane_d] {name}={block} must be an era boundary "
                             f"(a multiple of {era})")

    fb = _int(ld.get("from_block", 0), "from_block", lo=0)
    if fb:
        on_era("from_block", fb)
    ub = _int(ld.get("until_block", 0), "until_block", lo=0)
    if ub:
        if ub <= fb:
            raise ValueError(f"[lane_d] until_block={ub} must follow from_block={fb}")
        on_era("until_block", ub)
    share = _float(ld.get("share", 0.0), "share")
    if not 0.0 <= share < 1.0:
        raise ValueError(f"[lane_d] share must be in [0, 1); got {share!r}")
    takes = ld.get("takes_from", "lane_b")
    if takes not in LANE_D_TAKES_FROM:
        raise ValueError(f"[lane_d] takes_from must be one of {LANE_D_TAKES_FROM}; got {takes!r}")
    rem = ld.get("remainder", "burn")
    if rem not in LANE_D_REMAINDERS:
        raise ValueError(f"[lane_d] remainder must be one of {LANE_D_REMAINDERS}; got {rem!r}")

    contexts = _contexts(ld.get("contexts", list(LANE_D_CONTEXTS)))
    horizon = _int(ld.get("horizon", 64), "horizon", lo=1, hi=LANE_D_MAX_HORIZON)
    dtypes = _dtypes(ld.get("dtypes", list(LANE_D_DTYPES)))
    fwd = _float(ld.get("forward_margin_hours", 6.0), "forward_margin_hours")
    if fwd < 0.0:
        raise ValueError(f"[lane_d] forward_margin_hours must be >= 0; got {fwd!r}")
    lag = _float(ld.get("max_mirror_lag_hours", 6.0), "max_mirror_lag_hours")
    if lag <= 0.0:
        raise ValueError(f"[lane_d] max_mirror_lag_hours must be > 0; got {lag!r}")
    tau = _float(ld.get("tau_public", 0.01), "tau_public")
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"[lane_d] tau_public must be in [0, 1); got {tau!r}")
    suites = _names(ld.get("gate_suites", list(LANE_D_GATE_SUITES)), "gate_suites")
    gate_w = _gate_weights(ld.get("gate_weights"), suites)

    # W2e keys and their cross-key checks (re-run at every overlay adoption:
    # netconfig.validate_overlay rebuilds the merged config through here).
    min_windows = _int(ld.get("min_windows", 1000), "min_windows", lo=1)
    d_margin = _open_unit(ld.get("d_margin", 0.01), "d_margin")
    embargo = _float(ld.get("embargo_hours", 48.0), "embargo_hours")
    if embargo < 0.0:
        raise ValueError(f"[lane_d] embargo_hours must be >= 0; got {embargo!r}")
    max_windows = _int(ld.get("max_windows_per_settlement", 5000),
                       "max_windows_per_settlement", lo=1)
    if max_windows < min_windows:
        raise ValueError(f"[lane_d] max_windows_per_settlement={max_windows} must be >= "
                         f"min_windows={min_windows}")
    fresh = _float(ld.get("min_fresh_window_share", 0.5), "min_fresh_window_share")
    if not 0.0 <= fresh <= 1.0:
        raise ValueError(f"[lane_d] min_fresh_window_share must be in [0, 1]; got {fresh!r}")
    # rev6.18 A3 (F4): the public half's weight, range [0, 1), and the
    # per-window ratio clamp floor, range (0, 1). Not tied to any other key
    # (D24: the backtest cap and its loader inequality are deleted).
    public_weight = _float(ld.get("public_weight", 0.5), "public_weight")
    if not 0.0 <= public_weight < 1.0:
        raise ValueError(f"[lane_d] public_weight must be in [0, 1); got {public_weight!r}")
    ratio_floor = _open_unit(ld.get("window_ratio_floor", 0.5), "window_ratio_floor")
    notes = ld.get("notes", {})
    return LaneDConfig(
        from_block=fb, until_block=ub, share=share, takes_from=takes, remainder=rem,
        contexts=contexts, horizon=horizon, dtypes=dtypes,
        forward_margin_hours=fwd,
        min_windows=min_windows,
        min_clusters=_int(ld.get("min_clusters", 20), "min_clusters", lo=1),
        min_long_series=_int(ld.get("min_long_series", 300), "min_long_series", lo=0),
        max_source_share=_half_open_unit(ld.get("max_source_share", 0.25), "max_source_share"),
        d_margin=d_margin,
        alpha=_open_unit(ld.get("alpha", 0.05), "alpha"),
        bootstrap_B=_int(ld.get("bootstrap_B", 10_000), "bootstrap_B", lo=100),
        embargo_hours=embargo, max_windows_per_settlement=max_windows,
        min_fresh_window_share=fresh, public_weight=public_weight,
        window_ratio_floor=ratio_floor,
        gate_suites=suites, gate_weights=gate_w, tau_public=tau,
        reward_prior_kings=_int(ld.get("reward_prior_kings", 2), "reward_prior_kings", lo=0),
        king_decay=_half_open_unit(ld.get("king_decay", 0.5), "king_decay"),
        max_entries_per_settlement=_int(ld.get("max_entries_per_settlement", 8),
                                        "max_entries_per_settlement", lo=1),
        max_open_per_coldkey=_int(ld.get("max_open_per_coldkey", 2),
                                  "max_open_per_coldkey", lo=1),
        max_judged_per_coldkey_per_settlement=_int(
            ld.get("max_judged_per_coldkey_per_settlement", 2),
            "max_judged_per_coldkey_per_settlement", lo=1),
        max_coldkey_unresolved_settlements=_int(
            ld.get("max_coldkey_unresolved_settlements", 4),
            "max_coldkey_unresolved_settlements", lo=1),
        max_reveal_delay_days=_int(ld.get("max_reveal_delay_days", 14),
                                   "max_reveal_delay_days", lo=1),
        max_bench_holds=_int(ld.get("max_bench_holds", 4), "max_bench_holds", lo=0),
        loser_ttl_eras=_int(ld.get("loser_ttl_eras", 4), "loser_ttl_eras", lo=0),
        max_mirror_lag_hours=lag,
        ready_ttl_blocks=_int(ld.get("ready_ttl_blocks", 7200), "ready_ttl_blocks", lo=1),
        notes=dict(notes) if isinstance(notes, dict) else {},
    )


__all__ = [
    "LANE_D_CONTEXTS",
    "LANE_D_DTYPES",
    "LANE_D_GATE_SUITES",
    "LANE_D_MAX_CONTEXT",
    "LANE_D_MAX_HORIZON",
    "LANE_D_REMAINDERS",
    "LANE_D_TAKES_FROM",
    "RETIRED_LANE_D_F4_KEYS",
    "RETIRED_LANE_D_KEYS",
    "LaneDConfig",
    "parse_lane_d",
]
