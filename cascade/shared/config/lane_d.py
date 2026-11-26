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

import math
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
    # rev6.23 (owner 2026-11-24): the court pays explicit court_weights; the
    # geometric king_decay is read by no release. (Lane A's [scoring]
    # king_decay is a different, live key.)
    "king_decay",
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


def lane_d_crown_bar(start: float, end: float, warmup: int, tenure: int) -> float:
    """``start - (start - end) * tenure / warmup`` (rev6.23): exactly
    ``start`` at tenure 0 and exactly ``end`` from tenure ``warmup`` on. One
    expression, so every re-derivation gets the same float."""
    t, w = max(0, int(tenure)), int(warmup)
    if t >= w:
        return float(end)
    return float(start) - (float(start) - float(end)) * t / w


@dataclass(frozen=True)
class LaneDConfig:
    # Activation (D10/D45): 0 = off; an era-boundary block arms it. The lane
    # closes after ``until_block`` (0 = no end; the T21 kill switch).
    from_block: int = 0
    until_block: int = 0
    # DEC-CA-0075 rule menu: lane D runs the max-T crown test and pays the
    # court until beaten; the keys exist so the choice is explicit and
    # recorded (other menu entries are refused for lane D in this release)
    test_family: str = "maxt"
    pay_curve: str = "indefinite_until_beaten"
    # Emissions: ``share`` of the total, carved from ``takes_from`` (D1/D48
    # move it to lane_a by overlay); the unpaid remainder burns (D21).
    share: float = 0.0
    takes_from: str = "lane_b"
    remainder: str = "burn"
    # What an entry is judged at (§1.1, D2, D14).
    contexts: tuple[int, ...] = LANE_D_CONTEXTS
    horizon: int = 64
    dtypes: tuple[str, ...] = LANE_D_DTYPES
    # Forward-only judging (§1.3, D5). rev6.14 (a): two clocks per entry, T =
    # wall(max(T_E, T_K)): the eligibility / fresh boundary b = T +
    # forward_margin_hours (default 0) and the judging start j = T +
    # eval_delay_hours (default 6, the owner's "delay eval by 6 hours").
    forward_margin_hours: float = 0.0
    eval_delay_hours: float = 6.0
    min_windows: int = 1000
    min_clusters: int = 20
    min_long_series: int = 300
    max_source_share: float = 0.25
    # rev6.23 (owner 2026-11-24): the crown bar decays with KING_D's tenure,
    # linearly from ``d_margin_start`` to ``d_margin`` (the end, the floor)
    # over ``d_margin_warmup_settlements`` settlements (see :meth:`crown_bar`).
    d_margin: float = 0.01
    d_margin_start: float = 0.03
    d_margin_warmup_settlements: int = 32
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    # W2e (rev6.10 (a)/(b), rev6.11 (a), rev6.12 (b)): the per-series embargo
    # top-up (0 = the rev3 rule), the per-settlement window cap (>= min_windows)
    # and the minimum fresh share of a frozen set (1 = fresh windows only).
    embargo_hours: float = 48.0
    max_windows_per_settlement: int = 5000
    # Owner 2026-11-23 ("fresh data should always be in there, but trade off";
    # "first verdict ~6 h"): 0.2, was 0.5 (W1 forge: ~250 fresh windows at T+6 h).
    # Fresh windows are still drawn first and always included.
    min_fresh_window_share: float = 0.2
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
    # Court (§1.4, D7). rev6.23 (owner 2026-11-24): explicit slot weights,
    # KING_D first, one per slot (``1 + reward_prior_kings``), summing to 1;
    # they replace the retired geometric ``king_decay``.
    reward_prior_kings: int = 2
    court_weights: tuple[float, ...] = (0.80, 0.15, 0.05)
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

    def crown_bar(self, tenure: int) -> float:
        """rev6.23: the bar a combined LCB must reach to crown at a settlement
        where KING_D (a miner or the house) has held the crown ``tenure``
        lane D settlements: ``d_margin_start`` at tenure 0, falling linearly
        to ``d_margin`` at ``d_margin_warmup_settlements`` and flat after."""
        return lane_d_crown_bar(self.d_margin_start, self.d_margin,
                                self.d_margin_warmup_settlements, tenure)

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


def _court_weights(raw: object, reward_prior_kings: int) -> tuple[float, ...]:
    """rev6.23: one finite weight >= 0 per court slot (KING_D, then
    ``reward_prior_kings`` prior kings), summing to 1 within 1e-9."""
    want = 1 + int(reward_prior_kings)
    if not isinstance(raw, (list, tuple)) or len(raw) != want:
        raise ValueError(f"[lane_d] court_weights must be a list of {want} numbers "
                         f"(1 + reward_prior_kings); got {raw!r}")
    out = []
    for i, w in enumerate(raw):
        v = _float(w, f"court_weights[{i}]")
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"[lane_d] court_weights[{i}] must be a finite number >= 0; "
                             f"got {w!r}")
        out.append(v)
    if abs(math.fsum(out) - 1.0) > 1e-9:
        raise ValueError(f"[lane_d] court_weights must sum to 1 (within 1e-9); got "
                         f"{math.fsum(out)!r}")
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
    fwd = _float(ld.get("forward_margin_hours", 0.0), "forward_margin_hours")
    if fwd < 0.0:
        raise ValueError(f"[lane_d] forward_margin_hours must be >= 0; got {fwd!r}")
    delay = _float(ld.get("eval_delay_hours", 6.0), "eval_delay_hours")
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError(f"[lane_d] eval_delay_hours must be >= 0; got {delay!r}")
    if fwd > delay:
        # rev6.14 (a): b <= j, else judging could start before a fresh point exists.
        raise ValueError(f"[lane_d] forward_margin_hours ({fwd}) must be <= eval_delay_hours "
                         f"({delay})")
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
    # rev6.23: the tenure-decaying crown bar; ``d_margin`` is its end.
    d_start = _open_unit(ld.get("d_margin_start", 0.03), "d_margin_start")
    if d_margin > d_start:
        raise ValueError(f"[lane_d] d_margin={d_margin} (the bar's end) must be <= "
                         f"d_margin_start={d_start} (rev6.23)")
    d_warmup = _int(ld.get("d_margin_warmup_settlements", 32),
                    "d_margin_warmup_settlements", lo=1)
    rpk = _int(ld.get("reward_prior_kings", 2), "reward_prior_kings", lo=0)
    court_w = _court_weights(ld.get("court_weights", [0.80, 0.15, 0.05]), rpk)
    embargo = _float(ld.get("embargo_hours", 48.0), "embargo_hours")
    if embargo < 0.0:
        raise ValueError(f"[lane_d] embargo_hours must be >= 0; got {embargo!r}")
    max_windows = _int(ld.get("max_windows_per_settlement", 5000),
                       "max_windows_per_settlement", lo=1)
    if max_windows < min_windows:
        raise ValueError(f"[lane_d] max_windows_per_settlement={max_windows} must be >= "
                         f"min_windows={min_windows}")
    fresh = _float(ld.get("min_fresh_window_share", 0.2), "min_fresh_window_share")
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
    from ..rules_menu import parse_lane_menu

    family, curve = parse_lane_menu("lane_d", ld)
    return LaneDConfig(
        test_family=family, pay_curve=curve,
        from_block=fb, until_block=ub, share=share, takes_from=takes, remainder=rem,
        contexts=contexts, horizon=horizon, dtypes=dtypes,
        forward_margin_hours=fwd, eval_delay_hours=delay,
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
        d_margin_start=d_start, d_margin_warmup_settlements=d_warmup,
        reward_prior_kings=rpk, court_weights=court_w,
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


# ── rev6.18 B5: the qualified lane D scoring environments ────────────────────
#
# The list is a committed file (``deploy/laned/scoring_envs.toml``), not a
# chain.toml / overlay key: adding an environment is a release (PR + owner
# merge) and its ``from_block`` activates it, so every validator holds the
# entry before any report names it. Stdlib only: the audit imports these.

SCORING_ENVS_FILE = "deploy/laned/scoring_envs.toml"
SCORING_ENV_STATUSES = ("qualified", "candidate")
SCORING_ENV_KEYS = frozenset({"name", "status", "from_block", "until_block", "lock",
                              "lock_sha256", "torch_build", "gpu_models", "driver_min",
                              "driver_max", "evidence"})
# The report's ``host`` record (B7): exactly these keys.
HOST_KEYS = ("env", "gpu_model", "driver", "torch_build", "fingerprint", "identity")
_HEX64 = frozenset("0123456789abcdef")


class ScoringEnvError(ValueError):
    """The scoring environment list is unreadable or malformed."""


@dataclass(frozen=True)
class ScoringEnv:
    """One ``[[env]]`` of the scoring environment list (B5)."""

    name: str
    status: str
    from_block: int
    until_block: int
    lock: str
    lock_sha256: str
    torch_build: str
    gpu_models: tuple[str, ...]
    driver_min: str
    driver_max: str
    evidence: str

    def qualified_at(self, boundary: int) -> bool:
        """May this environment score the settlement at ``boundary``?"""
        b = int(boundary)
        return (self.status == "qualified" and b > self.from_block
                and (self.until_block <= 0 or b <= self.until_block))


def driver_version(raw: object) -> tuple[int, ...] | None:
    """``"580.65.06"`` -> ``(580, 65, 6)``; ``None`` for anything else."""
    if not isinstance(raw, str) or not raw:
        return None
    parts = raw.strip().split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _is_hex64(v: object) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX64


def _env_name_ok(v: object) -> bool:
    ok = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    return isinstance(v, str) and bool(v) and v[0].isalnum() and set(v) <= ok and v == v.lower()


def parse_scoring_envs(raw: object) -> tuple[ScoringEnv, ...]:
    """The ``[[env]]`` entries of a parsed scoring environment list (B5).
    Raises :class:`ScoringEnvError` on any defect (fail closed)."""
    def bad(why: str):
        raise ScoringEnvError(f"scoring envs: {why}")

    if not isinstance(raw, dict) or set(raw) != {"env"} or not isinstance(raw["env"], list) \
            or not raw["env"]:
        bad("the file must hold exactly a non-empty [[env]] list")
    out, names = [], set()
    for i, e in enumerate(raw["env"]):
        if not isinstance(e, dict) or set(e) != SCORING_ENV_KEYS:
            got = sorted(e) if isinstance(e, dict) else type(e).__name__
            bad(f"env[{i}] keys {got} != {sorted(SCORING_ENV_KEYS)}")
        name = e["name"]
        if not _env_name_ok(name) or name in names:
            bad(f"env[{i}] name {name!r} is malformed or duplicated")
        names.add(name)
        if e["status"] not in SCORING_ENV_STATUSES:
            bad(f"{name}: status {e['status']!r} not in {SCORING_ENV_STATUSES}")
        for k in ("from_block", "until_block"):
            if isinstance(e[k], bool) or not isinstance(e[k], int) or e[k] < 0:
                bad(f"{name}: {k} must be an integer >= 0")
        if e["until_block"] and e["until_block"] <= e["from_block"]:
            bad(f"{name}: until_block must be 0 or > from_block")
        for k in ("lock", "lock_sha256", "torch_build", "driver_min", "driver_max", "evidence"):
            if not isinstance(e[k], str):
                bad(f"{name}: {k} must be a string")
        if not e["torch_build"] or "/cuda-" not in e["torch_build"]:
            bad(f"{name}: torch_build must be '<torch version>/cuda-<cuda version>'")
        models = e["gpu_models"]
        if (not isinstance(models, list) or not models
                or not all(isinstance(m, str) and m.strip() == m and m for m in models)
                or len(set(models)) != len(models)):
            bad(f"{name}: gpu_models must be a non-empty list of unique exact names")
        lo = driver_version(e["driver_min"])
        hi = driver_version(e["driver_max"]) if e["driver_max"] else None
        if lo is None or (e["driver_max"] and (hi is None or hi < lo)):
            bad(f"{name}: driver_min / driver_max must be dotted integers, min <= max")
        if e["status"] == "qualified" and (not e["lock"] or not _is_hex64(e["lock_sha256"])
                                           or not e["evidence"]):
            bad(f"{name}: a qualified env needs lock, lock_sha256 (64 hex) and evidence")
        out.append(ScoringEnv(name=name, status=e["status"], from_block=int(e["from_block"]),
                              until_block=int(e["until_block"]), lock=e["lock"],
                              lock_sha256=e["lock_sha256"], torch_build=e["torch_build"],
                              gpu_models=tuple(models), driver_min=e["driver_min"],
                              driver_max=e["driver_max"], evidence=e["evidence"]))
    return tuple(out)


def scoring_envs_path(root=None):
    """The committed list's path under ``root`` (default: this checkout)."""
    from pathlib import Path

    base = Path(__file__).resolve().parents[3] if root is None else Path(root)
    return base / SCORING_ENVS_FILE


def load_scoring_envs(path=None) -> tuple[ScoringEnv, ...]:
    """The scoring environment list at ``path`` (default: the committed
    file of this checkout). Raises :class:`ScoringEnvError`."""
    import tomllib
    from pathlib import Path

    p = scoring_envs_path() if path is None else Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ScoringEnvError(f"scoring envs {p}: {e}") from e
    return parse_scoring_envs(raw)


def host_identity(env: str, gpu_model: str, driver: str, torch_build: str,
                  fingerprint: str) -> str:
    """B6: sha256 hex of the scoring host's identity (environment name, GPU
    model, driver, torch build and the host fingerprint), each field
    length-prefixed so no two tuples collide."""
    import hashlib

    h = hashlib.sha256(b"lane-d-host/1")
    for part in (env, gpu_model, driver, torch_build, fingerprint):
        b = str(part).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def host_problem(host: object, envs, boundary: int) -> str | None:
    """Why the report ``host`` record does not name an environment qualified
    at ``boundary`` (B5/B7), or ``None`` when it does. Pure: the validator
    checks it before scoring, the audit re-checks the signed record."""
    why = host_record_problem(host)
    if why:
        return why
    by_name = {e.name: e for e in envs}
    env = by_name.get(host["env"])
    if env is None:
        return f"scoring env {host['env']!r} is not in the scoring environment list"
    if not env.qualified_at(boundary):
        return (f"scoring env {env.name!r} ({env.status}, from_block {env.from_block}, "
                f"until_block {env.until_block}) is not qualified at boundary {int(boundary)}")
    if host["torch_build"] != env.torch_build:
        return f"torch build {host['torch_build']!r} != {env.name}'s {env.torch_build!r}"
    if host["gpu_model"] not in env.gpu_models:
        return f"GPU {host['gpu_model']!r} is not allowed in {env.name}"
    drv = driver_version(host["driver"])
    if drv is None or drv < driver_version(env.driver_min) or (
            env.driver_max and drv > driver_version(env.driver_max)):
        return (f"driver {host['driver']!r} outside {env.name}'s range "
                f"[{env.driver_min}, {env.driver_max or '-'}]")
    return None


def host_record_problem(host: object) -> str | None:
    """Why ``host`` is not a well-formed report ``host`` record (exactly
    :data:`HOST_KEYS`, non-empty strings, a 64-hex fingerprint and an
    ``identity`` that hashes the other fields), or ``None``."""
    if not isinstance(host, dict) or set(host) != set(HOST_KEYS) \
            or not all(isinstance(host[k], str) and host[k] for k in HOST_KEYS):
        return f"host record keys/types differ from {list(HOST_KEYS)}"
    if not _is_hex64(host["fingerprint"]):
        return "host fingerprint is not 64 lower-case hex"
    want = host_identity(host["env"], host["gpu_model"], host["driver"], host["torch_build"],
                         host["fingerprint"])
    if host["identity"] != want:
        return "host identity does not hash the host's fields"
    return None


__all__ = [
    "HOST_KEYS",
    "SCORING_ENVS_FILE",
    "SCORING_ENV_STATUSES",
    "ScoringEnv",
    "ScoringEnvError",
    "driver_version",
    "host_identity",
    "host_problem",
    "host_record_problem",
    "load_scoring_envs",
    "parse_scoring_envs",
    "scoring_envs_path",
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
    "lane_d_crown_bar",
    "parse_lane_d",
]
