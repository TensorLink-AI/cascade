"""Hard bounds on what a signed overlay may set (LANE_D_PLAN rev6.24, W-OV).

DEC-CA-0057 lets the owner change competition tunables by a signed overlay
instead of a fleet release. The loader already refuses values no lane can
run with (``alpha`` outside (0, 1), a negative window count, ...). This
module adds a tighter, overlay-only ENVELOPE: values the loader would accept
but that no owner decision should ever reach by overlay, because they would
let a null entry win (``alpha`` 0.4, a margin of 0), stall every satellite
(``bootstrap_B`` 10**8, a million windows), or silently move money (a lane
share the carve would clamp).

Rules (all enforced in code, none settable by an overlay):

1. :data:`BOUNDS` — per ``(table, key)`` numeric envelope, checked on the
   override VALUE after the loader accepted the merged table
   (:func:`value_refusals`). A key without a bound is still limited by the
   loader; a bound never loosens a loader check.
2. :data:`RELEASE_ONLY_KEYS` — keys inside an otherwise overlay-settable
   table that are data pins, storage locations, model pins, contract digests
   or fork-guard names. Refused by key presence (as the burn destinations
   are), so a node never adopts a different data root by overlay.
3. :func:`share_refusals` — the requested lane shares must fit without the
   carve clamping any of them (``lane_pool_shares`` clamps silently).

The bounds contain every value shipped in ``chain.toml`` and
``chain.testnet.toml`` (tested), so a restating overlay always passes. An
older node does not run these checks and accepts a superset; the owner's
publish tool (``plan_overlay``) runs them, so a published overlay is one
every node accepts. Changing a bound is a release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bound:
    """``lo``/``hi`` (``None`` = open-ended), each inclusive unless the
    matching ``*_open`` flag is set; ``integer`` refuses a non-int."""

    lo: float | None = None
    hi: float | None = None
    lo_open: bool = False
    hi_open: bool = False
    integer: bool = False
    why: str = ""

    def refusal(self, value: object) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"must be a number; got {value!r}"
        if self.integer and not isinstance(value, int):
            return f"must be an integer; got {value!r}"
        v = float(value)
        if not math.isfinite(v):
            return f"must be finite; got {value!r}"
        if self.lo is not None and (v < self.lo or (self.lo_open and v == self.lo)):
            return f"{value!r} is below the overlay envelope {self.span()}"
        if self.hi is not None and (v > self.hi or (self.hi_open and v == self.hi)):
            return f"{value!r} is above the overlay envelope {self.span()}"
        return None

    def span(self) -> str:
        lo = "-inf" if self.lo is None else repr(self.lo)
        hi = "inf" if self.hi is None else repr(self.hi)
        return f"{'(' if self.lo_open or self.lo is None else '['}{lo}, {hi}" \
               f"{')' if self.hi_open or self.hi is None else ']'}"


# Why-strings, one per danger class.
_FREE_WIN = "a looser test lets a null entry win"
_DOS = "satellites re-run this from signed rows; larger stalls the fleet"
_MONEY = "an emission share; larger moves pay away from the other lanes"
_STALL = "a lane that can never reach this never pays"


def _alpha() -> Bound:
    return Bound(0.0, 0.10, lo_open=True, why=_FREE_WIN)


def _margin() -> Bound:
    return Bound(0.002, 0.25, why=_FREE_WIN + " (a bar below 0.2 % is noise)")


def _boot() -> Bound:
    return Bound(200, 50_000, integer=True, why=_DOS)


def _windows(lo: int = 10, hi: int = 20_000) -> Bound:
    return Bound(lo, hi, integer=True, why=_DOS + "; " + _STALL)


def _clusters(lo: int = 2) -> Bound:
    return Bound(lo, 1000, integer=True, why=_FREE_WIN)


def _tau() -> Bound:
    return Bound(0.0, 0.10, why="a looser non-inferiority guard passes a regression")


def _count(lo: int, hi: int, why: str = _DOS) -> Bound:
    return Bound(lo, hi, integer=True, why=why)


def _hours(hi: float) -> Bound:
    return Bound(0.0, hi, lo_open=True, why="leg / wait length")


_DECAY = Bound(0.0, 1.0, lo_open=True, why="court split; 1.0 = flat")
# containment (owner 2026-11-24): a lane start / kill block is a chain block
# number (u32); the loader keeps it an era boundary, netconfig keeps it
# forward-only (never earlier than typed, never before the apply block)
_BLOCK = Bound(0, 2**32 - 1, integer=True,
               why="a chain block (u32); forward-only, era boundary")
_PRIOR_KINGS = _count(0, 8, why="court length")
_FRACTION = Bound(0.0, 1.0, why="a fraction")

# (table, key) -> envelope. Lane A ([scoring], [eval], listed [v2]), lane B,
# lane C, lane D and the listed [marketcast] (lane C's judge) tunables.
BOUNDS: dict[tuple[str, str], Bound] = {
    # ── lane A ────────────────────────────────────────────────────────────
    ("scoring", "bootstrap_B"): _boot(),
    ("scoring", "bootstrap_alpha"): Bound(0.0, 0.20, lo_open=True, why=_FREE_WIN),
    ("scoring", "king_decay"): _DECAY,
    ("scoring", "reward_prior_kings"): _PRIOR_KINGS,
    ("scoring", "min_windows"): _windows(1),
    ("scoring", "min_clusters"): _count(0, 1000, why=_FREE_WIN),
    ("scoring", "win_margin_start"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "win_margin_end"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "win_margin_start_prev"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "win_margin_start_v2"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "win_margin_end_v2"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "margin_increment_floor"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "bench_gate_tolerance"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "bench_gate_min_win_rate"): _FRACTION,
    ("scoring", "bench_gate_min_items"): _count(1, 10_000),
    ("scoring", "gift_gate_tolerance"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("scoring", "cascade_top_k"): _count(1, 16),
    ("eval", "n_windows"): _windows(100),
    ("eval", "shared_ladder_windows"): _windows(500),
    ("eval", "mix_target_windows"): _windows(0),
    ("eval", "num_samples"): _count(8, 1000),
    ("v2", "win_margin_start"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("v2", "win_margin_end"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("v2", "king_check_margin"): Bound(0.0, 1.0, why=_FREE_WIN),
    ("v2", "sentinel_tau"): _tau(),
    # ── lane B ────────────────────────────────────────────────────────────
    ("lane_b", "share"): Bound(0.0, 0.75, why=_MONEY),
    ("lane_b", "alpha"): _alpha(),
    ("lane_b", "bootstrap_B"): _boot(),
    ("lane_b", "gain_margin"): _margin(),
    ("lane_b", "min_target_windows"): _windows(10),
    ("lane_b", "min_target_clusters"): _clusters(),
    ("lane_b", "min_guard_windows"): _windows(10),
    ("lane_b", "affinity_windows"): _windows(16),
    ("lane_b", "tau_domain"): _tau(),
    ("lane_b", "tau_public"): _tau(),
    ("lane_b", "traffic_steal_bound"): Bound(0.0, 0.25, why=_FREE_WIN),
    ("lane_b", "leg_hours"): _hours(48.0),
    ("lane_b", "max_leg_hours"): _hours(48.0),
    ("lane_b", "max_leg_seconds"): _count(60, 2 * 48 * 3600, why="leg wall cap"),
    ("lane_b", "guard_factor"): Bound(1.0, 4.0, why="leg wall cap"),
    ("lane_b", "sentinel_max_defer_eras"): _count(0, 100, why=_STALL),
    ("lane_b", "until_block"): _BLOCK,
    # ── lane C ────────────────────────────────────────────────────────────
    ("lane_c", "share"): Bound(0.0, 0.5, why=_MONEY),
    ("lane_c", "from_block"): _BLOCK,
    ("lane_c", "until_block"): _BLOCK,
    ("lane_c", "alpha"): _alpha(),
    ("lane_c", "bootstrap_B"): _boot(),
    ("lane_c", "gain_margin"): _margin(),
    ("lane_c", "n_windows"): _windows(100),
    ("lane_c", "min_target_windows"): _windows(10),
    ("lane_c", "min_target_clusters"): _clusters(),
    ("lane_c", "tau_public"): _tau(),
    ("lane_c", "leg_hours"): _hours(48.0),
    ("lane_c", "legs_per_target"): _count(1, 16),
    # DEC-CA-0073 duration-scaled full pay of an mkt: win
    ("lane_c", "full_pay_min_days"): Bound(0.0, 365.0, lo_open=True, why=_MONEY),
    ("lane_c", "full_pay_max_days"): Bound(0.0, 365.0, lo_open=True, why=_MONEY),
    ("lane_c", "full_pay_max_gain"): Bound(0.0, 0.5, lo_open=True, why=_MONEY),
    # ── lane C's judge: the listed [marketcast] controls ──────────────────
    ("marketcast", "from_block"): _BLOCK,
    ("marketcast", "alpha"): _alpha(),
    ("marketcast", "bootstrap_B"): _boot(),
    ("marketcast", "gain_margin"): _margin(),
    ("marketcast", "min_windows"): _windows(50),
    ("marketcast", "min_clusters"): _clusters(),
    ("marketcast", "min_time_blocks"): _count(1, 100, why=_FREE_WIN),
    ("marketcast", "calibration_z"): Bound(1.0, 10.0, why=_FREE_WIN),
    ("marketcast", "cell_ratio_cap"): Bound(1.0, 100.0, lo_open=True, why=_FREE_WIN),
    ("marketcast", "near_dup_tol"): Bound(0.0, 0.5, why=_FREE_WIN),
    ("marketcast", "report_timeout_settlements"): _count(1, 100, why=_STALL),
    ("marketcast", "max_infra_deferrals"): _count(0, 100, why=_STALL),
    # lane C eval data (the grid): 66 assets x 6 timeframes x this per suite
    ("marketcast", "windows_per_cell"): _count(1, 512),
    # DEC-CA-0075: the embargo (recorded per leg at dispatch; the loader also
    # keeps embargo_eval_days < embargo_days)
    ("marketcast", "embargo_days"): _count(2, 120, why="the eval draw lies inside it"),
    # DEC-CA-0074 (the loader bounds them too: 1 .. embargo_days - 1, 1 .. 256, <= 3900)
    ("marketcast", "embargo_eval_days"): _count(1, 365),
    ("marketcast", "backtest_windows_per_cell"): _count(1, 256),
    ("marketcast", "verdict_wait_blocks"): _count(0, 3900, why=_STALL),
    # ── lane D ────────────────────────────────────────────────────────────
    ("lane_d", "share"): Bound(0.0, 0.5, why=_MONEY),
    ("lane_d", "until_block"): _BLOCK,
    ("lane_d", "alpha"): _alpha(),
    ("lane_d", "bootstrap_B"): _boot(),
    ("lane_d", "d_margin"): _margin(),
    # rev6.23 decaying crown bar: d_margin_start -> d_margin over the warmup
    ("lane_d", "d_margin_start"): _margin(),
    ("lane_d", "d_margin_warmup_settlements"): _count(0, 1000, why=_STALL),
    ("lane_d", "min_windows"): _windows(100),
    ("lane_d", "min_clusters"): _clusters(),
    ("lane_d", "max_windows_per_settlement"): _windows(1000),
    ("lane_d", "max_entries_per_settlement"): _count(1, 64),
    ("lane_d", "tau_public"): _tau(),
    ("lane_d", "reward_prior_kings"): _PRIOR_KINGS,
    ("lane_d", "embargo_hours"): Bound(0.0, 168.0, why="the reveal delay caps it"),
    ("lane_d", "forward_margin_hours"): Bound(0.0, 48.0, why="forward-only gap"),
    # rev6.18 D24 (F4): backtest_weight_cap / backtest_ratio_floor are retired
    # keys (no parser reads them), so they carry no overlay bound.
    # rev6.18 A3 (F4): the live/public split L = (1 - w) L_live + w L_pub,
    # overlay-tweakable after the release (owner 2026-11-24). The loader keeps
    # w < 1 (a live half always counts).
    ("lane_d", "public_weight"): Bound(0.0, 1.0, why="the live / public split of the verdict"),
    # The per-window ratio clamp [floor, 1 / floor] (0.5 = [0.5, 2]); there is
    # no separate ceiling key. Below 0.1 one window may move the score by more
    # than 10x (a single outlier decides); above 0.9 the clamp flattens every
    # window and nothing can clear.
    ("lane_d", "window_ratio_floor"): Bound(0.1, 0.9, why=_FREE_WIN + " (one window moves "
                                             "the score by at most 1 / floor; a floor near 1 "
                                             "stalls every verdict)"),
    ("lane_d", "max_source_share"): Bound(0.0, 1.0, lo_open=True, why=_FREE_WIN),
    ("lane_d", "max_open_per_coldkey"): _count(1, 64),
    ("lane_d", "max_judged_per_coldkey_per_settlement"): _count(1, 64),
    ("lane_d", "max_coldkey_unresolved_settlements"): _count(1, 1000, why=_STALL),
    ("lane_d", "max_reveal_delay_days"): _count(1, 365, why=_STALL),
    ("lane_d", "max_bench_holds"): _count(0, 64, why=_STALL),
    ("lane_d", "loser_ttl_eras"): _count(0, 1000, why=_STALL),
    ("lane_d", "max_mirror_lag_hours"): Bound(0.0, 168.0, lo_open=True, why=_STALL),
    ("lane_d", "ready_ttl_blocks"): _count(1, 1_000_000, why=_STALL),
}

# Release-only keys inside an overlay-settable table, with the reason the
# refusal prints. (The burn destinations are refused by
# netconfig.check_overrides itself; DEC-CA-0075 made [lane_b]
# contamination_hashset overlay-settable as a sha256-verified content hash.)
RELEASE_ONLY_KEYS: dict[tuple[str, str], str] = {
    ("scoring", "prior_contract_digest"): "a contract digest is release-only",
    ("eval", "shared_ladder_feature"): "a fork-guard feature name is release-only",
    ("eval", "benchmark_project_dir"): "an operator-local path is never an overlay",
    ("eval", "gift_gate_data_dir"): "an operator-local path is never an overlay",
    ("lane_d", "teacher_ref"): "the teacher model pin is release-only",
    ("lane_d", "teacher_api"): "an external endpoint is release-only",
    # Owner 2026-11-27: the per-window horizon rule. Validators that follow the
    # signed report run the pre-band checker; the band rule (and the pool
    # format it builds) changes only with a release.
    ("lane_d", "horizon_bands"): "the per-window horizon rule is release-only",
    ("lane_d", "max_horizon"): "the per-window horizon rule is release-only",
    ("lane_d", "band_slots"): "the per-window horizon rule is release-only",
    ("lane_d", "min_cell_windows"): "the per-window horizon rule is release-only",
}

# Eval DATA keys (owner 2026-11-24: what data a lane is judged on is
# overlay-settable; only rule logic needs a release). A data reference an
# overlay sets must be CONTENT-ADDRESSED — ``<repo>@<revision>`` — so every
# node loads the same bytes. A bucket prefix is content-addressed by the
# signed snapshot index it holds (each snapshot sha256-verified), so it
# needs no revision.
PINNED_REF_KEYS: dict[tuple[str, str], str] = {
    ("eval", "window_pool"): "the eval pool",
    ("lane_d", "real_corpus_ref"): "the real corpus",
}


def pin_refusals(overrides: dict) -> list[str]:
    """``[table] key`` for every :data:`PINNED_REF_KEYS` value an overlay
    sets that is not ``<repo>@<revision>`` (empty = off, allowed)."""
    out = []
    for (table, key), what in PINNED_REF_KEYS.items():
        kv = overrides.get(table)
        if not isinstance(kv, dict) or key not in kv:
            continue
        v = kv[key]
        repo, sep, rev = str(v).partition("@") if isinstance(v, str) else ("", "", "")
        if not isinstance(v, str) or (v and not (sep and repo.strip() and rev.strip())):
            out.append(f"[{table}] {key} = {v!r}: {what} must be content-addressed "
                       "(<repo>@<revision>)")
    return out


def key_refusals(overrides: dict) -> list[str]:
    """``[table] key (reason)`` for every :data:`RELEASE_ONLY_KEYS` key the
    overrides name (presence, whatever the value)."""
    out = []
    for table, kv in overrides.items():
        if not isinstance(kv, dict):
            continue
        for key in kv:
            why = RELEASE_ONLY_KEYS.get((table, key))
            if why:
                out.append(f"[{table}] {key} ({why})")
    return out


# (table, key) -> (per-element envelope, max length, min first element):
# list-valued tunables. The loader checks the sum and the length rule.
SEQ_BOUNDS: dict[tuple[str, str], tuple[Bound, int, float]] = {
    # rev6.23 lane D court split (KING_D first): KING_D keeps at least half,
    # at most 1 + 8 slots (reward_prior_kings <= 8)
    ("lane_d", "court_weights"): (Bound(0.0, 1.0, why=_MONEY), 9, 0.5),
}


def _seq_refusal(spec: tuple[Bound, int, float], value: object) -> str | None:
    elem, max_len, first_min = spec
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        return f"must be a non-empty list; got {value!r}"
    if len(value) > max_len:
        return f"has {len(value)} entries, above the overlay envelope's {max_len}"
    for i, v in enumerate(value):
        why = elem.refusal(v)
        if why:
            return f"[{i}] {why}"
    if float(value[0]) < first_min:
        return f"first entry {value[0]!r} is below the overlay envelope's {first_min}"
    return None


def value_refusals(overrides: dict) -> list[str]:
    """``[table] key: why`` for every override outside its :data:`BOUNDS`
    (or :data:`SEQ_BOUNDS`) envelope."""
    out = []
    for table, kv in overrides.items():
        if not isinstance(kv, dict):
            continue
        for key, value in kv.items():
            seq = SEQ_BOUNDS.get((table, key))
            if seq is not None:
                why = _seq_refusal(seq, value)
                if why:
                    out.append(f"[{table}] {key} {why} ({seq[0].why})")
                continue
            b = BOUNDS.get((table, key))
            if b is None:
                continue
            why = b.refusal(value)
            if why:
                out.append(f"[{table}] {key} {why} ({b.why})")
    return out


def out_of_bounds_keys(overrides: dict) -> list[tuple[str, str, str]]:
    """``(table, key, why)`` for every override the per-key envelope refuses
    (:func:`key_refusals`, :func:`pin_refusals`, :func:`value_refusals`), in
    the overrides' own order. #891: an IN-FORCE overlay (adopted under an
    older release) that fails this release's envelope keeps the release
    value for these keys instead of failing every round
    (:func:`cascade.shared.netconfig.build_config` ``in_force=True``)."""
    out = []
    for table, kv in overrides.items():
        if not isinstance(kv, dict):
            continue
        for key, value in kv.items():
            one = {table: {key: value}}
            why = key_refusals(one) + pin_refusals(one) + value_refusals(one)
            if why:
                out.append((table, key, "; ".join(why)))
    return out


def share_refusals(cfg) -> list[str]:
    """Lane shares that the carve would clamp (``lane_pool_shares`` caps a
    lane at what its source holds, silently): lane C's share must fit its
    ``takes_from`` source and lane D's its, every lane armed or not."""
    out = []
    b = float(cfg.lane_b.share)
    c = float(cfg.lane_c.share)
    d = float(cfg.lane_d.share)
    c_src = str(cfg.lane_c.takes_from)
    avail_c = b if c_src == "lane_b" else 1.0 - b
    if c > avail_c + 1e-12:
        out.append(f"[lane_c] share={c} exceeds its source {c_src} ({avail_c})")
    pool_b = b - c if c_src == "lane_b" else b
    pool_a = 1.0 - b - (0.0 if c_src == "lane_b" else c)
    d_src = str(cfg.lane_d.takes_from)
    avail_d = {"lane_a": pool_a, "lane_b": pool_b, "lane_c": c}.get(d_src, 0.0)
    if d > avail_d + 1e-12:
        out.append(f"[lane_d] share={d} exceeds its source {d_src} ({avail_d:.6g}): the "
                   "carve would clamp it")
    return out


def policy() -> dict:
    """The envelope as ``overlay_policy()`` publishes it."""
    return {
        "bounds": {f"{t}.{k}": {"span": b.span(), "integer": b.integer, "why": b.why}
                   for (t, k), b in sorted(BOUNDS.items())},
        "list_bounds": {f"{t}.{k}": {"each": e.span(), "max_len": n, "first_min": f,
                                     "why": e.why}
                        for (t, k), (e, n, f) in sorted(SEQ_BOUNDS.items())},
        "release_only_keys": {f"{t}.{k}": why for (t, k), why in sorted(RELEASE_ONLY_KEYS.items())},
        "pinned_refs": sorted(f"{t}.{k}" for t, k in PINNED_REF_KEYS),
        "never_overlay": ["the bounds themselves", "signature / trust rules",
                          "[netconfig] owner_hotkey", "burn destinations"],
    }


__all__ = ["BOUNDS", "PINNED_REF_KEYS", "RELEASE_ONLY_KEYS", "SEQ_BOUNDS", "Bound", "key_refusals",
           "out_of_bounds_keys", "pin_refusals", "policy", "share_refusals", "value_refusals"]
