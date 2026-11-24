"""Shared spec and fixture builders for the W7a report-schema tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W7a row, §1.2 (A8 caps,
coldkey resolution), §1.3 (both clocks, mirror lag, the confirmation duel and
D20), §1.3b (A10 low-frequency check), §1.4 (D21 house slot burns), §1.5
(the report's rev3 fields, the typed reason enum, B8, B10; rev6.11 (b): the
confirmation duel is dropped, so no confirm stage, statuses or block), §3.2 invariant 11,
§14.3 (forward fields), B7 (`scoring_method`) and B12 (first-seen Merkle
roots). The window refusal reasons are W5a's `REFUSAL_REASONS`
(tests/accept/w5a/README.md, branch laned/w5a-tests). README.md states the
module contract.

Nothing here imports a module in PATHS at import time, so on main every test
fails inside its own body: at its first import of a new name from
`cascade.validator.lane_d_report` (an `ImportError` for a path in PATHS) or
at an `assert`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCHEMA_FILE = REPO / "cascade" / "validator" / "schemas" / "lane_d_report_1.json"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT
BOUNDARY_2 = ARM_BLOCK + 2 * SETTLEMENT
READY_BLOCK = ARM_BLOCK + 40
REVEAL_BLOCK = ARM_BLOCK + 60

HORIZON = 64
CONTEXT = 4096
HISTORY = 256
N_SOURCES = 6
PER_SOURCE = 8
HOUR = 3600
T0 = 1_790_000_000                     # the first entry's reveal time (unix s)
HOUSE_T = T0 - 2 * 86_400
SNAPSHOT_START = {1: T0 + 12 * HOUR, 2: T0 + 12 * HOUR + 4 * 86_400}
FIRST_SEEN_LAG = 1800

CHALLENGER_UID = 6
LOSER_UID = 7
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)

ARMED = {
    "from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a", "remainder": "burn",
    "contexts": [4096, 8192, 16384], "horizon": HORIZON, "forward_margin_hours": 6.0,
    "min_windows": 24, "min_clusters": 4, "bootstrap_B": 200,
}

# ── the pinned enums (README "Enums") ───────────────────────────────────────

# rev6.11 (b): no `confirm_pending` status (the confirmation duel is dropped).
ENTRY_STATUSES = ("pending", "hold", "crowned", "lost", "closed")
SKIP_REASONS = ("pending_floor", "coldkey_cap_settlement", "coldkey_unresolved",
                "coldkey_changed", "king_unscored", "bench_hold", "deregistered",
                "ready_expired", "reveal_delay_exceeded", "duplicate_digest",
                "hotkey_has_open_entry", "context_not_armed", "checkpoint_fault",
                "mirror_lag_exceeded", "lane_closed", "context_below_armed",
                "before_eval_delay", "boundary_ts_missing")
# `context_below_armed` (plan rev6.15, owner 2026-11-23): with the settlement's
# armed record, a chosen entry whose declared context is below every armed
# variant is not judged and stays `pending` (never run above its declaration).
# `lane_closed` (W5b, T18): every open entry closes at `[lane_d] until_block`,
# listed in both entries[].reason and skipped[].reason.
# `checkpoint_fault` and `bench_hold` (W3b2 outcomes) are skip reasons: a
# faulty candidate is refused, a held one is not judged this settlement.
# `gate_regression` (RV4) is a judged loss on the §1.3b gate, like
# `lowfreq_regression`: an entry reason, never a skip reason.
# rev6.11 (b): no `confirm_pending` / `lost_confirmation`; a loss is `lost_duel`.
ENTRY_REASONS = SKIP_REASONS + ("lost_duel", "lowfreq_regression", "gate_regression",
                                "crowned")
BURN_REASONS = ("house_king", "deregistered")
# W5a's REFUSAL_REASONS (tests/accept/w5a/README.md "Reasons"), in precedence
# order: `unlisted_feed` (W5a) comes right after `missing_clock`.
WINDOW_REFUSAL_REASONS = ("missing_clock", "unlisted_feed", "forecast_product", "over_lag",
                          "before_margin", "publish_ahead", "late_mirror", "absent_from_s0",
                          "backfill")
# rev6.11 (b): the confirm stage, its rule and the verdict outcomes `confirm_pending`
# / `confirmed` are gone (README "Correction (rev6.11)").
REMOVED_STAGE = "confirm"
REMOVED_OUTCOMES = ("confirm_pending", "confirmed")
LOWFREQ_RESULTS = ("pass", "fail", "no_windows", "not_candidate")
# The §1.3b paired gate (W3b2 `lane_d_bench.OUTCOMES` and `REFS`), RV4.
GATE_OUTCOMES = ("pass", "gate_regression", "checkpoint_fault", "bench_hold")
GATE_REFS = ("king_d", "house")
GATE_METRICS = ("crps", "mase")
SCHEMA_ERROR_CODES = ("not_object", "unknown_schema", "missing_field", "unknown_field",
                      "bad_type", "bad_value", "rows_sha256_mismatch")

# $defs name in the schema file -> the module tuple name and the pinned values.
ENUM_DEFS = {
    "entry_status": ("ENTRY_STATUSES", ENTRY_STATUSES),
    "entry_reason": ("ENTRY_REASONS", ENTRY_REASONS),
    "skip_reason": ("SKIP_REASONS", SKIP_REASONS),
    "burn_reason": ("BURN_REASONS", BURN_REASONS),
    "window_refusal_reason": ("WINDOW_REFUSAL_REASONS", WINDOW_REFUSAL_REASONS),
    "lowfreq_result": ("LOWFREQ_RESULTS", LOWFREQ_RESULTS),
    "gate_outcome": ("GATE_OUTCOMES", GATE_OUTCOMES),
    "gate_ref": ("GATE_REFS", GATE_REFS),
}

TOP_LEVEL_REQUIRED = ("schema", "boundary", "signer", "signature", "scoring_method", "params",
                      "seed", "pool", "house", "king_before", "king_after", "crowned",
                      "prior_kings", "entries", "judged", "skipped", "cohorts", "windows",
                      "verdicts", "rows", "rows_sha256", "pay",
                      # rev6.23: {tenure, margin}, the crown bar applied
                      "crown_bar")
# `gate` (RV4): the §1.3b gate record, present iff a candidate reached the gate.
# rev6.6: `armed_contexts` + `armed_evidence`, present together iff the
# settlement was handed the W5d armed record (README "Armed contexts").
TOP_LEVEL_OPTIONAL = ("extensions", "gate", "armed_contexts", "armed_evidence", "draw",
                      "boundary_ts")   # rev6.14 (a)
PARAMS_FIELDS = ("share", "takes_from", "remainder", "horizon", "horizons", "contexts",
                 "forward_margin_hours", "max_mirror_lag_hours", "min_windows", "min_clusters",
                 "d_margin", "alpha", "bootstrap_B", "reward_prior_kings",
                 # rev6.23 (owner 2026-11-24): the tenure-decaying crown bar
                 # (d_margin is its end) and the explicit court weights, which
                 # replace king_decay.
                 "d_margin_start", "d_margin_warmup_settlements", "court_weights",
                 "max_entries_per_settlement",
                 "max_judged_per_coldkey_per_settlement", "scoring_levels", "gate_suites",
                 "tau_public", "lowfreq_freqs",
                 # RV4: the gate draw's weights (W3b1), W5b's source cap (B14),
                 # the context-arming floor (§1.7).
                 "gate_weights", "max_source_share", "min_long_series",
                 # rev6.10 / rev6.11 (a) / rev6.12 (b): the W2e keys a satellite needs
                 # to re-derive eligibility, the draw budget and the fresh share.
                 "embargo_hours", "max_windows_per_settlement", "min_fresh_window_share")
# The W2e defaults (plan: 48 h, 5000 windows, 0.5); read from cfg with these
# fall-backs, so a test never touches an attribute W2e has not added yet.
W2E_DEFAULTS = {"embargo_hours": 48.0, "max_windows_per_settlement": 5000,
                "min_fresh_window_share": 0.5}
GATE_FIELDS = ("suite", "block", "block_hash", "sentinel", "eligible", "results")
GATE_RESULT_FIELDS = ("digest", "outcome", "lcb", "blocked_by", "session")
# `snapshots` (RV4, W5b #581): every snapshot a judged window's version came
# from; `snapshot` / `sha256` / `horizon` stay the newest snapshot's.
POOL_FIELDS = ("snapshot", "sha256", "horizon", "first_seen_roots", "snapshots")
POOL_SNAPSHOT_FIELDS = ("snapshot", "sha256", "first_seen_roots")
ENTRY_FIELDS = ("digest", "hotkey", "coldkey", "context", "ready_block", "reveal_block",
                "reveal_ts", "status", "reason")
SKIP_FIELDS = ("digest", "reason")
# rev6.11: a cohort's `stage` is fixed to `first` or dropped (not pinned as required).
COHORT_FIELDS = ("index", "king_digest", "king_reveal_ts", "members", "t_max",
                 "window_ids", "set_sha256", "first_seen_roots", "refused",
                 # rev6.12 (b): the frozen set's fresh / embargo counts.
                 "fresh_windows", "embargo_windows")
# `snapshot_sha256` (RV4, W5b #581): the snapshot whose version was scored.
WINDOW_FIELDS = ("window_id", "source", "feed", "min_target_ts", "min_first_seen_ts",
                 "mirror_lag_hours", "snapshot_sha256",
                 # rev6.10 (a) / rev6.12: the effective embargo and the fresh / embargo
                 # class; rev6.10 (b) / rev6.9: the draw cell (names as W5j / W5i pin).
                 "freq", "embargo_hours", "min_release_ts", "rule", "embargo_window",
                 "timeframe", "bin", "kind", "freq_class", "h", "origin")
WINDOW_RULES = ("embargo", "embargo0")
WINDOW_KINDS = ("forward", "backtest")
WINDOW_BINS = ("forward", "short", "medium", "long")
FREQ_CLASSES = ("subdaily", "daily", "weekly", "monthly", "quarterly", "yearly")
# rev6.12 (b): per entry, the fresh and the embargo sub-score.
SUBSCORE_FIELDS = ("n", "geomean")
# The optional top-level W5j draw record (present iff a cohort was drawn).
DRAW_FIELDS = ("boundary_block_hash", "seed", "max_windows_per_settlement", "cohorts")
DRAW_COHORT_FIELDS = ("index", "quota", "capacity", "candidates_sha256")
# rev6.11: no `confirmation` block; `stage` fixed to `first` or dropped.
VERDICT_FIELDS = ("digest", "king_digest", "cohort", "lcb", "gain", "cleared",
                  "outcome", "lowfreq", "fresh", "embargo")
LOWFREQ_FIELDS = ("result", "freqs", "n_windows", "tau_public")
ROW_FIELDS = ("window_id", "digest", "cohort", "crps", "mase")
COURT_FIELDS = ("digest", "hotkey", "uid", "burn_reason")

# Fixture feeds (pool_d/1 "feeds", W5a format). W5a fails closed (RV3): a feed
# missing from `feeds` is refused `unlisted_feed`, so the rich pools list every
# feed their series name. src0's series report through feed-a (lag measured),
# src1 and src3 are listed with an unmeasured lag (null), the rest measured.
FEEDS = {"feed-a": {"mirror_lag_hours": 1.5, "forecast_product": False, "embargo_hours": 0.0},
         "src1": {"mirror_lag_hours": None, "forecast_product": False, "embargo_hours": 0.0},
         "src2": {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": 0.0},
         "src3": {"mirror_lag_hours": None, "forecast_product": False, "embargo_hours": 0.0},
         "src4": {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": 0.0},
         "src5": {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": 0.0}}
# W5a fails closed on a missing `frontier` too (`absent_from_s0`), so every
# rich series records its S0 as one `[as_of_ts, end_ts]` frontier record, as
# tests/accept/w1_5/helpers.py does: S0 was taken 3 h before the snapshot's
# first target (after every bound the fixture uses, T0 + 6 h) and held data
# up to 1 h before it was taken, so no target is a backfill.
FRONTIER_AS_OF = {snap: start - 3 * HOUR for snap, start in SNAPSHOT_START.items()}
FRONTIER_END_LAG = HOUR


def lane_d_cfg(**over):
    """The real W2 parse of ``ARMED`` with ``over`` applied."""
    from cascade.shared.config.lane_d import parse_lane_d

    return parse_lane_d({**ARMED, **over}, era_blocks_at=lambda _block: ERA)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fs_root(day: str) -> str:
    """A first-seen log's daily Merkle root (fixture: a sha256 of the day)."""
    return text_digest(f"w7a first-seen log {day}")


ROOTS = {1: sorted([fs_root("2026-09-21"), fs_root("2026-09-22")]),
         2: sorted([fs_root("2026-09-25"), fs_root("2026-09-26")])}


def window_id(series_id: str, target_ts) -> str:
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def canonical_rows_sha256(rows) -> str:
    """§1.5 rows digest, re-derived here: sha256 of the canonical JSON."""
    body = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def quantiles_around(median: np.ndarray, spread: float) -> np.ndarray:
    return np.stack([median + spread * z for z in _Z], axis=-1)


def _series(snapshot: int, seed: int, *, rich: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY, dtype=np.float64)
    out = []
    for j in range(N_SOURCES):
        for k in range(PER_SOURCE):
            sid = f"src{j}-s{k}"
            hist = 10.0 + 5.0 * j + k + 2.0 * np.sin(2 * np.pi * t / 24 + k) \
                + rng.normal(0, 0.5, HISTORY)
            start = SNAPSHOT_START[snapshot] + (j * PER_SOURCE + k) * HOUR
            ts = [int(start + i * HOUR) for i in range(HORIZON)]
            rec = {"series_id": sid, "source": f"src{j}", "history": hist,
                   "target_ts": ts, "first_seen_ts": [x + FIRST_SEEN_LAG for x in ts]}
            if rich and j == 0:
                rec["feed"] = "feed-a"
            out.append(rec)
    if rich:
        # One §1.3 trap: both clocks before the bound (W5a `before_margin`).
        old = [int(T0 - 70 * HOUR + i * HOUR) for i in range(HORIZON)]
        out.append({"series_id": "src2-trap", "source": "src2",
                    "history": 12.0 + rng.normal(0, 0.5, HISTORY), "target_ts": old,
                    "first_seen_ts": [x + FIRST_SEEN_LAG for x in old]})
        as_of = FRONTIER_AS_OF[snapshot]
        for rec in out:
            rec["frontier"] = [[as_of, as_of - FRONTIER_END_LAG]]
    return out


def _write_pool(directory: Path, snapshot: int, series: list[dict], targets: dict, *,
                rich: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "pool_d/1", "snapshot": f"w7a-{snapshot}", "horizon": HORIZON,
            "series": []}
    for s in series:
        rec = {"series_id": s["series_id"], "source": s["source"], "freq": "H",
               "seasonal_period": 24, "first_target_ts": s["target_ts"][0],
               "end_ts": s["target_ts"][-1], "target_ts": s["target_ts"],
               "first_seen_ts": s["first_seen_ts"]}
        for key in ("feed", "frontier"):
            if key in s:
                rec[key] = s[key]
        meta["series"].append(rec)
    if rich:
        meta["feeds"] = copy.deepcopy(FEEDS)
        meta["first_seen_roots"] = list(ROOTS[snapshot])
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    np.savez(directory / "hist.npz", **{s["series_id"]: s["history"] for s in series})
    np.savez(directory / "tgt.npz", **{s["series_id"]: np.asarray(targets[s["series_id"]])
                                       for s in series})
    return directory


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w7a-{name}")


def build_chain(tmp: Path, *, rich: bool = True, n_entries: int = 1) -> dict:
    """Two pool_d snapshots and the eval host's quantile forecasts (synthetic
    kind, no torch). Entry 0 is centred on the targets; the house is 3 sd off.
    Extra entries forecast like entry 0 under their own digests. `loser`
    (rev6.11) is 6 sd off: it loses to the house at its first verdict."""
    digests = [text_digest(f"w7a entry {i}") for i in range(n_entries)]
    house = text_digest("w7a house")
    loser = text_digest("w7a loser")
    rng = np.random.default_rng(7)
    pools, forecasts, series_by_snap = {}, {}, {}
    for snap in (1, 2):
        series = _series(snap, 100 + snap, rich=rich)
        targets, fc = {}, {d: {} for d in [*digests, house, loser]}
        for s in series:
            sd = float(np.std(s["history"]))
            target = np.resize(s["history"][-24:], HORIZON) + rng.normal(0, 0.3, HORIZON)
            targets[s["series_id"]] = target
            wid = window_id(s["series_id"], s["target_ts"])
            good = quantiles_around(target + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
            for d in digests:
                fc[d][wid] = good
            fc[house][wid] = quantiles_around(target + 3.0 * sd, 0.3 * sd)
            fc[loser][wid] = quantiles_around(target + 6.0 * sd, 0.3 * sd)
        pools[snap] = _write_pool(tmp / f"pool_d_{snap}", snap, series, targets, rich=rich)
        forecasts[snap] = fc
        series_by_snap[snap] = series
    return {"pools": pools, "forecasts": forecasts, "series": series_by_snap,
            "digests": digests, "loser": loser,
            "house": {"digest": house, "reveal_ts": HOUSE_T}}


def entry(digest: str, name: str, *, coldkey: str | None, reveal_offset: int = 0,
          ready_block: int = READY_BLOCK) -> dict:
    """A revealed entry as settle_lane_d takes it (W1.5 README)."""
    return {"hotkey": fixture_key(name).ss58_address, "digest": digest, "context": CONTEXT,
            "coldkey_at_ready": coldkey, "ready_block": int(ready_block),
            "reveal_block": REVEAL_BLOCK, "reveal_ts": T0 + int(reveal_offset)}


# ── gate inputs (W5e fail-closed, RV5) ──────────────────────────────────────
# Once W5e wires the §1.3b gate into `settle_lane_d`, an ARMED settlement that
# would crown without a bench session and a boundary block hash does not crown
# (`bench_hold`). `settle` therefore passes a healthy bench and the boundary's
# block hash WHEN `settle_lane_d` accepts them; on a main without the
# keywords it makes the legacy call, unchanged. The bench shape follows
# `tests/accept/w5e/helpers.py` (`passing_bench`, `settle_crowning`).

GATE_KEYWORDS = ("boundary_block_hash", "sentinel", "bench")


class HealthyBench:
    """A stand-in for the bench pod (an external service). Every digest but
    the current house gets the same rows (a tie, never a regression) and the
    house is 10% worse on every item, so every crowning candidate passes the
    paired gate on any drawn suite."""

    def __init__(self, house_digest: str):
        self.house_digest = house_digest
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites) -> dict:
        digests, suites = tuple(digests), tuple(suites)
        self.calls.append((digests, suites))
        return {d: {s: {"status": "ok",
                        "rows": _sweep_rows(s, "healthy",
                                            1.0 if d == self.house_digest else 0.9)}
                    for s in suites} for d in digests}


def gate_inputs(boundary: int, house_digest: str) -> dict:
    """The W5e gate keywords `settle_lane_d` accepts (none on today's main):
    the boundary's block hash, no §16 sentinel (W-S1 unarmed) and a healthy
    bench session."""
    import inspect

    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    if "bench" not in params and "boundary_block_hash" not in params:
        return {}
    kw: dict = {}
    if "boundary_block_hash" in params:
        kw["boundary_block_hash"] = gate_block_hash(boundary)
    if "sentinel" in params:
        kw["sentinel"] = None
    if "bench" in params:
        from cascade.validator.lane_d_bench import SweepSession

        kw["bench"] = SweepSession(f"w7a-sess-{int(boundary)}", HealthyBench(house_digest))
    return kw


def settle(chain: dict, cfg, state_path: Path, signer, *, boundary: int, snap: int,
           entries: list[dict], uid_of: dict):
    """`settle_lane_d`, with the W5e gate inputs when it accepts them (see
    `gate_inputs`): `two_settlements` crowns at BOUNDARY_2."""
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(**_rev614_ts(settle_lane_d, boundary), cfg=cfg, boundary=boundary, pool_dir=chain["pools"][snap],
                         entries=[dict(e) for e in entries], house=dict(chain["house"]),
                         forecasts=chain["forecasts"][snap], uid_of=dict(uid_of),
                         state_path=state_path, signer=signer,
                         **gate_inputs(boundary, chain["house"]["digest"]))


_CACHE: dict[str, dict] = {}


def coldkey(name: str = "coldkey-1") -> str:
    return fixture_key(name).ss58_address


def two_settlements(tmp: Path, *, rich: bool = True) -> dict:
    """The single-challenger chain settled at BOUNDARY_1 (snapshot 1) and
    BOUNDARY_2 (snapshot 2), unchanged for the WPs that import it (W9). Under
    rev6.11 (b) the challenger is crowned at BOUNDARY_1 already; W7a's own
    tests use `loser_then_crown`. Built once per scratch dir."""
    key = f"{tmp}:{rich}"
    if key not in _CACHE:
        chain = build_chain(tmp / "chain", rich=rich)
        e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
        uid_of = {e["hotkey"]: CHALLENGER_UID}
        cfg = lane_d_cfg()
        signer = fixture_key("signer")
        state = tmp / "state" / "lane_d_state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        s1 = settle(chain, cfg, state, signer, boundary=BOUNDARY_1, snap=1, entries=[e],
                    uid_of=uid_of)
        s2 = settle(chain, cfg, state, signer, boundary=BOUNDARY_2, snap=2, entries=[e],
                    uid_of=uid_of)
        _CACHE[key] = {"chain": chain, "entry": e, "uid_of": uid_of, "cfg": cfg,
                       "signer": signer, "state": state, "s1": s1, "s2": s2}
    return _CACHE[key]


def loser_then_crown(tmp: Path, *, rich: bool = True) -> dict:
    """The chain settled at BOUNDARY_1 (snapshot 1) and BOUNDARY_2 (snapshot 2).
    rev6.11 (b): a crown is decided at the FIRST verdict, so BOUNDARY_1
    judges the loser L alone (it loses to the house, which keeps the slot:
    `house_king`) and BOUNDARY_2 judges the challenger, revealed after
    BOUNDARY_1, whose first verdict clears and crowns through the gate.
    Built once per scratch dir; called from test bodies only."""
    key = f"{tmp}:{rich}:loser"
    if key not in _CACHE:
        chain = build_chain(tmp / "chain", rich=rich)
        e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
        loser = entry(chain["loser"], "loser", coldkey=coldkey("coldkey-loser"))
        uid_of = {e["hotkey"]: CHALLENGER_UID, loser["hotkey"]: LOSER_UID}
        cfg = lane_d_cfg()
        signer = fixture_key("signer")
        state = tmp / "state" / "lane_d_state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        s1 = settle(chain, cfg, state, signer, boundary=BOUNDARY_1, snap=1, entries=[loser],
                    uid_of=uid_of)
        s2 = settle(chain, cfg, state, signer, boundary=BOUNDARY_2, snap=2, entries=[e],
                    uid_of=uid_of)
        _CACHE[key] = {"chain": chain, "entry": e, "loser": loser, "uid_of": uid_of,
                       "cfg": cfg,
                       "signer": signer, "state": state, "s1": s1, "s2": s2}
    return _CACHE[key]


def reseal(report: dict) -> dict:
    """``report`` with ``rows_sha256`` recomputed over its (mutated) rows, so
    the only remaining defect is the one a test introduced."""
    out = copy.deepcopy(report)
    out["rows_sha256"] = canonical_rows_sha256(out["rows"])
    return out


def assert_valid(report) -> None:
    """`validate_report(report)`, a refusal surfacing as an `AssertionError`
    that names the code and field."""
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    try:
        assert validate_report(report) is None
    except ReportSchemaError as exc:
        raise AssertionError(f"report refused: {exc.code} at {exc.field}: {exc}") from exc


def with_snapshots(report: dict, older: dict | None = None) -> dict:
    """``report`` with the RV4 snapshot fields spliced in from its own `pool`
    (one snapshot, every window scored from it). ``older`` (``{snapshot,
    sha256, first_seen_roots}``) adds a second, older snapshot that the first
    window is scored from, as a W5b union would."""
    out = copy.deepcopy(report)
    pool = out["pool"]
    snaps = [{"snapshot": pool["snapshot"], "sha256": pool["sha256"],
              "first_seen_roots": list(pool["first_seen_roots"])}]
    for w in out["windows"]:
        w["snapshot_sha256"] = pool["sha256"]
    if older is not None:
        snaps.insert(0, copy.deepcopy(older))
        pool["first_seen_roots"] = sorted(set(pool["first_seen_roots"])
                                          | set(older["first_seen_roots"]))
        for c in out["cohorts"]:
            c["first_seen_roots"] = list(pool["first_seen_roots"])
        if out["windows"]:
            out["windows"][0]["snapshot_sha256"] = older["sha256"]
    pool["snapshots"] = snaps
    return out


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ── the §1.3b gate record (RV4) ─────────────────────────────────────────────
#
# What W5e will write as `report["gate"]`: W3b1's `gate_record(cfg, boundary,
# block_hash, sentinel=...)` (`suite`, `block`, `block_hash`, `sentinel`,
# `eligible`) plus `results`, one per candidate swept at that boundary,
# projected from W3b2's `paired_gate`: `digest`, `outcome`, `lcb` (the four
# LCBs, `null` on `checkpoint_fault` / `bench_hold`), `blocked_by` (`[]`
# unless `gate_regression`) and `session`. No free text (§14.3 item 4), so
# `paired_gate`'s `reason` string is not carried.

GATE_SENTINEL = "gift"
GATE_SESSION = "valpriv-4090/w7a/1"
GATE_ITEMS = 24


def gate_block_hash(boundary: int) -> str:
    """A fixture boundary block hash: ``0x`` + 64 lowercase hex."""
    return "0x" + text_digest(f"w7a boundary {int(boundary)}")


def hash_int(text: str) -> int:
    return int(text_digest(text)[:12], 16)


def _sweep_rows(suite: str, tag: str, factor: float) -> list[dict]:
    rng = np.random.default_rng(hash_int(f"{suite}|{tag}"))
    base = np.random.default_rng(hash_int(f"base|{suite}"))
    out = []
    for i in range(GATE_ITEMS):
        c, m = float(np.exp(base.normal(-0.7, 0.5))), float(np.exp(base.normal(0.1, 0.3)))
        out.append({"item": f"{suite}/task-{i:03d}",
                    "crps": c * factor * float(np.exp(rng.normal(0.0, 0.01))),
                    "mase": m * factor * float(np.exp(rng.normal(0.0, 0.01)))})
    return out


def sweep_records(candidate: str, suites, *, cand_factor: float = 1.0,
                  faulty: str | None = None) -> tuple[dict, dict, dict]:
    """``(candidate, king_d, house)`` sweep records of one host session.
    ``cand_factor`` > 1 makes the candidate worse than both references;
    ``faulty`` (a role) gives that role a NaN CRPS on every suite."""
    def rec(role: str, dg: str, factor: float) -> dict:
        suites_out = {}
        for s in suites:
            rows = _sweep_rows(s, role, factor)
            if role == faulty:
                rows[0]["crps"] = float("nan")
            suites_out[s] = {"status": "ok", "rows": rows}
        return {"session": GATE_SESSION, "digest": dg, "suites": suites_out}

    return (rec("candidate", candidate, cand_factor),
            rec("king_d", text_digest("w7a gate king_d"), 1.0),
            rec("house", text_digest("w7a gate house"), 1.0))


def gate_result(paired: dict, digest: str) -> dict:
    """A W3b2 `paired_gate` result projected onto one `gate.results[]` item."""
    lcb = paired.get("lcb")
    return {"digest": digest, "outcome": paired["outcome"],
            "lcb": None if lcb is None else {r: {m: float(lcb[r][m]) for m in GATE_METRICS}
                                             for r in GATE_REFS},
            "blocked_by": list(paired.get("blocked_by") or []),
            "session": paired["session"]}


def build_gate(cfg, boundary: int, candidates: dict[str, dict]) -> dict:
    """The gate record for ``boundary`` from the real W3b1 draw and W3b2 gate.
    ``candidates`` maps a digest to `sweep_records` keyword arguments."""
    from cascade.validator.lane_d_bench import paired_gate
    from cascade.validator.lane_d_gate import gate_record

    gate = gate_record(cfg, int(boundary), gate_block_hash(boundary), sentinel=GATE_SENTINEL)
    results = []
    for dg, kw in candidates.items():
        cand, king, house = sweep_records(dg, (gate["suite"], GATE_SENTINEL), **kw)
        paired = paired_gate(cand, king, house, suite=gate["suite"], alpha=float(cfg.alpha),
                             tau_public=float(cfg.tau_public), B=200, seed=f"w7a|{boundary}",
                             sentinel=GATE_SENTINEL)
        results.append(gate_result(paired, dg))
    return {**gate, "results": results}


# ── rev6.10-rev6.12 report fields, spliced (W5a / W5j / W5i not built yet) ──
#
# What the producer writes once W5a, W5j and W2e land; built here from the
# report's own windows, rows and the fixture chain so the validator tests can
# start from a correct report. README "Correction (rev6.11, rev6.12)".


def subscore(rows, digest: str, window_ids) -> dict:
    """`{n, geomean}` of `digest`'s rows on `window_ids` (the lane D `_geo`:
    sqrt of the CRPS and MASE geometric means, floored at 1e-12 / 1e-9);
    `geomean` is null when n = 0."""
    ids = set(window_ids)
    rs = [r for r in rows if r["digest"] == digest and r["window_id"] in ids]
    if not rs:
        return {"n": 0, "geomean": None}
    lc = sum(math.log(max(float(r["crps"]), 1e-12)) for r in rs) / len(rs)
    lm = sum(math.log(max(float(r["mase"]), 1e-9)) for r in rs) / len(rs)
    return {"n": len(rs), "geomean": math.sqrt(math.exp(lc) * math.exp(lm))}


def with_embargo_fields(report: dict, *, embargo_windows: int = 0) -> dict:
    """`report` with every rev6.10-rev6.12 field spliced in (fixture feeds pin
    e = 0, so every window is fresh, rule `embargo0`). `embargo_windows` = k
    turns the first k windows of cohort 0 into embargo windows (e = 48, rule
    `embargo`, data time moved inside (b - 48 h, b]) and recomputes the
    counts and sub-scores."""
    out = copy.deepcopy(report)
    params = out["params"]
    for k, v in W2E_DEFAULTS.items():
        params.setdefault(k, v)
    emb_ids: set[str] = set()
    if embargo_windows and out["cohorts"]:
        emb_ids = set(out["cohorts"][0]["window_ids"][:embargo_windows])
    margin = HOUR * float(params["forward_margin_hours"])
    b0 = int(out["cohorts"][0]["t_max"] + margin) if out["cohorts"] else 0
    for w in out["windows"]:
        sid, origin, htag = w["window_id"].rsplit(":", 2)
        emb = w["window_id"] in emb_ids
        e = float(params["embargo_hours"]) if emb else 0.0
        if emb:
            w["min_target_ts"] = b0 - 10 * HOUR
            w["min_first_seen_ts"] = b0 - 10 * HOUR + FIRST_SEEN_LAG
        w.update(freq="H", embargo_hours=e,
                 min_release_ts=int(w["min_first_seen_ts"]) + int(round(HOUR * e)),
                 rule="embargo" if e > 0 else "embargo0", embargo_window=emb,
                 timeframe="H", bin="forward", kind="forward", freq_class="subdaily",
                 h=int(htag[1:]), origin=int(origin))
    flag = {w["window_id"]: w["embargo_window"] for w in out["windows"]}
    for c in out["cohorts"]:
        c["fresh_windows"] = sum(1 for x in c["window_ids"] if not flag[x])
        c["embargo_windows"] = sum(1 for x in c["window_ids"] if flag[x])
    for v in out["verdicts"]:
        c = out["cohorts"][int(v["cohort"])]
        v["fresh"] = subscore(out["rows"], v["digest"],
                              [x for x in c["window_ids"] if not flag[x]])
        v["embargo"] = subscore(out["rows"], v["digest"],
                                [x for x in c["window_ids"] if flag[x]])
    if out["cohorts"]:
        bh = gate_block_hash(int(out["boundary"]))
        out["draw"] = {
            "boundary_block_hash": bh, "seed": f"{bh}:lane_d_windows",
            "max_windows_per_settlement": int(params["max_windows_per_settlement"]),
            "cohorts": [{"index": int(c["index"]),
                         "quota": {"H|forward": len(c["window_ids"])},
                         "capacity": {"H|forward": len(c["window_ids"])},
                         "candidates_sha256": hashlib.sha256(
                             "\n".join(sorted(c["window_ids"])).encode()).hexdigest()}
                        for c in out["cohorts"]]}
    return out


def _rev614_ts(settle, boundary) -> dict:
    """rev6.14 (a): the fixture's on-chain boundary timestamp (far after every
    fixture T + eval_delay_hours, 12 s per block), passed only where
    ``settle_lane_d`` takes it."""
    import inspect

    if "boundary_ts" not in inspect.signature(settle).parameters:
        return {}
    return {"boundary_ts": 2_000_000_000 + 12 * int(boundary)}
