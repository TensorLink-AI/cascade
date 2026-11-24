"""W5a through `settle_lane_d` (plan rev6 §3.2 invariant 4, T8, the W1.5 row
"every later WP keeps it green and adds >= 1 assertion").

* invariant 4: over random pool_d fixtures (both T orders, one or two
  entries, random margin / lag limits, every T8 case mixed in) no report row
  of a member is on a window that fails §1.3 for max(T_E, T_K);
* invariant 10 pair: unarmed nothing is judged or written; armed the trap
  windows are refused and the eligible ones scored;
* the skeleton end to end with trap windows on which the challenger is
  catastrophically wrong: refused, so it still clears, is crowned (rev6.11:
  at its first verdict) and is paid.

On main the settlement scores the clock-passing traps (over-lag, forecast
product, backfill, absent from S0), so these fail on an assertion.

RV3 BLOCKING 1 adds fail-closed traps: a series with no or a null
`frontier` (refused `absent_from_s0`) and a feed missing from `feeds`
(refused `unlisted_feed`). Main scores those too: AssertionError.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .helpers import (
    CLOCK_TRAPS,
    FAIL_CLOSED_KINDS,
    HORIZON,
    HOUR,
    T0,
    TRAP_KINDS,
    bound_of,
    case,
    feeds_block,
    fixture_key,
    gate_inputs,
    lane_d_cfg,
    oracle_reason,
    quantiles_around,
    target_values,
    window_id,
    write_pool,
)

ARM_BLOCK = 2700 * 3600
BOUNDARY_1 = ARM_BLOCK + 900
BOUNDARY_2 = ARM_BLOCK + 1800
HOUSE_T = T0 - 2 * 86_400
CHALLENGER_UID = 6
N_UIDS = 8


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _entry(name: str, reveal_ts: int) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": _digest(f"w5a {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": ARM_BLOCK + 40, "reveal_block": ARM_BLOCK + 60,
            "reveal_ts": int(reveal_ts)}


def _settle(cfg, boundary, pool_dir, entries, house, forecasts, uid_of, state_path):
    """`settle_lane_d`, with the W5e gate inputs when it accepts them
    (`helpers.gate_inputs`): the e2e skeleton crowns at BOUNDARY_2."""
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=pool_dir,
                         entries=[dict(e) for e in entries], house=dict(house),
                         forecasts=forecasts, uid_of=dict(uid_of), state_path=state_path,
                         signer=fixture_key("signer"),
                         **gate_inputs(boundary, house["digest"]))


def _good_forecasts(series, digests, spread=0.5):
    fc = {d: {} for d in digests}
    for s in series:
        _, y = target_values(s["series_id"])
        for d in digests:
            fc[d][window_id(s["series_id"], s["target_ts"])] = quantiles_around(y, spread)
    return fc


# ── invariant 4: property over random fixtures ──────────────────────────────


def test_inv4_settle_never_scores_an_ineligible_window(tmp_path):
    from cascade.validator.lane_d_forward import eligible_windows  # noqa: I001
    from cascade.validator.lane_d import load_pool

    seen_reasons: set[str] = set()
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        cfg = lane_d_cfg(forward_margin_hours=float(rng.choice([0.0, 6.0, 12.0])),
                         max_mirror_lag_hours=float(rng.choice([3.0, 6.0, 9.0])))
        t_king = int(T0 + int(rng.integers(-72, 73)) * HOUR)
        entries = [_entry(f"p{seed}-e{i}", T0 + int(rng.integers(0, 49)) * HOUR)
                   for i in range(int(rng.integers(1, 3)))]
        house = {"digest": _digest(f"w5a house {seed}"), "reveal_ts": t_king}
        bound_all = bound_of(max(e["reveal_ts"] for e in entries), t_king, cfg)
        sources = [f"src{j}" for j in range(4)]
        series = []
        for j, src in enumerate(sources):           # always eligible: meets the floors
            for k in range(4):
                series.append(case("eligible", int(bound_all), f"p{seed}-ok-{j}-{k}", src))
        kinds = ("eligible", *FAIL_CLOSED_KINDS, *TRAP_KINDS)
        for n in range(40):
            kind = str(rng.choice(kinds))
            shift = int(rng.choice([-72, -30, 0, 0, 24])) * HOUR
            series.append(case(kind, int(bound_all) + shift, f"p{seed}-r{n}",
                               sources[n % len(sources)]))
        feeds = feeds_block(sources)
        feeds["laggy"]["mirror_lag_hours"] = float(rng.uniform(1.0, 12.0))
        pool_dir = write_pool(tmp_path / f"pool{seed}", series, feeds=feeds,
                              snapshot=f"w5a-prop-{seed}")
        digests = [house["digest"], *(e["digest"] for e in entries)]
        uid_of = {e["hotkey"]: 5 + i for i, e in enumerate(entries)}
        s = _settle(cfg, BOUNDARY_1, pool_dir, entries, house,
                    _good_forecasts(series, digests), uid_of,
                    tmp_path / f"state{seed}" / "lane_d_state.json")
        assert s.report is not None, f"seed {seed}: armed settlement produced no report"
        by_wid = {window_id(r["series_id"], r["target_ts"]): r for r in series}
        t_of = {e["digest"]: e["reveal_ts"] for e in entries}
        member_rows = [r for r in s.report["rows"] if r["digest"] != house["digest"]]
        assert member_rows, f"seed {seed}: nothing judged (fixture meets the floors)"
        member_wids = {r["window_id"] for r in member_rows}
        for r in member_rows:
            bound = bound_of(t_of[r["digest"]], t_king, cfg)
            why = oracle_reason(by_wid[r["window_id"]], feeds, bound, cfg)
            assert why is None, (f"seed {seed}: scored {r['window_id']} for "
                                 f"{r['digest'][:8]} though it fails §1.3: {why}")
        king_wids = {r["window_id"] for r in s.report["rows"]
                     if r["digest"] == house["digest"]}
        assert king_wids <= member_wids, f"seed {seed}: king rows outside any member set"

        pool = load_pool(pool_dir, HORIZON)
        for e in entries:
            bound = bound_of(e["reveal_ts"], t_king, cfg)
            got = {v.window_id: (v.eligible, v.reason) for v in
                   eligible_windows(pool, t_entry=e["reveal_ts"], t_king=t_king, cfg=cfg)}
            want = {}
            for wid, rec in by_wid.items():
                why = oracle_reason(rec, feeds, bound, cfg)
                want[wid] = (why is None, why)
                if why:
                    seen_reasons.add(why)
            assert got == want, f"seed {seed}: eligible_windows disagrees with §1.3"
    missing = set(TRAP_KINDS) - seen_reasons
    assert not missing, f"fixture never exercised: {sorted(missing)}"


# ── shared trap fixture (one or two snapshots) ──────────────────────────────


def _trap_snapshot(tmp_path, snap: int, chal: str, house: str, bound: int):
    """48 eligible windows (6 sources x 8) on which the challenger is good and
    the house 3 sd off, plus traps of every T8 kind on which the house is exact
    and the challenger 1000 sd off. Scoring any trap sinks the challenger."""
    sources = [f"src{j}" for j in range(6)]
    series, traps = [], set()
    for src in sources:
        for k in range(8):
            series.append(case("eligible", bound + (snap - 1) * 4 * 86_400,
                               f"s{snap}-{src}-{k}", src))
    for kind in (*TRAP_KINDS, *FAIL_CLOSED_KINDS):
        for k in range(4 if kind in CLOCK_TRAPS else 12):
            rec = case(kind, bound, f"s{snap}-{kind}-{k}", sources[k % 6])
            series.append(rec)
            traps.add(window_id(rec["series_id"], rec["target_ts"]))
    rng = np.random.default_rng(snap)
    fc = {chal: {}, house: {}}
    for rec in series:
        _, y = target_values(rec["series_id"])
        sd = float(np.std(y)) or 1.0
        wid = window_id(rec["series_id"], rec["target_ts"])
        if wid in traps:
            fc[house][wid] = quantiles_around(y, 1e-6 * sd)
            fc[chal][wid] = quantiles_around(y + 1000.0 * sd, 0.3 * sd)
        else:
            fc[chal][wid] = quantiles_around(y + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
            fc[house][wid] = quantiles_around(y + 3.0 * sd, 0.3 * sd)
    feeds = feeds_block(sources)
    pool_dir = write_pool(tmp_path / f"pool_d_{snap}", series, feeds=feeds,
                          snapshot=f"w5a-trap-{snap}")
    return pool_dir, fc, traps, feeds, {window_id(r["series_id"], r["target_ts"]): r
                                        for r in series}


def _trap_chain(tmp_path):
    entry = _entry("challenger", T0)
    house = {"digest": _digest("w5a trap house"), "reveal_ts": HOUSE_T}
    cfg = lane_d_cfg(min_windows=24, min_clusters=4)
    bound = int(bound_of(T0, HOUSE_T, cfg))
    snaps = {n: _trap_snapshot(tmp_path, n, entry["digest"], house["digest"], bound)
             for n in (1, 2)}
    return cfg, entry, house, snaps, {entry["hotkey"]: CHALLENGER_UID}


def _assert_no_trap_scored(report, traps, by_wid, feeds, cfg, *, require_rows=True):
    wids = {r["window_id"] for r in report["rows"]}
    assert wids or not require_rows, "the armed settlement scored no rows"
    scored_traps = sorted(wids & traps)
    assert not scored_traps, f"ineligible windows were scored: {scored_traps[:5]}"
    bound = bound_of(T0, HOUSE_T, cfg)
    for wid in wids:
        assert oracle_reason(by_wid[wid], feeds, bound, cfg) is None, wid


# ── invariant 4 / invariant 10 pair ─────────────────────────────────────────


def test_inv4_window_set_unchanged_when_unarmed(tmp_path):
    cfg, entry, house, snaps, uid_of = _trap_chain(tmp_path)
    pool_dir, fc, *_ = snaps[1]
    unarmed = lane_d_cfg(from_block=0, min_windows=24, min_clusters=4)
    assert not unarmed.active(BOUNDARY_1)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    s = _settle(unarmed, BOUNDARY_1, pool_dir, [entry], house, fc, uid_of,
                state_dir / "lane_d_state.json")
    assert s.report is None and s.pay is None
    assert list(state_dir.iterdir()) == [], "unarmed lane D wrote state"


def test_inv4_window_set_armed_nonempty_lane_d_effect(tmp_path):
    cfg, entry, house, snaps, uid_of = _trap_chain(tmp_path)
    pool_dir, fc, traps, feeds, by_wid = snaps[1]
    s = _settle(cfg, BOUNDARY_1, pool_dir, [entry], house, fc, uid_of,
                tmp_path / "state" / "lane_d_state.json")
    assert s.report is not None and s.pay is not None
    _assert_no_trap_scored(s.report, traps, by_wid, feeds, cfg)
    member = {r["window_id"] for r in s.report["rows"] if r["digest"] == entry["digest"]}
    assert len(member) == 48, f"expected the 48 eligible windows scored; got {len(member)}"


# ── the W1.5 skeleton end to end, plus the W5a assertion ────────────────────


def test_e2e_skeleton_refuses_trap_windows_and_still_pays_the_challenger(tmp_path):
    from cascade.shared.emission import lane_d_vector

    cfg, entry, house, snaps, uid_of = _trap_chain(tmp_path)
    state_path = tmp_path / "state" / "lane_d_state.json"
    reports = []
    for boundary, n in ((BOUNDARY_1, 1), (BOUNDARY_2, 2)):
        pool_dir, fc, traps, feeds, by_wid = snaps[n]
        s = _settle(cfg, boundary, pool_dir, [entry], house, fc, uid_of, state_path)
        assert s.report is not None, f"boundary {boundary}: no report"
        # rev6.11 (b): the challenger is crowned at its first verdict
        # (BOUNDARY_1); BOUNDARY_2 may then judge nothing.
        _assert_no_trap_scored(s.report, traps, by_wid, feeds, cfg,
                               require_rows=boundary == BOUNDARY_1)
        reports.append(s)
    s2 = reports[-1]
    state = json.loads(state_path.read_text())
    assert state["king"]["digest"] == entry["digest"], \
        "the challenger was not crowned: trap windows sank it"
    assert s2.pay["court_uids"][0] == CHALLENGER_UID
    assert hasattr(cfg, "court_weights"), "[lane_d] court_weights is not parsed (rev6.23)"
    vec = lane_d_vector(N_UIDS, share=s2.pay["share"], court_uids=s2.pay["court_uids"],
                        court_weights=cfg.court_weights, burn_uid=0)
    assert vec[CHALLENGER_UID] > 0.0
