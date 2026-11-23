"""W5a embargo through `settle_lane_d` (plan rev6.10 (a), rev6.12 (a),
invariant 4 at the feed bound).

An armed settlement on a snapshot whose feeds pin e = 48: embargo windows
inside (b - 48 h, b], hourly and 5-min alike (rev6.12: every cadence), are
judgeable, and no member row is on a window the independent rev6.10 oracle
refuses. How many embargo windows a frozen set keeps is W5j's
`min_fresh_window_share`, not pinned here. `[lane_d] embargo_hours` comes from the real W2e parse
(`lane_d_cfg(embargo_hours=48)`).

On main no embargo window is scored: an AssertionError.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .helpers import (
    EMBARGO_H,
    HORIZON,
    HOUR,
    T0,
    bound_of,
    case,
    feeds_block,
    fixture_key,
    gate_inputs,
    hourly,
    lane_d_cfg,
    oracle_embargo_reason,
    quantiles_around,
    spec,
    target_values,
    window_id,
    write_pool,
)

ARM_BLOCK = 2700 * 3600
BOUNDARY_1 = ARM_BLOCK + 900
HOUSE_T = T0 - 2 * 86_400


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _entry(name: str, reveal_ts: int) -> dict:
    return {"hotkey": fixture_key(name).ss58_address, "digest": _digest(f"w5a {name}"),
            "context": 4096, "coldkey_at_ready": fixture_key(f"{name}-cold").ss58_address,
            "ready_block": ARM_BLOCK + 40, "reveal_block": ARM_BLOCK + 60,
            "reveal_ts": int(reveal_ts)}


def _embargoed(series_id: str, source: str, bound: int, k: int, freq: str) -> dict:
    ts = hourly(bound - (40 - k) * HOUR)
    seen = [bound - (30 - k) * HOUR + i for i in range(HORIZON)]
    return spec(series_id, source, ts, seen, freq=freq,
                frontier=[[bound - (44 - k) * HOUR, bound - (45 - k) * HOUR]])


def test_settle_judges_embargo_windows_of_every_cadence(tmp_path):
    from cascade.validator.lane_d import settle_lane_d

    entry = _entry("embargo-challenger", T0)
    house = {"digest": _digest("w5a embargo house"), "reveal_ts": HOUSE_T}
    cfg = lane_d_cfg(min_windows=24, min_clusters=4, embargo_hours=EMBARGO_H)
    bound = int(bound_of(T0, HOUSE_T, cfg))
    sources = [f"src{j}" for j in range(6)]
    series = [case("eligible", bound, f"fresh-{s}-{k}", s) for s in sources for k in range(8)]
    hourly_emb = [_embargoed(f"emb-h-{k}", sources[k % 6], bound, k, "H") for k in range(12)]
    fast_emb = [_embargoed(f"emb-5m-{k}", sources[k % 6], bound, k, "5min") for k in range(12)]
    series += hourly_emb + fast_emb
    feeds = feeds_block(sources, embargo_hours=EMBARGO_H)
    pool_dir = write_pool(tmp_path / "pool", series, feeds=feeds, snapshot="w5a-embargo")
    rng = np.random.default_rng(7)
    fc = {entry["digest"]: {}, house["digest"]: {}}
    for rec in series:
        _, y = target_values(rec["series_id"])
        sd = float(np.std(y)) or 1.0
        wid = window_id(rec["series_id"], rec["target_ts"])
        fc[entry["digest"]][wid] = quantiles_around(y + rng.normal(0, 0.05 * sd, HORIZON),
                                                    0.3 * sd)
        fc[house["digest"]][wid] = quantiles_around(y + 3.0 * sd, 0.3 * sd)
    s = settle_lane_d(cfg=cfg, boundary=BOUNDARY_1, pool_dir=pool_dir, entries=[dict(entry)],
                      house=dict(house), forecasts=fc, uid_of={entry["hotkey"]: 6},
                      state_path=tmp_path / "state" / "lane_d_state.json",
                      signer=fixture_key("signer"), **gate_inputs(BOUNDARY_1, house["digest"]))
    assert s.report is not None
    by_wid = {window_id(r["series_id"], r["target_ts"]): r for r in series}
    member = {r["window_id"] for r in s.report["rows"] if r["digest"] == entry["digest"]}
    assert member, "nothing judged (the fixture meets the floors)"
    for wid in member:
        why = oracle_embargo_reason(by_wid[wid], feeds, float(bound), cfg, EMBARGO_H)
        assert why is None, f"scored {wid} though the embargo-aware rule refuses it: {why}"
    emb = {window_id(r["series_id"], r["target_ts"]) for r in hourly_emb + fast_emb}
    assert member & emb, "no embargo window was judged at e = 48"
