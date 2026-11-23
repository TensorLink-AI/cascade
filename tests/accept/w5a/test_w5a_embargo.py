"""W5a embargo-aware forward eligibility (plan rev6.10 (a), rev6.12 (a)/(b);
§3.0 W5a row; README "Embargo").

    b = wall(max(T_E, T_K)) + forward_margin
    e = min(feeds[feed].embargo_hours, [lane_d] embargo_hours)   for EVERY freq
    eligible iff, for every target point, first_seen + e > b and data_ts > b - e;
    S0 / backfill at b - e.

e = 0 is the rev3 rule byte for byte. A listed feed whose `embargo_hours` is
missing or not a non-negative number fails closed as `unlisted_feed`. Each
eligible verdict carries its effective `embargo_hours` and `embargo_window`:
`False` for a fresh window (it passes at e = 0 on every point), `True` for an
embargo window (it doesn't, a target straddling `b` included).

On main the embargo is not implemented: the embargo windows are refused, the
invalid-feed windows are admitted and the verdicts carry no `embargo_hours` /
`embargo_window`, so the changed tests fail on an assert. The tests never
read `cfg.embargo_hours` (`helpers.with_embargo`).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from .helpers import (
    CADENCE_FREQS,
    EMBARGO_H,
    FAIL_CLOSED_KINDS,
    HORIZON,
    HOUR,
    MARGIN_H,
    T0,
    TRAP_KINDS,
    case,
    feeds_block,
    hourly,
    lane_d_cfg,
    oracle_embargo_reason,
    oracle_fresh,
    oracle_reason,
    spec,
    window_id,
    with_embargo,
    write_pool,
)

KING_T = T0 - 2 * 86_400
B = int(T0 + MARGIN_H * HOUR)          # b = max(T_E, T_K) + margin, T_E > T_K


def _cfg(e: float = EMBARGO_H, **over):
    return with_embargo(lane_d_cfg(**over), e)


def _verdicts(tmp_path, series, feeds, *, cfg=None, name="pool", t_entry=T0, t_king=KING_T):
    from cascade.validator.lane_d import load_pool  # noqa: I001
    from cascade.validator.lane_d_forward import eligible_windows

    pool = load_pool(write_pool(tmp_path / name, series, feeds=feeds), HORIZON)
    return list(eligible_windows(pool, t_entry=t_entry, t_king=t_king, cfg=cfg or _cfg()))


def _one(tmp_path, rec, feeds, **kw):
    (v,) = _verdicts(tmp_path, [rec], feeds, **kw)
    assert v.window_id == window_id(rec["series_id"], rec["target_ts"])
    return v


def _reason(v):
    return None if v.reason is None else str(v.reason)


def _embargoed(series_id: str, *, freq: str = "H", start_h: int = -40, seen_h: int = -30,
               source: str = "srcA", frontier=None, feed=None) -> dict:
    """A window whose data and first-seen clocks are both inside (b - 48 h, b]:
    eligible only through the embargo. Its S0 sits after b - 48 h, before
    the target."""
    ts = hourly(B + start_h * HOUR)
    seen = [B + seen_h * HOUR + i for i in range(HORIZON)]
    if frontier is None:
        frontier = [[B + (start_h - 4) * HOUR, B + (start_h - 5) * HOUR]]
    return spec(series_id, source, ts, seen, frontier=frontier, feed=feed, freq=freq)


def _mixed(series_id: str, *, freq: str = "H") -> dict:
    """A target straddling b: the first 30 points in (b - 48 h, b], the rest
    after b; every point first seen 30 min after its data time."""
    ts = hourly(B - 29 * HOUR)
    seen = [t + 1800 for t in ts]
    return spec(series_id, "srcA", ts, seen, freq=freq,
                frontier=[[B - 33 * HOUR, B - 34 * HOUR]])


# ── the rule at e > 0 ───────────────────────────────────────────────────────


def test_window_inside_the_embargo_is_eligible_at_e48(tmp_path):
    rec = _embargoed("emb")
    feeds = feeds_block({"srcA"}, embargo_hours=EMBARGO_H)
    v = _one(tmp_path, rec, feeds)
    assert v.eligible is True and v.reason is None, \
        f"a window inside (b - 48 h, b] must pass at e = 48; got {_reason(v)}"


def test_same_window_at_e0_is_the_rev3_refusal(tmp_path):
    rec = _embargoed("emb0")
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=0.0))
    assert (v.eligible, _reason(v)) == (False, "before_margin")
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=EMBARGO_H),
             cfg=_cfg(0.0), name="pool2")
    assert (v.eligible, _reason(v)) == (False, "before_margin"), \
        "[lane_d] embargo_hours = 0 must give the rev3 rule whatever the feed says"


@pytest.mark.parametrize("freq", CADENCE_FREQS)
def test_every_cadence_gets_the_embargo(tmp_path, freq):
    """rev6.12 (a): no sub-hourly zero; `freq` plays no part in e."""
    rec = _embargoed(f"cad-{freq}", freq=freq)
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=EMBARGO_H))
    assert v.eligible is True, f"freq {freq!r}: e = 48 for every cadence; got {_reason(v)}"


def test_a_target_straddling_b_is_eligible_at_e48_only(tmp_path):
    """rev6.12 (a): one window may mix embargoed and post-submission points;
    both halves pass the same rule. At e = 0 it is refused."""
    rec = _mixed("mix")
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=EMBARGO_H))
    assert v.eligible is True, f"a mixed target must pass at e = 48; got {_reason(v)}"
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=0.0), name="pool2")
    assert v.eligible is False and _reason(v) == "before_margin"


def test_catalog_value_can_only_shorten(tmp_path):
    """e = min(feed, [lane_d] embargo_hours): a feed value of 24 shortens the
    embargo, a feed value of 96 is clamped to the config's 48."""
    deep = _embargoed("deep", start_h=-30, seen_h=-26)       # inside 48 h, outside 24 h
    shallow = _embargoed("shallow", start_h=-20, seen_h=-18,   # inside 24 h
                         frontier=[[B - 22 * HOUR, B - 23 * HOUR]])
    v24 = {v.window_id: v for v in _verdicts(
        tmp_path, [deep, shallow], feeds_block({"srcA"}, embargo_hours=24.0))}
    assert _reason(v24[window_id("deep", deep["target_ts"])]) == "before_margin"
    assert v24[window_id("shallow", shallow["target_ts"])].eligible is True
    far = _embargoed("far", start_h=-60, seen_h=-55)          # inside 96 h, outside 48 h
    v = _one(tmp_path, far, feeds_block({"srcA"}, embargo_hours=96.0), name="pool2")
    assert (v.eligible, _reason(v)) == (False, "before_margin"), \
        "a feed embargo above [lane_d] embargo_hours must be clamped to it"
    v = _one(tmp_path, far, feeds_block({"srcA"}, embargo_hours=96.0), cfg=_cfg(96.0),
             name="pool3")
    assert v.eligible is True, "embargo_hours is read from cfg"


@pytest.mark.parametrize("clock", ["data", "seen"])
def test_feed_bound_is_strict_on_both_clocks(tmp_path, clock):
    """data_ts > b - e and first_seen > b - e (i.e. first_seen + e > b),
    strictly; the labels keep their meaning against the feed bound."""
    fb = int(B - EMBARGO_H * HOUR)
    ts = hourly(fb + 3 * HOUR)
    seen = [t + 1800 for t in ts]
    if clock == "data":
        ts, ts_ok, seen_ok = [fb] + ts[1:], [fb + 1] + ts[1:], seen
        want = "late_mirror"
    else:
        seen, seen_ok, ts_ok = [fb] + seen[1:], [fb + 1] + seen[1:], ts
        want = "publish_ahead"
    fr = [[fb + 2 * HOUR, fb - HOUR]]
    feeds = feeds_block({"srcA"}, embargo_hours=EMBARGO_H)
    v = _one(tmp_path, spec("edge", "srcA", ts, seen, frontier=fr), feeds)
    assert (v.eligible, _reason(v)) == (False, want)
    v = _one(tmp_path, spec("edge", "srcA", ts_ok, seen_ok, frontier=fr), feeds, name="pool2")
    assert v.eligible is True, f"one second after b - e must pass; got {_reason(v)}"


def test_every_target_point_counts_against_the_feed_bound(tmp_path):
    rec = _embargoed("tail")
    rec["first_seen_ts"][-1] = int(B - (EMBARGO_H + 1) * HOUR)
    v = _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=EMBARGO_H))
    assert (v.eligible, _reason(v)) == (False, "publish_ahead")


def test_s0_and_backfill_use_the_feed_bound(tmp_path):
    """S0 = the earliest frontier record with as_of_ts > b - e."""
    feeds = feeds_block({"srcA"}, embargo_hours=EMBARGO_H)
    before = _embargoed("s0-before", frontier=[[B - 50 * HOUR, B - 51 * HOUR]])
    assert _reason(_one(tmp_path, before, feeds)) == "absent_from_s0"
    after = _embargoed("s0-after", frontier=[[B - 50 * HOUR, B - 51 * HOUR],
                                             [B - 45 * HOUR, B - 46 * HOUR]])
    assert _one(tmp_path, after, feeds, name="pool2").eligible is True, \
        "a frontier record in (b - e, b] is S0 at e = 48"
    bf = _embargoed("bf", frontier=[[B - 36 * HOUR, B - 38 * HOUR]])   # target starts b - 40 h
    assert _reason(_one(tmp_path, bf, feeds, name="pool3")) == "backfill", \
        "a target starting at or before end_ts(S0) is a backfill"


@pytest.mark.parametrize("freq", ["10s", "H"])
@pytest.mark.parametrize("value", ["absent", None, "48", True, -1.0])
def test_bad_feed_embargo_fails_closed_as_unlisted_feed(tmp_path, value, freq):
    """W5a row: a missing or non-numeric `embargo_hours` in a listed feed fails
    closed as `unlisted_feed`, for every cadence (W6b refuses a negative
    catalog value the same way)."""
    ts = hourly(B + 3 * HOUR)
    rec = spec("bad-e", "srcA", ts, [t + 1800 for t in ts], frontier=[[B + 2 * HOUR, B + HOUR]],
               freq=freq)
    feeds = feeds_block({"srcA"})
    if value == "absent":
        del feeds["srcA"]["embargo_hours"]
    else:
        feeds["srcA"]["embargo_hours"] = value
    v = _one(tmp_path, rec, feeds)
    assert (v.eligible, _reason(v)) == (False, "unlisted_feed"), \
        f"feed embargo_hours {value!r}: got {_reason(v)}"


def test_bad_feed_embargo_ranks_after_missing_clock(tmp_path):
    rec = case("eligible", B, "miss", "srcA")
    rec["end_ts"] = None
    feeds = feeds_block({"srcA"})
    del feeds["srcA"]["embargo_hours"]
    assert _reason(_one(tmp_path, rec, feeds)) == "missing_clock"


def test_zero_is_a_valid_feed_embargo(tmp_path):
    rec = case("eligible", B, "zero", "srcA")
    assert _one(tmp_path, rec, feeds_block({"srcA"}, embargo_hours=0)).eligible is True


# ── verdict fields: the effective e and the fresh / embargo class ───────────


def test_verdict_carries_effective_embargo_and_embargo_window_flag(tmp_path):
    """README "Verdict fields": an eligible verdict carries `embargo_hours`
    (the effective e) and `embargo_window` (rev6.12 terms: `False` = fresh,
    passes at e = 0 on every point; `True` = embargo, incl. a mixed target).
    W5j's fresh-first draw and `min_fresh_window_share` read it."""
    fresh_ts = hourly(B + 3 * HOUR)
    fresh = spec("fresh", "srcA", fresh_ts, [t + 1800 for t in fresh_ts],
                 frontier=[[B + 2 * HOUR, B + HOUR]])
    fast = spec("fast", "srcA", fresh_ts, [t + 1800 for t in fresh_ts],
                frontier=[[B + 2 * HOUR, B + HOUR]], freq="1min")
    emb = _embargoed("emb")
    emb_fast = _embargoed("emb-fast", freq="10s")
    mix = _mixed("mix")
    series = [fresh, fast, emb, emb_fast, mix]
    got = {v.window_id: v for v in _verdicts(
        tmp_path, series, feeds_block({"srcA"}, embargo_hours=EMBARGO_H))}
    want = {"fresh": False, "fast": False, "emb": True, "emb-fast": True, "mix": True}
    for rec in series:
        sid = rec["series_id"]
        v = got[window_id(sid, rec["target_ts"])]
        assert v.eligible is True, f"{sid}: {_reason(v)}"
        e = getattr(v, "embargo_hours", None)
        flag = getattr(v, "embargo_window", None)
        assert e is not None and (float(e), flag) == (EMBARGO_H, want[sid]), \
            f"{sid}: (embargo_hours, embargo_window) = ({e!r}, {flag!r}); " \
            f"want ({EMBARGO_H}, {want[sid]})"
    got0 = _verdicts(tmp_path, [fresh], feeds_block({"srcA"}, embargo_hours=0.0), name="pool2")
    assert (float(getattr(got0[0], "embargo_hours", -1)),
            getattr(got0[0], "embargo_window", None)) == (0.0, False)


# ── e = 0 is byte-identical to the rev3 rule ────────────────────────────────


def _mixed_series(rng, bound: int, tag: str) -> list[dict]:
    kinds = ("eligible", *FAIL_CLOSED_KINDS, *TRAP_KINDS)
    out = []
    for n in range(60):
        kind = str(rng.choice(kinds))
        shift = int(rng.choice([-72, -50, -30, -10, 0, 0, 24])) * HOUR
        rec = case(kind, bound + shift, f"{tag}-{n}", f"src{n % 4}")
        rec["freq"] = str(rng.choice(CADENCE_FREQS))
        out.append(rec)
    return out


def _canon(verdicts) -> str:
    return json.dumps([[v.window_id, bool(v.eligible), _reason(v)] for v in verdicts])


@pytest.mark.parametrize("variant", ["feed_zero", "config_zero"])
def test_e0_is_byte_identical_to_the_rev3_verdicts(tmp_path, variant):
    """The test diffs both: the producer's verdicts at e = 0 serialise to the
    same bytes as the rev3 oracle (`helpers.oracle_reason` at bound b)."""
    rng = np.random.default_rng({"feed_zero": 12, "config_zero": 13}[variant])
    cfg0 = lane_d_cfg()
    bound = int(B)
    series = _mixed_series(rng, bound, variant)
    if variant == "feed_zero":
        feeds, cfg = feeds_block({f"src{j}" for j in range(4)}, 0.0), _cfg(EMBARGO_H)
    else:
        feeds, cfg = feeds_block({f"src{j}" for j in range(4)}, EMBARGO_H), _cfg(0.0)
    got = _verdicts(tmp_path, series, feeds, cfg=cfg)

    class _V:
        def __init__(self, rec):
            self.window_id = window_id(rec["series_id"], rec["target_ts"])
            self.reason = oracle_reason(rec, feeds, float(bound), cfg0)
            self.eligible = self.reason is None

    assert _canon(got) == _canon([_V(r) for r in series])


def test_embargo_rule_matches_the_oracle_over_random_fixtures(tmp_path):
    """Property: at e = 48 with mixed cadences, mixed feed values (0, 12, 48,
    96, missing) and every trap kind, every verdict, and every eligible
    verdict's fresh / embargo class, equals the independent rev6.10 oracle
    (`helpers.oracle_embargo_reason`, `helpers.oracle_fresh`)."""
    for seed in range(4):
        rng = np.random.default_rng(500 + seed)
        bound = int(B)
        series = _mixed_series(rng, bound, f"p{seed}")
        feeds = feeds_block({f"src{j}" for j in range(4)}, 0.0)
        for j, val in enumerate((12.0, EMBARGO_H, 96.0, "absent")):
            if val == "absent":
                del feeds[f"src{j}"]["embargo_hours"]
            else:
                feeds[f"src{j}"]["embargo_hours"] = val
        cfg = _cfg(EMBARGO_H)
        got = {v.window_id: (v.eligible, _reason(v),
                             getattr(v, "embargo_window", None) if v.eligible else None)
               for v in _verdicts(tmp_path, series, feeds, cfg=cfg, name=f"pool{seed}")}
        want = {}
        for rec in series:
            why = oracle_embargo_reason(rec, feeds, float(bound), cfg, EMBARGO_H)
            flag = None if why else not oracle_fresh(rec, feeds, float(bound), cfg)
            want[window_id(rec["series_id"], rec["target_ts"])] = (why is None, why, flag)
        diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
        assert not diff, f"seed {seed}: {len(diff)} verdicts differ, e.g. {list(diff.items())[:3]}"
