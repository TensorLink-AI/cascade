"""W7d (plan rev6.10 (a)/(b), rev6.11 (a), rev6.12 (a)/(b), W7d row): the
replay re-derives the per-window embargo record, the fresh / embargo class,
the minimum fresh share and the W5j seeded per-bin draw from the signed
report and `cfg`, never from producer code (invariant 12).

Every forgery is re-signed by the validator's key and its receipt re-issued
(`helpers.doctored`). Each mutator first asserts the field it doctors
exists, so on main (no embargo or draw fields) every case fails on an
assertion. README.md "rev6.11 / rev6.12" has the contract and field names.
"""

from __future__ import annotations

import hashlib

import pytest

from .helpers import (
    EMBARGO_HOURS,
    MARGIN_H,
    Chain,
    block_hash,
    build_all,
    describe,
    doctored,
    has_path,
    lane_d_cfg,
    replay,
    replay_module,
)

HOUR = 3600
SEED_TAG = "lane_d_windows"


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-windows"))


def _b(rep: dict) -> int:
    return int(rep["cohorts"][0]["t_max"]) + int(HOUR * MARGIN_H)


def _judged(rep: dict, i: int = 0) -> list[dict]:
    ids = set(rep["cohorts"][i]["window_ids"])
    return [w for w in rep["windows"] if w["window_id"] in ids]


def _draws(rep: dict) -> list[tuple[str, dict]]:
    """(path, record) of the signed draw record(s): a top-level `draw`, or
    one per cohort."""
    out = [("draw", rep["draw"])] if isinstance(rep.get("draw"), dict) else []
    out += [(f"cohorts[{i}].draw", co["draw"]) for i, co in enumerate(rep["cohorts"])
            if isinstance(co.get("draw"), dict)]
    return out


def _need(window: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in window]
    assert not missing, f"the signed window record lacks {missing} (W7a rev6.10 fields)"


def _lie(c, n, mutate, paths, *, alt_reason=None):
    mod = replay_module()
    reports, receipts = doctored(c.pub, n, mutate)
    res = replay(c.pub, reports=reports, receipts=receipts)
    for r in res[:n - 1]:
        assert r.status == mod.MATCH, describe(res)
    r = res[n - 1]
    if alt_reason is not None and r.status == mod.UNVERIFIABLE:
        assert r.reason == alt_reason, describe(res)
        return
    assert r.status == mod.MISMATCH, describe(res)
    assert any(has_path(r, p) for p in paths), f"no mismatch at {paths}; got {list(r.mismatches)}"


# ── the honest fields ───────────────────────────────────────────────────────


def _check_fields(rep: dict, cfg, bh: str, *, cohort: int = 0) -> None:
    p = rep["params"]
    assert p.get("embargo_hours") == EMBARGO_HOURS, p.get("embargo_hours")
    assert p.get("min_fresh_window_share") == 0.5, p.get("min_fresh_window_share")
    assert p.get("max_windows_per_settlement") == int(cfg.max_windows_per_settlement)
    assert "embargo_window_share_cap" not in p, "rev6.12 (b) replaced the share cap"
    b = int(rep["cohorts"][cohort]["t_max"]) + int(HOUR * MARGIN_H)
    ws = _judged(rep, cohort)
    for w in ws:
        _need(w, "embargo_hours", "embargo_window", "freq", "kind", "timeframe", "bin", "h",
              "origin")
        assert w["embargo_hours"] == EMBARGO_HOURS and w["kind"] == "forward"
        assert w["timeframe"] == w["freq"] == "H" and w["bin"] == "forward" and w["h"] == 64
        assert w["origin"] == w["min_target_ts"]
        fresh = w["min_target_ts"] > b and w["min_first_seen_ts"] > b
        assert w["embargo_window"] is (not fresh), w["window_id"]
    n_emb = sum(1 for w in ws if w["embargo_window"])
    assert len(ws) - n_emb >= 0.5 * len(ws), "the fresh share holds on a frozen set"
    ds = _draws(rep)
    assert ds, "the report carries the draw record (W7a rev6.10)"
    d = ds[cohort][1] if len(ds) > 1 else ds[0][1]
    assert d["boundary_block_hash"] == bh and d["seed"] == f"{bh}:{SEED_TAG}"
    assert d["max_windows_per_settlement"] == int(cfg.max_windows_per_settlement)


def test_rev610_the_crowning_report_carries_the_embargo_and_draw_fields(chains):
    c = chains["gated"]
    _check_fields(c.pub.report(2), c.cfg, block_hash("b2"))


def test_rev612_the_house_bar_cohort_holds_embargo_windows_within_the_share(chains):
    """Y (T_Y = T0 + 1 day) at b3: snapshot 1's windows have data time inside
    (b − 48 h, b] for Y, so they are eligible only through the embargo
    (embargo windows); snapshots 2 and 3 are fresh. 24 embargo + 48 fresh:
    the share 2/3 ≥ 0.5 holds and the honest chain MATCHes."""
    mod = replay_module()
    c = chains["house_bar"]
    rep = c.pub.report(3)
    _check_fields(rep, c.cfg, block_hash("b3"))
    ws = _judged(rep)
    assert sum(1 for w in ws if w["embargo_window"]) == 24 and len(ws) == 72, len(ws)
    res = replay(c.pub)
    assert all(r.status == mod.MATCH for r in res), describe(res)


# ── forgeries ───────────────────────────────────────────────────────────────


def _param(key, value):
    def m(rep):
        assert key in rep["params"], f"params echo lacks {key}"
        rep["params"][key] = value
    m.__name__ = f"params_{key}"
    return m, [f"params.{key}"]


def _window(edit):
    def m(rep):
        w = _judged(rep)[0]
        _need(w, "embargo_hours", "embargo_window")
        edit(w, _b(rep))
    m.__name__ = edit.__name__
    return m, ["windows", "cohorts[0]", "draw", "verdicts"]


def embargo_above_config(w, b):
    w["embargo_hours"] = 60.0


def embargo_flag_on_a_fresh_window(w, b):
    w["embargo_window"] = True


def first_seen_at_b_minus_e(w, b):
    w["min_first_seen_ts"] = int(b - EMBARGO_HOURS * HOUR)


def data_time_at_b_minus_e(w, b):
    w["min_target_ts"] = int(b - EMBARGO_HOURS * HOUR)


def _draw(edit):
    def m(rep):
        ds = _draws(rep)
        assert ds, "the report carries no draw record (W7a rev6.10)"
        edit(ds[0][1])
    m.__name__ = edit.__name__
    return m, ["draw", "cohorts[0]"]


def draw_seed(d):
    d["seed"] = str(d["seed"]) + "x"


def draw_block_hash(d):
    d["boundary_block_hash"] = "0x" + hashlib.sha256(b"w7d another block").hexdigest()


def _quota(d: dict) -> dict:
    """The per-cell quota of a draw record: `quota` at its top, or the first
    cohort's (`cohorts[0].quota`; W5j pins the cells, not the nesting)."""
    if isinstance(d.get("quota"), dict):
        return d["quota"]
    cos = d.get("cohorts") or []
    assert cos and isinstance(cos[0].get("quota"), dict), "the draw record has no quota"
    return cos[0]["quota"]


def draw_quota(d):
    q = _quota(d)
    cell = sorted(q)[0]
    q[cell] = int(q[cell]) + 1


FORGERIES = [_param("embargo_hours", 24.0), _param("min_fresh_window_share", 0.25),
             _param("max_windows_per_settlement", 4000),
             _window(embargo_above_config), _window(embargo_flag_on_a_fresh_window),
             _window(first_seen_at_b_minus_e), _window(data_time_at_b_minus_e),
             _draw(draw_seed), _draw(draw_block_hash), _draw(draw_quota)]


@pytest.mark.parametrize("forge", FORGERIES, ids=[f[0].__name__ for f in FORGERIES])
def test_rev610_a_forged_embargo_or_draw_record_is_a_mismatch(chains, forge):
    mutate, paths = forge
    _lie(chains["gated"], 2, mutate, paths)


def test_rev612_a_frozen_set_below_the_fresh_share_is_refused(chains):
    """More than half of X's judged windows restated as embargo windows (data
    time inside (b − e, b], flagged): F / (F + E) < 0.5, a set the producer
    never freezes. A MISMATCH, or UNVERIFIABLE `fresh_share_short`."""
    def m(rep):
        b = _b(rep)
        ws = _judged(rep)
        for w in ws[: len(ws) // 2 + 2]:
            _need(w, "embargo_window")
            w["min_target_ts"] = int(b - HOUR)
            w["embargo_window"] = True
        for _, d in _draws(rep):
            if "embargo_windows" in d:
                d["embargo_windows"] = sum(1 for w in ws if w["embargo_window"])

    _lie(chains["gated"], 2, m, ["cohorts[0]", "windows", "draw", "verdicts", "crowned"],
         alt_reason="fresh_share_short")


# ── a non-trivial draw: max_windows_per_settlement below the candidates ────

DRAW_CAP = 32       # below the 48 candidates of snapshots 1 + 2, above min_windows 24
_DRAW: dict = {}


def _draw_chain(tmp) -> dict:
    """X settled once, at b2 (snapshots 1 + 2: 48 candidates), under
    `max_windows_per_settlement = 32` and `min_windows = 24`, twice in
    separate directories (the draw is deterministic)."""
    if "chains" not in _DRAW:
        try:
            cfg = lane_d_cfg(min_windows=24, max_windows_per_settlement=DRAW_CAP)
        except ValueError as e:
            raise AssertionError(f"[lane_d] max_windows_per_settlement not parsed: {e}") from e
        out = []
        for k in (0, 1):
            c = Chain(tmp / f"draw{k}", cfg=cfg)
            c.step(2, entries=[c.x])
            out.append(c)
        _DRAW["chains"] = out
    return _DRAW


def test_rev610_the_draw_caps_the_set_and_the_replay_rederives_it(tmp_path_factory):
    mod = replay_module()
    c0, c1 = _draw_chain(tmp_path_factory.mktemp("w7d-draw"))["chains"]
    r0, r1 = c0.pub.report(1), c1.pub.report(1)
    assert r0["cohorts"], "fixture: b2 judges X on the draw"
    ids = r0["cohorts"][0]["window_ids"]
    assert 24 <= len(ids) <= DRAW_CAP, len(ids)
    by_src: dict = {}
    for w in _judged(r0):
        by_src[w["source"]] = by_src.get(w["source"], 0) + 1
    assert max(by_src.values()) <= float(c0.cfg.max_source_share) * len(ids)
    assert ids == r1["cohorts"][0]["window_ids"], "the seeded draw is deterministic"
    ds = _draws(r0)
    assert ds and sum(int(q) for q in _quota(ds[0][1]).values()) <= DRAW_CAP
    res = replay(c0.pub)
    assert [r.status for r in res] == [mod.MATCH], describe(res)


def test_rev610_a_dropped_drawn_window_is_a_mismatch(tmp_path_factory):
    c = _draw_chain(tmp_path_factory.mktemp("w7d-draw"))["chains"][0]

    def m(rep):
        assert rep["cohorts"] and _draws(rep), "fixture: a judged, drawn cohort"
        co = rep["cohorts"][0]
        gone = co["window_ids"].pop()
        co["set_sha256"] = hashlib.sha256("\n".join(co["window_ids"]).encode()).hexdigest()
        rep["windows"] = [w for w in rep["windows"] if w["window_id"] != gone]
        rep["rows"] = [r for r in rep["rows"] if r["window_id"] != gone]

    _lie(c, 1, m, ["cohorts[0]", "draw", "windows", "rows", "verdicts"])
