"""W9 (plan rev6.10 (b), rev6.11 (a)/(e), rev6.12 (b), W9 row): the audit
re-derives the per-window embargo record, the fresh / embargo class, the
minimum fresh share and the W5j seeded per-bin draw from the signed report,
independently of `lane_d.py`.

Every forgery is re-signed by the validator's key with `rows_sha256`
recomputed (`helpers.resign`). Each mutator first asserts the field it
doctors exists, so on main (no embargo or draw fields; the confirmation duel
still runs) every case fails on an assertion. README.md "rev6.11 / rev6.12"
has the contract and the field names.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from . import helpers as h

HOUR = 3600
E = 48.0
SEED_TAG = "lane_d_windows"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-windows"))


def _s1(run) -> dict:
    rep = copy.deepcopy(run["s1"].report)
    assert rep["cohorts"], "fixture: b1 judges X"
    return rep


def _b(rep: dict, cfg) -> int:
    return int(rep["cohorts"][0]["t_max"]) + int(HOUR * float(cfg.forward_margin_hours))


def _judged(rep: dict) -> list[dict]:
    ids = set(rep["cohorts"][0]["window_ids"])
    return [w for w in rep["windows"] if w["window_id"] in ids]


def _draws(rep: dict) -> list[dict]:
    """The signed draw record(s): a top-level `draw`, or one per cohort."""
    out = [rep["draw"]] if isinstance(rep.get("draw"), dict) else []
    out += [co["draw"] for co in rep["cohorts"] if isinstance(co.get("draw"), dict)]
    return out


def _need(window: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in window]
    assert not missing, f"the signed window record lacks {missing} (W7a rev6.10 fields)"


def _honest_fields(rep: dict, cfg, block_hash: str) -> None:
    p = rep["params"]
    assert p.get("embargo_hours") == E, p.get("embargo_hours")
    assert p.get("min_fresh_window_share") == 0.5, p.get("min_fresh_window_share")
    assert p.get("max_windows_per_settlement") == int(cfg.max_windows_per_settlement)
    assert "embargo_window_share_cap" not in p, "rev6.12 (b) replaced the share cap"
    b = _b(rep, cfg)
    for w in _judged(rep):
        _need(w, "embargo_hours", "embargo_window", "freq", "kind", "timeframe", "bin", "h",
              "origin")
        assert w["embargo_hours"] == E and w["kind"] == "forward"
        assert w["timeframe"] == w["freq"] and w["bin"] == "forward" and w["h"] == 64
        assert w["origin"] == w["min_target_ts"]
        fresh = w["min_target_ts"] > b and w["min_first_seen_ts"] > b
        assert w["embargo_window"] is (not fresh), w["window_id"]
    (d,) = _draws(rep)
    assert d["boundary_block_hash"] == block_hash
    assert d["seed"] == f"{block_hash}:{SEED_TAG}"
    assert d["max_windows_per_settlement"] == int(cfg.max_windows_per_settlement)


def test_rev610_the_crowning_report_carries_the_embargo_and_draw_fields(run):
    rep = _s1(run)
    _honest_fields(rep, run["cfg"], h.w7a.gate_block_hash(h.BOUNDARY_1))
    h.assert_pass(rep, run["cfg"], "honest b1 with its embargo and draw records")


# ── forgeries the audit FAILs ───────────────────────────────────────────────


def _param(key, value):
    def m(rep, cfg):
        assert key in rep["params"], f"params echo lacks {key}"
        rep["params"][key] = value
    m.__name__ = f"params_{key}"
    return m


def _window(edit):
    def m(rep, cfg):
        w = _judged(rep)[0]
        _need(w, "embargo_hours", "embargo_window")
        edit(w, _b(rep, cfg))
    m.__name__ = edit.__name__
    return m


def embargo_above_config(w, b):
    w["embargo_hours"] = 60.0


def embargo_flag_on_a_fresh_window(w, b):
    w["embargo_window"] = True


def first_seen_at_b_minus_e(w, b):
    w["min_first_seen_ts"] = int(b - E * HOUR)


def data_time_at_b_minus_e(w, b):
    w["min_target_ts"] = int(b - E * HOUR)


def _draw(edit):
    def m(rep, cfg):
        ds = _draws(rep)
        assert ds, "the report carries no draw record (W7a rev6.10)"
        edit(ds[0])
    m.__name__ = edit.__name__
    return m


def draw_seed(d):
    d["seed"] = str(d["seed"]) + "x"


def draw_block_hash(d):
    d["boundary_block_hash"] = "0x" + hashlib.sha256(b"w9 another block").hexdigest()


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


@pytest.mark.parametrize("forge", FORGERIES, ids=[f.__name__ for f in FORGERIES])
def test_rev610_a_forged_embargo_or_draw_record_fails(run, forge):
    rep = _s1(run)
    forge(rep, run["cfg"])
    h.assert_fail(h.resign(rep), run["cfg"], forge.__name__)


def test_rev612_a_frozen_set_below_the_fresh_share_fails(run):
    """More than half the judged windows restated as embargo windows (data
    time inside (b − e, b], flagged): F / (F + E) < `min_fresh_window_share`
    = 0.5, which the producer would never freeze (`fresh_share_short`)."""
    rep = _s1(run)
    cfg = run["cfg"]
    b = _b(rep, cfg)
    ws = _judged(rep)
    for w in ws[: len(ws) // 2 + 2]:
        _need(w, "embargo_window")
        w["min_target_ts"] = int(b - HOUR)
        w["embargo_window"] = True
    for d in _draws(rep):
        if "embargo_windows" in d:
            d["embargo_windows"] = sum(1 for w in ws if w["embargo_window"])
    res = h.audit(h.resign(rep), cfg)
    assert res.status == "FAIL", f"check_lane_d {res.status} ({res.detail})"
    assert "fresh_share_short" in str(res.detail), res.detail


# ── a non-trivial draw: max_windows_per_settlement below the candidates ────

_DRAW: dict = {}
DRAW_CAP = 32       # below snapshot 1's 48 eligible windows, above min_windows 24


def _draw_chain(tmp) -> dict:
    """b1 on W7a's snapshot 1 under `max_windows_per_settlement = 32`, settled
    twice in separate directories (the draw is deterministic)."""
    if "runs" not in _DRAW:
        try:
            cfg = h.w7a.lane_d_cfg(max_windows_per_settlement=DRAW_CAP)
        except ValueError as e:
            raise AssertionError(f"[lane_d] max_windows_per_settlement not parsed: {e}") from e
        runs = []
        for k in (0, 1):
            chain = h.pin_embargo(h.w7a.build_chain(tmp / f"draw_chain{k}"))
            e = h.w7a.entry(chain["digests"][0], "w9-draw", coldkey=h.w7a.coldkey("w9-draw"))
            path = tmp / f"draw_state{k}" / "lane_d_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            runs.append(h.w7a.settle(chain, cfg, path, h.signer(), boundary=h.BOUNDARY_1,
                                     snap=1, entries=[e], uid_of={e["hotkey"]: h.CHALLENGER_UID}))
        _DRAW.update(runs=runs, cfg=cfg)
    return _DRAW


def test_rev610_the_draw_caps_the_set_and_is_deterministic(tmp_path_factory):
    d = _draw_chain(tmp_path_factory.mktemp("w9-draw"))
    cfg = d["cfg"]
    r0, r1 = (s.report for s in d["runs"])
    assert r0["cohorts"], "fixture: b1 judges X on the draw"
    ids = r0["cohorts"][0]["window_ids"]
    assert int(cfg.min_windows) <= len(ids) <= DRAW_CAP, len(ids)
    by_src: dict = {}
    for w in _judged(r0):
        by_src[w["source"]] = by_src.get(w["source"], 0) + 1
    assert max(by_src.values()) <= float(cfg.max_source_share) * len(ids)
    assert ids == r1["cohorts"][0]["window_ids"], "the seeded draw is deterministic"
    (dr,) = _draws(r0)
    assert sum(int(q) for q in _quota(dr).values()) <= DRAW_CAP
    assert dr["max_windows_per_settlement"] == DRAW_CAP
    h.assert_pass(r0, cfg, "honest capped draw")


def test_rev610_a_dropped_drawn_window_fails(tmp_path_factory):
    """One drawn window removed (from the cohort set, the window records and
    the rows), the set digest recomputed: the re-derived draw holds it."""
    d = _draw_chain(tmp_path_factory.mktemp("w9-draw"))
    rep = copy.deepcopy(d["runs"][0].report)
    assert rep["cohorts"] and _draws(rep), "fixture: a judged, drawn cohort"
    co = rep["cohorts"][0]
    gone = co["window_ids"].pop()
    co["set_sha256"] = hashlib.sha256("\n".join(co["window_ids"]).encode()).hexdigest()
    rep["windows"] = [w for w in rep["windows"] if w["window_id"] != gone]
    rep["rows"] = [r for r in rep["rows"] if r["window_id"] != gone]
    h.assert_fail(h.resign(rep), d["cfg"], "a drawn window dropped from the frozen set")
